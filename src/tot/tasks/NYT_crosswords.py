import os
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from tot.tasks.base import Task, DATA_PATH
from tot.models import gpt, llama_instruct
from tot.cache import FileCache

from tot.prompts.NYT_crosswords import *  # noqa: F401,F403


# -----------------------------
# Helpers
# -----------------------------

_CONF_TO_SCORE = {"certain": 1.0, "high": 0.5, "medium": 0.2, "low": 0.1}


def _norm_dir(d: str) -> str:
    d0 = (d or "").strip().lower()
    if d0 in ("a", "across"):
        return "A"
    if d0 in ("d", "down"):
        return "D"
    # allow uppercase already
    if d0 in ("A".lower(), "D".lower()):
        return d0.upper()
    raise ValueError(f"Unknown direction: {d}")


def _normalize_answer_letters(s: str) -> str:
    # Keep A-Z only; your dataset converter already does this.
    return re.sub(r"[^A-Z]", "", (s or "").upper())


def _normalize_fill(raw: str, length: int) -> str:
    s = re.sub(r"[^A-Z_]", "", (raw or "").upper())
    s = s[:length]
    if len(s) < length:
        s += "_" * (length - len(s))
    return s


def _parse_y_entries(y: str) -> List[Dict[str, Any]]:
    """
    Parse y as JSON list of entry states:
      [{"entry_id": 12, "fill": "AB__E", ...}, ...]
    """
    if not y:
        return []

    text = y.strip()
    if text.startswith("Output:"):
        text = text.split("Output:", 1)[1].strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
    except Exception:
        return []

    if not isinstance(obj, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        eid = item.get("entry_id")
        fill = item.get("fill", "")
        if isinstance(eid, int) and isinstance(fill, str):
            out.append({"entry_id": eid, "fill": fill})
    return out


def _dump_y_entries(entries: List[Dict[str, Any]]) -> str:
    # Canonicalized serialization for stable visited-cache keys.
    entries = sorted(entries, key=lambda d: int(d["entry_id"]))
    return json.dumps(entries, ensure_ascii=False, separators=(",", ":"))


# -----------------------------
# Environment
# -----------------------------

@dataclass(frozen=True)
class EntrySpec:
    entry_id: int
    label: str          # clue number label, e.g. "65"
    direction: str      # "A" or "D"
    row: int
    col: int
    length: int
    clue: str
    answer: str         # ground truth string (letters only)
    cells: Tuple[int, ...]  # flattened cell indices


class NYTCrosswordsEnv:
    """
    Full crossword env that behaves similarly to MiniCrosswordsEnv but for arbitrary grid sizes and entry counts.

    Primary state:
      - entry_fills: list[str] (each len=entry.length, '_' for unknown)
      - status: list[int] per-entry: 0 unfilled, 1 filled (touched), 2 changed/conflicted-by-update (like mini)
    Derived state:
      - cell_letters: dict[cell_idx -> letter or '_'] (constructed from entry_fills and overlaps)
    """

    def __init__(self, file: str = "NYT_cw.json"):
        path = os.path.join(DATA_PATH, "crosswords", file)
        raw = json.load(open(path, "r", encoding="utf-8"))

        if isinstance(raw, dict) and "puzzles" in raw:
            self.puzzles = raw["puzzles"]
        elif isinstance(raw, list):
            self.puzzles = raw
        else:
            raise ValueError("Dataset must be a list of puzzles or a dict with key 'puzzles'.")

        self.n = len(self.puzzles)
        self.idx: Optional[int] = None

        # per-puzzle reset state
        self.width = 0
        self.height = 0
        self.entries: List[EntrySpec] = []
        self.status: List[int] = []
        self.entry_fills: List[str] = []
        self.steps = 0
        self.max_steps = 200

        # static per-puzzle indices
        self.key_to_eid: Dict[Tuple[str, str], int] = {}          # (dir,label)->entry_id
        self.eid_to_idx: Dict[int, int] = {}                      # entry_id -> position in self.entries list
        self.cell_to_refs: Dict[int, List[Tuple[int, int]]] = {}  # cell -> [(entry_idx, pos_in_entry), ...]
        self.fillable_cells: List[int] = []                       # union of cells in any entry

        # ground truth per cell for rewards
        self.gt_cell_letters: Dict[int, str] = {}

    def __len__(self):
        return self.n

    # ---- puzzle loading / reset ----

    def reset(
        self,
        idx: int,
        entry_fills: Optional[List[str]] = None,
        status: Optional[List[int]] = None,
        steps: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        self.idx = idx
        p = self.puzzles[idx]

        self.width = int(p.get("width") or p.get("cols"))
        self.height = int(p.get("height") or p.get("rows"))

        raw_entries = p["entries"]
        self.entries = []
        self.key_to_eid = {}
        self.eid_to_idx = {}
        self.cell_to_refs = {}
        self.gt_cell_letters = {}

        for epos, e in enumerate(raw_entries):
            entry_id = int(e["entry_id"])
            label = str(e.get("label", entry_id))
            direction = _norm_dir(e["direction"])
            row = int(e["row"])
            col = int(e["col"])
            length = int(e["length"])
            clue = str(e["clue"])
            answer = _normalize_answer_letters(str(e.get("answer", "")))

            # cells strongly recommended; if absent, derive contiguous cells
            if "cells" in e and isinstance(e["cells"], list) and len(e["cells"]) == length:
                cells = tuple(int(x) for x in e["cells"])
            else:
                step = 1 if direction == "A" else self.width
                start = row * self.width + col
                cells = tuple(start + i * step for i in range(length))

            spec = EntrySpec(
                entry_id=entry_id,
                label=label,
                direction=direction,
                row=row,
                col=col,
                length=length,
                clue=clue,
                answer=answer,
                cells=cells,
            )
            self.entries.append(spec)
            self.key_to_eid[(direction, label)] = entry_id
            self.eid_to_idx[entry_id] = epos

            # build cell refs
            for j, cell in enumerate(cells):
                self.cell_to_refs.setdefault(cell, []).append((epos, j))

            # build GT cell letters
            # If answer is empty or shorter (shouldn't happen), we skip those cells.
            if len(answer) == length:
                for j, cell in enumerate(cells):
                    self.gt_cell_letters[cell] = answer[j]

        self.fillable_cells = sorted(self.cell_to_refs.keys())

        # init mutable state
        nE = len(self.entries)
        self.entry_fills = ["_" * self.entries[i].length for i in range(nE)]
        self.status = [0] * nE
        self.steps = 0
        self.max_steps = max_steps or max(2 * nE, 200)

        if entry_fills is not None:
            if len(entry_fills) != nE:
                raise ValueError("entry_fills length mismatch.")
            self.entry_fills = entry_fills[:]
        if status is not None:
            if len(status) != nE:
                raise ValueError("status length mismatch.")
            self.status = status[:]
        if steps is not None:
            self.steps = int(steps)

        return self.render()

    # ---- derived cell letters ----

    def _cell_letters_from_entry_fills(self) -> Dict[int, str]:
        """
        Derive per-cell letters by taking any non-'_' letters from entry fills.
        If two entries disagree on a letter (shouldn't happen if action_valid is used), the latest state is inconsistent.
        """
        letters: Dict[int, str] = {c: "_" for c in self.fillable_cells}
        for eidx, spec in enumerate(self.entries):
            fill = self.entry_fills[eidx]
            for j, cell in enumerate(spec.cells):
                ch = fill[j]
                if ch != "_":
                    letters[cell] = ch
        return letters

    def _recompute_all_entry_fills_from_cells(self, cell_letters: Dict[int, str]) -> List[str]:
        fills: List[str] = []
        for spec in self.entries:
            chars = []
            for cell in spec.cells:
                chars.append(cell_letters.get(cell, "_"))
            fills.append("".join(chars))
        return fills

    # ---- rendering ----

    def render_board(self) -> str:
        cell_letters = self._cell_letters_from_entry_fills()
        # Blocks are cells not in fillable_cells
        s = "Current Board:\n"
        for r in range(self.height):
            row_chars = []
            for c in range(self.width):
                idx = r * self.width + c
                if idx in cell_letters:
                    row_chars.append(cell_letters[idx])
                else:
                    row_chars.append("#")
            s += "".join(row_chars) + "\n"
        return s

    def render_clues(self, status: Optional[int] = None) -> str:
        """
        Print all clues. Format:
          A65. clue text
          D12. clue text
        """
        lines = []
        for i, spec in enumerate(self.entries):
            if status is None or self.status[i] == status:
                lines.append(f"{spec.direction}{spec.label}. {spec.clue}")
        return "\n".join(lines) + ("\n" if lines else "")

    def render_ans(self, status: Optional[int] = None) -> str:
        """
        Print current fills aligned with clues.
        """
        lines = []
        for i, spec in enumerate(self.entries):
            if status is None or self.status[i] == status:
                fill_spaced = " ".join(self.entry_fills[i])
                lines.append(f"{spec.direction}{spec.label}. {spec.clue}: {fill_spaced}")
        return "\n".join(lines) + ("\n" if lines else "")

    def render_gt_ans(self, status: Optional[int] = None) -> str:
        lines = []
        for i, spec in enumerate(self.entries):
            if status is None or self.status[i] == status:
                lines.append(f"{spec.direction}{spec.label}. {spec.clue}: {spec.answer}")
        return "\n".join(lines) + ("\n" if lines else "")

    def render(self, status_view: bool = True) -> str:
        if not status_view:
            return self.render_board() + "\n" + self.render_ans()

        return (
            self.render_board()
            + "\nUnfilled:\n" + self.render_ans(status=0)
            + "\nFilled:\n" + self.render_ans(status=1)
            + "\nChanged:\n" + self.render_ans(status=2)
        )

    # ---- action validity / step ----

    def action_valid(self, entry_id: int, word: str) -> bool:
        """
        Valid if:
          - word length matches entry length
          - doesn't conflict with existing letters on intersecting cells
          - not a no-op (exactly same as current fill for that entry)
        """
        eidx = self.eid_to_idx.get(int(entry_id))
        if eidx is None:
            return False
        word = re.sub(r"[^A-Za-z]", "", (word or "")).upper()

        spec = self.entries[eidx]
        if len(word) != spec.length:
            return False

        current_fill = self.entry_fills[eidx]
        if word == current_fill:
            return False  # no-op

        cell_letters = self._cell_letters_from_entry_fills()
        # check conflicts at each cell
        for j, cell in enumerate(spec.cells):
            existing = cell_letters.get(cell, "_")
            wch = word[j]
            if existing != "_" and existing != wch:
                return False
        return True

    def step(self, entry_id: int, word: str):
        """
        Apply an entry fill. Updates:
          - entry_fills for all entries (via recompute from cell letters)
          - status for all entries (0/1/2)
          - rewards vs GT
        Returns:
          obs, r_all, done, info
        """
        self.steps += 1
        eidx = self.eid_to_idx.get(int(entry_id))
        if eidx is None:
            return f"Invalid! Unknown entry_id: {entry_id}", 0, False, {}
        word = re.sub(r"[^A-Za-z]", "", (word or "")).upper()

        spec = self.entries[eidx]
        if len(word) != spec.length:
            return f"Invalid! Word length must be {spec.length}.", 0, False, {}

        # validate conflicts
        if not self.action_valid(entry_id, word):
            return "Invalid! Conflicts with existing letters or no-op.", 0, False, {}

        old_fills = self.entry_fills[:]
        old_status = self.status[:]

        # update cell letters by writing the word letters onto spec cells
        cell_letters = self._cell_letters_from_entry_fills()
        for j, cell in enumerate(spec.cells):
            cell_letters[cell] = word[j]

        # recompute all fills from the updated cell letters
        new_fills = self._recompute_all_entry_fills_from_cells(cell_letters)

        # status update similar to mini:
        # - if any previously-known letter changes in an entry, mark it 2 (changed)
        # - mark the acted-on entry as filled (1)
        new_status = old_status[:]
        for i, (a, b, st) in enumerate(zip(old_fills, new_fills, old_status)):
            changed = any((ac != "_" and bc != ac) for ac, bc in zip(a, b))
            if changed:
                new_status[i] = 2
        new_status[eidx] = 1

        self.entry_fills = new_fills
        self.status = new_status

        # rewards:
        # - r_letter: fraction of fillable cells that are correct and filled
        # - r_word: fraction of entries fully correct (exact match) among all entries
        # - r_game: solved if all fillable cells match GT (and are filled)
        correct_filled = 0
        filled_cells = 0
        for cell in self.fillable_cells:
            ch = cell_letters.get(cell, "_")
            gt = self.gt_cell_letters.get(cell, None)
            if gt is None:
                continue
            if ch != "_":
                filled_cells += 1
                if ch == gt:
                    correct_filled += 1

        total_cells = sum(1 for c in self.fillable_cells if c in self.gt_cell_letters)
        r_letter = (correct_filled / total_cells) if total_cells > 0 else 0.0

        correct_words = 0
        for i, spec2 in enumerate(self.entries):
            if self.entry_fills[i] == spec2.answer and "_" not in self.entry_fills[i]:
                correct_words += 1
        r_word = correct_words / len(self.entries) if self.entries else 0.0

        solved = (filled_cells == total_cells) and (correct_filled == total_cells) and (total_cells > 0)

        done = solved or (self.steps >= self.max_steps)
        info = {"r_letter": r_letter, "r_word": r_word, "r_game": solved}
        return self.render(), solved, done, info

    # ---- evaluation helpers (sure/maybe/impossible) ----

    def iter_entry_lines_for_judge(self, min_filled_letters: int = 2) -> List[Tuple[int, str]]:
        """
        Returns list of (entry_idx, line_for_value_prompt).
        Like mini: skip very blank entries.
        """
        lines = []
        for i, spec in enumerate(self.entries):
            fill = self.entry_fills[i]
            filled = sum(ch != "_" for ch in fill)
            if filled < min_filled_letters:
                continue
            fill_spaced = " ".join(fill.lower())
            lines.append((i, f"{spec.direction}{spec.label}. {spec.clue}: {fill_spaced}"))
        return lines


# -----------------------------
# Task wrapper (TOT-style)
# -----------------------------

class NYTCrosswordsTask(Task):
    """
    Input (x): rendered clue list for a crossword puzzle
    Output (y): canonical JSON list of proposed entry fills, e.g.
        [{"entry_id":0,"label":"1","length":5,"fill":"SC___"}]
    Reward (r): derived from env info (word-level / letter-level / game-level)
    """

    def __init__(self, file: str = "NYT_cw.json"):
        super().__init__()
        self.env = NYTCrosswordsEnv(file)

        # Build xs (one per puzzle): render clues from a fresh reset
        self.xs: List[str] = []
        for idx in range(len(self.env)):
            self.env.reset(idx)
            self.xs.append(self.env.render_clues())

        # Default steps: scale with entry count at runtime; keep a safe upper bound here
        self.steps = 200

        # caches
        self.cache_proposals = {}
        self.value_caches: Dict[str, FileCache] = {}
        self.eval_caches: Dict[str, FileCache] = {}

    def __len__(self) -> int:
        return len(self.env)

    def get_input(self, idx: int) -> str:
        self.env.reset(idx)
        return self.env.render_clues()

    # ---- y replay / state setting ----

    def test_output(self, idx: int, output: str):
        """
        Reconstruct state from y JSON list (entry_id + fill).
        Returns info dict (r_word/r_letter/r_game).
        """
        y_entries = _parse_y_entries(output)
        self._set_env_from_y_entries(idx, y_entries)
        return self._info_from_env_state()

    def set_status(self, x: str, y: str):
        idx = self.xs.index(x)
        self.test_output(idx, y)  # updates env in-place

    def _set_env_from_y_entries(self, idx: int, y_entries: List[Dict[str, Any]]) -> None:
        self.env.reset(idx)

        nE = len(self.env.entries)
        entry_fills = ["_" * self.env.entries[i].length for i in range(nE)]
        status = [0] * nE

        for item in y_entries:
            eid = int(item["entry_id"])
            eidx = self.env.eid_to_idx.get(eid)
            if eidx is None:
                continue
            spec = self.env.entries[eidx]
            fill = _normalize_fill(item.get("fill", ""), spec.length)
            entry_fills[eidx] = fill
            if any(ch != "_" for ch in fill):
                status[eidx] = 1

        # Reconcile overlaps via per-cell view, mirroring env step semantics.
        cell_letters: Dict[int, str] = {c: "_" for c in self.env.fillable_cells}
        for eidx, spec in enumerate(self.env.entries):
            fill = entry_fills[eidx]
            for j, cell in enumerate(spec.cells):
                ch = fill[j]
                if ch == "_":
                    continue
                prev = cell_letters[cell]
                if prev == "_" or prev == ch:
                    cell_letters[cell] = ch
                else:
                    status[eidx] = 2

        new_fills = self.env._recompute_all_entry_fills_from_cells(cell_letters)
        for i, (old_fill, new_fill) in enumerate(zip(entry_fills, new_fills)):
            if old_fill != new_fill and any(ch != "_" for ch in old_fill):
                status[i] = 2

        self.env.entry_fills = new_fills
        self.env.status = status
        self.env.steps = sum(1 for s in status if s != 0)

    def _info_from_env_state(self) -> Dict[str, Any]:
        cell_letters = self.env._cell_letters_from_entry_fills()

        correct_filled = 0
        filled_cells = 0
        for cell in self.env.fillable_cells:
            ch = cell_letters.get(cell, "_")
            gt = self.env.gt_cell_letters.get(cell, None)
            if gt is None:
                continue
            if ch != "_":
                filled_cells += 1
                if ch == gt:
                    correct_filled += 1

        total_cells = sum(1 for c in self.env.fillable_cells if c in self.env.gt_cell_letters)
        r_letter = (correct_filled / total_cells) if total_cells > 0 else 0.0

        correct_words = 0
        for i, spec in enumerate(self.env.entries):
            if self.env.entry_fills[i] == spec.answer and "_" not in self.env.entry_fills[i]:
                correct_words += 1
        r_word = correct_words / len(self.env.entries) if self.env.entries else 0.0

        solved = (filled_cells == total_cells) and (correct_filled == total_cells) and (total_cells > 0)
        return {"r_letter": r_letter, "r_word": r_word, "r_game": solved, "r": r_word}

    def y_from_current_state(self, only_nonempty: bool = True) -> str:
        items: List[Dict[str, Any]] = []
        for i, spec in enumerate(self.env.entries):
            fill = self.env.entry_fills[i]
            if only_nonempty and not any(ch != "_" for ch in fill):
                continue
            items.append({
                "entry_id": spec.entry_id,
                "label": spec.label,
                "length": spec.length,
                "fill": fill,
            })
        return _dump_y_entries(items)

    # ---- active prompt/action path used by NYT search ----
    def propose_one_instruct_prompt_wrap(self, line: str, avoid_words: Optional[List[str]] = None):
        """
        line: typically "clue text: _ _ A _ _"
        """
        prompt_input = line
        if avoid_words:
            avoid_str = ", ".join(avoid_words)
            prompt_input += f"\nConstraint: Do NOT propose the following words: {avoid_str}\n"
        else:
            prompt_input += "\nConstraint: Do NOT propose the following words: N/A\n"
        return system_propose_one_prompt, user_propose_one_prompt.format(input=prompt_input)  # noqa: F405

    def get_entry_line(self, entry_idx: int) -> str:
        """
        Canonical one-line description for proposing a word for one entry.
        Format: {clue}: s p _ c e d
        """
        spec = self.env.entries[entry_idx]
        current_fill = self.env.entry_fills[entry_idx]
        fill_spaced = " ".join(current_fill.lower())
        return f"{spec.clue}: {fill_spaced}"

    def action_from_entry_word(self, entry_id: int, word: str) -> Optional[str]:
        """
        Build a deterministic action string from entry_id + proposed word.
        Returns None for unknown entry_id or empty/non-alpha words.
        """
        eidx = self.env.eid_to_idx.get(int(entry_id))
        if eidx is None:
            return None
        spec = self.env.entries[eidx]

        w = re.sub(r"[^A-Za-z]", "", (word or "")).upper()
        if not w:
            return None
        # keep full model output; env.action_valid enforces exact length.
        return f"e{spec.entry_id}. {w}"

    def propose_one_outputs_unwrap(self, x: str, y: str, outputs: List[str]):
        """
        Extract the best single WORD + aggregated score from free-form outputs that contain:
          WORD (high)
        Returns (word, best_score) or "NONE_SIGNAL".
        """
        proposals_to_scores: Dict[str, float] = {}
        for out in outputs:
            last = out.strip().splitlines()[-1].strip().lower() if out.strip() else ""
            if last == "none":
                return "NONE_SIGNAL"

            matches = re.findall(r"([A-Za-z]+)\s*\((certain|high|medium|low)\)", out, flags=re.IGNORECASE)
            for word, conf in matches:
                score = _CONF_TO_SCORE.get(conf.lower(), 0.0)
                w = word.upper()
                proposals_to_scores[w] = proposals_to_scores.get(w, 0.0) + score

        if not proposals_to_scores:
            return None

        best_word, best_score = max(proposals_to_scores.items(), key=lambda kv: kv[1])
        return best_word, best_score

    # ---- action validity ----

    def action_valid(self, entry_id: int, word: str, x: str, y: str) -> bool:
        self.set_status(x, y)
        return self.env.action_valid(entry_id, word)

    # ---- value + evaluation (sure/maybe/impossible) ----

    def _get_eval_cache(self, model: str) -> FileCache:
        key = (model or "unknown").strip().lower()
        if key not in self.eval_caches:
            safe = re.sub(r"[^A-Za-z0-9_]", "_", key)
            self.eval_caches[key] = FileCache(f"NYT_crossword/evaluation_{safe}.json")
        return self.eval_caches[key]

    def _parse_eval_label(self, text: str) -> Optional[str]:
        valid = ("sure", "maybe", "impossible")
        t = (text or "").lower()

        # same heuristic as mini: look for "conclusion" then next line
        lines = t.splitlines()
        for i, line in enumerate(lines):
            if "conclusion" in line and i + 1 < len(lines):
                cand = lines[i + 1].strip()
                for v in valid:
                    if v in cand:
                        return v

        for v in valid:
            if v in t:
                return v
        return None

    def score(self, x: str, y: str, n_evaluate_sample: int, model: str = "llama") -> Dict[str, int]:
        """
        Similar to mini: for entries that are not too blank, run a value prompt and count labels.
        """
        self.set_status(x, y)
        count = {"sure": 0, "maybe": 0, "impossible": 0}

        if model not in self.value_caches:
            safe = model.replace("/", "_").replace("-", "_")
            self.value_caches[model] = FileCache(f"NYT_crossword/value_cache_{safe}.json")
        cache = self.value_caches[model]

        for _, line in self.env.iter_entry_lines_for_judge(min_filled_letters=2):
            cached = cache.get(line)
            if cached in count:
                count[cached] += 1
                continue

            if "gpt" in model.lower():
                prompt = value_prompt.format(input=line)  # noqa: F405
                res = gpt(prompt, stop=None, max_tokens=200)[0].strip().lower()
            else:
                user_prompt = user_value_prompt.format(input=line)  # noqa: F405
                res = llama_instruct(user_prompt, system_value_prompt)[0].strip().lower()  # noqa: F405

            lab = self._parse_eval_label(res) or "impossible"
            cache.set(line, lab)
            count[lab] += 1

        return count

    def evaluate_state(self, x: str, y: str, model: str = "llama") -> Optional[List[int]]:
        """
        Returns list of entry indices judged "sure".
        If any entry is judged "impossible", returns None (prune).
        """
        self.set_status(x, y)
        cache = self._get_eval_cache(model)

        sure_list: List[int] = []
        for eidx, line in self.env.iter_entry_lines_for_judge(min_filled_letters=2):
            res = cache.get(line)
            if res is None:
                # try a few times like mini
                label = None
                for _ in range(3):
                    if "gpt" in model.lower():
                        prompt = value_prompt.format(input=line)  # noqa: F405
                        full = gpt(prompt)[0]
                    else:
                        user_prompt = user_evaluate_prompt.format(input=line)  # noqa: F405
                        full = llama_instruct(user_prompt, system_evaluate_prompt)[0]  # noqa: F405
                    label = self._parse_eval_label(full)
                    if label in ("sure", "maybe", "impossible"):
                        break
                res = label or "impossible"
                cache.set(line, res)

            if res == "sure":
                sure_list.append(eidx)
            elif res == "impossible":
                return None

        return sure_list

    def gpt_evaluate(self, x: str, y: str) -> List[int]:
        # Compatibility helper for existing search refinement flow.
        out = self.evaluate_state(x, y, model="gpt")
        return out or []

    def prune_grid_by_sure_list(self, x: str, y: str, sure_list: List[int]) -> str:
        # Keep only proposed entries that are currently judged "sure".
        self.set_status(x, y)
        sure_entry_ids = set()
        for eidx in sure_list:
            if 0 <= eidx < len(self.env.entries):
                sure_entry_ids.add(self.env.entries[eidx].entry_id)

        y_entries = _parse_y_entries(y)
        kept: List[Dict[str, Any]] = []
        for item in y_entries:
            if int(item["entry_id"]) in sure_entry_ids:
                kept.append(item)
        return _dump_y_entries(kept)

    # ---- visualization helpers ----

    def render_grid_only(self, x: str, y: str) -> str:
        """
        Convenience: reconstruct state from y then return just the rendered grid.
        """
        self.set_status(x, y)
        return self.env.render_board()

    def visualize(self, idx: int, y: str, include_entries: bool = False) -> str:
        x = self.get_input(idx)
        self.set_status(x, y)
        board = self.env.render_board()
        if not include_entries:
            return board
        return board + "\nFilled:\n" + self.env.render_ans(status=1) + "\nChanged:\n" + self.env.render_ans(status=2)
