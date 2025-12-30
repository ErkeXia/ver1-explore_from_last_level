import re
import os
import json
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.crosswords import * 
from tot.models import gpt, llama_instruct
from tot.cache import FileCache

class MiniCrosswordsEnv:
    def __init__(self, file='mini0505.json'):
        self.file = os.path.join(DATA_PATH, 'crosswords', file)

        self.file = json.load(open(self.file))
        self.n = len(self.file)
        self.cache = {}
        self.idx = None
        self.times = 0
        self.prompt_status_cache = {}

    def __len__(self):
        return self.n
    
    def reset(self, idx, board=None, status=None, steps=None):
        self.idx = idx
        self.data, self.board_gt = self.file[idx]
        self.board = ['_'] * 25
        self.ans = ['_____'] * 10
        self.ans_gt = self.get_ans(self.board_gt)
        self.steps = 0
        self.status = [0] * 10  # 0: unfilled; 1: filled; 2: filled then changed
        if board is not None:
            self.board = board
            self.ans = self.get_ans(self.board)
        if status is not None:
            self.status = status
        if steps is not None:
            self.steps = steps
        return self.render()
    

    def prompt_status(self):
        count = {'sure': 0, 'maybe': 0, 'impossible': 0}
        for ans, data, status in zip(self.ans, self.data, self.status):
            # if status != 0: continue
            if ans.count('_') >= 4: continue
            ans = ' '.join(ans.lower())
            line = f'{data}: {ans}'
            prompt = value_prompt.format(input=line)
            if prompt in self.prompt_status_cache:
                res = self.prompt_status_cache[prompt]
            else:
                res = gpt(prompt)[0]
                self.prompt_status_cache[prompt] = res
            # print(line)
            # print(res)
            # print()
            res = res.split('\n')[-1].strip()
            if res in count: count[res] += 1
        # print(count)
        return count
    
    def render_gt_board(self):
        s = "GT Board:\n"
        for i in range(5):
            s += ' '.join(self.board_gt[i*5:(i+1)*5]) + '\n'
        return s
    
    def render_board(self):
        s = "Current Board:\n"
        for i in range(5):
            s += ''.join(self.board[i*5:(i+1)*5]) + '\n'
        return s

    def render_clues(self, status=None):
        s = ""
        # s += "Horizontal:\n"
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.data[i] + '\n'
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + self.data[i] + '\n'
        return s
    
    def render_ans(self, status=None):
        s = ""
        # s += "Horizontal:\n"
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.data[i] + ': ' + self.ans[i] + '\n'
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + self.data[i] + ': ' + self.ans[i] + '\n'
        return s
    
    def render_gt_ans(self, status=None):
        s = ""
        # s += "Horizontal:\n"
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.data[i] + ': ' + self.ans_gt[i] + '\n'
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + self.data[i] + ': ' + self.ans_gt[i] + '\n'
        return s

    def render(self, status=True):
        if status:
            return self.render_board() + '\nUnfilled:\n' + self.render_ans(status=0) + '\nFilled:\n' + self.render_ans(status=1) + '\nChanged:\n' + self.render_ans(status=2)
        else:
            return self.render_board() + '\n' + self.render_ans()
    
    def get_ans(self, board):
        ans = [''] * 10
        for i in range(5):
            ans[i] = ''.join(board[i*5:(i+1)*5])
        for i in range(5):
            ans[i+5] = ''.join(board[i::5])
        return ans
    
    def step(self, action):
        self.steps += 1
        action = action.split('\n')[-1]
        action = action.split('. ')
        if len(action) != 2:
            return 'Invalid! Format should be like "h1. apple"', 0, False, {}
        pos, word = action

        if len(word) != 5:
            return 'Invalid! Word should have 5 letters.', 0, False, {}
        if pos.startswith('h'):
            idx = int(pos[1:]) - 1
            self.board[idx*5:(idx+1)*5] = list(word.upper())
        elif pos.startswith('v'):
            idx = int(pos[1:]) - 1
            self.board[idx::5] = list(word.upper())
            idx += 5  # for later status update
        else:
            return 'Invalid! Position should be h1-h5 or v1-v5', 0, False, {}
        
        self.new_ans = self.get_ans(self.board)
        # self.status = [2 if (status == 1 and ans != new_ans) else status for status, ans, new_ans in zip(self.status, self.ans, self.new_ans)]
        self.status = [2 if any(letter != new_letter and letter != '_' for letter, new_letter in zip(ans, new_ans)) else status for status, ans, new_ans in zip(self.status, self.ans, self.new_ans)]
        self.status[idx] = 1
        self.ans = self.new_ans
        r_all = (self.board == self.board_gt)
        r_letter = sum(a == b for a, b in zip(self.board, self.board_gt)) / 25
        r_word = sum(a == b for a, b in zip(self.ans, self.ans_gt)) / 10
        return self.render(), r_all, (r_all or self.steps >= 20), {'r_letter': r_letter, 'r_word': r_word, 'r_game': r_all}


class MiniCrosswordsTask(Task):
    """
    Input (x)   : Decription of a 5x5 mini crossword
    Output (y)  : List of 10 words to fill in the crossword
    Reward (r)  : word level and game level
    Input Example: 
    Output Example: 
    """
    def __init__(self, file='mini0505.json'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        self.env = MiniCrosswordsEnv(file)  # use it as a stateless tool
        self.xs = []
        for idx in range(len(self.env)):
            self.env.reset(idx)
            self.xs.append(self.env.render_clues())
        self.steps = 10  # TODO: variable steps??
        self.cache_proposals = {}
        self.value_caches = {}
        self.eval_cache = FileCache("gpt3_5_evaluation.json")

    def __len__(self) -> int:
        return len(self.env)
    
    def get_input(self, idx: int) -> str:
        self.env.reset(idx)
        return self.env.render_clues()

    
    def test_output(self, idx: int, output: str):
        self.env.reset(idx)
        output = output.split('Output:\n')[-1]
        info = {'r_word': 0, 'r_letter': 0, 'r_game': 0}
        for i, line in enumerate(output.strip().split('\n')[-5:], 1):
            letters = line.split(' ')[:5]
            word = ''.join(letters)
            word = word + '_' * (5 - len(word))
            action = f'h{i}. {word}'
            # print(action)
            _, _, _, info = self.env.step(action)
        info['r'] = info['r_word']
        return info

    def set_status(self, x: str, y: str):
        idx = self.xs.index(x)
        self.test_output(idx, y)  # update self.env
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y
    
    def propose_prompt_wrap(self, x: str, y: str='') -> str:
        self.set_status(x, y)
        return propose_prompt.format(input=self.env.render())
    
    def propose_instruct_prompt_wrap(self, x: str, y: str='') -> str:
        self.set_status(x, y)
        return system_propose_prompt, user_propose_prompt.format(input=self.env.render(), board = y)
    
    def propose_one_instruct_prompt_wrap(self, line: str, avoid_words: list = None):
        prompt_input = line
        if avoid_words and len(avoid_words) > 0:
            avoid_str = ", ".join(avoid_words)
            # Insert the constraint before "Your Output:"
            constraint_text = f"\nConstraint: Do NOT propose the following words: {avoid_str}\n"
            prompt_input = prompt_input + constraint_text
        else:
            prompt_input = prompt_input + "\nConstraint: Do NOT propose the following words: N/A \n"
        return system_propose_one_prompt, user_propose_one_prompt.format(input=prompt_input)
    
    def propose_outputs_unwrap(self, x: str, y: str, outputs: list, n_max_propose: int) -> list:
        confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}  # TODO: ad hoc
        proposals_to_scores = {}
        for output in outputs:
            lines = output.split('\n')
            pattern = r'^([hv][1-5])\. ([a-zA-Z]{5,5}) \((certain|high|medium|low)\).*$'
            for line in lines:
                match = re.match(pattern, line.lower())
                if match:
                    parts = [match.group(1), match.group(2), match.group(3)]
                    proposal = parts[0].lower() + '. ' + parts[1].lower()
                    score = confidence_to_value.get(parts[2], 0)
                    proposals_to_scores[proposal] = proposals_to_scores.get(proposal, 0) + score
        
        proposals = sorted(proposals_to_scores.items(), key=lambda x: x[1], reverse=True)
        if n_max_propose != -1:
            proposals = proposals[:n_max_propose]
        proposals = [proposal[0] + '\n' for proposal in proposals]
        self.cache_proposals[(x, y, n_max_propose)] = proposals
        return proposals
    
    def propose_one_outputs_unwrap(self, x: str, y: str, outputs: list) -> str:
        confidence_to_value = {'certain': 1.0, 'high': 0.5, 'medium': 0.2, 'low': 0.1}
        proposals_to_scores = {}
        
        for output in outputs:
            last_line = output.strip().split('\n')[-1].strip().lower()
            if 'none' in last_line and len(last_line) < 6: # Safety check length to avoid false positives in text
                return "NONE_SIGNAL"
            
            # This regex is more flexible, looking for the last instance of the pattern
            pattern = r'([a-zA-Z]{5,5})\s*\((certain|high|medium|low)\)'
            matches = re.findall(pattern, output.lower())
            
            for match in matches:
                word, confidence = match
                score = confidence_to_value.get(confidence, 0)
                # We add score to handle multiple generations of the same word
                proposals_to_scores[word] = proposals_to_scores.get(word, 0) + score
                
        if not proposals_to_scores:
            return None

        # Find the best proposal among all parsed outputs
        best_proposal, best_score = max(proposals_to_scores.items(), key=lambda item: item[1])
        
        return best_proposal, best_score
    
    def action_valid(self, action: str, x: str, y: str) -> bool:
        """
        Checks if a proposed action (e.g., "h1. apple") is valid by ensuring
        it does not conflict with any existing letters on the board.

        Args:
            action: The proposed move string.
            y: The current board state as a flat list of 25 characters.

        Returns:
            True if the move is valid (no conflicts), False otherwise.
        """
        self.set_status(x, y) 
        current_board = list(self.env.board)
        try:
            pos, word = action.split('. ')
            word = word.strip().upper()

            if len(word) != 5:
                return False

            if pos.startswith('h'):
                row_idx = int(pos[1:]) - 1
                
                # --- START: New Logic ---
                # Get the current word from the board at that position
                current_word_on_board = "".join(current_board[row_idx * 5 : (row_idx * 5) + 5])
                
                # If the proposed word is exactly the same as what's already there, it's not a valid *new* move.
                if word == current_word_on_board:
                    return False  # Exact match found, so invalid
                # --- END: New Logic ---

                # Original conflict check (remains the same)
                for i in range(5):
                    board_char = current_board[row_idx * 5 + i]
                    word_char = word[i]
                    if board_char != '_' and board_char != word_char:
                        return False  # Conflict found

            elif pos.startswith('v'):
                col_idx = int(pos[1:]) - 1

                # --- START: New Logic ---
                # Get the current word from the board at that position
                current_word_on_board = "".join(current_board[i * 5 + col_idx] for i in range(5))

                # If the proposed word is exactly the same, it's not a valid *new* move.
                if word == current_word_on_board:
                    return False # Exact match found, so invalid
                # --- END: New Logic ---

                # Original conflict check (remains the same)
                for i in range(5):
                    board_char = current_board[i * 5 + col_idx]
                    word_char = word[i]
                    if board_char != '_' and board_char != word_char:
                        return False  # Conflict found
            else:
                return False # Invalid position format
                
        except (ValueError, IndexError):
            return False # Malformed action string

        return True # No conflicts found and not an exact match
    
    def evaluate(self, x: str, y: str, n_evaluate_sample: int, model: str = 'llama') -> int:
        self.set_status(x, y)
        count = {'sure': 0, 'maybe': 0, 'impossible': 0}
        
        if model not in self.value_caches:
            safe_model_name = model.replace('/', '_').replace('-', '_')
            self.value_caches[model] = FileCache(f"crossword_value_cache_{safe_model_name}.json")
            
        current_cache = self.value_caches[model]
        
        for ans, data, status in zip(self.env.ans, self.env.data, self.env.status):
            if ans.count('_') >= 3: continue
            ans = ' '.join(ans.lower())
            line = f'{data}: {ans}'
            # prompt = value_prompt.format(input=line)
            # res = gpt(prompt)[0]
            # res = base_model(prompt)[0]
            res = current_cache.get(line)
            if res is not None:
                print(res)
                count[res] += 1
                continue
            
            if 'gpt' in model.lower():
                 # Use standard prompt for GPT models
                 prompt = value_prompt.format(input=line)
                 res = gpt(prompt, stop=None, max_tokens=200)[0].strip().lower()
            else:
                 # Use instruct prompts for Llama/local models
                 user_prompt = user_value_prompt.format(input=line)
                 res = llama_instruct(user_prompt, system_value_prompt)[0].strip().lower()
            
            print(line)
            print(res)
            print()
            res = res.split('\n')[-1].strip()
            if res in count: 
                count[res] += 1
                current_cache.set(line, res)
        print(count)
        return count
    
    def gpt_evaluate(self, x: str, y: str) -> list:
        self.set_status(x, y)
        sure_lst = []
        for i, (ans, data, status) in enumerate(zip(self.env.ans, self.env.data, self.env.status)):
            ans = ' '.join(ans.lower())
            line = f'{data}: {ans}'
            
            res = self.eval_cache.get(line)
            if res is None:
                prompt = value_prompt.format(input=line)
                res = gpt(prompt)[0]
                self.eval_cache.set(line, res)
                
            print(line)
            print(res)
            print()
            res = res.split('\n')[-1].strip()
            if res == 'sure':
                sure_lst.append(i)
            if res == 'impossible':
                return None
        print(sure_lst)
        return sure_lst
    
    def prune_grid_by_sure_list(self, x: str, y: str, sure_list: list[int]) -> str:
        """Keeps only the words at indices in sure_list and blanks out the rest."""
        self.set_status(x, y) 
        board = list(self.env.board)
        if board is None: 
            # If parsing fails, return the original string to be safe
            return y_grid_string

        # Create a boolean mask of 25 cells to keep
        keep_mask = [False] * 25
        for idx in sure_list:
            if 0 <= idx < 5: # Horizontal word (indices 0-4)
                for i in range(5): keep_mask[idx * 5 + i] = True
            elif 5 <= idx < 10: # Vertical word (indices 5-9)
                col_idx = idx - 5
                for i in range(5): keep_mask[i * 5 + col_idx] = True

        # Build the new pruned board
        pruned_board = [board[i] if keep_mask[i] else '_' for i in range(25)]
        
        # Format it back into a grid string
        rows = [" ".join(pruned_board[i*5:(i+1)*5]) for i in range(5)]
        return "Output:\n" + "\n".join(rows) + "\n"
