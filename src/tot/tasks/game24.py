import re
import os
import sympy
import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.game24 import * 


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]

pat = re.compile(r'^\s*\d+(?:\.\d+)?\s*[.:)]\s*')

def clean(text: str) -> str:
    return '\n'.join(
        pat.sub('', line)
        for line in text.splitlines()
        if line.strip()
    )

class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='24.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, '24', file)
        self.data = list(pd.read_csv(path)['Puzzles'])
        self.value_cache = {}
        self.steps = 4
        self.stops = ['\n'] * 4

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            # print(sympy.simplify(expression))
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # print(e)
            return {'r': 0}
            
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)
        print(f'Current number is: {current_numbers}\n')
        if current_numbers == '24':
            print(f'Found the answer! ')
            prompt = cot_user_prompt.format(input=x) + 'Steps:' + y + 'Answer:'
            system = cot_system_prompt
            print(f'system prompt: {system}\n user prompt: {prompt}')
        else:
            prompt = propose_user_prompt.format(input=current_numbers)
            system = propose_system_prompt
        return system, prompt
    
    @staticmethod
    def propose_prompt_unwrap(value_outputs: list) -> list:
        filtered = [clean(s) for s in value_outputs if s and 'are' not in s and 'steps' not in s]
        return filtered
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '')
            # print(ans)
            # print([value_last_step_prompt.format(input=x, answer=ans)])
            return value_last_step_prompt_system, value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        # print(current_numbers)
        return value_system_prompt, value_user_prompt.format(input=current_numbers)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        # value_names = [_.split('\n')[-1] for _ in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        # value = sum(value * value_names.count(name) for name, value in value_map.items())
        value = sum(
            value * sum(s.count(name) for s in value_outputs)
            for name, value in value_map.items()
        )
        return value
    
    @staticmethod
    def validate_prompt_wrap(x: str, ys: list) -> str:
        numbered_steps = '\n'.join(f"{i + 1}: {step}" for i, step in enumerate(ys))
        print(f'numbered steps : \n{numbered_steps}')
        return evaluate_prompt.format(input = x, f_step = numbered_steps)
    
    @staticmethod
    def validate_unwrap(validate_output: str) ->  tuple[int, str]:
        if('Yes' in validate_output or 'yes' in validate_output):
            return -1, validate_output[validate_output.find('Ans'):]
        match = re.search(r"No, invalid at step (\d+)", validate_output)
        redo_s = int(match.group(1)) - 1
        if('Should' in validate_output):
            return redo_s, validate_output[(validate_output.find('Should be:') + 10):].strip()
        return redo_s, ""
            