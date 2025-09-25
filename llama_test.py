from tot.models import model_setup, llama_instruct, llama_usage, base_model
import sys
import time


propose_prompt = '''
You aim to use numbers and basic arithmetic operations (+ - * /) to obtain 24.
You now should provide eight possible next steps for the given input like the example.
EXAMPLE:
Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
TASK:
Input: 1 5 5 5
Possible next steps:
'''

propose_system_prompt = '''
You aim to use numbers and basic arithmetic operations (+ - * /) to obtain 24.
You now should provide eight possible next steps for the given input like the example.
EXAMPLE:
Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
'''

propose_user_prompt = '''
TASK:
Input: 4 5 6 10
Possible next steps:
'''

value_system_prompt2 = '''
Evaluate if the given numbers can reach 24 using basic arithmetic operations (+, -, *, /).
**Think step-by-step internally** and perform calculations logically.
Produce output **in exactly this format**:

a op b = c         # optional  
Analysis           # optional  
<final> sure | likely | impossible

You may write at most **five lines** in total. The <final> decision must be on the last line.

EXAMPLES:
Input: 10 14
10 + 14 = 24  
sure

Input: 11 12
11 + 12 = 23  
12 - 11 = 1  
11 * 12 = 132  
11 / 12 = 0.91  
impossible

Input: 4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure

Input: 4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure

Input: 5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely

Input: 5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely

Input: 10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 11 are all too big
impossible

Input: 1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible
'''

value_system_prompt_m = '''
You are given a set of numbers. Your task is to evaluate if these numbers can reach 24 using basic arithmetic operations (+, -, *, /).
Think step-by-step by trying some further operations, and give your final answer in the last line.

Format for your response:
Analysis           # optional  
<final> sure | likely | impossible

The last line must contain your final decision in the above format.

### EXAMPLES:

Input: 10 14  
10 + 14 = 24  
sure

Input: 11 12  
11 + 12 = 23  
12 - 11 = 1  
11 * 12 = 132  
11 / 12 = 0.91  
impossible

Input: 5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely
'''

value_system_prompt = '''
You are given a set of numbers. Your task is to evaluate if these numbers can reach 24 using basic arithmetic operations (+, -, *, /).
**Think step-by-step internally**, and provide a **logical breakdown** of your reasoning.

**Follow these specific instructions**:
1. Perform arithmetic operations on the numbers to check if 24 can be obtained.
2. Write your steps in this **exact format**. No extra commentary or explanations. Only perform arithmetic and show the result in the specified format.

Format for your response:
a op b = c        (remaining: …)   # optional  
Analysis           # optional  
<final> sure | likely | impossible

You may write at most **five lines** total. The last line must contain your final decision.

### EXAMPLES:

Input: 10 14  
10 + 14 = 24  
sure

Input: 11 12  
11 + 12 = 23  
12 - 11 = 1  
11 * 12 = 132  
11 / 12 = 0.91  
impossible

Input: 5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely
'''


value_user_prompt = '''
TASK:
Input: 4 10 30
'''

value_user_prompt2 = '''
TASK:
Input: 9 9 9
'''

cot_system_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. 
You are given the steps to obtain 24. 
Return only the final answer
Examples:
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24
Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24
Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24
Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24
Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24
'''

cot_user_prompt = '''
Input: 4 5 6 10
Steps:10 + 6 = 16 (left: 4 5 16)
16 + 4 = 20 (left: 4 20)
4 + 20 = 24 (left: 24)
Answer: 
'''

propose_system_prompt_act = '''
You are playing the Game of 24, where you use numbers and basic arithmetic operations (+, -, *, /) to obtain 24. 
Given the current set of available numbers, your task is to go one step further.
Provide a list of eight possible *next steps*, where each step involves applying one arithmetic operation to two numbers from the list and showing the result.

**Respond with the possible next steps** that you think are most likely to help achieve 24.
Respond **only** with the possible next steps in the format below. Do **not** include any extra commentary or explanations.

Your response should be in the following format:

EXAMPLE1:
Input: 2 8 8 14
Possible next steps:
2 + 8 = 10
8 / 2 = 4
14 + 2 = 16
2 * 8 = 16
8 - 2 = 6
14 - 8 = 6
14 / 2 = 7
14 - 2 = 12

EXAMPLE2:
Input: 2 12
Possible next steps:
2 * 12 = 24
2 + 12 = 14
12 - 2 = 10
2 - 12 = -10
12 / 2 = 6
12 + 2 = 14
12 * 2 = 24
2 / 12 = 0.17
'''

propose_prompt_sys = """
You are playing the Game of 24, where you use numbers and basic arithmetic operations (+, -, *, /) to obtain 24. 
Given the current set of available numbers, your task is to output **exactly eight possible next steps with 2 available numbers**, nothing else.

Format:
<expression> = <result>

EXAMPLE:
Input: 2 8 8 14
Possible next steps:
2 + 8 = 10
8 / 2 = 4
14 + 2 = 16
2 * 8 = 16
8 - 2 = 6
14 - 8 = 6
14 / 2 = 7
14 - 2 = 12

TASK:
Input: 4 7 13 13
Possible next steps:
"""

value_prompt = '''Evaluate if given numbers can reach 24 with basic arithmetic operations (+ - * /).  
You must always show at least one line of reasoning (operations you try), and then end with exactly one of: "sure", "likely", or "impossible".  

EXAMPLES:
Input: 10 14
10 + 14 = 24
sure

Input: 11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible

Input: 4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure

Input: 4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure

Input: 5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely

Input: 5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely

Input: 10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible

Input: 1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible

TASK:
Input: 8 32
'''

# value_prompt_o = '''Evaluate if given numbers can reach 24 (sure/likely/impossible)
# 10 14
# 10 + 14 = 24
# sure
# 11 12
# 11 + 12 = 23
# 12 - 11 = 1
# 11 * 12 = 132
# 11 / 12 = 0.91
# impossible
# 4 4 10
# 4 + 4 + 10 = 8 + 10 = 18
# 4 * 10 - 4 = 40 - 4 = 36
# (10 - 4) * 4 = 6 * 4 = 24
# sure
# 4 9 11
# 9 + 11 + 4 = 20 + 4 = 24
# sure
# 5 7 8
# 5 + 7 + 8 = 12 + 8 = 20
# (8 - 5) * 7 = 3 * 7 = 21
# I cannot obtain 24 now, but numbers are within a reasonable range
# likely
# 5 6 6
# 5 + 6 + 6 = 17
# (6 - 5) * 6 = 1 * 6 = 6
# I cannot obtain 24 now, but numbers are within a reasonable range
# likely
# 10 10 11
# 10 + 10 + 11 = 31
# (11 - 10) * 10 = 10
# 10 10 10 are all too big
# impossible
# 1 3 3
# 1 * 3 * 3 = 9
# (1 + 3) * 3 = 12
# 1 3 3 are all too small
# impossible
# {input}
# '''

# print(value_prompt)
model_setup('mistral', TGI_arg = False)
start = time.perf_counter()
# output = llama_B([value_user_prompt], value_system_prompt_m, n=3, stop=None, temperature=0.7)
# elapsed = time.perf_counter() - start
# start = time.perf_counter()
# print(f"{elapsed:.6f} seconds")
# output = llama_B([value_user_prompt], value_system_prompt_m, n=3, stop=None, temperature=0.7)
# elapsed = time.perf_counter() - start
# start = time.perf_counter()
# print(f"{elapsed:.6f} seconds")
# output = llama(value_user_prompt, value_system_prompt_m, n=1, stop=None, temperature=0.7, max_tokens = 200)
for i in range(10):
    output = base_model(value_prompt, max_tokens = 200, stop = ['\n\n'])
    output = output[0]
    print(f'--{i}--')
    print(output)
elapsed = time.perf_counter() - start
print(f"{elapsed:.6f} seconds")
# output = llama(value_user_prompt, value_system_prompt_m, n=1, stop=None, temperature=0.7)
# elapsed = time.perf_counter() - start
# start = time.perf_counter()
# print(f"{elapsed:.6f} seconds")

# output = llama(value_user_prompt, value_system_prompt, n=1, stop=None, temperature=0.7, max_tokens = 200)
