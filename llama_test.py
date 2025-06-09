from tot.models import llama
import sys

propose_prompt = '''
You aim to use numbers and basic arithmetic operations (+ - * /) to obtain 24.
You now should provide eight possible next steps for the given input.
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
Input: 4 5 10 10
Possible next steps:
'''

value_prompt = '''Evaluate if given numbers can reach 24 with basic arithmetic operations (+ - * /) 
You should response (sure/likely/impossible)
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
Input: 5 9 9
'''
value_system_prompt = '''Evaluate if given numbers can reach 24 with basic arithmetic operations (+ - * /) 
You should response to the task with some reasoning steps and sure/likely/impossible
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
'''

value_user_prompt = '''
TASK:
Input: 5 9 9
'''

with open('output.txt', 'w', buffering=1) as f:
    sys.stdout = f
    print(value_prompt)
    output = llama(value_user_prompt, value_system_prompt, n=5, stop=None)