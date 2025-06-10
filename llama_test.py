from tot.models import llama
import sys

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
Input: 1 5 5 5
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
THINK step-by-step **internally**
Produce output in *exactly* this format:
a  op  b  =  c        (remaining: …)   # optional
c  op  d  =  e                         # optional
<final>    sure | likely | impossible
You may write at most five lines total

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
Input: 9 10 10
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


with open('output.txt', 'w', buffering=1) as f:
    sys.stdout = f
    # print(value_prompt)
    output = llama(cot_user_prompt, cot_system_prompt, n=5, stop=None, temperature=0.7)
    print(output)