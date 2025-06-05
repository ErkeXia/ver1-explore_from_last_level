# 5-shot
standard_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
'''

# 5-shot
cot_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
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
Input: {input}
'''

# 1-shot
propose_prompt = '''Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {input}
Possible next steps:
'''

value_prompt = '''Evaluate if given numbers can reach 24 (sure/likely/impossible)
10 14
10 + 14 = 24
sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible
4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure
4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure
5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely
5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely
10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible
1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible
{input}
'''

value_last_step_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge: 
sure
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge: 
sure
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge: 
sure
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: 
impossible
Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: 
impossible
Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: 
impossible
Input: {input}
Answer: {answer}
Judge:'''

# evaluate_prompt = '''You and your colleague are working on Game of 24: 
# use numbers and basic arithmetic operations (+ - * /) to obtain 24. 
# Your colleague has provided a possible answers with multiple steps.
# Given the input and each steps of the answer, give a judgement (sure/impossible) if the answer is correct, 
# i.e. it uses each input exactly once and no other numbers, and reach 24.
# You should look at each step and evaluate if it is valid. 
# If you think the answer is impossible to be correct, please step does it start to go wrong. 
# Input: 4 4 6 8
# Steps: 
# Step 1:
# 4 + 8 = 12 (left: 4 6 12)
# Step 2:
# 4 + 8 = 12 (left: 4 6 12)
# 6 - 4 = 2 (left: 2 12)
# Step 3:
# 4 + 8 = 12 (left: 4 6 12)
# 6 - 4 = 2 (left: 2 12)
# 2 * 12 = 24 (left: 24)
# Step 4:
# 4 + 8 = 12 (left: 4 6 12)
# 6 - 4 = 2 (left: 2 12)
# 2 * 12 = 24 (left: 24)
# Answer: (4 + 8) * (6 - 4) = 24
# Judge:
# sure

# Input: 4 5 10 10
# Steps:
# Step 1:
# 10 - 4 = 6 (left: 6 5 10)
# Step 2:
# 10 - 4 = 6 (left: 6 5 10)
# 8 / 2 = 4 (left: 4 6)
# Step 3:
# 10 - 4 = 6 (left: 6 5 10)
# 8 / 2 = 4 (left: 4 6)
# 4 * 6 = 24 (left: 24)
# Step 4:
# 10 - 4 = 6 (left: 6 5 10)
# 8 / 2 = 4 (left: 4 6)
# 4 * 6 = 24 (left: 24)
# Answer: (10 - 4) * (5 + 10) = 24
# Judge:
# impossible, invalid at step 2.
# '''

# evaluate_prompt = """
# You are an expert verifier for the Game of 24.

# Objective  
# Check whether a proposed multi-step solution transforms the four given numbers into **24**, using **only** +, -, *, /, **each starting number exactly once**, and no extra numbers.

# Input format
# ------------
# Input: a b c d
# Steps:
# Step k:
# x op y = z (left: L)    # x and y must be in the current multiset L; z must be the correct result;  
#                         # L is the multiset after replacing x and y with z.

# Task
# ----
# 1. Process the steps in order, updating the multiset.  
# 2. At the first violation (wrong operands, wrong arithmetic, bad “left” list, division by zero, etc.)  
#    stop and output:  

#    No, invalid at step N

# 3. If no violation occurs **and** the final multiset is exactly 24, output:  

#    Yes

# Output **only** that single line—no extra text.

# Examples
# --------
# Input: 4 4 6 8
# Steps:
# 1. 4 + 8 = 12 (left: 4 6 12)
# 2. 6 - 4 = 2 (left: 2 12)
# 3. 2 * 12 = 24 (left: 24)
# 4. Answer: (4 + 8) * (6 - 4)
# Judge:
# Yes

# Input: 4 5 10 10
# Steps:
# 1. 10 - 4 = 6 (left: 6 5 10)
# 2. 8 / 2 = 4 (left: 4 6)
# 3. 4 * 6 = 24 (left: 24)
# Judge:
# No, invalid at step 2

# Input: {input}
# Steps:
# {f_step}
# Judge:
# """

evaluate_prompt = """
You are an expert verifier and coach for the Game of 24.

Goal  
Check a multi-step attempt that should turn four numbers into **24** using only +, -, *, /.  
Besides legality, you must detect the first step that makes the target **unreachable**.

Definitions
-----------
• blocking step = a legal step after which **no sequence** of further legal operations can yield 24.

Required output
---------------
Exactly one line, with one of three possible forms:

1. Yes                              # all steps legal, final left number = 24
2. No, invalid at step N            # first illegal step
3. No, blocking at step N           # first legal but hopeless step

Procedure
---------
1. Walk through the steps in order, checking:
   • x and y are in the set of remaining number.  
   • z is the correct result of x op y (no division by zero).  
   • left list matches the updated multiset.

2. If a check fails → form 2.

3. After **each legal step**, decide whether 24 is still reachable from the new multiset
   If not reachable → form 3 using this step's index.

4. When the steps end, the only number left is 24 -> form 1.

Examples
--------
Input: 4 4 6 8
Steps:
1: 4 + 8 = 12 (left: 4 6 12)
2: 6 - 4 = 2 (left: 2 12)
3: 2 * 12 = 24 (left: 24)
Judge:
Yes

Input: 4 5 10 10
Steps:
1: 10 - 4 = 6 (left: 6 5 10)
2: 8 / 2 = 4 (left: 4 6)           # 8 and 2 not in multiset
3: 4 * 6 = 24 (left: 24)
Judge:
No, invalid at step 2

Input: 1 1 6 8
Steps:
1: 1 + 1 = 2 (left: 2 6 8)
2: 2 + 6 = 8 (left: 8 8)         # after this, 24 is impossible
Judge:
No, blocking at step 2


Input: {input}
Steps:
{f_step}
Judge:
"""
