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
Input: {input}
'''


# 1-shot
# propose_prompt = '''Input: 2 8 8 14
# Possible next steps:
# 2 + 8 = 10 (left: 8 10 14)
# 8 / 2 = 4 (left: 4 8 14)
# 14 + 2 = 16 (left: 8 8 16)
# 2 * 8 = 16 (left: 8 14 16)
# 8 - 2 = 6 (left: 6 8 14)
# 14 - 8 = 6 (left: 2 6 8)
# 14 /  2 = 7 (left: 7 8 8)
# 14 - 2 = 12 (left: 8 8 12)
# Input: {input}
# Possible next steps:
# '''

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
Input: {input}
Possible next steps:
'''

propose_system_prompt = '''
You aim to use numbers and basic arithmetic operations (+ - * /) to obtain 24.
You now should provide eight possible next steps for the given input like the example.
EXAMPLE1:
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
EXAMPLE2:
Input: 2 12
Possible next steps:
2 * 12 = 24 (left: 24)
2 + 12 = 14 (left: 14)
12 - 2 = 10 (left: 10)
2 - 12 = -10 (left: -10)
12 / 2 = 6 (left: 6)
12 + 2 = 14 (left: 14)
12 * 2 = 24 (left: 24)
2 / 12 = 0.17 (left: 0.17)
'''

propose_user_prompt = '''
TASK:
Input: {input}
Possible next steps:
'''

# value_prompt = '''Evaluate if given numbers can reach 24 with basic arithmetic operations (+ - * /) 
# You should response to the task with some reasoning steps and sure/likely/impossible
# EXAMPLES:
# Input: 10 14
# 10 + 14 = 24
# sure

# Input: 11 12
# 11 + 12 = 23
# 12 - 11 = 1
# 11 * 12 = 132
# 11 / 12 = 0.91
# impossible

# Input: 4 4 10
# 4 + 4 + 10 = 8 + 10 = 18
# 4 * 10 - 4 = 40 - 4 = 36
# (10 - 4) * 4 = 6 * 4 = 24
# sure

# Input: 4 9 11
# 9 + 11 + 4 = 20 + 4 = 24
# sure

# Input: 5 7 8
# 5 + 7 + 8 = 12 + 8 = 20
# (8 - 5) * 7 = 3 * 7 = 21
# I cannot obtain 24 now, but numbers are within a reasonable range
# likely

# Input: 5 6 6
# 5 + 6 + 6 = 17
# (6 - 5) * 6 = 1 * 6 = 6
# I cannot obtain 24 now, but numbers are within a reasonable range
# likely

# Input: 10 10 11
# 10 + 10 + 11 = 31
# (11 - 10) * 10 = 10
# 10 10 10 are all too big
# impossible

# Input: 1 3 3
# 1 * 3 * 3 = 9
# (1 + 3) * 3 = 12
# 1 3 3 are all too small
# impossible

# TASK:
# Input: {input}
# '''

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
Input: {input}
'''

value_last_step_prompt_system = '''
Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
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
'''

value_last_step_prompt = '''
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
Check a multi-step attempt that should turn four numbers into **24** using only + - * /.  
Besides legality, detect the first step after which **no further legal moves can ever reach 24**.

Required output
---------------
Return **one line** in **one** of these three forms:

1. Yes - Answer: a op b op c op d = 24  
   # all steps legal, final remaining number is 24

2. No, invalid at step N - Should be: x op y = z (left: …)  
   # first illegal or blocking step **and** you can suggest a concrete fix

3. No, invalid at step N  
   # first illegal or blocking step, but no clear single-step fix exists

Procedure
---------
• Walk through the steps in order, ensuring  
   x and y are in the current multiset,  
   z is the correct result of x op y (no ÷0),  
   the stated “left” multiset is correct.  

• If any check fails or the new multiset can never make 24, emit form 2 or 3.  
  (Use form 2 only when you can give one better replacement line.)

• When all steps finish:  
   one remaining number = 24 → form 1  
   otherwise → “invalid” at the last step (form 3).

Examples
Input: 4 4 6 8
Steps:
1: 4 + 8 = 12 (left: 4 6 12)
2: 6 - 4 = 2  (left: 2 12)
3: 2 * 12 = 24 (left: 24)
Judge:
Yes - Answer: (4 + 8) * (6 - 4) = 24

Input: 4 5 10 10
Steps:
1: 10 - 4 = 6 (left: 6 5 10)
2: 8 / 2 = 4 (left: 4 6)        # 8 and 2 not present
3: 4 * 6 = 24 (left: 24)
Judge:
No, invalid at step 2 - Should be: 5 + 10 = 15 (left: 6 15)

Input: 1 1 6 8
Steps:
1: 1 + 1 = 2 (left: 2 6 8)
2: 2 + 6 = 8 (left: 8 8)        # 24 now impossible
Judge:
No, invalid at step 2

Input: 4 5 6 10
Steps:
1: 10 - 6 = 4 (left: 4 4 5)
2: 4 * 5 = 20 (left: 4 4 20)
3: 4 + 20 = 24 (left: 4 24)
Judge:
No, invalid at step 2 - Should be:  4 * 5 = 20 (left: 4 20)

TASK
Input: {input}
Steps:
{f_step}
Judge:

"""






# evaluate_prompt = '''
# You are an expert verifier and coach for the Game of 24.

# Goal  
# Evaluate a multi-step attempt that should turn four numbers into **24** using only + - * /.  
# Besides legality, detect the first step that makes 24 unreachable, **then give one short tip** the next
# model can use.

# Definitions
# -----------
# • blocking step = legal step after which **no sequence** of further legal operations can yield 24.

# Required output
# ---------------
# Exactly **two lines**:

# 1. One of
#    • Yes                              # all steps legal, final left number = 24  
#    • No, invalid at step N            # first illegal step  
#    • No, blocking at step N           # first legal but hopeless step  

# 2. ≤ 20 words of advice (no extra lines).  
#    • If verdict is Yes → a brief praise (“Good job”).  
#    • If verdict is No → a concrete, actionable hint (e.g.  
#      “Step 2 uses 8,2 not present” or  
#      “Avoid merging 1 and 6; keep 6*4”)

# Procedure
# ---------
# 1. Walk through the steps in order, checking  
#    • operands are in the remaining set,  
#    • result is correct (no division by zero),  
#    • “left” list matches the updated multiset.  
#    If a check fails → verdict form 2.

# 2. After each legal step, decide whether 24 is reachable from the new multiset.  
#    If unreachable → verdict form 3 for this step.

# 3. When the steps end  
#    • single number = 24 → Yes  
#    • single number ≠ 24 → No, blocking at last step

# Remember: **only two lines** of output—no code fences, no commentary.

# Examples
# --------
# Input: 4 4 6 8
# Steps:
# 1: 4 + 8 = 12 (left: 4 6 12)
# 2: 6 - 4 = 2  (left: 2 12)
# 3: 2 * 12 = 24 (left: 24)
# Judge:
# Yes
# Good job

# Input: 4 5 10 10
# Steps:
# 1: 10 - 4 = 6  (left: 6 5 10)
# 2: 8 / 2 = 4  (left: 4 6)          # 8 and 2 not present
# 3: 4 * 6 = 24 (left: 24)
# Judge:
# No, invalid at step 2
# Step 2 uses numbers not in set

# Input: 1 1 6 8
# Steps:
# 1: 1 + 1 = 2  (left: 2 6 8)
# 2: 2 + 6 = 8  (left: 8 8)          # 24 now impossible
# Judge:
# No, blocking at step 2
# Keep 6 for 6*4 later

# Input: {input}
# Steps:
# {f_step}
# Judge:
# '''