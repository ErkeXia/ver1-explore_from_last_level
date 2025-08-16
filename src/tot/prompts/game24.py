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

propose_system_prompt_act = '''
You aim to use numbers and basic arithmetic operations (+ - * /) to obtain 24.
You now should provide eight possible next steps for the given input like the example.
EXAMPLE1:
Input: 2 8 8 14
Possible next steps:
2 + 8 = 10
8 / 2 = 4
14 + 2 = 16
2 * 8 = 16
8 - 2 = 6
14 - 8 = 6
14 /  2 = 7
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

# propose_system_prompt_act = '''
# You are playing the Game of 24, where you use numbers and basic arithmetic operations (+, -, *, /) to obtain 24. 
# Given the current set of available numbers, your task is to go one step further.
# Provide a list of eight possible *next steps*, where each step involves applying one arithmetic operation to two numbers from the list and showing the result.

# **Respond with the possible next steps** that you think are most likely to help achieve 24.
# Respond **only** with the possible next steps in the format below. Do **not** include any extra commentary or explanations.

# Your response should be in the following format:

# EXAMPLE1:
# Input: 2 8 8 14
# Possible next steps:
# 2 + 8 = 10
# 8 / 2 = 4
# 14 + 2 = 16
# 2 * 8 = 16
# 8 - 2 = 6
# 14 - 8 = 6
# 14 / 2 = 7
# 14 - 2 = 12

# EXAMPLE2:
# Input: 2 12
# Possible next steps:
# 2 * 12 = 24
# 2 + 12 = 14
# 12 - 2 = 10
# 2 - 12 = -10
# 12 / 2 = 6
# 12 + 2 = 14
# 12 * 2 = 24
# 2 / 12 = 0.17
# '''


propose_user_prompt = '''
TASK:
Input: {input}
Possible next steps:
'''

propose_gpt_4_1 = '''
You aim to use numbers and basic arithmetic operations (+ - * /) to obtain 24.
You now should provide eight possible next steps for the given input in the exact format shown in examples.
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
TASK:
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

# value_system_prompt_mistral = '''
# You are given a set of numbers. Your task is to evaluate if these numbers can reach 24 using basic arithmetic operations (+, -, *, /).
# **Think step-by-step internally**, and provide a **logical breakdown** of your reasoning.

# **Follow these specific instructions**:
# 1. Perform arithmetic operations on the numbers to check if 24 can be obtained.
# 2. Write your steps in this **exact format**. No extra commentary or explanations. Only perform arithmetic and show the result in the specified format.

# Format for your response:
# a op b = c        (remaining: …)   # optional  
# Analysis           # optional  
# <final> sure | likely | impossible

# You may write at most **five lines** total. The last line must contain your final decision.

# ### EXAMPLES:

# Input: 10 14  
# 10 + 14 = 24  
# sure

# Input: 11 12  
# 11 + 12 = 23  
# 12 - 11 = 1  
# 11 * 12 = 132  
# 11 / 12 = 0.91  
# impossible
# '''

# value_system_prompt = '''
# Evaluate if the given numbers can reach 24 using basic arithmetic operations (+, -, *, /).
# **Think step-by-step internally** and perform calculations logically.
# Produce output **in exactly this format**:

# a op b = c         # optional  
# Analysis           # optional  
# <final> sure | likely | impossible

# You may write at most **five lines** in total. The <final> decision must be on the last line.

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
# '''

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
Check a multi-step attempt that should turn four numbers into **24** using only + - * /. Each number can be used once. 
Check if the numbers used in each step is available, if the left list is correct, if there is arithmetic problem. 
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
2: 2 + 6 = 8 (left: 8 8)        # 24 now impossible with 8 8 left
Judge:
No, invalid at step 2

Input: 4 5 6 10
Steps:
1: 10 - 6 = 4 (left: 4 4 5)
2: 4 * 5 = 20 (left: 4 4 20)
3: 4 + 20 = 24 (left: 4 24)
Judge:
No, invalid at step 2 - Should be:  4 * 5 = 20 (left: 4 20)

Input: 4 5 10 10
Steps:
1. 4 + 10 = 14 (left: 14 10)    # 5 should be in the left list
2. 14 + 10 = 24 (left: 24) 
Judge:
No, invalid at step 1 - Should be:  4 + 10 = 14 (left: 14 10 5)

Input: 1 2 4 7
Steps:
1: 7 - 1 = 6 (left: 2 6 4)
2: 4 * 6 = 24 (left: 24)        # 2 should be in the left list
Judge:
No, invalid at step 2

TASK
Input: {input}
Steps:
{f_step}
Judge:

"""


evaluate_prompt_sys = """
You are an expert verifier and coach for the Game of 24.

Goal  
Check a multi-step attempt that should turn four numbers into **24** using only + - * /. Each number can be used once. 
Check if the numbers used in each step is available and if there are arithmetic problems. 
Besides legality, detect the first step after which **no further legal moves can ever reach 24**.

Required output
---------------
Return **one line** in **one** of these two forms:

1. Yes - Answer: a op b op c op d = 24  
   # all steps legal, final remaining number is 24

2. No, invalid at step N - Should be: x op y = z (left: …)  
   # first illegal or blocking step **and** you can suggest a concrete fix

Procedure
---------
• Walk through the steps in order, ensuring 
   the step is in the form of x op y = z,
   x and y are available,
   z is the correct result of x op y (no ÷0).

• If any check fails or after this step, available numbers can never make 24, emit form 2.  

• When all steps finish:  
   one remaining number = 24 → form 1  
   otherwise → form 2, give your suggestion.

Examples
Input: 4 4 6 8
Steps:
1: 4 + 8 = 12
2: 6 - 4 = 2
3: 2 * 12 = 24
Judge:
Yes - Answer: (4 + 8) * (6 - 4) = 24

Input: 4 5 10 10
Steps:
1: 10 - 4 = 6
2: 8 / 2 = 4        # 8 and 2 not present
3: 4 * 6 = 24
Judge:
No, invalid at step 2 - Should be: 5 + 10 = 15

Input: 1 1 6 8
Steps:
1: 1 + 1 = 2
2: 2 + 6 = 8        # 24 now impossible with 8 8 left
Judge:
No, invalid at step 2 - Should be: 6 / 2 = 3

Input: 4 5 6 10
Steps:
1: 10 - 6 = 4
2: 4 * 5 = 20
3: 4 + 20 = 24
Judge:
No, invalid at step 2 - Should be:  4 * 5 = 20

Input: 4 5 10 10
Steps:
1. 4 + 10 = 14    
2. 14 + 10 = 24      # 5 should be used for once
Judge:
No, invalid at step 2 - Should be:  5 + 14 = 19

TASK
Input: {input}
Steps:
{f_step}
Judge:

"""

suggest_prompt_sys = """
You are a coach for the Game of 24.

Goal  
You are given an incorrect answer to a game of 24.
Detect the first step after which **no further legal moves can ever reach 24**, either due to illegal or wrong steps taken.

Required output
---------------
Think step by step (you may show your reasoning).  
Return your **final decision in the last line** in this exact form:

   Invalid at step N - Should be: x op y = z (left: …)  
      # first illegal or blocking step **and** a concrete fix

Procedure
---------
• Walk through the steps in order, ensuring after this step, it is still possible to get 24.
• For each step, check:
   - Format is exactly: x op y = z (ops allowed: + - * /; division by zero is illegal).  
   - x and y are available in the current multiset (numbers used once each).  
   - Arithmetic is correct (use tolerance 1e-6 for floats).  
• If any check fails, or after applying this step the remaining numbers can never reach 24, stop and give your suggestion.
• Your suggestion must be legal from the state **before** step N

Examples
Input: 4 5 10 10
Steps:
1: 10 - 4 = 6
2: 8 / 2 = 4       
3: 4 * 6 = 24
Judge:
--your reasoning steps--
Invalid at step 2 - Should be: 5 + 10 = 15

Input: 1 1 6 8
Steps:
1: 1 + 1 = 2
2: 2 + 6 = 8       
3: 8 + 8 = 16
Judge:
--your reasoning steps--
Invalid at step 2 - Should be: 6 / 2 = 3

Input: 4 5 10 10
Steps:
1. 4 + 10 = 14    
2. 14 + 10 = 24      # 5 should be used for once
Judge:
--your reasoning steps--
Invalid at step 2 - Should be:  5 + 14 = 19

TASK
Input: {input}
Steps:
{f_step}
Judge:
"""

correctness_prompt_sys = """
You are an expert verifier for the Game of 24.

Goal  
Check a multi-step attempt that should turn four numbers into **24** using only + - * /. Each number can be used once. 
Check if the numbers used in each step is available and if there are arithmetic problems. 

Required output
---------------
Think step by step. 
Return your final decision in the last line in **one** of these two forms:

1. Yes - Answer: a op b op c op d = 24  
   # all steps legal, final remaining number is 24

2. No
   # This is not a valid answer

Procedure
---------
• Walk through the steps in order, ensuring 
   the step is in the form of x op y = z,
   x and y are available,
   z is the correct result of x op y (no ÷0),

• If any check fails emit form 2.  

• When all steps finish:  
   one remaining number = 24 → form 1  
   otherwise → form 2

Examples
Input: 4 4 6 8
Steps:
1: 4 + 8 = 12
2: 6 - 4 = 2
3: 2 * 12 = 24
Judge:
Yes - Answer: (4 + 8) * (6 - 4) = 24

Input: 4 5 10 10
Steps:
1: 10 - 4 = 6
2: 8 / 2 = 4       
3: 4 * 6 = 24
Judge:
No

Input: 1 1 6 8
Steps:
1: 1 + 1 = 2
2: 2 + 6 = 8       
3: 8 + 8 = 16
Judge:
No

Input: 4 5 10 10
Steps:
1. 4 + 10 = 14    
2. 14 + 10 = 24      # 5 should be used for once
Judge:
No

TASK
Input: {input}
Steps:
{f_step}
Judge:

"""

