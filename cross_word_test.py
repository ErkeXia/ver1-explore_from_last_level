import argparse
import sys
import time
import openai
# from tot.methods.bfs import solve, solve_v1
from tot.methods.dfs import solve
from tot.tasks.crosswords import MiniCrosswordsTask
import torch
import transformers

task2 = MiniCrosswordsTask()
x = task2.get_input(0)
print(x)