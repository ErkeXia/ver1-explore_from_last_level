import os
from pathlib import Path

# directory where this script lives
HERE = Path(__file__).resolve().parent

# if kaggle.json is in ../kaggle.json relative to this script:
os.environ["KAGGLE_CONFIG_DIR"] = str(HERE.parent)   # must be a directory

# (optional sanity print)
cfg = os.environ["KAGGLE_CONFIG_DIR"]
print("KAGGLE_CONFIG_DIR =", cfg)
print("kaggle.json exists?", (Path(cfg) / "kaggle.json").exists())

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
# quick test: list datasets (may take a moment)
# print(api.dataset_list(search="crossword"))

# # Download datasets
# api.dataset_download_files('darinhawley/new-york-times-crossword-clues-answers-19932021', path='data', unzip=True)

import pandas as pd

# Load the dataset
df = pd.read_csv('data/nytcrosswords.csv', encoding="latin1")

# Display the first few rows to check the data
print(df.head())

# Filter puzzles with answers having 25 characters (exclude blank blocks)
df = df[df['answer'].apply(lambda x: len(str(x).replace(" ", "")) == 25)]  # removing spaces

# Now let's create a list of the clues and answers in the required format
formatted_data = []

for index, row in df.iterrows():
    # Assuming that 'clues' and 'answer' columns exist in the data
    clues = row['clues']  # Get the clues (could be horizontal/vertical)
    answer = row['answer']  # Get the answer
    
    # Split the answer into individual letters
    answer_letters = list(answer.replace(" ", "").upper())  # remove spaces and convert to list of letters
    
    # Split the clues into horizontal and vertical (assuming you have separate columns for them)
    horizontal_clues = clues[:5]  # Assuming first 5 are horizontal
    vertical_clues = clues[5:]  # Assuming next 5 are vertical
    
    # Append the formatted puzzle to the list
    formatted_data.append([horizontal_clues, answer_letters[:5], vertical_clues, answer_letters[5:]])

# Now save the formatted data to a JSON file
import json

with open('NYT_mini.json', 'w') as json_file:
    json.dump(formatted_data, json_file)

