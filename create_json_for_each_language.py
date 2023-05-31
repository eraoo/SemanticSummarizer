# Developer: Erfan Raoofian
# License: Apache 2.0
"""
This script processes a directory containing code files organized
into directories for each problem and programming language.
It reads all the code files and organizes them into a dictionary,
which is then saved into a JSON file for each programming language.
The JSON file contains a dictionary where each problem is a key and
the value is a list of code submissions, with each submission being
a dictionary of its own with the submission ID as the key and
the code as the value.
"""
import os
import json
from tqdm import tqdm

# The directory where the code files are stored
data_dir = 'data'

def read_code_file(file_path):
    """Reads a code file and returns it as a string.

    Args:
        file_path (str): The path to the code file.

    Returns:
        str: The content of the code file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Get all the problem directories in the 'data' directory
problem_folders = sorted([entry for entry in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, entry))])

# Initialize a dictionary to store code submissions for each language
submissions_by_language = {}

# Iterate through each problem directory
for problem_folder in tqdm(problem_folders, desc='Processing problems'):
    problem_number = int(problem_folder[1:])  # Convert 'p00001' to 1
    problem_path = os.path.join(data_dir, problem_folder)

    # Iterate through each language directory in the problem directory
    for language_folder in os.listdir(problem_path):
        language_path = os.path.join(problem_path, language_folder)

        # Initialize the problem entry in the dictionary
        if language_folder not in submissions_by_language:
            submissions_by_language[language_folder] = {}

        # Add the problem to the language's submissions
        submissions_by_language[language_folder][f'p{problem_number}'] = []

        # Iterate through each submission file in the language directory
        for submission_file in os.listdir(language_path):
            submission_path = os.path.join(language_path, submission_file)
            submission_code = read_code_file(submission_path)
            submission_number = int(submission_file[1:].split('.')[0])  # Convert 's00001' to 1

            # Add the submission to the problem's submissions
            submissions_by_language[language_folder][f'p{problem_number}'].append({
                f's{submission_number}': submission_code
            })

# Create a JSON file for each programming language
for language, submissions in tqdm(submissions_by_language.items(), desc='Creating JSON files'):
    with open(f'{language}_submissions.json', 'w', encoding='utf-8') as file:
        json.dump(submissions, file, ensure_ascii=False, indent=2)
