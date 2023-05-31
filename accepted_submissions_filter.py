#Author: Erfan Raoofian
#License: Apache 2.0

"""
This script processes JSON files containing embeddings of code submissions.
It goes through each problem in a JSON file and cross-references with
a corresponding CSV file that contains metadata about the submissions.
It filters out the submissions that are not accepted based on the metadata.
After that, it saves the filtered data back into JSON files,
essentially updating the original JSON files to only include accepted submissions.
"""
# Import necessary libraries
import os
import json
import csv
import pandas as pd
from tqdm import tqdm

# Define directories
json_dir = 'embeddings'   # Directory containing JSON files with code embeddings
csv_dir = 'metadata'      # Directory containing metadata about the code submissions
updated_dir = 'updatedJsons'  # Directory to save the updated JSON files

# Create updated directory if it does not exist
os.makedirs(updated_dir, exist_ok=True)

# Get list of JSON files in the 'embeddings' directory
json_files = os.listdir(json_dir)

# Loop through each JSON file
for json_filename in tqdm(json_files, desc='Processing JSON files', unit='file'):
    print(f'Processing {json_filename}...')
    
    # Open the JSON file
    with open(os.path.join(json_dir, json_filename), 'r') as json_file:
        # Load the JSON data
        data = json.load(json_file)

    # Loop through each problem in the JSON data
    for problem_id in list(data.keys()):
        # Format the problem_id to match the CSV filename
        problem_csv_filename = f"p{int(problem_id[1:]):05}.csv"

        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(os.path.join(csv_dir, problem_csv_filename))
        
        # Filter the DataFrame to only rows where the status is "Accepted"
        df = df[df['status'] == 'Accepted']
        
        # Get the set of accepted submission IDs
        accepted_submissions = set(df['submission_id'])
        
        # Filter the submissions in the JSON data to only include accepted ones
        data[problem_id] = [submission for submission in data[problem_id] if list(submission.keys())[0] in accepted_submissions]
        
        # If the problem has no accepted submissions, remove it from the JSON data
        if not data[problem_id]:
            del data[problem_id]

    # Save the updated JSON data in the 'updatedJsons' directory
    with open(os.path.join(updated_dir, json_filename), 'w') as updated_json_file:
        json.dump(data, updated_json_file, indent=2)
