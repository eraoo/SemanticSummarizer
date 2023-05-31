#Author: Erfan Raoofian
#License: Apache 2.0
"""
This script processes JSON files containing embeddings of accepted code submissions.
It calculates the average embedding vector for each problem by aggregating all
the embeddings associated with the problem. The average embedding represents
the mean vector of all submission embeddings for a given problem.
The script then replaces the original list of embeddings with this average embedding
in the JSON files, thereby creating a new set of JSON files that contain
average embeddings for each problem.
"""

# Import necessary libraries
import os
import json
import numpy as np
from tqdm import tqdm

# Define directories
source_dir = 'updatedJsons'     # Directory containing the filtered JSON files
output_dir = 'average_embeddings'  # Directory to save the updated JSON files with average embeddings

# Create average directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Get list of JSON files in the 'updatedJsons' directory
json_files = os.listdir(source_dir)

# Loop through each JSON file
for json_filename in tqdm(json_files, desc='Processing JSON files', unit='file'):
    # Open the JSON file
    with open(os.path.join(source_dir, json_filename), 'r') as json_file:
        # Load the JSON data
        data = json.load(json_file)

    # Loop through each problem in the JSON data
    for problem_id in list(data.keys()):
        # Collect all embeddings associated with the problem
        embeddings = [list(submission.values())[0] for submission in data[problem_id]]
        
        # Calculate the average embedding across all submissions for the problem
        average_embedding = list(np.mean(embeddings, axis=0))
        
        # Replace the list of embeddings with the average embedding
        data[problem_id] = average_embedding

    # Save the updated JSON data in the 'average_embeddings' directory
    with open(os.path.join(output_dir, json_filename), 'w') as average_json_file:
        json.dump(data, average_json_file, indent=2)
