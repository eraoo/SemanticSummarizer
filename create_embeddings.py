# Developer: Erfan Raoofian
# License: Apache 2.0
"""
This script takes a directory of JSON files where each file corresponds
to code submissions. It creates embeddings for each code snippet using
the SentenceTransformer model. If a memory error occurs while generating
the embeddings, the batch size is halved until the process is successful.
The generated embeddings are then saved to JSON files in the 'embeddings'
directory. Each problem has an associated list of embeddings, where each
embedding corresponds to a code submission.
"""
import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# The model to be used for generating embeddings
model_path = "models/"
model_name = "st-codesearch-distilroberta-base"

# Load the model from the local directory
model = SentenceTransformer(model_path)

# Path to the directory containing JSON files of code submissions
jsons_dir = 'jsons'

# Directory to store generated embeddings
embeddings_dir = 'embeddings'

# Create the embeddings directory if it doesn't exist
if not os.path.exists(embeddings_dir):
    os.makedirs(embeddings_dir)

# Function to encode the code and return the embeddings
def get_code_embeddings(codes, batch_size):
    """Generates embeddings for given codes using the SentenceTransformer model.

    Args:
        codes (list): List of code snippets.
        batch_size (int): Size of the batch to process at a time.

    Returns:
        list: List of embeddings generated for the code snippets.
    """
    code_embeddings = []
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        code_embeddings.extend(model.encode(batch))
    return code_embeddings

initial_batch_size = 1024

# Iterate through each JSON file in the 'jsons' folder
for json_file in os.listdir(jsons_dir):
    json_path = os.path.join(jsons_dir, json_file)

    # Load the submissions data from the JSON file
    with open(json_path, 'r', encoding='utf-8') as file:
        submissions_data = json.load(file)

    # Initialize a new dictionary for the embeddings
    embeddings_data = {}

    # Iterate through each problem and its submissions
    for problem, submissions in tqdm(submissions_data.items(), desc=f'Processing {json_file}'):
        embeddings_data[problem] = []

        codes = []
        submission_ids = []

        # Iterate through each submission and get its code
        for submission in submissions:
            for submission_id, code in submission.items():
                codes.append(code)
                submission_ids.append(submission_id)

        # Initialize batch size
        batch_size = initial_batch_size

        # Process the codes in batches
        while batch_size > 0:
            try:
                # Get the code embeddings
                code_embeddings = get_code_embeddings(codes, batch_size)

                # Add the code embeddings to the problem's embeddings
                for idx, code_embedding in enumerate(code_embeddings):
                    embeddings_data[problem].append({
                        submission_ids[idx]: code_embedding.tolist()  # Convert ndarray to list
                    })

                # Break the loop if successful
                break
            except RuntimeError:
                # Reduce the batch size if OOM error occurs
                batch_size //= 2

    # Save the embeddings data to a new JSON file in the 'embeddings' folder
    embeddings_file = os.path.join(embeddings_dir, json_file)
    with open(embeddings_file, 'w', encoding='utf-8') as file:
        json.dump(embeddings_data, file, ensure_ascii=False, indent=2)
