# Developer: Erfan Raoofian
# License: Apache 2.0
"""
This script takes a directory of JSON files where each file corresponds
to code submissions. This script takes in JSON files containing codes
from the 'jsons' directory,
splits them into smaller chunks (each containing at most 100 problems),
computes embeddings for each chunk, and then saves the embeddings to
new JSON files in the 'embeddings' directory. It does this using
the SentenceTransformer model loaded from the 'models' directory.
"""
import os
import json
import math
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Directory paths for input JSONs and output embeddings
jsons_dir = 'jsons'
embeddings_dir = 'embeddings'
temp_jsons_dir = 'temp_jsons'
model_path = 'sroberta/'

# Load the SentenceTransformer model from the local directory
model = SentenceTransformer(model_path)
#model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")

# Create the necessary directories if they don't exist
if not os.path.exists(embeddings_dir):
    os.makedirs(embeddings_dir)
if not os.path.exists(temp_jsons_dir):
    os.makedirs(temp_jsons_dir)

def get_code_embeddings(codes, batch_size):
    """Computes embeddings for the given codes using the SentenceTransformer model.

    Args:
        codes (list[str]): The codes to compute embeddings for.
        batch_size (int): The size of the batches to split the codes into when computing embeddings.

    Returns:
        list[list[float]]: The computed embeddings.
    """
    code_embeddings = []
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        code_embeddings.extend(model.encode(batch))
    return code_embeddings

# Initial batch size
initial_batch_size = 1024

def process_file(json_file, input_dir, output_dir):
    """Processes a single JSON file by computing embeddings for all codes and saving them to a new JSON file.

    Args:
        json_file (str): The name of the JSON file to process.
        input_dir (str): The directory containing the input JSON file.
        output_dir (str): The directory to save the output JSON file to.
    """
    json_path = os.path.join(input_dir, json_file)

    with open(json_path, 'r', encoding='utf-8') as file:
        submissions_data = json.load(file)

    embeddings_data = {}

    for problem, submissions in tqdm(submissions_data.items(), desc=f'Processing {json_file}'):
        embeddings_data[problem] = []

        codes = []
        submission_ids = []

        for submission in submissions:
            for submission_id, code in submission.items():
                codes.append(code)
                submission_ids.append(submission_id)

        batch_size = initial_batch_size

        while batch_size > 0:
            try:
                code_embeddings = get_code_embeddings(codes, batch_size)

                for idx, code_embedding in enumerate(code_embeddings):
                    embeddings_data[problem].append({
                        submission_ids[idx]: code_embedding.tolist()
                    })

                break
            except RuntimeError:
                batch_size //= 2

    embeddings_file = os.path.join(output_dir, json_file)
    with open(embeddings_file, 'w', encoding='utf-8') as file:
        json.dump(embeddings_data, file, ensure_ascii=False, indent=2)

def split_json(json_file, jsons_dir, temp_jsons_dir):
    """Splits a large JSON file into smaller ones containing at most 100 problems each.

    Args:
        json_file (str): The name of the JSON file to split.
        jsons_dir (str): The directory containing the JSON file.
        temp_jsons_dir (str): The directory to save the smaller JSON files to.
    """
    json_path = os.path.join(jsons_dir, json_file)

    with open(json_path, 'r', encoding='utf-8') as file:
        submissions_data = json.load(file)

    num_problems = len(submissions_data)
    num_parts = math.ceil(num_problems / 100)
    problems_per_part = num_problems
    # The number of problems to include in each part
    problems_per_part = num_problems // num_parts

    # Split the submissions_data into smaller chunks and save each chunk to a new JSON file
    for i in range(num_parts):
        start_idx = i * problems_per_part
        end_idx = start_idx + problems_per_part if i < num_parts - 1 else num_problems
        partial_data = dict(list(submissions_data.items())[start_idx:end_idx])

        temp_json_file = os.path.join(temp_jsons_dir, f'{json_file[:-5]}_part{i}.json')
        with open(temp_json_file, 'w', encoding='utf-8') as temp_file:
            json.dump(partial_data, temp_file, ensure_ascii=False, indent=2)

# Split each JSON file in the 'jsons' directory into smaller parts and compute embeddings for each part
for json_file in os.listdir(jsons_dir):
    split_json(json_file, jsons_dir, temp_jsons_dir)

    # Process each split JSON file
    for temp_json_file in [f for f in os.listdir(temp_jsons_dir) if f.startswith(json_file[:-5])]:
        process_file(temp_json_file, temp_jsons_dir, embeddings_dir)
        # Optionally, remove the temporary JSON file after processing it
        # os.remove(os.path.join(temp_jsons_dir, temp_json_file))
