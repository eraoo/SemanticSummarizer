
#Author: Erfan Raoofian
#License: Apache 2.0
"""
This script processes JSON files that contain the average embeddings for
each problem in the CodeNet dataset. The embeddings are vectors of length 768,
which were likely computed from a sentence transformer model.
Each problem also has an associated problem description, which is stored in an HTML file.
The script opens each JSON file and for each problem, it loads the associated HTML file,
extracts the text of the problem description using BeautifulSoup,
and stores the problem ID, problem description, and average embedding in separate lists.
It also prepares the payload data to be pushed to the Qdrant server.
The script batches the data and sends it to the Qdrant server once
it has processed 10 problems. The upsert function is used,
which inserts new data or updates existing data. The data sent includes a unique ID
for each problem (generated using the uuid library), the problem's embedding,
and the payload (problem number and problem description).
if an HTML file does not exist for a problem or if the file cannot be read due to
a UnicodeDecodeError, the problem is skipped and a message is printed to the console.
The skipped problem does not affect the processing of the remaining problems.
"""

# Import necessary libraries
import os
import json
import uuid
from bs4 import BeautifulSoup
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Define directories
average_dir = 'average_embeddings'  # Directory containing JSON files with average embeddings
html_dir = 'problem_descriptions'   # Directory containing HTML files with problem descriptions

# Initialize a Qdrant client
client = QdrantClient(host='localhost', port=6333)

# Recreate the Qdrant collection named "codenet"
print('Creating collection "codenet"...')
client.recreate_collection(
    collection_name='codenet',
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
)

# Get list of JSON files in the 'average_embeddings' directory
json_files = os.listdir(average_dir)

# Loop through each JSON file
for json_filename in tqdm(json_files, desc='Processing JSON files', unit='file'):
    # Initialize lists to store data
    ids = []
    problem_numbers = []
    problem_descriptions = []
    embeddings = []
    payloads = list()

    # Open the JSON file
    with open(os.path.join(average_dir, json_filename), 'r') as json_file:
        # Load the JSON data
        data = json.load(json_file)

    # Loop through each problem in the JSON data
    for problem_id in list(data.keys()):
        # Generate the filename for the corresponding HTML file
        html_filename = os.path.join(html_dir, f"p{int(problem_id[1:]):05}.html")
        
        # If the HTML file exists, open it and extract the problem description
        if os.path.isfile(html_filename):
            try:
                with open(html_filename, 'r', encoding='utf-8') as html_file:
                    soup = BeautifulSoup(html_file, 'html.parser')
                    problem_description = soup.text
            except UnicodeDecodeError:
                problem_description = "#non english problem description#"
        else:
            print(f"HTML file {html_filename} does not exist. Skipping...")
            continue

        # Append the data to the lists
        ids.append(str(uuid.uuid4()))
        problem_numbers.append(problem_id)
        problem_descriptions.append(problem_description)
        embeddings.append(data[problem_id])
        payloads.append({"problem_number": problem_id, "problem_description": problem_description})

        # If we have a batch of 10 problems, push them to the Qdrant collection
        if len(ids) == 10:
            client.upsert(
                collection_name="codenet",
                points=models.Batch(
                    ids=ids,
                    vectors=embeddings,
                    payloads=payloads
                ),
            )
            # Clear the lists
            ids = []
            problem_numbers = []
            problem_descriptions = []
            embeddings = []
            payloads = []

    # If we have a remaining batch smaller than 10, push it to the Qdrant collection
    if ids:
        client.upsert(
            collection_name="codenet",
            points=models.Batch(
                ids=ids,
                vectors=embeddings,
                payload={"problem_numbers": problem_numbers, "problem_descriptions": problem_descriptions}
            ),
        )
