# SemanticSummarizer
A course project leveraging sentence transformers and large language models to automate code summarization tasks. The project aims to enhance code comprehension by creating meaningful and context-rich summaries from a diverse collection of code examples.

This project aims to leverage machine learning techniques to provide automated code summarization, generating concise and informative code summaries from the IBM's Project CodeNet dataset.

## Repository Structure

The repository contains Python scripts for data processing, creating embeddings, filtering, and averaging embeddings. Here is a brief description of each script:

- `create_json_for_each_language.py`: Organizes code files into a dictionary and saves them into a JSON file for each programming language.
- `create_embeddings_large_files.py`: Splits JSON files into smaller chunks, computes embeddings for each chunk, and saves the embeddings to new JSON files.
- `accepted_submissions_filter.py`: Filters out non-accepted submissions based on metadata and updates the original JSON files to only include accepted submissions.
- `average_embeddings.py`: Calculates the average embedding vector for each problem by aggregating all the embeddings associated with the problem and creates a new set of JSON files.
- `insert_qdrant.py`: Processes JSON files containing average embeddings for each problem and sends the data to the Qdrant server.
- `flask_code_search.py`: Builds a simple search application with Flask, allowing you to search through the CodeNet problems using a code snippet as the query.

## Prerequisites

Before using the scripts, make sure to download the Project CodeNet dataset from [here](link_to_dataset) and store it in a directory named `data` at the root of the project repository.

Ensure that you have the following installed:

- Python 3.7+
- Flask
- BeautifulSoup
- SentenceTransformer
- Qdrant

You can install these packages with pip:

```bash
pip install flask beautifulsoup4 sentence-transformers qdrant
```

## How to Use this Repository

1. Clone the repository:

   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```