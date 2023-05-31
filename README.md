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

Before using the scripts, make sure to download the Project CodeNet dataset from [here](https://developer.ibm.com/exchanges/data/all/project-codenet/) and store it in a directory named `data` at the root of the project repository.

Ensure that you have the following installed:

- Python 3.7+
- Flask
- BeautifulSoup
- SentenceTransformer
- Qdrant
- numpy
- tqdm
- pandas

You can install these packages with pip:

```bash
pip install numpy tqdm sentence-transformers flask qdrant qdrant-client beautifulsoup4 pandas

```

## How to Use this Repository

1. Clone the repository:

   ```bash
   git clone https://github.com/eraoo/SemanticSummarizer.git
   cd SemanticSummarizer
   ```



2. Download and store the CodeNet dataset:

Download the dataset from the link provided in the Prerequisites section.
Once downloaded, extract the contents of the zip file and place them in a directory named data.
Create JSON files for each language:

```bash
python create_json_for_each_language.py
```

4. Create embeddings for code submissions:
```bash
python create_embeddings_large_files.py
```

5. Filter out non-accepted submissions:
```bash
python only_accepted.py
```

6. Compute average embeddings:
```bash
python average_embeddings.py
```

7. Insert data into the Qdrant server:
```bash
python insert_qdrant.py
```

8. Launch the Flask application:
```bash
python flask_web_interface.py
```

Then, open a web browser and navigate to http://localhost:5000 to use the application.

Remember, each step is dependent on the previous ones, so ensure you run the scripts in the order mentioned above.

## Contributing
We welcome contributions to this project! Please feel free to submit issues for bug reporting or enhancements.