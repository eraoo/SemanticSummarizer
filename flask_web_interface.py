
#Developer: Erfan Raoofian
#License: Apache 2.0
"""
This script builds a simple search application with Flask,
which allows you to search through the CodeNet problems using a code snippet as the query. 
When a user submits a search, the application encodes the query into an embedding vector
using a pre-trained model. It then searches the database for the most similar code snippets
based on these embeddings. The application is web-based and runs on a Flask server.
It uses the Qdrant database system to store and retrieve data.
"""



import os
import json
import flask
import logging
import numpy as np
from time import time
from uuid import uuid4
from flask import request
from pprint import pprint
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = flask.Flask('codenet-search')

base_html = '''
<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Science Search!</title>
<!-- Add Bootstrap CSS -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-KyZXEAg3QhqLMpG8r+Knujsl7/1L_dstPt3HV5HzF6Gvk/e3zzE5GgJTszoo8f+" crossorigin="anonymous">
<!-- Custom CSS -->
<style>
  body {
    margin: 20px;
  }

  a {
    text-decoration: none;
    color: #0a2d66;
  }

  a:hover {
    color: #0a2d66;
  }

  .search-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
  }

  .search-container input {
    flex-grow: 1;
    margin-right: 10px;
    height: 80px; /* Make the search bar larger vertically */
  }

  .results-container {
    max-width: 800px; /* Limit the width of the search results */
    margin: 0 auto;
  }

  .result-title {
    margin-bottom: 5px;
  }

  .score {
    margin-bottom: 10px;
  }
</style>
</head>
<body>
<div style="text-align: center; margin-bottom: 30px;">
  <img src="/static/logo.png" alt="Logo">
</div>

<div class="search-container">
  <form action="http://127.0.0.1/" method="post" style="width: 100%;">
    <textarea placeholder="Enter code snippet here" name="search" class="form-control" style="display: inline-block; width: calc(100% - 80px);" rows="3"></textarea>
    <button type="submit" style="background: none; border: none; padding: 0; display: inline-block; width: 80px;"><img src="/static/search.png" alt="Search" style="width: 100%;"></button>
  </form>
</div>

<!-- ... unchanged HTML ... -->
'''
html_close = '''
<!-- Add Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.6/dist/umd/popper.min.js" integrity="sha384-oBqDVmMz4fnFO9gybBud7GEl1QzK7x2Qb4rofzX9qdFh2F1/9R4p8w4j1f8t3L7z" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.min.js" integrity="sha384-cn7l7gDp0eyniUwwAZgrzD06kc/tftFf19TOAs2zVinnD/C7E91j9yyk5//jjpt/" crossorigin="anonymous"></script>
</body>
</html>
'''
# ... rest of the code

@app.route('/', methods=['POST', 'GET'])
def home():
    if flask.request.method == 'GET':
        return base_html + html_close
    elif flask.request.method == 'POST':
        start = time()
        query = flask.request.form['search']
        print('Query received:', query)
        embeddings = model.encode([query])
        vectors = embeddings.tolist()
        results = client.search(
            collection_name='codenet',
            query_vector=vectors[0],
            limit=10)
        pprint(results)
        html = base_html + '<div class="results-container">'
        html = html + '<p>Query time: %s seconds</p>' % (time() - start)
        html = html + '<p>Search query: %s</p>' % query
        for result in results:
            html = html + '<div class="card mb-3">'
            html = html + '<div class="card-body">'
            html = html + '<h2 class="card-title result-title"><a href="problem_descriptions/%s.html" target="_blank">%s</a></h2>' % (result.payload['problem_number'], result.payload['problem_number'])
            html = html + '<p class="score">Score: <b>%s</b></p>' % result.score
            html = html + '<blockquote class="card-text">%s</blockquote>' % result.payload['problem_description'][:500]  # Showing the first 500 characters
            html = html + '</div></div>'
        html = html + '</div>' + html_close
        return html
if __name__ == '__main__':
    model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
    client = QdrantClient(host='localhost', port=6333)
    app.run(host='0.0.0.0', port=80)
