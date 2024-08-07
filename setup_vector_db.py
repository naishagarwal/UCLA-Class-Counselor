from elasticsearch import Elasticsearch, helpers
from transformers import BertModel, BertTokenizer
import torch
import numpy as np

### set up elastic index
# elasticsearch connection details
es_host = "https://localhost:9200"
es_username = "elastic"
es_password = "<ES_PASSWORD>"
ca_cert_path = "/usr/local/share/ca-certificates/ca_bundle.crt"

# initialize the Elasticsearch client with authentication
es = Elasticsearch(
    [es_host],
    basic_auth=(es_username, es_password),
    ca_certs = ca_cert_path,
    verify_certs=False
)

# delete index if needed
#es.options(ignore_status=[400,404]).indices.delete(index='embedding_index')


# check if the Elasticsearch cluster is up and running
if not es.ping():
    raise ValueError("Connection failed")

# Define the index settings and mappings
index_name = "embedding_index"
index_settings = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": 768  # dimension of BERT embeddings
            }
        }
    }
}

# create the index
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=index_settings)
    print(f"Index '{index_name}' created.")

    # new settings to update for larger max_result_window
    new_settings = {
        "index": {
            "max_result_window": 20000  
        }
    }

    # update the index settings
    try:
        response = es.indices.put_settings(
            index=index_name,
            body=new_settings
        )
        print(f"Settings updated for index '{index_name}': {response}")
    except Exception as e:
        print(f"Error updating settings for index '{index_name}': {e}")
else:
    print(f"Index '{index_name}' already exists.")

### set up embeddings
# load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# create function for generating embeddings
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

'''
#documents = data

# generate embeddings for each document
documents = [
    "This is a sample document.",
    "This document is another example.",
    "Elasticsearch is a powerful search engine.",
    "Machine learning can generate embeddings."
]

embeddings = [generate_embeddings(doc) for doc in documents]


# function to yield documents in batches
def generate_actions(embeddings, index_name, batch_size=1000):
    for i in range(0, len(embeddings), batch_size):
        embs = embeddings[i:i + batch_size]
        docs = documents[i:i + batch_size]
        for i, emb in enumerate(embs):
            yield {
                "_index": index_name,
                "_source": {
                    "text": docs[i],
                    "embedding": emb
                }
            }


# bulk index the sample data in batches
try:
    helpers.bulk(es, generate_actions(embeddings, index_name))
    print(f"{len(documents)} documents indexed in '{index_name}'.")
except Exception as e:
    print(f"Error during bulk indexing: {e}")
'''


# call function to generate embeddings
def generate_query_vector(query_text):
    return generate_embeddings(query_text)

query_vector = generate_query_vector("Search for machine learning")

# conduct knn search
knn_query = {
    "size": 10,
    "query": {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector
            }
        }
    }

# get results and print
response = es.search(index=index_name, body=knn_query)
print("Search results:")
for hit in response['hits']['hits']:
    print(hit["_source"]["text"])