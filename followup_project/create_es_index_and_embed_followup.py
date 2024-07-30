import os
from elasticsearch import Elasticsearch, helpers
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import json

os.environ['CURL_CA_BUNDLE'] = ''

# url for localhost: https://localhost:9200/<INDEX_NAME>/_search?&from=<STARTING_POINTING>&size=<RESULTS_SIZE>&pretty
### set up elastic index
# elasticsearch connection details
es_host = "https://localhost:9200"
es_username = "elastic"
es_password = "<ES_PASSWORD>"
ca_cert_path = "/usr/local/share/ca-certificates/ca_bundle.crt"

# initialize the elasticsearch client w/o authentication
es = Elasticsearch(
    [es_host],
    basic_auth=(es_username, es_password),
    ca_certs = ca_cert_path,
    verify_certs=False
)

# delete index if needed
#es.options(ignore_status=[400,404]).indices.delete(index='classes_embedding')


# check if the elasticsearch cluster is up and running
if not es.ping():
    raise ValueError("Connection failed")

# Define the index settings and mappings
index_name = "f_24_embedding"
index_settings = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": 768  # Dimension of BERT embeddings
            }
        }
    }
}

# create the index
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=index_settings)
    print(f"Index '{index_name}' created.")

    # new settings to update for larger max_result_window (default is 10k)
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

'''
# data to be indexed
with open('cleaned_ucla_class_info.json', 'r') as fp:
    data = json.load(fp)
documents = data

# generate embeddings
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

embeddings = [generate_embeddings(str(doc)) for doc in documents]
'''

# data to be indexed
base_dir = os.path.dirname(os.path.realpath('embedding_input_followup.txt'))
with open(base_dir + '/followup_project/data/fall_2024/embedding_input_followup.txt') as file:
    documents = [line.rstrip() for line in file]

# generate embeddings
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

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
                    "text": str(docs[i]),
                    "embedding": emb
                }
            }

# bulk index the sample data in batches
try:
    helpers.bulk(es, generate_actions(embeddings, index_name))
    print(f"{len(documents)} documents indexed in '{index_name}'.")
except Exception as e:
    print(f"Error during bulk indexing: {e}")