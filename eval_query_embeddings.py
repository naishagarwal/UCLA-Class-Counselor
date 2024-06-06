from elasticsearch import Elasticsearch, helpers
from transformers import BertModel, BertTokenizer
import torch

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

index_name = "classes_embedding_2"

# delete index if needed
#es.options(ignore_status=[400,404]).indices.delete(index='classes_embedding')

# check if the elasticsearch cluster is up and running
if not es.ping():
    raise ValueError("Connection failed")

# load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# generate embeddings
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# embed query vector
def generate_query_vector(query_text):
    return generate_embeddings(query_text)

### update query vector to whatever search you want to embded
import json

# ingest randomized prompts
with open('random_prompts.json', 'r') as fp:
    prompt_list = json.load(fp)

import random

end_point = 0
prompt_size = 100
prompt_list = prompt_list[750:750+prompt_size]

# create keys and list for output json
keys = [str(item) for item in range(0, len(prompt_list))]
dumps_list = []

# iterate over each individual prompt
for i, prompt in enumerate(prompt_list):
    query_vector = generate_query_vector(prompt_list[i]['user'])
    print(prompt_list[i]['user'])

    knn_size = 5
    # perform knn search
    knn_query = {
        "size": knn_size, # how many top k results you want returned back 
        "query": {
            "knn": {
                "field": "embedding",
                "query_vector": query_vector
                }
            }
        }

    # search es using knn_query
    response = es.search(index=index_name, body=knn_query)

    # print results
    print(f"Total documents retrieved: {len(response['hits']['hits'])}")
    print("Search results:")

    # create and populate results_list
    results_list = []
    for hit in response['hits']['hits']:
        results_list.append(hit["_source"]["text"])

    # create list of dictionaries
    res = dict({keys[i]: results_list})
    dumps_list.append(res)

# write list of dictionaries to file as json
with open("knn_{}.json".format(knn_size), 'a') as fp:  
    #json.dumps(res)
    #json.dump('{},\n'.format(res), fp)
    json.dump(dumps_list, fp, indent=4)