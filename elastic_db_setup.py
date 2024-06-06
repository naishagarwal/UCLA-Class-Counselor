from elasticsearch import Elasticsearch, helpers
import json

### search using https://localhost:9200/<INDEX_NAME>/_search?&from=<STARTING_POINTING>&size=100&pretty

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

response = es.search(
    index="ucla_classes",
    body={
        "query": {
            "match_all": {}
        }
    }
)

# print results
print("Number of results in index: {}".format(response["hits"]["total"]["value"]))
print("Query results:")
for hit in response["hits"]["hits"]:
    print(hit["_source"])