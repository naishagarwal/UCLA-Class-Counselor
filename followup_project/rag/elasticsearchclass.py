from elasticsearch import Elasticsearch
from langchain.schema import Document
from transformers import BertModel, BertTokenizer
import torch
from typing import List

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class ElasticsearchRetriever:
    def __init__(self, es_host, es_username, es_password, indices, ca_cert_path):
        self.client = Elasticsearch(
            [es_host],
            basic_auth = (es_username, es_password),
            ca_certs = ca_cert_path,
            verify_certs = False
        )
        self.indices = indices
    
    def generate_embeddings(self, text):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def get_relevant_documents(self, query: str, top_k: int = 10) -> List[Document]:
        query_vector = self.generate_embeddings(query)
        knn_query = {
            "size": top_k, # how many top k results you want returned back 
            "query": {
            "knn": {
                "field": "embedding",
                "query_vector": query_vector
                }
            }
        }

        response = self.client.search(index=self.indices, body=knn_query)
        hits = response['hits']['hits']
        for hit in hits:
            documents = [Document(page_content = hit["_source"]["text"], metadata = {"score": hit['_score']})]
        return documents
    
    def __call__(self, inputs: dict) -> dict:
        query = inputs.get("question", "")
        documents = self.get_relevant_documents(query)
        context = " ".join([doc.page_content for doc in documents])
        return {"context": context}
