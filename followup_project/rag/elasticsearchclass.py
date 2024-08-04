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

    def query_with_embedding(self, index_name, query_vector, top_k):
        knn_query = {
            "size": top_k, # how many top k results you want returned back 
            "query": {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector
                    }
                }
            }

        # search es using knn_query
        response = self.client.search(index=index_name, body=knn_query)

        hits = response['hits']['hits']
        for hit in hits:
            documents = [Document(page_content = hit["_source"]["text"], metadata = {"score": hit['_score']})]
        return documents

    def query_with_text(self, index_name, query_text, top_k):
        knn_query = {
            "size": top_k, # how many top k results you want returned back 
            "query": {
                "match": {
                    "text": query_text
                    }
                }
            }

        # search es using knn_query
        response = self.client.search(index=index_name, body=knn_query)

        hits = response['hits']['hits']
        for hit in hits:
            documents = [Document(page_content = hit["_source"]["text"], metadata = {"score": hit['_score']})]
        return documents

    def get_relevant_documents(self, query: str, top_k: int = 1) -> List[Document]:

        combined_documents = []
        query_vector = self.generate_embeddings(query)
        for index in self.indices:
            documents_embeddings = self.query_with_embedding(index, query_vector, top_k)
            documents_text = self.query_with_text(index, query, top_k)
            combined_documents.extend(documents_embeddings)
            combined_documents.extend(documents_text)
        return combined_documents

        # knn_query = {
        #     "size": top_k, # how many top k results you want returned back 
        #     "query": {
        #     "knn": {
        #         "field": "embedding",
        #         "query_vector": query_vector
        #         }
        #     }
        # }

        # response = self.client.search(index=self.indices, body=knn_query)
        # hits = response['hits']['hits']
        # for hit in hits:
        #     documents = [Document(page_content = hit["_source"]["text"], metadata = {"score": hit['_score']})]
        # return documents
    
     #perform knn search
    
    
    def __call__(self, inputs: dict) -> dict:
        query = inputs.get("question", "")
        documents = self.get_relevant_documents(query)
        context = " ".join([doc.page_content for doc in documents])
        return {"context": context}
