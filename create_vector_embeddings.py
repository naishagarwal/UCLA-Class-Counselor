from transformers import BertModel, BertTokenizer
import torch
import numpy as np

# load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# generate embeddings for each document
documents = [
    "This is a sample document.",
    "This document is another example.",
    "Elasticsearch is a powerful search engine.",
    "Machine learning can generate embeddings."
]
embeddings = [generate_embeddings(doc) for doc in documents]

print(embeddings)