import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Placeholder: in production, use persistent DB and batch indexing
class SimpleVectorSearch:
    def __init__(self):
        self.texts = []
        self.embeddings = None
        self.index = None
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def add(self, text):
        self.texts.append(text)
        emb = self.embed([text])
        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def embed(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().numpy()

    def search(self, query, k=3):
        q_emb = self.embed([query])
        D, I = self.index.search(q_emb, k)
        return [self.texts[i] for i in I[0]]

vector_db = SimpleVectorSearch()

def search_similar(query):
    return vector_db.search(query)
