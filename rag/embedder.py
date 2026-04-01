
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class RAGEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.texts = []

    def build_index(self, texts):
        self.texts = texts

        embeddings = self.model.encode(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)

        self.index.add(np.array(embeddings))

    def search(self, query, top_k=3):
        query_vec = self.model.encode([query])

        distances, indices = self.index.search(np.array(query_vec), top_k)

        results = [self.texts[i] for i in indices[0]]

        return results
