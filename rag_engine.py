# rag_engine.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from urls import url

# Prepare text list
route_texts = [f"{name.replace('_', ' ').capitalize()}: {path}" for name, path in url.items()]
route_keys = list(url.keys())
route_paths = list(url.values())

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
route_embeddings = embedder.encode(route_texts, convert_to_numpy=True)

# Create FAISS index
dimension = route_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(route_embeddings)

def search_routes(query: str, k=3):
    query_vector = embedder.encode([query])
    _, indices = index.search(np.array(query_vector), k)
    return [(route_keys[i], route_paths[i]) for i in indices[0]]
