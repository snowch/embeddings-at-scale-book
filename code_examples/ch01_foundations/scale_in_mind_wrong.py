# Code from Chapter 01
# Book: Embeddings at Scale

import faiss
import numpy as np

embeddings = np.load("embeddings.npy")  # Doesn't scale
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)  # In-memory only
index.add(embeddings)
