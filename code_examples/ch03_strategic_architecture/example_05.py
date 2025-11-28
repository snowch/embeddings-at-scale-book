# Code from Chapter 02
# Book: Embeddings at Scale

import faiss
import numpy as np


# Placeholder encoder
class PlaceholderTextEncoder:
    """Placeholder text encoder. Replace with actual model."""

    def encode(self, text):
        return np.random.randn(768).astype(np.float32)


text_encoder = PlaceholderTextEncoder()

# Placeholder index
index = faiss.IndexFlatL2(768)

# User query: "red summer dress"
query_embedding = text_encoder.encode("red summer dress")
results = index.search(query_embedding.reshape(1, -1), k=10)
# Returns products with text matching "red summer dress"
# Misses: visually similar dresses described differently
