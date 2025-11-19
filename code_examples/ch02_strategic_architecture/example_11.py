# Code from Chapter 02
# Book: Embeddings at Scale

import faiss
import numpy as np


# Placeholder encoder with multiple encoding methods
class MultiModalEncoder:
    """Placeholder multi-modal encoder. Replace with actual model."""
    def encode_text(self, text):
        return np.random.randn(768).astype(np.float32)

    def encode_image(self, image):
        return np.random.randn(768).astype(np.float32)

encoder = MultiModalEncoder()

# Placeholder index with multimodal search
class PlaceholderIndex:
    """Placeholder index. Replace with actual implementation."""
    def __init__(self):
        self.index = faiss.IndexFlatL2(768)

    def search_multimodal(self, query_embs, modality_weights=None, k=10):
        # Simple placeholder: just use first embedding
        emb = list(query_embs.values())[0]
        distances, indices = self.index.search(emb.reshape(1, -1), k)
        return [{'idx': idx, 'distance': dist} for idx, dist in zip(indices[0], distances[0])]

index = PlaceholderIndex()

# Example image path
uploaded_image = "example_uploaded.jpg"

# Image query
image_emb = encoder.encode_image(uploaded_image)

# Initial results
initial_results = index.search_multimodal({'image': image_emb}, k=100)

# Text refinement
text_emb = encoder.encode_text("in blue")

# Combined query
refined_results = index.search_multimodal(
    {'image': image_emb, 'text': text_emb},
    modality_weights={'image': 0.7, 'text': 0.3},  # Image is primary
    k=20
)
