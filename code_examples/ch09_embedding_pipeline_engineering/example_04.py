# Code from Chapter 09
# Book: Embeddings at Scale

import numpy as np


# Placeholder embedding service
class PlaceholderEmbeddingService:
    """Placeholder embedding service. Replace with actual implementation."""
    def get_embedding(self, query, model_version=None):
        return np.random.randn(768).astype(np.float32)

embedding_service = PlaceholderEmbeddingService()

# Allow clients to specify model version explicitly
query_embedding = embedding_service.get_embedding(
    query="...",
    model_version="v1.2.3"  # Pin to specific version
)
