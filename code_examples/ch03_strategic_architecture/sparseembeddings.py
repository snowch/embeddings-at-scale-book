import numpy as np

# Code from Chapter 02
# Book: Embeddings at Scale


class SparseEmbeddings:
    """Sparse embedding optimization"""

    def densify_top_k(self, embedding, k=64):
        """Keep only top-k values, zero out rest"""
        # Find top-k indices
        top_k_indices = np.argsort(np.abs(embedding))[-k:]

        # Create sparse embedding
        sparse = np.zeros_like(embedding)
        sparse[top_k_indices] = embedding[top_k_indices]

        sparsity = 1 - (k / len(embedding))

        return {
            "sparse_embedding": sparse,
            "sparsity": sparsity,
            "storage_savings": sparsity,  # Can use sparse storage format
        }
