import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Code from Chapter 03
# Book: Embeddings at Scale

# Utility function
def top_k(similarities, k=10):
    """Return indices of top-k similarities. Placeholder implementation."""
    return np.argsort(similarities)[-k:][::-1]

# Traditional database query
def find_customer(database, customer_id):
    """O(log N) with B-tree index"""
    return database.index['customer_id'].lookup(customer_id)
    # 256 trillion rows: ~48 comparisons

# Naive embedding search
def find_similar_naive(query_embedding, all_embeddings):
    """O(N * D) where N=rows, D=dimensions"""
    similarities = []
    for embedding in all_embeddings:  # 256 trillion iterations
        similarity = cosine_similarity(query_embedding, embedding)  # 768 multiplications
        similarities.append(similarity)
    return top_k(similarities, k=10)

# Cost calculation:
# 256 trillion rows × 768 dimensions = 196 quadrillion operations
# At 1 billion ops/second: 6 years per query
