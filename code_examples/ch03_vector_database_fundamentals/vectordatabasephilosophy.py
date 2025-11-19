# Code from Chapter 03
# Book: Embeddings at Scale

class VectorDatabasePhilosophy:
    """Core philosophical differences from traditional databases"""

    def traditional_db_guarantee(self):
        """Traditional DB: Exact results, guaranteed correctness"""
        return {
            'correctness': '100% - returns exactly what was requested',
            'performance': 'O(log N) with index, O(N) without',
            'use_case': 'Exact match, range queries, transactions'
        }

    def vector_db_guarantee(self):
        """Vector DB: Approximate results, high probability of correctness"""
        return {
            'correctness': '95-99% - returns approximately most similar',
            'performance': 'O(log N) even without perfect accuracy',
            'use_case': 'Semantic similarity, nearest neighbors, recommendations',
            'key_insight': 'Trading small accuracy for massive speed gains'
        }

# Example: Finding top-10 most similar items
# Exact approach: Scan all 256T items - infeasible
# Approximate approach: HNSW index finds 95%+ correct top-10 in <100ms
