# Code from Chapter 03
# Book: Embeddings at Scale

class HierarchicalNavigation:
    """How hierarchical structures enable fast search"""

    def flat_index_problem(self):
        """Why flat indices don't scale"""
        # Flat index: Compare query with all N vectors
        # Time: O(N * D) where D = embedding dimension
        # For 256T vectors @ 768 dims: ~6 years per query

        return "Infeasible at scale"

    def hierarchical_solution(self):
        """Multi-level navigation reduces comparisons"""

        # Level 0 (coarsest): 1,000 centroids
        # Level 1: 1,000 x 1,000 = 1M centroids
        # Level 2: 1,000 x 1M = 1B centroids
        # Level 3: 1,000 x 1B = 1T vectors
        # Level 4: 256 x 1T = 256T vectors

        # Search process:
        # 1. Compare with 1,000 Level 0 centroids
        # 2. Descend to best centroid's children (1,000 comparisons)
        # 3. Repeat for Level 1, 2, 3
        # Total: ~5,000 comparisons instead of 256 trillion

        comparisons_per_level = 1000
        num_levels = 5
        total_comparisons = comparisons_per_level * num_levels

        return {
            'total_comparisons': total_comparisons,
            'speedup': f'{256_000_000_000_000 / total_comparisons:.2e}x faster',
            'latency': '<100ms vs 6 years'
        }
