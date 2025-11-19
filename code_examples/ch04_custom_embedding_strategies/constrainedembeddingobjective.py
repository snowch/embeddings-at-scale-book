# Code from Chapter 04
# Book: Embeddings at Scale

class ConstrainedEmbeddingObjective:
    """
    Optimize embeddings with hard constraints
    """

    def __init__(self):
        self.primary_objective = 'relevance'  # What we primarily optimize
        self.constraints = [
            {'type': 'diversity', 'threshold': 0.3},  # Min 30% diversity in results
            {'type': 'freshness', 'threshold': 0.5},  # Min 50% results from last 30 days
            {'type': 'price_range', 'threshold': 0.2}  # Min 20% coverage across price ranges
        ]

    def search_with_constraints(self, query, k=20):
        """
        Retrieve results satisfying constraints
        """
        # Initial retrieval (larger set)
        candidates = self.retrieve_candidates(query, k=k*10)  # 10x oversampling

        # Rerank to satisfy constraints
        final_results = self.constrained_reranking(
            candidates,
            constraints=self.constraints,
            k=k
        )

        return final_results

    def constrained_reranking(self, candidates, constraints, k):
        """
        Rerank candidates to satisfy constraints while maximizing primary objective
        """
        selected = []
        remaining = candidates.copy()

        # Greedy selection with constraint checking
        while len(selected) < k and remaining:
            # Find best candidate that maintains constraints
            best_candidate = None
            best_score = -float('inf')

            for candidate in remaining:
                # Check if adding this candidate maintains constraints
                temp_selected = selected + [candidate]
                if self.satisfies_constraints(temp_selected, constraints):
                    if candidate['relevance_score'] > best_score:
                        best_candidate = candidate
                        best_score = candidate['relevance_score']

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                # No candidate satisfies constraints - relax slightly
                break

        # Fill remaining slots if needed
        if len(selected) < k:
            selected.extend(remaining[:k - len(selected)])

        return selected

    def satisfies_constraints(self, selected, constraints):
        """
        Check if selected results satisfy all constraints
        """
        for constraint in constraints:
            if constraint['type'] == 'diversity':
                # Check diversity
                diversity_score = self.compute_diversity(selected)
                if diversity_score < constraint['threshold']:
                    return False

            elif constraint['type'] == 'freshness':
                # Check freshness
                recent_count = sum(
                    1 for item in selected
                    if item['days_since_published'] <= 30
                )
                freshness_ratio = recent_count / len(selected) if selected else 0
                if freshness_ratio < constraint['threshold']:
                    return False

            elif constraint['type'] == 'price_range':
                # Check price range coverage
                price_ranges = set(item['price_tier'] for item in selected)
                range_coverage = len(price_ranges) / 5  # Assuming 5 price tiers
                if range_coverage < constraint['threshold']:
                    return False

        return True
