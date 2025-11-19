import numpy as np

# Code from Chapter 08
# Book: Embeddings at Scale


class HierarchicalProductRecommender:
    """
    Product recommendations that respect category structure

    Benefits:
    - Diversification: Recommend across subcategories
    - Substitution: Find products at same hierarchy level
    - Upselling: Navigate up taxonomy for premium alternatives
    - Cross-selling: Navigate to related branches
    """

    def __init__(self, hyperbolic_embeddings, product_metadata):
        self.embeddings = hyperbolic_embeddings
        self.metadata = product_metadata

    def recommend_substitutes(self, product_id, top_k=10):
        """
        Find substitute products (same level in hierarchy)

        E.g., gaming_laptop → [other gaming laptops]
        """
        # Find items at similar distance from root
        # (same hierarchy level often correlates with distance from origin)
        emb = self.embeddings.get_embedding(product_id)
        depth = np.linalg.norm(emb)  # Distance from origin ≈ depth

        similar = self.embeddings.find_similar(product_id, top_k * 3)

        # Filter to items at similar depth
        substitutes = []
        for item, dist in similar:
            item_emb = self.embeddings.get_embedding(item)
            item_depth = np.linalg.norm(item_emb)

            if abs(depth - item_depth) < 0.1:  # Similar depth
                substitutes.append((item, dist))

            if len(substitutes) >= top_k:
                break

        return substitutes

    def recommend_upsell(self, product_id, budget_multiplier=1.5):
        """
        Find premium alternatives (move up in hierarchy, then down to premium branch)

        E.g., business_laptop → ultrabook (higher-end subcategory)
        """
        # Move toward origin (up hierarchy), then explore nearby branches
        # that are associated with higher price points
        pass  # Implementation depends on price metadata
