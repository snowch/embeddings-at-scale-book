# Code from Chapter 03
# Book: Embeddings at Scale

import numpy as np


class GeometricIntuition:
    """Understanding vector similarity through geometry"""

    def demonstrate_similarity_measures(self):
        """Common similarity metrics and their geometric meaning"""

        # Two embeddings
        embedding_a = np.array([0.5, 0.3, 0.8, 0.1])
        embedding_b = np.array([0.6, 0.4, 0.7, 0.2])

        # 1. Euclidean distance (L2)
        # Geometric meaning: Straight-line distance in space
        l2_distance = np.linalg.norm(embedding_a - embedding_b)
        # Use case: When magnitude matters

        # 2. Cosine similarity
        # Geometric meaning: Angle between vectors
        cosine_sim = np.dot(embedding_a, embedding_b) / (
            np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b)
        )
        # Use case: When direction matters more than magnitude
        # Most common for embeddings

        # 3. Inner product (dot product)
        # Geometric meaning: Projection of one vector onto another
        inner_product = np.dot(embedding_a, embedding_b)
        # Use case: When both direction and magnitude matter

        return {
            "l2_distance": l2_distance,
            "cosine_similarity": cosine_sim,
            "inner_product": inner_product,
            "recommendation": "Use cosine similarity for normalized embeddings",
        }

    def curse_of_dimensionality(self, dimensions):
        """Why high dimensions are both blessing and curse"""

        # In high dimensions, most points are far from each other
        # BUT similar points cluster together more distinctly

        # Concentration phenomenon
        # As dimensions increase, distances concentrate around mean

        # Blessing: Easier to find clusters
        # Curse: All distances become similar, making discrimination harder

        return {
            "dimensions": dimensions,
            "challenge": "Distances concentrate in high-D space",
            "solution": "Use index structures that exploit local geometry",
            "key_insight": "Locality-sensitive hashing and graph-based indices",
        }
