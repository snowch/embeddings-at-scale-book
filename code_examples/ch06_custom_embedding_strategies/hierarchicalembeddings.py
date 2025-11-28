# Code from Chapter 04
# Book: Embeddings at Scale

import numpy as np


# Placeholder encoder class
class Encoder:
    """Placeholder encoder. Replace with actual model."""

    def __init__(self, dim=512):
        self.dim = dim

    def encode(self, text):
        return np.random.randn(self.dim).astype(np.float32)


class HierarchicalEmbeddings:
    """
    Preserve hierarchical structure in embedding space
    """

    def __init__(self):
        self.level_encoders = {
            "category": Encoder(dim=256),  # Coarse level
            "subcategory": Encoder(dim=512),  # Medium level
            "product": Encoder(dim=768),  # Fine level
        }

    def encode_hierarchical(self, item, level="product"):
        """
        Encode at different hierarchy levels

        Example:
          Category: "Electronics"
          Subcategory: "Smartphones"
          Product: "iPhone 15 Pro Max 256GB"
        """
        embeddings = {}

        # Encode at each level in hierarchy
        for level_name in ["category", "subcategory", "product"]:
            if level_name in item:
                embeddings[level_name] = self.level_encoders[level_name].encode(item[level_name])

            # Stop at requested level
            if level_name == level:
                break

        return embeddings

    def hierarchical_search(self, query, level="product"):
        """
        Search at appropriate hierarchy level

        Coarse queries ("electronics") match at category level
        Fine queries ("iphone 15 pro max") match at product level
        """
        # Classify query specificity
        query_level = self.infer_query_level(query)

        # Encode at appropriate level
        query_emb = self.level_encoders[query_level].encode(query)

        # Search at that level
        results = self.search_at_level(query_emb, level=query_level)

        return results
