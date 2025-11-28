# Code from Chapter 04
# Book: Embeddings at Scale

import numpy as np


# Placeholder encoder classes for different tiers
class HighDimEncoder:
    """Placeholder high-dimensional encoder. Replace with actual model."""

    def __init__(self, dim=768):
        self.dim = dim

    def encode(self, item):
        return np.random.randn(self.dim).astype(np.float32)


class MediumDimEncoder:
    """Placeholder medium-dimensional encoder. Replace with actual model."""

    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, item):
        return np.random.randn(self.dim).astype(np.float32)


class LowDimEncoder:
    """Placeholder low-dimensional encoder. Replace with actual model."""

    def __init__(self, dim=128):
        self.dim = dim

    def encode(self, item):
        return np.random.randn(self.dim).astype(np.float32)


class TieredEmbeddings:
    """
    Different embedding dimensions for different data tiers
    """

    def __init__(self):
        self.hot_encoder = HighDimEncoder(dim=768)  # Frequent queries
        self.warm_encoder = MediumDimEncoder(dim=384)  # Moderate queries
        self.cold_encoder = LowDimEncoder(dim=128)  # Rare queries

    def encode_with_tier(self, item, access_frequency):
        """
        Encode with appropriate dimension based on access frequency
        """
        if access_frequency > 1000:  # >1000 queries/day
            # Hot tier: high quality, high cost justified
            return self.hot_encoder.encode(item), "hot"
        elif access_frequency > 10:
            # Warm tier: good quality, moderate cost
            return self.warm_encoder.encode(item), "warm"
        else:
            # Cold tier: acceptable quality, low cost
            return self.cold_encoder.encode(item), "cold"


# Cost savings:
# - 90% of embeddings in cold tier (128-dim): 83% storage savings
# - 9% in warm tier (384-dim): 50% savings
# - 1% in hot tier (768-dim): full quality
# - Overall: ~80% storage cost reduction
