import numpy as np
import torch

# Code from Chapter 04
# Book: Embeddings at Scale


# Placeholder encoder classes for temporal embeddings
class StaticEncoder:
    """Placeholder static encoder. Replace with actual model."""

    def __init__(self, dim):
        self.dim = dim

    def encode(self, content):
        return torch.randn(self.dim)


class TimeEncoder:
    """Placeholder time encoder. Replace with actual model."""

    def __init__(self, dim):
        self.dim = dim

    def encode(self, timestamp):
        return torch.randn(self.dim)


class TemporalEmbeddings:
    """
    Handle time-varying embeddings
    """

    def __init__(self, embedding_dim=512, time_encoding_dim=64):
        self.static_encoder = StaticEncoder(embedding_dim - time_encoding_dim)
        self.time_encoder = TimeEncoder(time_encoding_dim)
        self.embedding_dim = embedding_dim

    def encode_with_time(self, content, timestamp):
        """
        Encode content with temporal context
        """
        # Static content embedding
        static_emb = self.static_encoder.encode(content)

        # Time encoding (positional encoding or learned)
        time_emb = self.time_encoder.encode(timestamp)

        # Concatenate
        temporal_emb = torch.cat([static_emb, time_emb], dim=-1)

        return temporal_emb

    def time_decayed_similarity(self, query_time, document_time, document_emb):
        """
        Adjust similarity based on temporal distance
        """
        time_diff_days = abs((query_time - document_time).days)

        # Exponential decay: more recent = more relevant
        decay_factor = np.exp(-time_diff_days / 180)  # 180-day half-life

        return document_emb * decay_factor


# Domains requiring temporal awareness:
temporal_use_cases = [
    {
        "domain": "News Search",
        "requirement": "Recent articles more relevant for most queries",
        "approach": "Time decay on similarity scores",
    },
    {
        "domain": "Social Media",
        "requirement": "Trending topics change rapidly",
        "approach": "Short-window embeddings, frequent retraining",
    },
    {
        "domain": "Fashion/Trends",
        "requirement": "Style similarity depends on current trends",
        "approach": "Time-conditioned embeddings, seasonal retraining",
    },
    {
        "domain": "Scientific Research",
        "requirement": "Paradigm shifts change what's similar",
        "approach": "Period-specific embeddings (pre/post major discoveries)",
    },
]
