# Code from Chapter 02
# Book: Embeddings at Scale


class TieredEmbeddingStorage:
    """Implement tiered storage for cost optimization"""

    def __init__(self):
        self.hot_storage = {}  # In-memory (expensive, fast)
        self.warm_storage = {}  # SSD (moderate, medium speed)
        self.cold_storage = {}  # Object storage (cheap, slow)

        self.access_counts = {}

    def get_embedding(self, embedding_id):
        """Retrieve embedding with tiered storage"""
        # Try hot storage first
        if embedding_id in self.hot_storage:
            self.access_counts[embedding_id] += 1
            return self.hot_storage[embedding_id]

        # Try warm storage
        if embedding_id in self.warm_storage:
            emb = self.warm_storage[embedding_id]
            self.access_counts[embedding_id] += 1

            # Promote to hot if frequently accessed
            if self.access_counts[embedding_id] > 100:  # Threshold
                self.promote_to_hot(embedding_id, emb)

            return emb

        # Fall back to cold storage
        if embedding_id in self.cold_storage:
            emb = self.cold_storage[embedding_id]
            self.access_counts[embedding_id] = 1
            return emb

    def tier_management(self):
        """Automatically manage tiers based on access patterns"""
        # Demote infrequently accessed embeddings from hot → warm → cold
        for emb_id, count in self.access_counts.items():
            if count < 10 and emb_id in self.hot_storage:
                # Demote to warm
                self.demote_to_warm(emb_id)
            elif count < 1 and emb_id in self.warm_storage:
                # Demote to cold
                self.demote_to_cold(emb_id)
