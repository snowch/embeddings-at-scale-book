# Code from Chapter 08
# Book: Embeddings at Scale

class StreamingEmbeddingService:
    """
    Production service for real-time dynamic embeddings

    Architecture:
    - Serving layer: Fast reads from current embeddings
    - Update layer: Asynchronous writes from interaction stream
    - Version management: Consistent reads during updates

    Scales to:
    - 100M+ users
    - 1B+ interactions/day
    - <10ms p99 latency for reads
    - <5min lag for updates
    """

    def __init__(self, embedding_model, update_interval=300):
        """
        Args:
            embedding_model: DynamicEmbedding or TemporalUserEmbedding
            update_interval: Seconds between batch updates (default 5min)
        """
        self.model = embedding_model
        self.update_interval = update_interval

        # Interaction queue (in production: Kafka, Kinesis, Pub/Sub)
        self.interaction_queue = []

        # Current embeddings (read-optimized cache)
        self.embedding_cache = {}

        # Last update time
        self.last_update = datetime.now()

    def get_embedding(self, user_id):
        """Fast read from cache"""
        if user_id in self.embedding_cache:
            return self.embedding_cache[user_id]

        # Cache miss: compute and cache
        with torch.no_grad():
            emb = self.model.get_user_embedding(torch.tensor([user_id]))
            self.embedding_cache[user_id] = emb
            return emb

    def record_interaction(self, user_id, item_id, interaction_type):
        """
        Record new interaction (fast, asynchronous)

        In production: Write to message queue, return immediately
        """
        self.interaction_queue.append({
            'user_id': user_id,
            'item_id': item_id,
            'type': interaction_type,
            'timestamp': datetime.now()
        })

    def process_updates(self):
        """
        Batch process accumulated interactions

        Called periodically (every 5-15 minutes)
        In production: Separate worker process
        """
        if not self.interaction_queue:
            return

        print(f"Processing {len(self.interaction_queue)} interactions...")

        # Group by user for efficiency
        user_interactions = {}
        for interaction in self.interaction_queue:
            user_id = interaction['user_id']
            if user_id not in user_interactions:
                user_interactions[user_id] = []
            user_interactions[user_id].append(interaction)

        # Update embeddings
        for user_id, interactions in user_interactions.items():
            for interaction in interactions:
                self.model.update_from_interaction(
                    torch.tensor([user_id]),
                    torch.tensor([interaction['item_id']]),
                    interaction['type']
                )

            # Invalidate cache
            if user_id in self.embedding_cache:
                del self.embedding_cache[user_id]

        # Clear queue
        self.interaction_queue = []
        self.last_update = datetime.now()

        print("Update complete.")
