# Code from Chapter 06
# Book: Embeddings at Scale

class SiameseEmbeddingService:
    """
    Production service for Siamese network embeddings

    Key features:
    - Embedding caching to avoid recomputation
    - Batch processing for efficiency
    - Health monitoring and fallbacks
    - GPU/CPU flexibility
    """

    def __init__(
        self,
        model,
        cache_size=100000,
        batch_size=256,
        device='cuda'
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size

        # LRU cache for embeddings
        from functools import lru_cache
        import hashlib

        self.embedding_cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_cache_key(self, item):
        """Generate cache key for item"""
        # Convert item to bytes and hash
        item_bytes = item.cpu().numpy().tobytes()
        return hashlib.md5(item_bytes).hexdigest()

    def get_embedding(self, item, use_cache=True):
        """
        Get embedding for single item

        Args:
            item: Input to embed
            use_cache: Whether to use cache

        Returns:
            Embedding tensor
        """
        if use_cache:
            cache_key = self._get_cache_key(item)

            if cache_key in self.embedding_cache:
                self.cache_hits += 1
                return self.embedding_cache[cache_key]

            self.cache_misses += 1

        # Compute embedding
        with torch.no_grad():
            embedding = self.model.get_embedding(item.to(self.device))

        # Store in cache
        if use_cache:
            if len(self.embedding_cache) >= self.cache_size:
                # Remove oldest entry (FIFO)
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]

            self.embedding_cache[cache_key] = embedding.cpu()

        return embedding

    def get_embeddings_batch(self, items):
        """
        Get embeddings for batch of items efficiently

        Args:
            items: Batch of inputs (N, ...)

        Returns:
            Embeddings (N, embedding_dim)
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]

            with torch.no_grad():
                batch_embeddings = self.model.get_embedding(batch.to(self.device))

            embeddings.append(batch_embeddings.cpu())

        return torch.cat(embeddings, dim=0)

    def compare(self, item1, item2):
        """
        Compare two items

        Returns:
            Similarity score (higher = more similar)
        """
        embedding1 = self.get_embedding(item1)
        embedding2 = self.get_embedding(item2)

        # Cosine similarity
        similarity = F.cosine_similarity(
            embedding1,
            embedding2,
            dim=0
        ).item()

        return similarity

    def find_similar(self, query, candidates, top_k=10):
        """
        Find most similar candidates to query

        Args:
            query: Query item
            candidates: List of candidate items
            top_k: Number of results to return

        Returns:
            Indices and similarities of top-k candidates
        """
        query_embedding = self.get_embedding(query)
        candidate_embeddings = self.get_embeddings_batch(
            torch.stack(candidates)
        )

        # Compute similarities
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            candidate_embeddings,
            dim=1
        )

        # Get top-k
        top_k_sims, top_k_indices = torch.topk(similarities, k=min(top_k, len(candidates)))

        return top_k_indices.tolist(), top_k_sims.tolist()

    def get_cache_stats(self):
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            'cache_size': len(self.embedding_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }
