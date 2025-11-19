# Code from Chapter 02
# Book: Embeddings at Scale

# Placeholder cache class
class EmbeddingCache:
    """Simple cache for embeddings. Placeholder implementation."""
    def __init__(self, max_size=10_000_000):
        self.cache = {}
        self.max_size = max_size

    def get(self, key):
        return self.cache.get(key)

    def put(self, key, value):
        if len(self.cache) < self.max_size:
            self.cache[key] = value

class EfficientMultiModalEncoding:
    """Optimize multi-modal encoding costs"""

    def __init__(self):
        # Cache encoded embeddings
        self.embedding_cache = EmbeddingCache(max_size=10_000_000)

        # Batch processing for efficiency
        self.batch_size = 128

    def encode_batch(self, items, modalities=['text', 'image']):
        """Encode multiple items in batch"""
        results = []

        for modality in modalities:
            # Collect all data for this modality
            modality_data = [item.get(modality) for item in items]

            # Check cache
            cached_indices = []
            uncached_data = []
            uncached_indices = []

            for idx, data in enumerate(modality_data):
                cache_key = self.get_cache_key(modality, data)
                cached_emb = self.embedding_cache.get(cache_key)

                if cached_emb is not None:
                    cached_indices.append(idx)
                    results.append((idx, modality, cached_emb))
                else:
                    uncached_data.append(data)
                    uncached_indices.append(idx)

            # Encode uncached data in batch
            if uncached_data:
                if modality == 'text':
                    embeddings = self.text_encoder.encode(uncached_data, batch_size=self.batch_size)
                elif modality == 'image':
                    embeddings = self.image_encoder.encode_batch(uncached_data)

                # Cache and store results
                for idx, emb in zip(uncached_indices, embeddings):
                    cache_key = self.get_cache_key(modality, modality_data[idx])
                    self.embedding_cache.put(cache_key, emb)
                    results.append((idx, modality, emb))

        # Reorganize by item
        items_embeddings = {}
        for idx, modality, emb in results:
            if idx not in items_embeddings:
                items_embeddings[idx] = {}
            items_embeddings[idx][modality] = emb

        return items_embeddings
