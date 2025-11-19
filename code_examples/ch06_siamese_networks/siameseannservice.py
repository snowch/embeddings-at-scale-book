import torch.nn.functional as F

# Code from Chapter 06
# Book: Embeddings at Scale

class SiameseANNService:
    """
    Siamese network integrated with Approximate Nearest Neighbor search

    For production scale:
    - Embed items once, store in ANN index
    - Sub-millisecond similarity search across billions of items
    - Periodic index updates without downtime
    """

    def __init__(self, siamese_service, embedding_dim=512):
        self.siamese_service = siamese_service
        self.embedding_dim = embedding_dim

        # Use FAISS for ANN search
        try:
            import faiss

            # Create index: Inner product (for normalized embeddings = cosine similarity)
            self.index = faiss.IndexFlatIP(embedding_dim)

            # For very large scale, use IVF or HNSW
            # self.index = faiss.IndexIVFFlat(
            #     faiss.IndexFlatIP(embedding_dim),
            #     embedding_dim,
            #     n_lists=100
            # )

        except ImportError:
            print("FAISS not installed. Install with: pip install faiss-cpu")
            self.index = None

        self.id_to_index = {}  # Map item IDs to index positions
        self.index_to_id = {}  # Map index positions to item IDs

    def add_items(self, item_ids, items):
        """
        Add items to search index

        Args:
            item_ids: List of item identifiers
            items: Batch of items to embed and index
        """
        if self.index is None:
            raise RuntimeError("FAISS not available")

        # Get embeddings
        embeddings = self.siamese_service.get_embeddings_batch(items)

        # Normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()

        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings)

        # Update mappings
        for i, item_id in enumerate(item_ids):
            idx = start_idx + i
            self.id_to_index[item_id] = idx
            self.index_to_id[idx] = item_id

    def search(self, query, top_k=10):
        """
        Search for similar items

        Args:
            query: Query item
            top_k: Number of results

        Returns:
            List of (item_id, similarity) tuples
        """
        if self.index is None:
            raise RuntimeError("FAISS not available")

        # Get query embedding
        query_embedding = self.siamese_service.get_embedding(query)
        query_embedding = F.normalize(query_embedding, p=2, dim=1).cpu().numpy()

        # Search index
        similarities, indices = self.index.search(query_embedding, top_k)

        # Convert to item IDs
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx in self.index_to_id:
                item_id = self.index_to_id[idx]
                results.append((item_id, float(sim)))

        return results

    def update_item(self, item_id, new_item):
        """
        Update an item in the index

        Note: FAISS doesn't support in-place updates. For production,
        use a write-ahead log and periodic full rebuilds.
        """
        if item_id not in self.id_to_index:
            raise ValueError(f"Item {item_id} not in index")

        # For simplicity, we'll remove and re-add
        # In production, batch updates and rebuild index periodically
        print("Warning: Item updates require index rebuild in production")

    def get_statistics(self):
        """Get index statistics"""
        return {
            'total_items': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embedding_dim,
            'cache_stats': self.siamese_service.get_cache_stats()
        }
