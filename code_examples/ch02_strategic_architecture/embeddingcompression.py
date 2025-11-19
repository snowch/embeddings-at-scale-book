# Code from Chapter 02
# Book: Embeddings at Scale


class EmbeddingCompression:
    """Advanced compression techniques"""

    def product_quantization(self, embeddings, num_subvectors=8, bits_per_subvector=8):
        """
        Product Quantization: decompose embeddings into subvectors
        Example: 768-dim float32 (3,072 bytes) → 8 bytes = 384x compression
        """
        import faiss

        dim = embeddings.shape[1]
        dim // num_subvectors

        # Train product quantizer
        pq = faiss.IndexPQ(dim, num_subvectors, bits_per_subvector)
        pq.train(embeddings)

        # Encode
        codes = pq.sa_encode(embeddings)

        # Calculate compression ratio
        # Original: dim * 4 bytes (float32)
        # Compressed: num_subvectors * (bits_per_subvector / 8) bytes
        bytes_per_code = (num_subvectors * bits_per_subvector) / 8
        compression_ratio = (dim * 4) / bytes_per_code

        return {
            "codes": codes,
            "quantizer": pq,
            "compression_ratio": compression_ratio,
            "storage_savings": 1 - (1 / compression_ratio),
        }
