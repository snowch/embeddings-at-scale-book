# Code from Chapter 04
# Book: Embeddings at Scale

class BinaryEmbeddings:
    """
    Ultra-compressed binary embeddings for massive scale
    """

    def binarize(self, embeddings):
        """
        Convert float embeddings to binary

        768-dim float32 → 96 bytes
        768-dim binary → 96 bits = 12 bytes (8x compression)
        """
        # Threshold at 0
        binary = (embeddings > 0).astype(np.uint8)

        # Pack into bits
        packed = np.packbits(binary, axis=1)

        return packed

    def hamming_similarity(self, binary1, binary2):
        """
        Ultra-fast similarity using Hamming distance
        """
        # XOR gives differing bits
        xor = np.bitwise_xor(binary1, binary2)

        # Count differing bits
        hamming_dist = np.unpackbits(xor).sum()

        # Convert to similarity
        max_dist = len(binary1) * 8
        similarity = 1 - (hamming_dist / max_dist)

        return similarity


# Binary embeddings enable:
# - 8x storage compression
# - 10-100x faster search (Hamming distance via POPCOUNT instruction)
# - Billion-scale search on single machine
