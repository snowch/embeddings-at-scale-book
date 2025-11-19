import numpy as np

# Code from Chapter 02
# Book: Embeddings at Scale

class EmbeddingQuantization:
    """Quantize embeddings to reduce storage"""

    def quantize_float32_to_int8(self, embeddings):
        """
        float32 (4 bytes) → int8 (1 byte) = 75% storage savings
        """
        # Find min/max for normalization
        min_val = embeddings.min()
        max_val = embeddings.max()

        # Scale to 0-255
        scaled = (embeddings - min_val) / (max_val - min_val) * 255

        # Convert to int8
        quantized = scaled.astype(np.uint8)

        # Store scale factors for dequantization
        scale_factors = {
            'min': min_val,
            'max': max_val
        }

        return quantized, scale_factors

    def dequantize_int8_to_float32(self, quantized, scale_factors):
        """Dequantize back to float32"""
        scaled = quantized.astype(np.float32) / 255
        dequantized = scaled * (scale_factors['max'] - scale_factors['min']) + scale_factors['min']
        return dequantized
