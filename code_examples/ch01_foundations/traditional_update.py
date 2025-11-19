# Code from Chapter 01
# Book: Embeddings at Scale

import numpy as np


# Placeholder functions for demonstration
def train_from_scratch(all_data):
    """Train model from scratch. Placeholder implementation."""
    return None  # Return placeholder model

class PlaceholderEncoder:
    """Placeholder encoder for demonstration. Replace with actual model."""
    def encode(self, data):
        if isinstance(data, str):
            return np.random.randn(768).astype(np.float32)
        else:
            return np.random.randn(len(data), 768).astype(np.float32)

    def fine_tune(self, new_data, existing_embeddings):
        """Fine-tune encoder. Placeholder implementation."""
        pass

encoder = PlaceholderEncoder()

def concatenate(arr1, arr2):
    """Concatenate arrays. Placeholder implementation."""
    return np.concatenate([arr1, arr2], axis=0)

# Traditional approach: retrain everything
def traditional_update(all_data):
    model = train_from_scratch(all_data)  # Expensive, slow
    return model

# Embedding approach: incremental updates
def embedding_update(existing_embeddings, new_data):
    # New items immediately positioned in learned space
    new_embeddings = encoder.encode(new_data)

    # Optional: fine-tune the encoder with new patterns
    encoder.fine_tune(new_data, existing_embeddings)

    # The space evolves without losing accumulated knowledge
    return concatenate(existing_embeddings, new_embeddings)
