# Code from Chapter 01
# Book: Embeddings at Scale

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
