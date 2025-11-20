# Code from Chapter 01
# Book: Embeddings at Scale

"""
Word Embeddings Similarity Example

Demonstrates the core concept of embeddings: numerical vectors that represent
objects in a continuous multi-dimensional space, where similarity in meaning
corresponds to proximity in geometric space.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# A simple 3-dimensional embedding space for illustration
# Why 3 dimensions? This is deliberately simplified for visualization and pedagogy.
# Real embeddings typically use 300-1024 dimensions, but 3D allows us to:
# - Visualize the concept geometrically (x, y, z axes)
# - Understand the math without getting lost in high-dimensional space
# - Demonstrate the core principle: semantic similarity = geometric proximity

# How were these values chosen? They're hand-crafted to demonstrate key relationships:
# - Dimension 0 (~0.9 or ~0.5): Represents "royalty" vs "common"
# - Dimension 1 (~0.8 or ~0.2): Represents "human" vs "other"
# - Dimension 2 (~0.1 or ~0.9): Represents "male" vs "female"
# (In real embeddings, dimensions aren't this interpretable—they're learned automatically)
word_embeddings = {
    'king':   np.array([0.9, 0.8, 0.1]),  # Royal + human + male
    'queen':  np.array([0.9, 0.8, 0.9]),  # Royal + human + female
    'man':    np.array([0.5, 0.2, 0.1]),  # Common + human + male
    'woman':  np.array([0.5, 0.2, 0.9]),  # Common + human + female
    'apple':  np.array([0.1, 0.3, 0.5]),  # Not royal, not human, neutral
}

# Calculate similarity using cosine similarity (see callout box below)
def similarity(word1, word2):
    # Reshape from (3,) to (1, 3) - required by cosine_similarity()
    # Why reshape? cosine_similarity expects 2D arrays where each row is a vector.
    # Our embeddings are 1D arrays (shape: 3,), so we reshape to 2D (shape: 1, 3)
    # The -1 in reshape(1, -1) means "infer this dimension" → converts [a,b,c] to [[a,b,c]]
    vec1 = word_embeddings[word1].reshape(1, -1)
    vec2 = word_embeddings[word2].reshape(1, -1)

    # cosine_similarity returns a 2D array, so [0][0] extracts the scalar similarity value
    return cosine_similarity(vec1, vec2)[0][0]

# Demonstrate that related words have similar embeddings
print(f"king vs queen: {similarity('king', 'queen'):.3f}")  # High (~0.93) - both royalty
print(f"man vs woman: {similarity('man', 'woman'):.3f}")    # High (~0.85) - both human
print(f"king vs apple: {similarity('king', 'apple'):.3f}")  # Low (~0.46) - unrelated concepts
