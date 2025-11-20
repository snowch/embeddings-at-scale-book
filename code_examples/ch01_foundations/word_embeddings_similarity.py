# Code from Chapter 01
# Book: Embeddings at Scale

"""
Word Embeddings Similarity Example

Demonstrates the core concept of embeddings: numerical vectors that represent
objects in a continuous multi-dimensional space, where similarity in meaning
corresponds to proximity in geometric space.
"""

from scipy.spatial.distance import cosine

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

# fmt: off
word_embeddings = {
    "king":  [0.9, 0.8, 0.1],  # Royal + human + male
    "queen": [0.9, 0.8, 0.9],  # Royal + human + female
    "man":   [0.5, 0.8, 0.1],  # Common + human + male
    "woman": [0.5, 0.8, 0.9],  # Common + human + female
    "apple": [0.1, 0.3, 0.5],  # Not royal, not human, neutral
}
# fmt: on


# Calculate similarity using cosine similarity
# cosine() returns distance (0 = identical), so we convert to similarity (1 = identical)
def similarity(word1, word2):
    """Higher similarity = more similar concepts (range: -1 to 1, typically 0 to 1)"""
    return 1 - cosine(word_embeddings[word1], word_embeddings[word2])


# Demonstrate that related words have similar embeddings
print(f"king vs queen: {similarity('king', 'queen'):.3f}")  # High (~0.85) - both royalty
print(f"man vs woman: {similarity('man', 'woman'):.3f}")  # High (~0.80) - both human
print(f"king vs apple: {similarity('king', 'apple'):.3f}")  # Low (~0.53) - unrelated concepts
