# Code from Chapter 01
# Book: Embeddings at Scale

"""
Semantic Distance Function

Demonstrates how geometric distance between embeddings reflects semantic similarity.
Smaller distance indicates more similar concepts.
"""

from scipy.spatial.distance import cosine

def semantic_distance(word1, word2, embeddings):
    """Smaller distance = more similar concepts"""
    return cosine(embeddings[word1], embeddings[word2])

# Example usage (requires embeddings dictionary)
if __name__ == '__main__':
    import numpy as np

    # Example embeddings
    embeddings = {
        'cat':    [0.8, 0.6, 0.1, 0.2],  # Close to 'kitten'
        'kitten': [0.8, 0.5, 0.2, 0.3],  # Close to 'cat'
        'dog':    [0.7, 0.6, 0.1, 0.8],  # Close to 'puppy', related to 'cat'
        'puppy':  [0.7, 0.5, 0.2, 0.9],  # Close to 'dog'
        'car':    [0.1, 0.2, 0.9, 0.1],  # Far from animals
    }

    # Animals are close to each other
    print(f"cat ↔ dog: {semantic_distance('cat', 'dog', embeddings):.3f}")

    # Animals far from vehicles
    print(f"cat ↔ car: {semantic_distance('cat', 'car', embeddings):.3f}")
