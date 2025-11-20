# Code from Chapter 01
# Book: Embeddings at Scale

"""
Vector Analogy Function

Demonstrates how mathematical operations on vectors correspond to semantic operations.
The famous example: king - man + woman ≈ queen

Note: This requires embeddings trained on large datasets to work reliably.
"""

from scipy.spatial.distance import cosine


def vector_analogy(a, b, c, embeddings):
    """Solve: a is to b as c is to ?"""
    # a - b + c
    result_vector = embeddings[a] - embeddings[b] + embeddings[c]

    # Find closest word to result_vector
    closest_word = None
    closest_distance = float('inf')

    for word, vec in embeddings.items():
        if word in [a, b, c]:  # Skip input words
            continue
        dist = cosine(result_vector, vec)
        if dist < closest_distance:
            closest_distance = dist
            closest_word = word

    return closest_word

# Examples with properly trained embeddings:
# vector_analogy('king', 'man', 'woman') → 'queen'
# vector_analogy('Paris', 'France', 'Germany') → 'Berlin'
# vector_analogy('swimming', 'swimmer', 'running') → 'runner'

if __name__ == '__main__':
    print("Vector analogy demonstration")
    print("Note: This function requires properly trained embeddings to work")
    print("Examples that work with real word embeddings:")
    print("  king - man + woman ≈ queen")
    print("  Paris - France + Germany ≈ Berlin")
    print("  swimming - swimmer + running ≈ runner")
