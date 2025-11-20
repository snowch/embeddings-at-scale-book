# Code from Chapter 01
# Book: Embeddings at Scale

"""
Word Embeddings with SentenceTransformer

Demonstrates how to use pre-trained models to create word embeddings and
measure similarity between words.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings for words/sentences
words = ["cat", "dog", "puppy", "kitten", "automobile", "car"]
embeddings = model.encode(words)

# Find similar words
similarities = cosine_similarity(embeddings)

# 'cat' is most similar to 'kitten'
# 'automobile' is most similar to 'car'
print("Similarity matrix:")
for i, word1 in enumerate(words):
    for j, word2 in enumerate(words):
        if i < j:
            print(f"{word1} ↔ {word2}: {similarities[i][j]:.3f}")
