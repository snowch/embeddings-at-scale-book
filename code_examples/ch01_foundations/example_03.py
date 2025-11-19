# Code from Chapter 01
# Book: Embeddings at Scale

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example data for demonstration
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing deals with text data.",
]
query = "What is machine learning?"

# Better, but still term-based
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([query])
similarities = cosine_similarity(query_vector, doc_vectors)
