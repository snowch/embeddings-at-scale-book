# Code from Chapter 01
# Book: Embeddings at Scale

"""
Simple Embedding-Based Search Engine

A minimal but complete embedding-based search system that demonstrates
semantic search - understanding meaning rather than just matching keywords.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleEmbeddingSearch:
    """A minimal embedding-based search engine"""

    def __init__(self):
        # Load pre-trained embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents):
        """Index documents by creating embeddings"""
        self.documents = documents
        print(f"Creating embeddings for {len(documents)} documents...")
        self.embeddings = self.model.encode(documents, show_progress_bar=True)
        print(f"Indexed {len(documents)} documents")

    def search(self, query, top_k=5):
        """Search for documents similar to query"""
        # Embed the query
        query_embedding = self.model.encode([query])[0]

        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Get top-k results
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({"document": self.documents[idx], "score": similarities[idx]})

        return results


# Example usage
if __name__ == "__main__":
    search_engine = SimpleEmbeddingSearch()

    # Add documents
    documents = [
        "The cat sat on the mat",
        "Dogs are loyal pets",
        "Python is a programming language",
        "Machine learning uses neural networks",
        "Cats and dogs are popular pets",
        "Deep learning is a subset of machine learning",
    ]

    search_engine.add_documents(documents)

    # Search with semantic understanding
    results = search_engine.search("feline animals")

    print("\nQuery: 'feline animals'")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['score']:.3f}] {result['document']}")

    # Expected: Cat-related documents rank highest, even though
    # the word "feline" doesn't appear in any document!
