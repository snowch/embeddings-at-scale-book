"""Evaluate different chunk sizes for retrieval quality."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EvaluationResult:
    """Results from chunk size evaluation."""
    chunk_size: int
    num_chunks: int
    avg_chunk_length: float
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    avg_relevance_score: float


def evaluate_chunk_sizes(
    documents: List[str],
    queries: List[str],
    ground_truth: List[List[int]],  # For each query, list of relevant doc indices
    chunk_sizes: Optional[List[int]] = None,
    overlap_ratio: float = 0.1,
    top_k: int = 5
) -> List[EvaluationResult]:
    """
    Evaluate retrieval quality across different chunk sizes.

    Args:
        documents: List of documents to chunk and index
        queries: List of test queries
        ground_truth: For each query, indices of relevant documents
        chunk_sizes: List of chunk sizes to evaluate
        overlap_ratio: Overlap as fraction of chunk size
        top_k: Number of results to retrieve

    Returns:
        List of EvaluationResult for each chunk size
    """
    from fixed_size_chunking import chunk_by_tokens
    from sentence_transformers import SentenceTransformer

    if chunk_sizes is None:
        chunk_sizes = [128, 256, 512, 1024]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    results = []

    for chunk_size in chunk_sizes:
        overlap = int(chunk_size * overlap_ratio)

        # Chunk all documents
        all_chunks = []
        chunk_to_doc = []  # Map chunk index to source document

        for doc_idx, doc in enumerate(documents):
            doc_chunks = chunk_by_tokens(doc, chunk_size, overlap)
            for chunk in doc_chunks:
                all_chunks.append(chunk)
                chunk_to_doc.append(doc_idx)

        # Embed all chunks
        chunk_embeddings = model.encode(all_chunks)

        # Evaluate retrieval for each query
        precisions = []
        recalls = []
        relevance_scores = []

        for query, relevant_docs in zip(queries, ground_truth):
            query_embedding = model.encode(query)

            # Calculate similarities
            similarities = np.dot(chunk_embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            # Get unique documents from top chunks
            retrieved_docs = list({chunk_to_doc[i] for i in top_indices})

            # Calculate metrics
            relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))

            precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
            recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0

            precisions.append(precision)
            recalls.append(recall)
            relevance_scores.append(np.mean([similarities[i] for i in top_indices]))

        # Calculate aggregate metrics
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

        results.append(EvaluationResult(
            chunk_size=chunk_size,
            num_chunks=len(all_chunks),
            avg_chunk_length=np.mean([len(c) for c in all_chunks]),
            retrieval_precision=avg_precision,
            retrieval_recall=avg_recall,
            retrieval_f1=f1,
            avg_relevance_score=np.mean(relevance_scores)
        ))

    return results


def find_optimal_chunk_size(
    documents: List[str],
    sample_queries: List[str],
    ground_truth: List[List[int]],
    search_range: Tuple[int, int] = (64, 1024),
    num_iterations: int = 5
) -> int:
    """
    Use binary search to find optimal chunk size.

    Args:
        documents: Test documents
        sample_queries: Sample queries for evaluation
        ground_truth: Relevance labels
        search_range: Min and max chunk sizes to search
        num_iterations: Number of binary search iterations

    Returns:
        Optimal chunk size
    """
    low, high = search_range

    for _ in range(num_iterations):
        mid = (low + high) // 2

        # Evaluate three points
        sizes = [low, mid, high]
        results = evaluate_chunk_sizes(
            documents, sample_queries, ground_truth,
            chunk_sizes=sizes
        )

        # Find best F1 score
        best_idx = np.argmax([r.retrieval_f1 for r in results])

        # Narrow search range
        if best_idx == 0:
            high = mid
        elif best_idx == 2:
            low = mid
        else:
            # Best is in the middle, narrow both sides
            low = (low + mid) // 2
            high = (mid + high) // 2

    return (low + high) // 2


def analyze_chunk_statistics(chunks: List[str]) -> Dict:
    """Analyze statistics of a chunking result."""
    lengths = [len(c) for c in chunks]

    return {
        'num_chunks': len(chunks),
        'total_chars': sum(lengths),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'median_length': np.median(lengths),
        'length_distribution': {
            '0-100': sum(1 for length in lengths if length < 100),
            '100-250': sum(1 for length in lengths if 100 <= length < 250),
            '250-500': sum(1 for length in lengths if 250 <= length < 500),
            '500-1000': sum(1 for length in lengths if 500 <= length < 1000),
            '1000+': sum(1 for length in lengths if length >= 1000),
        }
    }


# Example usage
if __name__ == "__main__":
    # Sample data for demonstration
    sample_docs = [
        """Machine learning is a subset of artificial intelligence that focuses
        on building systems that learn from data. It enables computers to
        improve their performance on tasks through experience.""" * 5,

        """Neural networks are computing systems inspired by biological neural
        networks. They consist of interconnected nodes that process information
        using connectionist approaches to computation.""" * 5,

        """Deep learning is part of a broader family of machine learning methods
        based on artificial neural networks with representation learning.
        Learning can be supervised, semi-supervised or unsupervised.""" * 5,
    ]

    sample_queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What is deep learning?"
    ]

    # Ground truth: which documents are relevant for each query
    ground_truth = [
        [0],        # Query 1 relates to doc 0
        [1],        # Query 2 relates to doc 1
        [2],        # Query 3 relates to doc 2
    ]

    print("Evaluating chunk sizes...\n")
    results = evaluate_chunk_sizes(
        sample_docs,
        sample_queries,
        ground_truth,
        chunk_sizes=[128, 256, 512],
        top_k=3
    )

    print("Results:")
    print("-" * 70)
    print(f"{'Size':>6} | {'Chunks':>6} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
    print("-" * 70)

    for r in results:
        print(f"{r.chunk_size:>6} | {r.num_chunks:>6} | {r.retrieval_precision:>9.3f} | {r.retrieval_recall:>6.3f} | {r.retrieval_f1:>6.3f}")

    # Find best
    best = max(results, key=lambda x: x.retrieval_f1)
    print(f"\nBest chunk size: {best.chunk_size} (F1={best.retrieval_f1:.3f})")
