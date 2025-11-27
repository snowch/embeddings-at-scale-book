"""Evaluate chunk quality for RAG systems."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ChunkQualityMetrics:
    """Quality metrics for a set of chunks."""
    # Size metrics
    avg_chunk_size: float
    std_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int

    # Content metrics
    avg_word_count: float
    unique_terms_ratio: float

    # Retrieval metrics (if ground truth available)
    precision_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    mrr: Optional[float] = None  # Mean Reciprocal Rank


def evaluate_chunk_quality(
    chunks: List[str],
    queries: Optional[List[str]] = None,
    ground_truth: Optional[List[List[int]]] = None,
    k: int = 5
) -> ChunkQualityMetrics:
    """
    Evaluate the quality of a chunking strategy.

    Args:
        chunks: List of text chunks
        queries: Optional list of test queries
        ground_truth: For each query, indices of relevant chunks
        k: Number of results for precision/recall@k

    Returns:
        ChunkQualityMetrics object
    """
    # Size metrics
    sizes = [len(c) for c in chunks]
    word_counts = [len(c.split()) for c in chunks]

    # Unique terms ratio
    all_terms = []
    for chunk in chunks:
        all_terms.extend(chunk.lower().split())
    unique_ratio = len(set(all_terms)) / len(all_terms) if all_terms else 0

    metrics = ChunkQualityMetrics(
        avg_chunk_size=np.mean(sizes),
        std_chunk_size=np.std(sizes),
        min_chunk_size=min(sizes),
        max_chunk_size=max(sizes),
        avg_word_count=np.mean(word_counts),
        unique_terms_ratio=unique_ratio
    )

    # Retrieval metrics if ground truth provided
    if queries and ground_truth:
        p_at_k, r_at_k, mrr = _evaluate_retrieval(chunks, queries, ground_truth, k)
        metrics.precision_at_k = p_at_k
        metrics.recall_at_k = r_at_k
        metrics.mrr = mrr

    return metrics


def _evaluate_retrieval(
    chunks: List[str],
    queries: List[str],
    ground_truth: List[List[int]],
    k: int
) -> Tuple[float, float, float]:
    """Evaluate retrieval performance."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed chunks
    chunk_embeddings = model.encode(chunks)

    precisions = []
    recalls = []
    reciprocal_ranks = []

    for query, relevant_indices in zip(queries, ground_truth):
        # Embed query
        query_embedding = model.encode(query)

        # Calculate similarities
        similarities = np.dot(chunk_embeddings, query_embedding)
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        # Precision@k
        relevant_retrieved = len(set(top_k_indices) & set(relevant_indices))
        precision = relevant_retrieved / k
        precisions.append(precision)

        # Recall@k
        recall = relevant_retrieved / len(relevant_indices) if relevant_indices else 0
        recalls.append(recall)

        # Reciprocal Rank
        for rank, idx in enumerate(top_k_indices, 1):
            if idx in relevant_indices:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(precisions), np.mean(recalls), np.mean(reciprocal_ranks)


def compare_chunking_strategies(
    text: str,
    strategies: Dict[str, callable],
    queries: List[str],
    ground_truth: List[List[int]]
) -> Dict[str, ChunkQualityMetrics]:
    """
    Compare multiple chunking strategies on the same document.

    Args:
        text: Document to chunk
        strategies: Dict of strategy_name -> chunking_function
        queries: Test queries
        ground_truth: Relevant chunk indices for each query

    Returns:
        Dict of strategy_name -> metrics
    """
    results = {}

    for name, chunker_fn in strategies.items():
        chunks = chunker_fn(text)
        metrics = evaluate_chunk_quality(chunks, queries, ground_truth)
        results[name] = metrics

    return results


def analyze_failure_cases(
    chunks: List[str],
    queries: List[str],
    ground_truth: List[List[int]],
    k: int = 5
) -> List[Dict]:
    """
    Identify queries where retrieval failed.

    Returns details about each failure for debugging.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_embeddings = model.encode(chunks)

    failures = []

    for query, relevant_indices in zip(queries, ground_truth):
        query_embedding = model.encode(query)
        similarities = np.dot(chunk_embeddings, query_embedding)
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        relevant_retrieved = set(top_k_indices) & set(relevant_indices)

        if len(relevant_retrieved) == 0:
            # Complete failure - no relevant chunks retrieved
            failures.append({
                'query': query,
                'expected_chunks': [chunks[i][:100] for i in relevant_indices],
                'retrieved_chunks': [chunks[i][:100] for i in top_k_indices],
                'expected_similarities': [similarities[i] for i in relevant_indices],
                'retrieved_similarities': [similarities[i] for i in top_k_indices],
                'failure_type': 'complete'
            })
        elif len(relevant_retrieved) < len(relevant_indices):
            # Partial failure - some relevant chunks missed
            missed = set(relevant_indices) - relevant_retrieved
            failures.append({
                'query': query,
                'missed_chunks': [chunks[i][:100] for i in missed],
                'missed_similarities': [similarities[i] for i in missed],
                'failure_type': 'partial'
            })

    return failures


def suggest_improvements(metrics: ChunkQualityMetrics) -> List[str]:
    """
    Suggest improvements based on quality metrics.
    """
    suggestions = []

    # Chunk size analysis
    if metrics.std_chunk_size > metrics.avg_chunk_size * 0.5:
        suggestions.append(
            "High chunk size variance detected. Consider using fixed-size "
            "chunking or adjusting separators for more consistent sizes."
        )

    if metrics.avg_chunk_size < 100:
        suggestions.append(
            "Chunks are very small. Consider increasing chunk size to "
            "preserve more context."
        )

    if metrics.avg_chunk_size > 1000:
        suggestions.append(
            "Chunks are quite large. Consider reducing chunk size for "
            "better retrieval precision."
        )

    # Retrieval analysis
    if metrics.precision_at_k and metrics.precision_at_k < 0.3:
        suggestions.append(
            "Low precision detected. Try:\n"
            "- Smaller chunk sizes for finer granularity\n"
            "- Better query preprocessing\n"
            "- Hybrid search (keyword + semantic)"
        )

    if metrics.recall_at_k and metrics.recall_at_k < 0.5:
        suggestions.append(
            "Low recall detected. Try:\n"
            "- Larger overlap between chunks\n"
            "- Retrieving more chunks (higher k)\n"
            "- Document expansion or query expansion"
        )

    if metrics.unique_terms_ratio < 0.3:
        suggestions.append(
            "Low unique terms ratio indicates repetitive content. "
            "Consider deduplication or content filtering."
        )

    return suggestions


# Example usage
if __name__ == "__main__":
    # Sample evaluation
    sample_chunks = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning uses multiple layers for feature extraction.",
        "Supervised learning requires labeled training data.",
        "Clustering groups similar data points together.",
    ]

    sample_queries = [
        "What is machine learning?",
        "How do neural networks work?",
    ]

    sample_ground_truth = [
        [0],  # First query relates to chunk 0
        [1],  # Second query relates to chunk 1
    ]

    metrics = evaluate_chunk_quality(
        sample_chunks,
        sample_queries,
        sample_ground_truth,
        k=3
    )

    print("Chunk Quality Metrics:")
    print(f"  Avg chunk size: {metrics.avg_chunk_size:.0f} chars")
    print(f"  Size std dev: {metrics.std_chunk_size:.0f}")
    print(f"  Avg word count: {metrics.avg_word_count:.1f}")
    print(f"  Unique terms ratio: {metrics.unique_terms_ratio:.2%}")

    if metrics.precision_at_k:
        print(f"  Precision@3: {metrics.precision_at_k:.2%}")
        print(f"  Recall@3: {metrics.recall_at_k:.2%}")
        print(f"  MRR: {metrics.mrr:.3f}")

    print("\nSuggested Improvements:")
    for suggestion in suggest_improvements(metrics):
        print(f"  - {suggestion}")
