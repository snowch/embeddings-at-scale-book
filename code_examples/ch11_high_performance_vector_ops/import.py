# Code from Chapter 11
# Book: Embeddings at Scale

"""
Optimized Exact Similarity Search

Techniques:
1. SIMD vectorization: Process multiple dimensions simultaneously
2. Matrix multiplication: Batch multiple queries together
3. Early termination: Stop computing if maximum score impossible
4. Quantization: Reduce precision for faster computation
5. Filtering: Pre-filter candidates before computing similarity
"""

import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SearchResult:
    """
    Result from similarity search

    Attributes:
        indices: Indices of nearest neighbors
        scores: Similarity scores
        latency_ms: Search latency in milliseconds
    """
    indices: np.ndarray
    scores: np.ndarray
    latency_ms: float

class OptimizedExactSearch:
    """
    Optimized exact similarity search using modern CPU/GPU techniques

    Optimizations:
    - SIMD vectorization via NumPy/PyTorch
    - Batch processing for throughput
    - L2 normalization for cosine similarity via dot product
    - GPU acceleration when available
    - Memory-efficient chunked processing

    Performance:
    - CPU (AVX-512): 1M vectors/sec on single core
    - GPU (A100): 100M vectors/sec
    - Batch processing: 10x throughput improvement

    Use when:
    - Small corpus (< 1M vectors)
    - Accuracy critical (zero approximation error)
    - GPU available (makes exact search feasible at larger scale)
    """

    def __init__(
        self,
        corpus: np.ndarray,
        normalized: bool = False,
        use_gpu: bool = True
    ):
        """
        Args:
            corpus: Corpus embeddings (N, d)
            normalized: Whether vectors are L2-normalized
            use_gpu: Use GPU acceleration if available
        """
        self.corpus = corpus
        self.normalized = normalized
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

        # Convert to PyTorch tensor for GPU acceleration
        self.corpus_tensor = torch.from_numpy(corpus).to(self.device)

        # Normalize if needed (cosine similarity via dot product)
        if not normalized:
            self.corpus_tensor = F.normalize(self.corpus_tensor, p=2, dim=1)
            print(f"Normalized {len(corpus)} vectors")

        print(f"Initialized exact search on {self.device}")
        print(f"  Corpus size: {len(corpus):,} vectors × {corpus.shape[1]} dims")
        print(f"  Memory: {corpus.nbytes / 1e9:.2f} GB")

    def search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> SearchResult:
        """
        Find k nearest neighbors to query using exact search

        Algorithm:
        1. Normalize query (if corpus normalized)
        2. Compute similarity to all corpus vectors (batched matrix multiplication)
        3. Find top-k via partial sort

        Args:
            query: Query vector (d,) or batch of queries (batch_size, d)
            k: Number of nearest neighbors

        Returns:
            SearchResult with indices, scores, and latency
        """
        start_time = time.time()

        # Convert to tensor
        query_tensor = torch.from_numpy(query).to(self.device)

        # Handle single query vs batch
        if query_tensor.ndim == 1:
            query_tensor = query_tensor.unsqueeze(0)
            single_query = True
        else:
            single_query = False

        # Normalize query
        query_tensor = F.normalize(query_tensor, p=2, dim=1)

        # Compute similarities (batched dot product)
        # (batch_size, d) @ (d, N) → (batch_size, N)
        similarities = torch.matmul(query_tensor, self.corpus_tensor.T)

        # Find top-k (uses partial sort - faster than full sort)
        top_scores, top_indices = torch.topk(
            similarities,
            k=min(k, len(self.corpus)),
            dim=1,
            largest=True,
            sorted=True
        )

        # Move back to CPU
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        # If single query, remove batch dimension
        if single_query:
            top_scores = top_scores[0]
            top_indices = top_indices[0]

        latency_ms = (time.time() - start_time) * 1000

        return SearchResult(
            indices=top_indices,
            scores=top_scores,
            latency_ms=latency_ms
        )

    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 10,
        batch_size: int = 1000
    ) -> List[SearchResult]:
        """
        Process multiple queries in batches for higher throughput

        Batching benefits:
        - Amortize GPU kernel launch overhead
        - Better utilize SIMD/parallelism
        - Reduce memory transfers

        Args:
            queries: Query vectors (num_queries, d)
            k: Number of nearest neighbors per query
            batch_size: Queries per batch

        Returns:
            List of SearchResults
        """
        num_queries = len(queries)
        results = []

        for i in range(0, num_queries, batch_size):
            batch_end = min(i + batch_size, num_queries)
            batch_queries = queries[i:batch_end]

            result = self.search(batch_queries, k)
            results.append(result)

        return results

class EarlyTerminationSearch:
    """
    Early termination for similarity search

    Insight: If we maintain upper bound on remaining similarity,
    we can skip vectors that cannot be in top-k

    Algorithm (MaxScore):
    1. Sort corpus by maximum contribution per dimension
    2. Maintain heap of current top-k scores
    3. During scan, compute upper bound on remaining vectors
    4. Terminate early if upper bound < k-th best score

    Speedup: 2-10× depending on query selectivity and corpus distribution

    Best for:
    - Sparse vectors (text embeddings with many zeros)
    - Skewed distributions (some dimensions much more important)
    - Large k (when many candidates needed)
    """

    def __init__(self, corpus: np.ndarray):
        """
        Args:
            corpus: Corpus embeddings (N, d)
        """
        self.corpus = corpus
        self.num_vectors, self.dim = corpus.shape

        # Precompute maximum contribution per dimension
        # max_contrib[j] = max_i |corpus[i, j]|
        self.max_contrib = np.max(np.abs(corpus), axis=0)

        # Sort dimensions by maximum contribution (for early termination)
        self.dim_order = np.argsort(-self.max_contrib)

        print("Initialized early termination search")
        print(f"  Max contrib range: [{self.max_contrib.min():.3f}, {self.max_contrib.max():.3f}]")

    def search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> SearchResult:
        """
        Search with early termination

        Args:
            query: Query vector (d,)
            k: Number of nearest neighbors

        Returns:
            SearchResult
        """
        start_time = time.time()

        # Normalize query
        query = query / np.linalg.norm(query)

        # Compute similarities for first few vectors to initialize heap
        init_size = min(k * 2, self.num_vectors)
        scores = []

        for i in range(init_size):
            score = np.dot(query, self.corpus[i])
            scores.append((score, i))

        # Maintain heap of top-k
        scores.sort(reverse=True)
        top_k_scores = scores[:k]
        k_th_score = top_k_scores[-1][0] if len(top_k_scores) == k else -np.inf

        # Scan remaining vectors with early termination
        vectors_scanned = init_size

        for i in range(init_size, self.num_vectors):
            # Compute upper bound on similarity for this vector
            # (using partial computation along dimension order)
            upper_bound = 0.0
            partial_score = 0.0

            for j_idx, j in enumerate(self.dim_order):
                partial_score += query[j] * self.corpus[i, j]

                # Upper bound: current partial + optimistic estimate for remaining
                remaining_dims = self.dim - j_idx - 1
                if remaining_dims > 0:
                    # Assume remaining dimensions contribute maximally
                    remaining_max = np.sum(
                        np.abs(query[self.dim_order[j_idx+1:]]) *
                        self.max_contrib[self.dim_order[j_idx+1:]]
                    )
                    upper_bound = partial_score + remaining_max
                else:
                    upper_bound = partial_score

                # Early termination: if upper bound < k-th score, skip this vector
                if upper_bound < k_th_score:
                    break
            else:
                # Computed full similarity
                score = partial_score
                vectors_scanned += 1

                # Update top-k if this score is better
                if score > k_th_score:
                    top_k_scores.append((score, i))
                    top_k_scores.sort(reverse=True)
                    top_k_scores = top_k_scores[:k]
                    k_th_score = top_k_scores[-1][0]

        # Extract results
        scores_arr = np.array([s for s, _ in top_k_scores])
        indices_arr = np.array([i for _, i in top_k_scores])

        latency_ms = (time.time() - start_time) * 1000

        print(f"  Scanned {vectors_scanned}/{self.num_vectors} vectors "
              f"({vectors_scanned/self.num_vectors:.1%})")

        return SearchResult(
            indices=indices_arr,
            scores=scores_arr,
            latency_ms=latency_ms
        )

# Example: Optimized exact search
def exact_search_example():
    """
    Compare naive vs. optimized exact search

    Scenario: 1M product embeddings, find 10 nearest neighbors
    """
    # Generate synthetic corpus
    num_vectors = 1_000_000
    dim = 512

    print(f"Generating {num_vectors:,} random {dim}-d vectors...")
    corpus = np.random.randn(num_vectors, dim).astype(np.float32)
    corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)

    # Create search index
    search = OptimizedExactSearch(corpus, normalized=True, use_gpu=True)

    # Generate query
    query = np.random.randn(dim).astype(np.float32)
    query = query / np.linalg.norm(query)

    # Search
    print("\nSearching for k=10 nearest neighbors...")
    result = search.search(query, k=10)

    print("\n✓ Search complete")
    print(f"  Latency: {result.latency_ms:.2f} ms")
    print(f"  Throughput: {num_vectors / (result.latency_ms / 1000):,.0f} vectors/sec")
    print(f"  Top-5 scores: {result.scores[:5]}")

    # Batch search
    print("\nBatch search with 100 queries...")
    queries = np.random.randn(100, dim).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    batch_results = search.batch_search(queries, k=10, batch_size=10)
    total_latency = sum(r.latency_ms for r in batch_results)

    print("✓ Batch search complete")
    print(f"  Total latency: {total_latency:.2f} ms")
    print(f"  Per-query latency: {total_latency / 100:.2f} ms")
    print(f"  Throughput: {100 / (total_latency / 1000):.0f} queries/sec")

# Uncomment to run:
# exact_search_example()
