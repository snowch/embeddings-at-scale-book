# Code from Chapter 11
# Book: Embeddings at Scale

"""
Approximate Nearest Neighbor Implementations

Algorithms:
1. IVF (Inverted File Index): Clustering-based partitioning
2. HNSW (Hierarchical Navigable Small World): Graph-based navigation
3. Product Quantization: Vector compression for memory efficiency
"""

import heapq
import time
from collections import defaultdict

# Placeholder classes - see import.py for full implementations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SearchResult:
    """Placeholder for SearchResult."""
    indices: np.ndarray
    scores: np.ndarray
    latency_ms: float

class OptimizedExactSearch:
    """Placeholder for OptimizedExactSearch."""
    def __init__(self):
        pass

    def search(self, query_vector, k=10):
        return SearchResult(
            indices=np.array([0]),
            scores=np.array([1.0]),
            latency_ms=0.1
        )

class IVFIndex:
    """
    Inverted File Index (IVF) for approximate nearest neighbor search

    Algorithm:
    1. Training: Cluster corpus into C centroids using k-means
    2. Indexing: Assign each vector to nearest centroid
    3. Search:
       - Find nearest centroids to query
       - Search only vectors in those clusters
       - Merge and rank results

    Parameters:
    - num_clusters (C): More clusters = better recall, slower search
    - num_probes (P): Search P nearest clusters (P=1 fastest, P=C exact)

    Performance:
    - Build time: O(N × C × iterations)
    - Search time: O(C + (N/C) × P)  << O(N) for P << C
    - Memory: Same as corpus (no compression)

    Typical settings:
    - 1M vectors: C=1000, P=10 → 100× speedup, 95% recall
    - 100M vectors: C=10000, P=50 → 200× speedup, 98% recall
    - 1B vectors: C=100000, P=100 → 1000× speedup, 99% recall
    """

    def __init__(
        self,
        num_clusters: int = 1000,
        num_probes: int = 10
    ):
        """
        Args:
            num_clusters: Number of clusters (more = better recall, slower build)
            num_probes: Number of clusters to search (more = better recall, slower search)
        """
        self.num_clusters = num_clusters
        self.num_probes = num_probes

        # Index structures
        self.centroids = None  # Cluster centroids (C, d)
        self.inverted_lists = None  # cluster_id → [vector_indices]
        self.corpus = None

        print("Initialized IVF index")
        print(f"  Clusters: {num_clusters}")
        print(f"  Probes: {num_probes}")

    def train(self, corpus: np.ndarray, max_iterations: int = 50):
        """
        Train IVF index by clustering corpus

        Uses k-means clustering to find centroids

        Args:
            corpus: Training vectors (N, d)
            max_iterations: K-means iterations
        """
        print(f"Training IVF with {len(corpus):,} vectors...")
        start_time = time.time()

        self.corpus = corpus
        num_vectors, dim = corpus.shape

        # Initialize centroids randomly
        centroid_indices = np.random.choice(
            num_vectors,
            size=self.num_clusters,
            replace=False
        )
        self.centroids = corpus[centroid_indices].copy()

        # K-means clustering
        for iteration in range(max_iterations):
            # Assign vectors to nearest centroids
            assignments = self._assign_to_nearest_centroid(corpus)

            # Update centroids
            old_centroids = self.centroids.copy()
            for cluster_id in range(self.num_clusters):
                cluster_vectors = corpus[assignments == cluster_id]
                if len(cluster_vectors) > 0:
                    self.centroids[cluster_id] = cluster_vectors.mean(axis=0)

            # Check convergence
            centroid_shift = np.linalg.norm(self.centroids - old_centroids)
            if centroid_shift < 1e-4:
                print(f"  Converged at iteration {iteration + 1}")
                break

        # Build inverted lists
        self.inverted_lists = defaultdict(list)
        assignments = self._assign_to_nearest_centroid(corpus)

        for vector_id, cluster_id in enumerate(assignments):
            self.inverted_lists[cluster_id].append(vector_id)

        # Statistics
        avg_list_size = np.mean([len(lst) for lst in self.inverted_lists.values()])
        max_list_size = max([len(lst) for lst in self.inverted_lists.values()])

        elapsed = time.time() - start_time

        print(f"✓ Training complete in {elapsed:.1f}s")
        print(f"  Avg cluster size: {avg_list_size:.0f} vectors")
        print(f"  Max cluster size: {max_list_size} vectors")

    def _assign_to_nearest_centroid(self, vectors: np.ndarray) -> np.ndarray:
        """
        Assign vectors to nearest centroids

        Returns:
            assignments: (N,) array of cluster IDs
        """
        # Compute distances to all centroids
        # (N, d) @ (d, C) → (N, C)
        similarities = np.dot(vectors, self.centroids.T)

        # Find nearest centroid for each vector
        assignments = np.argmax(similarities, axis=1)

        return assignments

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        num_probes: Optional[int] = None
    ) -> SearchResult:
        """
        Search for k nearest neighbors using IVF

        Args:
            query: Query vector (d,)
            k: Number of nearest neighbors
            num_probes: Override default num_probes

        Returns:
            SearchResult
        """
        start_time = time.time()

        if num_probes is None:
            num_probes = self.num_probes

        # Normalize query
        query = query / np.linalg.norm(query)

        # Find nearest centroids to query
        centroid_scores = np.dot(self.centroids, query)
        nearest_centroids = np.argsort(-centroid_scores)[:num_probes]

        # Search vectors in selected clusters
        candidate_indices = []
        for cluster_id in nearest_centroids:
            candidate_indices.extend(self.inverted_lists[cluster_id])

        # Compute exact similarities for candidates
        candidates = self.corpus[candidate_indices]
        scores = np.dot(candidates, query)

        # Find top-k
        if len(scores) > k:
            top_k_positions = np.argpartition(-scores, k)[:k]
            top_k_positions = top_k_positions[np.argsort(-scores[top_k_positions])]
        else:
            top_k_positions = np.argsort(-scores)

        top_indices = np.array([candidate_indices[i] for i in top_k_positions])
        top_scores = scores[top_k_positions]

        latency_ms = (time.time() - start_time) * 1000

        return SearchResult(
            indices=top_indices,
            scores=top_scores,
            latency_ms=latency_ms
        )

class HNSWIndex:
    """
    Hierarchical Navigable Small World (HNSW) graph index

    Architecture:
    - Multi-layer proximity graph (skip list structure)
    - Layer 0: All vectors, dense connections
    - Layer 1+: Subset of vectors, sparser connections
    - Higher layers enable long-range navigation

    Algorithm:
    1. Insert: Add vector at random layer, connect to nearest neighbors
    2. Search:
       - Start at top layer
       - Greedy navigate to local minimum
       - Descend to next layer
       - Repeat until layer 0
       - Return neighbors at layer 0

    Parameters:
    - M: Max connections per layer (higher = better recall, more memory)
    - ef_construction: Candidate list size during build (higher = better index)
    - ef_search: Candidate list size during search (higher = better recall)

    Performance:
    - Build time: O(N × log N × M × ef_construction)
    - Search time: O(log N × M × ef_search)
    - Memory: ~(M × 4 bytes) × N = ~16MB per 1M vectors (M=4)

    State-of-the-art trade-off:
    - 99.5% recall @ 0.5ms latency (1B vectors, A100 GPU)
    - 95% recall @ 0.1ms latency

    Best for:
    - High-dimensional data (100-1000 dims)
    - Need excellent recall (>95%)
    - Have memory for graph structure
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        max_layer: int = 5
    ):
        """
        Args:
            dim: Vector dimension
            M: Max connections per node per layer
            ef_construction: Size of candidate list during construction
            ef_search: Size of candidate list during search
            max_layer: Maximum layer index
        """
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_layer = max_layer

        # Graph structure: layer → node_id → [neighbor_ids]
        self.graph = defaultdict(lambda: defaultdict(list))

        # Vectors
        self.vectors = []
        self.vector_layers = []  # Layer of each vector

        # Entry point (top layer, highest node)
        self.entry_point = None
        self.entry_layer = -1

        print("Initialized HNSW index")
        print(f"  M: {M}, ef_construction: {ef_construction}, ef_search: {ef_search}")

    def add(self, vector: np.ndarray):
        """
        Add vector to HNSW index

        Args:
            vector: Vector to add (d,)
        """
        # Normalize
        vector = vector / np.linalg.norm(vector)

        # Determine layer for this vector
        layer = self._random_layer()

        # Add to index
        node_id = len(self.vectors)
        self.vectors.append(vector)
        self.vector_layers.append(layer)

        # Update entry point if this is highest layer
        if layer > self.entry_layer:
            self.entry_layer = layer
            self.entry_point = node_id

        # If first vector, done
        if node_id == 0:
            return

        # Insert into graph layers
        # Start from top, navigate to nearest neighbors, insert connections
        current_nearest = [self.entry_point]

        for lc in range(self.entry_layer, -1, -1):
            # Navigate to nearest neighbor at this layer
            current_nearest = self._search_layer(
                vector,
                current_nearest,
                layer=lc,
                ef=self.ef_construction
            )

            # If this layer <= node's layer, create connections
            if lc <= layer:
                # Get M nearest neighbors
                neighbors = current_nearest[:self.M]

                # Add bidirectional connections
                for neighbor_id in neighbors:
                    self.graph[lc][node_id].append(neighbor_id)
                    self.graph[lc][neighbor_id].append(node_id)

                    # Prune neighbor's connections if exceeds M
                    if len(self.graph[lc][neighbor_id]) > self.M:
                        self._prune_connections(neighbor_id, lc)

    def _random_layer(self) -> int:
        """
        Sample layer for new node using exponential decay

        Probability of layer l: ~ (1/2)^l
        Creates skip list structure
        """
        layer = 0
        while layer < self.max_layer and np.random.random() < 0.5:
            layer += 1
        return layer

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        layer: int,
        ef: int
    ) -> List[int]:
        """
        Greedy search within single layer

        Args:
            query: Query vector
            entry_points: Starting nodes
            layer: Layer to search
            ef: Size of candidate list

        Returns:
            List of nearest node IDs (sorted by similarity)
        """
        visited = set(entry_points)
        candidates = []

        # Initialize with entry points
        for node_id in entry_points:
            similarity = np.dot(query, self.vectors[node_id])
            heapq.heappush(candidates, (-similarity, node_id))

        best_candidates = [(-s, i) for s, i in candidates]  # Max heap for best

        while candidates:
            # Get closest unvisited candidate
            current_sim, current_id = heapq.heappop(candidates)
            current_sim = -current_sim

            # If this is worse than ef-th best, stop
            if len(best_candidates) >= ef:
                ef_th_best_sim = -best_candidates[ef - 1][0]
                if current_sim < ef_th_best_sim:
                    break

            # Explore neighbors
            if layer in self.graph and current_id in self.graph[layer]:
                for neighbor_id in self.graph[layer][current_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        similarity = np.dot(query, self.vectors[neighbor_id])

                        # Add to candidates
                        heapq.heappush(candidates, (-similarity, neighbor_id))
                        heapq.heappush(best_candidates, (-similarity, neighbor_id))

                        # Keep only ef best
                        if len(best_candidates) > ef:
                            heapq.heappop(best_candidates)  # Remove worst

        # Return ef nearest
        result = sorted(best_candidates, reverse=True)  # Sort by similarity
        return [node_id for _, node_id in result]

    def _prune_connections(self, node_id: int, layer: int):
        """
        Prune connections to maintain max M neighbors

        Keep M most similar neighbors
        """
        neighbors = self.graph[layer][node_id]
        if len(neighbors) <= self.M:
            return

        # Compute similarities
        node_vector = self.vectors[node_id]
        similarities = [
            (np.dot(node_vector, self.vectors[n_id]), n_id)
            for n_id in neighbors
        ]

        # Keep M best
        similarities.sort(reverse=True)
        self.graph[layer][node_id] = [n_id for _, n_id in similarities[:self.M]]

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None
    ) -> SearchResult:
        """
        Search for k nearest neighbors

        Args:
            query: Query vector (d,)
            k: Number of nearest neighbors
            ef: Search candidate list size (default: self.ef_search)

        Returns:
            SearchResult
        """
        start_time = time.time()

        if ef is None:
            ef = self.ef_search

        if len(self.vectors) == 0:
            return SearchResult(
                indices=np.array([]),
                scores=np.array([]),
                latency_ms=0.0
            )

        # Normalize query
        query = query / np.linalg.norm(query)

        # Start from entry point at top layer
        current_nearest = [self.entry_point]

        # Navigate down through layers
        for layer in range(self.entry_layer, -1, -1):
            current_nearest = self._search_layer(
                query,
                current_nearest,
                layer=layer,
                ef=ef if layer == 0 else 1
            )

        # Extract top-k
        top_k = current_nearest[:k]
        top_scores = np.array([np.dot(query, self.vectors[i]) for i in top_k])
        top_indices = np.array(top_k)

        latency_ms = (time.time() - start_time) * 1000

        return SearchResult(
            indices=top_indices,
            scores=top_scores,
            latency_ms=latency_ms
        )

# Example: Compare IVF vs HNSW
def ann_comparison_example():
    """
    Compare IVF and HNSW on 100K vectors

    Metrics:
    - Build time
    - Search latency
    - Recall @ k=10
    - Memory usage
    """
    # Generate corpus
    num_vectors = 100_000
    dim = 128
    k = 10

    print(f"Generating {num_vectors:,} {dim}-d vectors...")
    corpus = np.random.randn(num_vectors, dim).astype(np.float32)
    corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)

    # Ground truth (exact search)
    print("\nComputing ground truth with exact search...")
    exact = OptimizedExactSearch(corpus, normalized=True, use_gpu=False)

    queries = np.random.randn(100, dim).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    ground_truth = []
    for query in queries:
        result = exact.search(query, k=k)
        ground_truth.append(set(result.indices))

    # IVF Index
    print("\n=== IVF Index ===")
    ivf_start = time.time()
    ivf = IVFIndex(num_clusters=1000, num_probes=10)
    ivf.train(corpus)
    ivf_build_time = time.time() - ivf_start

    print(f"Build time: {ivf_build_time:.1f}s")

    # IVF search
    ivf_latencies = []
    ivf_recalls = []

    for i, query in enumerate(queries):
        result = ivf.search(query, k=k)
        ivf_latencies.append(result.latency_ms)

        recall = len(set(result.indices) & ground_truth[i]) / k
        ivf_recalls.append(recall)

    print(f"Avg search latency: {np.mean(ivf_latencies):.2f} ms")
    print(f"Recall@{k}: {np.mean(ivf_recalls):.3f}")

    # HNSW Index
    print("\n=== HNSW Index ===")
    hnsw_start = time.time()
    hnsw = HNSWIndex(dim=dim, M=16, ef_construction=200, ef_search=50)

    for vector in corpus:
        hnsw.add(vector)

    hnsw_build_time = time.time() - hnsw_start
    print(f"Build time: {hnsw_build_time:.1f}s")

    # HNSW search
    hnsw_latencies = []
    hnsw_recalls = []

    for i, query in enumerate(queries):
        result = hnsw.search(query, k=k)
        hnsw_latencies.append(result.latency_ms)

        recall = len(set(result.indices) & ground_truth[i]) / k
        hnsw_recalls.append(recall)

    print(f"Avg search latency: {np.mean(hnsw_latencies):.2f} ms")
    print(f"Recall@{k}: {np.mean(hnsw_recalls):.3f}")

    # Summary
    print("\n=== Comparison ===")
    print(f"{'Method':<10} {'Build (s)':<12} {'Search (ms)':<14} {'Recall@10':<12}")
    print(f"{'IVF':<10} {ivf_build_time:<12.1f} {np.mean(ivf_latencies):<14.2f} {np.mean(ivf_recalls):<12.3f}")
    print(f"{'HNSW':<10} {hnsw_build_time:<12.1f} {np.mean(hnsw_latencies):<14.2f} {np.mean(hnsw_recalls):<12.3f}")

# Uncomment to run:
# ann_comparison_example()
