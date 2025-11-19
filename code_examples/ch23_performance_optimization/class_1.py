# Code from Chapter 23
# Book: Embeddings at Scale

"""
Index Tuning for Specific Workloads

Architecture:
1. Workload profiler: Analyze query patterns, data characteristics
2. Benchmark suite: Test index configurations on representative data
3. Parameter optimizer: Search parameter space for optimal config
4. Validation framework: Test on production-like traffic
5. Adaptive tuning: Continuously monitor and adjust

Techniques:
- HNSW tuning: ef_construction, M, ef_search, layer scaling
- IVF tuning: n_clusters, n_probe, training set size
- PQ tuning: n_subvectors, n_bits, training method
- LSH tuning: n_tables, n_bits, hash function selection
- Hybrid indexes: Combine multiple strategies

Index selection criteria:
- Query latency requirements
- Recall requirements
- Update frequency
- Dataset size and growth
- Memory/disk constraints
- Hardware availability
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class WorkloadProfile:
    """
    Characterization of vector workload for index tuning
    
    Attributes:
        total_vectors: Total number of vectors in dataset
        vector_dim: Dimensionality of vectors
        queries_per_second: Average query rate
        k_distribution: Distribution of k values (histogram)
        filter_rate: Fraction of queries with filters
        update_rate: Vectors added/updated per second
        batch_size: Typical batch size (1 for single queries)
        latency_p50_requirement: 50th percentile latency requirement (ms)
        latency_p99_requirement: 99th percentile latency requirement (ms)
        recall_requirement: Minimum acceptable recall
        memory_budget_gb: Available RAM for indexes
        storage_budget_gb: Available disk for vectors
        has_gpu: Whether GPU acceleration available
    """
    total_vectors: int
    vector_dim: int
    queries_per_second: float
    k_distribution: Dict[int, float] = field(default_factory=dict)  # k → probability
    filter_rate: float = 0.0
    update_rate: float = 0.0
    batch_size: int = 1
    latency_p50_requirement: float = 50.0  # ms
    latency_p99_requirement: float = 100.0  # ms
    recall_requirement: float = 0.95
    memory_budget_gb: float = 32.0
    storage_budget_gb: float = 1000.0
    has_gpu: bool = False

@dataclass
class IndexConfig:
    """
    Configuration for vector index
    
    Attributes:
        index_type: Type of index (hnsw, ivf, pq, lsh, hybrid)
        parameters: Index-specific parameters
        hardware: Hardware configuration (cpu, gpu, mixed)
        build_time_hours: Time to build index (estimated)
        memory_gb: RAM required for index
        storage_gb: Disk space required
        expected_latency_p50: Expected 50th percentile latency
        expected_latency_p99: Expected 99th percentile latency
        expected_recall: Expected recall at these latencies
        expected_throughput: Expected queries per second
    """
    index_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    hardware: str = "cpu"
    build_time_hours: float = 1.0
    memory_gb: float = 10.0
    storage_gb: float = 100.0
    expected_latency_p50: float = 20.0
    expected_latency_p99: float = 50.0
    expected_recall: float = 0.95
    expected_throughput: float = 1000.0

class HNSWTuner:
    """
    HNSW index parameter tuning
    
    HNSW parameters:
    - M: Number of connections per layer (16-64 typical)
    - ef_construction: Size of dynamic candidate list during construction (100-500)
    - ef_search: Size of dynamic candidate list during search (k to 1000+)
    - max_layers: Maximum number of layers (automatically computed)
    """

    def __init__(self, profile: WorkloadProfile):
        self.profile = profile

    def tune(self) -> IndexConfig:
        """
        Tune HNSW parameters for workload
        
        Returns:
            config: Optimized HNSW configuration
        """
        # Estimate optimal M based on dimensionality and recall requirement
        if self.profile.recall_requirement >= 0.98:
            M = 48  # Higher M for high recall
        elif self.profile.recall_requirement >= 0.95:
            M = 32  # Moderate M for balanced performance
        else:
            M = 16  # Lower M for speed

        # Adjust M for high dimensionality
        if self.profile.vector_dim >= 768:
            M = min(M * 1.5, 64)

        # Estimate ef_construction based on M and recall
        ef_construction = max(M * 4, 200)
        if self.profile.recall_requirement >= 0.98:
            ef_construction = max(ef_construction * 2, 400)

        # Estimate ef_search based on k distribution and latency requirements
        typical_k = self._get_typical_k()
        ef_search = typical_k * 2

        # Adjust for recall requirements
        if self.profile.recall_requirement >= 0.98:
            ef_search = typical_k * 4
        elif self.profile.recall_requirement >= 0.95:
            ef_search = typical_k * 2
        else:
            ef_search = max(typical_k * 1.5, 50)

        # Adjust for latency requirements
        if self.profile.latency_p50_requirement < 10:
            # Aggressive latency requirement - reduce ef_search
            ef_search = max(ef_search * 0.7, typical_k)

        # Estimate resource requirements
        memory_gb = self._estimate_memory(M, ef_construction)
        storage_gb = self._estimate_storage(M)
        build_time_hours = self._estimate_build_time(M, ef_construction)

        # Estimate performance
        expected_latency_p50 = self._estimate_latency(M, ef_search, 50)
        expected_latency_p99 = self._estimate_latency(M, ef_search, 99)
        expected_recall = self._estimate_recall(M, ef_search)
        expected_throughput = 1000 / expected_latency_p50  # Per core

        return IndexConfig(
            index_type="hnsw",
            parameters={
                "M": int(M),
                "ef_construction": int(ef_construction),
                "ef_search": int(ef_search),
            },
            hardware="cpu",
            build_time_hours=build_time_hours,
            memory_gb=memory_gb,
            storage_gb=storage_gb,
            expected_latency_p50=expected_latency_p50,
            expected_latency_p99=expected_latency_p99,
            expected_recall=expected_recall,
            expected_throughput=expected_throughput,
        )

    def _get_typical_k(self) -> int:
        """Get typical k value from distribution"""
        if not self.profile.k_distribution:
            return 10  # Default

        # Weighted average
        total = sum(self.profile.k_distribution.values())
        avg_k = sum(
            k * prob for k, prob in self.profile.k_distribution.items()
        ) / total

        return int(avg_k)

    def _estimate_memory(self, M: float, ef_construction: float) -> float:
        """
        Estimate memory requirements for HNSW index
        
        Memory = vectors + graph structure + construction buffers
        """
        # Vector storage
        vector_memory = (
            self.profile.total_vectors *
            self.profile.vector_dim *
            4 / 1024**3  # float32 in GB
        )

        # Graph structure (adjacency lists)
        avg_layers = np.log2(self.profile.total_vectors) / np.log2(1.0 / np.log(2))
        graph_memory = (
            self.profile.total_vectors *
            M * avg_layers *
            8 / 1024**3  # Pointers in GB
        )

        # Construction buffers
        construction_memory = (
            ef_construction *
            self.profile.vector_dim *
            4 / 1024**3
        )

        return vector_memory + graph_memory + construction_memory

    def _estimate_storage(self, M: float) -> float:
        """Estimate disk storage requirements"""
        # Similar to memory but can use compression
        return self._estimate_memory(M, 0) * 0.8

    def _estimate_build_time(self, M: float, ef_construction: float) -> float:
        """
        Estimate index construction time
        
        Roughly linear in dataset size, quadratic in ef_construction
        """
        base_time_per_million = 0.1  # hours per million vectors
        complexity_factor = (M / 32) * (ef_construction / 200)

        return (
            self.profile.total_vectors / 1e6 *
            base_time_per_million *
            complexity_factor
        )

    def _estimate_latency(
        self,
        M: float,
        ef_search: float,
        percentile: int
    ) -> float:
        """
        Estimate query latency at given percentile
        
        Latency increases with ef_search and dimensionality
        """
        # Base latency for distance computation
        base_latency = self.profile.vector_dim * 1e-6  # 1μs per dimension

        # Graph traversal cost
        traversal_latency = ef_search * base_latency * np.log(M)

        # Add overhead for sorting, result assembly
        overhead = 0.5  # ms

        latency_p50 = traversal_latency * 1000 + overhead

        # p99 is typically 2-3× p50 for HNSW
        if percentile == 99:
            return latency_p50 * 2.5
        else:
            return latency_p50

    def _estimate_recall(self, M: float, ef_search: float) -> float:
        """
        Estimate recall for given parameters
        
        Higher M and ef_search → higher recall
        """
        # Empirical formula (approximation)
        typical_k = self._get_typical_k()

        if ef_search >= typical_k * 4 and M >= 32:
            return 0.98
        elif ef_search >= typical_k * 2 and M >= 24:
            return 0.95
        elif ef_search >= typical_k * 1.5 and M >= 16:
            return 0.90
        else:
            return 0.85

class IVFTuner:
    """
    IVF (Inverted File Index) parameter tuning
    
    IVF parameters:
    - n_clusters: Number of Voronoi cells (sqrt(N) to N/100 typical)
    - n_probe: Number of clusters to search (1 to n_clusters)
    - training_size: Number of vectors for k-means training
    """

    def __init__(self, profile: WorkloadProfile):
        self.profile = profile

    def tune(self) -> IndexConfig:
        """
        Tune IVF parameters for workload
        
        Returns:
            config: Optimized IVF configuration
        """
        # Estimate optimal n_clusters
        # Rule of thumb: sqrt(N) to N/100 depending on requirements
        if self.profile.recall_requirement >= 0.98:
            # Fewer clusters for high recall
            n_clusters = max(int(np.sqrt(self.profile.total_vectors)), 256)
        elif self.profile.recall_requirement >= 0.95:
            # Moderate clusters
            n_clusters = max(
                int(np.sqrt(self.profile.total_vectors) * 2),
                512
            )
        else:
            # Many clusters for speed
            n_clusters = max(
                int(np.sqrt(self.profile.total_vectors) * 4),
                1024
            )

        # Cap at reasonable maximum
        n_clusters = min(n_clusters, 65536)

        # Estimate optimal n_probe
        typical_k = self._get_typical_k()

        if self.profile.recall_requirement >= 0.98:
            n_probe = max(int(n_clusters * 0.05), 50)  # Search 5% of clusters
        elif self.profile.recall_requirement >= 0.95:
            n_probe = max(int(n_clusters * 0.02), 20)  # Search 2% of clusters
        else:
            n_probe = max(int(n_clusters * 0.01), 10)  # Search 1% of clusters

        # Adjust for latency requirements
        if self.profile.latency_p50_requirement < 10:
            n_probe = max(n_probe // 2, 5)

        # Training size (typically 10-100× n_clusters)
        training_size = min(
            n_clusters * 50,
            int(self.profile.total_vectors * 0.1)
        )

        # Estimate resources
        memory_gb = self._estimate_memory(n_clusters)
        storage_gb = self._estimate_storage(n_clusters)
        build_time_hours = self._estimate_build_time(n_clusters, training_size)

        # Estimate performance
        expected_latency_p50 = self._estimate_latency(n_clusters, n_probe, 50)
        expected_latency_p99 = self._estimate_latency(n_clusters, n_probe, 99)
        expected_recall = self._estimate_recall(n_clusters, n_probe)
        expected_throughput = 1000 / expected_latency_p50

        return IndexConfig(
            index_type="ivf",
            parameters={
                "n_clusters": n_clusters,
                "n_probe": n_probe,
                "training_size": training_size,
            },
            hardware="cpu",
            build_time_hours=build_time_hours,
            memory_gb=memory_gb,
            storage_gb=storage_gb,
            expected_latency_p50=expected_latency_p50,
            expected_latency_p99=expected_latency_p99,
            expected_recall=expected_recall,
            expected_throughput=expected_throughput,
        )

    def _get_typical_k(self) -> int:
        """Get typical k value from distribution"""
        if not self.profile.k_distribution:
            return 10
        total = sum(self.profile.k_distribution.values())
        avg_k = sum(
            k * prob for k, prob in self.profile.k_distribution.items()
        ) / total
        return int(avg_k)

    def _estimate_memory(self, n_clusters: int) -> float:
        """
        Estimate memory for IVF index
        
        Memory = vectors + centroids + inverted lists
        """
        # Vector storage
        vector_memory = (
            self.profile.total_vectors *
            self.profile.vector_dim *
            4 / 1024**3
        )

        # Centroid storage
        centroid_memory = (
            n_clusters *
            self.profile.vector_dim *
            4 / 1024**3
        )

        # Inverted list pointers
        list_memory = (
            self.profile.total_vectors *
            8 / 1024**3  # Pointer per vector
        )

        return vector_memory + centroid_memory + list_memory

    def _estimate_storage(self, n_clusters: int) -> float:
        """Estimate disk storage"""
        return self._estimate_memory(n_clusters) * 0.9

    def _estimate_build_time(
        self,
        n_clusters: int,
        training_size: int
    ) -> float:
        """
        Estimate build time
        
        Dominated by k-means clustering time
        """
        # k-means iterations (typically 10-50)
        n_iterations = 20

        # Time per iteration (empirical)
        time_per_iteration = (
            training_size * n_clusters * self.profile.vector_dim * 1e-9
        )  # hours

        kmeans_time = n_iterations * time_per_iteration

        # Assignment time (assign all vectors to clusters)
        assignment_time = (
            self.profile.total_vectors * n_clusters *
            self.profile.vector_dim * 1e-10
        )

        return kmeans_time + assignment_time

    def _estimate_latency(
        self,
        n_clusters: int,
        n_probe: int,
        percentile: int
    ) -> float:
        """Estimate query latency"""
        # Coarse quantization cost (find nearest centroids)
        coarse_cost = n_clusters * self.profile.vector_dim * 1e-6

        # Fine search cost (search n_probe clusters)
        vectors_per_cluster = self.profile.total_vectors / n_clusters
        candidates = n_probe * vectors_per_cluster
        fine_cost = candidates * self.profile.vector_dim * 1e-6

        latency_p50 = (coarse_cost + fine_cost) * 1000  # to ms

        if percentile == 99:
            return latency_p50 * 2.0
        else:
            return latency_p50

    def _estimate_recall(self, n_clusters: int, n_probe: int) -> float:
        """Estimate recall"""
        # Probability of finding nearest neighbor
        # Depends on cluster quality and n_probe
        probe_ratio = n_probe / n_clusters

        if probe_ratio >= 0.05:
            return 0.98
        elif probe_ratio >= 0.02:
            return 0.95
        elif probe_ratio >= 0.01:
            return 0.90
        else:
            return 0.85

class IndexSelector:
    """
    Select optimal index type and configuration for workload
    
    Compares HNSW, IVF, PQ, and hybrid approaches
    """

    def __init__(self, profile: WorkloadProfile):
        self.profile = profile
        self.hnsw_tuner = HNSWTuner(profile)
        self.ivf_tuner = IVFTuner(profile)

    def select(self) -> IndexConfig:
        """
        Select best index configuration for workload
        
        Returns:
            config: Recommended index configuration
        """
        # Generate candidate configurations
        candidates = []

        # HNSW candidate
        hnsw_config = self.hnsw_tuner.tune()
        candidates.append(hnsw_config)

        # IVF candidate
        ivf_config = self.ivf_tuner.tune()
        candidates.append(ivf_config)

        # Score each candidate
        scores = []
        for config in candidates:
            score = self._score_config(config)
            scores.append((score, config))

        # Return best scoring configuration
        scores.sort(reverse=True, key=lambda x: x[0])
        best_config = scores[0][1]

        return best_config

    def _score_config(self, config: IndexConfig) -> float:
        """
        Score configuration based on how well it meets requirements
        
        Returns:
            score: Higher is better (0-100)
        """
        score = 100.0

        # Latency penalty
        if config.expected_latency_p50 > self.profile.latency_p50_requirement:
            score -= 30 * (
                config.expected_latency_p50 /
                self.profile.latency_p50_requirement - 1
            )

        if config.expected_latency_p99 > self.profile.latency_p99_requirement:
            score -= 20 * (
                config.expected_latency_p99 /
                self.profile.latency_p99_requirement - 1
            )

        # Recall penalty
        if config.expected_recall < self.profile.recall_requirement:
            score -= 50 * (
                self.profile.recall_requirement - config.expected_recall
            )

        # Memory penalty
        if config.memory_gb > self.profile.memory_budget_gb:
            score -= 25 * (
                config.memory_gb / self.profile.memory_budget_gb - 1
            )

        # Build time penalty (minor)
        if config.build_time_hours > 24:
            score -= 5 * (config.build_time_hours / 24 - 1)

        return max(score, 0)
