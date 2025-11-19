# Code from Chapter 03
# Book: Embeddings at Scale

class HNSWDeepDive:
    """Understanding HNSW internals for scale"""

    def __init__(self, M=16, ef_construction=200, max_level=5):
        """
        M: Max connections per node (typical: 16-64)
        ef_construction: Candidates considered during build (typical: 100-500)
        max_level: Number of layers (auto-computed from dataset size)
        """
        self.M = M
        self.ef_construction = ef_construction
        self.max_level = max_level

        # Graph structure: list of layers
        # Each layer: dict of {node_id: [neighbor_ids]}
        self.layers = [dict() for _ in range(max_level + 1)]

        # Entry point for search
        self.entry_point = None

    def insert(self, vector_id, embedding, level):
        """
        Insert vector into HNSW graph

        Algorithm:
        1. Start at entry point in top layer
        2. Greedily navigate to nearest neighbor in each layer
        3. At target layer, find ef_construction nearest neighbors
        4. Connect to M best neighbors (by proximity)
        5. Prune connections to maintain M limit
        """

        # Find nearest neighbors at insertion layer
        neighbors = self.find_neighbors_at_layer(
            embedding,
            layer=level,
            ef=self.ef_construction
        )

        # Select M best connections
        connections = self.select_neighbors_heuristic(
            embedding,
            neighbors,
            M=self.M
        )

        # Add bidirectional edges
        for neighbor_id in connections:
            self.add_edge(vector_id, neighbor_id, level)
            self.add_edge(neighbor_id, vector_id, level)

            # Prune neighbor's connections if exceeds M
            self.prune_connections(neighbor_id, level)

    def search(self, query_embedding, ef_search=100, k=10):
        """
        Search for k nearest neighbors

        Algorithm:
        1. Start at entry point in top layer
        2. Greedy descent to layer 0, maintaining ef_search candidates
        3. At layer 0, expand ef_search candidates
        4. Return top-k by similarity
        """

        # Start from entry point
        current_nearest = [self.entry_point]

        # Descend through layers
        for layer in range(self.max_level, 0, -1):
            current_nearest = self.search_layer(
                query_embedding,
                entry_points=current_nearest,
                layer=layer,
                ef=1  # Greedy search in upper layers
            )

        # Final layer: expand to ef_search candidates
        candidates = self.search_layer(
            query_embedding,
            entry_points=current_nearest,
            layer=0,
            ef=ef_search
        )

        # Return top-k
        return sorted(candidates, key=lambda x: x.distance)[:k]

    def complexity_analysis(self, num_vectors):
        """Analyze HNSW complexity"""

        import math

        # Number of layers: log(N) / log(M)
        num_layers = int(math.log(num_vectors) / math.log(self.M))

        # Expected comparisons per query
        comparisons_per_layer = self.M  # Examine M connections per layer
        total_comparisons = num_layers * comparisons_per_layer

        # Memory per vector
        avg_connections = self.M * 1.5  # Higher in layer 0
        bytes_per_connection = 8  # 64-bit ID
        memory_per_vector = avg_connections * bytes_per_connection

        # Build time per vector
        # Must compare with ef_construction candidates
        build_comparisons = self.ef_construction * num_layers

        return {
            'num_layers': num_layers,
            'comparisons_per_query': total_comparisons,
            'memory_overhead_bytes': memory_per_vector,
            'build_comparisons_per_vector': build_comparisons,
            'query_complexity': f'O(log N) ≈ {total_comparisons} comparisons',
            'build_complexity': 'O(N log N) for full dataset'
        }

    def tune_for_scale(self, target_recall=0.95, target_latency_ms=50):
        """Tuning guidelines for trillion-scale"""

        recommendations = {
            'M': {
                'small_scale_1m': 16,
                'medium_scale_100m': 32,
                'large_scale_10b': 48,
                'trillion_scale_100t': 64,
                'rationale': 'Higher M = more connections = better recall but more memory'
            },

            'ef_construction': {
                'fast_build': 100,
                'balanced': 200,
                'high_quality': 400,
                'trillion_scale_recommendation': 300,
                'rationale': 'Higher ef = better index quality but slower builds'
            },

            'ef_search': {
                'very_fast_low_recall': 50,
                'balanced': 100,
                'high_recall': 200,
                'maximum_recall': 500,
                'trillion_scale_recommendation': 150,
                'rationale': 'Tune based on latency budget and recall requirements'
            },

            'optimization_tips': [
                'Use SIMD for distance calculations (4-8x speedup)',
                'Prefetch graph neighbors to avoid cache misses',
                'Store layer 0 on fast SSD, upper layers in RAM',
                'Batch queries to amortize graph traversal overhead',
                'Use progressive search (start low ef, increase if needed)',
                'Consider quantization (product quantization) for memory'
            ]
        }

        return recommendations

# Example: HNSW for 100B vectors
hnsw = HNSWDeepDive(M=48, ef_construction=300)
analysis = hnsw.complexity_analysis(num_vectors=100_000_000_000)

print("HNSW Analysis for 100B vectors:")
print(f"  Layers: {analysis['num_layers']}")
print(f"  Comparisons per query: {analysis['comparisons_per_query']}")
print(f"  Memory overhead: {analysis['memory_overhead_bytes']} bytes/vector")
