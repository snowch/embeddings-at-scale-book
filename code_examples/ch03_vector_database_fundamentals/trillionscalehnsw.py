# Code from Chapter 03
# Book: Embeddings at Scale

class TrillionScaleHNSW:
    """Production patterns for HNSW at massive scale"""

    def sharding_strategy(self):
        """Distribute HNSW across shards"""

        return {
            'approach': 'Horizontal sharding with random assignment',
            'shard_size': '100M-1B vectors per shard',
            'search_strategy': 'Query all shards in parallel, merge results',

            'optimization_merge_strategy': {
                'naive': 'Search all shards, merge top-k',
                'smart': 'Search subset of shards, expand if needed',
                'adaptive': 'Learn which shards have relevant vectors'
            },

            'alternative_ivf_hnsw_hybrid': {
                'description': 'Use IVF for coarse partitioning, HNSW within partitions',
                'benefit': 'Only search relevant partitions',
                'search_flow': [
                    '1. IVF coarse search → identify relevant partitions',
                    '2. HNSW fine search within partitions',
                    '3. Merge results across partitions'
                ],
                'speedup': '5-10x at trillion scale'
            }
        }

    def incremental_updates(self):
        """Handle updates without full rebuild"""

        return {
            'online_inserts': {
                'strategy': 'Add to existing HNSW graph',
                'cost': f'O(log N * M * ef_construction)',
                'latency': '10-100ms per insert',
                'when_to_use': 'Continuous product catalog updates'
            },

            'batch_inserts': {
                'strategy': 'Build mini-HNSW, merge with main graph',
                'cost': 'Lower per-vector cost than online',
                'latency': 'Amortized over batch',
                'when_to_use': 'Daily/hourly batch loads'
            },

            'deletions': {
                'soft_delete': 'Mark deleted, filter at query time',
                'hard_delete': 'Remove from graph, reconnect neighbors',
                'recommendation': 'Soft delete with periodic rebuild'
            },

            'updates': {
                'strategy': 'Delete + re-insert',
                'optimization': 'If vector change is small, update in place',
                'when_to_rebuild': 'Rebuild when >20% of vectors changed'
            }
        }

    def memory_optimization(self):
        """Reduce memory footprint"""

        return {
            'graph_compression': {
                'technique': 'Compress connection lists',
                'savings': '30-50%',
                'tradeoff': 'Slight decompression overhead'
            },

            'tiered_storage': {
                'layer_0_hot': 'Keep in RAM (accessed most)',
                'upper_layers_warm': 'Keep on fast SSD (accessed less)',
                'embedding_vectors_cold': 'Keep on slow storage (only for reranking)',
                'savings': '60-80% memory reduction',
                'latency_impact': '< 10ms added for SSD access'
            },

            'mmap_strategy': {
                'description': 'Memory-map index file, let OS manage paging',
                'benefit': 'Automatic hot/cold page management',
                'works_well_for': 'Indices larger than RAM'
            },

            'quantization': {
                'technique': 'Store vector IDs as 32-bit instead of 64-bit',
                'savings': '50% on graph structure',
                'limitation': 'Limits to 4B vectors per shard (usually fine)'
            }
        }

# Calculate memory savings
def calculate_memory_savings():
    vectors_per_shard = 1_000_000_000  # 1B
    M = 48
    embedding_dim = 768

    # Baseline: everything in RAM
    baseline_memory_gb = (
        vectors_per_shard * embedding_dim * 4 +  # Embeddings (float32)
        vectors_per_shard * M * 8  # Graph (64-bit IDs)
    ) / (1024 ** 3)

    # Optimized: tiered storage + compression
    optimized_memory_gb = (
        vectors_per_shard * M * 4 * 0.7  # Graph (32-bit IDs + compression)
    ) / (1024 ** 3)
    # Embeddings on SSD, not in RAM

    savings_pct = (1 - optimized_memory_gb / baseline_memory_gb) * 100

    return {
        'baseline_memory_gb': baseline_memory_gb,
        'optimized_memory_gb': optimized_memory_gb,
        'savings_pct': savings_pct,
        'savings_explanation': 'Move embeddings to SSD, compress graph, use 32-bit IDs'
    }

print(calculate_memory_savings())
