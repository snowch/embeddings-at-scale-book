# Code from Chapter 03
# Book: Embeddings at Scale

class ProductionVectorDatabaseArchitecture:
    """Reference architecture for trillion-scale vector DB"""

    def __init__(self):
        self.components = self.define_components()

    def define_components(self):
        """Core components of production vector DB"""

        return {
            'ingestion_layer': {
                'responsibility': 'Accept and validate embeddings',
                'components': [
                    'API gateway (REST/gRPC)',
                    'Validation (dimension check, normalization)',
                    'Batching (group inserts for efficiency)',
                    'Rate limiting (protect from overload)'
                ],
                'throughput': '100K-1M embeddings/second',
                'key_challenges': [
                    'Hot partitions (uneven write distribution)',
                    'Duplicate detection at scale',
                    'Schema evolution (embedding dimension changes)'
                ]
            },

            'storage_layer': {
                'responsibility': 'Persist embeddings and indices',
                'components': [
                    'Raw embedding storage (object storage like S3)',
                    'Index storage (fast SSD/NVMe)',
                    'Metadata storage (traditional DB for filtering)',
                    'WAL (write-ahead log for durability)'
                ],
                'characteristics': {
                    'raw_embeddings': 'Cold storage, rarely accessed after indexing',
                    'indices': 'Hot storage, constantly accessed',
                    'metadata': 'Separate system, joined at query time'
                },
                'cost_optimization': [
                    'Tier embeddings: hot (SSD) → warm (HDD) → cold (S3)',
                    'Compress embeddings in cold storage',
                    'Index-only serving (keep raw embeddings offline)'
                ]
            },

            'index_layer': {
                'responsibility': 'Fast similarity search',
                'components': [
                    'HNSW graphs (primary index)',
                    'IVF indices (coarse quantization)',
                    'Product quantization (compression)',
                    'Index builder (background reindexing)'
                ],
                'key_parameters': {
                    'hnsw_m': '16-64 (connections per vertex)',
                    'hnsw_ef_construction': '200-500 (build accuracy)',
                    'hnsw_ef_search': '50-200 (query accuracy)',
                    'tradeoff': 'Higher values = better accuracy, more memory/time'
                },
                'build_strategy': [
                    'Incremental updates for real-time inserts',
                    'Batch rebuilds for major reindexing',
                    'Multi-version indices for zero-downtime updates'
                ]
            },

            'query_layer': {
                'responsibility': 'Execute searches efficiently',
                'components': [
                    'Query parser (parse filters + vector query)',
                    'Query planner (optimize execution)',
                    'Distributed query executor (fan-out to shards)',
                    'Result aggregator (merge + re-rank)',
                    'Cache (LRU cache for hot queries)'
                ],
                'optimizations': [
                    'Early termination (stop after k good results)',
                    'Adaptive search (adjust ef_search based on quality)',
                    'Pre-filtering (apply metadata filters before vector search)',
                    'Post-filtering (apply filters after vector search)',
                    'Hybrid search (combine vector + keyword)'
                ]
            },

            'metadata_layer': {
                'responsibility': 'Filter and join with attributes',
                'components': [
                    'Metadata database (PostgreSQL, Elasticsearch)',
                    'Filter optimizer (push down filters)',
                    'Join coordinator (combine vector + metadata results)'
                ],
                'patterns': {
                    'pre_filtering': 'Filter metadata → search embeddings',
                    'post_filtering': 'Search embeddings → filter results',
                    'hybrid': 'Parallel search + filter → merge',
                    'choice_depends_on': 'Selectivity of filters'
                }
            },

            'serving_layer': {
                'responsibility': 'Serve queries with SLA guarantees',
                'components': [
                    'Load balancer (distribute queries)',
                    'Query router (route to appropriate shards)',
                    'Circuit breaker (fail fast on overload)',
                    'Adaptive throttling (shed load gracefully)'
                ],
                'sla_targets': {
                    'p50_latency': '<20ms',
                    'p95_latency': '<50ms',
                    'p99_latency': '<100ms',
                    'availability': '99.99%',
                    'throughput': '100K QPS per region'
                }
            },

            'monitoring_layer': {
                'responsibility': 'Observe system health and quality',
                'metrics': [
                    'Query latency (p50, p95, p99, p99.9)',
                    'Throughput (QPS, inserts/sec)',
                    'Index quality (recall@10, recall@100)',
                    'Resource utilization (CPU, memory, disk I/O)',
                    'Error rates (timeouts, failures)',
                    'Data quality (embedding distribution, anomalies)'
                ],
                'alerts': [
                    'Latency SLA breach (p99 > threshold)',
                    'Recall degradation (index needs rebuild)',
                    'Resource saturation (scale up needed)',
                    'Skew detection (hot shards)',
                    'Index corruption (checksum failures)'
                ]
            }
        }

    def design_pattern_for_scale(self):
        """Design patterns that enable trillion-row scale"""

        return {
            'sharding': {
                'strategy': 'Horizontal partitioning across machines',
                'sharding_key': 'Hash of vector ID or random',
                'num_shards': '1000-10000 for trillion-scale',
                'shard_size': '100M-1B vectors per shard',
                'rebalancing': 'Online resharding with zero downtime'
            },

            'replication': {
                'strategy': 'Multi-copy for availability and performance',
                'replication_factor': '3x (tolerates 2 failures)',
                'consistency_model': 'Eventual consistency for inserts',
                'read_strategy': 'Read from nearest replica',
                'write_strategy': 'Async replication after write acknowledgment'
            },

            'caching': {
                'query_cache': 'LRU cache for frequent queries',
                'index_cache': 'Keep hot index pages in memory',
                'embedding_cache': 'Cache frequently accessed embeddings',
                'cache_size': '10-20% of total data',
                'cache_hit_rate': '>80% for production workloads'
            },

            'versioning': {
                'index_versions': 'Multiple index versions for AB testing',
                'schema_versions': 'Support dimension changes gracefully',
                'rollback': 'Quick rollback to previous index version',
                'blue_green': 'Zero-downtime index updates'
            }
        }

# Example: Calculating shard configuration
class ShardCalculator:
    """Calculate optimal sharding configuration"""

    def calculate_shards(self, total_vectors, vectors_per_shard_target, replication_factor=3):
        """
        Determine optimal shard count

        Example: 256T vectors, target 256M vectors/shard
        """
        ideal_shards = total_vectors / vectors_per_shard_target

        # Round to nearest power-of-2 for consistent hashing
        import math
        actual_shards = 2 ** math.ceil(math.log2(ideal_shards))

        vectors_per_shard = total_vectors / actual_shards

        # Storage calculation
        embedding_dim = 768
        bytes_per_vector = embedding_dim * 4  # float32

        # HNSW index overhead (~1.5x raw data)
        index_overhead = 1.5

        storage_per_shard_gb = (
            vectors_per_shard * bytes_per_vector * index_overhead / (1024**3)
        )

        total_storage_gb = storage_per_shard_gb * actual_shards * replication_factor

        return {
            'total_vectors': total_vectors,
            'num_shards': actual_shards,
            'vectors_per_shard': vectors_per_shard,
            'storage_per_shard_gb': storage_per_shard_gb,
            'total_storage_gb': total_storage_gb,
            'total_storage_tb': total_storage_gb / 1024,
            'replication_factor': replication_factor,
            'recommended_machine_spec': {
                'memory_gb': storage_per_shard_gb * 1.2,  # 20% overhead
                'cpu_cores': 16,
                'disk_type': 'NVMe SSD',
                'network': '10Gbps+'
            }
        }

# Calculate for 256T vectors
calc = ShardCalculator()
config = calc.calculate_shards(
    total_vectors=256_000_000_000_000,
    vectors_per_shard_target=256_000_000
)

print(f"Shard configuration for 256T vectors:")
print(f"  Shards: {config['num_shards']:,}")
print(f"  Vectors/shard: {config['vectors_per_shard']:,}")
print(f"  Storage/shard: {config['storage_per_shard_gb']:.1f} GB")
print(f"  Total storage: {config['total_storage_tb']:.1f} TB")
