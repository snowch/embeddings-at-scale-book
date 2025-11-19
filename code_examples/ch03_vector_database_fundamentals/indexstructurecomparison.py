# Code from Chapter 03
# Book: Embeddings at Scale

class IndexStructureComparison:
    """Compare major index structures"""

    def compare_structures(self):
        """Index structure trade-offs"""

        return {
            'flat_index': {
                'structure': 'Scan all vectors',
                'build_time': 'O(N)',
                'query_time': 'O(N * D)',
                'memory': 'O(N * D)',
                'accuracy': '100%',
                'max_scale': '~1M vectors',
                'use_case': 'Ground truth, small datasets'
            },

            'ivf': {
                'name': 'Inverted File Index',
                'structure': 'Partition space into Voronoi cells',
                'build_time': 'O(N * k) where k = num partitions',
                'query_time': 'O((N/k) * D * n_probe)',
                'memory': 'O(N * D + k * D)',
                'accuracy': '80-95% (depends on n_probe)',
                'max_scale': '~1B vectors',
                'use_case': 'Balanced speed/accuracy'
            },

            'hnsw': {
                'name': 'Hierarchical Navigable Small World',
                'structure': 'Multi-layer proximity graph',
                'build_time': 'O(N * log(N) * M) where M = connections',
                'query_time': 'O(log(N) * M)',
                'memory': 'O(N * (D + M)) - vectors plus graph edges',
                'accuracy': '95-99%',
                'max_scale': '100B+ vectors',
                'use_case': 'High-performance production systems',
                'why_best': 'Best accuracy/speed tradeoff at scale'
            },

            'lsh': {
                'name': 'Locality-Sensitive Hashing',
                'structure': 'Hash similar vectors to same buckets',
                'build_time': 'O(N * L) where L = hash tables',
                'query_time': 'O(L * bucket_size)',
                'memory': 'O(N * L)',
                'accuracy': '70-90%',
                'max_scale': 'Trillion+ vectors',
                'use_case': 'Ultra-massive scale, can tolerate lower accuracy'
            },

            'pq': {
                'name': 'Product Quantization',
                'structure': 'Compress vectors via quantization',
                'build_time': 'O(N * iterations)',
                'query_time': 'O(N) with compressed distance',
                'memory': 'O(N * code_size) - very low',
                'accuracy': '85-95%',
                'max_scale': '10B+ vectors',
                'use_case': 'Memory-constrained environments',
                'often_combined_with': 'IVF for IVF-PQ hybrid'
            }
        }
