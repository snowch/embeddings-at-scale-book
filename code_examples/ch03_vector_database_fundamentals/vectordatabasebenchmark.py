# Code from Chapter 03
# Book: Embeddings at Scale

class VectorDatabaseBenchmark:
    """Comprehensive benchmarking framework"""

    def __init__(self, dataset_size, embedding_dim, index_type):
        self.dataset_size = dataset_size
        self.embedding_dim = embedding_dim
        self.index_type = index_type

    def benchmark_dimensions(self):
        """What to benchmark"""

        return {
            'index_build_performance': {
                'metrics': [
                    'Build time (hours)',
                    'Build throughput (vectors/second)',
                    'Peak memory usage (GB)',
                    'CPU utilization (%)',
                    'Disk I/O (MB/s)'
                ],
                'variables': [
                    'Dataset size',
                    'Embedding dimensions',
                    'Index parameters (M, ef_construction)',
                    'Hardware (CPU, RAM, disk type)'
                ]
            },

            'query_performance': {
                'metrics': [
                    'p50, p95, p99, p99.9 latency',
                    'Throughput (QPS)',
                    'Recall@10, Recall@100',
                    'Memory usage during queries',
                    'CPU usage during queries'
                ],
                'variables': [
                    'Number of queries (load testing)',
                    'K (number of neighbors requested)',
                    'ef_search parameter',
                    'Query distribution (random vs clustered)',
                    'Concurrent query load'
                ]
            },

            'update_performance': {
                'metrics': [
                    'Insert latency',
                    'Insert throughput',
                    'Query latency during inserts (degradation)',
                    'Index quality after inserts (recall drift)'
                ],
                'variables': [
                    'Insert rate (vectors/second)',
                    'Fraction of dataset updated',
                    'Insert pattern (random vs sequential)'
                ]
            },

            'scalability': {
                'metrics': [
                    'Latency vs dataset size',
                    'Memory vs dataset size',
                    'Build time vs dataset size',
                    'Throughput vs number of shards'
                ],
                'test': 'Run same benchmark at 1M, 10M, 100M, 1B, 10B vectors'
            },

            'cost_efficiency': {
                'metrics': [
                    'Cost per million queries',
                    'Cost per billion embeddings stored',
                    'Infrastructure cost ($/month)',
                    'Cost vs recall tradeoff'
                ],
                'calculation': 'Amortize hardware + electricity + personnel costs'
            }
        }

    def standard_benchmark_datasets(self):
        """Industry-standard datasets for comparison"""

        return {
            'sift1m': {
                'size': 1_000_000,
                'dimensions': 128,
                'domain': 'Images (SIFT descriptors)',
                'use': 'Small-scale baseline',
                'download': 'http://corpus-texmex.irisa.fr/'
            },

            'deep1b': {
                'size': 1_000_000_000,
                'dimensions': 96,
                'domain': 'Images (deep learning features)',
                'use': 'Billion-scale benchmark',
                'download': 'http://sites.skoltech.ru/compvision/noimi/'
            },

            'msturing1b': {
                'size': 1_000_000_000,
                'dimensions': 100,
                'domain': 'Web documents',
                'use': 'Production-scale benchmark',
                'download': 'https://github.com/microsoft/SPTAG'
            },

            'laion5b': {
                'size': 5_000_000_000,
                'dimensions': 768,
                'domain': 'Image-text embeddings (CLIP)',
                'use': 'Multi-modal, massive scale',
                'download': 'https://laion.ai/blog/laion-5b/'
            },

            'custom': {
                'recommendation': 'Use your own production data for most accurate benchmark',
                'reason': 'Production queries have different distribution than academic datasets'
            }
        }

    def run_benchmark_suite(self):
        """Execute comprehensive benchmark"""

        import time
        import numpy as np

        results = {}

        # 1. Index Build Benchmark
        print(f"Building index for {self.dataset_size:,} vectors...")
        build_start = time.time()

        # Generate synthetic embeddings
        embeddings = np.random.randn(self.dataset_size, self.embedding_dim).astype(np.float32)

        # Build index (pseudo-code for illustration)
        # In real benchmark, use actual vector DB
        index = self.build_index(embeddings)

        build_time = time.time() - build_start
        results['build_time_seconds'] = build_time
        results['build_throughput_vec_per_sec'] = self.dataset_size / build_time

        # 2. Query Latency Benchmark
        print("Benchmarking query latency...")
        num_queries = 10000
        query_latencies = []

        for i in range(num_queries):
            query = np.random.randn(self.embedding_dim).astype(np.float32)

            start = time.time()
            results_ids = index.search(query, k=10)
            latency_ms = (time.time() - start) * 1000

            query_latencies.append(latency_ms)

        query_latencies = np.array(query_latencies)
        results['query_latency_p50_ms'] = np.percentile(query_latencies, 50)
        results['query_latency_p95_ms'] = np.percentile(query_latencies, 95)
        results['query_latency_p99_ms'] = np.percentile(query_latencies, 99)

        # 3. Recall Benchmark
        print("Benchmarking recall...")
        results['recall_at_10'] = self.measure_recall(index, embeddings, k=10)
        results['recall_at_100'] = self.measure_recall(index, embeddings, k=100)

        # 4. Throughput Benchmark
        print("Benchmarking throughput...")
        duration_seconds = 60
        results['throughput_qps'] = self.measure_throughput(index, duration_seconds)

        return results

    def measure_recall(self, index, embeddings, k=10, num_test_queries=100):
        """Measure recall@k"""

        total_recall = 0

        for _ in range(num_test_queries):
            # Random query from dataset
            query_idx = np.random.randint(0, len(embeddings))
            query = embeddings[query_idx]

            # Ground truth: exact nearest neighbors (brute force)
            distances = np.linalg.norm(embeddings - query, axis=1)
            true_top_k = set(np.argsort(distances)[:k])

            # Approximate search
            approx_top_k = set(index.search(query, k=k))

            # Recall: overlap / k
            recall = len(true_top_k & approx_top_k) / k
            total_recall += recall

        return total_recall / num_test_queries

    def measure_throughput(self, index, duration_seconds):
        """Measure queries per second"""

        import time

        query_count = 0
        end_time = time.time() + duration_seconds

        while time.time() < end_time:
            query = np.random.randn(self.embedding_dim).astype(np.float32)
            index.search(query, k=10)
            query_count += 1

        return query_count / duration_seconds

    def build_index(self, embeddings):
        """Placeholder for actual index building"""
        # In real implementation, use actual vector DB library
        # e.g., Faiss, Milvus, Pinecone SDK

        class DummyIndex:
            def search(self, query, k):
                # Simulate search
                return list(range(k))

        return DummyIndex()

# Example: Run benchmark
benchmark = VectorDatabaseBenchmark(
    dataset_size=10_000_000,  # 10M vectors
    embedding_dim=768,
    index_type='HNSW'
)

# Note: This is illustrative - real benchmarks take hours/days
# results = benchmark.run_benchmark_suite()
# print(results)
