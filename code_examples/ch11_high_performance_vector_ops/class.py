# Code from Chapter 11
# Book: Embeddings at Scale

"""
Parallel Query Processing

Strategies:
1. Thread pooling: Handle multiple queries concurrently
2. Request batching: Group queries for GPU efficiency
3. Load balancing: Distribute queries across replicas
4. Query caching: Cache results for repeated queries
"""

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List

import numpy as np


# Placeholder classes - see import.py for full implementations
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
        return SearchResult(indices=np.array([0]), scores=np.array([1.0]), latency_ms=0.1)


# Placeholder for GPUVectorSearch - see gpuvectorsearch.py for full implementation
class GPUVectorSearch:
    """Placeholder for GPUVectorSearch."""

    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim

    def search(self, query, k=10):
        return SearchResult(
            indices=np.array(list(range(k))),
            scores=np.array([1.0 - i * 0.1 for i in range(k)]),
            latency_ms=1.0,
        )


@dataclass
class Query:
    """Query request"""

    query_id: str
    vector: np.ndarray
    k: int = 10
    timestamp: float = 0.0


@dataclass
class QueryResponse:
    """Query response"""

    query_id: str
    indices: np.ndarray
    scores: np.ndarray
    latency_ms: float


class ParallelVectorSearch:
    """
    Parallel query processing with thread pooling

    Architecture:
    - Request queue: Incoming queries
    - Worker pool: N threads processing queries
    - Response queue: Completed queries

    Concurrency model:
    - Each worker handles one query at a time
    - Multiple workers run in parallel
    - Thread-safe index access (read-only)

    Throughput:
    - Single-threaded: 100 queries/sec
    - 8-thread pool: 800 queries/sec (linear scaling for CPU-bound)
    - 8-thread + GPU: 5000 queries/sec (GPU handles actual search)

    Use for:
    - Serving layer (multiple concurrent users)
    - Batch processing (process dataset in parallel)
    - Load testing (simulate concurrent load)
    """

    def __init__(self, index, num_workers: int = 8):
        """
        Args:
            index: Search index (OptimizedExactSearch, IVFIndex, etc.)
            num_workers: Number of worker threads
        """
        self.index = index
        self.num_workers = num_workers

        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        print(f"Initialized parallel search with {num_workers} workers")

    def process_queries(self, queries: List[Query]) -> List[QueryResponse]:
        """
        Process multiple queries in parallel

        Args:
            queries: List of queries to process

        Returns:
            List of query responses
        """
        start_time = time.time()

        # Submit all queries to thread pool
        futures = {
            self.executor.submit(self._process_single_query, query): query for query in queries
        }

        # Collect results as they complete
        responses = []
        for future in as_completed(futures):
            response = future.result()
            responses.append(response)

        total_elapsed = (time.time() - start_time) * 1000

        print(f"Processed {len(queries)} queries in {total_elapsed:.2f} ms")
        print(f"  Throughput: {len(queries) / (total_elapsed / 1000):.0f} queries/sec")
        print(f"  Avg latency: {total_elapsed / len(queries):.2f} ms/query")

        return responses

    def _process_single_query(self, query: Query) -> QueryResponse:
        """
        Process single query (called by worker thread)

        Args:
            query: Query to process

        Returns:
            QueryResponse
        """
        query_start = time.time()

        # Search index
        result = self.index.search(query.vector, k=query.k)

        latency_ms = (time.time() - query_start) * 1000

        return QueryResponse(
            query_id=query.query_id,
            indices=result.indices,
            scores=result.scores,
            latency_ms=latency_ms,
        )

    def shutdown(self):
        """Shutdown thread pool"""
        self.executor.shutdown(wait=True)


class BatchedQueryProcessor:
    """
    Batched query processing for GPU efficiency

    Strategy:
    - Accumulate queries for short time window (10-50ms)
    - Process batch together on GPU
    - Amortize GPU kernel launch overhead

    Trade-off:
    - Added latency: 10-50ms queueing time
    - Increased throughput: 10-100× on GPU

    Typical settings:
    - Max batch size: 1000 queries
    - Max wait time: 20ms
    - Result: 95th percentile latency < 30ms, 50K queries/sec

    Use when:
    - High query volume (>1K queries/sec)
    - Latency tolerance (can wait 10-50ms)
    - GPU available (batching benefits)
    """

    def __init__(
        self,
        gpu_index: "GPUVectorSearch",
        max_batch_size: int = 1000,
        max_wait_time_ms: float = 20.0,
    ):
        """
        Args:
            gpu_index: GPU search index
            max_batch_size: Maximum queries per batch
            max_wait_time_ms: Maximum time to wait for batch to fill
        """
        self.gpu_index = gpu_index
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms

        # Query queue
        self.query_queue = queue.Queue()

        # Processing thread
        self.processing_thread = threading.Thread(target=self._batch_processing_loop)
        self.processing_thread.daemon = True
        self.running = False

        print("Initialized batched query processor")
        print(f"  Max batch size: {max_batch_size}")
        print(f"  Max wait time: {max_wait_time_ms} ms")

    def start(self):
        """Start background processing thread"""
        self.running = True
        self.processing_thread.start()
        print("Started batch processing thread")

    def stop(self):
        """Stop background processing thread"""
        self.running = False
        self.processing_thread.join()
        print("Stopped batch processing thread")

    def submit_query(self, query: Query, response_callback: Callable[[QueryResponse], None]):
        """
        Submit query for batched processing

        Args:
            query: Query to process
            response_callback: Callback to invoke with response
        """
        self.query_queue.put((query, response_callback))

    def _batch_processing_loop(self):
        """
        Background thread: Accumulate and process batches
        """
        while self.running:
            batch_queries = []
            batch_callbacks = []

            batch_start_time = time.time()

            # Accumulate batch
            while len(batch_queries) < self.max_batch_size:
                # Check if wait time exceeded
                elapsed_ms = (time.time() - batch_start_time) * 1000
                if elapsed_ms >= self.max_wait_time_ms and len(batch_queries) > 0:
                    break

                # Get next query (with timeout)
                timeout = (self.max_wait_time_ms - elapsed_ms) / 1000
                timeout = max(0.001, timeout)  # At least 1ms

                try:
                    query, callback = self.query_queue.get(timeout=timeout)
                    batch_queries.append(query)
                    batch_callbacks.append(callback)
                except queue.Empty:
                    if len(batch_queries) > 0:
                        break  # Process partial batch
                    continue  # Keep waiting

            # Process batch if non-empty
            if batch_queries:
                self._process_batch(batch_queries, batch_callbacks)

    def _process_batch(self, batch_queries: List[Query], batch_callbacks: List[Callable]):
        """
        Process batch of queries on GPU

        Args:
            batch_queries: Queries to process
            batch_callbacks: Response callbacks
        """
        # Extract query vectors
        query_vectors = np.array([q.vector for q in batch_queries])

        # Process batch on GPU
        result = self.gpu_index.search(query_vectors, k=batch_queries[0].k)

        # Dispatch responses
        for i, query in enumerate(batch_queries):
            response = QueryResponse(
                query_id=query.query_id,
                indices=result.indices[i] if result.indices.ndim > 1 else result.indices,
                scores=result.scores[i] if result.scores.ndim > 1 else result.scores,
                latency_ms=result.latency_ms / len(batch_queries),
            )
            batch_callbacks[i](response)


class LoadBalancedSearchCluster:
    """
    Load-balanced search cluster across multiple replicas

    Architecture:
    - N replicas of search index (identical copies)
    - Load balancer distributes queries
    - Strategies: Round-robin, least-load, latency-weighted

    Benefits:
    - Horizontal scaling (2× replicas → 2× throughput)
    - High availability (replica failures don't break service)
    - Geographic distribution (replicas near users)

    Typical deployment:
    - 3-10 replicas per region
    - 3-5 regions globally
    - Total: 10-50 replicas
    - Throughput: 1M queries/sec
    - Availability: 99.99%
    """

    def __init__(self, replicas: List):
        """
        Args:
            replicas: List of search index replicas
        """
        self.replicas = replicas
        self.num_replicas = len(replicas)

        # Round-robin counter
        self.next_replica = 0
        self.lock = threading.Lock()

        # Replica health tracking
        self.replica_health = [True] * self.num_replicas

        print(f"Initialized load-balanced cluster with {self.num_replicas} replicas")

    def search(self, query: np.ndarray, k: int = 10) -> SearchResult:
        """
        Search using load balancing

        Strategy: Round-robin across healthy replicas

        Args:
            query: Query vector
            k: Number of nearest neighbors

        Returns:
            SearchResult
        """
        # Select replica (round-robin)
        with self.lock:
            # Find next healthy replica
            attempts = 0
            while attempts < self.num_replicas:
                replica_idx = self.next_replica
                self.next_replica = (self.next_replica + 1) % self.num_replicas

                if self.replica_health[replica_idx]:
                    break

                attempts += 1
            else:
                raise RuntimeError("No healthy replicas available")

        # Search on selected replica
        try:
            result = self.replicas[replica_idx].search(query, k=k)
            return result
        except Exception as e:
            # Mark replica as unhealthy
            self.replica_health[replica_idx] = False
            print(f"⚠️  Replica {replica_idx} failed: {e}")

            # Retry on another replica
            return self.search(query, k=k)


# Example: Parallel query processing
def parallel_processing_example():
    """
    Demonstrate parallel query processing

    Scenario: 1K concurrent queries on 1M vector index
    """
    # Create index
    num_vectors = 100_000
    dim = 512

    print(f"Creating index with {num_vectors:,} vectors...")
    corpus = np.random.randn(num_vectors, dim).astype(np.float32)
    corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)

    index = OptimizedExactSearch(corpus, normalized=True, use_gpu=False)

    # Generate queries
    num_queries = 1000
    queries = [
        Query(query_id=f"query_{i}", vector=np.random.randn(dim).astype(np.float32), k=10)
        for i in range(num_queries)
    ]

    # Sequential processing (baseline)
    print("\n=== Sequential Processing ===")
    seq_start = time.time()
    for query in queries:
        index.search(query.vector, k=query.k)
    seq_elapsed = (time.time() - seq_start) * 1000

    print(f"Total time: {seq_elapsed:.2f} ms")
    print(f"Throughput: {num_queries / (seq_elapsed / 1000):.0f} queries/sec")

    # Parallel processing
    print("\n=== Parallel Processing (8 workers) ===")
    parallel_search = ParallelVectorSearch(index, num_workers=8)
    responses = parallel_search.process_queries(queries)
    parallel_search.shutdown()

    print(f"Speedup: {seq_elapsed / (sum(r.latency_ms for r in responses) / len(responses)):.1f}×")


# Uncomment to run:
# parallel_processing_example()
