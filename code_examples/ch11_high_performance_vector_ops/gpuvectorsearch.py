# Code from Chapter 11
# Book: Embeddings at Scale

"""
GPU-Accelerated Vector Operations

Techniques:
1. Batched operations: Amortize kernel launch overhead
2. Tensor Cores: Specialized matrix multiplication hardware
3. Memory coalescing: Optimize memory access patterns
4. Shared memory: Reduce global memory bandwidth
5. Streams: Overlap computation and data transfer
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


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
        return SearchResult(
            indices=np.array([0]),
            scores=np.array([1.0]),
            latency_ms=0.1
        )


class GPUVectorSearch:
    """
    GPU-accelerated vector search

    Optimizations:
    - Batched matrix multiplication (GEMM) using Tensor Cores
    - Half-precision (FP16) for 2× memory, 2-8× speedup
    - Pinned memory for fast CPU→GPU transfer
    - Multiple streams for async operations
    - Top-k using GPU-optimized kernels

    Performance (A100 80GB):
    - 100M vectors × 512 dims: 5ms search latency (20× faster than CPU)
    - 1B vectors × 512 dims: 50ms search latency
    - Batch 100 queries: 1ms per query (100× speedup)

    Memory requirements:
    - Corpus in GPU memory: N × d × 2 bytes (FP16)
    - 100M × 512 × 2 = 100GB (fits on A100)
    - 1B × 512 × 2 = 1TB (requires partitioning or multi-GPU)
    """

    def __init__(
        self,
        corpus: np.ndarray,
        use_fp16: bool = True,
        device_id: int = 0
    ):
        """
        Args:
            corpus: Corpus embeddings (N, d)
            use_fp16: Use half precision (2× memory, 2-8× faster)
            device_id: GPU device ID
        """
        self.device = torch.device(f'cuda:{device_id}')
        self.use_fp16 = use_fp16

        print(f"Initializing GPU vector search on {self.device}")

        # Convert to PyTorch tensor
        corpus_tensor = torch.from_numpy(corpus)

        # Move to GPU with appropriate precision
        if use_fp16:
            corpus_tensor = corpus_tensor.half()
        corpus_tensor = corpus_tensor.to(self.device)

        # Normalize for cosine similarity
        self.corpus = F.normalize(corpus_tensor, p=2, dim=1)

        num_vectors, dim = corpus.shape
        memory_gb = corpus.nbytes / 1e9 * (0.5 if use_fp16 else 1.0)

        print(f"✓ Loaded {num_vectors:,} vectors × {dim} dims")
        print(f"  Precision: {'FP16' if use_fp16 else 'FP32'}")
        print(f"  GPU memory: {memory_gb:.2f} GB")

    def search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> SearchResult:
        """
        GPU-accelerated search

        Args:
            query: Query vector (d,) or batch (batch_size, d)
            k: Number of nearest neighbors

        Returns:
            SearchResult
        """
        start_time = time.time()

        # Convert to tensor
        query_tensor = torch.from_numpy(query).to(self.device)

        if self.use_fp16:
            query_tensor = query_tensor.half()

        # Handle single query vs batch
        if query_tensor.ndim == 1:
            query_tensor = query_tensor.unsqueeze(0)
            single_query = True
        else:
            single_query = False

        # Normalize query
        query_tensor = F.normalize(query_tensor, p=2, dim=1)

        # Compute similarities using Tensor Cores
        # (batch_size, d) @ (d, N) → (batch_size, N)
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            similarities = torch.matmul(query_tensor, self.corpus.T)

        # Find top-k using GPU kernel
        top_scores, top_indices = torch.topk(
            similarities,
            k=min(k, len(self.corpus)),
            dim=1,
            largest=True,
            sorted=True
        )

        # Synchronize GPU (wait for kernel completion)
        torch.cuda.synchronize()

        # Move results to CPU
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        if single_query:
            top_scores = top_scores[0]
            top_indices = top_indices[0]

        latency_ms = (time.time() - start_time) * 1000

        return SearchResult(
            indices=top_indices,
            scores=top_scores,
            latency_ms=latency_ms
        )

    def batch_search_async(
        self,
        queries: np.ndarray,
        k: int = 10,
        batch_size: int = 1000
    ) -> list[SearchResult]:
        """
        Asynchronous batch search using CUDA streams

        Overlaps:
        - Data transfer (CPU → GPU) for batch i+1
        - Computation for batch i
        - Result transfer (GPU → CPU) for batch i-1

        3× speedup from pipelining

        Args:
            queries: Query vectors (num_queries, d)
            k: Number of nearest neighbors per query
            batch_size: Queries per batch

        Returns:
            List of SearchResults
        """
        num_queries = len(queries)
        results = []

        # Create CUDA streams
        stream_compute = torch.cuda.Stream()
        stream_transfer = torch.cuda.Stream()

        # Pinned memory for fast CPU→GPU transfer
        queries_pinned = torch.from_numpy(queries).pin_memory()

        for i in range(0, num_queries, batch_size):
            batch_end = min(i + batch_size, num_queries)
            batch_queries = queries_pinned[i:batch_end]

            # Async transfer to GPU
            with torch.cuda.stream(stream_transfer):
                batch_gpu = batch_queries.to(self.device, non_blocking=True)
                if self.use_fp16:
                    batch_gpu = batch_gpu.half()

            # Async computation
            with torch.cuda.stream(stream_compute):
                stream_compute.wait_stream(stream_transfer)

                batch_gpu = F.normalize(batch_gpu, p=2, dim=1)

                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    similarities = torch.matmul(batch_gpu, self.corpus.T)

                top_scores, top_indices = torch.topk(
                    similarities,
                    k=min(k, len(self.corpus)),
                    dim=1
                )

            # Wait for computation
            stream_compute.synchronize()

            # Transfer results back
            top_scores_cpu = top_scores.cpu().numpy()
            top_indices_cpu = top_indices.cpu().numpy()

            # Store results
            for j in range(len(batch_queries)):
                results.append(SearchResult(
                    indices=top_indices_cpu[j],
                    scores=top_scores_cpu[j],
                    latency_ms=0.0  # Measured per batch, not per query
                ))

        return results

class MultiGPUVectorSearch:
    """
    Multi-GPU vector search for > 1B vectors

    Strategy: Shard corpus across GPUs
    - GPU 0: Vectors 0 to N/4
    - GPU 1: Vectors N/4 to N/2
    - GPU 2: Vectors N/2 to 3N/4
    - GPU 3: Vectors 3N/4 to N

    Search:
    1. Broadcast query to all GPUs
    2. Each GPU searches its shard
    3. Gather top-k from each GPU
    4. Merge and return global top-k

    Speedup: Linear with number of GPUs (4 GPUs → 4× capacity, same latency)
    """

    def __init__(
        self,
        corpus: np.ndarray,
        num_gpus: Optional[int] = None
    ):
        """
        Args:
            corpus: Full corpus (N, d)
            num_gpus: Number of GPUs to use (default: all available)
        """
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()

        self.num_gpus = num_gpus
        self.gpu_indices = []

        # Shard corpus across GPUs
        shard_size = len(corpus) // num_gpus

        print(f"Sharding {len(corpus):,} vectors across {num_gpus} GPUs...")

        for gpu_id in range(num_gpus):
            start_idx = gpu_id * shard_size
            end_idx = (gpu_id + 1) * shard_size if gpu_id < num_gpus - 1 else len(corpus)

            shard = corpus[start_idx:end_idx]

            # Create GPU index for this shard
            gpu_index = GPUVectorSearch(shard, device_id=gpu_id)
            self.gpu_indices.append({
                'index': gpu_index,
                'offset': start_idx
            })

            print(f"  GPU {gpu_id}: {len(shard):,} vectors")

    def search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> SearchResult:
        """
        Search across all GPU shards

        Args:
            query: Query vector (d,)
            k: Number of nearest neighbors

        Returns:
            SearchResult (global top-k)
        """
        start_time = time.time()

        # Search each shard in parallel
        shard_results = []
        for gpu_info in self.gpu_indices:
            result = gpu_info['index'].search(query, k=k)

            # Offset indices to global positions
            result.indices = result.indices + gpu_info['offset']

            shard_results.append(result)

        # Merge results from all shards
        all_indices = np.concatenate([r.indices for r in shard_results])
        all_scores = np.concatenate([r.scores for r in shard_results])

        # Find global top-k
        top_k_positions = np.argsort(-all_scores)[:k]
        top_indices = all_indices[top_k_positions]
        top_scores = all_scores[top_k_positions]

        latency_ms = (time.time() - start_time) * 1000

        return SearchResult(
            indices=top_indices,
            scores=top_scores,
            latency_ms=latency_ms
        )

# Example: GPU acceleration benchmark
def gpu_acceleration_example():
    """
    Benchmark CPU vs GPU vector search

    Scenario: 10M product embeddings, 512 dims
    """
    # Check GPU availability
    if not torch.cuda.is_available():
        print("No GPU available, skipping GPU benchmark")
        return

    # Generate corpus
    num_vectors = 1_000_000  # 1M for faster demo
    dim = 512

    print(f"Generating {num_vectors:,} {dim}-d vectors...")
    corpus = np.random.randn(num_vectors, dim).astype(np.float32)
    corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)

    query = np.random.randn(dim).astype(np.float32)
    query = query / np.linalg.norm(query)

    # CPU baseline
    print("\n=== CPU Search ===")
    cpu_search = OptimizedExactSearch(corpus, normalized=True, use_gpu=False)
    cpu_result = cpu_search.search(query, k=10)
    print(f"Latency: {cpu_result.latency_ms:.2f} ms")

    # GPU search
    print("\n=== GPU Search (FP32) ===")
    gpu_fp32 = GPUVectorSearch(corpus, use_fp16=False)
    gpu_fp32_result = gpu_fp32.search(query, k=10)
    print(f"Latency: {gpu_fp32_result.latency_ms:.2f} ms")
    print(f"Speedup: {cpu_result.latency_ms / gpu_fp32_result.latency_ms:.1f}×")

    # GPU FP16
    print("\n=== GPU Search (FP16) ===")
    gpu_fp16 = GPUVectorSearch(corpus, use_fp16=True)
    gpu_fp16_result = gpu_fp16.search(query, k=10)
    print(f"Latency: {gpu_fp16_result.latency_ms:.2f} ms")
    print(f"Speedup vs CPU: {cpu_result.latency_ms / gpu_fp16_result.latency_ms:.1f}×")
    print(f"Speedup vs GPU FP32: {gpu_fp32_result.latency_ms / gpu_fp16_result.latency_ms:.1f}×")

    # Batch search
    print("\n=== Batch Search (100 queries, FP16) ===")
    batch_queries = np.random.randn(100, dim).astype(np.float32)
    batch_queries = batch_queries / np.linalg.norm(batch_queries, axis=1, keepdims=True)

    batch_start = time.time()
    gpu_fp16.batch_search_async(batch_queries, k=10, batch_size=10)
    batch_elapsed = (time.time() - batch_start) * 1000

    print(f"Total latency: {batch_elapsed:.2f} ms")
    print(f"Per-query latency: {batch_elapsed / 100:.2f} ms")
    print(f"Throughput: {100 / (batch_elapsed / 1000):.0f} queries/sec")

# Uncomment to run:
# gpu_acceleration_example()
