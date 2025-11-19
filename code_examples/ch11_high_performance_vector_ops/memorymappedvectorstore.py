# Code from Chapter 11
# Book: Embeddings at Scale

"""
Memory-Mapped Vector Storage

Benefits:
- Access > RAM datasets (1B+ vectors on 128GB machine)
- Fast startup (no loading time)
- Shared memory across processes
- OS-managed caching (frequently accessed vectors cached)

Challenges:
- Slower than RAM (disk I/O latency)
- Random access patterns cause cache misses
- Need to optimize data layout for access patterns
"""

import os
import time

# Placeholder classes - see import.py for full implementations
from dataclasses import dataclass
from pathlib import Path

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

class MemoryMappedVectorStore:
    """
    Memory-mapped vector storage for > RAM datasets

    File format:
    - Binary file: All vectors concatenated
    - Layout: [vec0_dim0, vec0_dim1, ..., vec0_dimD, vec1_dim0, ...]
    - Enables fast sequential scans

    Access patterns:
    - Sequential scan: ~1GB/sec (disk throughput)
    - Random access: ~10K vectors/sec (seek latency)

    Optimization:
    - Cluster similar vectors together (spatial locality)
    - Align to page boundaries (4KB)
    - Prefetch next chunk while processing current

    Typical usage:
    - 1B vectors × 512 dims × 4 bytes = 2TB
    - Machine: 128GB RAM
    - OS caches frequently accessed ~10% (200M vectors)
    - Rest paged from SSD as needed
    """

    def __init__(
        self,
        file_path: str,
        dim: int,
        mode: str = 'r'
    ):
        """
        Args:
            file_path: Path to memory-mapped file
            dim: Embedding dimension
            mode: 'r' (read-only), 'w+' (read-write), or 'c' (copy-on-write)
        """
        self.file_path = Path(file_path)
        self.dim = dim
        self.mode = mode

        # Memory-map file
        if self.file_path.exists():
            # Load existing file
            self.mmap = np.memmap(
                str(self.file_path),
                dtype=np.float32,
                mode=mode
            )

            # Infer number of vectors
            self.num_vectors = len(self.mmap) // dim

            # Reshape to (N, d)
            self.vectors = self.mmap.reshape((self.num_vectors, dim))

            print("Loaded memory-mapped vector store")
            print(f"  Path: {file_path}")
            print(f"  Vectors: {self.num_vectors:,} × {dim} dims")
            print(f"  Size: {self.file_path.stat().st_size / 1e9:.2f} GB")
        else:
            # Create new file
            self.mmap = None
            self.vectors = None
            self.num_vectors = 0
            print(f"Created new memory-mapped vector store at {file_path}")

    def append(self, vectors: np.ndarray):
        """
        Append vectors to store

        Args:
            vectors: Vectors to append (N, d)
        """
        if self.mode == 'r':
            raise ValueError("Cannot append to read-only store")

        # Flatten vectors
        flat_vectors = vectors.reshape(-1).astype(np.float32)

        if self.mmap is None:
            # Create new file
            self.mmap = np.memmap(
                str(self.file_path),
                dtype=np.float32,
                mode='w+',
                shape=(len(flat_vectors),)
            )
            self.mmap[:] = flat_vectors
            self.num_vectors = len(vectors)
        else:
            # Append to existing file
            old_size = len(self.mmap)
            new_size = old_size + len(flat_vectors)

            # Resize memory map
            self.mmap.flush()
            self.mmap = np.memmap(
                str(self.file_path),
                dtype=np.float32,
                mode='r+',
                shape=(new_size,)
            )

            # Append new vectors
            self.mmap[old_size:] = flat_vectors
            self.num_vectors = new_size // self.dim

        # Reshape
        self.vectors = self.mmap.reshape((self.num_vectors, self.dim))

        # Flush to disk
        self.mmap.flush()

        print(f"Appended {len(vectors):,} vectors (total: {self.num_vectors:,})")

    def get_vector(self, idx: int) -> np.ndarray:
        """
        Get single vector by index

        Random access - may trigger disk I/O if not cached

        Args:
            idx: Vector index

        Returns:
            Vector (d,)
        """
        if idx >= self.num_vectors:
            raise IndexError(f"Index {idx} out of range (have {self.num_vectors} vectors)")

        return self.vectors[idx].copy()

    def get_vectors(self, indices: np.ndarray) -> np.ndarray:
        """
        Get multiple vectors by indices

        Args:
            indices: Vector indices (K,)

        Returns:
            Vectors (K, d)
        """
        return self.vectors[indices].copy()

    def scan(
        self,
        query: np.ndarray,
        k: int = 10,
        batch_size: int = 10000
    ) -> SearchResult:
        """
        Sequential scan for nearest neighbors

        Strategy:
        - Scan vectors in batches (sequential I/O - fast)
        - Maintain top-k heap
        - Process entire corpus

        Performance:
        - Sequential disk read: ~1GB/sec (SSD)
        - 1B vectors × 512 dims × 4 bytes = 2TB
        - Scan time: ~2000 seconds (33 minutes)

        Optimization:
        - Use GPU for batch similarity computation
        - Prefetch next batch while processing current
        - Early termination if upper bound known

        Args:
            query: Query vector (d,)
            k: Number of nearest neighbors
            batch_size: Vectors per batch

        Returns:
            SearchResult
        """
        start_time = time.time()

        # Normalize query
        query = query / np.linalg.norm(query)

        # Top-k heap
        top_k_scores = []
        top_k_indices = []

        # Scan in batches
        for i in range(0, self.num_vectors, batch_size):
            batch_end = min(i + batch_size, self.num_vectors)

            # Load batch (triggers disk I/O if not cached)
            batch = self.vectors[i:batch_end]

            # Compute similarities
            similarities = np.dot(batch, query)

            # Update top-k
            for j, sim in enumerate(similarities):
                if len(top_k_scores) < k:
                    top_k_scores.append(sim)
                    top_k_indices.append(i + j)
                elif sim > min(top_k_scores):
                    min_idx = top_k_scores.index(min(top_k_scores))
                    top_k_scores[min_idx] = sim
                    top_k_indices[min_idx] = i + j

        # Sort top-k
        sorted_pairs = sorted(zip(top_k_scores, top_k_indices), reverse=True)
        top_scores = np.array([s for s, _ in sorted_pairs])
        top_indices = np.array([i for _, i in sorted_pairs])

        latency_ms = (time.time() - start_time) * 1000

        return SearchResult(
            indices=top_indices,
            scores=top_scores,
            latency_ms=latency_ms
        )

    def close(self):
        """Flush and close memory-mapped file"""
        if self.mmap is not None:
            self.mmap.flush()
            del self.mmap
            del self.vectors

class TieredVectorStore:
    """
    Tiered storage: Hot vectors in RAM, cold vectors on disk

    Architecture:
    - Tier 1 (RAM): 10% most frequently accessed vectors
    - Tier 2 (Disk): Remaining 90% (memory-mapped)

    Access pattern tracking:
    - Count accesses per vector
    - Periodically promote hot vectors to RAM tier
    - Demote cold vectors to disk tier

    Performance:
    - 90% of queries hit RAM tier (0.1ms latency)
    - 10% of queries hit disk tier (10ms latency)
    - Average latency: 0.1 × 0.9 + 10 × 0.1 = 1.09ms

    Speedup: 10× vs. all-disk, 90% reduction vs. all-RAM cost
    """

    def __init__(
        self,
        disk_store: MemoryMappedVectorStore,
        ram_cache_size: int = 100000
    ):
        """
        Args:
            disk_store: Memory-mapped disk store
            ram_cache_size: Number of vectors to keep in RAM
        """
        self.disk_store = disk_store
        self.ram_cache_size = ram_cache_size

        # RAM cache: index → vector
        self.ram_cache = {}

        # Access tracking: index → access_count
        self.access_counts = {}

        print("Initialized tiered vector store")
        print(f"  Total vectors: {disk_store.num_vectors:,}")
        print(f"  RAM cache size: {ram_cache_size:,} ({ram_cache_size / disk_store.num_vectors:.1%})")

    def get_vector(self, idx: int) -> np.ndarray:
        """
        Get vector with tiered caching

        Args:
            idx: Vector index

        Returns:
            Vector (d,)
        """
        # Track access
        self.access_counts[idx] = self.access_counts.get(idx, 0) + 1

        # Check RAM cache
        if idx in self.ram_cache:
            return self.ram_cache[idx]

        # Load from disk
        vector = self.disk_store.get_vector(idx)

        # Add to cache if space available
        if len(self.ram_cache) < self.ram_cache_size:
            self.ram_cache[idx] = vector
        else:
            # Evict least frequently accessed
            self._evict_lfu()
            self.ram_cache[idx] = vector

        return vector

    def _evict_lfu(self):
        """Evict least frequently used vector from cache"""
        if not self.ram_cache:
            return

        # Find least frequently accessed cached vector
        cached_indices = list(self.ram_cache.keys())
        lfu_idx = min(cached_indices, key=lambda i: self.access_counts.get(i, 0))

        # Evict
        del self.ram_cache[lfu_idx]

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        total_accesses = sum(self.access_counts.values())
        cached_accesses = sum(
            count for idx, count in self.access_counts.items()
            if idx in self.ram_cache
        )

        hit_rate = cached_accesses / total_accesses if total_accesses > 0 else 0

        return {
            'cache_size': len(self.ram_cache),
            'cache_capacity': self.ram_cache_size,
            'total_accesses': total_accesses,
            'cache_hit_rate': hit_rate
        }

# Example: Memory-mapped storage
def memory_mapped_example():
    """
    Demonstrate memory-mapped vector storage

    Scenario: 1M vectors, simulate > RAM dataset
    """
    dim = 512
    num_vectors = 1_000_000
    file_path = "/tmp/vectors.mmap"

    # Create store
    print("Creating memory-mapped store...")
    store = MemoryMappedVectorStore(file_path, dim=dim, mode='w+')

    # Append vectors in batches (simulating large-scale ingestion)
    batch_size = 100000
    for i in range(0, num_vectors, batch_size):
        batch = np.random.randn(batch_size, dim).astype(np.float32)
        batch = batch / np.linalg.norm(batch, axis=1, keepdims=True)
        store.append(batch)
        print(f"  Progress: {i + batch_size:,}/{num_vectors:,}")

    store.close()

    # Reopen for reading
    print("\nReopening store...")
    store = MemoryMappedVectorStore(file_path, dim=dim, mode='r')

    # Random access
    print("\nRandom access test...")
    random_indices = np.random.randint(0, num_vectors, size=100)

    start_time = time.time()
    for idx in random_indices:
        store.get_vector(idx)
    elapsed = time.time() - start_time

    print(f"Random access: {elapsed:.3f}s for 100 vectors ({elapsed/100*1000:.2f} ms/vector)")

    # Sequential scan search
    print("\nSequential scan search...")
    query = np.random.randn(dim).astype(np.float32)
    result = store.scan(query, k=10, batch_size=10000)

    print(f"Scan complete: {result.latency_ms:.2f} ms")
    print(f"Top-5 scores: {result.scores[:5]}")

    # Clean up
    store.close()
    os.remove(file_path)
    print(f"\nCleaned up {file_path}")

# Uncomment to run:
# memory_mapped_example()
