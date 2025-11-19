import json

# Code from Chapter 23
# Book: Embeddings at Scale

"""
Multi-Tier Caching for Hot Embeddings

Architecture:
1. L1 cache: Hot embeddings in process memory (GB scale)
2. L2 cache: Warm embeddings in shared memory (10s GB scale)
3. L3 cache: Cold embeddings on local SSD (100s GB scale)
4. L4 storage: Full dataset in distributed storage (PB scale)
5. Result cache: Cache entire query results
6. Negative cache: Cache "not found" to avoid repeated lookups

Techniques:
- Adaptive replacement: Learn access patterns, optimize eviction
- Prefetching: Predict likely queries, preload embeddings
- Compression: Store compressed embeddings in cache
- Hierarchical caching: Small hot cache + large warm cache
- Query result caching: Cache top-k results for repeated queries
- Probabilistic structures: Bloom filters for negative caching

Performance targets:
- Cache hit rate: >70% for L1+L2, >90% for L1+L2+L3
- Cache lookup latency: <0.1ms L1, <1ms L2, <5ms L3
- Memory efficiency: >100× compression with <5% accuracy loss
- Invalidation latency: <100ms for critical updates
"""

import hashlib
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CacheEntry:
    """
    Cached embedding or query result
    
    Attributes:
        key: Cache key (embedding ID or query hash)
        value: Cached data (embedding vector or query results)
        size_bytes: Memory size of cached data
        access_count: Number of times accessed
        last_access: Timestamp of last access
        creation_time: When entry was added to cache
        ttl_seconds: Time-to-live (None = no expiration)
        version: Data version (for invalidation)
        compressed: Whether value is compressed
    """
    key: str
    value: Any
    size_bytes: int
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    creation_time: datetime = field(default_factory=datetime.now)
    ttl_seconds: Optional[int] = None
    version: int = 1
    compressed: bool = False

@dataclass
class CacheStats:
    """
    Cache performance statistics
    
    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        evictions: Number of entries evicted
        invalidations: Number of entries invalidated
        total_size_bytes: Current cache size
        hit_rate: hits / (hits + misses)
        avg_lookup_ms: Average lookup latency
        entries: Number of cached entries
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    total_size_bytes: int = 0
    avg_lookup_ms: float = 0.0
    entries: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class MultiTierCache:
    """
    Multi-tier LRU cache for hot embeddings
    
    L1: Hot cache (most frequently accessed)
    L2: Warm cache (recently accessed)
    L3: Cold cache (less frequently accessed)
    
    Each tier has different size/latency trade-offs
    """

    def __init__(
        self,
        l1_capacity_mb: float = 1024,      # 1GB L1
        l2_capacity_mb: float = 10240,     # 10GB L2
        l3_capacity_mb: float = 102400,    # 100GB L3
        enable_compression: bool = True
    ):
        self.l1_capacity_bytes = int(l1_capacity_mb * 1024 * 1024)
        self.l2_capacity_bytes = int(l2_capacity_mb * 1024 * 1024)
        self.l3_capacity_bytes = int(l3_capacity_mb * 1024 * 1024)
        self.enable_compression = enable_compression

        # L1 cache: OrderedDict for LRU behavior
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l1_size_bytes = 0

        # L2 cache: Larger, for warm entries
        self.l2_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l2_size_bytes = 0

        # L3 cache: Even larger, may use compression
        self.l3_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l3_size_bytes = 0

        # Statistics
        self.stats = {
            'l1': CacheStats(),
            'l2': CacheStats(),
            'l3': CacheStats(),
            'total': CacheStats()
        }

        # Access frequency tracking for promotion/demotion
        self.access_freq: Dict[str, int] = defaultdict(int)

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache, checking all tiers
        
        Args:
            key: Cache key
        
        Returns:
            value: Cached value or None if not found
        """
        start_time = time.time()

        # Check L1 cache
        if key in self.l1_cache:
            entry = self.l1_cache[key]
            entry.access_count += 1
            entry.last_access = datetime.now()
            self.access_freq[key] += 1

            # Move to end (most recently used)
            self.l1_cache.move_to_end(key)

            self.stats['l1'].hits += 1
            self.stats['total'].hits += 1
            self._update_latency(start_time, 'l1')

            return entry.value

        # Check L2 cache
        if key in self.l2_cache:
            entry = self.l2_cache[key]
            entry.access_count += 1
            entry.last_access = datetime.now()
            self.access_freq[key] += 1

            self.l2_cache.move_to_end(key)

            self.stats['l2'].hits += 1
            self.stats['total'].hits += 1

            # Promote to L1 if accessed frequently
            if entry.access_count >= 10:
                self._promote_to_l1(key, entry)

            self._update_latency(start_time, 'l2')
            return entry.value

        # Check L3 cache
        if key in self.l3_cache:
            entry = self.l3_cache[key]
            entry.access_count += 1
            entry.last_access = datetime.now()
            self.access_freq[key] += 1

            self.l3_cache.move_to_end(key)

            self.stats['l3'].hits += 1
            self.stats['total'].hits += 1

            # Decompress if needed
            value = entry.value
            if entry.compressed:
                value = self._decompress(value)

            # Promote to L2 if accessed frequently
            if entry.access_count >= 5:
                self._promote_to_l2(key, entry)

            self._update_latency(start_time, 'l3')
            return value

        # Cache miss
        self.stats['l1'].misses += 1
        self.stats['l2'].misses += 1
        self.stats['l3'].misses += 1
        self.stats['total'].misses += 1

        return None

    def put(
        self,
        key: str,
        value: Any,
        tier: str = 'l1'
    ) -> None:
        """
        Put value into cache at specified tier
        
        Args:
            key: Cache key
            value: Value to cache
            tier: Which cache tier ('l1', 'l2', 'l3')
        """
        # Calculate size
        size_bytes = self._estimate_size(value)

        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size_bytes
        )

        # Put into appropriate tier
        if tier == 'l1':
            self._put_l1(key, entry)
        elif tier == 'l2':
            self._put_l2(key, entry)
        elif tier == 'l3':
            # Compress for L3 if enabled
            if self.enable_compression and isinstance(value, np.ndarray):
                entry.value = self._compress(value)
                entry.compressed = True
                entry.size_bytes = self._estimate_size(entry.value)
            self._put_l3(key, entry)

    def _put_l1(self, key: str, entry: CacheEntry) -> None:
        """Put entry into L1 cache"""
        # Evict if needed
        while self.l1_size_bytes + entry.size_bytes > self.l1_capacity_bytes:
            if not self.l1_cache:
                break
            self._evict_l1()

        # Add entry
        self.l1_cache[key] = entry
        self.l1_size_bytes += entry.size_bytes
        self.stats['l1'].entries += 1
        self.stats['l1'].total_size_bytes = self.l1_size_bytes

    def _put_l2(self, key: str, entry: CacheEntry) -> None:
        """Put entry into L2 cache"""
        while self.l2_size_bytes + entry.size_bytes > self.l2_capacity_bytes:
            if not self.l2_cache:
                break
            self._evict_l2()

        self.l2_cache[key] = entry
        self.l2_size_bytes += entry.size_bytes
        self.stats['l2'].entries += 1
        self.stats['l2'].total_size_bytes = self.l2_size_bytes

    def _put_l3(self, key: str, entry: CacheEntry) -> None:
        """Put entry into L3 cache"""
        while self.l3_size_bytes + entry.size_bytes > self.l3_capacity_bytes:
            if not self.l3_cache:
                break
            self._evict_l3()

        self.l3_cache[key] = entry
        self.l3_size_bytes += entry.size_bytes
        self.stats['l3'].entries += 1
        self.stats['l3'].total_size_bytes = self.l3_size_bytes

    def _evict_l1(self) -> None:
        """Evict LRU entry from L1, demote to L2"""
        key, entry = self.l1_cache.popitem(last=False)
        self.l1_size_bytes -= entry.size_bytes
        self.stats['l1'].evictions += 1
        self.stats['l1'].entries -= 1

        # Demote to L2
        self._put_l2(key, entry)

    def _evict_l2(self) -> None:
        """Evict LRU entry from L2, demote to L3"""
        key, entry = self.l2_cache.popitem(last=False)
        self.l2_size_bytes -= entry.size_bytes
        self.stats['l2'].evictions += 1
        self.stats['l2'].entries -= 1

        # Demote to L3
        self._put_l3(key, entry)

    def _evict_l3(self) -> None:
        """Evict LRU entry from L3"""
        key, entry = self.l3_cache.popitem(last=False)
        self.l3_size_bytes -= entry.size_bytes
        self.stats['l3'].evictions += 1
        self.stats['l3'].entries -= 1

    def _promote_to_l1(self, key: str, entry: CacheEntry) -> None:
        """Promote entry from L2 to L1"""
        # Remove from L2
        if key in self.l2_cache:
            del self.l2_cache[key]
            self.l2_size_bytes -= entry.size_bytes
            self.stats['l2'].entries -= 1

            # Add to L1
            self._put_l1(key, entry)

    def _promote_to_l2(self, key: str, entry: CacheEntry) -> None:
        """Promote entry from L3 to L2"""
        # Decompress if needed
        if entry.compressed:
            entry.value = self._decompress(entry.value)
            entry.compressed = False
            entry.size_bytes = self._estimate_size(entry.value)

        # Remove from L3
        if key in self.l3_cache:
            del self.l3_cache[key]
            self.l3_size_bytes -= entry.size_bytes
            self.stats['l3'].entries -= 1

            # Add to L2
            self._put_l2(key, entry)

    def invalidate(self, key: str) -> None:
        """
        Invalidate cache entry across all tiers
        
        Args:
            key: Cache key to invalidate
        """
        # Remove from L1
        if key in self.l1_cache:
            entry = self.l1_cache.pop(key)
            self.l1_size_bytes -= entry.size_bytes
            self.stats['l1'].invalidations += 1
            self.stats['l1'].entries -= 1

        # Remove from L2
        if key in self.l2_cache:
            entry = self.l2_cache.pop(key)
            self.l2_size_bytes -= entry.size_bytes
            self.stats['l2'].invalidations += 1
            self.stats['l2'].entries -= 1

        # Remove from L3
        if key in self.l3_cache:
            entry = self.l3_cache.pop(key)
            self.l3_size_bytes -= entry.size_bytes
            self.stats['l3'].invalidations += 1
            self.stats['l3'].entries -= 1

        # Remove from frequency tracking
        if key in self.access_freq:
            del self.access_freq[key]

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, (list, tuple)):
            return len(value) * 8  # Rough estimate
        elif isinstance(value, bytes):
            return len(value)
        else:
            return 1024  # Default estimate

    def _compress(self, vector: np.ndarray) -> bytes:
        """
        Compress vector for storage
        
        Uses quantization for ~4× compression
        """
        # Simple quantization: float32 → uint8
        # Map [min, max] → [0, 255]
        vmin, vmax = vector.min(), vector.max()
        quantized = ((vector - vmin) / (vmax - vmin) * 255).astype(np.uint8)

        # Store min/max for decompression
        metadata = np.array([vmin, vmax], dtype=np.float32)

        return metadata.tobytes() + quantized.tobytes()

    def _decompress(self, compressed: bytes) -> np.ndarray:
        """Decompress vector"""
        # Extract metadata
        metadata = np.frombuffer(compressed[:8], dtype=np.float32)
        vmin, vmax = metadata[0], metadata[1]

        # Extract quantized values
        quantized = np.frombuffer(compressed[8:], dtype=np.uint8)

        # Dequantize
        vector = quantized.astype(np.float32) / 255.0 * (vmax - vmin) + vmin

        return vector

    def _update_latency(self, start_time: float, tier: str) -> None:
        """Update average latency statistics"""
        latency_ms = (time.time() - start_time) * 1000

        # Exponential moving average
        alpha = 0.1
        current_avg = self.stats[tier].avg_lookup_ms
        self.stats[tier].avg_lookup_ms = (
            alpha * latency_ms + (1 - alpha) * current_avg
        )

    def get_stats(self) -> Dict[str, CacheStats]:
        """Get cache statistics"""
        # Update total stats
        self.stats['total'].entries = (
            self.stats['l1'].entries +
            self.stats['l2'].entries +
            self.stats['l3'].entries
        )
        self.stats['total'].total_size_bytes = (
            self.l1_size_bytes +
            self.l2_size_bytes +
            self.l3_size_bytes
        )

        return self.stats

class QueryResultCache:
    """
    Cache complete query results
    
    Caches (query_vector, k, filters) → [(id, score), ...]
    """

    def __init__(self, capacity_mb: float = 1024):
        self.capacity_bytes = int(capacity_mb * 1024 * 1024)
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.size_bytes = 0
        self.stats = CacheStats()

    def get(
        self,
        query_vector: np.ndarray,
        k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Get cached query results
        
        Args:
            query_vector: Query embedding
            k: Number of results
            filters: Query filters
        
        Returns:
            results: Cached results or None if not found
        """
        key = self._make_key(query_vector, k, filters)

        if key in self.cache:
            entry = self.cache[key]
            entry.access_count += 1
            entry.last_access = datetime.now()

            # Move to end (most recently used)
            self.cache.move_to_end(key)

            self.stats.hits += 1
            return entry.value

        self.stats.misses += 1
        return None

    def put(
        self,
        query_vector: np.ndarray,
        k: int,
        filters: Optional[Dict[str, Any]],
        results: List[Tuple[str, float]]
    ) -> None:
        """
        Cache query results
        
        Args:
            query_vector: Query embedding
            k: Number of results
            filters: Query filters
            results: Query results to cache
        """
        key = self._make_key(query_vector, k, filters)

        # Estimate size (results + metadata)
        size_bytes = len(results) * (64 + 8)  # ID + score

        # Evict if needed
        while self.size_bytes + size_bytes > self.capacity_bytes:
            if not self.cache:
                break
            self._evict()

        # Create entry
        entry = CacheEntry(
            key=key,
            value=results,
            size_bytes=size_bytes
        )

        self.cache[key] = entry
        self.size_bytes += size_bytes
        self.stats.entries += 1

    def _make_key(
        self,
        query_vector: np.ndarray,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> str:
        """
        Create cache key from query parameters
        
        Hash query vector to fixed-length key
        """
        # Hash query vector
        vector_hash = hashlib.sha256(query_vector.tobytes()).hexdigest()[:16]

        # Hash filters
        if filters:
            filter_str = json.dumps(filters, sort_keys=True)
            filter_hash = hashlib.sha256(filter_str.encode()).hexdigest()[:8]
        else:
            filter_hash = "nofilter"

        return f"{vector_hash}_{k}_{filter_hash}"

    def _evict(self) -> None:
        """Evict LRU entry"""
        key, entry = self.cache.popitem(last=False)
        self.size_bytes -= entry.size_bytes
        self.stats.evictions += 1
        self.stats.entries -= 1

class AdaptivePrefetcher:
    """
    Adaptive prefetching based on query patterns
    
    Learns which embeddings likely to be accessed next
    """

    def __init__(self, cache: MultiTierCache):
        self.cache = cache
        self.query_history: List[str] = []
        self.transition_probs: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.history_size = 1000

    def record_access(self, key: str) -> None:
        """
        Record access pattern for learning
        
        Args:
            key: Accessed cache key
        """
        if self.query_history:
            # Update transition probabilities
            prev_key = self.query_history[-1]
            self.transition_probs[prev_key][key] += 1

        # Add to history
        self.query_history.append(key)

        # Trim history
        if len(self.query_history) > self.history_size:
            self.query_history = self.query_history[-self.history_size:]

    def prefetch(self, current_key: str, n: int = 10) -> List[str]:
        """
        Predict likely next accesses and prefetch
        
        Args:
            current_key: Current access
            n: Number of entries to prefetch
        
        Returns:
            prefetch_keys: Predicted next accesses
        """
        if current_key not in self.transition_probs:
            return []

        # Get transition probabilities
        transitions = self.transition_probs[current_key]

        # Normalize
        total = sum(transitions.values())
        if total == 0:
            return []

        normalized = {
            k: v / total
            for k, v in transitions.items()
        }

        # Sort by probability
        sorted_keys = sorted(
            normalized.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top-n
        return [k for k, _ in sorted_keys[:n]]
