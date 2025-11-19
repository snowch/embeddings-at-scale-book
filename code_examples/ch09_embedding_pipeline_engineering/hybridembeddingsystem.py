# Code from Chapter 09
# Book: Embeddings at Scale

import torch
import torch.nn as nn
from typing import Optional, List, Dict
import asyncio
from datetime import datetime, timedelta
import numpy as np
from collections import deque

class HybridEmbeddingSystem:
    """
    Hybrid system combining batch and real-time embedding generation

    Architecture:
    - Batch layer: Pre-compute embeddings for stable entities (products, documents)
    - Real-time layer: Generate embeddings for dynamic entities (user queries, sessions)
    - Caching layer: Cache frequently accessed real-time embeddings
    - Routing logic: Determine batch vs. real-time for each request

    Decision matrix:
    | Entity Type | Update Frequency | Access Pattern | Strategy |
    |-------------|-----------------|----------------|----------|
    | Products    | Daily           | Random         | Batch    |
    | Documents   | Hourly          | Zipfian        | Batch + Cache |
    | User Queries| Per request     | Unique         | Real-time |
    | User Profiles| Weekly         | Personalized   | Batch + Real-time |
    """

    def __init__(
        self,
        model_registry,
        model_id: str,
        cache_size: int = 100000,
        real_time_threshold_ms: int = 50
    ):
        """
        Args:
            model_registry: Registry to load embedding model
            model_id: Model to use
            cache_size: Number of embeddings to cache
            real_time_threshold_ms: Max latency for real-time generation
        """
        self.model, self.metadata = model_registry.load_model(model_id, 'cuda')
        self.model.eval()

        # Batch embedding storage (in production: vector database)
        self.batch_embeddings: Dict[str, np.ndarray] = {}
        self.batch_embedding_timestamps: Dict[str, datetime] = {}

        # Real-time embedding cache (LRU)
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_access_times = deque(maxlen=cache_size)

        # Metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_lookups = 0
        self.realtime_generations = 0

    def get_embedding(
        self,
        entity_id: str,
        entity_type: str,
        features: Optional[torch.Tensor] = None,
        max_staleness: Optional[timedelta] = None
    ) -> np.ndarray:
        """
        Get embedding using appropriate strategy

        Args:
            entity_id: Unique identifier for entity
            entity_type: 'product', 'document', 'query', 'user'
            features: Features for real-time generation
            max_staleness: Maximum acceptable age for batch embeddings

        Returns:
            embedding: Vector representation
        """
        # Route based on entity type
        if entity_type in ['query', 'session']:
            # Always generate real-time for transient entities
            return self._generate_realtime(entity_id, features)

        elif entity_type in ['product', 'document']:
            # Prefer batch, fallback to real-time
            batch_emb = self._lookup_batch(entity_id, max_staleness)
            if batch_emb is not None:
                return batch_emb
            else:
                return self._generate_realtime(entity_id, features)

        elif entity_type == 'user':
            # Hybrid: batch base + real-time personalization
            base_emb = self._lookup_batch(entity_id, max_staleness)
            if base_emb is None:
                base_emb = self._generate_realtime(entity_id, features)

            # Personalize based on current session
            personalized_emb = self._personalize_embedding(base_emb, features)
            return personalized_emb

        else:
            raise ValueError(f"Unknown entity type: {entity_type}")

    def _lookup_batch(
        self,
        entity_id: str,
        max_staleness: Optional[timedelta]
    ) -> Optional[np.ndarray]:
        """
        Lookup pre-computed batch embedding

        Returns None if:
        - Embedding doesn't exist
        - Embedding is too stale
        """
        if entity_id not in self.batch_embeddings:
            return None

        # Check staleness
        if max_staleness is not None:
            embedding_time = self.batch_embedding_timestamps[entity_id]
            age = datetime.now() - embedding_time
            if age > max_staleness:
                return None  # Too stale, need fresh embedding

        self.batch_lookups += 1
        return self.batch_embeddings[entity_id]

    def _generate_realtime(
        self,
        entity_id: str,
        features: torch.Tensor
    ) -> np.ndarray:
        """
        Generate embedding in real-time

        With caching for frequently accessed entities
        """
        # Check cache first
        if entity_id in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[entity_id]

        self.cache_misses += 1
        self.realtime_generations += 1

        # Generate embedding
        start_time = datetime.now()

        with torch.no_grad():
            features = features.to('cuda')
            embedding = self.model(features).cpu().numpy()

        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Cache if latency-sensitive
        if latency_ms > 10:  # Cache expensive computations
            self._add_to_cache(entity_id, embedding)

        return embedding

    def _personalize_embedding(
        self,
        base_embedding: np.ndarray,
        session_features: torch.Tensor
    ) -> np.ndarray:
        """
        Personalize base embedding with real-time session context

        Techniques:
        - Additive: base + alpha * session_vector
        - Attention: weighted combination based on context
        - Learned: small neural network for personalization
        """
        # Simple additive personalization
        with torch.no_grad():
            session_features = session_features.to('cuda')
            session_vector = self.model(session_features).cpu().numpy()

        # Weighted combination (80% base, 20% session)
        personalized = 0.8 * base_embedding + 0.2 * session_vector

        return personalized

    def _add_to_cache(self, entity_id: str, embedding: np.ndarray):
        """
        Add to LRU cache

        Evict oldest if cache full
        """
        if len(self.embedding_cache) >= len(self.cache_access_times):
            # Evict oldest
            oldest_id = self.cache_access_times.popleft()
            if oldest_id in self.embedding_cache:
                del self.embedding_cache[oldest_id]

        self.embedding_cache[entity_id] = embedding
        self.cache_access_times.append(entity_id)

    def batch_update(
        self,
        entity_ids: List[str],
        embeddings: List[np.ndarray]
    ):
        """
        Update batch embeddings (called by batch processing job)

        Args:
            entity_ids: IDs of entities
            embeddings: Pre-computed embeddings
        """
        timestamp = datetime.now()

        for entity_id, embedding in zip(entity_ids, embeddings):
            self.batch_embeddings[entity_id] = embedding
            self.batch_embedding_timestamps[entity_id] = timestamp

        print(f"✓ Updated {len(entity_ids)} batch embeddings")

        # Invalidate cache for updated entities
        for entity_id in entity_ids:
            if entity_id in self.embedding_cache:
                del self.embedding_cache[entity_id]

    def get_metrics(self) -> Dict:
        """
        Get system metrics for monitoring

        Key metrics:
        - Cache hit rate
        - Batch vs. real-time ratio
        - Average latency per strategy
        """
        total_requests = self.batch_lookups + self.realtime_generations
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0

        return {
            'total_requests': total_requests,
            'batch_lookups': self.batch_lookups,
            'batch_ratio': self.batch_lookups / total_requests if total_requests > 0 else 0,
            'realtime_generations': self.realtime_generations,
            'realtime_ratio': self.realtime_generations / total_requests if total_requests > 0 else 0,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.embedding_cache)
        }

class StreamingEmbeddingPipeline:
    """
    Streaming pipeline for near-real-time embedding updates

    Architecture:
    - Consume events from stream (Kafka, Kinesis, Pub/Sub)
    - Micro-batch embeddings (100-1000 items every 10-60 seconds)
    - Update vector index incrementally
    - Balance freshness vs. throughput

    Use cases:
    - News articles (publish → embed → searchable in <1 minute)
    - E-commerce products (inventory update → embed → discoverable)
    - Social media posts (post → embed → recommendable)
    - Log events (event → embed → queryable)
    """

    def __init__(
        self,
        model,
        batch_window_seconds: int = 30,
        max_batch_size: int = 500
    ):
        """
        Args:
            model: Embedding model
            batch_window_seconds: Time window for micro-batching
            max_batch_size: Max items per micro-batch
        """
        self.model = model
        self.model.eval()

        self.batch_window = timedelta(seconds=batch_window_seconds)
        self.max_batch_size = max_batch_size

        # Accumulation buffer
        self.buffer: List[Dict] = []
        self.last_flush = datetime.now()

        # Metrics
        self.items_processed = 0
        self.batches_processed = 0

    async def process_stream(self, event_stream):
        """
        Process streaming events in micro-batches

        Args:
            event_stream: Async iterator of events
        """
        async for event in event_stream:
            # Add to buffer
            self.buffer.append(event)

            # Flush if buffer full or time window expired
            should_flush = (
                len(self.buffer) >= self.max_batch_size or
                (datetime.now() - self.last_flush) >= self.batch_window
            )

            if should_flush:
                await self._flush_buffer()

    async def _flush_buffer(self):
        """
        Process accumulated buffer as micro-batch

        Benefits of micro-batching:
        - Amortize model overhead
        - Better GPU utilization
        - Reduce index update frequency
        """
        if not self.buffer:
            return

        print(f"Flushing buffer with {len(self.buffer)} items...")

        # Extract features from events
        features = torch.stack([self._extract_features(event) for event in self.buffer])

        # Generate embeddings (batched)
        with torch.no_grad():
            embeddings = self.model(features.to('cuda')).cpu().numpy()

        # Update vector index
        entity_ids = [event['entity_id'] for event in self.buffer]
        await self._update_index(entity_ids, embeddings)

        # Metrics
        self.items_processed += len(self.buffer)
        self.batches_processed += 1

        # Clear buffer
        self.buffer = []
        self.last_flush = datetime.now()

        print(f"✓ Processed {len(entity_ids)} embeddings (total: {self.items_processed})")

    def _extract_features(self, event: Dict) -> torch.Tensor:
        """Extract features from streaming event"""
        # Implementation depends on event schema
        return torch.randn(100)  # Placeholder

    async def _update_index(self, entity_ids: List[str], embeddings: np.ndarray):
        """
        Update vector index with new embeddings

        In production: Call vector database API (Pinecone, Weaviate, etc.)
        """
        # Asynchronous index update
        await asyncio.sleep(0.01)  # Simulate async update
        pass

# Example: E-commerce product embeddings
def ecommerce_hybrid_example():
    """
    E-commerce scenario with hybrid approach

    Products (batch):
    - 10M products
    - Update daily (new products, price changes, inventory)
    - Batch process overnight

    User Queries (real-time):
    - 100M queries/day
    - Generate on-demand
    - Cache popular queries

    User Profiles (hybrid):
    - 50M users
    - Base profile updated weekly (batch)
    - Personalized with current session (real-time)
    """
    from datetime import timedelta

    # Initialize hybrid system
    registry = EmbeddingModelRegistry()
    # Assume model is registered
    hybrid_system = HybridEmbeddingSystem(
        model_registry=registry,
        model_id="ecommerce-v1.0.0",
        cache_size=100000
    )

    # Scenario 1: Product lookup (batch)
    product_embedding = hybrid_system.get_embedding(
        entity_id="product_12345",
        entity_type="product",
        max_staleness=timedelta(days=1)
    )
    print(f"Product embedding: {product_embedding.shape}")

    # Scenario 2: User query (real-time)
    query_features = torch.randn(1, 100)
    query_embedding = hybrid_system.get_embedding(
        entity_id="query_unique_789",
        entity_type="query",
        features=query_features
    )
    print(f"Query embedding: {query_embedding.shape}")

    # Scenario 3: Personalized user (hybrid)
    user_session_features = torch.randn(1, 100)
    user_embedding = hybrid_system.get_embedding(
        entity_id="user_456",
        entity_type="user",
        features=user_session_features,
        max_staleness=timedelta(days=7)
    )
    print(f"User embedding: {user_embedding.shape}")

    # Metrics
    metrics = hybrid_system.get_metrics()
    print(f"\nSystem metrics:")
    print(f"  Batch ratio: {metrics['batch_ratio']:.2%}")
    print(f"  Real-time ratio: {metrics['realtime_ratio']:.2%}")
    print(f"  Cache hit rate: {metrics['cache_hit_rate']:.2%}")

# Uncomment to run:
# ecommerce_hybrid_example()
