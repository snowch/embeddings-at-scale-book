# Code from Chapter 12
# Book: Embeddings at Scale

"""
Streaming Embedding Pipeline

Architecture:
1. Event Stream: Kafka, Kinesis, Pub/Sub
2. Stream Processor: Flink, Spark Streaming, custom consumers
3. Embedding Generator: Real-time model inference
4. Vector Index: Incremental updates (HNSW, Faiss)

Latency budget:
- Event ingestion: 10-100ms
- Feature extraction: 10-50ms
- Embedding generation: 50-200ms
- Index update: 10-50ms
- Total: 100-400ms (< 1 second)
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


@dataclass
class StreamEvent:
    """
    Event in embedding stream

    Attributes:
        event_id: Unique identifier
        event_type: Type of event (create, update, delete)
        entity_id: ID of entity to embed
        data: Entity data
        timestamp: Event timestamp
    """
    event_id: str
    event_type: str  # 'create', 'update', 'delete'
    entity_id: str
    data: Dict
    timestamp: datetime = field(default_factory=datetime.now)

class StreamingEmbeddingPipeline:
    """
    Real-time embedding pipeline with micro-batching

    Architecture:
    - Consume events from stream (Kafka topic, Kinesis stream)
    - Micro-batch events (10-100 items, 100-1000ms window)
    - Generate embeddings (batched inference on GPU)
    - Update vector index (incremental HNSW update)
    - Emit updated embeddings downstream

    Guarantees:
    - At-least-once processing (events may be reprocessed)
    - Eventual consistency (index eventually reflects all events)
    - Low latency (p99 < 1 second)

    Fault tolerance:
    - Checkpointing to recover from failures
    - Exactly-once semantics via idempotent updates
    - Dead letter queue for failed events
    """

    def __init__(
        self,
        embedding_model,
        vector_index,
        batch_window_ms: int = 500,
        max_batch_size: int = 100,
        enable_checkpointing: bool = True
    ):
        """
        Args:
            embedding_model: Model for generating embeddings
            vector_index: Vector index for storing embeddings (HNSW, Faiss)
            batch_window_ms: Time window for micro-batching (milliseconds)
            max_batch_size: Maximum events per micro-batch
            enable_checkpointing: Enable fault-tolerant checkpointing
        """
        self.embedding_model = embedding_model
        self.vector_index = vector_index
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size
        self.enable_checkpointing = enable_checkpointing

        # Event queue for micro-batching
        self.event_queue = queue.Queue()

        # Processing thread
        self.processing_thread = None
        self.running = False

        # Metrics
        self.events_processed = 0
        self.batches_processed = 0
        self.total_latency_ms = 0
        self.errors = 0

        # Checkpointing
        self.last_checkpoint_offset = 0

        print("Initialized Streaming Embedding Pipeline")
        print(f"  Batch window: {batch_window_ms}ms")
        print(f"  Max batch size: {max_batch_size}")

    def start(self):
        """Start background processing thread"""
        if self.running:
            print("Pipeline already running")
            return

        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        print("✓ Started streaming pipeline")

    def stop(self):
        """Stop background processing thread"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)

        print("✓ Stopped streaming pipeline")

    def ingest_event(self, event: StreamEvent):
        """
        Ingest event into stream

        Args:
            event: Stream event to process
        """
        self.event_queue.put(event)

    def _processing_loop(self):
        """
        Background thread: Consume events and process in micro-batches

        Loop:
        1. Accumulate events for batch_window_ms
        2. Process batch (extract features, generate embeddings, update index)
        3. Checkpoint offset
        4. Repeat
        """
        while self.running:
            batch = self._accumulate_batch()

            if batch:
                self._process_batch(batch)

    def _accumulate_batch(self) -> List[StreamEvent]:
        """
        Accumulate events into micro-batch

        Strategy:
        - Wait up to batch_window_ms for events
        - Return batch when either:
          - batch_window_ms elapsed
          - max_batch_size reached

        Returns:
            List of events in batch
        """
        batch = []
        batch_start = time.time()

        while len(batch) < self.max_batch_size:
            # Calculate remaining wait time
            elapsed_ms = (time.time() - batch_start) * 1000
            remaining_ms = max(0, self.batch_window_ms - elapsed_ms)

            # If window elapsed and we have events, return batch
            if remaining_ms == 0 and len(batch) > 0:
                break

            # Wait for next event
            try:
                timeout = remaining_ms / 1000
                event = self.event_queue.get(timeout=max(0.001, timeout))
                batch.append(event)
            except queue.Empty:
                # Timeout - return current batch if non-empty
                if len(batch) > 0:
                    break
                continue

        return batch

    def _process_batch(self, batch: List[StreamEvent]):
        """
        Process micro-batch of events

        Steps:
        1. Extract features from events
        2. Generate embeddings (batched inference)
        3. Update vector index
        4. Checkpoint progress

        Args:
            batch: Events to process
        """
        batch_start = time.time()

        try:
            # 1. Extract features
            features_list = []
            entity_ids = []

            for event in batch:
                features = self._extract_features(event)
                if features is not None:
                    features_list.append(features)
                    entity_ids.append(event.entity_id)

            if not features_list:
                return

            # 2. Generate embeddings (batched)
            embeddings = self._generate_embeddings_batch(features_list)

            # 3. Update vector index
            self._update_index(entity_ids, embeddings, batch)

            # 4. Checkpoint
            if self.enable_checkpointing:
                self._checkpoint(batch[-1].event_id)

            # Metrics
            batch_latency_ms = (time.time() - batch_start) * 1000
            self.events_processed += len(batch)
            self.batches_processed += 1
            self.total_latency_ms += batch_latency_ms

            # Log progress
            avg_latency = self.total_latency_ms / self.batches_processed
            throughput = self.events_processed / (self.total_latency_ms / 1000)

            if self.batches_processed % 10 == 0:
                print(f"Processed {self.events_processed:,} events in {self.batches_processed} batches")
                print(f"  Avg latency: {avg_latency:.1f}ms")
                print(f"  Throughput: {throughput:.0f} events/sec")

        except Exception as e:
            print(f"⚠️  Batch processing failed: {e}")
            self.errors += 1

            # Send failed events to dead letter queue
            self._send_to_dlq(batch, error=str(e))

    def _extract_features(self, event: StreamEvent) -> Optional[np.ndarray]:
        """
        Extract features from event

        Args:
            event: Stream event

        Returns:
            Feature vector or None if extraction fails
        """
        try:
            # Extract text features
            text_parts = []
            for field in ['title', 'description', 'content']:
                if field in event.data:
                    text_parts.append(str(event.data[field]))

            if not text_parts:
                return None

            " ".join(text_parts)

            # In production: Use proper feature extraction
            # For now: Return dummy features
            return np.random.randn(512).astype(np.float32)

        except Exception as e:
            print(f"⚠️  Feature extraction failed for event {event.event_id}: {e}")
            return None

    def _generate_embeddings_batch(
        self,
        features_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        Generate embeddings for batch (GPU-accelerated)

        Args:
            features_list: List of feature vectors

        Returns:
            Batch of embeddings (N, embedding_dim)
        """
        # Stack features into batch
        np.stack(features_list)

        # Generate embeddings
        # In production: Use actual embedding model
        # For now: Return random embeddings
        embeddings = np.random.randn(len(features_list), 256).astype(np.float32)

        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings

    def _update_index(
        self,
        entity_ids: List[str],
        embeddings: np.ndarray,
        events: List[StreamEvent]
    ):
        """
        Update vector index with new embeddings

        Operations:
        - CREATE: Add new vector to index
        - UPDATE: Replace existing vector
        - DELETE: Remove vector from index

        Args:
            entity_ids: Entity identifiers
            embeddings: Embedding vectors
            events: Original events (for event_type)
        """
        for entity_id, embedding, event in zip(entity_ids, embeddings, events):
            if event.event_type == 'create':
                # Add to index
                self.vector_index.add(entity_id, embedding)

            elif event.event_type == 'update':
                # Replace in index (delete + add)
                self.vector_index.delete(entity_id)
                self.vector_index.add(entity_id, embedding)

            elif event.event_type == 'delete':
                # Remove from index
                self.vector_index.delete(entity_id)

    def _checkpoint(self, last_event_id: str):
        """
        Checkpoint progress for fault tolerance

        In production: Write to persistent storage (database, S3)

        Args:
            last_event_id: Last processed event ID
        """
        self.last_checkpoint_offset = last_event_id
        # In production: Persist to database

    def _send_to_dlq(self, batch: List[StreamEvent], error: str):
        """
        Send failed events to dead letter queue

        Args:
            batch: Failed events
            error: Error message
        """
        print(f"⚠️  Sending {len(batch)} events to DLQ: {error}")
        # In production: Write to Kafka DLQ topic, SQS DLQ, etc.

    def get_metrics(self) -> Dict:
        """Get pipeline metrics"""
        return {
            'events_processed': self.events_processed,
            'batches_processed': self.batches_processed,
            'avg_latency_ms': self.total_latency_ms / max(1, self.batches_processed),
            'throughput_eps': self.events_processed / max(1, self.total_latency_ms / 1000),
            'errors': self.errors,
            'error_rate': self.errors / max(1, self.batches_processed)
        }

class MockVectorIndex:
    """Mock vector index for demonstration"""
    def __init__(self):
        self.vectors = {}

    def add(self, entity_id: str, embedding: np.ndarray):
        self.vectors[entity_id] = embedding

    def delete(self, entity_id: str):
        if entity_id in self.vectors:
            del self.vectors[entity_id]

    def search(self, query: np.ndarray, k: int = 10):
        # Mock search
        return list(self.vectors.keys())[:k]

# Example: Real-time news article embeddings
def streaming_news_example():
    """
    Streaming pipeline for news article embeddings

    Scenario:
    - News articles published throughout the day
    - Need embeddings within 1 second for recommendations
    - 1000 articles/hour = 0.3 articles/second

    Architecture:
    - Kafka topic: news_articles
    - Streaming pipeline: Consume, embed, index
    - Vector index: HNSW for fast search
    """

    # Mock embedding model
    class MockEmbeddingModel:
        def embed(self, texts):
            return np.random.randn(len(texts), 256).astype(np.float32)

    # Initialize components
    embedding_model = MockEmbeddingModel()
    vector_index = MockVectorIndex()

    # Initialize streaming pipeline
    pipeline = StreamingEmbeddingPipeline(
        embedding_model=embedding_model,
        vector_index=vector_index,
        batch_window_ms=500,
        max_batch_size=50
    )

    # Start pipeline
    pipeline.start()

    # Simulate incoming events
    print("Simulating news article stream...")
    for i in range(100):
        event = StreamEvent(
            event_id=f"event_{i}",
            event_type='create',
            entity_id=f"article_{i}",
            data={
                'title': f'Breaking News {i}',
                'content': f'This is article content for story {i}',
                'category': ['Politics', 'Sports', 'Tech'][i % 3]
            }
        )

        pipeline.ingest_event(event)

        # Simulate event arrival rate
        time.sleep(0.01)  # 100 events/sec

    # Wait for processing to complete
    time.sleep(2)

    # Stop pipeline
    pipeline.stop()

    # Print metrics
    metrics = pipeline.get_metrics()
    print("\n✓ Streaming pipeline metrics:")
    print(f"  Events processed: {metrics['events_processed']}")
    print(f"  Batches processed: {metrics['batches_processed']}")
    print(f"  Avg latency: {metrics['avg_latency_ms']:.1f}ms")
    print(f"  Throughput: {metrics['throughput_eps']:.0f} events/sec")
    print(f"  Errors: {metrics['errors']}")

# Uncomment to run:
# streaming_news_example()
