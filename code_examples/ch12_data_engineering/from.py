# Code from Chapter 12
# Book: Embeddings at Scale

"""
Embedding-Aware ETL Pipeline

Key differences from traditional ETL:
1. Preserve semantic relationships during transformation
2. Generate features that capture similarity, not just attributes
3. Handle multimodal data (text, images, structured)
4. Maintain data lineage for debugging embedding quality
5. Support incremental updates for continuous training
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import hashlib

@dataclass
class DataRecord:
    """
    Raw data record from source system

    Attributes:
        record_id: Unique identifier
        data: Raw data payload
        source: Source system identifier
        timestamp: When record was created
        metadata: Additional context
    """
    record_id: str
    data: Dict[str, Any]
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class EmbeddingFeatures:
    """
    Transformed features ready for embedding generation

    Attributes:
        record_id: Links back to original record
        text_features: Text content for embedding
        structured_features: Numerical/categorical features
        context_features: Additional context (user, session, etc.)
        labels: Labels for supervised learning (optional)
        data_hash: Hash for duplicate detection
    """
    record_id: str
    text_features: Optional[str] = None
    structured_features: Optional[Dict[str, float]] = None
    context_features: Optional[Dict[str, Any]] = None
    labels: Optional[List[str]] = None
    data_hash: Optional[str] = None

class EmbeddingETLPipeline:
    """
    ETL pipeline for embedding generation at scale

    Architecture:
    1. Extract: Pull data from multiple sources (databases, APIs, files)
    2. Transform: Feature engineering + quality validation
    3. Load: Write to training system (cloud storage, feature store)

    Design principles:
    - Idempotent: Re-running produces same results
    - Incremental: Process only new/changed records
    - Traceable: Full lineage from raw data to embeddings
    - Scalable: Handles billions of records via partitioning

    Production considerations:
    - Checkpointing for fault tolerance
    - Monitoring for data drift
    - Schema validation at each stage
    - Resource optimization (memory, compute)
    """

    def __init__(
        self,
        output_path: str,
        checkpoint_path: Optional[str] = None,
        batch_size: int = 10000
    ):
        """
        Args:
            output_path: Where to write transformed features
            checkpoint_path: Where to save progress checkpoints
            batch_size: Records per batch
        """
        self.output_path = Path(output_path)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.batch_size = batch_size

        # Statistics
        self.records_processed = 0
        self.records_skipped = 0
        self.records_failed = 0

        # State
        self.last_checkpoint_id: Optional[str] = None
        self._load_checkpoint()

        print(f"Initialized Embedding ETL Pipeline")
        print(f"  Output: {output_path}")
        print(f"  Batch size: {batch_size:,}")
        if self.last_checkpoint_id:
            print(f"  Resuming from checkpoint: {self.last_checkpoint_id}")

    def extract(
        self,
        source_iterator,
        start_time: Optional[datetime] = None
    ) -> List[DataRecord]:
        """
        Extract data from source system

        Supports incremental extraction:
        - Read only records after last checkpoint
        - Handle pagination for large datasets
        - Implement retry logic for transient failures

        Args:
            source_iterator: Iterator over source records
            start_time: Only extract records after this time (incremental)

        Returns:
            List of DataRecords
        """
        records = []

        for raw_record in source_iterator:
            # Skip records before checkpoint (incremental processing)
            if start_time and raw_record.get('timestamp') < start_time:
                continue

            # Parse into DataRecord
            try:
                record = DataRecord(
                    record_id=raw_record['id'],
                    data=raw_record.get('data', {}),
                    source=raw_record.get('source', 'unknown'),
                    timestamp=raw_record.get('timestamp', datetime.now()),
                    metadata=raw_record.get('metadata', {})
                )
                records.append(record)

            except Exception as e:
                print(f"⚠️  Failed to parse record: {e}")
                self.records_failed += 1
                continue

        print(f"Extracted {len(records):,} records from source")
        return records

    def transform(
        self,
        records: List[DataRecord]
    ) -> List[EmbeddingFeatures]:
        """
        Transform raw records into embedding features

        Key transformations:
        1. Text normalization (preserve semantic meaning)
        2. Feature extraction (capture relationships)
        3. Context enrichment (add user/session context)
        4. Deduplication (hash-based)
        5. Quality validation

        Args:
            records: Raw data records

        Returns:
            List of EmbeddingFeatures
        """
        features_list = []
        seen_hashes = set()

        for record in records:
            try:
                # Extract features
                features = self._extract_features(record)

                # Generate hash for deduplication
                features.data_hash = self._compute_hash(features)

                # Skip duplicates
                if features.data_hash in seen_hashes:
                    self.records_skipped += 1
                    continue
                seen_hashes.add(features.data_hash)

                # Validate quality
                if not self._validate_quality(features):
                    self.records_skipped += 1
                    continue

                features_list.append(features)
                self.records_processed += 1

            except Exception as e:
                print(f"⚠️  Transform failed for record {record.record_id}: {e}")
                self.records_failed += 1
                continue

        print(f"Transformed {len(features_list):,} records")
        print(f"  Skipped: {self.records_skipped:,} (duplicates + quality)")
        print(f"  Failed: {self.records_failed:,}")

        return features_list

    def _extract_features(
        self,
        record: DataRecord
    ) -> EmbeddingFeatures:
        """
        Extract features from raw record

        Feature engineering strategies:
        - Text: Combine title, description, tags into single string
        - Structured: Normalize numerical features, encode categoricals
        - Context: Add temporal, user, session information

        Args:
            record: Raw data record

        Returns:
            EmbeddingFeatures
        """
        # Text features: Combine multiple text fields
        text_parts = []
        for field in ['title', 'description', 'content', 'tags']:
            if field in record.data and record.data[field]:
                text_parts.append(str(record.data[field]))

        text_features = " ".join(text_parts) if text_parts else None

        # Structured features: Extract numerical/categorical
        structured_features = {}
        for key, value in record.data.items():
            if isinstance(value, (int, float)):
                structured_features[key] = float(value)
            elif isinstance(value, bool):
                structured_features[key] = float(value)

        # Context features: Metadata that provides additional signal
        context_features = {
            'source': record.source,
            'timestamp': record.timestamp.isoformat(),
            **record.metadata
        }

        # Labels: Extract from metadata if available
        labels = record.metadata.get('labels', None)

        return EmbeddingFeatures(
            record_id=record.record_id,
            text_features=text_features,
            structured_features=structured_features if structured_features else None,
            context_features=context_features,
            labels=labels
        )

    def _compute_hash(self, features: EmbeddingFeatures) -> str:
        """
        Compute hash for deduplication

        Hash content (not metadata):
        - Text features
        - Structured features

        Exclude:
        - record_id (same content, different IDs should dedupe)
        - timestamps (same content, different times should dedupe)

        Args:
            features: Features to hash

        Returns:
            Hash string
        """
        hash_input = {
            'text': features.text_features,
            'structured': features.structured_features
        }

        hash_str = json.dumps(hash_input, sort_keys=True)
        return hashlib.md5(hash_str.encode()).hexdigest()

    def _validate_quality(self, features: EmbeddingFeatures) -> bool:
        """
        Validate feature quality

        Quality checks:
        - Has at least one feature type (text or structured)
        - Text is not empty or too short
        - Structured features are valid numbers
        - No extreme outliers

        Args:
            features: Features to validate

        Returns:
            True if valid, False otherwise
        """
        # Must have at least text or structured features
        if not features.text_features and not features.structured_features:
            return False

        # Text validation
        if features.text_features:
            # Must have minimum length
            if len(features.text_features.strip()) < 10:
                return False

            # Must not be too long (likely corrupted)
            if len(features.text_features) > 100000:
                return False

        # Structured features validation
        if features.structured_features:
            for key, value in features.structured_features.items():
                # Must be valid number
                if not np.isfinite(value):
                    return False

                # Check for extreme outliers (likely errors)
                if abs(value) > 1e10:
                    return False

        return True

    def load(
        self,
        features_list: List[EmbeddingFeatures],
        output_format: str = 'jsonl'
    ):
        """
        Load features to output destination

        Output formats:
        - jsonl: JSON Lines (one record per line)
        - parquet: Columnar format (efficient for large datasets)
        - tfrecord: TensorFlow format (for TF-based training)

        Partitioning strategy:
        - Partition by date for time-based incremental processing
        - Partition by hash for parallel processing

        Args:
            features_list: Features to write
            output_format: Output format ('jsonl', 'parquet', 'tfrecord')
        """
        print(f"Loading {len(features_list):,} records to {self.output_path}")

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

        if output_format == 'jsonl':
            self._load_jsonl(features_list)
        elif output_format == 'parquet':
            self._load_parquet(features_list)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        print(f"✓ Loaded {len(features_list):,} records")

    def _load_jsonl(self, features_list: List[EmbeddingFeatures]):
        """Write features as JSON Lines"""
        output_file = self.output_path / f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        with open(output_file, 'w') as f:
            for features in features_list:
                record = {
                    'record_id': features.record_id,
                    'text_features': features.text_features,
                    'structured_features': features.structured_features,
                    'context_features': features.context_features,
                    'labels': features.labels,
                    'data_hash': features.data_hash
                }
                f.write(json.dumps(record) + '\n')

        print(f"  Wrote {output_file}")

    def _load_parquet(self, features_list: List[EmbeddingFeatures]):
        """Write features as Parquet"""
        # Convert to DataFrame
        records = []
        for features in features_list:
            records.append({
                'record_id': features.record_id,
                'text_features': features.text_features,
                'structured_features': json.dumps(features.structured_features),
                'context_features': json.dumps(features.context_features),
                'labels': json.dumps(features.labels),
                'data_hash': features.data_hash
            })

        df = pd.DataFrame(records)

        output_file = self.output_path / f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        df.to_parquet(output_file, index=False)

        print(f"  Wrote {output_file}")

    def run(
        self,
        source_iterator,
        incremental: bool = True,
        output_format: str = 'jsonl'
    ):
        """
        Run complete ETL pipeline

        Args:
            source_iterator: Iterator over source records
            incremental: Only process new records since last checkpoint
            output_format: Output format
        """
        print("Starting ETL pipeline...")
        start_time = datetime.now()

        # Extract
        extract_start = None
        if incremental and self.last_checkpoint_id:
            # In production: Query checkpoint to get timestamp
            extract_start = datetime.now() - timedelta(days=1)  # Placeholder

        records = self.extract(source_iterator, start_time=extract_start)

        if not records:
            print("No new records to process")
            return

        # Transform
        features_list = self.transform(records)

        if not features_list:
            print("No valid features after transformation")
            return

        # Load
        self.load(features_list, output_format=output_format)

        # Checkpoint
        if self.checkpoint_path:
            self._save_checkpoint(records[-1].record_id if records else None)

        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"\n✓ ETL pipeline complete")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Throughput: {self.records_processed / elapsed:.0f} records/sec")
        print(f"  Success rate: {self.records_processed / (self.records_processed + self.records_failed):.2%}")

    def _save_checkpoint(self, last_record_id: Optional[str]):
        """Save checkpoint for incremental processing"""
        if not self.checkpoint_path or not last_record_id:
            return

        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'last_record_id': last_record_id,
            'timestamp': datetime.now().isoformat(),
            'records_processed': self.records_processed,
            'records_skipped': self.records_skipped,
            'records_failed': self.records_failed
        }

        checkpoint_file = self.checkpoint_path / 'checkpoint.json'
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"✓ Saved checkpoint: {last_record_id}")

    def _load_checkpoint(self):
        """Load checkpoint for resuming"""
        if not self.checkpoint_path:
            return

        checkpoint_file = self.checkpoint_path / 'checkpoint.json'
        if not checkpoint_file.exists():
            return

        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)

        self.last_checkpoint_id = checkpoint.get('last_record_id')

class DistributedETLPipeline:
    """
    Distributed ETL for trillion-row scale

    Architecture:
    - Partition data by key (user_id, date, hash)
    - Process partitions in parallel (100-1000 workers)
    - Shuffle and merge for global operations (deduplication)
    - Write to distributed storage (S3, GCS, HDFS)

    Technologies:
    - Spark for distributed processing
    - Delta Lake for ACID transactions
    - Airflow for orchestration

    Performance:
    - Single-node: 100K records/sec
    - 100-node cluster: 10M records/sec
    - 1000-node cluster: 100M records/sec
    """

    def __init__(
        self,
        num_partitions: int = 100,
        output_path: str = "s3://embeddings/features/"
    ):
        """
        Args:
            num_partitions: Number of partitions for parallel processing
            output_path: Distributed storage path
        """
        self.num_partitions = num_partitions
        self.output_path = output_path

        print(f"Initialized Distributed ETL Pipeline")
        print(f"  Partitions: {num_partitions}")
        print(f"  Output: {output_path}")

    def partition_data(
        self,
        records: List[DataRecord],
        partition_key: str = 'hash'
    ) -> Dict[int, List[DataRecord]]:
        """
        Partition data for parallel processing

        Partition strategies:
        - hash: Hash record_id for even distribution
        - date: Partition by timestamp for temporal locality
        - key: Partition by specific field (user_id, category)

        Args:
            records: Records to partition
            partition_key: Partitioning strategy

        Returns:
            Dict mapping partition_id to records
        """
        partitions = {i: [] for i in range(self.num_partitions)}

        for record in records:
            if partition_key == 'hash':
                partition_id = hash(record.record_id) % self.num_partitions
            elif partition_key == 'date':
                partition_id = record.timestamp.day % self.num_partitions
            else:
                partition_id = hash(str(record.data.get(partition_key, ''))) % self.num_partitions

            partitions[partition_id].append(record)

        print(f"Partitioned {len(records):,} records into {self.num_partitions} partitions")

        # Print partition sizes
        sizes = [len(p) for p in partitions.values()]
        print(f"  Partition sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.0f}")

        return partitions

    def process_partition(
        self,
        partition_id: int,
        records: List[DataRecord]
    ) -> List[EmbeddingFeatures]:
        """
        Process single partition (runs on worker node)

        Args:
            partition_id: Partition identifier
            records: Records in this partition

        Returns:
            Transformed features
        """
        print(f"Processing partition {partition_id} with {len(records):,} records")

        # Create single-node pipeline for this partition
        pipeline = EmbeddingETLPipeline(
            output_path=f"{self.output_path}/partition_{partition_id}",
            batch_size=10000
        )

        # Transform records
        features_list = pipeline.transform(records)

        return features_list

# Example: E-commerce product ETL
def ecommerce_etl_example():
    """
    ETL pipeline for e-commerce product embeddings

    Source: Product catalog (database)
    Transform: Combine title, description, category, attributes
    Load: Training-ready features

    Scale: 100M products, updated daily
    """

    # Simulate source data
    def generate_source_records(count=1000):
        """Simulate product catalog records"""
        for i in range(count):
            yield {
                'id': f'product_{i}',
                'data': {
                    'title': f'Product {i}',
                    'description': f'This is a great product for {i % 10} use cases',
                    'category': ['Electronics', 'Clothing', 'Books'][i % 3],
                    'price': 10.0 + (i % 100),
                    'rating': 3.0 + (i % 5) * 0.5,
                    'tags': ['tag1', 'tag2', 'tag3']
                },
                'source': 'product_db',
                'timestamp': datetime.now() - timedelta(hours=i % 24),
                'metadata': {
                    'labels': [['Electronics', 'Clothing', 'Books'][i % 3]]
                }
            }

    # Initialize pipeline
    pipeline = EmbeddingETLPipeline(
        output_path='/tmp/embeddings/features',
        checkpoint_path='/tmp/embeddings/checkpoints',
        batch_size=100
    )

    # Run ETL
    source_iterator = generate_source_records(1000)
    pipeline.run(
        source_iterator,
        incremental=True,
        output_format='jsonl'
    )

    print(f"\n✓ E-commerce ETL complete")
    print(f"  Output: {pipeline.output_path}")

# Uncomment to run:
# ecommerce_etl_example()
