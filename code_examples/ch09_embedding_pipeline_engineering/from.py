# Code from Chapter 09
# Book: Embeddings at Scale

"""
Production Embedding System Architecture

Components:
1. Training Pipeline: Generates embedding models
2. Inference Pipeline: Applies models to data at scale
3. Vector Store: Stores and indexes embeddings
4. Serving Layer: Handles real-time queries
5. Monitoring System: Tracks quality and performance
6. Orchestration: Coordinates updates and rollouts

Data flow:
Raw Data → Feature Engineering → Model Training → Model Registry
Raw Data → Feature Engineering → Batch Inference → Vector Store → Serving Layer
Real-time Data → Feature Engineering → Online Inference → Vector Store → Serving Layer
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class EmbeddingModelMetadata:
    """
    Metadata for tracking embedding models in production

    Critical for:
    - Version management across pipeline stages
    - Reproducing embeddings for debugging
    - Audit trails for compliance
    - Impact analysis for changes
    """
    model_id: str  # Unique identifier (e.g., "user-embeddings-v1.2.3")
    model_version: str  # Semantic version
    training_date: datetime
    training_data_hash: str  # Hash of training data for reproducibility
    hyperparameters: Dict
    performance_metrics: Dict  # Validation metrics
    embedding_dimension: int
    input_features: List[str]
    preprocessing_config: Dict

    # Deployment tracking
    deployed_to_staging: Optional[datetime] = None
    deployed_to_production: Optional[datetime] = None
    rollback_model_id: Optional[str] = None  # Previous version to rollback to

    # Model lineage
    parent_model_id: Optional[str] = None  # For fine-tuned models
    training_commit_hash: str = ""  # Git commit of training code

    def to_dict(self) -> Dict:
        """Serialize for storage in model registry"""
        return {
            'model_id': self.model_id,
            'model_version': self.model_version,
            'training_date': self.training_date.isoformat(),
            'training_data_hash': self.training_data_hash,
            'hyperparameters': self.hyperparameters,
            'performance_metrics': self.performance_metrics,
            'embedding_dimension': self.embedding_dimension,
            'input_features': self.input_features,
            'preprocessing_config': self.preprocessing_config,
            'deployed_to_staging': self.deployed_to_staging.isoformat() if self.deployed_to_staging else None,
            'deployed_to_production': self.deployed_to_production.isoformat() if self.deployed_to_production else None,
            'rollback_model_id': self.rollback_model_id,
            'parent_model_id': self.parent_model_id,
            'training_commit_hash': self.training_commit_hash
        }

class EmbeddingModelRegistry:
    """
    Central registry for embedding models across training and serving

    Functions:
    - Store trained models with metadata
    - Version management (semantic versioning)
    - Model lineage tracking
    - Deployment state management
    - Model discovery for inference pipelines

    In production: Use MLflow, Weights & Biases, or custom S3/GCS-based registry
    """

    def __init__(self, storage_backend: str = "local"):
        """
        Args:
            storage_backend: 's3', 'gcs', 'mlflow', or 'local'
        """
        self.storage_backend = storage_backend
        self.models: Dict[str, EmbeddingModelMetadata] = {}

    def register_model(
        self,
        model: nn.Module,
        metadata: EmbeddingModelMetadata,
        artifacts: Optional[Dict] = None
    ) -> str:
        """
        Register a trained embedding model

        Args:
            model: Trained PyTorch model
            metadata: Model metadata
            artifacts: Additional artifacts (preprocessing objects, config files)

        Returns:
            model_id: Unique identifier for registered model
        """
        # Validate model
        self._validate_model(model, metadata)

        # Store model weights
        model_path = self._store_model_weights(model, metadata.model_id)

        # Store artifacts (preprocessing pipelines, vocabularies, etc.)
        if artifacts:
            artifact_path = self._store_artifacts(artifacts, metadata.model_id)

        # Register in metadata store
        self.models[metadata.model_id] = metadata
        self._persist_metadata(metadata)

        print(f"✓ Registered model: {metadata.model_id}")
        print(f"  Version: {metadata.model_version}")
        print(f"  Embedding dim: {metadata.embedding_dimension}")
        print(f"  Training date: {metadata.training_date}")

        return metadata.model_id

    def load_model(
        self,
        model_id: str,
        device: str = 'cpu'
    ) -> tuple[nn.Module, EmbeddingModelMetadata]:
        """
        Load model and metadata from registry

        Args:
            model_id: Model identifier
            device: 'cpu', 'cuda', or specific device

        Returns:
            (model, metadata): Loaded model and metadata
        """
        if model_id not in self.models:
            # Try loading from persistent storage
            metadata = self._load_metadata(model_id)
            if metadata is None:
                raise ValueError(f"Model {model_id} not found in registry")
            self.models[model_id] = metadata
        else:
            metadata = self.models[model_id]

        # Load model weights
        model = self._load_model_weights(model_id, metadata, device)

        print(f"✓ Loaded model: {model_id}")
        print(f"  Version: {metadata.model_version}")

        return model, metadata

    def promote_to_production(
        self,
        model_id: str,
        validation_results: Dict
    ) -> bool:
        """
        Promote model from staging to production

        Validates:
        - Performance metrics meet thresholds
        - Staging testing completed
        - Rollback plan exists

        Args:
            model_id: Model to promote
            validation_results: Results from staging validation

        Returns:
            success: True if promotion succeeded
        """
        metadata = self.models.get(model_id)
        if metadata is None:
            raise ValueError(f"Model {model_id} not found")

        # Validation checks
        if metadata.deployed_to_staging is None:
            raise ValueError("Model must be deployed to staging before production")

        # Check performance thresholds
        required_metrics = ['retrieval_recall@10', 'embedding_quality_score']
        for metric in required_metrics:
            if metric not in validation_results:
                raise ValueError(f"Missing required validation metric: {metric}")

        # Example thresholds (customize per use case)
        if validation_results['retrieval_recall@10'] < 0.85:
            print(f"✗ Promotion failed: recall@10 ({validation_results['retrieval_recall@10']}) < 0.85")
            return False

        # Find current production model for rollback
        current_prod_model = self._get_current_production_model()
        if current_prod_model:
            metadata.rollback_model_id = current_prod_model.model_id

        # Promote
        metadata.deployed_to_production = datetime.now()
        self._persist_metadata(metadata)

        # Update production pointer
        self._update_production_pointer(model_id)

        print(f"✓ Promoted {model_id} to production")
        if metadata.rollback_model_id:
            print(f"  Rollback target: {metadata.rollback_model_id}")

        return True

    def rollback(self) -> str:
        """
        Rollback to previous production model

        Returns:
            model_id: ID of model rolled back to
        """
        current_prod = self._get_current_production_model()
        if current_prod is None:
            raise ValueError("No production model to rollback from")

        if current_prod.rollback_model_id is None:
            raise ValueError("No rollback target defined")

        rollback_model_id = current_prod.rollback_model_id

        # Update production pointer
        self._update_production_pointer(rollback_model_id)

        print(f"✓ Rolled back to {rollback_model_id}")
        print(f"  From: {current_prod.model_id}")

        return rollback_model_id

    def _validate_model(self, model: nn.Module, metadata: EmbeddingModelMetadata):
        """Validate model matches metadata"""
        # Check model produces correct embedding dimension
        dummy_input = torch.randn(1, len(metadata.input_features))
        try:
            output = model(dummy_input)
            if output.shape[-1] != metadata.embedding_dimension:
                raise ValueError(
                    f"Model output dim ({output.shape[-1]}) != "
                    f"metadata dim ({metadata.embedding_dimension})"
                )
        except Exception as e:
            raise ValueError(f"Model validation failed: {e}")

    def _store_model_weights(self, model: nn.Module, model_id: str) -> str:
        """Store model weights to backend storage"""
        # In production: Upload to S3/GCS
        # For now: Local storage
        path = f"models/{model_id}/model.pt"
        torch.save(model.state_dict(), path)
        return path

    def _store_artifacts(self, artifacts: Dict, model_id: str) -> str:
        """Store additional artifacts"""
        path = f"models/{model_id}/artifacts.json"
        with open(path, 'w') as f:
            json.dump(artifacts, f)
        return path

    def _persist_metadata(self, metadata: EmbeddingModelMetadata):
        """Persist metadata to storage"""
        path = f"models/{metadata.model_id}/metadata.json"
        with open(path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def _load_metadata(self, model_id: str) -> Optional[EmbeddingModelMetadata]:
        """Load metadata from storage"""
        # Implementation depends on storage backend
        return None

    def _load_model_weights(
        self,
        model_id: str,
        metadata: EmbeddingModelMetadata,
        device: str
    ) -> nn.Module:
        """Load model weights from storage"""
        # In production: Download from S3/GCS
        path = f"models/{model_id}/model.pt"

        # Reconstruct model architecture (stored in metadata.hyperparameters)
        model = self._reconstruct_model_architecture(metadata)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()

        return model

    def _reconstruct_model_architecture(self, metadata: EmbeddingModelMetadata) -> nn.Module:
        """Reconstruct model architecture from metadata"""
        # Simplified example - real implementation would use config files
        class SimpleEmbedding(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, output_dim)
                )

            def forward(self, x):
                return self.encoder(x)

        return SimpleEmbedding(
            len(metadata.input_features),
            metadata.embedding_dimension
        )

    def _get_current_production_model(self) -> Optional[EmbeddingModelMetadata]:
        """Get current production model"""
        # In production: Read from production pointer file/database
        prod_models = [
            m for m in self.models.values()
            if m.deployed_to_production is not None
        ]
        if not prod_models:
            return None
        return max(prod_models, key=lambda m: m.deployed_to_production)

    def _update_production_pointer(self, model_id: str):
        """Update production pointer to new model"""
        # In production: Update pointer file in S3, or database entry
        # This is what serving layer reads to determine which model to use
        pointer_path = "production/current_model.txt"
        with open(pointer_path, 'w') as f:
            f.write(model_id)

class EmbeddingInferencePipeline:
    """
    Batch inference pipeline for generating embeddings at scale

    Handles:
    - Large-scale batch processing (billions of items)
    - Checkpointing for fault tolerance
    - Distributed processing across workers
    - Resource optimization (GPU utilization, batching)
    - Progress tracking and monitoring

    Typical throughput:
    - Single GPU: 10K-100K embeddings/second
    - 8 GPU node: 100K-1M embeddings/second
    - 100 GPU cluster: 10M-100M embeddings/second
    """

    def __init__(
        self,
        model_registry: EmbeddingModelRegistry,
        model_id: str,
        batch_size: int = 1024,
        num_workers: int = 4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model_registry: Registry to load model from
            model_id: Model to use for inference
            batch_size: Batch size for inference
            num_workers: Data loading workers
            device: Inference device
        """
        self.model_registry = model_registry
        self.model_id = model_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        # Load model
        self.model, self.metadata = model_registry.load_model(model_id, device)
        self.model.eval()

        # Metrics
        self.processed_count = 0
        self.start_time = None

    def process_batch(
        self,
        data_iterator,
        output_writer,
        checkpoint_every: int = 100000
    ):
        """
        Process large dataset in batches

        Args:
            data_iterator: Iterator yielding batches of data
            output_writer: Writer for generated embeddings
            checkpoint_every: Save checkpoint every N items
        """
        self.start_time = datetime.now()
        batch_buffer = []

        print(f"Starting batch inference with model {self.model_id}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Device: {self.device}")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_iterator):
                # Move to device
                if isinstance(batch_data, torch.Tensor):
                    batch_data = batch_data.to(self.device)

                # Generate embeddings
                embeddings = self.model(batch_data)

                # Write to output
                batch_buffer.extend(embeddings.cpu().numpy())
                self.processed_count += len(embeddings)

                # Flush buffer periodically
                if len(batch_buffer) >= checkpoint_every:
                    output_writer.write(batch_buffer)
                    batch_buffer = []
                    self._save_checkpoint(batch_idx)
                    self._log_progress()

                # Memory management
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Final flush
        if batch_buffer:
            output_writer.write(batch_buffer)

        self._log_progress(final=True)

    def _save_checkpoint(self, batch_idx: int):
        """Save checkpoint for fault tolerance"""
        checkpoint = {
            'model_id': self.model_id,
            'processed_count': self.processed_count,
            'batch_idx': batch_idx,
            'timestamp': datetime.now().isoformat()
        }
        # In production: Save to persistent storage
        with open('checkpoint.json', 'w') as f:
            json.dump(checkpoint, f)

    def _log_progress(self, final: bool = False):
        """Log progress metrics"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        throughput = self.processed_count / elapsed if elapsed > 0 else 0

        if final:
            print("\n✓ Batch inference complete")
            print(f"  Total processed: {self.processed_count:,}")
            print(f"  Elapsed time: {elapsed:.1f}s")
            print(f"  Throughput: {throughput:,.0f} embeddings/second")
        else:
            print(f"  Progress: {self.processed_count:,} embeddings "
                  f"({throughput:,.0f}/sec)", end='\r')

# Example: Complete MLOps workflow
def embedding_mlops_example():
    """
    End-to-end MLOps workflow for embeddings

    Steps:
    1. Train model
    2. Register in model registry
    3. Deploy to staging
    4. Validate in staging
    5. Promote to production
    6. Run batch inference
    7. Monitor in production
    """

    # 1. Train model (simplified)
    class SimpleEmbeddingModel(nn.Module):
        def __init__(self, input_dim, embedding_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, embedding_dim)
            )

        def forward(self, x):
            return self.encoder(x)

    model = SimpleEmbeddingModel(input_dim=100, embedding_dim=256)

    # 2. Create metadata
    metadata = EmbeddingModelMetadata(
        model_id="product-embeddings-v1.0.0",
        model_version="1.0.0",
        training_date=datetime.now(),
        training_data_hash="abc123",
        hyperparameters={
            'embedding_dim': 256,
            'learning_rate': 0.001,
            'batch_size': 512
        },
        performance_metrics={
            'retrieval_recall@10': 0.89,
            'embedding_quality_score': 0.92
        },
        embedding_dimension=256,
        input_features=['feature_' + str(i) for i in range(100)],
        preprocessing_config={'normalization': 'standard'},
        training_commit_hash="def456"
    )

    # 3. Register model
    registry = EmbeddingModelRegistry(storage_backend='local')
    model_id = registry.register_model(model, metadata)

    # 4. Deploy to staging
    metadata.deployed_to_staging = datetime.now()
    registry._persist_metadata(metadata)
    print(f"\n✓ Deployed to staging: {model_id}")

    # 5. Validate in staging
    staging_validation_results = {
        'retrieval_recall@10': 0.89,
        'embedding_quality_score': 0.92,
        'latency_p99_ms': 15
    }

    # 6. Promote to production
    success = registry.promote_to_production(model_id, staging_validation_results)

    if success:
        # 7. Run batch inference
        pipeline = EmbeddingInferencePipeline(
            model_registry=registry,
            model_id=model_id,
            batch_size=1024
        )

        print("\n✓ MLOps workflow complete")
        print(f"  Model in production: {model_id}")

# Uncomment to run:
# embedding_mlops_example()
