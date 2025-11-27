"""Production image embedding pipeline."""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Iterator, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Image processing modes."""
    STANDARD = "standard"     # Single embedding per image
    MULTI_CROP = "multi_crop"  # Multiple crops aggregated
    TILED = "tiled"           # For large images
    OBJECT_LEVEL = "object"   # Embed detected objects


@dataclass
class ImageMetadata:
    """Metadata for a processed image."""
    image_id: str
    source_path: Optional[str] = None
    width: int = 0
    height: int = 0
    format: str = ""
    processing_mode: str = ""
    quality_score: float = 0.0
    processed_at: datetime = field(default_factory=datetime.now)
    custom: Dict = field(default_factory=dict)


@dataclass
class ProcessedImage:
    """Result of image processing."""
    image_id: str
    embedding: np.ndarray
    metadata: ImageMetadata
    additional_embeddings: Optional[Dict[str, np.ndarray]] = None


@dataclass
class PipelineConfig:
    """Configuration for the image pipeline."""
    # Model settings
    model_name: str = "resnet50"
    embedding_dim: int = 2048

    # Processing settings
    target_size: tuple = (224, 224)
    processing_mode: ProcessingMode = ProcessingMode.STANDARD
    normalize_embeddings: bool = True

    # Quality filtering
    enable_quality_filter: bool = True
    min_resolution: int = 100
    blur_threshold: float = 100.0

    # Tiling settings (for TILED mode)
    tile_overlap: float = 0.1
    max_tiles: int = 50

    # Batch processing
    batch_size: int = 32


class ImageEmbeddingPipeline:
    """
    Production-ready image embedding pipeline.

    Features:
    - Multiple processing modes
    - Quality filtering
    - Batch processing with GPU support
    - Metadata tracking
    - Error handling and logging
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._encoder = None
        self._initialized = False

    def initialize(self):
        """Initialize models (lazy loading)."""
        if self._initialized:
            return

        logger.info(f"Initializing pipeline with {self.config.model_name}")

        # Initialize encoder based on model name
        if 'vit' in self.config.model_name.lower():
            from vit_embeddings import ViTEmbedder
            self._encoder = ViTEmbedder(
                model_name=self.config.model_name,
                batch_size=self.config.batch_size
            )
        else:
            from cnn_embeddings import CNNEmbedder
            self._encoder = CNNEmbedder(
                model_name=self.config.model_name,
                batch_size=self.config.batch_size
            )

        self._initialized = True
        logger.info("Pipeline initialized")

    def process_image(
        self,
        image,
        image_id: Optional[str] = None,
        source_path: Optional[str] = None,
        custom_metadata: Optional[Dict] = None
    ) -> Optional[ProcessedImage]:
        """
        Process a single image through the pipeline.

        Args:
            image: PIL Image or numpy array
            image_id: Unique identifier (generated if not provided)
            source_path: Original file path
            custom_metadata: Additional metadata to store

        Returns:
            ProcessedImage or None if image fails quality checks
        """
        from PIL import Image

        self.initialize()

        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Generate ID if not provided
        if image_id is None:
            image_id = self._generate_id(pil_image)

        # Quality check
        if self.config.enable_quality_filter:
            from quality_assessment import assess_image_quality
            quality = assess_image_quality(
                pil_image,
                min_resolution=self.config.min_resolution,
                blur_threshold=self.config.blur_threshold
            )
            if not quality.passed:
                logger.warning(f"Image {image_id} failed quality check: {quality.issues}")
                return None
            quality_score = quality.blur_score
        else:
            quality_score = 0.0

        # Process based on mode
        if self.config.processing_mode == ProcessingMode.STANDARD:
            embedding = self._process_standard(pil_image)
            additional = None

        elif self.config.processing_mode == ProcessingMode.MULTI_CROP:
            embedding, additional = self._process_multi_crop(pil_image)

        elif self.config.processing_mode == ProcessingMode.TILED:
            embedding, additional = self._process_tiled(pil_image)

        elif self.config.processing_mode == ProcessingMode.OBJECT_LEVEL:
            embedding, additional = self._process_objects(pil_image)

        else:
            raise ValueError(f"Unknown mode: {self.config.processing_mode}")

        # Normalize if requested
        if self.config.normalize_embeddings:
            embedding = embedding / np.linalg.norm(embedding)

        # Create metadata
        w, h = pil_image.size
        metadata = ImageMetadata(
            image_id=image_id,
            source_path=source_path,
            width=w,
            height=h,
            format=pil_image.format or 'unknown',
            processing_mode=self.config.processing_mode.value,
            quality_score=quality_score,
            custom=custom_metadata or {}
        )

        return ProcessedImage(
            image_id=image_id,
            embedding=embedding,
            metadata=metadata,
            additional_embeddings=additional
        )

    def process_batch(
        self,
        images: List,
        image_ids: Optional[List[str]] = None,
        skip_failures: bool = True
    ) -> Iterator[ProcessedImage]:
        """
        Process multiple images efficiently.

        Args:
            images: List of PIL Images or numpy arrays
            image_ids: Optional list of IDs
            skip_failures: Whether to skip failed images or raise

        Yields:
            ProcessedImage for each successfully processed image
        """
        self.initialize()

        if image_ids is None:
            image_ids = [None] * len(images)

        for img, img_id in zip(images, image_ids):
            try:
                result = self.process_image(img, image_id=img_id)
                if result is not None:
                    yield result
            except Exception as e:
                logger.error(f"Error processing image {img_id}: {e}")
                if not skip_failures:
                    raise

    def _process_standard(self, image) -> np.ndarray:
        """Standard single-embedding processing."""
        return self._encoder.encode([image])[0]

    def _process_multi_crop(self, image) -> tuple:
        """Multi-crop processing with aggregation."""
        from resolution_handling import multi_crop_embedding

        # Get aggregated embedding
        embedding = multi_crop_embedding(
            image,
            self._encoder,
            target_size=self.config.target_size,
            num_crops=5,
            aggregation='mean'
        )

        return embedding, None

    def _process_tiled(self, image) -> tuple:
        """Tiled processing for large images."""
        from tiling_strategy import TiledImageProcessor

        processor = TiledImageProcessor(
            tile_size=self.config.target_size,
            overlap=self.config.tile_overlap,
            max_tiles=self.config.max_tiles
        )

        result = processor.process(image, self._encoder)

        additional = {
            'tile_embeddings': result['tile_embeddings'],
            'tile_positions': result['tile_positions']
        }

        return result['aggregate_embedding'], additional

    def _process_objects(self, image) -> tuple:
        """Object-level processing."""
        from object_detection_embedding import ObjectLevelEmbedder

        embedder = ObjectLevelEmbedder(
            embedding_model=self.config.model_name
        )

        result = embedder.process_image(image)

        # Use scene embedding as primary
        embedding = result['scene_embedding']

        # Store object embeddings as additional
        object_embeddings = {}
        for i, obj in enumerate(result['objects']):
            if obj.embedding is not None:
                key = f"object_{i}_{obj.class_name}"
                object_embeddings[key] = obj.embedding

        return embedding, object_embeddings if object_embeddings else None

    def _generate_id(self, image) -> str:
        """Generate unique ID from image content."""
        img_array = np.array(image)
        content_hash = hashlib.md5(img_array.tobytes()).hexdigest()
        return f"img_{content_hash[:12]}"


def prepare_for_vector_db(
    processed_images: List[ProcessedImage]
) -> List[Dict]:
    """
    Prepare processed images for vector database insertion.

    Returns list of records with id, embedding, and metadata.
    """
    records = []

    for img in processed_images:
        record = {
            'id': img.image_id,
            'embedding': img.embedding.tolist(),
            'metadata': {
                'width': img.metadata.width,
                'height': img.metadata.height,
                'format': img.metadata.format,
                'processing_mode': img.metadata.processing_mode,
                'quality_score': img.metadata.quality_score,
                'processed_at': img.metadata.processed_at.isoformat(),
                **img.metadata.custom
            }
        }
        records.append(record)

        # Add additional embeddings as separate records if present
        if img.additional_embeddings:
            for key, emb in img.additional_embeddings.items():
                if isinstance(emb, np.ndarray):
                    records.append({
                        'id': f"{img.image_id}_{key}",
                        'embedding': emb.tolist(),
                        'metadata': {
                            'parent_id': img.image_id,
                            'embedding_type': key,
                            **img.metadata.custom
                        }
                    })

    return records


# Example usage
if __name__ == "__main__":
    from PIL import Image

    print("Image Embedding Pipeline Demo")
    print("=" * 50)

    # Create sample images
    sample_images = [
        Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        for _ in range(3)
    ]

    # Configure pipeline
    config = PipelineConfig(
        model_name='resnet50',
        processing_mode=ProcessingMode.STANDARD,
        enable_quality_filter=False,  # Disable for demo
        batch_size=8
    )

    pipeline = ImageEmbeddingPipeline(config)

    print(f"Processing {len(sample_images)} images...")
    print(f"Mode: {config.processing_mode.value}")
    print(f"Model: {config.model_name}")
    print()

    # Process images
    results = list(pipeline.process_batch(sample_images))

    print(f"Successfully processed: {len(results)} images")
    for result in results:
        print(f"\nImage: {result.image_id}")
        print(f"  Size: {result.metadata.width}x{result.metadata.height}")
        print(f"  Embedding shape: {result.embedding.shape}")
        print(f"  Embedding norm: {np.linalg.norm(result.embedding):.3f}")

    # Prepare for vector DB
    records = prepare_for_vector_db(results)
    print(f"\nPrepared {len(records)} records for vector database")
