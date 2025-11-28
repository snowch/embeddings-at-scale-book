"""Batch processing utilities for image embeddings."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class BatchResult:
    """Result of batch processing."""

    embeddings: list[np.ndarray]
    image_ids: list[str]
    failed: list[str]
    processing_time: float
    metadata: dict = field(default_factory=dict)


@dataclass
class ProcessingStats:
    """Statistics for batch processing."""

    total_images: int
    successful: int
    failed: int
    total_time: float
    images_per_second: float
    average_time_per_image: float


class BatchImageProcessor:
    """Process images in batches for embedding."""

    def __init__(
        self,
        encoder,
        batch_size: int = 32,
        target_size: int = 224,
        num_workers: int = 4,
        error_handling: str = "skip",
    ):
        """
        Initialize batch processor.

        Args:
            encoder: Image embedding model
            batch_size: Number of images per batch
            target_size: Model input size
            num_workers: Number of parallel workers for preprocessing
            error_handling: How to handle errors ('skip', 'raise', 'placeholder')
        """
        self.encoder = encoder
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_workers = num_workers
        self.error_handling = error_handling

    def process_images(
        self,
        image_paths: list[str | Path],
    ) -> BatchResult:
        """
        Process a list of images.

        Args:
            image_paths: List of image file paths

        Returns:
            BatchResult with embeddings and metadata
        """
        start_time = time.time()

        # Preprocess images in parallel
        preprocessed = self._parallel_preprocess(image_paths)

        # Batch encode
        all_embeddings = []
        successful_ids = []
        failed_ids = []

        for batch_start in range(0, len(preprocessed), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(preprocessed))
            batch_items = preprocessed[batch_start:batch_end]

            # Separate successful and failed
            batch_images = []
            batch_ids = []

            for path, result in batch_items:
                if result is not None:
                    batch_images.append(result)
                    batch_ids.append(str(path))
                else:
                    failed_ids.append(str(path))

            if batch_images:
                # Encode batch
                embeddings = self._encode_batch(batch_images)
                all_embeddings.extend(embeddings)
                successful_ids.extend(batch_ids)

        processing_time = time.time() - start_time

        return BatchResult(
            embeddings=all_embeddings,
            image_ids=successful_ids,
            failed=failed_ids,
            processing_time=processing_time,
            metadata={
                "batch_size": self.batch_size,
                "total_processed": len(successful_ids),
                "images_per_second": len(successful_ids) / processing_time
                if processing_time > 0
                else 0,
            },
        )

    def _parallel_preprocess(
        self,
        image_paths: list[str | Path],
    ) -> list[tuple[str | Path, Image.Image | None]]:
        """Preprocess images in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._preprocess_single, path): path for path in image_paths}

            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    results.append((path, result))
                except Exception:
                    if self.error_handling == "raise":
                        raise
                    results.append((path, None))

        return results

    def _preprocess_single(self, path: str | Path) -> Image.Image | None:
        """Preprocess a single image."""
        try:
            image = Image.open(path)

            # Convert to RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize
            image = image.resize(
                (self.target_size, self.target_size),
                Image.Resampling.LANCZOS,
            )

            return image

        except Exception:
            if self.error_handling == "raise":
                raise
            return None

    def _encode_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Encode a batch of images."""
        # If encoder supports batch encoding
        if hasattr(self.encoder, "encode_batch"):
            return self.encoder.encode_batch(images)

        # Otherwise encode one by one
        return [self.encoder.encode(img) for img in images]


class StreamingBatchProcessor:
    """Process images in a streaming fashion for very large datasets."""

    def __init__(
        self,
        encoder,
        batch_size: int = 32,
        target_size: int = 224,
        buffer_size: int = 100,
    ):
        """
        Initialize streaming processor.

        Args:
            encoder: Image embedding model
            batch_size: Number of images per batch
            target_size: Model input size
            buffer_size: Size of preprocessing buffer
        """
        self.encoder = encoder
        self.batch_size = batch_size
        self.target_size = target_size
        self.buffer_size = buffer_size

    def process_stream(self, image_iterator):
        """
        Process images from an iterator, yielding embeddings.

        Args:
            image_iterator: Iterator yielding (image_id, image_path) tuples

        Yields:
            Tuples of (image_id, embedding)
        """
        batch_images = []
        batch_ids = []

        for image_id, image_path in image_iterator:
            try:
                # Load and preprocess
                image = Image.open(image_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image = image.resize(
                    (self.target_size, self.target_size),
                    Image.Resampling.LANCZOS,
                )

                batch_images.append(image)
                batch_ids.append(image_id)

                # Process batch when full
                if len(batch_images) >= self.batch_size:
                    embeddings = self._encode_batch(batch_images)
                    yield from zip(batch_ids, embeddings)
                    batch_images = []
                    batch_ids = []

            except Exception:
                # Skip failed images in streaming mode
                continue

        # Process remaining images
        if batch_images:
            embeddings = self._encode_batch(batch_images)
            yield from zip(batch_ids, embeddings)

    def _encode_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Encode a batch of images."""
        if hasattr(self.encoder, "encode_batch"):
            return self.encoder.encode_batch(images)
        return [self.encoder.encode(img) for img in images]


def compute_processing_stats(results: list[BatchResult]) -> ProcessingStats:
    """
    Compute aggregate statistics from multiple batch results.

    Args:
        results: List of BatchResult objects

    Returns:
        ProcessingStats with aggregate metrics
    """
    total_successful = sum(len(r.embeddings) for r in results)
    total_failed = sum(len(r.failed) for r in results)
    total_time = sum(r.processing_time for r in results)

    return ProcessingStats(
        total_images=total_successful + total_failed,
        successful=total_successful,
        failed=total_failed,
        total_time=total_time,
        images_per_second=total_successful / total_time if total_time > 0 else 0,
        average_time_per_image=total_time / total_successful if total_successful > 0 else 0,
    )


def process_directory(
    directory: str | Path,
    encoder,
    batch_size: int = 32,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> BatchResult:
    """
    Process all images in a directory.

    Args:
        directory: Directory path
        encoder: Image embedding model
        batch_size: Batch size for processing
        extensions: Valid image extensions

    Returns:
        BatchResult with all embeddings
    """
    directory = Path(directory)

    # Find all images
    image_paths = []
    for ext in extensions:
        image_paths.extend(directory.glob(f"**/*{ext}"))
        image_paths.extend(directory.glob(f"**/*{ext.upper()}"))

    # Process
    processor = BatchImageProcessor(encoder=encoder, batch_size=batch_size)
    return processor.process_images(image_paths)
