"""Multi-resolution pyramid embedding for scale-invariant retrieval."""

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class MultiResolutionEmbedding:
    """Embedding with multiple resolution levels."""

    embeddings: dict[str, np.ndarray]  # resolution_name -> embedding
    original_size: tuple[int, int]
    levels: list[str]


class MultiResolutionEmbedder:
    """Create embeddings at multiple scales for scale-invariant retrieval."""

    def __init__(
        self,
        encoder,
        scales: list[float] | None = None,
        target_size: int = 224,
    ):
        """
        Initialize multi-resolution embedder.

        Args:
            encoder: Image embedding model with encode() method
            scales: Scale factors (default: [0.5, 1.0, 2.0])
            target_size: Model input size
        """
        self.encoder = encoder
        self.scales = scales or [0.5, 1.0, 2.0]
        self.target_size = target_size

    def embed(self, image: Image.Image) -> MultiResolutionEmbedding:
        """
        Create embeddings at multiple resolutions.

        Args:
            image: Input PIL Image

        Returns:
            MultiResolutionEmbedding with embeddings at each scale
        """
        original_size = image.size
        embeddings = {}

        for scale in self.scales:
            # Resize to scaled version
            new_width = int(image.width * scale)
            new_height = int(image.height * scale)

            if new_width < self.target_size or new_height < self.target_size:
                # Skip if scaled version is smaller than model input
                continue

            scaled_image = image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS,
            )

            # Center crop to target size
            left = (new_width - self.target_size) // 2
            top = (new_height - self.target_size) // 2
            cropped = scaled_image.crop(
                (left, top, left + self.target_size, top + self.target_size)
            )

            # Generate embedding
            level_name = f"scale_{scale:.1f}"
            embeddings[level_name] = self.encoder.encode(cropped)

        return MultiResolutionEmbedding(
            embeddings=embeddings,
            original_size=original_size,
            levels=list(embeddings.keys()),
        )

    def embed_pyramid(
        self,
        image: Image.Image,
        num_levels: int = 3,
    ) -> MultiResolutionEmbedding:
        """
        Create image pyramid embeddings.

        Args:
            image: Input PIL Image
            num_levels: Number of pyramid levels

        Returns:
            MultiResolutionEmbedding with pyramid levels
        """
        embeddings = {}
        current_image = image

        for level in range(num_levels):
            level_name = f"level_{level}"

            # Resize to target size for embedding
            resized = current_image.resize(
                (self.target_size, self.target_size),
                Image.Resampling.LANCZOS,
            )

            embeddings[level_name] = self.encoder.encode(resized)

            # Downsample for next level
            current_image = current_image.resize(
                (current_image.width // 2, current_image.height // 2),
                Image.Resampling.LANCZOS,
            )

            if current_image.width < self.target_size:
                break

        return MultiResolutionEmbedding(
            embeddings=embeddings,
            original_size=image.size,
            levels=list(embeddings.keys()),
        )


def aggregate_multi_resolution(
    mr_embedding: MultiResolutionEmbedding,
    method: str = "mean",
    weights: list[float] | None = None,
) -> np.ndarray:
    """
    Aggregate multi-resolution embeddings into single vector.

    Args:
        mr_embedding: Multi-resolution embedding
        method: Aggregation method ('mean', 'max', 'weighted')
        weights: Weights for weighted aggregation

    Returns:
        Single aggregated embedding vector
    """
    embeddings = list(mr_embedding.embeddings.values())

    if method == "mean":
        return np.mean(embeddings, axis=0)
    elif method == "max":
        return np.max(embeddings, axis=0)
    elif method == "weighted":
        if weights is None:
            # Default: weight higher resolutions more
            weights = [1.0 / (i + 1) for i in range(len(embeddings))]
        weights = np.array(weights) / sum(weights)
        return np.average(embeddings, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def concatenate_multi_resolution(
    mr_embedding: MultiResolutionEmbedding,
) -> np.ndarray:
    """
    Concatenate all resolution embeddings.

    Results in higher-dimensional but more expressive vector.

    Args:
        mr_embedding: Multi-resolution embedding

    Returns:
        Concatenated embedding vector
    """
    embeddings = [mr_embedding.embeddings[level] for level in mr_embedding.levels]
    return np.concatenate(embeddings)
