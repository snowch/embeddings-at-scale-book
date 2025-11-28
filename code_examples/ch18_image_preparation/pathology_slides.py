"""Digital pathology slide processing for embeddings."""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class PathologyPatch:
    """A patch from a pathology slide."""

    image: Image.Image
    location: tuple[int, int]  # (x, y) in full resolution
    level: int  # Pyramid level
    patch_id: str = ""
    tissue_ratio: float = 1.0


@dataclass
class PathologyEmbedding:
    """Embedding for a pathology patch."""

    embedding: np.ndarray
    patch_id: str
    location: tuple[int, int]
    level: int
    metadata: dict = field(default_factory=dict)


class PathologySlideProcessor:
    """Process whole slide images (WSI) for embedding."""

    def __init__(
        self,
        encoder,
        patch_size: int = 256,
        target_magnification: float = 20.0,
        tissue_threshold: float = 0.5,
        overlap: float = 0.0,
    ):
        """
        Initialize pathology slide processor.

        Args:
            encoder: Image embedding model
            patch_size: Size of patches in pixels
            target_magnification: Target magnification (e.g., 20x, 40x)
            tissue_threshold: Minimum tissue content ratio
            overlap: Overlap ratio between patches
        """
        self.encoder = encoder
        self.patch_size = patch_size
        self.target_magnification = target_magnification
        self.tissue_threshold = tissue_threshold
        self.overlap = overlap

    def extract_patches(
        self,
        slide_image: Image.Image,
        tissue_mask: np.ndarray | None = None,
    ) -> list[PathologyPatch]:
        """
        Extract tissue patches from slide image.

        Args:
            slide_image: Slide image (or region)
            tissue_mask: Binary mask of tissue regions

        Returns:
            List of PathologyPatch objects
        """
        width, height = slide_image.size
        step = int(self.patch_size * (1 - self.overlap))

        # Generate tissue mask if not provided
        if tissue_mask is None:
            tissue_mask = self._detect_tissue(slide_image)

        patches = []
        patch_idx = 0

        for y in range(0, height - self.patch_size + 1, step):
            for x in range(0, width - self.patch_size + 1, step):
                # Check tissue content
                mask_patch = tissue_mask[y : y + self.patch_size, x : x + self.patch_size]
                tissue_ratio = np.mean(mask_patch)

                if tissue_ratio < self.tissue_threshold:
                    continue

                # Extract patch
                patch_image = slide_image.crop((x, y, x + self.patch_size, y + self.patch_size))

                patches.append(
                    PathologyPatch(
                        image=patch_image,
                        location=(x, y),
                        level=0,
                        patch_id=f"patch_{patch_idx}",
                        tissue_ratio=tissue_ratio,
                    )
                )
                patch_idx += 1

        return patches

    def embed_patches(
        self,
        patches: list[PathologyPatch],
    ) -> list[PathologyEmbedding]:
        """
        Generate embeddings for pathology patches.

        Args:
            patches: List of PathologyPatch objects

        Returns:
            List of PathologyEmbedding objects
        """
        embeddings = []

        for patch in patches:
            # Preprocess for pathology
            processed = self._preprocess_pathology(patch.image)

            # Generate embedding
            embedding = self.encoder.encode(processed)

            embeddings.append(
                PathologyEmbedding(
                    embedding=embedding,
                    patch_id=patch.patch_id,
                    location=patch.location,
                    level=patch.level,
                    metadata={
                        "tissue_ratio": patch.tissue_ratio,
                        "patch_size": self.patch_size,
                    },
                )
            )

        return embeddings

    def _detect_tissue(self, image: Image.Image) -> np.ndarray:
        """
        Detect tissue regions using color thresholding.

        Args:
            image: Input image

        Returns:
            Binary mask of tissue regions
        """
        # Convert to numpy
        img_array = np.array(image)

        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Simple thresholding (tissue is darker than background)
        # Background in H&E slides is typically white (>220)
        tissue_mask = gray < 220

        # Also filter out very dark regions (artifacts)
        tissue_mask = tissue_mask & (gray > 20)

        return tissue_mask.astype(np.uint8)

    def _preprocess_pathology(self, image: Image.Image) -> Image.Image:
        """
        Preprocess pathology image for embedding.

        Args:
            image: Input patch image

        Returns:
            Preprocessed image
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Stain normalization (simplified - production would use Macenko)
        img_array = np.array(image).astype(np.float32)

        # Simple contrast adjustment
        for i in range(3):
            channel = img_array[:, :, i]
            p2, p98 = np.percentile(channel, (2, 98))
            img_array[:, :, i] = np.clip((channel - p2) / (p98 - p2) * 255, 0, 255)

        return Image.fromarray(img_array.astype(np.uint8))


def aggregate_slide_embeddings(
    embeddings: list[PathologyEmbedding],
    method: str = "mean",
) -> np.ndarray:
    """
    Aggregate patch embeddings to slide-level embedding.

    Args:
        embeddings: List of patch embeddings
        method: Aggregation method ('mean', 'max', 'attention')

    Returns:
        Slide-level embedding
    """
    vectors = np.array([e.embedding for e in embeddings])

    if method == "mean":
        return np.mean(vectors, axis=0)
    elif method == "max":
        return np.max(vectors, axis=0)
    elif method == "attention":
        # Simple attention based on tissue ratio
        weights = np.array([e.metadata.get("tissue_ratio", 1.0) for e in embeddings])
        weights = weights / weights.sum()
        return np.average(vectors, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
