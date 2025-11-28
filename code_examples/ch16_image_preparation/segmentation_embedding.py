"""Segmentation-based region embedding extraction."""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class SegmentedRegion:
    """A segmented region from an image."""

    image: Image.Image
    mask: np.ndarray
    label: str
    instance_id: int = 0
    area: int = 0
    bbox: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2)


@dataclass
class SegmentationEmbedding:
    """Embedding for a segmented region."""

    embedding: np.ndarray
    label: str
    instance_id: int
    bbox: tuple[int, int, int, int]
    metadata: dict = field(default_factory=dict)


class SegmentationEmbedder:
    """Extract embeddings from segmented regions."""

    def __init__(
        self,
        encoder,
        target_size: int = 224,
        min_area: int = 100,
        background_handling: str = "mask",
    ):
        """
        Initialize segmentation embedder.

        Args:
            encoder: Image embedding model
            target_size: Model input size
            min_area: Minimum region area in pixels
            background_handling: How to handle background ('mask', 'crop', 'pad')
        """
        self.encoder = encoder
        self.target_size = target_size
        self.min_area = min_area
        self.background_handling = background_handling

    def embed_regions(
        self,
        image: Image.Image,
        segmentation_mask: np.ndarray,
        labels: dict[int, str] | None = None,
    ) -> list[SegmentationEmbedding]:
        """
        Extract and embed segmented regions.

        Args:
            image: Original image
            segmentation_mask: Integer mask where each value is a segment ID
            labels: Mapping from segment ID to label name

        Returns:
            List of embeddings for each valid region
        """
        labels = labels or {}
        embeddings = []

        # Get unique segment IDs (excluding 0 which is typically background)
        segment_ids = np.unique(segmentation_mask)
        segment_ids = segment_ids[segment_ids != 0]

        for seg_id in segment_ids:
            # Create binary mask for this segment
            mask = (segmentation_mask == seg_id).astype(np.uint8)
            area = np.sum(mask)

            if area < self.min_area:
                continue

            # Get bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            bbox = (x1, y1, x2 + 1, y2 + 1)

            # Extract region image
            region_image = self._extract_region(image, mask, bbox)

            # Generate embedding
            embedding = self.encoder.encode(region_image)

            embeddings.append(
                SegmentationEmbedding(
                    embedding=embedding,
                    label=labels.get(seg_id, f"segment_{seg_id}"),
                    instance_id=int(seg_id),
                    bbox=bbox,
                    metadata={
                        "area": int(area),
                        "background_handling": self.background_handling,
                    },
                )
            )

        return embeddings

    def _extract_region(
        self,
        image: Image.Image,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> Image.Image:
        """
        Extract region using specified background handling.

        Args:
            image: Original image
            mask: Binary mask for region
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Processed region image
        """
        x1, y1, x2, y2 = bbox
        img_array = np.array(image)

        if self.background_handling == "mask":
            # Apply mask, set background to white
            masked = img_array.copy()
            mask_3d = np.stack([mask] * 3, axis=2)
            masked[mask_3d == 0] = 255

            # Crop to bounding box
            cropped = masked[y1:y2, x1:x2]

        elif self.background_handling == "crop":
            # Simple crop without masking
            cropped = img_array[y1:y2, x1:x2]

        elif self.background_handling == "pad":
            # Crop and pad to square
            cropped = img_array[y1:y2, x1:x2]
            mask_3d = np.stack([mask[y1:y2, x1:x2]] * 3, axis=2)
            cropped[mask_3d == 0] = 255

        else:
            raise ValueError(f"Unknown background handling: {self.background_handling}")

        # Convert to PIL and resize
        region_image = Image.fromarray(cropped)
        region_image = self._resize_with_padding(region_image)

        return region_image

    def _resize_with_padding(self, image: Image.Image) -> Image.Image:
        """Resize image preserving aspect ratio with padding."""
        aspect = image.width / image.height

        if aspect > 1:
            new_width = self.target_size
            new_height = int(self.target_size / aspect)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect)

        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create white background
        padded = Image.new("RGB", (self.target_size, self.target_size), (255, 255, 255))
        paste_x = (self.target_size - new_width) // 2
        paste_y = (self.target_size - new_height) // 2
        padded.paste(resized, (paste_x, paste_y))

        return padded


def embed_instance_segmentation(
    image: Image.Image,
    instances: list[dict],
    encoder,
    target_size: int = 224,
) -> list[SegmentationEmbedding]:
    """
    Embed regions from instance segmentation results.

    Args:
        image: Original image
        instances: List of instance dicts with 'mask', 'label', 'score' keys
        encoder: Image embedding model
        target_size: Model input size

    Returns:
        List of embeddings for each instance
    """
    embedder = SegmentationEmbedder(
        encoder=encoder,
        target_size=target_size,
        background_handling="mask",
    )

    # Create combined mask
    h, w = np.array(image).shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.int32)
    labels = {}

    for idx, instance in enumerate(instances, start=1):
        mask = instance["mask"]
        combined_mask[mask > 0] = idx
        labels[idx] = instance.get("label", f"instance_{idx}")

    return embedder.embed_regions(image, combined_mask, labels)


def embed_semantic_segmentation(
    image: Image.Image,
    semantic_mask: np.ndarray,
    class_names: dict[int, str],
    encoder,
    target_size: int = 224,
) -> list[SegmentationEmbedding]:
    """
    Embed regions from semantic segmentation.

    Args:
        image: Original image
        semantic_mask: Semantic segmentation mask
        class_names: Mapping from class ID to name
        encoder: Image embedding model
        target_size: Model input size

    Returns:
        List of embeddings for each semantic class
    """
    embedder = SegmentationEmbedder(
        encoder=encoder,
        target_size=target_size,
        min_area=500,  # Larger min area for semantic regions
        background_handling="mask",
    )

    return embedder.embed_regions(image, semantic_mask, class_names)
