"""Attention-guided region extraction for embeddings."""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class AttentionRegion:
    """A region identified by attention analysis."""

    image: Image.Image
    attention_score: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    region_id: int = 0


@dataclass
class AttentionEmbedding:
    """Embedding for an attention-identified region."""

    embedding: np.ndarray
    attention_score: float
    bbox: tuple[int, int, int, int]
    region_id: int
    metadata: dict = field(default_factory=dict)


class AttentionRegionExtractor:
    """Extract important regions using attention maps."""

    def __init__(
        self,
        encoder,
        target_size: int = 224,
        num_regions: int = 5,
        attention_threshold: float = 0.3,
        min_region_size: int = 32,
    ):
        """
        Initialize attention region extractor.

        Args:
            encoder: Image embedding model (ideally with attention output)
            target_size: Model input size
            num_regions: Maximum number of regions to extract
            attention_threshold: Minimum attention threshold
            min_region_size: Minimum region size in pixels
        """
        self.encoder = encoder
        self.target_size = target_size
        self.num_regions = num_regions
        self.attention_threshold = attention_threshold
        self.min_region_size = min_region_size

    def extract_attention_regions(
        self,
        image: Image.Image,
        attention_map: np.ndarray | None = None,
    ) -> list[AttentionRegion]:
        """
        Extract regions based on attention.

        Args:
            image: Input image
            attention_map: Pre-computed attention map (optional)

        Returns:
            List of high-attention regions
        """
        if attention_map is None:
            # Generate attention map using gradient-based method
            attention_map = self._compute_attention_map(image)

        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (
            attention_map.max() - attention_map.min() + 1e-8
        )

        # Find high-attention regions
        regions = self._find_attention_peaks(image, attention_map)

        return regions

    def embed_attention_regions(
        self,
        image: Image.Image,
        attention_map: np.ndarray | None = None,
    ) -> list[AttentionEmbedding]:
        """
        Extract and embed attention-based regions.

        Args:
            image: Input image
            attention_map: Pre-computed attention map (optional)

        Returns:
            List of embeddings for attention regions
        """
        regions = self.extract_attention_regions(image, attention_map)
        embeddings = []

        for region in regions:
            # Resize region for embedding
            resized = region.image.resize(
                (self.target_size, self.target_size),
                Image.Resampling.LANCZOS,
            )

            # Generate embedding
            embedding = self.encoder.encode(resized)

            embeddings.append(
                AttentionEmbedding(
                    embedding=embedding,
                    attention_score=region.attention_score,
                    bbox=region.bbox,
                    region_id=region.region_id,
                    metadata={
                        "original_size": region.image.size,
                    },
                )
            )

        return embeddings

    def _compute_attention_map(self, image: Image.Image) -> np.ndarray:
        """
        Compute attention map using simple saliency.

        In production, use model's actual attention weights.

        Args:
            image: Input image

        Returns:
            Attention map (H, W)
        """
        img_array = np.array(image.convert("RGB")).astype(np.float32)

        # Simple saliency based on color contrast
        # Production would extract attention from ViT or use GradCAM

        # Compute mean color
        mean_color = np.mean(img_array, axis=(0, 1))

        # Distance from mean color
        color_distance = np.sqrt(np.sum((img_array - mean_color) ** 2, axis=2))

        # Normalize
        saliency = color_distance / (color_distance.max() + 1e-8)

        # Apply Gaussian smoothing simulation
        from PIL import ImageFilter

        saliency_img = Image.fromarray((saliency * 255).astype(np.uint8))
        saliency_img = saliency_img.filter(ImageFilter.GaussianBlur(radius=5))
        saliency = np.array(saliency_img).astype(np.float32) / 255

        return saliency

    def _find_attention_peaks(
        self,
        image: Image.Image,
        attention_map: np.ndarray,
    ) -> list[AttentionRegion]:
        """
        Find peak attention regions.

        Args:
            image: Original image
            attention_map: Normalized attention map

        Returns:
            List of AttentionRegion objects
        """
        height, width = attention_map.shape
        regions = []

        # Find connected components (simplified)
        # Production would use scipy.ndimage.label

        # Grid-based region finding
        grid_size = max(self.min_region_size, min(height, width) // 8)

        region_candidates = []

        for y in range(0, height - grid_size, grid_size // 2):
            for x in range(0, width - grid_size, grid_size // 2):
                region_attention = attention_map[y : y + grid_size, x : x + grid_size]
                mean_attention = np.mean(region_attention)

                if mean_attention > self.attention_threshold:
                    region_candidates.append(
                        {
                            "bbox": (x, y, x + grid_size, y + grid_size),
                            "attention": mean_attention,
                        }
                    )

        # Sort by attention and take top regions
        region_candidates.sort(key=lambda r: r["attention"], reverse=True)

        # Merge overlapping regions and select top N
        selected_bboxes = []
        for candidate in region_candidates:
            if len(selected_bboxes) >= self.num_regions:
                break

            bbox = candidate["bbox"]

            # Check overlap with existing regions
            overlaps = False
            for existing in selected_bboxes:
                if self._boxes_overlap(bbox, existing):
                    overlaps = True
                    break

            if not overlaps:
                selected_bboxes.append(bbox)
                region_image = image.crop(bbox)

                regions.append(
                    AttentionRegion(
                        image=region_image,
                        attention_score=candidate["attention"],
                        bbox=bbox,
                        region_id=len(regions),
                    )
                )

        return regions

    def _boxes_overlap(
        self,
        box1: tuple[int, int, int, int],
        box2: tuple[int, int, int, int],
        threshold: float = 0.3,
    ) -> bool:
        """Check if two boxes overlap significantly."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return False

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # IoU
        iou = intersection / (area1 + area2 - intersection)

        return iou > threshold


def extract_vit_attention_regions(
    image: Image.Image,
    vit_model,
    encoder,
    num_regions: int = 5,
) -> list[AttentionEmbedding]:
    """
    Extract regions using ViT attention weights.

    Args:
        image: Input image
        vit_model: Vision Transformer model with attention output
        encoder: Embedding encoder
        num_regions: Number of regions to extract

    Returns:
        List of attention-based embeddings
    """
    # This would extract actual attention weights from ViT
    # Simplified version uses computed saliency

    extractor = AttentionRegionExtractor(
        encoder=encoder,
        num_regions=num_regions,
    )

    return extractor.embed_attention_regions(image)
