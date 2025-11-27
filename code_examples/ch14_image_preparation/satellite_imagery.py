"""Satellite and aerial imagery embedding strategies."""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class GeoTile:
    """A georeferenced image tile."""

    image: np.ndarray
    bounds: tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    crs: str = "EPSG:4326"
    tile_id: str = ""
    zoom_level: int = 0


@dataclass
class SatelliteEmbedding:
    """Embedding for a satellite image tile."""

    embedding: np.ndarray
    tile_id: str
    bounds: tuple[float, float, float, float]
    metadata: dict = field(default_factory=dict)


class SatelliteImageProcessor:
    """Process satellite imagery for embedding."""

    def __init__(
        self,
        encoder,
        tile_size: int = 256,
        overlap: float = 0.1,
        min_valid_ratio: float = 0.7,
    ):
        """
        Initialize satellite image processor.

        Args:
            encoder: Image embedding model
            tile_size: Size of tiles in pixels
            overlap: Overlap ratio between tiles
            min_valid_ratio: Minimum ratio of valid (non-nodata) pixels
        """
        self.encoder = encoder
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_valid_ratio = min_valid_ratio

    def process_large_image(
        self,
        image: np.ndarray,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> list[SatelliteEmbedding]:
        """
        Process large satellite image into embedded tiles.

        Args:
            image: Large satellite image array (H, W, C)
            bounds: Geographic bounds (optional)

        Returns:
            List of SatelliteEmbedding for each valid tile
        """
        height, width = image.shape[:2]
        step = int(self.tile_size * (1 - self.overlap))

        embeddings = []
        tile_idx = 0

        for y in range(0, height - self.tile_size + 1, step):
            for x in range(0, width - self.tile_size + 1, step):
                # Extract tile
                tile = image[y : y + self.tile_size, x : x + self.tile_size]

                # Check for valid data
                if not self._is_valid_tile(tile):
                    continue

                # Calculate tile bounds if geographic bounds provided
                tile_bounds = None
                if bounds is not None:
                    tile_bounds = self._calculate_tile_bounds(bounds, (width, height), (x, y))

                # Preprocess and embed
                processed = self._preprocess_satellite_tile(tile)
                embedding = self.encoder.encode(processed)

                embeddings.append(
                    SatelliteEmbedding(
                        embedding=embedding,
                        tile_id=f"tile_{tile_idx}",
                        bounds=tile_bounds or (x, y, x + self.tile_size, y + self.tile_size),
                        metadata={
                            "pixel_x": x,
                            "pixel_y": y,
                            "size": self.tile_size,
                        },
                    )
                )
                tile_idx += 1

        return embeddings

    def _is_valid_tile(self, tile: np.ndarray) -> bool:
        """Check if tile has enough valid data."""
        # Check for nodata (typically 0 or 255 for all bands)
        if len(tile.shape) == 3:
            # RGB or multispectral
            valid_pixels = np.all(tile > 0, axis=2) & np.all(tile < 255, axis=2)
        else:
            valid_pixels = (tile > 0) & (tile < 255)

        valid_ratio = np.mean(valid_pixels)
        return valid_ratio >= self.min_valid_ratio

    def _calculate_tile_bounds(
        self,
        image_bounds: tuple[float, float, float, float],
        image_size: tuple[int, int],
        tile_origin: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        """Calculate geographic bounds for a tile."""
        min_x, min_y, max_x, max_y = image_bounds
        width, height = image_size
        tx, ty = tile_origin

        # Calculate pixel size
        px_width = (max_x - min_x) / width
        px_height = (max_y - min_y) / height

        # Calculate tile bounds
        tile_min_x = min_x + tx * px_width
        tile_max_x = tile_min_x + self.tile_size * px_width
        tile_max_y = max_y - ty * px_height
        tile_min_y = tile_max_y - self.tile_size * px_height

        return (tile_min_x, tile_min_y, tile_max_x, tile_max_y)

    def _preprocess_satellite_tile(self, tile: np.ndarray) -> Image.Image:
        """Preprocess satellite tile for embedding model."""
        # Handle different band configurations
        if len(tile.shape) == 2:
            # Single band - convert to RGB
            tile = np.stack([tile] * 3, axis=2)
        elif tile.shape[2] > 3:
            # Multispectral - use RGB bands (typically 0, 1, 2)
            tile = tile[:, :, :3]

        # Normalize to 0-255 uint8
        if tile.dtype != np.uint8:
            tile = ((tile - tile.min()) / (tile.max() - tile.min()) * 255).astype(np.uint8)

        # Apply histogram stretching for better contrast
        tile = self._histogram_stretch(tile)

        return Image.fromarray(tile)

    def _histogram_stretch(
        self,
        image: np.ndarray,
        percentile_low: float = 2,
        percentile_high: float = 98,
    ) -> np.ndarray:
        """Apply percentile-based histogram stretching."""
        result = np.zeros_like(image)

        for i in range(image.shape[2]):
            band = image[:, :, i]
            p_low = np.percentile(band, percentile_low)
            p_high = np.percentile(band, percentile_high)

            stretched = np.clip((band - p_low) / (p_high - p_low) * 255, 0, 255)
            result[:, :, i] = stretched.astype(np.uint8)

        return result


def create_spatial_index(
    embeddings: list[SatelliteEmbedding],
) -> dict:
    """
    Create spatial index for efficient geographic queries.

    Args:
        embeddings: List of georeferenced embeddings

    Returns:
        Spatial index structure
    """
    # Simple grid-based index (production would use R-tree)
    index = {"tiles": {}, "bounds": None}

    for emb in embeddings:
        index["tiles"][emb.tile_id] = {
            "embedding": emb.embedding,
            "bounds": emb.bounds,
            "metadata": emb.metadata,
        }

        # Track overall bounds
        if index["bounds"] is None:
            index["bounds"] = list(emb.bounds)
        else:
            index["bounds"][0] = min(index["bounds"][0], emb.bounds[0])
            index["bounds"][1] = min(index["bounds"][1], emb.bounds[1])
            index["bounds"][2] = max(index["bounds"][2], emb.bounds[2])
            index["bounds"][3] = max(index["bounds"][3], emb.bounds[3])

    return index
