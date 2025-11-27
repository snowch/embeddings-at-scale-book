"""Tiling strategies for large images."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Tile:
    """A tile extracted from a larger image."""
    image: np.ndarray
    x: int  # Top-left x coordinate in original image
    y: int  # Top-left y coordinate in original image
    width: int
    height: int
    row: int
    col: int


def tile_image(
    image,
    tile_size: Tuple[int, int] = (224, 224),
    overlap: float = 0.1,
    min_coverage: float = 0.5
) -> List[Tile]:
    """
    Split a large image into overlapping tiles.

    Args:
        image: PIL Image or numpy array
        tile_size: Size of each tile (width, height)
        overlap: Fraction of overlap between adjacent tiles (0-1)
        min_coverage: Minimum fraction of tile that must contain image data

    Returns:
        List of Tile objects
    """
    from PIL import Image

    if isinstance(image, Image.Image):
        image = np.array(image)

    h, w = image.shape[:2]
    tile_w, tile_h = tile_size

    # Calculate stride
    stride_x = int(tile_w * (1 - overlap))
    stride_y = int(tile_h * (1 - overlap))

    tiles = []
    row = 0

    y = 0
    while y < h:
        col = 0
        x = 0
        while x < w:
            # Extract tile
            x_end = min(x + tile_w, w)
            y_end = min(y + tile_h, h)

            tile_img = image[y:y_end, x:x_end]

            # Check coverage (for edge tiles)
            coverage = (tile_img.shape[0] * tile_img.shape[1]) / (tile_w * tile_h)

            if coverage >= min_coverage:
                # Pad if necessary
                if tile_img.shape[0] < tile_h or tile_img.shape[1] < tile_w:
                    padded = np.zeros((tile_h, tile_w, image.shape[2]), dtype=image.dtype)
                    padded[:tile_img.shape[0], :tile_img.shape[1]] = tile_img
                    tile_img = padded

                tiles.append(Tile(
                    image=tile_img,
                    x=x,
                    y=y,
                    width=x_end - x,
                    height=y_end - y,
                    row=row,
                    col=col
                ))

            x += stride_x
            col += 1

        y += stride_y
        row += 1

    return tiles


def embed_tiled_image(
    image,
    encoder,
    tile_size: Tuple[int, int] = (224, 224),
    overlap: float = 0.1,
    aggregation: str = 'mean'
) -> np.ndarray:
    """
    Create embedding for large image using tiling.

    Args:
        image: Large PIL Image or numpy array
        encoder: Embedding model with encode() method
        tile_size: Size of each tile
        overlap: Overlap fraction
        aggregation: How to combine tile embeddings ('mean', 'max', 'weighted')

    Returns:
        Aggregated embedding vector
    """
    from PIL import Image

    tiles = tile_image(image, tile_size, overlap)

    if not tiles:
        raise ValueError("No tiles extracted from image")

    # Convert tiles to PIL for encoder
    tile_images = [Image.fromarray(t.image) for t in tiles]

    # Get embeddings for all tiles
    embeddings = encoder.encode(tile_images)

    # Aggregate
    if aggregation == 'mean':
        return np.mean(embeddings, axis=0)
    elif aggregation == 'max':
        return np.max(embeddings, axis=0)
    elif aggregation == 'weighted':
        # Weight by tile coverage and distance from center
        weights = []
        h, w = image.shape[:2] if isinstance(image, np.ndarray) else image.size[::-1]
        center_x, center_y = w / 2, h / 2

        for tile in tiles:
            # Coverage weight
            coverage = (tile.width * tile.height) / (tile_size[0] * tile_size[1])

            # Distance from center weight (closer = higher weight)
            tile_center_x = tile.x + tile.width / 2
            tile_center_y = tile.y + tile.height / 2
            distance = np.sqrt((tile_center_x - center_x)**2 + (tile_center_y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            distance_weight = 1 - (distance / max_distance)

            weights.append(coverage * distance_weight)

        weights = np.array(weights)
        weights = weights / weights.sum()

        return np.average(embeddings, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def reconstruct_from_tiles(
    tiles: List[Tile],
    original_size: Tuple[int, int]
) -> np.ndarray:
    """
    Reconstruct image from tiles (for visualization/debugging).

    Uses averaging in overlap regions.
    """
    h, w = original_size
    channels = tiles[0].image.shape[2] if len(tiles[0].image.shape) == 3 else 1

    # Accumulator and count for averaging
    accumulated = np.zeros((h, w, channels), dtype=np.float32)
    counts = np.zeros((h, w, 1), dtype=np.float32)

    for tile in tiles:
        x, y = tile.x, tile.y
        tile_h, tile_w = tile.height, tile.width

        accumulated[y:y+tile_h, x:x+tile_w] += tile.image[:tile_h, :tile_w].astype(np.float32)
        counts[y:y+tile_h, x:x+tile_w] += 1

    # Average where tiles overlap
    counts = np.maximum(counts, 1)  # Avoid division by zero
    result = (accumulated / counts).astype(np.uint8)

    return result


class TiledImageProcessor:
    """
    Process large images using tiling with configurable strategies.
    """

    def __init__(
        self,
        tile_size: Tuple[int, int] = (224, 224),
        overlap: float = 0.1,
        max_tiles: Optional[int] = None,
        selection_strategy: str = 'all'  # 'all', 'grid', 'important'
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.max_tiles = max_tiles
        self.selection_strategy = selection_strategy

    def process(self, image, encoder) -> dict:
        """
        Process a large image and return embeddings with metadata.
        """
        from PIL import Image

        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image

        # Get all tiles
        all_tiles = tile_image(image_array, self.tile_size, self.overlap)

        # Select tiles based on strategy
        if self.selection_strategy == 'grid' and self.max_tiles:
            selected_tiles = self._select_grid(all_tiles)
        elif self.selection_strategy == 'important':
            selected_tiles = self._select_important(all_tiles)
        else:
            selected_tiles = all_tiles

        # Limit if needed
        if self.max_tiles and len(selected_tiles) > self.max_tiles:
            selected_tiles = selected_tiles[:self.max_tiles]

        # Embed
        tile_images = [Image.fromarray(t.image) for t in selected_tiles]
        embeddings = encoder.encode(tile_images)

        return {
            'tile_embeddings': embeddings,
            'tile_positions': [(t.x, t.y, t.width, t.height) for t in selected_tiles],
            'aggregate_embedding': np.mean(embeddings, axis=0),
            'num_tiles': len(selected_tiles),
            'original_size': image_array.shape[:2]
        }

    def _select_grid(self, tiles: List[Tile]) -> List[Tile]:
        """Select tiles in a uniform grid pattern."""
        if not tiles:
            return tiles

        max_row = max(t.row for t in tiles)
        max_col = max(t.col for t in tiles)

        # Calculate step to get approximately max_tiles
        if self.max_tiles:
            total_tiles = (max_row + 1) * (max_col + 1)
            step = max(1, int(np.sqrt(total_tiles / self.max_tiles)))
        else:
            step = 1

        selected = [t for t in tiles if t.row % step == 0 and t.col % step == 0]
        return selected

    def _select_important(self, tiles: List[Tile]) -> List[Tile]:
        """Select tiles with highest variance (most information)."""
        # Score tiles by variance
        scored = []
        for tile in tiles:
            variance = np.var(tile.image)
            scored.append((variance, tile))

        # Sort by variance descending
        scored.sort(key=lambda x: x[0], reverse=True)

        n = self.max_tiles or len(tiles)
        return [t for _, t in scored[:n]]


# Example usage
if __name__ == "__main__":

    # Create a large sample image
    large_image = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)
    print(f"Original image size: {large_image.shape}")

    # Tile the image
    tiles = tile_image(large_image, tile_size=(224, 224), overlap=0.1)
    print(f"Number of tiles: {len(tiles)}")

    # Show tile grid
    rows = max(t.row for t in tiles) + 1
    cols = max(t.col for t in tiles) + 1
    print(f"Tile grid: {rows} x {cols}")

    # Show first few tiles
    print("\nFirst 5 tiles:")
    for tile in tiles[:5]:
        print(f"  Tile ({tile.row}, {tile.col}): pos=({tile.x}, {tile.y}), size={tile.image.shape}")

    # Test reconstruction
    reconstructed = reconstruct_from_tiles(tiles, (1000, 800))
    print(f"\nReconstructed shape: {reconstructed.shape}")
