"""Handle different image resolutions and aspect ratios."""

from typing import List, Tuple, Optional
import numpy as np
from enum import Enum


class ResizeStrategy(Enum):
    """Available resize strategies."""
    STRETCH = "stretch"           # Resize ignoring aspect ratio
    CENTER_CROP = "center_crop"   # Resize then crop center
    RANDOM_CROP = "random_crop"   # Resize then random crop (for training)
    PAD = "pad"                   # Resize with padding to preserve aspect ratio
    MULTI_CROP = "multi_crop"     # Multiple crops for ensemble embedding


def resize_for_embedding(
    image,
    target_size: Tuple[int, int] = (224, 224),
    strategy: ResizeStrategy = ResizeStrategy.CENTER_CROP
) -> np.ndarray:
    """
    Resize image using specified strategy.

    Args:
        image: PIL Image or numpy array
        target_size: Target (width, height)
        strategy: How to handle aspect ratio

    Returns:
        Resized image as numpy array
    """
    from PIL import Image

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if strategy == ResizeStrategy.STRETCH:
        return np.array(image.resize(target_size))

    elif strategy == ResizeStrategy.CENTER_CROP:
        return _center_crop(image, target_size)

    elif strategy == ResizeStrategy.PAD:
        return _resize_with_pad(image, target_size)

    elif strategy == ResizeStrategy.MULTI_CROP:
        # Returns multiple crops - handled separately
        raise ValueError("Use multi_crop_embedding() for MULTI_CROP strategy")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _center_crop(image, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize and center crop."""
    from PIL import Image

    w, h = image.size
    target_w, target_h = target_size

    # Resize so smaller dimension matches, then crop
    scale = max(target_w / w, target_h / h)
    new_size = (int(w * scale), int(h * scale))
    image = image.resize(new_size)

    # Center crop
    new_w, new_h = image.size
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    cropped = image.crop((left, top, left + target_w, top + target_h))

    return np.array(cropped)


def _resize_with_pad(
    image,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """Resize preserving aspect ratio, pad to target size."""
    from PIL import Image

    w, h = image.size
    target_w, target_h = target_size

    # Calculate size that fits within target
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize
    image = image.resize((new_w, new_h))

    # Create padded canvas
    padded = Image.new('RGB', target_size, pad_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded.paste(image, (paste_x, paste_y))

    return np.array(padded)


def multi_crop_embedding(
    image,
    encoder,
    target_size: Tuple[int, int] = (224, 224),
    num_crops: int = 5,
    aggregation: str = 'mean'
) -> np.ndarray:
    """
    Create embedding from multiple crops of the image.

    Crops include:
    - 4 corners
    - Center
    - Optionally: random crops

    Args:
        image: PIL Image
        encoder: Embedding model with encode() method
        target_size: Size for each crop
        num_crops: Number of crops (1=center, 5=corners+center, 10=+flips)
        aggregation: How to combine embeddings ('mean', 'max', 'concat')

    Returns:
        Aggregated embedding
    """
    from PIL import Image

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    crops = []

    # Resize to slightly larger than target
    w, h = image.size
    target_w, target_h = target_size
    scale = max(target_w / w, target_h / h) * 1.14  # ~256/224
    new_size = (int(w * scale), int(h * scale))
    image = image.resize(new_size)
    w, h = image.size

    # Center crop
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    crops.append(image.crop((left, top, left + target_w, top + target_h)))

    if num_crops >= 5:
        # Four corners
        crops.append(image.crop((0, 0, target_w, target_h)))  # Top-left
        crops.append(image.crop((w - target_w, 0, w, target_h)))  # Top-right
        crops.append(image.crop((0, h - target_h, target_w, h)))  # Bottom-left
        crops.append(image.crop((w - target_w, h - target_h, w, h)))  # Bottom-right

    if num_crops >= 10:
        # Horizontal flips of each crop
        crops.extend([c.transpose(Image.FLIP_LEFT_RIGHT) for c in crops[:5]])

    # Get embeddings
    embeddings = encoder.encode(crops)

    # Aggregate
    if aggregation == 'mean':
        return np.mean(embeddings, axis=0)
    elif aggregation == 'max':
        return np.max(embeddings, axis=0)
    elif aggregation == 'concat':
        return embeddings.flatten()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def adaptive_resize(
    image,
    max_size: int = 1024,
    min_size: int = 224
) -> np.ndarray:
    """
    Adaptively resize image based on its dimensions.

    Preserves aspect ratio while ensuring:
    - Neither dimension exceeds max_size
    - Neither dimension is smaller than min_size

    Args:
        image: PIL Image
        max_size: Maximum dimension
        min_size: Minimum dimension

    Returns:
        Resized image
    """
    from PIL import Image

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    w, h = image.size

    # Check if resize needed
    if max(w, h) <= max_size and min(w, h) >= min_size:
        return np.array(image)

    # Calculate scale
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
    elif min(w, h) < min_size:
        scale = min_size / min(w, h)
    else:
        return np.array(image)

    new_w, new_h = int(w * scale), int(h * scale)
    return np.array(image.resize((new_w, new_h)))


# Example usage
if __name__ == "__main__":
    from PIL import Image

    # Create sample non-square image
    img_array = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
    sample_image = Image.fromarray(img_array)
    print(f"Original size: {sample_image.size}")

    print("\nResize strategies comparison:")
    print("-" * 50)

    for strategy in ResizeStrategy:
        if strategy == ResizeStrategy.MULTI_CROP:
            continue  # Needs encoder

        result = resize_for_embedding(sample_image, (224, 224), strategy)
        print(f"{strategy.value:15} -> shape: {result.shape}")

    # Adaptive resize examples
    print("\nAdaptive resize examples:")
    print("-" * 50)

    for size in [(100, 80), (500, 400), (2000, 1500)]:
        img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
        result = adaptive_resize(img)
        print(f"  {size} -> {result.shape[:2]}")
