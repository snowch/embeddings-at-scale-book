"""Color space handling and normalization for image embeddings."""

from typing import Tuple

import numpy as np

# Standard normalization values for common models
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD
) -> np.ndarray:
    """
    Apply channel-wise normalization.

    Args:
        image: Image array with values in [0, 1] or [0, 255]
        mean: Per-channel mean values
        std: Per-channel standard deviation values

    Returns:
        Normalized image array
    """
    image = image.astype(np.float32)

    # Scale to [0, 1] if needed
    if image.max() > 1.0:
        image = image / 255.0

    # Apply normalization
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    return (image - mean) / std


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD
) -> np.ndarray:
    """
    Reverse normalization for visualization.
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image


def convert_color_space(
    image: np.ndarray,
    source: str = 'RGB',
    target: str = 'RGB'
) -> np.ndarray:
    """
    Convert between color spaces.

    Supported: RGB, BGR, GRAY, LAB, HSV
    """
    import cv2

    conversions = {
        ('RGB', 'BGR'): cv2.COLOR_RGB2BGR,
        ('BGR', 'RGB'): cv2.COLOR_BGR2RGB,
        ('RGB', 'GRAY'): cv2.COLOR_RGB2GRAY,
        ('BGR', 'GRAY'): cv2.COLOR_BGR2GRAY,
        ('GRAY', 'RGB'): cv2.COLOR_GRAY2RGB,
        ('RGB', 'LAB'): cv2.COLOR_RGB2LAB,
        ('LAB', 'RGB'): cv2.COLOR_LAB2RGB,
        ('RGB', 'HSV'): cv2.COLOR_RGB2HSV,
        ('HSV', 'RGB'): cv2.COLOR_HSV2RGB,
    }

    key = (source.upper(), target.upper())
    if key in conversions:
        return cv2.cvtColor(image, conversions[key])
    elif source == target:
        return image
    else:
        raise ValueError(f"Unsupported conversion: {source} -> {target}")


def histogram_equalization(
    image: np.ndarray,
    clip_limit: float = 2.0
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Useful for images with poor contrast or lighting.
    """
    import cv2

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to RGB
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def white_balance(
    image: np.ndarray,
    method: str = 'gray_world'
) -> np.ndarray:
    """
    Apply white balance correction.

    Methods:
    - gray_world: Assumes average color should be gray
    - max_white: Assumes brightest pixels are white
    """
    image = image.astype(np.float32)

    if method == 'gray_world':
        # Calculate average of each channel
        avg_r = np.mean(image[:, :, 0])
        avg_g = np.mean(image[:, :, 1])
        avg_b = np.mean(image[:, :, 2])
        avg_gray = (avg_r + avg_g + avg_b) / 3

        # Scale each channel
        image[:, :, 0] = image[:, :, 0] * (avg_gray / avg_r)
        image[:, :, 1] = image[:, :, 1] * (avg_gray / avg_g)
        image[:, :, 2] = image[:, :, 2] * (avg_gray / avg_b)

    elif method == 'max_white':
        # Use percentile to avoid outliers
        max_r = np.percentile(image[:, :, 0], 99)
        max_g = np.percentile(image[:, :, 1], 99)
        max_b = np.percentile(image[:, :, 2], 99)

        image[:, :, 0] = image[:, :, 0] * (255.0 / max_r)
        image[:, :, 1] = image[:, :, 1] * (255.0 / max_g)
        image[:, :, 2] = image[:, :, 2] * (255.0 / max_b)

    return np.clip(image, 0, 255).astype(np.uint8)


class ColorNormalizer:
    """
    Comprehensive color normalization for consistent embeddings.
    """

    def __init__(
        self,
        model_type: str = 'imagenet',
        apply_white_balance: bool = False,
        apply_histogram_eq: bool = False
    ):
        self.model_type = model_type
        self.apply_white_balance = apply_white_balance
        self.apply_histogram_eq = apply_histogram_eq

        # Set normalization values based on model
        if model_type == 'imagenet':
            self.mean = IMAGENET_MEAN
            self.std = IMAGENET_STD
        elif model_type == 'clip':
            self.mean = CLIP_MEAN
            self.std = CLIP_STD
        else:
            self.mean = (0.5, 0.5, 0.5)
            self.std = (0.5, 0.5, 0.5)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full normalization pipeline.
        """
        # Ensure RGB
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # Apply preprocessing
        if self.apply_white_balance:
            image = white_balance(image)

        if self.apply_histogram_eq:
            image = histogram_equalization(image)

        # Apply model-specific normalization
        return normalize_image(image, self.mean, self.std)


# Example usage
if __name__ == "__main__":
    # Create sample image
    sample = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    print("Color Normalization Demo")
    print("=" * 50)

    # Test normalization
    normalized = normalize_image(sample)
    print(f"Original range: [{sample.min()}, {sample.max()}]")
    print(f"Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")

    # Test denormalization roundtrip
    recovered = denormalize_image(normalized)
    diff = np.abs(sample.astype(float) - recovered.astype(float)).mean()
    print(f"Roundtrip difference: {diff:.2f}")

    # Test color normalizer
    normalizer = ColorNormalizer(
        model_type='imagenet',
        apply_white_balance=True
    )
    result = normalizer.normalize(sample)
    print(f"\nWith white balance: range [{result.min():.2f}, {result.max():.2f}]")
