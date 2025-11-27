"""Standard image preprocessing pipeline for embeddings."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PreprocessConfig:
    """Configuration for image preprocessing."""

    target_size: Tuple[int, int] = (224, 224)
    resize_method: str = "resize"  # 'resize', 'crop', 'pad'
    normalize: bool = True
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    color_mode: str = "RGB"


class ImagePreprocessor:
    """
    Standard preprocessing pipeline for image embeddings.

    Handles:
    - Resizing/cropping/padding
    - Color space conversion
    - Normalization
    - Tensor conversion
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def preprocess(self, image) -> np.ndarray:
        """
        Preprocess a single image.

        Args:
            image: PIL Image or numpy array

        Returns:
            Preprocessed numpy array ready for model input
        """
        from PIL import Image

        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Color space conversion
        if self.config.color_mode == "RGB" and image.mode != "RGB":
            image = image.convert("RGB")
        elif self.config.color_mode == "L" and image.mode != "L":
            image = image.convert("L")

        # Resize/crop/pad
        if self.config.resize_method == "resize":
            image = self._resize(image)
        elif self.config.resize_method == "crop":
            image = self._center_crop(image)
        elif self.config.resize_method == "pad":
            image = self._resize_with_pad(image)

        # Convert to numpy
        img_array = np.array(image, dtype=np.float32)

        # Scale to [0, 1]
        if img_array.max() > 1.0:
            img_array = img_array / 255.0

        # Normalize
        if self.config.normalize:
            img_array = self._normalize(img_array)

        # Ensure channel dimension
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)

        return img_array

    def preprocess_batch(self, images: List) -> np.ndarray:
        """Preprocess a batch of images."""
        return np.stack([self.preprocess(img) for img in images])

    def _resize(self, image) -> "Image":
        """Simple resize to target size."""
        return image.resize(self.config.target_size)

    def _center_crop(self, image) -> "Image":
        """Resize then center crop to target size."""
        # First resize so smaller dimension matches target
        w, h = image.size
        target_w, target_h = self.config.target_size

        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h))

        # Center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        return image.crop((left, top, left + target_w, top + target_h))

    def _resize_with_pad(self, image) -> "Image":
        """Resize preserving aspect ratio with padding."""
        from PIL import Image as PILImage

        w, h = image.size
        target_w, target_h = self.config.target_size

        # Calculate scale to fit within target
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        image = image.resize((new_w, new_h))

        # Create padded image
        padded = PILImage.new(image.mode, self.config.target_size, (128, 128, 128))
        left = (target_w - new_w) // 2
        top = (target_h - new_h) // 2
        padded.paste(image, (left, top))

        return padded

    def _normalize(self, img_array: np.ndarray) -> np.ndarray:
        """Apply ImageNet-style normalization."""
        mean = np.array(self.config.mean)
        std = np.array(self.config.std)

        if len(img_array.shape) == 3 and img_array.shape[-1] == 3:
            return (img_array - mean) / std
        return img_array


def create_torchvision_transform(config: PreprocessConfig):
    """
    Create a torchvision transform from config.

    Useful for PyTorch DataLoader integration.
    """
    import torchvision.transforms as T

    transforms = []

    # Resize/crop
    if config.resize_method == "resize":
        transforms.append(T.Resize(config.target_size))
    elif config.resize_method == "crop":
        larger_size = max(config.target_size) + 32
        transforms.append(T.Resize(larger_size))
        transforms.append(T.CenterCrop(config.target_size))
    elif config.resize_method == "pad":
        # Custom padding transform
        transforms.append(T.Resize(config.target_size))

    transforms.append(T.ToTensor())

    if config.normalize:
        transforms.append(T.Normalize(mean=config.mean, std=config.std))

    return T.Compose(transforms)


# Example usage
if __name__ == "__main__":
    from PIL import Image

    # Create sample image with non-square aspect ratio
    img_array = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
    sample_image = Image.fromarray(img_array)
    print(f"Original size: {sample_image.size}")

    # Test different resize methods
    for method in ["resize", "crop", "pad"]:
        config = PreprocessConfig(target_size=(224, 224), resize_method=method, normalize=True)
        preprocessor = ImagePreprocessor(config)
        result = preprocessor.preprocess(sample_image)
        print(f"\n{method.upper()} method:")
        print(f"  Output shape: {result.shape}")
        print(f"  Value range: [{result.min():.3f}, {result.max():.3f}]")
