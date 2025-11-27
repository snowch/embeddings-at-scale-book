"""Domain-specific augmentation strategies."""

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass
class DomainAugmentationConfig:
    """Configuration for domain-specific augmentation."""

    domain: str
    flip_horizontal: bool = True
    flip_vertical: bool = False
    rotation_range: tuple[float, float] = (-15, 15)
    scale_range: tuple[float, float] = (0.8, 1.2)
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    blur_prob: float = 0.1
    noise_prob: float = 0.1
    custom_transforms: list = None


class DomainAugmenter:
    """Apply domain-appropriate augmentations."""

    def __init__(self, config: DomainAugmentationConfig | None = None):
        """
        Initialize domain augmenter.

        Args:
            config: Domain-specific configuration
        """
        self.config = config or DomainAugmentationConfig(domain="general")

    def augment(self, image: Image.Image) -> Image.Image:
        """
        Apply domain-appropriate augmentations.

        Args:
            image: Input image

        Returns:
            Augmented image
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        augmented = image.copy()

        # Geometric transforms
        augmented = self._apply_geometric(augmented)

        # Photometric transforms
        augmented = self._apply_photometric(augmented)

        # Domain-specific transforms
        if self.config.custom_transforms:
            for transform in self.config.custom_transforms:
                augmented = transform(augmented)

        return augmented

    def _apply_geometric(self, image: Image.Image) -> Image.Image:
        """Apply geometric augmentations."""
        # Horizontal flip
        if self.config.flip_horizontal and random.random() < 0.5:
            image = ImageOps.mirror(image)

        # Vertical flip
        if self.config.flip_vertical and random.random() < 0.5:
            image = ImageOps.flip(image)

        # Rotation
        if self.config.rotation_range:
            angle = random.uniform(*self.config.rotation_range)
            image = image.rotate(angle, fillcolor=(128, 128, 128), expand=False)

        # Scale (zoom)
        if self.config.scale_range:
            scale = random.uniform(*self.config.scale_range)
            w, h = image.size
            new_w, new_h = int(w * scale), int(h * scale)

            if scale > 1:
                # Zoom in: resize up then crop center
                scaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                image = scaled.crop((left, top, left + w, top + h))
            else:
                # Zoom out: resize down then pad
                scaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                padded = Image.new("RGB", (w, h), (128, 128, 128))
                paste_x = (w - new_w) // 2
                paste_y = (h - new_h) // 2
                padded.paste(scaled, (paste_x, paste_y))
                image = padded

        return image

    def _apply_photometric(self, image: Image.Image) -> Image.Image:
        """Apply photometric augmentations."""
        # Brightness
        if self.config.brightness_range:
            factor = random.uniform(*self.config.brightness_range)
            image = ImageEnhance.Brightness(image).enhance(factor)

        # Contrast
        if self.config.contrast_range:
            factor = random.uniform(*self.config.contrast_range)
            image = ImageEnhance.Contrast(image).enhance(factor)

        # Blur
        if random.random() < self.config.blur_prob:
            radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        # Noise
        if random.random() < self.config.noise_prob:
            image = self._add_noise(image)

        return image

    def _add_noise(self, image: Image.Image, std: float = 10) -> Image.Image:
        """Add Gaussian noise."""
        img_array = np.array(image).astype(np.float32)
        noise = np.random.normal(0, std, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)


def create_medical_augmenter() -> DomainAugmenter:
    """Create augmenter for medical imaging."""
    config = DomainAugmentationConfig(
        domain="medical",
        flip_horizontal=True,
        flip_vertical=True,  # Valid for many medical images
        rotation_range=(-180, 180),  # Full rotation often valid
        scale_range=(0.9, 1.1),  # Conservative scaling
        brightness_range=(0.9, 1.1),  # Conservative brightness
        contrast_range=(0.9, 1.1),
        blur_prob=0.05,  # Less blur to preserve detail
        noise_prob=0.1,
    )
    return DomainAugmenter(config)


def create_satellite_augmenter() -> DomainAugmenter:
    """Create augmenter for satellite imagery."""
    config = DomainAugmentationConfig(
        domain="satellite",
        flip_horizontal=True,
        flip_vertical=True,  # Satellite images can be flipped
        rotation_range=(-180, 180),  # Full rotation valid
        scale_range=(0.8, 1.2),
        brightness_range=(0.7, 1.3),  # Handle varying lighting
        contrast_range=(0.8, 1.2),
        blur_prob=0.1,  # Atmospheric effects
        noise_prob=0.15,  # Sensor noise
    )
    return DomainAugmenter(config)


def create_retail_augmenter() -> DomainAugmenter:
    """Create augmenter for product/retail images."""
    config = DomainAugmentationConfig(
        domain="retail",
        flip_horizontal=True,
        flip_vertical=False,  # Products have orientation
        rotation_range=(-15, 15),  # Limited rotation
        scale_range=(0.85, 1.15),
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        blur_prob=0.1,
        noise_prob=0.05,
    )
    return DomainAugmenter(config)


def create_document_augmenter() -> DomainAugmenter:
    """Create augmenter for document images."""
    config = DomainAugmentationConfig(
        domain="document",
        flip_horizontal=False,  # Documents have orientation
        flip_vertical=False,
        rotation_range=(-5, 5),  # Slight rotation (skew)
        scale_range=(0.95, 1.05),  # Very limited scaling
        brightness_range=(0.85, 1.15),
        contrast_range=(0.9, 1.1),
        blur_prob=0.15,  # Simulate focus issues
        noise_prob=0.1,  # Scanning artifacts
    )
    return DomainAugmenter(config)


def create_face_augmenter() -> DomainAugmenter:
    """Create augmenter for face images."""
    config = DomainAugmentationConfig(
        domain="face",
        flip_horizontal=True,  # Faces can be mirrored
        flip_vertical=False,  # Upside-down faces not useful
        rotation_range=(-20, 20),  # Head tilt
        scale_range=(0.9, 1.1),
        brightness_range=(0.7, 1.3),  # Lighting variation
        contrast_range=(0.8, 1.2),
        blur_prob=0.1,
        noise_prob=0.05,
    )
    return DomainAugmenter(config)


# Factory function
def get_domain_augmenter(domain: str) -> DomainAugmenter:
    """
    Get appropriate augmenter for domain.

    Args:
        domain: Domain name ('medical', 'satellite', 'retail', 'document', 'face')

    Returns:
        Configured DomainAugmenter
    """
    factories = {
        "medical": create_medical_augmenter,
        "satellite": create_satellite_augmenter,
        "retail": create_retail_augmenter,
        "document": create_document_augmenter,
        "face": create_face_augmenter,
    }

    factory = factories.get(domain)
    if factory:
        return factory()

    # Default general augmenter
    return DomainAugmenter(DomainAugmentationConfig(domain=domain))
