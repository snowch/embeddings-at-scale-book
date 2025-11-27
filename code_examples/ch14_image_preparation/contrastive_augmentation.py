"""Augmentation strategies for contrastive learning."""

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass
class AugmentationPair:
    """A pair of augmented views for contrastive learning."""

    view1: Image.Image
    view2: Image.Image
    original_id: str = ""


class ContrastiveAugmenter:
    """Generate augmented view pairs for contrastive learning."""

    def __init__(
        self,
        target_size: int = 224,
        strength: str = "medium",
    ):
        """
        Initialize contrastive augmenter.

        Args:
            target_size: Output image size
            strength: Augmentation strength ('weak', 'medium', 'strong')
        """
        self.target_size = target_size
        self.strength = strength
        self.augmentations = self._get_augmentations(strength)

    def _get_augmentations(self, strength: str) -> list[Callable]:
        """Get augmentation functions based on strength."""
        base_augs = [
            self._random_crop_resize,
            self._horizontal_flip,
            self._color_jitter,
        ]

        if strength in ("medium", "strong"):
            base_augs.extend(
                [
                    self._grayscale,
                    self._gaussian_blur,
                ]
            )

        if strength == "strong":
            base_augs.extend(
                [
                    self._solarize,
                    self._random_rotation,
                ]
            )

        return base_augs

    def generate_pair(
        self,
        image: Image.Image,
        image_id: str = "",
    ) -> AugmentationPair:
        """
        Generate two augmented views of the same image.

        Args:
            image: Original image
            image_id: Image identifier

        Returns:
            AugmentationPair with two different views
        """
        view1 = self._apply_augmentations(image)
        view2 = self._apply_augmentations(image)

        return AugmentationPair(view1=view1, view2=view2, original_id=image_id)

    def _apply_augmentations(self, image: Image.Image) -> Image.Image:
        """Apply random augmentations to create a view."""
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        augmented = image.copy()

        # Always apply random crop resize
        augmented = self._random_crop_resize(augmented)

        # Apply other augmentations with probability
        for aug_fn in self.augmentations[1:]:  # Skip crop which was already applied
            if random.random() < 0.5:
                augmented = aug_fn(augmented)

        return augmented

    def _random_crop_resize(self, image: Image.Image) -> Image.Image:
        """Random crop and resize to target size."""
        # Random scale between 0.2 and 1.0 of original area
        min_scale = 0.2 if self.strength == "strong" else 0.5
        scale = random.uniform(min_scale, 1.0)

        # Calculate crop size
        width, height = image.size
        crop_size = int(min(width, height) * np.sqrt(scale))

        # Random crop location
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)

        cropped = image.crop((left, top, left + crop_size, top + crop_size))

        # Resize to target
        return cropped.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)

    def _horizontal_flip(self, image: Image.Image) -> Image.Image:
        """Random horizontal flip."""
        if random.random() < 0.5:
            return ImageOps.mirror(image)
        return image

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        """Random color jittering."""
        # Brightness
        factor = random.uniform(0.6, 1.4)
        image = ImageEnhance.Brightness(image).enhance(factor)

        # Contrast
        factor = random.uniform(0.6, 1.4)
        image = ImageEnhance.Contrast(image).enhance(factor)

        # Saturation
        factor = random.uniform(0.6, 1.4)
        image = ImageEnhance.Color(image).enhance(factor)

        # Hue shift (simplified via channel shuffle)
        if random.random() < 0.1:
            img_array = np.array(image)
            channels = list(range(3))
            random.shuffle(channels)
            img_array = img_array[:, :, channels]
            image = Image.fromarray(img_array)

        return image

    def _grayscale(self, image: Image.Image) -> Image.Image:
        """Convert to grayscale with probability."""
        if random.random() < 0.2:
            gray = image.convert("L")
            return gray.convert("RGB")
        return image

    def _gaussian_blur(self, image: Image.Image) -> Image.Image:
        """Apply Gaussian blur."""
        if random.random() < 0.5:
            radius = random.uniform(0.1, 2.0)
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image

    def _solarize(self, image: Image.Image) -> Image.Image:
        """Apply solarization."""
        if random.random() < 0.2:
            threshold = random.randint(128, 255)
            return ImageOps.solarize(image, threshold=threshold)
        return image

    def _random_rotation(self, image: Image.Image) -> Image.Image:
        """Apply random rotation."""
        if random.random() < 0.3:
            angle = random.uniform(-30, 30)
            return image.rotate(angle, fillcolor=(128, 128, 128))
        return image


@dataclass
class SimCLRAugmentation:
    """SimCLR-style augmentation configuration."""

    crop_scale: tuple[float, float] = (0.08, 1.0)
    crop_ratio: tuple[float, float] = (0.75, 1.33)
    color_jitter_prob: float = 0.8
    grayscale_prob: float = 0.2
    blur_prob: float = 0.5
    blur_sigma: tuple[float, float] = (0.1, 2.0)


def create_simclr_views(
    image: Image.Image,
    config: SimCLRAugmentation | None = None,
    target_size: int = 224,
) -> tuple[Image.Image, Image.Image]:
    """
    Create SimCLR-style augmented view pair.

    Args:
        image: Original image
        config: Augmentation configuration
        target_size: Output size

    Returns:
        Tuple of two augmented views
    """
    config = config or SimCLRAugmentation()

    def apply_simclr_aug(img: Image.Image) -> Image.Image:
        # Random resized crop
        scale = random.uniform(*config.crop_scale)
        ratio = random.uniform(*config.crop_ratio)

        width, height = img.size
        area = width * height * scale

        w = int(np.sqrt(area * ratio))
        h = int(np.sqrt(area / ratio))

        w = min(w, width)
        h = min(h, height)

        left = random.randint(0, width - w)
        top = random.randint(0, height - h)

        cropped = img.crop((left, top, left + w, top + h))
        resized = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

        # Horizontal flip
        if random.random() < 0.5:
            resized = ImageOps.mirror(resized)

        # Color jitter
        if random.random() < config.color_jitter_prob:
            resized = ImageEnhance.Brightness(resized).enhance(random.uniform(0.6, 1.4))
            resized = ImageEnhance.Contrast(resized).enhance(random.uniform(0.6, 1.4))
            resized = ImageEnhance.Color(resized).enhance(random.uniform(0.6, 1.4))

        # Grayscale
        if random.random() < config.grayscale_prob:
            resized = resized.convert("L").convert("RGB")

        # Gaussian blur
        if random.random() < config.blur_prob:
            sigma = random.uniform(*config.blur_sigma)
            resized = resized.filter(ImageFilter.GaussianBlur(radius=sigma))

        return resized

    if image.mode != "RGB":
        image = image.convert("RGB")

    view1 = apply_simclr_aug(image)
    view2 = apply_simclr_aug(image)

    return view1, view2
