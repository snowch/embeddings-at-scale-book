"""Image augmentation for training embedding models."""

from typing import Callable, List, Tuple

import numpy as np


def create_augmentation_pipeline(
    mode: str = 'standard',
    target_size: Tuple[int, int] = (224, 224)
) -> Callable:
    """
    Create an augmentation pipeline for embedding training.

    Modes:
    - standard: Basic augmentations for general use
    - contrastive: Strong augmentations for contrastive learning
    - minimal: Light augmentations for fine-tuning

    Returns:
        Augmentation function that takes PIL Image and returns augmented image
    """
    import torchvision.transforms as T

    if mode == 'standard':
        transform = T.Compose([
            T.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif mode == 'contrastive':
        # Stronger augmentations for SimCLR/MoCo style training
        transform = T.Compose([
            T.RandomResizedCrop(target_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif mode == 'minimal':
        transform = T.Compose([
            T.Resize(int(target_size[0] * 1.1)),
            T.CenterCrop(target_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return transform


def generate_positive_pairs(
    image,
    augment_fn: Callable,
    num_pairs: int = 2
) -> List:
    """
    Generate augmented positive pairs for contrastive learning.

    Args:
        image: Original PIL Image
        augment_fn: Augmentation function
        num_pairs: Number of augmented versions to generate

    Returns:
        List of augmented tensors (all are positive pairs of original)
    """
    return [augment_fn(image) for _ in range(num_pairs)]


class ContrastiveAugmentor:
    """
    Augmentation strategy for contrastive learning (SimCLR, MoCo).

    Creates multiple views of each image for self-supervised training.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        num_views: int = 2
    ):
        import torchvision.transforms as T

        self.num_views = num_views

        # Define augmentation pipeline
        self.transform = T.Compose([
            T.RandomResizedCrop(target_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                )
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([
                T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __call__(self, image) -> List:
        """Generate multiple augmented views of an image."""
        return [self.transform(image) for _ in range(self.num_views)]


class DomainSpecificAugmentor:
    """
    Domain-specific augmentation strategies.
    """

    @staticmethod
    def medical_imaging(target_size: Tuple[int, int] = (224, 224)):
        """
        Augmentations suitable for medical images.

        Preserves diagnostic features while adding realistic variations.
        """
        import torchvision.transforms as T

        return T.Compose([
            T.RandomResizedCrop(target_size, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            # Careful with color - preserve diagnostic info
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def satellite_imagery(target_size: Tuple[int, int] = (224, 224)):
        """
        Augmentations for satellite/aerial images.

        Handles rotation invariance and atmospheric variations.
        """
        import torchvision.transforms as T

        return T.Compose([
            T.RandomResizedCrop(target_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=180),  # Satellites can view from any angle
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def product_photography(target_size: Tuple[int, int] = (224, 224)):
        """
        Augmentations for e-commerce product images.

        Preserves product identity while varying presentation.
        """
        import torchvision.transforms as T

        return T.Compose([
            T.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            # Product color is important - light augmentation
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            T.RandomPerspective(distortion_scale=0.1, p=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def document_images(target_size: Tuple[int, int] = (224, 224)):
        """
        Augmentations for document/text images.

        Simulates scanning variations while preserving readability.
        """
        import torchvision.transforms as T

        return T.Compose([
            T.Resize(int(target_size[0] * 1.1)),
            T.RandomCrop(target_size),
            T.RandomRotation(degrees=2),  # Slight rotation only
            T.RandomAffine(degrees=0, shear=2),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# Example usage
if __name__ == "__main__":
    from PIL import Image

    print("Image Augmentation for Embeddings")
    print("=" * 50)

    # Create sample image
    sample = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))

    # Test different modes
    for mode in ['standard', 'contrastive', 'minimal']:
        augment_fn = create_augmentation_pipeline(mode=mode)
        augmented = augment_fn(sample)
        print(f"\n{mode.upper()} augmentation:")
        print(f"  Output shape: {augmented.shape}")
        print(f"  Value range: [{augmented.min():.2f}, {augmented.max():.2f}]")

    # Test contrastive augmentor
    print("\nContrastive Learning Augmentor:")
    augmentor = ContrastiveAugmentor(num_views=2)
    views = augmentor(sample)
    print(f"  Generated {len(views)} views")
    print(f"  Each view shape: {views[0].shape}")

    # Test domain-specific
    print("\nDomain-Specific Augmentations:")
    for name, fn in [
        ('Medical', DomainSpecificAugmentor.medical_imaging),
        ('Satellite', DomainSpecificAugmentor.satellite_imagery),
        ('Product', DomainSpecificAugmentor.product_photography),
    ]:
        aug = fn()
        result = aug(sample)
        print(f"  {name}: shape={result.shape}")
