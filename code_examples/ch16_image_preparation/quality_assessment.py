"""Image quality assessment for filtering before embedding."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class QualityResult:
    """Result of image quality assessment."""

    passed: bool
    blur_score: float
    brightness: float
    contrast: float
    resolution_ok: bool
    aspect_ratio_ok: bool
    issues: List[str]


def assess_image_quality(
    image,
    min_resolution: int = 100,
    max_aspect_ratio: float = 4.0,
    min_brightness: float = 0.1,
    max_brightness: float = 0.9,
    min_contrast: float = 0.1,
    blur_threshold: float = 100.0,
) -> QualityResult:
    """
    Assess image quality for embedding suitability.

    Args:
        image: PIL Image or numpy array
        min_resolution: Minimum pixels on smallest side
        max_aspect_ratio: Maximum width/height ratio
        min_brightness: Minimum average brightness (0-1)
        max_brightness: Maximum average brightness (0-1)
        min_contrast: Minimum contrast (std dev of pixel values)
        blur_threshold: Laplacian variance threshold (higher = sharper)

    Returns:
        QualityResult with pass/fail and metrics
    """
    import cv2
    from PIL import Image

    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image

    issues = []

    # Resolution check
    h, w = image_array.shape[:2]
    min_dim = min(h, w)
    resolution_ok = min_dim >= min_resolution
    if not resolution_ok:
        issues.append(f"Resolution too low: {min_dim}px < {min_resolution}px")

    # Aspect ratio check
    aspect_ratio = max(w, h) / min(w, h)
    aspect_ratio_ok = aspect_ratio <= max_aspect_ratio
    if not aspect_ratio_ok:
        issues.append(f"Aspect ratio too extreme: {aspect_ratio:.1f} > {max_aspect_ratio}")

    # Convert to grayscale for analysis
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array

    # Brightness check
    brightness = np.mean(gray) / 255.0
    brightness_ok = min_brightness <= brightness <= max_brightness
    if not brightness_ok:
        issues.append(f"Brightness out of range: {brightness:.2f}")

    # Contrast check
    contrast = np.std(gray) / 255.0
    contrast_ok = contrast >= min_contrast
    if not contrast_ok:
        issues.append(f"Low contrast: {contrast:.3f} < {min_contrast}")

    # Blur detection using Laplacian variance
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_ok = blur_score >= blur_threshold
    if not blur_ok:
        issues.append(f"Image too blurry: {blur_score:.1f} < {blur_threshold}")

    passed = all([resolution_ok, aspect_ratio_ok, brightness_ok, contrast_ok, blur_ok])

    return QualityResult(
        passed=passed,
        blur_score=blur_score,
        brightness=brightness,
        contrast=contrast,
        resolution_ok=resolution_ok,
        aspect_ratio_ok=aspect_ratio_ok,
        issues=issues,
    )


def detect_duplicate_images(images: List, threshold: float = 0.95) -> List[Tuple[int, int]]:
    """
    Detect duplicate or near-duplicate images using perceptual hashing.

    Args:
        images: List of PIL Images
        threshold: Similarity threshold (0-1) for duplicate detection

    Returns:
        List of (idx1, idx2) pairs that are duplicates
    """
    import imagehash
    from PIL import Image

    # Compute perceptual hashes
    hashes = []
    for img in images:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        phash = imagehash.phash(img)
        hashes.append(phash)

    # Find duplicates
    duplicates = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            # Compute similarity (inverse of Hamming distance)
            distance = hashes[i] - hashes[j]
            max_distance = len(hashes[i].hash.flatten()) * len(hashes[i].hash.flatten())
            similarity = 1 - (distance / max_distance)

            if similarity >= threshold:
                duplicates.append((i, j))

    return duplicates


def detect_corrupted_image(image_path: str) -> Tuple[bool, Optional[str]]:
    """
    Check if an image file is corrupted.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (is_valid, error_message)
    """
    from PIL import Image

    try:
        with Image.open(image_path) as img:
            # Try to load the full image
            img.load()

            # Verify it's a valid image
            img.verify()

        return True, None

    except Exception as e:
        return False, str(e)


def compute_image_statistics(image) -> Dict:
    """
    Compute detailed statistics about an image.

    Useful for understanding dataset characteristics.
    """
    from PIL import Image

    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image

    stats = {
        "height": image_array.shape[0],
        "width": image_array.shape[1],
        "channels": image_array.shape[2] if len(image_array.shape) == 3 else 1,
        "dtype": str(image_array.dtype),
    }

    # Per-channel statistics
    if len(image_array.shape) == 3:
        for i, channel in enumerate(["red", "green", "blue"]):
            ch = image_array[:, :, i]
            stats[f"{channel}_mean"] = float(np.mean(ch))
            stats[f"{channel}_std"] = float(np.std(ch))
            stats[f"{channel}_min"] = int(np.min(ch))
            stats[f"{channel}_max"] = int(np.max(ch))
    else:
        stats["gray_mean"] = float(np.mean(image_array))
        stats["gray_std"] = float(np.std(image_array))

    # Overall statistics
    stats["aspect_ratio"] = stats["width"] / stats["height"]
    stats["total_pixels"] = stats["width"] * stats["height"]

    return stats


class ImageQualityFilter:
    """
    Filter images based on quality criteria.

    Use as a preprocessing step before embedding.
    """

    def __init__(
        self,
        min_resolution: int = 100,
        max_aspect_ratio: float = 4.0,
        blur_threshold: float = 100.0,
        remove_duplicates: bool = True,
    ):
        self.min_resolution = min_resolution
        self.max_aspect_ratio = max_aspect_ratio
        self.blur_threshold = blur_threshold
        self.remove_duplicates = remove_duplicates

    def filter(self, images: List) -> Tuple[List, List[int]]:
        """
        Filter images and return valid ones.

        Args:
            images: List of images

        Returns:
            Tuple of (filtered_images, kept_indices)
        """
        # Quality assessment
        valid_indices = []
        for i, img in enumerate(images):
            result = assess_image_quality(
                img,
                min_resolution=self.min_resolution,
                max_aspect_ratio=self.max_aspect_ratio,
                blur_threshold=self.blur_threshold,
            )
            if result.passed:
                valid_indices.append(i)

        valid_images = [images[i] for i in valid_indices]

        # Duplicate removal
        if self.remove_duplicates and len(valid_images) > 1:
            duplicates = detect_duplicate_images(valid_images)
            dup_indices = {j for _, j in duplicates}  # Remove second of each pair

            final_images = []
            final_indices = []
            for i, (img, orig_idx) in enumerate(zip(valid_images, valid_indices)):
                if i not in dup_indices:
                    final_images.append(img)
                    final_indices.append(orig_idx)

            return final_images, final_indices

        return valid_images, valid_indices


# Example usage
if __name__ == "__main__":
    from PIL import Image

    print("Image Quality Assessment Demo")
    print("=" * 50)

    # Test with different quality images
    test_cases = [
        ("Good quality", np.random.randint(0, 255, (500, 400, 3), dtype=np.uint8)),
        ("Too small", np.random.randint(0, 255, (50, 40, 3), dtype=np.uint8)),
        ("Too dark", np.random.randint(0, 30, (200, 200, 3), dtype=np.uint8)),
        (
            "Low contrast",
            np.full((200, 200, 3), 128, dtype=np.uint8)
            + np.random.randint(-5, 5, (200, 200, 3), dtype=np.int8).astype(np.uint8),
        ),
    ]

    for name, img_array in test_cases:
        result = assess_image_quality(Image.fromarray(img_array))
        print(f"\n{name}:")
        print(f"  Passed: {result.passed}")
        print(f"  Blur score: {result.blur_score:.1f}")
        print(f"  Brightness: {result.brightness:.2f}")
        print(f"  Contrast: {result.contrast:.3f}")
        if result.issues:
            print(f"  Issues: {', '.join(result.issues)}")
