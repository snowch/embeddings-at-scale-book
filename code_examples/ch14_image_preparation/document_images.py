"""Document image processing for embeddings."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from PIL import Image, ImageFilter, ImageOps


class DocumentRegionType(Enum):
    """Types of document regions."""

    TEXT = "text"
    FIGURE = "figure"
    TABLE = "table"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"


@dataclass
class DocumentRegion:
    """A region within a document image."""

    image: Image.Image
    region_type: DocumentRegionType
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    page_number: int = 0
    confidence: float = 1.0


@dataclass
class DocumentEmbedding:
    """Embedding for a document region or page."""

    embedding: np.ndarray
    region_type: DocumentRegionType | None
    page_number: int
    bbox: tuple[int, int, int, int] | None = None
    metadata: dict = field(default_factory=dict)


class DocumentImageProcessor:
    """Process document images for embedding."""

    def __init__(
        self,
        encoder,
        target_size: int = 224,
        enhance_text: bool = True,
        split_columns: bool = True,
    ):
        """
        Initialize document image processor.

        Args:
            encoder: Image embedding model
            target_size: Model input size
            enhance_text: Whether to enhance text regions
            split_columns: Whether to detect and split columns
        """
        self.encoder = encoder
        self.target_size = target_size
        self.enhance_text = enhance_text
        self.split_columns = split_columns

    def process_page(
        self,
        image: Image.Image,
        page_number: int = 0,
    ) -> list[DocumentEmbedding]:
        """
        Process a document page into embeddings.

        Args:
            image: Document page image
            page_number: Page number

        Returns:
            List of embeddings for page and regions
        """
        embeddings = []

        # Preprocess entire page
        processed_page = self._preprocess_document(image)

        # Full page embedding
        page_resized = processed_page.resize(
            (self.target_size, self.target_size),
            Image.Resampling.LANCZOS,
        )
        page_embedding = self.encoder.encode(page_resized)

        embeddings.append(
            DocumentEmbedding(
                embedding=page_embedding,
                region_type=None,
                page_number=page_number,
                bbox=(0, 0, image.width, image.height),
                metadata={"type": "full_page"},
            )
        )

        # Detect and embed regions
        regions = self._detect_regions(processed_page)

        for region in regions:
            region_embedding = self._embed_region(region)
            embeddings.append(
                DocumentEmbedding(
                    embedding=region_embedding,
                    region_type=region.region_type,
                    page_number=page_number,
                    bbox=region.bbox,
                    metadata={
                        "confidence": region.confidence,
                    },
                )
            )

        return embeddings

    def _preprocess_document(self, image: Image.Image) -> Image.Image:
        """
        Preprocess document image.

        Args:
            image: Input document image

        Returns:
            Preprocessed image
        """
        # Convert to grayscale for processing
        if image.mode != "L":
            gray = image.convert("L")
        else:
            gray = image.copy()

        # Deskew (simplified)
        # In production, use more sophisticated deskew algorithms

        # Enhance contrast
        enhanced = ImageOps.autocontrast(gray, cutoff=2)

        # Denoise
        enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))

        # Binarize for text regions
        if self.enhance_text:
            # Adaptive thresholding simulation
            img_array = np.array(enhanced)
            threshold = np.mean(img_array) - 0.1 * np.std(img_array)
            binary = (img_array > threshold) * 255
            enhanced = Image.fromarray(binary.astype(np.uint8))

        # Convert back to RGB for embedding model
        return enhanced.convert("RGB")

    def _detect_regions(self, image: Image.Image) -> list[DocumentRegion]:
        """
        Detect regions in document image.

        Args:
            image: Preprocessed document image

        Returns:
            List of detected regions
        """
        regions = []
        img_array = np.array(image.convert("L"))

        height, width = img_array.shape

        # Simple horizontal projection-based detection
        # Production would use ML-based layout analysis

        # Detect text blocks using horizontal projection
        h_projection = np.mean(img_array, axis=1)
        text_rows = h_projection < 250  # Text regions are darker

        # Find contiguous text regions
        in_region = False
        region_start = 0

        for i, is_text in enumerate(text_rows):
            if is_text and not in_region:
                in_region = True
                region_start = i
            elif not is_text and in_region:
                in_region = False
                region_height = i - region_start

                if region_height > 20:  # Minimum region height
                    # Determine region type based on position and size
                    if region_start < height * 0.1:
                        region_type = DocumentRegionType.HEADER
                    elif region_start > height * 0.9:
                        region_type = DocumentRegionType.FOOTER
                    elif region_height > height * 0.3:
                        region_type = DocumentRegionType.TEXT
                    else:
                        region_type = DocumentRegionType.TEXT

                    region_image = image.crop((0, region_start, width, i))

                    regions.append(
                        DocumentRegion(
                            image=region_image,
                            region_type=region_type,
                            bbox=(0, region_start, width, i),
                        )
                    )

        return regions

    def _embed_region(self, region: DocumentRegion) -> np.ndarray:
        """
        Generate embedding for a document region.

        Args:
            region: Document region

        Returns:
            Embedding vector
        """
        # Resize preserving aspect ratio
        image = region.image
        aspect = image.width / image.height

        if aspect > 1:
            new_width = self.target_size
            new_height = int(self.target_size / aspect)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect)

        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Pad to target size
        padded = Image.new("RGB", (self.target_size, self.target_size), (255, 255, 255))
        paste_x = (self.target_size - new_width) // 2
        paste_y = (self.target_size - new_height) // 2
        padded.paste(resized, (paste_x, paste_y))

        return self.encoder.encode(padded)


def process_multi_page_document(
    pages: list[Image.Image],
    processor: DocumentImageProcessor,
) -> list[DocumentEmbedding]:
    """
    Process multi-page document.

    Args:
        pages: List of page images
        processor: DocumentImageProcessor instance

    Returns:
        List of all embeddings across pages
    """
    all_embeddings = []

    for page_num, page_image in enumerate(pages):
        page_embeddings = processor.process_page(page_image, page_number=page_num)
        all_embeddings.extend(page_embeddings)

    return all_embeddings
