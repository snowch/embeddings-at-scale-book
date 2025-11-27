"""Object detection followed by individual object embedding."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DetectedObject:
    """An object detected in an image."""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_name: str
    confidence: float
    cropped_image: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None


def detect_objects_yolo(
    image, confidence_threshold: float = 0.5, model_name: str = "yolov5s"
) -> List[DetectedObject]:
    """
    Detect objects using YOLOv5.

    Args:
        image: PIL Image or numpy array
        confidence_threshold: Minimum confidence for detection
        model_name: YOLOv5 model variant

    Returns:
        List of DetectedObject with bounding boxes
    """
    import torch
    from PIL import Image

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Load YOLOv5 model
    model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)
    model.conf = confidence_threshold

    # Run detection
    results = model(image)

    # Parse results
    detections = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names[int(cls)]

        # Crop object from image
        img_array = np.array(image)
        cropped = img_array[y1:y2, x1:x2]

        detections.append(
            DetectedObject(
                bbox=(x1, y1, x2, y2),
                class_name=class_name,
                confidence=float(conf),
                cropped_image=cropped,
            )
        )

    return detections


def embed_detected_objects(
    detections: List[DetectedObject], encoder, min_size: int = 32
) -> List[DetectedObject]:
    """
    Generate embeddings for each detected object.

    Args:
        detections: List of DetectedObject from detection
        encoder: Image embedding model
        min_size: Minimum object size to embed

    Returns:
        Same detections with embeddings added
    """
    from PIL import Image

    valid_objects = []
    valid_images = []

    for det in detections:
        if det.cropped_image is None:
            continue

        h, w = det.cropped_image.shape[:2]
        if h < min_size or w < min_size:
            continue

        valid_objects.append(det)
        valid_images.append(Image.fromarray(det.cropped_image))

    if valid_images:
        embeddings = encoder.encode(valid_images)

        for det, emb in zip(valid_objects, embeddings):
            det.embedding = emb

    return detections


class ObjectLevelEmbedder:
    """
    Create embeddings at the object level within images.

    Workflow:
    1. Detect objects in image
    2. Crop each object
    3. Embed each crop individually
    4. Return structured embeddings with metadata
    """

    def __init__(
        self,
        detection_model: str = "yolov5s",
        embedding_model: str = "resnet50",
        confidence_threshold: float = 0.5,
        min_object_size: int = 32,
        target_classes: Optional[List[str]] = None,
    ):
        self.detection_model = detection_model
        self.confidence_threshold = confidence_threshold
        self.min_object_size = min_object_size
        self.target_classes = target_classes

        # Initialize embedding model
        from cnn_embeddings import CNNEmbedder

        self.encoder = CNNEmbedder(model_name=embedding_model)

    def process_image(self, image) -> Dict:
        """
        Process a single image to extract object-level embeddings.

        Returns dict with:
        - objects: List of detected objects with embeddings
        - scene_embedding: Embedding of full image
        - metadata: Processing information
        """
        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image
            image = np.array(image)

        # Detect objects
        detections = detect_objects_yolo(
            pil_image,
            confidence_threshold=self.confidence_threshold,
            model_name=self.detection_model,
        )

        # Filter by target classes if specified
        if self.target_classes:
            detections = [d for d in detections if d.class_name in self.target_classes]

        # Embed objects
        detections = embed_detected_objects(detections, self.encoder, min_size=self.min_object_size)

        # Get scene-level embedding
        scene_embedding = self.encoder.encode([pil_image])[0]

        return {
            "objects": detections,
            "scene_embedding": scene_embedding,
            "num_objects": len([d for d in detections if d.embedding is not None]),
            "image_size": image.shape[:2],
        }

    def batch_process(self, images: List) -> List[Dict]:
        """Process multiple images."""
        return [self.process_image(img) for img in images]


def create_object_index(
    processed_results: List[Dict], include_scenes: bool = True
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Create a searchable index from object embeddings.

    Args:
        processed_results: Results from ObjectLevelEmbedder.batch_process()
        include_scenes: Whether to include scene-level embeddings

    Returns:
        Tuple of (embeddings array, metadata list)
    """
    embeddings = []
    metadata = []

    for img_idx, result in enumerate(processed_results):
        # Add object embeddings
        for obj in result["objects"]:
            if obj.embedding is not None:
                embeddings.append(obj.embedding)
                metadata.append(
                    {
                        "type": "object",
                        "image_idx": img_idx,
                        "class": obj.class_name,
                        "bbox": obj.bbox,
                        "confidence": obj.confidence,
                    }
                )

        # Add scene embedding
        if include_scenes:
            embeddings.append(result["scene_embedding"])
            metadata.append(
                {"type": "scene", "image_idx": img_idx, "num_objects": result["num_objects"]}
            )

    return np.array(embeddings), metadata


# Example usage
if __name__ == "__main__":
    # This example requires YOLOv5 and an image with objects
    print("Object Detection + Embedding Pipeline")
    print("=" * 50)

    # Create sample image (in practice, use real images)
    sample_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

    print("This example requires:")
    print("  - YOLOv5: pip install ultralytics")
    print("  - Real images with detectable objects")
    print()

    # Example workflow (pseudocode):
    print("Example workflow:")
    print("  1. Load image")
    print("  2. Detect objects with YOLO")
    print("  3. Crop detected objects")
    print("  4. Embed each crop")
    print("  5. Store embeddings with metadata")
    print()
    print("Use cases:")
    print("  - Search for specific objects within images")
    print("  - Find images containing similar objects")
    print("  - Object-level similarity comparison")
