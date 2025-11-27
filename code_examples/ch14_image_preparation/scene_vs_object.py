"""Scene-level vs object-level embedding strategies."""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class DetectedObject:
    """An object detected in an image."""

    image: Image.Image
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    object_id: int = 0


@dataclass
class SceneEmbedding:
    """Scene-level embedding for an image."""

    embedding: np.ndarray
    image_id: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ObjectEmbedding:
    """Object-level embedding."""

    embedding: np.ndarray
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    object_id: int
    parent_image_id: str = ""


class SceneObjectEmbedder:
    """Generate both scene and object level embeddings."""

    def __init__(
        self,
        encoder,
        detector=None,
        target_size: int = 224,
        min_object_confidence: float = 0.5,
        min_object_area: int = 1000,
    ):
        """
        Initialize scene/object embedder.

        Args:
            encoder: Image embedding model
            detector: Object detection model (optional)
            target_size: Model input size
            min_object_confidence: Minimum detection confidence
            min_object_area: Minimum object area in pixels
        """
        self.encoder = encoder
        self.detector = detector
        self.target_size = target_size
        self.min_object_confidence = min_object_confidence
        self.min_object_area = min_object_area

    def embed_scene(
        self,
        image: Image.Image,
        image_id: str = "",
    ) -> SceneEmbedding:
        """
        Generate scene-level embedding.

        Args:
            image: Input image
            image_id: Unique image identifier

        Returns:
            SceneEmbedding for the full image
        """
        # Resize for model
        resized = image.resize(
            (self.target_size, self.target_size),
            Image.Resampling.LANCZOS,
        )

        # Generate embedding
        embedding = self.encoder.encode(resized)

        return SceneEmbedding(
            embedding=embedding,
            image_id=image_id,
            metadata={
                "original_size": image.size,
                "type": "scene",
            },
        )

    def embed_objects(
        self,
        image: Image.Image,
        detections: list[dict] | None = None,
        image_id: str = "",
    ) -> list[ObjectEmbedding]:
        """
        Generate object-level embeddings.

        Args:
            image: Input image
            detections: Pre-computed detections (optional)
            image_id: Parent image identifier

        Returns:
            List of ObjectEmbedding for detected objects
        """
        if detections is None:
            detections = self._detect_objects(image)

        embeddings = []

        for idx, det in enumerate(detections):
            # Filter by confidence
            if det["confidence"] < self.min_object_confidence:
                continue

            # Filter by area
            bbox = det["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < self.min_object_area:
                continue

            # Crop object
            object_image = image.crop(bbox)

            # Resize for model
            resized = self._resize_object(object_image)

            # Generate embedding
            embedding = self.encoder.encode(resized)

            embeddings.append(
                ObjectEmbedding(
                    embedding=embedding,
                    label=det["label"],
                    confidence=det["confidence"],
                    bbox=bbox,
                    object_id=idx,
                    parent_image_id=image_id,
                )
            )

        return embeddings

    def embed_scene_and_objects(
        self,
        image: Image.Image,
        image_id: str = "",
        detections: list[dict] | None = None,
    ) -> tuple[SceneEmbedding, list[ObjectEmbedding]]:
        """
        Generate both scene and object embeddings.

        Args:
            image: Input image
            image_id: Image identifier
            detections: Pre-computed detections (optional)

        Returns:
            Tuple of (scene_embedding, list of object_embeddings)
        """
        scene_emb = self.embed_scene(image, image_id)
        object_embs = self.embed_objects(image, detections, image_id)

        return scene_emb, object_embs

    def _detect_objects(self, image: Image.Image) -> list[dict]:
        """
        Detect objects in image.

        Args:
            image: Input image

        Returns:
            List of detection dicts with bbox, label, confidence
        """
        if self.detector is not None:
            return self.detector.detect(image)

        # Placeholder - return empty list if no detector
        # In production, integrate with YOLO, Detectron2, etc.
        return []

    def _resize_object(self, image: Image.Image) -> Image.Image:
        """Resize object image preserving aspect ratio."""
        # Calculate new size preserving aspect ratio
        aspect = image.width / image.height

        if aspect > 1:
            new_width = self.target_size
            new_height = int(self.target_size / aspect)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect)

        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Pad to square
        padded = Image.new("RGB", (self.target_size, self.target_size), (128, 128, 128))
        paste_x = (self.target_size - new_width) // 2
        paste_y = (self.target_size - new_height) // 2
        padded.paste(resized, (paste_x, paste_y))

        return padded


def compare_scene_vs_object_retrieval(
    query_image: Image.Image,
    scene_embeddings: list[SceneEmbedding],
    object_embeddings: list[ObjectEmbedding],
    encoder,
    top_k: int = 5,
) -> dict:
    """
    Compare scene vs object retrieval results.

    Args:
        query_image: Query image
        scene_embeddings: Database of scene embeddings
        object_embeddings: Database of object embeddings
        encoder: Embedding encoder
        top_k: Number of results to return

    Returns:
        Dict with scene and object retrieval results
    """
    # Embed query
    query_resized = query_image.resize((224, 224), Image.Resampling.LANCZOS)
    query_embedding = encoder.encode(query_resized)

    # Scene retrieval
    scene_scores = []
    for scene in scene_embeddings:
        similarity = np.dot(query_embedding, scene.embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(scene.embedding)
        )
        scene_scores.append((scene.image_id, float(similarity)))

    scene_scores.sort(key=lambda x: x[1], reverse=True)

    # Object retrieval
    object_scores = []
    for obj in object_embeddings:
        similarity = np.dot(query_embedding, obj.embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(obj.embedding)
        )
        object_scores.append(
            {
                "image_id": obj.parent_image_id,
                "object_id": obj.object_id,
                "label": obj.label,
                "similarity": float(similarity),
            }
        )

    object_scores.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "scene_results": scene_scores[:top_k],
        "object_results": object_scores[:top_k],
    }
