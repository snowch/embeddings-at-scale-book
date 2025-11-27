"""Hybrid scene + object embedding strategies."""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class HybridEmbedding:
    """Combined scene and object embedding."""

    scene_embedding: np.ndarray
    object_embeddings: list[np.ndarray]
    combined_embedding: np.ndarray | None = None
    image_id: str = ""
    object_labels: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class HybridEmbedder:
    """Create hybrid embeddings combining scene and object information."""

    def __init__(
        self,
        encoder,
        detector=None,
        target_size: int = 224,
        combination_method: str = "concat",
        max_objects: int = 10,
    ):
        """
        Initialize hybrid embedder.

        Args:
            encoder: Image embedding model
            detector: Object detection model
            target_size: Model input size
            combination_method: How to combine embeddings
            max_objects: Maximum objects to include
        """
        self.encoder = encoder
        self.detector = detector
        self.target_size = target_size
        self.combination_method = combination_method
        self.max_objects = max_objects

    def embed(
        self,
        image: Image.Image,
        image_id: str = "",
        detections: list[dict] | None = None,
    ) -> HybridEmbedding:
        """
        Generate hybrid embedding.

        Args:
            image: Input image
            image_id: Image identifier
            detections: Pre-computed detections

        Returns:
            HybridEmbedding with scene and object information
        """
        # Scene embedding
        scene_resized = image.resize(
            (self.target_size, self.target_size),
            Image.Resampling.LANCZOS,
        )
        scene_embedding = self.encoder.encode(scene_resized)

        # Object embeddings
        if detections is None and self.detector is not None:
            detections = self.detector.detect(image)
        detections = detections or []

        object_embeddings = []
        object_labels = []

        # Sort by confidence and take top objects
        sorted_detections = sorted(detections, key=lambda x: x.get("confidence", 0), reverse=True)

        for det in sorted_detections[: self.max_objects]:
            bbox = det["bbox"]
            obj_image = image.crop(bbox)

            # Resize preserving aspect ratio
            obj_resized = self._resize_preserve_aspect(obj_image)

            obj_embedding = self.encoder.encode(obj_resized)
            object_embeddings.append(obj_embedding)
            object_labels.append(det.get("label", "unknown"))

        # Combine embeddings
        combined = self._combine_embeddings(scene_embedding, object_embeddings)

        return HybridEmbedding(
            scene_embedding=scene_embedding,
            object_embeddings=object_embeddings,
            combined_embedding=combined,
            image_id=image_id,
            object_labels=object_labels,
            metadata={
                "num_objects": len(object_embeddings),
                "combination_method": self.combination_method,
            },
        )

    def _resize_preserve_aspect(self, image: Image.Image) -> Image.Image:
        """Resize image preserving aspect ratio with padding."""
        aspect = image.width / image.height

        if aspect > 1:
            new_width = self.target_size
            new_height = int(self.target_size / aspect)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect)

        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        padded = Image.new("RGB", (self.target_size, self.target_size), (128, 128, 128))
        paste_x = (self.target_size - new_width) // 2
        paste_y = (self.target_size - new_height) // 2
        padded.paste(resized, (paste_x, paste_y))

        return padded

    def _combine_embeddings(
        self,
        scene_embedding: np.ndarray,
        object_embeddings: list[np.ndarray],
    ) -> np.ndarray | None:
        """
        Combine scene and object embeddings.

        Args:
            scene_embedding: Scene-level embedding
            object_embeddings: List of object embeddings

        Returns:
            Combined embedding or None if no combination
        """
        if not object_embeddings:
            return scene_embedding

        if self.combination_method == "concat":
            # Concatenate scene with pooled object embeddings
            obj_pooled = np.mean(object_embeddings, axis=0)
            return np.concatenate([scene_embedding, obj_pooled])

        elif self.combination_method == "weighted_sum":
            # Weighted combination (scene gets more weight)
            obj_pooled = np.mean(object_embeddings, axis=0)
            return 0.7 * scene_embedding + 0.3 * obj_pooled

        elif self.combination_method == "attention":
            # Simple attention-based combination
            return self._attention_combination(scene_embedding, object_embeddings)

        else:
            return None

    def _attention_combination(
        self,
        scene_embedding: np.ndarray,
        object_embeddings: list[np.ndarray],
    ) -> np.ndarray:
        """
        Combine using attention mechanism.

        Args:
            scene_embedding: Scene embedding (query)
            object_embeddings: Object embeddings (keys/values)

        Returns:
            Attention-weighted combination
        """
        # Compute attention scores
        scores = []
        for obj_emb in object_embeddings:
            # Dot product attention
            score = np.dot(scene_embedding, obj_emb)
            scores.append(score)

        # Softmax
        scores = np.array(scores)
        exp_scores = np.exp(scores - np.max(scores))
        attention_weights = exp_scores / (exp_scores.sum() + 1e-8)

        # Weighted sum of object embeddings
        weighted_objects = np.zeros_like(scene_embedding)
        for weight, obj_emb in zip(attention_weights, object_embeddings):
            weighted_objects += weight * obj_emb

        # Combine with scene
        return 0.5 * scene_embedding + 0.5 * weighted_objects


def search_hybrid_embeddings(
    query_embedding: np.ndarray,
    hybrid_embeddings: list[HybridEmbedding],
    search_mode: str = "combined",
    top_k: int = 10,
) -> list[dict]:
    """
    Search using hybrid embeddings.

    Args:
        query_embedding: Query vector
        hybrid_embeddings: Database of hybrid embeddings
        search_mode: 'combined', 'scene', or 'objects'
        top_k: Number of results

    Returns:
        List of search results with scores
    """
    results = []

    for hybrid in hybrid_embeddings:
        if search_mode == "combined" and hybrid.combined_embedding is not None:
            target = hybrid.combined_embedding
        elif search_mode == "scene":
            target = hybrid.scene_embedding
        elif search_mode == "objects" and hybrid.object_embeddings:
            # Max similarity across objects
            obj_similarities = [
                np.dot(query_embedding, obj)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(obj))
                for obj in hybrid.object_embeddings
            ]
            max_idx = np.argmax(obj_similarities)
            results.append(
                {
                    "image_id": hybrid.image_id,
                    "similarity": float(obj_similarities[max_idx]),
                    "matched_object": hybrid.object_labels[max_idx]
                    if hybrid.object_labels
                    else None,
                }
            )
            continue
        else:
            target = hybrid.scene_embedding

        similarity = np.dot(query_embedding, target) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(target)
        )
        results.append(
            {
                "image_id": hybrid.image_id,
                "similarity": float(similarity),
            }
        )

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]
