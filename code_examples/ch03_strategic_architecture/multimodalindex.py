import faiss
import numpy as np
import torch

# Code from Chapter 02
# Book: Embeddings at Scale


# ModalityFusion placeholder - see modalityfusion.py for full implementation
class ModalityFusion:
    """Placeholder for ModalityFusion."""

    @staticmethod
    def early_fusion(modality_embeddings, weights=None):
        if weights is None:
            weights = [1.0 / len(modality_embeddings)] * len(modality_embeddings)
        # Convert to tensors if needed
        embeddings_tensors = []
        for emb in modality_embeddings:
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)
            embeddings_tensors.append(emb)
        fused = sum(w * emb for w, emb in zip(weights, embeddings_tensors))
        return fused / torch.norm(fused)


class MultiModalIndex:
    """Scalable multi-modal indexing"""

    def __init__(self, embedding_dim=512):
        self.embedding_dim = embedding_dim

        # Separate indices per modality (for modality-specific search)
        self.text_index = faiss.IndexHNSWFlat(embedding_dim, 32)
        self.image_index = faiss.IndexHNSWFlat(embedding_dim, 32)
        self.video_index = faiss.IndexHNSWFlat(embedding_dim, 32)

        # Unified index (for fused multi-modal embeddings)
        self.unified_index = faiss.IndexHNSWFlat(embedding_dim, 32)

        # Metadata storage
        self.metadata = []

    def add_multimodal_item(
        self, item_id, text_emb=None, image_emb=None, video_emb=None, metadata=None
    ):
        """Add item with multiple modalities"""
        # Add to modality-specific indices
        if text_emb is not None:
            self.text_index.add(text_emb.reshape(1, -1))
        if image_emb is not None:
            self.image_index.add(image_emb.reshape(1, -1))
        if video_emb is not None:
            self.video_index.add(video_emb.reshape(1, -1))

        # Create fused embedding for unified index
        available_embs = [emb for emb in [text_emb, image_emb, video_emb] if emb is not None]
        if available_embs:
            fused_emb = ModalityFusion.early_fusion(available_embs)
            self.unified_index.add(fused_emb.reshape(1, -1))

        # Store metadata
        self.metadata.append(
            {
                "item_id": item_id,
                "has_text": text_emb is not None,
                "has_image": image_emb is not None,
                "has_video": video_emb is not None,
                "metadata": metadata,
            }
        )

    def search_multimodal(self, query_embs, modality_weights=None, k=10):
        """
        Search with multi-modal query
        query_embs: dict like {'text': text_emb, 'image': image_emb}
        """
        if modality_weights is None:
            modality_weights = {mod: 1.0 / len(query_embs) for mod in query_embs}

        # Search each modality
        results_by_modality = {}
        if "text" in query_embs:
            distances, indices = self.text_index.search(query_embs["text"].reshape(1, -1), k)
            results_by_modality["text"] = list(zip(indices[0], distances[0]))

        if "image" in query_embs:
            distances, indices = self.image_index.search(query_embs["image"].reshape(1, -1), k)
            results_by_modality["image"] = list(zip(indices[0], distances[0]))

        if "video" in query_embs:
            distances, indices = self.video_index.search(query_embs["video"].reshape(1, -1), k)
            results_by_modality["video"] = list(zip(indices[0], distances[0]))

        # Combine results with late fusion
        combined_scores = {}
        for modality, results in results_by_modality.items():
            weight = modality_weights.get(modality, 1.0)
            for idx, distance in results:
                # Convert distance to similarity (smaller distance = higher similarity)
                similarity = 1.0 / (1.0 + distance)
                combined_scores[idx] = combined_scores.get(idx, 0) + weight * similarity

        # Sort by combined score
        top_k = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return [
            {
                "item_id": self.metadata[idx]["item_id"],
                "score": score,
                "metadata": self.metadata[idx]["metadata"],
            }
            for idx, score in top_k
        ]
