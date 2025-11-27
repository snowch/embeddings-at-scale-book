from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ForensicSearchConfig:
    """Configuration for forensic video search."""

    frame_size: int = 224
    embedding_dim: int = 512
    hidden_dim: int = 1024
    n_attributes: int = 50  # Color, object type, etc.


class VisualSearchEncoder(nn.Module):
    """
    Encoder for visual similarity search.

    Enables query-by-example search to find visually
    similar frames across video archives.
    """

    def __init__(self, config: ForensicSearchConfig):
        super().__init__()
        self.config = config

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.projection = nn.Linear(512, config.embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images for similarity search.

        Args:
            images: [batch, 3, H, W] images/frames

        Returns:
            embeddings: [batch, embedding_dim] visual embeddings
        """
        features = self.backbone(images).squeeze(-1).squeeze(-1)
        embeddings = self.projection(features)
        return F.normalize(embeddings, dim=-1)


class AttributeExtractor(nn.Module):
    """
    Extract structured attributes from frames.

    Enables attribute-based search like "person in red shirt"
    or "white sedan".
    """

    def __init__(self, config: ForensicSearchConfig):
        super().__init__()
        self.config = config

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Multi-label attribute classifier
        self.attribute_head = nn.Linear(256, config.n_attributes)

        # Attribute names (example subset)
        self.attribute_names = [
            "person_present",
            "vehicle_present",
            "red_clothing",
            "blue_clothing",
            "black_clothing",
            "white_clothing",
            "backpack",
            "bag",
            "hat",
            "glasses",
            "sedan",
            "suv",
            "truck",
            "motorcycle",
            "bicycle",
            # ... more attributes
        ]

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Extract attributes from images.

        Args:
            images: [batch, 3, H, W] images/frames

        Returns:
            attribute_logits: Raw attribute predictions
            attributes: Dict mapping attribute names to probabilities
        """
        features = self.backbone(images).squeeze(-1).squeeze(-1)
        logits = self.attribute_head(features)
        probs = torch.sigmoid(logits)

        attributes = {}
        for i, name in enumerate(self.attribute_names):
            if i < logits.shape[-1]:
                attributes[name] = probs[:, i]

        return logits, attributes


class TextToVideoEncoder(nn.Module):
    """
    Text-to-video search encoder.

    Enables natural language queries like
    "person running through parking lot".
    """

    def __init__(self, config: ForensicSearchConfig, vocab_size: int = 30000):
        super().__init__()
        self.config = config

        # Text encoder
        self.token_embed = nn.Embedding(vocab_size, config.embedding_dim)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim,
                batch_first=True,
            ),
            num_layers=4,
        )

        # Video encoder
        self.video_encoder = VisualSearchEncoder(config)

        # Projection to shared space
        self.text_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.video_projection = nn.Linear(config.embedding_dim, config.embedding_dim)

    def encode_text(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text query.

        Args:
            token_ids: [batch, seq_len] token indices
            attention_mask: [batch, seq_len] valid token mask

        Returns:
            text_embedding: [batch, embedding_dim] text embedding
        """
        x = self.token_embed(token_ids)

        if attention_mask is not None:
            x = self.text_encoder(x, src_key_padding_mask=~attention_mask.bool())
        else:
            x = self.text_encoder(x)

        # Mean pooling
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            pooled = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        embedding = self.text_projection(pooled)
        return F.normalize(embedding, dim=-1)

    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames.

        Args:
            frames: [batch, 3, H, W] video frames

        Returns:
            video_embedding: [batch, embedding_dim] video embedding
        """
        raw_emb = self.video_encoder(frames)
        embedding = self.video_projection(raw_emb)
        return F.normalize(embedding, dim=-1)


class VideoArchiveIndex:
    """
    Searchable index of video archive.

    Maintains embeddings and metadata for efficient
    forensic search across large video collections.
    """

    def __init__(self, config: ForensicSearchConfig):
        self.config = config
        self.visual_encoder = VisualSearchEncoder(config)
        self.attribute_extractor = AttributeExtractor(config)

        # Index storage
        self.frame_embeddings: list[torch.Tensor] = []
        self.frame_attributes: list[dict] = []
        self.frame_metadata: list[dict] = []

    def index_frame(
        self,
        frame: torch.Tensor,
        camera_id: str,
        timestamp: float,
        frame_number: int,
    ) -> int:
        """
        Add frame to index.

        Returns:
            frame_id: Index of added frame
        """
        # Compute embedding
        embedding = self.visual_encoder(frame.unsqueeze(0)).squeeze(0)

        # Extract attributes
        _, attributes = self.attribute_extractor(frame.unsqueeze(0))
        attr_dict = {k: v.item() for k, v in attributes.items()}

        # Store
        frame_id = len(self.frame_embeddings)
        self.frame_embeddings.append(embedding)
        self.frame_attributes.append(attr_dict)
        self.frame_metadata.append({
            "frame_id": frame_id,
            "camera_id": camera_id,
            "timestamp": timestamp,
            "frame_number": frame_number,
        })

        return frame_id

    def search_by_example(
        self,
        query_frame: torch.Tensor,
        k: int = 20,
        time_range: Optional[tuple[float, float]] = None,
        camera_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Search by visual example.

        Args:
            query_frame: Query image
            k: Number of results
            time_range: Optional (start, end) timestamp filter
            camera_ids: Optional camera filter

        Returns:
            List of matching results with metadata
        """
        query_emb = self.visual_encoder(query_frame.unsqueeze(0)).squeeze(0)

        # Stack all embeddings
        all_embs = torch.stack(self.frame_embeddings)

        # Compute similarities
        similarities = F.cosine_similarity(query_emb.unsqueeze(0), all_embs)

        # Apply filters
        mask = torch.ones_like(similarities, dtype=torch.bool)

        if time_range is not None:
            start, end = time_range
            for i, meta in enumerate(self.frame_metadata):
                if not (start <= meta["timestamp"] <= end):
                    mask[i] = False

        if camera_ids is not None:
            for i, meta in enumerate(self.frame_metadata):
                if meta["camera_id"] not in camera_ids:
                    mask[i] = False

        # Apply mask
        similarities[~mask] = -1

        # Get top k
        top_k = torch.topk(similarities, min(k, mask.sum().item()))

        results = []
        for idx, sim in zip(top_k.indices, top_k.values):
            if sim.item() < 0:
                continue
            results.append({
                **self.frame_metadata[idx],
                "similarity": sim.item(),
                "attributes": self.frame_attributes[idx],
            })

        return results

    def search_by_attributes(
        self,
        attributes: dict[str, float],
        k: int = 20,
        threshold: float = 0.5,
    ) -> list[dict]:
        """
        Search by attribute constraints.

        Args:
            attributes: Dict of attribute name -> minimum probability
            k: Number of results
            threshold: Minimum match score

        Returns:
            List of matching results
        """
        results = []

        for i, frame_attrs in enumerate(self.frame_attributes):
            # Compute attribute match score
            scores = []
            for attr_name, min_prob in attributes.items():
                if attr_name in frame_attrs:
                    if frame_attrs[attr_name] >= min_prob:
                        scores.append(frame_attrs[attr_name])
                    else:
                        scores.append(0)
                else:
                    scores.append(0)

            if scores:
                match_score = sum(scores) / len(scores)
                if match_score >= threshold:
                    results.append({
                        **self.frame_metadata[i],
                        "match_score": match_score,
                        "attributes": frame_attrs,
                    })

        # Sort by match score
        results.sort(key=lambda x: x["match_score"], reverse=True)

        return results[:k]

    def get_timeline(
        self,
        query_embedding: torch.Tensor,
        time_window_seconds: float = 3600,
        n_buckets: int = 24,
    ) -> list[dict]:
        """
        Get timeline distribution of search results.

        Shows when matching content appears across time.
        """
        all_embs = torch.stack(self.frame_embeddings)
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), all_embs)

        # Get timestamps
        timestamps = [m["timestamp"] for m in self.frame_metadata]
        min_ts = min(timestamps)
        max_ts = max(timestamps)

        bucket_size = (max_ts - min_ts) / n_buckets

        timeline = []
        for i in range(n_buckets):
            bucket_start = min_ts + i * bucket_size
            bucket_end = bucket_start + bucket_size

            # Get frames in this bucket
            bucket_mask = [
                bucket_start <= ts < bucket_end for ts in timestamps
            ]
            bucket_sims = similarities[bucket_mask]

            if len(bucket_sims) > 0:
                timeline.append({
                    "start": bucket_start,
                    "end": bucket_end,
                    "count": len(bucket_sims),
                    "max_similarity": bucket_sims.max().item(),
                    "avg_similarity": bucket_sims.mean().item(),
                })
            else:
                timeline.append({
                    "start": bucket_start,
                    "end": bucket_end,
                    "count": 0,
                    "max_similarity": 0,
                    "avg_similarity": 0,
                })

        return timeline
