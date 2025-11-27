import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class GEOINTConfig:
    """Configuration for geospatial intelligence embedding models."""
    image_size: int = 512
    n_spectral_bands: int = 4  # RGB + NIR or multispectral
    embedding_dim: int = 512
    hidden_dim: int = 1024
    n_object_classes: int = 50  # Military vehicles, infrastructure, etc.
    patch_size: int = 32


class SatelliteImageEncoder(nn.Module):
    """
    Encode satellite/aerial imagery into embeddings.

    Handles multi-spectral imagery with varying resolutions
    for scene understanding and object detection.
    """

    def __init__(self, config: GEOINTConfig):
        super().__init__()
        self.config = config

        # Multi-spectral feature extraction
        self.backbone = nn.Sequential(
            nn.Conv2d(config.n_spectral_bands, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.projection = nn.Linear(512, config.embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode satellite imagery.

        Args:
            images: [batch, n_bands, height, width] multi-spectral imagery

        Returns:
            embeddings: [batch, embedding_dim] scene embeddings
        """
        features = self.backbone(images).squeeze(-1).squeeze(-1)
        embeddings = self.projection(features)
        return F.normalize(embeddings, dim=-1)


class ChangeDetectionEncoder(nn.Module):
    """
    Detect changes between temporal satellite image pairs.

    Identifies construction, destruction, vehicle movement,
    and other activity indicators.
    """

    def __init__(self, config: GEOINTConfig):
        super().__init__()
        self.config = config

        # Siamese encoder for bi-temporal images
        self.encoder = SatelliteImageEncoder(config)

        # Change analysis network
        self.change_analyzer = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.embedding_dim)
        )

        # Change type classifier
        self.change_classifier = nn.Linear(config.embedding_dim, 10)  # Change types

    def forward(
        self,
        image_before: torch.Tensor,
        image_after: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Analyze changes between two temporal images.

        Args:
            image_before: Earlier satellite image
            image_after: Later satellite image

        Returns:
            change_embedding: Embedding capturing the change
            change_logits: Classification of change type
        """
        emb_before = self.encoder(image_before)
        emb_after = self.encoder(image_after)

        # Concatenate and analyze difference
        combined = torch.cat([emb_before, emb_after], dim=-1)
        change_embedding = self.change_analyzer(combined)
        change_embedding = F.normalize(change_embedding, dim=-1)

        # Classify change type
        change_logits = self.change_classifier(change_embedding)

        return change_embedding, change_logits


class ObjectDetectionEncoder(nn.Module):
    """
    Detect and classify objects in satellite imagery.

    Identifies military vehicles, aircraft, vessels, and
    infrastructure with fine-grained classification.
    """

    def __init__(self, config: GEOINTConfig):
        super().__init__()
        self.config = config

        # Feature pyramid network style architecture
        self.backbone = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(config.n_spectral_bands, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
        ])

        # Object embedding from ROI features
        self.roi_embed = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(),
            nn.Linear(512 * 49, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        # Classification head
        self.classifier = nn.Linear(config.embedding_dim, config.n_object_classes)

    def encode_region(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode detected object regions.

        Args:
            images: Full satellite images
            boxes: [batch, n_boxes, 4] bounding boxes (x1, y1, x2, y2)

        Returns:
            object_embeddings: [batch, n_boxes, embedding_dim]
        """
        # Extract features
        x = images
        for layer in self.backbone:
            x = layer(x)

        # For simplicity, using ROI align approximation
        # Real implementation would use torchvision.ops.roi_align
        batch_size, n_boxes = boxes.shape[:2]

        object_embeddings = []
        for b in range(batch_size):
            batch_embs = []
            for i in range(n_boxes):
                # Extract and resize ROI (simplified)
                box = boxes[b, i]
                x1, y1, x2, y2 = box.int()
                # Scale to feature map size
                scale = x.shape[-1] / images.shape[-1]
                fx1, fy1 = int(x1 * scale), int(y1 * scale)
                fx2, fy2 = int(x2 * scale), int(y2 * scale)
                fx2 = max(fx2, fx1 + 1)
                fy2 = max(fy2, fy1 + 1)

                roi = x[b:b+1, :, fy1:fy2, fx1:fx2]
                emb = self.roi_embed(roi)
                batch_embs.append(emb)

            object_embeddings.append(torch.cat(batch_embs, dim=0))

        return torch.stack(object_embeddings)

    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Detect and classify objects.

        Returns:
            embeddings: Object embeddings
            class_logits: Classification scores
        """
        embeddings = self.encode_region(images, boxes)
        embeddings = F.normalize(embeddings, dim=-1)

        class_logits = self.classifier(embeddings)

        return embeddings, class_logits


class ActivityPatternEncoder(nn.Module):
    """
    Encode activity patterns from temporal image sequences.

    Identifies patterns of activity at facilities over time
    (e.g., vehicle counts, construction progress, operational tempo).
    """

    def __init__(self, config: GEOINTConfig):
        super().__init__()
        self.config = config

        # Per-image encoder
        self.image_encoder = SatelliteImageEncoder(config)

        # Temporal modeling
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim,
                batch_first=True
            ),
            num_layers=4
        )

        # Activity pattern embedding
        self.pattern_embed = nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(
        self,
        image_sequence: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode activity pattern from image sequence.

        Args:
            image_sequence: [batch, time, channels, height, width]
            timestamps: [batch, time] observation times

        Returns:
            pattern_embedding: [batch, embedding_dim] activity pattern
        """
        batch_size, seq_len = image_sequence.shape[:2]

        # Encode each image
        images_flat = image_sequence.flatten(0, 1)
        embeddings_flat = self.image_encoder(images_flat)
        embeddings = embeddings_flat.view(batch_size, seq_len, -1)

        # Temporal modeling
        temporal_features = self.temporal_encoder(embeddings)

        # Pool to single pattern embedding
        pattern = temporal_features.mean(dim=1)
        pattern_embedding = self.pattern_embed(pattern)

        return F.normalize(pattern_embedding, dim=-1)


class GEOINTSearchSystem:
    """
    Search system for geospatial imagery archive.
    """

    def __init__(self, config: GEOINTConfig):
        self.config = config
        self.encoder = SatelliteImageEncoder(config)
        self.change_detector = ChangeDetectionEncoder(config)

        # Archive index (would be backed by vector database)
        self.archive_embeddings = None
        self.archive_metadata = None

    def search_similar_scenes(
        self,
        query_image: torch.Tensor,
        k: int = 10
    ) -> list[dict]:
        """
        Find similar scenes in imagery archive.
        """
        query_emb = self.encoder(query_image)

        if self.archive_embeddings is None:
            raise ValueError("Archive not indexed")

        similarities = F.cosine_similarity(
            query_emb, self.archive_embeddings
        )

        top_k = torch.topk(similarities, k)

        results = []
        for idx, sim in zip(top_k.indices, top_k.values):
            results.append({
                "image_id": self.archive_metadata[idx]["id"],
                "location": self.archive_metadata[idx]["location"],
                "timestamp": self.archive_metadata[idx]["timestamp"],
                "similarity": sim.item()
            })

        return results

    def detect_significant_changes(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        threshold: float = 0.3
    ) -> dict:
        """
        Detect if significant change occurred between images.
        """
        change_emb, change_logits = self.change_detector(before, after)

        # Change magnitude from embedding difference
        before_emb = self.encoder(before)
        after_emb = self.encoder(after)
        change_magnitude = 1 - F.cosine_similarity(before_emb, after_emb)

        change_probs = F.softmax(change_logits, dim=-1)
        predicted_type = torch.argmax(change_probs, dim=-1)

        return {
            "significant_change": change_magnitude.item() > threshold,
            "change_magnitude": change_magnitude.item(),
            "change_type": predicted_type.item(),
            "change_embedding": change_emb
        }
