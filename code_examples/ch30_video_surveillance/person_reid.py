from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReIDConfig:
    """Configuration for person re-identification."""

    image_height: int = 256
    image_width: int = 128
    embedding_dim: int = 512
    hidden_dim: int = 1024
    n_parts: int = 6  # Body part divisions


class PartBasedReIDEncoder(nn.Module):
    """
    Part-based person re-identification encoder.

    Divides person image into horizontal stripes
    for robust matching across viewpoints.
    """

    def __init__(self, config: ReIDConfig):
        super().__init__()
        self.config = config

        # Backbone CNN
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            self._resblock(64, 128, stride=2),
            self._resblock(128, 256, stride=2),
            self._resblock(256, 512, stride=1),
        )

        # Part-based pooling
        self.part_pool = nn.AdaptiveAvgPool2d((config.n_parts, 1))

        # Part embeddings
        self.part_embeddings = nn.ModuleList(
            [nn.Linear(512, config.embedding_dim // config.n_parts) for _ in range(config.n_parts)]
        )

        # Global embedding
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_embedding = nn.Linear(512, config.embedding_dim)

    def _resblock(self, in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
        """Residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract person embeddings.

        Args:
            images: [batch, 3, H, W] person crops

        Returns:
            global_emb: Global person embedding
            part_embs: Part-level embeddings
        """
        features = self.backbone(images)

        # Global embedding
        global_feat = self.global_pool(features).squeeze(-1).squeeze(-1)
        global_emb = self.global_embedding(global_feat)
        global_emb = F.normalize(global_emb, dim=-1)

        # Part embeddings
        part_feats = self.part_pool(features).squeeze(-1)  # [batch, 512, n_parts]
        part_embs = []
        for i in range(self.config.n_parts):
            part_emb = self.part_embeddings[i](part_feats[:, :, i])
            part_embs.append(F.normalize(part_emb, dim=-1))

        part_embs = torch.stack(part_embs, dim=1)  # [batch, n_parts, dim/n_parts]

        return global_emb, part_embs


class AttentionReIDEncoder(nn.Module):
    """
    Attention-based person re-identification.

    Uses self-attention to focus on discriminative
    regions of person appearance.
    """

    def __init__(self, config: ReIDConfig):
        super().__init__()
        self.config = config

        # Backbone
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
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Spatial attention
        self.attention_conv = nn.Conv2d(512, 1, 1)

        # Projection
        self.projection = nn.Linear(512, config.embedding_dim)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract attention-weighted embeddings.

        Args:
            images: [batch, 3, H, W] person crops

        Returns:
            embedding: Person embedding
            attention_map: Spatial attention weights
        """
        features = self.backbone(images)

        # Compute attention
        attention = self.attention_conv(features)
        attention = torch.sigmoid(attention)

        # Attention-weighted pooling
        weighted = features * attention
        pooled = weighted.sum(dim=(2, 3)) / (attention.sum(dim=(2, 3)) + 1e-6)

        embedding = self.projection(pooled)
        embedding = F.normalize(embedding, dim=-1)

        return embedding, attention


class PersonTracker:
    """
    Multi-camera person tracking system.

    Maintains gallery of tracked individuals and
    matches new detections across cameras.
    """

    def __init__(self, config: ReIDConfig):
        self.config = config
        self.encoder = PartBasedReIDEncoder(config)

        # Gallery of tracked persons
        self.gallery_embeddings: list[torch.Tensor] = []
        self.gallery_metadata: list[dict] = []

        # Matching threshold
        self.match_threshold = 0.7

    def add_to_gallery(
        self,
        embedding: torch.Tensor,
        metadata: dict,
    ) -> int:
        """
        Add new person to tracking gallery.

        Returns:
            person_id: Assigned person ID
        """
        person_id = len(self.gallery_embeddings)
        self.gallery_embeddings.append(embedding)
        self.gallery_metadata.append({**metadata, "person_id": person_id})
        return person_id

    def match_person(
        self,
        query_embedding: torch.Tensor,
        camera_id: str,
        timestamp: float,
    ) -> Optional[dict]:
        """
        Match query embedding against gallery.

        Args:
            query_embedding: Embedding of detected person
            camera_id: Camera where person was detected
            timestamp: Detection timestamp

        Returns:
            Match result with person_id and confidence, or None
        """
        if len(self.gallery_embeddings) == 0:
            return None

        gallery = torch.stack(self.gallery_embeddings)
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), gallery)

        best_idx = similarities.argmax().item()
        best_sim = similarities[best_idx].item()

        if best_sim >= self.match_threshold:
            return {
                "person_id": self.gallery_metadata[best_idx]["person_id"],
                "confidence": best_sim,
                "previous_camera": self.gallery_metadata[best_idx].get("last_camera"),
                "time_gap": timestamp - self.gallery_metadata[best_idx].get("last_seen", 0),
            }

        return None

    def update_gallery(
        self,
        person_id: int,
        new_embedding: torch.Tensor,
        camera_id: str,
        timestamp: float,
        momentum: float = 0.9,
    ) -> None:
        """
        Update gallery embedding with new observation.

        Uses exponential moving average to smooth updates.
        """
        old_emb = self.gallery_embeddings[person_id]
        updated_emb = momentum * old_emb + (1 - momentum) * new_embedding
        self.gallery_embeddings[person_id] = F.normalize(updated_emb, dim=-1)

        self.gallery_metadata[person_id]["last_camera"] = camera_id
        self.gallery_metadata[person_id]["last_seen"] = timestamp

    def get_person_trajectory(self, person_id: int) -> list[dict]:
        """Get tracking history for a person."""
        if person_id >= len(self.gallery_metadata):
            return []
        return self.gallery_metadata[person_id].get("trajectory", [])


class TripletLoss(nn.Module):
    """
    Triplet loss for re-identification training.

    Pulls same-person embeddings together while
    pushing different-person embeddings apart.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor person embedding
            positive: Same person, different image
            negative: Different person

        Returns:
            loss: Triplet loss value
        """
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class HardTripletMiner:
    """
    Hard triplet mining for effective training.

    Selects challenging triplets that violate the
    margin constraint for faster convergence.
    """

    def __init__(self, margin: float = 0.3):
        self.margin = margin

    def mine_triplets(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine hard triplets from batch.

        Args:
            embeddings: [batch, dim] embeddings
            labels: [batch] person IDs

        Returns:
            anchor, positive, negative indices
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Compute pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings)

        anchors, positives, negatives = [], [], []

        for i in range(batch_size):
            # Find hardest positive (same person, largest distance)
            same_person = labels == labels[i]
            same_person[i] = False  # Exclude self

            if same_person.sum() == 0:
                continue

            pos_dists = dist_matrix[i][same_person]
            hardest_pos_idx = torch.where(same_person)[0][pos_dists.argmax()]

            # Find hardest negative (different person, smallest distance)
            diff_person = labels != labels[i]

            if diff_person.sum() == 0:
                continue

            neg_dists = dist_matrix[i][diff_person]
            hardest_neg_idx = torch.where(diff_person)[0][neg_dists.argmin()]

            # Check if this is a valid triplet (violates margin)
            if dist_matrix[i, hardest_pos_idx] - dist_matrix[i, hardest_neg_idx] + self.margin > 0:
                anchors.append(i)
                positives.append(hardest_pos_idx.item())
                negatives.append(hardest_neg_idx.item())

        if len(anchors) == 0:
            # Return dummy triplet if no hard triplets found
            return (
                torch.tensor([0], device=device),
                torch.tensor([0], device=device),
                torch.tensor([1] if batch_size > 1 else [0], device=device),
            )

        return (
            torch.tensor(anchors, device=device),
            torch.tensor(positives, device=device),
            torch.tensor(negatives, device=device),
        )
