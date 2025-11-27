from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RetailConfig:
    """Configuration for retail video analytics."""

    frame_size: int = 224
    embedding_dim: int = 256
    hidden_dim: int = 512
    n_behavior_classes: int = 20


class ShopliftingDetector(nn.Module):
    """
    Shoplifting behavior detection.

    Identifies suspicious behaviors like concealment,
    unusual checkout patterns, and merchandise handling.
    """

    def __init__(self, config: RetailConfig):
        super().__init__()
        self.config = config

        # Spatial encoder
        self.spatial_encoder = nn.Sequential(
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

        # Temporal modeling
        self.temporal_encoder = nn.LSTM(
            input_size=256,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Behavior classifier
        self.behavior_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, config.n_behavior_classes),
        )

        # Suspicion score (binary)
        self.suspicion_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Behavior labels
        self.behavior_names = [
            "normal_shopping",
            "browsing",
            "examining_product",
            "returning_product",
            "concealment_attempt",
            "bag_stuffing",
            "ticket_switching",
            "package_manipulation",
            "lookout_behavior",
            "distraction_technique",
            "checkout_avoidance",
            "exit_without_purchase",
            "nervous_behavior",
            "repeated_visits",
            "unusual_clothing",
            # ... more behaviors
        ]

    def forward(self, video_clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Analyze video for shoplifting indicators.

        Args:
            video_clip: [batch, T, 3, H, W] video clip

        Returns:
            behavior_logits: Behavior classification
            suspicion_score: Overall suspicion level (0-1)
            embedding: Temporal embedding for further analysis
        """
        batch_size, n_frames = video_clip.shape[:2]

        # Encode each frame
        frames_flat = video_clip.flatten(0, 1)
        spatial_feats = self.spatial_encoder(frames_flat).squeeze(-1).squeeze(-1)
        spatial_feats = spatial_feats.view(batch_size, n_frames, -1)

        # Temporal modeling
        temporal_feats, _ = self.temporal_encoder(spatial_feats)

        # Use final hidden state
        embedding = temporal_feats[:, -1]

        # Classify behavior
        behavior_logits = self.behavior_classifier(embedding)

        # Compute suspicion score
        suspicion_score = self.suspicion_head(embedding)

        return behavior_logits, suspicion_score, embedding


class CustomerJourneyTracker(nn.Module):
    """
    Track customer journey through store.

    Analyzes movement patterns, dwell times, and
    interactions for retail analytics.
    """

    def __init__(self, config: RetailConfig):
        super().__init__()
        self.config = config

        # Person encoder
        self.person_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.person_projection = nn.Linear(128, config.embedding_dim)

        # Zone classifier (store areas)
        self.zone_classifier = nn.Linear(config.embedding_dim, 20)  # Store zones

        # Zone names
        self.zone_names = [
            "entrance",
            "checkout",
            "electronics",
            "clothing",
            "grocery",
            "pharmacy",
            "customer_service",
            "fitting_room",
            "exit",
            # ... more zones
        ]

    def forward(
        self, person_crop: torch.Tensor, scene_context: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode person and predict zone.

        Args:
            person_crop: [batch, 3, H, W] person detection crop
            scene_context: Optional scene context

        Returns:
            person_embedding: Person appearance embedding
            zone_logits: Zone classification
        """
        features = self.person_encoder(person_crop).squeeze(-1).squeeze(-1)
        embedding = self.person_projection(features)
        embedding = F.normalize(embedding, dim=-1)

        zone_logits = self.zone_classifier(embedding)

        return embedding, zone_logits


class QueueAnalyzer(nn.Module):
    """
    Queue and wait time analysis.

    Monitors checkout queues for length, wait times,
    and customer satisfaction indicators.
    """

    def __init__(self, config: RetailConfig):
        super().__init__()
        self.config = config

        # Scene encoder
        self.scene_encoder = nn.Sequential(
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

        # Queue length regression
        self.queue_length_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU(),  # Non-negative
        )

        # Wait time estimation (minutes)
        self.wait_time_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU(),
        )

        # Frustration indicators
        self.frustration_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, queue_region: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Analyze queue from checkout region.

        Args:
            queue_region: [batch, 3, H, W] checkout area image

        Returns:
            Dictionary with queue metrics
        """
        features = self.scene_encoder(queue_region).squeeze(-1).squeeze(-1)

        return {
            "queue_length": self.queue_length_head(features),
            "estimated_wait_minutes": self.wait_time_head(features),
            "frustration_indicator": self.frustration_head(features),
        }


class RetailAnalyticsSystem:
    """
    Complete retail video analytics system.
    """

    def __init__(self, config: RetailConfig):
        self.config = config
        self.shoplifting_detector = ShopliftingDetector(config)
        self.journey_tracker = CustomerJourneyTracker(config)
        self.queue_analyzer = QueueAnalyzer(config)

        # Alert thresholds
        self.suspicion_threshold = 0.7
        self.queue_alert_length = 5

    def analyze_customer_behavior(
        self,
        video_clip: torch.Tensor,
        camera_zone: str,
    ) -> dict:
        """
        Comprehensive customer behavior analysis.

        Args:
            video_clip: Video of customer activity
            camera_zone: Store zone where camera is located

        Returns:
            Analysis results
        """
        behavior_logits, suspicion, embedding = self.shoplifting_detector(video_clip)

        behavior_probs = F.softmax(behavior_logits, dim=-1)
        top_behavior_idx = behavior_probs.argmax(dim=-1).item()
        top_behavior = (
            self.shoplifting_detector.behavior_names[top_behavior_idx]
            if top_behavior_idx < len(self.shoplifting_detector.behavior_names)
            else f"behavior_{top_behavior_idx}"
        )

        return {
            "primary_behavior": top_behavior,
            "behavior_confidence": behavior_probs.max().item(),
            "suspicion_score": suspicion.item(),
            "requires_attention": suspicion.item() > self.suspicion_threshold,
            "camera_zone": camera_zone,
            "embedding": embedding,
        }

    def analyze_queue(
        self,
        checkout_image: torch.Tensor,
        register_id: str,
    ) -> dict:
        """
        Analyze checkout queue status.

        Args:
            checkout_image: Image of checkout area
            register_id: Register identifier

        Returns:
            Queue analysis results
        """
        metrics = self.queue_analyzer(checkout_image.unsqueeze(0))

        queue_length = metrics["queue_length"].item()
        wait_time = metrics["estimated_wait_minutes"].item()

        return {
            "register_id": register_id,
            "queue_length": int(round(queue_length)),
            "estimated_wait_minutes": round(wait_time, 1),
            "frustration_level": metrics["frustration_indicator"].item(),
            "needs_additional_register": queue_length > self.queue_alert_length,
        }

    def generate_store_insights(
        self,
        all_analyses: list[dict],
        time_period_hours: float = 1.0,
    ) -> dict:
        """
        Generate aggregate store insights.

        Args:
            all_analyses: List of individual analyses
            time_period_hours: Time period covered

        Returns:
            Aggregate insights
        """
        if not all_analyses:
            return {"error": "No analyses provided"}

        # Aggregate metrics
        suspicion_scores = [
            a.get("suspicion_score", 0) for a in all_analyses if "suspicion_score" in a
        ]
        queue_lengths = [a.get("queue_length", 0) for a in all_analyses if "queue_length" in a]

        # Count behaviors
        behavior_counts: dict[str, int] = {}
        for a in all_analyses:
            if "primary_behavior" in a:
                behavior = a["primary_behavior"]
                behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1

        return {
            "time_period_hours": time_period_hours,
            "total_analyses": len(all_analyses),
            "avg_suspicion_score": (
                sum(suspicion_scores) / len(suspicion_scores) if suspicion_scores else 0
            ),
            "high_suspicion_incidents": sum(
                1 for s in suspicion_scores if s > self.suspicion_threshold
            ),
            "avg_queue_length": (sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0),
            "max_queue_length": max(queue_lengths) if queue_lengths else 0,
            "behavior_distribution": behavior_counts,
        }
