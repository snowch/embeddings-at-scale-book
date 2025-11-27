from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SmartCityConfig:
    """Configuration for smart city video analytics."""

    frame_size: int = 224
    embedding_dim: int = 256
    hidden_dim: int = 512
    n_vehicle_types: int = 10
    n_incident_types: int = 15


class TrafficAnalyzer(nn.Module):
    """
    Traffic monitoring and analysis.

    Counts vehicles, estimates speed, detects incidents,
    and analyzes traffic flow patterns.
    """

    def __init__(self, config: SmartCityConfig):
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

        # Vehicle counting (regression)
        self.vehicle_count_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU(),
        )

        # Traffic density estimation
        self.density_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 0-1 density
        )

        # Congestion classification
        self.congestion_head = nn.Linear(256, 4)  # free, light, moderate, heavy

    def forward(self, traffic_image: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Analyze traffic scene.

        Args:
            traffic_image: [batch, 3, H, W] traffic camera image

        Returns:
            Traffic metrics
        """
        features = self.scene_encoder(traffic_image).squeeze(-1).squeeze(-1)

        return {
            "vehicle_count": self.vehicle_count_head(features),
            "traffic_density": self.density_head(features),
            "congestion_logits": self.congestion_head(features),
        }


class IncidentDetector(nn.Module):
    """
    Traffic incident detection.

    Detects accidents, breakdowns, wrong-way driving,
    and other traffic incidents.
    """

    def __init__(self, config: SmartCityConfig):
        super().__init__()
        self.config = config

        # Temporal encoder for incident detection
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.temporal_model = nn.LSTM(
            input_size=128,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Incident classifier
        self.incident_classifier = nn.Linear(config.hidden_dim, config.n_incident_types)

        # Incident severity
        self.severity_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # low, medium, high
        )

        self.incident_names = [
            "normal_traffic",
            "accident",
            "breakdown",
            "wrong_way_driving",
            "debris_on_road",
            "pedestrian_on_road",
            "animal_on_road",
            "stopped_vehicle",
            "slow_vehicle",
            "emergency_vehicle",
            "construction",
            "weather_hazard",
            "signal_malfunction",
            # ... more incidents
        ]

    def forward(self, video_clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect incidents in traffic video.

        Args:
            video_clip: [batch, T, 3, H, W] traffic video

        Returns:
            incident_logits: Incident type classification
            severity_logits: Severity classification
            embedding: Temporal embedding
        """
        batch_size, n_frames = video_clip.shape[:2]

        # Encode frames
        frames_flat = video_clip.flatten(0, 1)
        frame_feats = self.frame_encoder(frames_flat).squeeze(-1).squeeze(-1)
        frame_feats = frame_feats.view(batch_size, n_frames, -1)

        # Temporal modeling
        temporal_out, (hidden, _) = self.temporal_model(frame_feats)
        embedding = hidden[-1]

        # Classify incident
        incident_logits = self.incident_classifier(embedding)
        severity_logits = self.severity_head(embedding)

        return incident_logits, severity_logits, embedding


class CrowdAnalyzer(nn.Module):
    """
    Crowd monitoring and analysis.

    Estimates crowd density, detects anomalies,
    and monitors crowd flow for public safety.
    """

    def __init__(self, config: SmartCityConfig):
        super().__init__()
        self.config = config

        # Density estimation network
        self.density_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Upsampling for density map
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),  # Density map
        )

        # Global scene analysis
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Crowd behavior classifier
        self.behavior_head = nn.Linear(128, 5)  # normal, gathering, dispersing, panic, protest

    def forward(self, crowd_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Analyze crowd scene.

        Args:
            crowd_image: [batch, 3, H, W] crowd scene image

        Returns:
            density_map: Spatial density estimation
            total_count: Estimated total people count
            behavior_logits: Crowd behavior classification
        """
        # Density map
        density_map = F.relu(self.density_encoder(crowd_image))

        # Total count from density map
        total_count = density_map.sum(dim=(1, 2, 3))

        # Scene-level features
        scene_feats = self.scene_encoder(crowd_image).squeeze(-1).squeeze(-1)
        behavior_logits = self.behavior_head(scene_feats)

        return density_map, total_count, behavior_logits


class SmartCityAnalyticsSystem:
    """
    Complete smart city video analytics system.
    """

    def __init__(self, config: SmartCityConfig):
        self.config = config
        self.traffic_analyzer = TrafficAnalyzer(config)
        self.incident_detector = IncidentDetector(config)
        self.crowd_analyzer = CrowdAnalyzer(config)

        # Alert thresholds
        self.congestion_threshold = 0.7
        self.crowd_density_threshold = 100  # people per frame

    def analyze_traffic(
        self,
        image: torch.Tensor,
        camera_id: str,
        timestamp: float,
    ) -> dict:
        """
        Analyze single traffic image.

        Args:
            image: Traffic camera image
            camera_id: Camera identifier
            timestamp: Capture timestamp

        Returns:
            Traffic analysis results
        """
        metrics = self.traffic_analyzer(image.unsqueeze(0))

        congestion_probs = F.softmax(metrics["congestion_logits"], dim=-1)
        congestion_level = ["free", "light", "moderate", "heavy"][
            congestion_probs.argmax(dim=-1).item()
        ]

        return {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "vehicle_count": int(round(metrics["vehicle_count"].item())),
            "traffic_density": metrics["traffic_density"].item(),
            "congestion_level": congestion_level,
            "congestion_confidence": congestion_probs.max().item(),
        }

    def detect_traffic_incident(
        self,
        video_clip: torch.Tensor,
        camera_id: str,
        timestamp: float,
    ) -> dict:
        """
        Detect incidents in traffic video.

        Args:
            video_clip: Video clip to analyze
            camera_id: Camera identifier
            timestamp: Start timestamp

        Returns:
            Incident detection results
        """
        incident_logits, severity_logits, _ = self.incident_detector(video_clip.unsqueeze(0))

        incident_probs = F.softmax(incident_logits, dim=-1)
        incident_idx = incident_probs.argmax(dim=-1).item()
        incident_type = (
            self.incident_detector.incident_names[incident_idx]
            if incident_idx < len(self.incident_detector.incident_names)
            else f"incident_{incident_idx}"
        )

        severity_probs = F.softmax(severity_logits, dim=-1)
        severity = ["low", "medium", "high"][severity_probs.argmax(dim=-1).item()]

        is_incident = incident_type != "normal_traffic"

        return {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "incident_detected": is_incident,
            "incident_type": incident_type,
            "incident_confidence": incident_probs.max().item(),
            "severity": severity if is_incident else None,
            "severity_confidence": severity_probs.max().item() if is_incident else None,
        }

    def analyze_crowd(
        self,
        image: torch.Tensor,
        camera_id: str,
        location: str,
    ) -> dict:
        """
        Analyze crowd scene.

        Args:
            image: Crowd scene image
            camera_id: Camera identifier
            location: Location name

        Returns:
            Crowd analysis results
        """
        density_map, total_count, behavior_logits = self.crowd_analyzer(image.unsqueeze(0))

        behavior_probs = F.softmax(behavior_logits, dim=-1)
        behavior_names = ["normal", "gathering", "dispersing", "panic", "protest"]
        behavior = behavior_names[behavior_probs.argmax(dim=-1).item()]

        count = int(round(total_count.item()))

        return {
            "camera_id": camera_id,
            "location": location,
            "estimated_count": count,
            "crowd_behavior": behavior,
            "behavior_confidence": behavior_probs.max().item(),
            "density_alert": count > self.crowd_density_threshold,
            "density_map": density_map.squeeze(0),  # For visualization
        }
