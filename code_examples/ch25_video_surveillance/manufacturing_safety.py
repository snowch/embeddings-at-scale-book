from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ManufacturingConfig:
    """Configuration for manufacturing safety analytics."""

    frame_size: int = 224
    embedding_dim: int = 256
    hidden_dim: int = 512
    n_ppe_types: int = 10
    n_safety_violations: int = 15


class PPEDetector(nn.Module):
    """
    Personal Protective Equipment detection.

    Detects presence and proper wearing of hard hats,
    safety vests, goggles, gloves, and other PPE.
    """

    def __init__(self, config: ManufacturingConfig):
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
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Multi-label PPE classifier
        self.ppe_classifier = nn.Linear(256, config.n_ppe_types)

        # PPE properly worn (per item)
        self.proper_wear_classifier = nn.Linear(256, config.n_ppe_types)

        self.ppe_names = [
            "hard_hat",
            "safety_vest",
            "safety_goggles",
            "safety_gloves",
            "safety_boots",
            "ear_protection",
            "face_shield",
            "respirator",
            "fall_harness",
            "high_visibility_clothing",
        ]

    def forward(
        self, person_crop: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Detect PPE on person.

        Args:
            person_crop: [batch, 3, H, W] person detection crop

        Returns:
            ppe_present: Binary presence of each PPE type
            properly_worn: Whether each PPE is properly worn
            ppe_status: Dict with detailed status per PPE type
        """
        features = self.person_encoder(person_crop).squeeze(-1).squeeze(-1)

        ppe_logits = self.ppe_classifier(features)
        ppe_present = torch.sigmoid(ppe_logits)

        proper_logits = self.proper_wear_classifier(features)
        properly_worn = torch.sigmoid(proper_logits)

        # Build status dictionary
        ppe_status = {}
        for i, name in enumerate(self.ppe_names):
            if i < ppe_present.shape[-1]:
                ppe_status[name] = {
                    "present": ppe_present[0, i].item() > 0.5,
                    "present_confidence": ppe_present[0, i].item(),
                    "properly_worn": properly_worn[0, i].item() > 0.5,
                    "proper_confidence": properly_worn[0, i].item(),
                }

        return ppe_present, properly_worn, ppe_status


class SafetyViolationDetector(nn.Module):
    """
    Detect safety violations in manufacturing environment.

    Identifies unsafe behaviors, zone violations,
    and equipment-related hazards.
    """

    def __init__(self, config: ManufacturingConfig):
        super().__init__()
        self.config = config

        # Temporal encoder for behavior analysis
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

        # Multi-label violation classifier
        self.violation_classifier = nn.Linear(
            config.hidden_dim, config.n_safety_violations
        )

        # Risk level assessment
        self.risk_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # low, medium, high risk
        )

        self.violation_names = [
            "no_violation",
            "missing_ppe",
            "improper_ppe",
            "restricted_zone_entry",
            "unsafe_lifting",
            "running",
            "horseplay",
            "blocked_exit",
            "unsafe_machine_operation",
            "too_close_to_equipment",
            "fall_hazard",
            "chemical_handling_violation",
            "electrical_hazard",
            "fire_hazard",
            "housekeeping_violation",
        ]

    def forward(
        self, video_clip: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect safety violations.

        Args:
            video_clip: [batch, T, 3, H, W] video clip

        Returns:
            violation_logits: Multi-label violation classification
            risk_logits: Risk level classification
            embedding: Temporal embedding
        """
        batch_size, n_frames = video_clip.shape[:2]

        # Encode frames
        frames_flat = video_clip.flatten(0, 1)
        frame_feats = self.frame_encoder(frames_flat).squeeze(-1).squeeze(-1)
        frame_feats = frame_feats.view(batch_size, n_frames, -1)

        # Temporal modeling
        _, (hidden, _) = self.temporal_model(frame_feats)
        embedding = hidden[-1]

        violation_logits = self.violation_classifier(embedding)
        risk_logits = self.risk_head(embedding)

        return violation_logits, risk_logits, embedding


class ZoneMonitor(nn.Module):
    """
    Restricted zone monitoring.

    Detects unauthorized entry into dangerous areas
    and monitors safe working distances.
    """

    def __init__(self, config: ManufacturingConfig):
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

        # Zone occupancy
        self.occupancy_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Distance estimation (to dangerous equipment)
        self.distance_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),  # Non-negative
        )

    def forward(
        self, zone_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monitor restricted zone.

        Args:
            zone_image: [batch, 3, H, W] zone camera image

        Returns:
            zone_occupied: Probability zone is occupied
            min_distance: Estimated minimum distance to danger
        """
        features = self.scene_encoder(zone_image).squeeze(-1).squeeze(-1)

        zone_occupied = self.occupancy_head(features)
        min_distance = self.distance_head(features)

        return zone_occupied, min_distance


class ManufacturingSafetySystem:
    """
    Complete manufacturing safety analytics system.
    """

    def __init__(self, config: ManufacturingConfig):
        self.config = config
        self.ppe_detector = PPEDetector(config)
        self.violation_detector = SafetyViolationDetector(config)
        self.zone_monitor = ZoneMonitor(config)

        # Required PPE per zone (example configuration)
        self.zone_ppe_requirements: dict[str, list[str]] = {
            "welding_area": [
                "hard_hat",
                "safety_goggles",
                "safety_gloves",
                "face_shield",
            ],
            "warehouse": ["hard_hat", "safety_vest", "safety_boots"],
            "chemical_storage": ["safety_goggles", "safety_gloves", "respirator"],
            "assembly_line": ["safety_goggles", "safety_gloves"],
            "construction_site": [
                "hard_hat",
                "safety_vest",
                "safety_boots",
                "safety_goggles",
            ],
        }

    def check_ppe_compliance(
        self,
        person_crop: torch.Tensor,
        zone: str,
        worker_id: Optional[str] = None,
    ) -> dict:
        """
        Check PPE compliance for a worker.

        Args:
            person_crop: Person detection crop
            zone: Work zone name
            worker_id: Optional worker identifier

        Returns:
            Compliance check results
        """
        _, _, ppe_status = self.ppe_detector(person_crop.unsqueeze(0))

        required_ppe = self.zone_ppe_requirements.get(zone, [])

        violations = []
        for ppe in required_ppe:
            if ppe in ppe_status:
                status = ppe_status[ppe]
                if not status["present"]:
                    violations.append({"ppe": ppe, "issue": "missing"})
                elif not status["properly_worn"]:
                    violations.append({"ppe": ppe, "issue": "improper"})

        return {
            "worker_id": worker_id,
            "zone": zone,
            "compliant": len(violations) == 0,
            "ppe_status": ppe_status,
            "violations": violations,
            "required_ppe": required_ppe,
        }

    def analyze_safety(
        self,
        video_clip: torch.Tensor,
        camera_id: str,
        zone: str,
        timestamp: float,
    ) -> dict:
        """
        Comprehensive safety analysis.

        Args:
            video_clip: Video clip to analyze
            camera_id: Camera identifier
            zone: Work zone
            timestamp: Capture timestamp

        Returns:
            Safety analysis results
        """
        violation_logits, risk_logits, _ = self.violation_detector(
            video_clip.unsqueeze(0)
        )

        # Get violations above threshold
        violation_probs = torch.sigmoid(violation_logits)[0]
        detected_violations = []
        for i, prob in enumerate(violation_probs):
            if prob.item() > 0.5 and i > 0:  # Skip "no_violation"
                violation_name = (
                    self.violation_detector.violation_names[i]
                    if i < len(self.violation_detector.violation_names)
                    else f"violation_{i}"
                )
                detected_violations.append({
                    "violation": violation_name,
                    "confidence": prob.item(),
                })

        # Risk assessment
        risk_probs = F.softmax(risk_logits, dim=-1)[0]
        risk_levels = ["low", "medium", "high"]
        risk_level = risk_levels[risk_probs.argmax().item()]

        return {
            "camera_id": camera_id,
            "zone": zone,
            "timestamp": timestamp,
            "violations_detected": len(detected_violations) > 0,
            "violations": detected_violations,
            "risk_level": risk_level,
            "risk_confidence": risk_probs.max().item(),
            "requires_immediate_action": risk_level == "high"
            or any(v["confidence"] > 0.8 for v in detected_violations),
        }

    def monitor_zone(
        self,
        zone_image: torch.Tensor,
        zone_name: str,
        min_safe_distance: float = 2.0,
    ) -> dict:
        """
        Monitor restricted zone status.

        Args:
            zone_image: Zone camera image
            zone_name: Zone identifier
            min_safe_distance: Minimum safe distance in meters

        Returns:
            Zone monitoring results
        """
        occupied, distance = self.zone_monitor(zone_image.unsqueeze(0))

        is_occupied = occupied.item() > 0.5
        est_distance = distance.item()
        distance_violation = est_distance < min_safe_distance and is_occupied

        return {
            "zone": zone_name,
            "occupied": is_occupied,
            "occupancy_confidence": occupied.item(),
            "estimated_distance": est_distance,
            "min_safe_distance": min_safe_distance,
            "distance_violation": distance_violation,
            "alert_level": "high" if distance_violation else ("medium" if is_occupied else "low"),
        }
