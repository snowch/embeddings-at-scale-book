from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PrivacyConfig:
    """Configuration for privacy-preserving analytics."""

    frame_size: int = 224
    embedding_dim: int = 256
    noise_scale: float = 0.1  # Differential privacy noise


class FaceAnonymizer(nn.Module):
    """
    Face detection and anonymization.

    Detects faces and applies blurring or replacement
    while preserving other scene content.
    """

    def __init__(self, config: PrivacyConfig):
        super().__init__()
        self.config = config

        # Simple face detector (in production, use specialized model)
        self.face_detector = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),  # Face probability map
            nn.Sigmoid(),
        )

        # Blur kernel
        self.blur_kernel_size = 21
        self.blur_sigma = 10.0

    def detect_faces(self, image: torch.Tensor) -> torch.Tensor:
        """
        Detect face regions.

        Args:
            image: [batch, 3, H, W] input image

        Returns:
            face_mask: [batch, 1, H', W'] face probability map
        """
        return self.face_detector(image)

    def apply_blur(
        self, image: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Apply Gaussian blur to masked regions.

        Args:
            image: Input image
            mask: Region mask (upsampled to image size)
            threshold: Mask threshold for blurring

        Returns:
            anonymized: Image with faces blurred
        """
        # Upsample mask to image size
        mask = F.interpolate(mask, size=image.shape[2:], mode="bilinear")
        binary_mask = (mask > threshold).float()

        # Create Gaussian kernel
        kernel_size = self.blur_kernel_size
        sigma = self.blur_sigma

        x = torch.arange(kernel_size, device=image.device) - kernel_size // 2
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_2d = gaussian_1d.outer(gaussian_1d)
        gaussian_2d = gaussian_2d / gaussian_2d.sum()
        kernel = gaussian_2d.view(1, 1, kernel_size, kernel_size)

        # Apply blur per channel
        blurred_channels = []
        for c in range(3):
            channel = image[:, c : c + 1]
            blurred = F.conv2d(channel, kernel, padding=kernel_size // 2)
            blurred_channels.append(blurred)
        blurred = torch.cat(blurred_channels, dim=1)

        # Blend original and blurred
        anonymized = image * (1 - binary_mask) + blurred * binary_mask

        return anonymized

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Anonymize faces in image.

        Args:
            image: Input image

        Returns:
            anonymized: Image with faces blurred
            face_mask: Detected face regions
        """
        face_mask = self.detect_faces(image)
        anonymized = self.apply_blur(image, face_mask)
        return anonymized, face_mask


class PrivacyPreservingEncoder(nn.Module):
    """
    Extract analytics embeddings without identifying information.

    Uses differential privacy and feature suppression to
    protect individual privacy while enabling analytics.
    """

    def __init__(self, config: PrivacyConfig):
        super().__init__()
        self.config = config

        # Scene encoder (non-biometric features)
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

        self.projection = nn.Linear(256, config.embedding_dim)

        # Privacy gradient reversal (during training)
        # Ensures embeddings don't leak identity

    def add_dp_noise(
        self, embedding: torch.Tensor, epsilon: float = 1.0
    ) -> torch.Tensor:
        """
        Add differential privacy noise.

        Args:
            embedding: Input embedding
            epsilon: Privacy budget (smaller = more private)

        Returns:
            noisy_embedding: Privacy-protected embedding
        """
        # Calibrate noise to embedding sensitivity
        sensitivity = 1.0  # Assuming normalized embeddings
        noise_scale = sensitivity / epsilon

        noise = torch.randn_like(embedding) * noise_scale
        noisy = embedding + noise

        return F.normalize(noisy, dim=-1)

    def forward(
        self, image: torch.Tensor, add_noise: bool = True
    ) -> torch.Tensor:
        """
        Extract privacy-preserving embedding.

        Args:
            image: Input image (preferably pre-anonymized)
            add_noise: Whether to add DP noise

        Returns:
            embedding: Privacy-protected embedding
        """
        features = self.scene_encoder(image).squeeze(-1).squeeze(-1)
        embedding = self.projection(features)
        embedding = F.normalize(embedding, dim=-1)

        if add_noise:
            embedding = self.add_dp_noise(embedding)

        return embedding


class OnDeviceAnalytics(nn.Module):
    """
    On-device analytics that never transmits raw video.

    Processes video locally and only sends aggregate
    statistics or anonymized metadata.
    """

    def __init__(self, config: PrivacyConfig):
        super().__init__()
        self.config = config

        # Lightweight on-device model
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Count head (people, vehicles, etc.)
        self.count_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # 5 object categories
            nn.ReLU(),
        )

        # Activity classification
        self.activity_head = nn.Linear(128, 10)

    def forward(self, frame: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Process frame on-device.

        Args:
            frame: Input video frame

        Returns:
            Aggregate statistics (no raw data)
        """
        features = self.encoder(frame).squeeze(-1).squeeze(-1)

        counts = self.count_head(features)
        activity_logits = self.activity_head(features)

        return {
            "object_counts": counts,  # [people, vehicles, ...]
            "activity_distribution": F.softmax(activity_logits, dim=-1),
            # No images, no embeddings - only statistics
        }


class PrivacyAuditLogger:
    """
    Audit logging for privacy compliance.

    Tracks all data access and processing for
    regulatory compliance and accountability.
    """

    def __init__(self):
        self.logs: list[dict] = []

    def log_access(
        self,
        user_id: str,
        action: str,
        data_type: str,
        camera_id: Optional[str] = None,
        time_range: Optional[tuple[float, float]] = None,
        purpose: Optional[str] = None,
    ) -> str:
        """
        Log data access event.

        Args:
            user_id: User performing access
            action: Type of action (view, search, export)
            data_type: Type of data accessed
            camera_id: Camera accessed (if applicable)
            time_range: Time range accessed
            purpose: Stated purpose for access

        Returns:
            log_id: Unique log entry identifier
        """
        import time
        import uuid

        log_id = str(uuid.uuid4())

        self.logs.append({
            "log_id": log_id,
            "timestamp": time.time(),
            "user_id": user_id,
            "action": action,
            "data_type": data_type,
            "camera_id": camera_id,
            "time_range": time_range,
            "purpose": purpose,
        })

        return log_id

    def log_processing(
        self,
        process_type: str,
        input_data: str,
        output_data: str,
        privacy_measures: list[str],
    ) -> str:
        """
        Log data processing event.

        Args:
            process_type: Type of processing performed
            input_data: Description of input data
            output_data: Description of output data
            privacy_measures: Privacy measures applied

        Returns:
            log_id: Unique log entry identifier
        """
        import time
        import uuid

        log_id = str(uuid.uuid4())

        self.logs.append({
            "log_id": log_id,
            "timestamp": time.time(),
            "event_type": "processing",
            "process_type": process_type,
            "input_data": input_data,
            "output_data": output_data,
            "privacy_measures": privacy_measures,
        })

        return log_id

    def generate_compliance_report(
        self,
        start_time: float,
        end_time: float,
    ) -> dict:
        """
        Generate compliance report for time period.

        Args:
            start_time: Report start timestamp
            end_time: Report end timestamp

        Returns:
            Compliance report summary
        """
        relevant_logs = [
            log
            for log in self.logs
            if start_time <= log.get("timestamp", 0) <= end_time
        ]

        # Aggregate statistics
        access_by_user: dict[str, int] = {}
        access_by_action: dict[str, int] = {}

        for log in relevant_logs:
            user = log.get("user_id", "unknown")
            action = log.get("action", "unknown")

            access_by_user[user] = access_by_user.get(user, 0) + 1
            access_by_action[action] = access_by_action.get(action, 0) + 1

        return {
            "report_period": {"start": start_time, "end": end_time},
            "total_events": len(relevant_logs),
            "access_by_user": access_by_user,
            "access_by_action": access_by_action,
            "detailed_logs": relevant_logs,
        }


class PrivacyPreservingAnalyticsSystem:
    """
    Complete privacy-preserving video analytics system.
    """

    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.anonymizer = FaceAnonymizer(config)
        self.encoder = PrivacyPreservingEncoder(config)
        self.on_device = OnDeviceAnalytics(config)
        self.audit_logger = PrivacyAuditLogger()

    def process_frame_private(
        self,
        frame: torch.Tensor,
        camera_id: str,
        anonymize: bool = True,
        add_dp_noise: bool = True,
    ) -> dict:
        """
        Process frame with privacy protections.

        Args:
            frame: Input video frame
            camera_id: Camera identifier
            anonymize: Whether to blur faces
            add_dp_noise: Whether to add differential privacy noise

        Returns:
            Privacy-protected analytics results
        """
        # Log processing
        self.audit_logger.log_processing(
            process_type="frame_analysis",
            input_data=f"frame from {camera_id}",
            output_data="aggregate statistics",
            privacy_measures=[
                "face_anonymization" if anonymize else None,
                "differential_privacy" if add_dp_noise else None,
            ],
        )

        # Anonymize if requested
        if anonymize:
            frame, _ = self.anonymizer(frame)

        # Get on-device statistics (no raw data leaves)
        stats = self.on_device(frame)

        # Get privacy-preserving embedding
        embedding = self.encoder(frame, add_noise=add_dp_noise)

        return {
            "camera_id": camera_id,
            "statistics": {
                "object_counts": stats["object_counts"].tolist(),
                "activity_distribution": stats["activity_distribution"].tolist(),
            },
            "embedding": embedding,  # DP-protected
            "privacy_measures": {
                "faces_anonymized": anonymize,
                "dp_noise_added": add_dp_noise,
            },
        }
