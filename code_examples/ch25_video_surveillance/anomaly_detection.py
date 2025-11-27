from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AnomalyConfig:
    """Configuration for video anomaly detection."""

    frame_size: int = 224
    clip_length: int = 16
    embedding_dim: int = 256
    hidden_dim: int = 512
    memory_size: int = 100  # Number of normal prototypes


class VideoAutoencoder(nn.Module):
    """
    Video autoencoder for reconstruction-based anomaly detection.

    Learns to reconstruct normal video; anomalies have
    high reconstruction error.
    """

    def __init__(self, config: AnomalyConfig):
        super().__init__()
        self.config = config

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, config.embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(config.embedding_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                config.embedding_dim, 256, kernel_size=3, stride=2, padding=1,
                output_padding=1
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.ConvTranspose3d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(
                64, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(
        self, video: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode and reconstruct video.

        Args:
            video: [batch, 3, T, H, W] video clip

        Returns:
            reconstructed: Reconstructed video
            embedding: Latent representation
            reconstruction_error: Per-sample reconstruction error
        """
        embedding = self.encoder(video)
        reconstructed = self.decoder(embedding)

        # Compute reconstruction error
        error = F.mse_loss(reconstructed, video, reduction="none")
        error = error.mean(dim=(1, 2, 3, 4))  # Per-sample error

        return reconstructed, embedding, error


class FuturePredictionModel(nn.Module):
    """
    Future frame prediction for anomaly detection.

    Predicts future frames from past; anomalies are
    unpredictable and have high prediction error.
    """

    def __init__(self, config: AnomalyConfig):
        super().__init__()
        self.config = config

        # Temporal encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.ConvTranspose3d(
                256, 128, kernel_size=3, stride=(2, 2, 2), padding=1, output_padding=1
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(
                128, 64, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(
                64, 3, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)
            ),
            nn.Sigmoid(),
        )

    def forward(
        self, past_frames: torch.Tensor, future_frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future frames and compute error.

        Args:
            past_frames: [batch, 3, T_past, H, W] past video
            future_frames: [batch, 3, T_future, H, W] ground truth future

        Returns:
            predicted: Predicted future frames
            prediction_error: Per-sample prediction error
        """
        encoded = self.encoder(past_frames)
        predicted = self.predictor(encoded)

        # Compute prediction error
        error = F.mse_loss(predicted, future_frames, reduction="none")
        error = error.mean(dim=(1, 2, 3, 4))

        return predicted, error


class MemoryAugmentedAnomalyDetector(nn.Module):
    """
    Memory-augmented anomaly detection.

    Stores prototypes of normal patterns and detects
    anomalies as inputs far from all prototypes.
    """

    def __init__(self, config: AnomalyConfig):
        super().__init__()
        self.config = config

        # Video encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )

        self.projection = nn.Linear(256, config.embedding_dim)

        # Memory bank of normal prototypes
        self.memory = nn.Parameter(
            torch.randn(config.memory_size, config.embedding_dim) * 0.02
        )

        # Decoder from memory-augmented representation
        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, 256 * 2 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (256, 2, 2, 2)),
            nn.ConvTranspose3d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self, video: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect anomalies using memory augmentation.

        Args:
            video: [batch, 3, T, H, W] video clip

        Returns:
            reconstructed: Memory-augmented reconstruction
            attention_weights: Attention over memory slots
            anomaly_score: Per-sample anomaly score
        """
        # Encode input
        encoded = self.encoder(video).flatten(1)
        query = self.projection(encoded)
        query = F.normalize(query, dim=-1)

        # Normalize memory
        memory_norm = F.normalize(self.memory, dim=-1)

        # Compute attention over memory
        similarity = torch.matmul(query, memory_norm.t())  # [batch, memory_size]
        attention = F.softmax(similarity * 10, dim=-1)  # Temperature scaling

        # Read from memory
        memory_read = torch.matmul(attention, memory_norm)  # [batch, embedding_dim]

        # Reconstruct from memory
        reconstructed = self.decoder(memory_read)

        # Anomaly score: distance to nearest memory slot
        max_similarity = similarity.max(dim=-1).values
        anomaly_score = 1 - max_similarity  # Higher = more anomalous

        return reconstructed, attention, anomaly_score


class VideoAnomalyDetectionSystem:
    """
    Complete video anomaly detection system.
    """

    def __init__(self, config: AnomalyConfig):
        self.config = config

        # Multiple detection methods for robustness
        self.autoencoder = VideoAutoencoder(config)
        self.predictor = FuturePredictionModel(config)
        self.memory_detector = MemoryAugmentedAnomalyDetector(config)

        # Anomaly threshold (tuned per camera/context)
        self.threshold = 0.5

        # Context-specific baselines
        self.context_baselines: dict[str, dict] = {}

    def compute_anomaly_score(
        self,
        video: torch.Tensor,
        method: str = "ensemble",
    ) -> torch.Tensor:
        """
        Compute anomaly score for video clip.

        Args:
            video: [batch, 3, T, H, W] video clip
            method: 'autoencoder', 'predictor', 'memory', or 'ensemble'

        Returns:
            anomaly_scores: [batch] anomaly scores (higher = more anomalous)
        """
        scores = []

        if method in ["autoencoder", "ensemble"]:
            _, _, ae_error = self.autoencoder(video)
            scores.append(ae_error)

        if method in ["predictor", "ensemble"]:
            # Split into past/future
            t = video.shape[2]
            past = video[:, :, : t // 2]
            future = video[:, :, t // 2:]
            _, pred_error = self.predictor(past, future)
            scores.append(pred_error)

        if method in ["memory", "ensemble"]:
            _, _, mem_score = self.memory_detector(video)
            scores.append(mem_score)

        if len(scores) == 1:
            return scores[0]

        # Ensemble: average normalized scores
        stacked = torch.stack(scores, dim=1)
        # Normalize each method's scores
        normalized = (stacked - stacked.mean(dim=0)) / (stacked.std(dim=0) + 1e-6)
        return normalized.mean(dim=1)

    def detect_anomalies(
        self,
        video: torch.Tensor,
        camera_id: str,
        timestamp: float,
    ) -> list[dict]:
        """
        Detect anomalies in video with context.

        Args:
            video: Video clip to analyze
            camera_id: Camera identifier for context-specific threshold
            timestamp: Detection timestamp

        Returns:
            List of detected anomalies
        """
        scores = self.compute_anomaly_score(video, method="ensemble")

        # Get context-specific threshold
        threshold = self.context_baselines.get(camera_id, {}).get(
            "threshold", self.threshold
        )

        anomalies = []
        for i, score in enumerate(scores):
            if score.item() > threshold:
                anomalies.append({
                    "batch_idx": i,
                    "anomaly_score": score.item(),
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "confidence": min(1.0, (score.item() - threshold) / threshold),
                })

        return anomalies

    def update_baseline(
        self,
        camera_id: str,
        normal_scores: list[float],
        percentile: float = 95,
    ) -> None:
        """
        Update context-specific baseline from normal data.

        Args:
            camera_id: Camera identifier
            normal_scores: Anomaly scores from known-normal data
            percentile: Percentile to use as threshold
        """
        import numpy as np

        scores_array = np.array(normal_scores)
        threshold = np.percentile(scores_array, percentile)

        self.context_baselines[camera_id] = {
            "threshold": threshold,
            "mean": float(scores_array.mean()),
            "std": float(scores_array.std()),
            "n_samples": len(normal_scores),
        }
