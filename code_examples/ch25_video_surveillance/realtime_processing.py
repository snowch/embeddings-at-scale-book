from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VideoProcessingConfig:
    """Configuration for real-time video processing."""

    frame_size: int = 224
    clip_length: int = 16
    embedding_dim: int = 512
    hidden_dim: int = 1024
    fps: int = 30
    batch_size: int = 32


class FrameEncoder(nn.Module):
    """
    Efficient frame encoder for real-time processing.

    Extracts lightweight embeddings from individual frames
    for rapid scene understanding and change detection.
    """

    def __init__(self, config: VideoProcessingConfig):
        super().__init__()
        self.config = config

        # Lightweight CNN backbone (MobileNet-style)
        self.backbone = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            # Depthwise separable convolutions
            self._depthwise_block(32, 64, stride=2),
            self._depthwise_block(64, 128, stride=2),
            self._depthwise_block(128, 256, stride=2),
            self._depthwise_block(256, 512, stride=2),
            # Global pooling
            nn.AdaptiveAvgPool2d(1),
        )

        self.projection = nn.Linear(512, config.embedding_dim)

    def _depthwise_block(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> nn.Sequential:
        """Depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=stride,
                padding=1,
                groups=in_channels,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode frames to embeddings.

        Args:
            frames: [batch, 3, H, W] RGB frames

        Returns:
            embeddings: [batch, embedding_dim] frame embeddings
        """
        features = self.backbone(frames).squeeze(-1).squeeze(-1)
        embeddings = self.projection(features)
        return F.normalize(embeddings, dim=-1)


class ClipEncoder(nn.Module):
    """
    Temporal clip encoder for action understanding.

    Processes short video clips to capture motion and
    temporal patterns for activity recognition.
    """

    def __init__(self, config: VideoProcessingConfig):
        super().__init__()
        self.config = config

        # 3D CNN for spatiotemporal features
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(
                128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(
                256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )

        self.projection = nn.Linear(512, config.embedding_dim)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Encode video clips.

        Args:
            clips: [batch, 3, T, H, W] video clips

        Returns:
            embeddings: [batch, embedding_dim] clip embeddings
        """
        features = self.conv3d(clips).squeeze(-1).squeeze(-1).squeeze(-1)
        embeddings = self.projection(features)
        return F.normalize(embeddings, dim=-1)


class HierarchicalVideoProcessor(nn.Module):
    """
    Hierarchical video processing for efficiency.

    Uses fast frame-level detection to trigger
    more expensive clip-level analysis.
    """

    def __init__(self, config: VideoProcessingConfig):
        super().__init__()
        self.config = config

        self.frame_encoder = FrameEncoder(config)
        self.clip_encoder = ClipEncoder(config)

        # Fast event detector (frame-level)
        self.event_detector = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Detailed classifier (clip-level)
        self.event_classifier = nn.Linear(config.embedding_dim, 20)  # Event types

    def process_frame(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fast frame-level processing.

        Returns:
            frame_embedding: Frame representation
            event_score: Probability of interesting event
        """
        embedding = self.frame_encoder(frame)
        event_score = self.event_detector(embedding)
        return embedding, event_score

    def process_clip(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Detailed clip-level analysis.

        Returns:
            clip_embedding: Clip representation
            event_logits: Event classification scores
        """
        embedding = self.clip_encoder(clip)
        event_logits = self.event_classifier(embedding)
        return embedding, event_logits

    def forward(
        self, frames: torch.Tensor, process_clips: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Process video with optional clip analysis.

        Args:
            frames: [batch, T, 3, H, W] video frames
            process_clips: Whether to run clip-level analysis

        Returns:
            Dictionary with embeddings and scores
        """
        batch_size, n_frames = frames.shape[:2]

        # Frame-level processing
        frames_flat = frames.flatten(0, 1)  # [batch*T, 3, H, W]
        frame_embs = self.frame_encoder(frames_flat)
        frame_embs = frame_embs.view(batch_size, n_frames, -1)

        event_scores = self.event_detector(frame_embs)

        results = {"frame_embeddings": frame_embs, "event_scores": event_scores}

        if process_clips:
            # Reshape for 3D conv: [batch, 3, T, H, W]
            clips = frames.transpose(1, 2)
            clip_embs, event_logits = self.process_clip(clips)
            results["clip_embeddings"] = clip_embs
            results["event_logits"] = event_logits

        return results


class StreamProcessor:
    """
    Production video stream processor.

    Manages multiple camera streams with efficient
    batching and resource allocation.
    """

    def __init__(self, config: VideoProcessingConfig):
        self.config = config
        self.processor = HierarchicalVideoProcessor(config)

        # Detection threshold for triggering clip analysis
        self.event_threshold = 0.5

        # Frame buffers per camera
        self.frame_buffers: dict[str, list[torch.Tensor]] = {}

    def process_stream_batch(
        self, camera_id: str, frames: torch.Tensor
    ) -> list[dict]:
        """
        Process batch of frames from a camera stream.

        Args:
            camera_id: Unique camera identifier
            frames: [batch, 3, H, W] frames from stream

        Returns:
            List of detection results
        """
        results = []

        # Frame-level detection
        frame_embs, event_scores = self.processor.process_frame(frames)

        for i, (emb, score) in enumerate(zip(frame_embs, event_scores)):
            result = {
                "camera_id": camera_id,
                "frame_index": i,
                "embedding": emb,
                "event_score": score.item(),
            }

            # Trigger clip analysis if event detected
            if score.item() > self.event_threshold:
                result["triggered_analysis"] = True
                # In production, would queue for clip analysis

            results.append(result)

        return results

    def get_processing_stats(self) -> dict:
        """Get processing statistics."""
        return {
            "event_threshold": self.event_threshold,
            "active_cameras": len(self.frame_buffers),
        }
