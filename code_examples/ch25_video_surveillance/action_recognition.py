from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActionConfig:
    """Configuration for action recognition."""

    clip_length: int = 16
    frame_size: int = 224
    embedding_dim: int = 512
    hidden_dim: int = 1024
    n_action_classes: int = 50


class TwoStreamEncoder(nn.Module):
    """
    Two-stream action recognition encoder.

    Combines RGB (appearance) and optical flow (motion)
    streams for robust action understanding.
    """

    def __init__(self, config: ActionConfig):
        super().__init__()
        self.config = config

        # RGB stream (spatial)
        self.rgb_encoder = nn.Sequential(
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

        # Optical flow stream (temporal)
        # Flow has 2 channels (u, v) * clip_length frames
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(2 * config.clip_length, 64, 7, stride=2, padding=3),
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

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(512, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

    def forward(
        self,
        rgb_frames: torch.Tensor,
        flow_stack: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode action from RGB and flow.

        Args:
            rgb_frames: [batch, 3, H, W] sampled RGB frame
            flow_stack: [batch, 2*T, H, W] stacked optical flow

        Returns:
            embedding: [batch, embedding_dim] action embedding
        """
        rgb_feat = self.rgb_encoder(rgb_frames).squeeze(-1).squeeze(-1)
        flow_feat = self.flow_encoder(flow_stack).squeeze(-1).squeeze(-1)

        combined = torch.cat([rgb_feat, flow_feat], dim=-1)
        embedding = self.fusion(combined)

        return F.normalize(embedding, dim=-1)


class SlowFastEncoder(nn.Module):
    """
    SlowFast action recognition encoder.

    Combines slow pathway (high resolution, low frame rate)
    with fast pathway (low resolution, high frame rate).
    """

    def __init__(self, config: ActionConfig):
        super().__init__()
        self.config = config

        # Slow pathway: high spatial, low temporal
        self.slow_pathway = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )

        # Fast pathway: low spatial, high temporal
        self.fast_pathway = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + 32, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

    def forward(
        self,
        slow_input: torch.Tensor,
        fast_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode action from slow and fast pathways.

        Args:
            slow_input: [batch, 3, T_slow, H, W] low frame rate video
            fast_input: [batch, 3, T_fast, H, W] high frame rate video

        Returns:
            embedding: [batch, embedding_dim] action embedding
        """
        slow_feat = self.slow_pathway(slow_input).flatten(1)
        fast_feat = self.fast_pathway(fast_input).flatten(1)

        combined = torch.cat([slow_feat, fast_feat], dim=-1)
        embedding = self.fusion(combined)

        return F.normalize(embedding, dim=-1)


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer-based temporal encoder for actions.

    Uses self-attention to capture long-range temporal
    dependencies in video sequences.
    """

    def __init__(self, config: ActionConfig):
        super().__init__()
        self.config = config

        # Frame encoder (shared across frames)
        self.frame_encoder = nn.Sequential(
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

        # Project to transformer dimension
        self.input_projection = nn.Linear(256, config.embedding_dim)

        # Learnable temporal position encoding
        self.temporal_pos = nn.Parameter(
            torch.randn(1, config.clip_length, config.embedding_dim) * 0.02
        )

        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # CLS token for action representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embedding_dim) * 0.02)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video clip with temporal transformer.

        Args:
            video: [batch, T, 3, H, W] video clip

        Returns:
            embedding: [batch, embedding_dim] action embedding
        """
        batch_size, n_frames = video.shape[:2]

        # Encode each frame
        frames_flat = video.flatten(0, 1)  # [batch*T, 3, H, W]
        frame_feats = self.frame_encoder(frames_flat).squeeze(-1).squeeze(-1)
        frame_feats = frame_feats.view(batch_size, n_frames, -1)

        # Project to embedding dim
        frame_embs = self.input_projection(frame_feats)

        # Add temporal position encoding
        frame_embs = frame_embs + self.temporal_pos[:, :n_frames]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, frame_embs], dim=1)

        # Temporal transformer
        output = self.transformer(tokens)

        # Use CLS token as action embedding
        embedding = output[:, 0]

        return F.normalize(embedding, dim=-1)


class ActionRecognitionSystem:
    """
    Complete action recognition system.

    Supports multiple action categories and provides
    structured output for downstream processing.
    """

    def __init__(self, config: ActionConfig):
        self.config = config
        self.encoder = TemporalTransformerEncoder(config)

        # Action classifier
        self.classifier = nn.Linear(config.embedding_dim, config.n_action_classes)

        # Action category names (example)
        self.action_names = [
            "walking",
            "running",
            "standing",
            "sitting",
            "falling",
            "fighting",
            "shoplifting",
            "loitering",
            "tailgating",
            "climbing",
            # ... more actions
        ]

    def classify_action(
        self,
        video_clip: torch.Tensor,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Classify action in video clip.

        Args:
            video_clip: [batch, T, 3, H, W] video clip
            top_k: Number of top predictions to return

        Returns:
            List of predictions with action name and confidence
        """
        embedding = self.encoder(video_clip)
        logits = self.classifier(embedding)
        probs = F.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

        predictions = []
        for i in range(video_clip.shape[0]):
            batch_preds = []
            for j in range(top_k):
                idx = top_indices[i, j].item()
                action_name = (
                    self.action_names[idx]
                    if idx < len(self.action_names)
                    else f"action_{idx}"
                )
                batch_preds.append({
                    "action": action_name,
                    "confidence": top_probs[i, j].item(),
                    "action_id": idx,
                })
            predictions.append(batch_preds)

        return predictions

    def detect_safety_violations(
        self,
        video_clip: torch.Tensor,
        violation_actions: Optional[list[int]] = None,
        threshold: float = 0.5,
    ) -> list[dict]:
        """
        Detect safety-related actions.

        Args:
            video_clip: Video clip to analyze
            violation_actions: List of action IDs considered violations
            threshold: Confidence threshold for detection

        Returns:
            List of detected violations
        """
        if violation_actions is None:
            # Default safety-relevant actions
            violation_actions = [4, 5, 9]  # falling, fighting, climbing

        embedding = self.encoder(video_clip)
        logits = self.classifier(embedding)
        probs = F.softmax(logits, dim=-1)

        violations = []
        for i in range(video_clip.shape[0]):
            for action_id in violation_actions:
                if probs[i, action_id] >= threshold:
                    action_name = (
                        self.action_names[action_id]
                        if action_id < len(self.action_names)
                        else f"action_{action_id}"
                    )
                    violations.append({
                        "batch_idx": i,
                        "action": action_name,
                        "confidence": probs[i, action_id].item(),
                        "action_id": action_id,
                    })

        return violations
