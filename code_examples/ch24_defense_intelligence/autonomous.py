from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AutonomousConfig:
    """Configuration for autonomous systems embedding models."""
    lidar_points: int = 16384
    image_size: int = 256
    radar_bins: int = 256
    embedding_dim: int = 512
    hidden_dim: int = 1024
    n_semantic_classes: int = 20


class MultiSensorFusion(nn.Module):
    """
    Fuse multiple sensor modalities for robust perception.

    Combines camera, lidar, and radar for reliable sensing
    in degraded environments.
    """

    def __init__(self, config: AutonomousConfig):
        super().__init__()
        self.config = config

        # Camera encoder
        self.camera_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8)
        )
        self.camera_projection = nn.Linear(256 * 64, config.embedding_dim)

        # LiDAR encoder (point cloud)
        self.lidar_encoder = PointCloudEncoder(config)

        # Radar encoder
        self.radar_encoder = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding=3),  # range, velocity, azimuth, elevation
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        self.radar_projection = nn.Linear(256 * 16, config.embedding_dim)

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            config.embedding_dim, num_heads=8, batch_first=True
        )

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(config.embedding_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        # Confidence estimation per modality
        self.confidence_heads = nn.ModuleDict({
            'camera': nn.Linear(config.embedding_dim, 1),
            'lidar': nn.Linear(config.embedding_dim, 1),
            'radar': nn.Linear(config.embedding_dim, 1)
        })

    def forward(
        self,
        camera: Optional[torch.Tensor] = None,
        lidar: Optional[torch.Tensor] = None,
        radar: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Fuse available sensor modalities.

        Args:
            camera: [batch, 3, H, W] camera images
            lidar: [batch, N, 3] point cloud (x, y, z)
            radar: [batch, 4, M] radar detections

        Returns:
            scene_embedding: Fused scene representation
            confidences: Per-modality confidence scores
        """
        embeddings = []
        modalities = []
        confidences = {}

        if camera is not None:
            cam_feat = self.camera_encoder(camera).flatten(1)
            cam_emb = self.camera_projection(cam_feat)
            cam_emb = F.normalize(cam_emb, dim=-1)
            embeddings.append(cam_emb)
            modalities.append('camera')
            confidences['camera'] = torch.sigmoid(
                self.confidence_heads['camera'](cam_emb)
            )

        if lidar is not None:
            lidar_emb = self.lidar_encoder(lidar)
            embeddings.append(lidar_emb)
            modalities.append('lidar')
            confidences['lidar'] = torch.sigmoid(
                self.confidence_heads['lidar'](lidar_emb)
            )

        if radar is not None:
            radar_feat = self.radar_encoder(radar).flatten(1)
            radar_emb = self.radar_projection(radar_feat)
            radar_emb = F.normalize(radar_emb, dim=-1)
            embeddings.append(radar_emb)
            modalities.append('radar')
            confidences['radar'] = torch.sigmoid(
                self.confidence_heads['radar'](radar_emb)
            )

        if len(embeddings) == 0:
            raise ValueError("At least one sensor modality required")

        if len(embeddings) == 1:
            return embeddings[0], confidences

        # Stack for attention
        stacked = torch.stack(embeddings, dim=1)

        # Cross-modal attention
        attended, _ = self.cross_attention(stacked, stacked, stacked)

        # Weighted fusion by confidence
        weights = torch.cat([confidences[m] for m in modalities], dim=-1)
        weights = F.softmax(weights, dim=-1).unsqueeze(-1)

        fused = (attended * weights).sum(dim=1)

        return F.normalize(fused, dim=-1), confidences


class PointCloudEncoder(nn.Module):
    """
    Encode 3D point clouds from LiDAR.
    """

    def __init__(self, config: AutonomousConfig):
        super().__init__()
        self.config = config

        # PointNet-style architecture
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        # Global feature aggregation
        self.global_encoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, config.embedding_dim)
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Encode point cloud.

        Args:
            points: [batch, N, 3] xyz coordinates

        Returns:
            embedding: [batch, embedding_dim]
        """
        # Per-point features
        point_features = self.point_encoder(points)

        # Global max pooling
        global_features = point_features.max(dim=1)[0]

        # Project to embedding
        embedding = self.global_encoder(global_features)
        return F.normalize(embedding, dim=-1)


class TerrainEncoder(nn.Module):
    """
    Encode terrain for navigation in GPS-denied environments.

    Learns terrain representations for localization
    via terrain matching.
    """

    def __init__(self, config: AutonomousConfig):
        super().__init__()
        self.config = config

        # Elevation/terrain encoder
        self.terrain_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # Elevation map
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)
        )

        self.projection = nn.Linear(256 * 16, config.embedding_dim)

    def forward(self, elevation_map: torch.Tensor) -> torch.Tensor:
        """
        Encode terrain patch.

        Args:
            elevation_map: [batch, 1, H, W] elevation data

        Returns:
            terrain_embedding: [batch, embedding_dim]
        """
        features = self.terrain_encoder(elevation_map).flatten(1)
        embedding = self.projection(features)
        return F.normalize(embedding, dim=-1)


class MissionContextEncoder(nn.Module):
    """
    Encode mission context for decision making.

    Represents mission objectives, constraints, and
    current state for context-aware autonomy.
    """

    def __init__(self, config: AutonomousConfig):
        super().__init__()
        self.config = config

        # Mission parameters encoder
        self.mission_encoder = nn.Sequential(
            nn.Linear(100, config.hidden_dim),  # Mission parameters
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        # Constraint encoder
        self.constraint_encoder = nn.Sequential(
            nn.Linear(50, config.hidden_dim),  # ROE, boundaries, etc.
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        # State encoder (platform status, resources)
        self.state_encoder = nn.Sequential(
            nn.Linear(30, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.embedding_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

    def forward(
        self,
        mission_params: torch.Tensor,
        constraints: torch.Tensor,
        platform_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode mission context.

        Args:
            mission_params: Mission objectives and parameters
            constraints: Rules of engagement, boundaries
            platform_state: Current platform status

        Returns:
            context_embedding: Mission context representation
        """
        mission_emb = F.normalize(self.mission_encoder(mission_params), dim=-1)
        constraint_emb = F.normalize(self.constraint_encoder(constraints), dim=-1)
        state_emb = F.normalize(self.state_encoder(platform_state), dim=-1)

        combined = torch.cat([mission_emb, constraint_emb, state_emb], dim=-1)
        context_emb = self.fusion(combined)

        return F.normalize(context_emb, dim=-1)


class MultiAgentCoordination(nn.Module):
    """
    Coordinate multiple autonomous agents.

    Learns team representations for distributed
    task allocation and deconfliction.
    """

    def __init__(self, config: AutonomousConfig):
        super().__init__()
        self.config = config

        # Agent state encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(50, config.hidden_dim),  # Position, heading, status, capabilities
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        # Team coordination via attention
        self.team_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim,
                batch_first=True
            ),
            num_layers=4
        )

        # Task assignment head
        self.task_head = nn.Linear(config.embedding_dim, 10)  # Task types

    def forward(
        self,
        agent_states: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Coordinate agent team.

        Args:
            agent_states: [batch, n_agents, 50] agent states
            agent_mask: [batch, n_agents] valid agent mask

        Returns:
            agent_embeddings: Coordinated agent representations
            task_assignments: Suggested task assignments
        """
        # Encode individual agents
        agent_emb = self.agent_encoder(agent_states)

        # Team coordination
        if agent_mask is not None:
            coordinated = self.team_attention(
                agent_emb, src_key_padding_mask=~agent_mask.bool()
            )
        else:
            coordinated = self.team_attention(agent_emb)

        coordinated = F.normalize(coordinated, dim=-1)

        # Task assignments
        task_logits = self.task_head(coordinated)

        return coordinated, task_logits


class AutonomousNavigationSystem:
    """
    Navigation system for autonomous platforms.
    """

    def __init__(self, config: AutonomousConfig):
        self.config = config
        self.sensor_fusion = MultiSensorFusion(config)
        self.terrain_encoder = TerrainEncoder(config)
        self.context_encoder = MissionContextEncoder(config)

        # Reference terrain database for localization
        self.terrain_database = None
        self.terrain_positions = None

    def localize(
        self,
        observed_terrain: torch.Tensor,
        search_radius_km: float = 10.0
    ) -> dict:
        """
        Localize using terrain matching (GPS-denied).
        """
        if self.terrain_database is None:
            raise ValueError("Terrain database not loaded")

        # Encode observed terrain
        obs_emb = self.terrain_encoder(observed_terrain)

        # Match against database
        similarities = F.cosine_similarity(
            obs_emb, self.terrain_database
        )

        # Find best match
        best_idx = similarities.argmax()
        best_sim = similarities[best_idx]

        return {
            "position": self.terrain_positions[best_idx],
            "confidence": best_sim.item(),
            "match_index": best_idx.item()
        }

    def plan_path(
        self,
        current_embedding: torch.Tensor,
        goal_embedding: torch.Tensor,
        context_embedding: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Plan path considering mission context.

        Returns list of waypoint embeddings.
        """
        # Simplified planning - real system would use full planner
        # Interpolate between current and goal
        n_waypoints = 10
        waypoints = []

        for i in range(n_waypoints):
            alpha = i / (n_waypoints - 1)
            waypoint = (1 - alpha) * current_embedding + alpha * goal_embedding
            waypoint = F.normalize(waypoint, dim=-1)
            waypoints.append(waypoint)

        return waypoints
