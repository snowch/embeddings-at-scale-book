import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ClimateEmbeddingConfig:
    """Configuration for climate and weather embedding models."""
    n_pressure_levels: int = 13  # Vertical levels
    n_surface_vars: int = 4  # 2m temp, 10m wind u/v, mslp
    n_atmospheric_vars: int = 5  # T, u, v, q, z per level
    lat_size: int = 181  # 1-degree resolution
    lon_size: int = 360
    embedding_dim: int = 512
    hidden_dim: int = 1024
    patch_size: int = 8  # For vision transformer approach


class SphericalPositionalEncoding(nn.Module):
    """
    Positional encoding for spherical Earth coordinates.

    Uses spherical harmonics-inspired encoding to properly handle
    the geometry of global climate data.
    """

    def __init__(self, d_model: int, max_lat: int = 181, max_lon: int = 360):
        super().__init__()
        self.d_model = d_model

        # Create positional encodings for lat/lon
        lat = torch.linspace(-90, 90, max_lat)
        lon = torch.linspace(0, 360, max_lon)

        # Convert to radians
        lat_rad = lat * math.pi / 180
        lon_rad = lon * math.pi / 180

        # Create encoding dimensions
        d_per_coord = d_model // 4

        # Frequency scales
        freqs = torch.exp(torch.arange(0, d_per_coord, 2) *
                         -(math.log(10000.0) / d_per_coord))

        # Latitude encoding (account for poles)
        lat_enc = torch.zeros(max_lat, d_per_coord)
        for i, f in enumerate(freqs):
            lat_enc[:, 2*i] = torch.sin(lat_rad * f)
            lat_enc[:, 2*i+1] = torch.cos(lat_rad * f)

        # Longitude encoding (periodic)
        lon_enc = torch.zeros(max_lon, d_per_coord)
        for i, f in enumerate(freqs):
            lon_enc[:, 2*i] = torch.sin(lon_rad * f)
            lon_enc[:, 2*i+1] = torch.cos(lon_rad * f)

        # Combine into full positional encoding
        pe = torch.zeros(max_lat, max_lon, d_model)
        for i in range(max_lat):
            for j in range(max_lon):
                pe[i, j, :d_per_coord] = lat_enc[i]
                pe[i, j, d_per_coord:2*d_per_coord] = lon_enc[j]
                # Cross terms for spherical geometry
                pe[i, j, 2*d_per_coord:3*d_per_coord] = lat_enc[i] * torch.cos(lon_rad[j])
                pe[i, j, 3*d_per_coord:] = lat_enc[i] * torch.sin(lon_rad[j])

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.shape[1], :x.shape[2], :x.shape[3]]


class WeatherStateEncoder(nn.Module):
    """
    Encode atmospheric state for weather prediction.

    Based on architectures like GraphCast and Pangu-Weather,
    learns representations of 3D atmospheric fields.
    """

    def __init__(self, config: ClimateEmbeddingConfig):
        super().__init__()
        self.config = config

        # Total input channels: surface + atmospheric at each level
        n_input = config.n_surface_vars + config.n_atmospheric_vars * config.n_pressure_levels

        # Patch embedding (like Vision Transformer)
        self.patch_embed = nn.Conv2d(
            n_input,
            config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        # Calculate number of patches
        n_lat_patches = config.lat_size // config.patch_size
        n_lon_patches = config.lon_size // config.patch_size
        n_patches = n_lat_patches * n_lon_patches

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, config.hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)

        # Project to embedding dimension
        self.projection = nn.Linear(config.hidden_dim, config.embedding_dim)

    def forward(
        self,
        surface_vars: torch.Tensor,
        atmospheric_vars: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode weather state.

        Args:
            surface_vars: [batch, n_surface, lat, lon] surface variables
            atmospheric_vars: [batch, n_atmos, n_levels, lat, lon] 3D fields

        Returns:
            embeddings: [batch, embedding_dim] state embeddings
        """
        _batch_size = surface_vars.shape[0]  # noqa: F841

        # Flatten atmospheric levels into channels
        atmos_flat = atmospheric_vars.flatten(1, 2)  # [batch, n_atmos*n_levels, lat, lon]

        # Concatenate surface and atmospheric
        x = torch.cat([surface_vars, atmos_flat], dim=1)

        # Patch embedding
        x = self.patch_embed(x)  # [batch, hidden, n_lat_patches, n_lon_patches]
        x = x.flatten(2).transpose(1, 2)  # [batch, n_patches, hidden]

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling over patches
        x = x.mean(dim=1)

        # Project to embedding
        embeddings = self.projection(x)
        return F.normalize(embeddings, dim=-1)


class ClimatePatternEncoder(nn.Module):
    """
    Encode climate patterns for regime identification.

    Identifies large-scale climate modes (El Nino, NAO, etc.)
    from atmospheric and oceanic state variables.
    """

    def __init__(self, config: ClimateEmbeddingConfig):
        super().__init__()
        self.config = config

        # Separate encoders for atmosphere and ocean
        self.atmos_encoder = nn.Sequential(
            nn.Conv2d(config.n_surface_vars, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8)
        )

        self.ocean_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # SST, SSH, salinity
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8)
        )

        # Fusion and projection
        self.fusion = nn.Sequential(
            nn.Linear(256 * 64 * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

    def forward(
        self,
        atmosphere: torch.Tensor,
        ocean: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode coupled atmosphere-ocean state.

        Args:
            atmosphere: [batch, n_vars, lat, lon] atmospheric fields
            ocean: [batch, 3, lat, lon] ocean fields (SST, SSH, salinity)

        Returns:
            embeddings: [batch, embedding_dim] climate state embeddings
        """
        atmos_features = self.atmos_encoder(atmosphere).flatten(1)
        ocean_features = self.ocean_encoder(ocean).flatten(1)

        combined = torch.cat([atmos_features, ocean_features], dim=-1)
        embeddings = self.fusion(combined)

        return F.normalize(embeddings, dim=-1)


class SatelliteImageEncoder(nn.Module):
    """
    Encode satellite imagery for Earth observation.

    Handles multi-spectral satellite data for applications like
    land use classification, vegetation monitoring, and ice extent.
    """

    def __init__(self, n_channels: int = 13, embedding_dim: int = 256):
        super().__init__()

        # Multi-spectral encoder (e.g., Sentinel-2 with 13 bands)
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.projection = nn.Linear(512, embedding_dim)

    def forward(
        self,
        imagery: torch.Tensor,
        cloud_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode multi-spectral satellite imagery.

        Args:
            imagery: [batch, n_channels, height, width] satellite image
            cloud_mask: [batch, 1, height, width] optional cloud mask

        Returns:
            embeddings: [batch, embedding_dim] image embeddings
        """
        if cloud_mask is not None:
            # Mask cloudy pixels (simple approach - set to mean)
            imagery = imagery * (1 - cloud_mask) + imagery.mean() * cloud_mask

        features = self.encoder(imagery).squeeze(-1).squeeze(-1)
        embeddings = self.projection(features)

        return F.normalize(embeddings, dim=-1)


class WeatherForecastSystem:
    """
    Weather forecasting system using embeddings.

    Demonstrates how learned atmospheric state embeddings
    enable efficient weather prediction.
    """

    def __init__(self, config: ClimateEmbeddingConfig):
        self.config = config
        self.encoder = WeatherStateEncoder(config)

        # Forecast model: predict next state embedding from current
        self.forecast_model = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        # Decoder: reconstruct atmospheric state from embedding
        self.decoder = None  # Would be a full decoder network

    def forecast(
        self,
        current_surface: torch.Tensor,
        current_atmos: torch.Tensor,
        forecast_hours: int = 24
    ) -> list[torch.Tensor]:
        """
        Generate weather forecast.

        Args:
            current_surface: Current surface variables
            current_atmos: Current atmospheric state
            forecast_hours: Hours to forecast (6-hour steps)

        Returns:
            List of forecast embeddings at each time step
        """
        # Encode current state
        current_emb = self.encoder(current_surface, current_atmos)

        forecasts = [current_emb]
        emb = current_emb

        # Roll out forecast
        n_steps = forecast_hours // 6
        for _ in range(n_steps):
            # Predict next state embedding
            emb = self.forecast_model(emb)
            emb = F.normalize(emb, dim=-1)
            forecasts.append(emb)

        return forecasts
