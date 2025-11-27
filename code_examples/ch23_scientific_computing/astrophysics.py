from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AstroEmbeddingConfig:
    """Configuration for astronomical embedding models."""
    image_size: int = 64  # Galaxy image size
    n_bands: int = 5  # Multi-band imaging (u, g, r, i, z)
    embedding_dim: int = 256
    n_spectral_bins: int = 4096  # Spectral resolution
    lightcurve_length: int = 1000  # Time series points
    hidden_dim: int = 512


class GalaxyMorphologyEncoder(nn.Module):
    """
    Encode galaxy images into embeddings for morphological classification.

    Handles multi-band astronomical imaging with rotation invariance
    for galaxy morphology (spiral, elliptical, irregular, merger).
    """

    def __init__(self, config: AstroEmbeddingConfig):
        super().__init__()
        self.config = config

        # Multi-band image encoder with rotation augmentation awareness
        self.conv_layers = nn.Sequential(
            nn.Conv2d(config.n_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )

        self.projection = nn.Sequential(
            nn.Linear(512, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode multi-band galaxy images.

        Args:
            images: [batch, n_bands, height, width] multi-band images

        Returns:
            embeddings: [batch, embedding_dim] galaxy embeddings
        """
        features = self.conv_layers(images)
        features = features.squeeze(-1).squeeze(-1)
        embeddings = self.projection(features)
        return F.normalize(embeddings, dim=-1)


class StellarSpectrumEncoder(nn.Module):
    """
    Encode stellar spectra for classification and parameter estimation.

    Learns representations that capture stellar temperature, metallicity,
    surface gravity, and chemical abundances from spectroscopic data.
    """

    def __init__(self, config: AstroEmbeddingConfig):
        super().__init__()
        self.config = config

        # 1D convolutions for spectral feature extraction
        self.spectral_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )

        # Attention over spectral regions (emission/absorption lines)
        self.attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)

        self.projection = nn.Sequential(
            nn.Linear(256 * 32, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

    def forward(
        self,
        spectra: torch.Tensor,
        wavelength_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode stellar spectra.

        Args:
            spectra: [batch, n_spectral_bins] flux values
            wavelength_mask: [batch, n_spectral_bins] mask for valid wavelengths

        Returns:
            embeddings: [batch, embedding_dim] spectral embeddings
        """
        # Add channel dimension
        x = spectra.unsqueeze(1)

        # Spectral convolutions
        x = self.spectral_conv(x)  # [batch, 256, 32]

        # Self-attention over spectral regions
        x = x.transpose(1, 2)  # [batch, 32, 256]
        x, _ = self.attention(x, x, x)

        # Project to embedding
        x = x.flatten(1)
        embeddings = self.projection(x)
        return F.normalize(embeddings, dim=-1)


class TransientLightCurveEncoder(nn.Module):
    """
    Encode astronomical light curves for transient classification.

    Handles irregularly sampled time series from survey telescopes
    for classification of supernovae, variable stars, and other transients.
    """

    def __init__(self, config: AstroEmbeddingConfig):
        super().__init__()
        self.config = config

        # Embed time, magnitude, and error
        self.input_projection = nn.Linear(3, 64)  # (time, mag, error)

        # Transformer for irregular time series
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, 64))

        self.projection = nn.Linear(64, config.embedding_dim)

    def forward(
        self,
        times: torch.Tensor,
        magnitudes: torch.Tensor,
        errors: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode light curves with irregular sampling.

        Args:
            times: [batch, seq_len] observation times (MJD)
            magnitudes: [batch, seq_len] observed magnitudes
            errors: [batch, seq_len] measurement uncertainties
            mask: [batch, seq_len] valid observation mask

        Returns:
            embeddings: [batch, embedding_dim] light curve embeddings
        """
        batch_size = times.shape[0]

        # Stack inputs and project
        x = torch.stack([times, magnitudes, errors], dim=-1)
        x = self.input_projection(x)

        # Prepend classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transform with attention
        if mask is not None:
            # Extend mask for cls token
            cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([cls_mask, mask], dim=1)
            x = self.transformer(x, src_key_padding_mask=~mask.bool())
        else:
            x = self.transformer(x)

        # Use cls token as representation
        cls_output = x[:, 0]
        embeddings = self.projection(cls_output)
        return F.normalize(embeddings, dim=-1)


class AstronomicalSearchSystem:
    """
    End-to-end system for astronomical object search and classification.
    """

    def __init__(self, config: AstroEmbeddingConfig):
        self.config = config
        self.galaxy_encoder = GalaxyMorphologyEncoder(config)
        self.spectrum_encoder = StellarSpectrumEncoder(config)
        self.lightcurve_encoder = TransientLightCurveEncoder(config)

        # Object catalog embeddings (would be loaded from database)
        self.catalog_embeddings = None
        self.catalog_metadata = None

    def classify_transient(
        self,
        times: torch.Tensor,
        mags: torch.Tensor,
        errors: torch.Tensor,
        reference_embeddings: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """
        Classify a transient event by comparing to reference classes.

        Args:
            times, mags, errors: Light curve observations
            reference_embeddings: Dict mapping class names to prototype embeddings

        Returns:
            Class probabilities
        """
        # Encode the light curve
        embedding = self.lightcurve_encoder(
            times.unsqueeze(0), mags.unsqueeze(0), errors.unsqueeze(0)
        )

        # Compare to reference classes
        similarities = {}
        for class_name, ref_emb in reference_embeddings.items():
            sim = F.cosine_similarity(embedding, ref_emb.unsqueeze(0))
            similarities[class_name] = sim.item()

        # Convert to probabilities
        total = sum(max(0, s) for s in similarities.values())
        if total > 0:
            probs = {k: max(0, v) / total for k, v in similarities.items()}
        else:
            probs = {k: 1/len(similarities) for k in similarities}

        return probs

    def find_similar_objects(
        self,
        query_embedding: torch.Tensor,
        k: int = 10
    ) -> list[dict]:
        """
        Find k most similar astronomical objects in catalog.
        """
        if self.catalog_embeddings is None:
            raise ValueError("Catalog not loaded")

        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.catalog_embeddings
        )

        top_k = torch.topk(similarities, k)

        results = []
        for idx, sim in zip(top_k.indices, top_k.values):
            results.append({
                "object_id": self.catalog_metadata[idx]["id"],
                "similarity": sim.item(),
                "metadata": self.catalog_metadata[idx]
            })

        return results
