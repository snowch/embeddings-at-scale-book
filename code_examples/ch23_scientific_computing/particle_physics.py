import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParticlePhysicsConfig:
    """Configuration for particle physics embedding models."""
    particle_features: int = 7  # pt, eta, phi, E, charge, pid, etc.
    max_particles: int = 128
    hidden_dim: int = 256
    embedding_dim: int = 128
    n_heads: int = 8
    n_layers: int = 6


class ParticleFeatureEmbedding(nn.Module):
    """
    Embed particle-level features for collision events.

    Handles continuous kinematic features (pt, eta, phi, E)
    and categorical features (particle ID, charge).
    """

    def __init__(self, config: ParticlePhysicsConfig):
        super().__init__()

        # Kinematic feature projection (pt, eta, phi, E)
        self.kinematic_embed = nn.Linear(4, config.hidden_dim // 2)

        # Particle ID embedding (electron, muon, photon, jet, etc.)
        self.pid_embed = nn.Embedding(20, config.hidden_dim // 4)

        # Charge embedding (-1, 0, +1)
        self.charge_embed = nn.Embedding(3, config.hidden_dim // 4)

    def forward(
        self,
        kinematics: torch.Tensor,
        particle_ids: torch.Tensor,
        charges: torch.Tensor
    ) -> torch.Tensor:
        """
        Embed particle features.

        Args:
            kinematics: [batch, n_particles, 4] (pt, eta, phi, E)
            particle_ids: [batch, n_particles] particle type
            charges: [batch, n_particles] charge (-1, 0, 1) -> (0, 1, 2)

        Returns:
            embeddings: [batch, n_particles, hidden_dim]
        """
        kin_emb = self.kinematic_embed(kinematics)
        pid_emb = self.pid_embed(particle_ids)
        charge_emb = self.charge_embed(charges + 1)  # Shift to 0-indexed

        return torch.cat([kin_emb, pid_emb, charge_emb], dim=-1)


class ParticleCloudEncoder(nn.Module):
    """
    Encode collision events as point clouds.

    Based on ParticleNet/LorentzNet, handles variable-length
    sets of particles with permutation invariance.
    """

    def __init__(self, config: ParticlePhysicsConfig):
        super().__init__()
        self.config = config

        # Particle embedding
        self.particle_embed = ParticleFeatureEmbedding(config)

        # Learnable CLS token for event-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))

        # Transformer encoder (permutation equivariant)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Project to embedding dimension
        self.projection = nn.Linear(config.hidden_dim, config.embedding_dim)

    def forward(
        self,
        kinematics: torch.Tensor,
        particle_ids: torch.Tensor,
        charges: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode collision event.

        Args:
            kinematics: [batch, max_particles, 4] particle 4-vectors
            particle_ids: [batch, max_particles] particle types
            charges: [batch, max_particles] particle charges
            mask: [batch, max_particles] valid particle mask

        Returns:
            embeddings: [batch, embedding_dim] event embeddings
        """
        batch_size = kinematics.shape[0]

        # Embed particles
        x = self.particle_embed(kinematics, particle_ids, charges)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Extend mask for CLS token
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([cls_mask, mask], dim=1)

        # Transform
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask.bool())
        else:
            x = self.transformer(x)

        # Use CLS token as event representation
        event_repr = x[:, 0]

        embeddings = self.projection(event_repr)
        return F.normalize(embeddings, dim=-1)


class JetEncoder(nn.Module):
    """
    Encode hadronic jets for tagging and classification.

    Jets are collimated sprays of particles from quarks/gluons.
    Jet tagging identifies the originating particle (b-quark, top, etc.)
    """

    def __init__(self, config: ParticlePhysicsConfig):
        super().__init__()
        self.config = config

        # Constituent embedding
        self.constituent_embed = nn.Linear(config.particle_features, config.hidden_dim)

        # Edge convolution layers (for local structure)
        self.edge_convs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim)
            ) for _ in range(3)
        ])

        # Global attention
        self.attention = nn.MultiheadAttention(
            config.hidden_dim, config.n_heads, batch_first=True
        )

        self.readout = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

    def forward(
        self,
        constituents: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode jet from constituents.

        Args:
            constituents: [batch, max_constituents, features] particle features
            mask: [batch, max_constituents] valid constituent mask

        Returns:
            embeddings: [batch, embedding_dim] jet embeddings
        """
        # Initial embedding
        x = self.constituent_embed(constituents)

        # Edge convolutions (simplified - full version uses kNN graph)
        for edge_conv in self.edge_convs:
            # Global pooling as proxy for edge aggregation
            global_feat = x.mean(dim=1, keepdim=True).expand_as(x)
            edge_input = torch.cat([x, global_feat], dim=-1)
            x = x + edge_conv(edge_input)

        # Self-attention
        if mask is not None:
            x, _ = self.attention(x, x, x, key_padding_mask=~mask.bool())
        else:
            x, _ = self.attention(x, x, x)

        # Global pooling
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            pooled = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        embeddings = self.readout(pooled)
        return F.normalize(embeddings, dim=-1)


class LorentzInvariantEncoder(nn.Module):
    """
    Physics-informed encoder respecting Lorentz symmetry.

    Uses Lorentz-invariant quantities (masses, angles, boosts)
    rather than raw 4-vectors for better generalization.
    """

    def __init__(self, config: ParticlePhysicsConfig):
        super().__init__()
        self.config = config

        # Invariant feature extraction
        self.invariant_mlp = nn.Sequential(
            nn.Linear(config.max_particles * 3, config.hidden_dim),  # masses, delta_R, etc.
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Combine with particle-level features
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        self.particle_encoder = ParticleCloudEncoder(config)

    def compute_invariants(self, kinematics: torch.Tensor) -> torch.Tensor:
        """
        Compute Lorentz-invariant quantities from 4-vectors.

        Args:
            kinematics: [batch, n_particles, 4] (pt, eta, phi, E)

        Returns:
            invariants: [batch, n_invariants] invariant masses, angles, etc.
        """
        batch_size = kinematics.shape[0]
        n_particles = kinematics.shape[1]

        # Convert to (px, py, pz, E)
        pt = kinematics[:, :, 0]
        eta = kinematics[:, :, 1]
        phi = kinematics[:, :, 2]
        E = kinematics[:, :, 3]

        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)

        # Invariant mass of each particle
        m2 = E**2 - px**2 - py**2 - pz**2
        masses = torch.sqrt(F.relu(m2))

        # Pairwise delta R (angular separation)
        delta_eta = eta.unsqueeze(2) - eta.unsqueeze(1)
        delta_phi = phi.unsqueeze(2) - phi.unsqueeze(1)
        # Wrap phi difference to [-pi, pi]
        delta_phi = torch.remainder(delta_phi + 3.14159, 2*3.14159) - 3.14159
        delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

        # Flatten invariants
        invariants = torch.cat([
            masses,
            delta_R[:, :, :n_particles//2].flatten(1)  # Subsample for fixed size
        ], dim=-1)

        return invariants

    def forward(
        self,
        kinematics: torch.Tensor,
        particle_ids: torch.Tensor,
        charges: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode event with Lorentz invariance.
        """
        # Compute invariant features
        invariants = self.compute_invariants(kinematics)
        invariant_emb = self.invariant_mlp(invariants)

        # Particle-level encoding
        particle_emb = self.particle_encoder(kinematics, particle_ids, charges, mask)

        # Fuse representations
        combined = torch.cat([invariant_emb, particle_emb], dim=-1)
        embeddings = self.fusion(combined)

        return F.normalize(embeddings, dim=-1)


class AnomalyDetector:
    """
    Anomaly detection for new physics searches.

    Uses embeddings to identify events that differ from
    Standard Model predictions, potentially indicating new particles.
    """

    def __init__(self, config: ParticlePhysicsConfig):
        self.encoder = ParticleCloudEncoder(config)

        # Autoencoder for reconstruction-based anomaly detection
        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.max_particles * 4)  # Reconstruct kinematics
        )

    def compute_anomaly_score(
        self,
        kinematics: torch.Tensor,
        particle_ids: torch.Tensor,
        charges: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute anomaly score for events.

        Higher scores indicate more anomalous events that
        deviate from learned Standard Model patterns.
        """
        # Encode
        embedding = self.encoder(kinematics, particle_ids, charges, mask)

        # Decode (reconstruct)
        reconstructed = self.decoder(embedding)
        reconstructed = reconstructed.view(-1, kinematics.shape[1], 4)

        # Reconstruction error as anomaly score
        if mask is not None:
            error = ((reconstructed - kinematics) ** 2).sum(dim=-1)
            error = (error * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        else:
            error = ((reconstructed - kinematics) ** 2).mean(dim=(1, 2))

        return error


class TriggerClassifier:
    """
    Real-time event classification for trigger systems.

    Must run in microseconds to decide which events to keep
    from the ~40 MHz collision rate at the LHC.
    """

    def __init__(self, config: ParticlePhysicsConfig):
        # Lightweight encoder for real-time inference
        self.encoder = nn.Sequential(
            nn.Linear(config.particle_features * 10, config.hidden_dim),  # Top 10 particles
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        self.classifier = nn.Linear(config.embedding_dim, 5)  # Signal categories

    def classify(self, top_particles: torch.Tensor) -> torch.Tensor:
        """
        Fast classification of collision events.

        Args:
            top_particles: [batch, 10, features] top 10 particles by pT

        Returns:
            logits: [batch, n_classes] classification logits
        """
        x = top_particles.flatten(1)
        embedding = F.relu(self.encoder(x))
        logits = self.classifier(embedding)
        return logits
