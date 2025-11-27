from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EcologyConfig:
    """Configuration for ecology embedding models."""
    image_size: int = 224
    audio_length: int = 10  # seconds
    sample_rate: int = 22050
    n_mels: int = 128
    sequence_length: int = 256  # DNA barcode length
    embedding_dim: int = 256
    hidden_dim: int = 512
    n_species: int = 10000  # Known species for classification


class SpeciesImageEncoder(nn.Module):
    """
    Encode species images for identification.

    Used for camera trap analysis, citizen science apps,
    and automated biodiversity monitoring.
    """

    def __init__(self, config: EcologyConfig):
        super().__init__()
        self.config = config

        # CNN backbone (simplified - would use pretrained ResNet/ViT)
        self.backbone = nn.Sequential(
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
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.projection = nn.Linear(512, config.embedding_dim)

        # Hierarchical classification heads (Kingdom -> Phylum -> ... -> Species)
        self.taxonomy_heads = nn.ModuleDict({
            'kingdom': nn.Linear(config.embedding_dim, 5),
            'phylum': nn.Linear(config.embedding_dim, 35),
            'class': nn.Linear(config.embedding_dim, 100),
            'order': nn.Linear(config.embedding_dim, 500),
            'family': nn.Linear(config.embedding_dim, 1500),
            'genus': nn.Linear(config.embedding_dim, 5000),
            'species': nn.Linear(config.embedding_dim, config.n_species)
        })

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Encode species image with hierarchical classification.

        Args:
            images: [batch, 3, height, width] RGB images

        Returns:
            embeddings: [batch, embedding_dim] species embeddings
            taxonomy_logits: dict of logits at each taxonomic level
        """
        features = self.backbone(images).squeeze(-1).squeeze(-1)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, dim=-1)

        # Hierarchical predictions
        taxonomy_logits = {
            level: head(embeddings)
            for level, head in self.taxonomy_heads.items()
        }

        return embeddings, taxonomy_logits


class BioacousticEncoder(nn.Module):
    """
    Encode audio recordings for species identification.

    Processes spectrograms to identify species from vocalizations
    (bird songs, whale calls, bat echolocation, etc.)
    """

    def __init__(self, config: EcologyConfig):
        super().__init__()
        self.config = config

        # Spectrogram dimensions
        _n_frames = int(config.audio_length * config.sample_rate / 512)  # noqa: F841 hop_length=512

        # CNN on mel spectrogram
        self.spectrogram_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.projection = nn.Sequential(
            nn.Linear(256 * 16, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """
        Encode audio spectrogram.

        Args:
            spectrograms: [batch, n_mels, n_frames] mel spectrograms

        Returns:
            embeddings: [batch, embedding_dim] audio embeddings
        """
        # Add channel dimension
        x = spectrograms.unsqueeze(1)

        features = self.spectrogram_encoder(x)
        features = features.flatten(1)

        embeddings = self.projection(features)
        return F.normalize(embeddings, dim=-1)


class DNABarcodeEncoder(nn.Module):
    """
    Encode DNA barcode sequences for species identification.

    Uses standard barcode genes (COI for animals, rbcL/matK for plants)
    to identify species from environmental DNA samples.
    """

    def __init__(self, config: EcologyConfig):
        super().__init__()
        self.config = config

        # Nucleotide embedding (A, C, G, T, N)
        self.nucleotide_embed = nn.Embedding(5, 64)

        # 1D convolutions for motif detection
        self.conv_layers = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )

        # Transformer for long-range dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.projection = nn.Linear(512, config.embedding_dim)

    def forward(
        self,
        sequences: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode DNA barcode sequence.

        Args:
            sequences: [batch, seq_len] nucleotide indices (0-4)
            mask: [batch, seq_len] valid position mask

        Returns:
            embeddings: [batch, embedding_dim] sequence embeddings
        """
        # Embed nucleotides
        x = self.nucleotide_embed(sequences)  # [batch, seq_len, 64]
        x = x.transpose(1, 2)  # [batch, 64, seq_len]

        # Convolutional feature extraction
        x = self.conv_layers(x)  # [batch, 512, 16]
        x = x.transpose(1, 2)  # [batch, 16, 512]

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        embeddings = self.projection(x)
        return F.normalize(embeddings, dim=-1)


class MultiModalSpeciesEncoder(nn.Module):
    """
    Fuse multiple modalities for robust species identification.

    Combines image, audio, and DNA evidence when available
    for more accurate and confident identification.
    """

    def __init__(self, config: EcologyConfig):
        super().__init__()
        self.config = config

        self.image_encoder = SpeciesImageEncoder(config)
        self.audio_encoder = BioacousticEncoder(config)
        self.dna_encoder = DNABarcodeEncoder(config)

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

        # Modality-specific confidence
        self.confidence_heads = nn.ModuleDict({
            'image': nn.Linear(config.embedding_dim, 1),
            'audio': nn.Linear(config.embedding_dim, 1),
            'dna': nn.Linear(config.embedding_dim, 1)
        })

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        spectrogram: Optional[torch.Tensor] = None,
        dna_sequence: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Encode species from available modalities.

        Args:
            image: Optional species image
            spectrogram: Optional audio spectrogram
            dna_sequence: Optional DNA barcode sequence

        Returns:
            embedding: Fused species embedding
            confidences: Per-modality confidence scores
        """
        embeddings = []
        modality_names = []
        confidences = {}

        if image is not None:
            img_emb, _ = self.image_encoder(image)
            embeddings.append(img_emb)
            modality_names.append('image')
            confidences['image'] = torch.sigmoid(
                self.confidence_heads['image'](img_emb)
            )

        if spectrogram is not None:
            audio_emb = self.audio_encoder(spectrogram)
            embeddings.append(audio_emb)
            modality_names.append('audio')
            confidences['audio'] = torch.sigmoid(
                self.confidence_heads['audio'](audio_emb)
            )

        if dna_sequence is not None:
            dna_emb = self.dna_encoder(dna_sequence)
            embeddings.append(dna_emb)
            modality_names.append('dna')
            confidences['dna'] = torch.sigmoid(
                self.confidence_heads['dna'](dna_emb)
            )

        if len(embeddings) == 0:
            raise ValueError("At least one modality must be provided")

        if len(embeddings) == 1:
            return embeddings[0], confidences

        # Stack embeddings for attention
        stacked = torch.stack(embeddings, dim=1)  # [batch, n_modalities, dim]

        # Cross-modal attention
        attended, _ = self.cross_attention(stacked, stacked, stacked)

        # Weighted fusion based on confidence
        weights = torch.cat([confidences[m] for m in modality_names], dim=-1)
        weights = F.softmax(weights, dim=-1).unsqueeze(-1)

        fused = (attended * weights).sum(dim=1)

        return F.normalize(fused, dim=-1), confidences


class BiodiversitySurveySystem:
    """
    End-to-end biodiversity monitoring system.

    Processes data from multiple sources (camera traps, acoustic
    recorders, eDNA samples) to assess ecosystem health.
    """

    def __init__(self, config: EcologyConfig):
        self.config = config
        self.encoder = MultiModalSpeciesEncoder(config)

        # Reference database of known species
        self.species_embeddings = None  # [n_species, embedding_dim]
        self.species_metadata = None  # List of species info

    def identify_species(
        self,
        embedding: torch.Tensor,
        k: int = 5,
        threshold: float = 0.7
    ) -> list[dict]:
        """
        Identify species from embedding.

        Args:
            embedding: Query embedding
            k: Number of candidates to return
            threshold: Minimum similarity for confident match

        Returns:
            List of candidate species with confidence scores
        """
        if self.species_embeddings is None:
            raise ValueError("Species database not loaded")

        similarities = F.cosine_similarity(
            embedding.unsqueeze(0),
            self.species_embeddings
        )

        top_k = torch.topk(similarities, k)

        candidates = []
        for idx, sim in zip(top_k.indices, top_k.values):
            if sim.item() >= threshold:
                candidates.append({
                    'species_id': self.species_metadata[idx]['id'],
                    'scientific_name': self.species_metadata[idx]['scientific_name'],
                    'common_name': self.species_metadata[idx]['common_name'],
                    'confidence': sim.item(),
                    'taxonomy': self.species_metadata[idx]['taxonomy']
                })

        return candidates

    def compute_diversity_metrics(
        self,
        site_embeddings: list[torch.Tensor]
    ) -> dict:
        """
        Compute biodiversity metrics from species detections at a site.

        Args:
            site_embeddings: List of species embeddings detected at site

        Returns:
            Diversity metrics (richness, evenness, etc.)
        """
        if len(site_embeddings) == 0:
            return {'richness': 0, 'shannon_index': 0, 'simpson_index': 0}

        # Stack embeddings
        embeddings = torch.stack(site_embeddings)

        # Cluster to estimate species count (richness)
        # Simplified - real implementation would use proper clustering
        unique_species = len(embeddings)  # Placeholder

        # Shannon diversity index (using embedding similarity as proxy)
        # H = -sum(p_i * log(p_i))
        # Simplified calculation
        shannon_index = torch.log(torch.tensor(float(unique_species)))

        # Simpson's index (probability two random individuals are same species)
        # Simplified
        simpson_index = 1.0 - (1.0 / unique_species) if unique_species > 0 else 0

        return {
            'richness': unique_species,
            'shannon_index': shannon_index.item(),
            'simpson_index': simpson_index
        }
