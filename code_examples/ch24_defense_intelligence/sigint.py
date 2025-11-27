from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SIGINTConfig:
    """Configuration for signals intelligence embedding models."""
    vocab_size: int = 50000
    max_seq_length: int = 512
    embedding_dim: int = 768
    hidden_dim: int = 3072
    n_heads: int = 12
    n_layers: int = 6
    n_languages: int = 100


class MultilingualTextEncoder(nn.Module):
    """
    Encode text in any language to unified embedding space.

    Enables cross-lingual search and analysis without
    requiring translation.
    """

    def __init__(self, config: SIGINTConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embed = nn.Embedding(config.max_seq_length, config.embedding_dim)

        # Language embedding (for language-aware processing)
        self.language_embed = nn.Embedding(config.n_languages, config.embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        self.layer_norm = nn.LayerNorm(config.embedding_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        language_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text to language-agnostic embeddings.

        Args:
            input_ids: [batch, seq_len] token indices
            attention_mask: [batch, seq_len] mask for padding
            language_ids: [batch] language identifiers

        Returns:
            embeddings: [batch, embedding_dim] text embeddings
        """
        batch_size, seq_len = input_ids.shape

        # Token + position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.position_embed(positions)

        # Add language embedding if provided
        if language_ids is not None:
            lang_emb = self.language_embed(language_ids).unsqueeze(1)
            x = x + lang_emb

        # Transform
        if attention_mask is not None:
            x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        else:
            x = self.transformer(x)

        x = self.layer_norm(x)

        # Mean pooling over non-padded tokens
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            embeddings = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            embeddings = x.mean(dim=1)

        return F.normalize(embeddings, dim=-1)


class EntityEmbedding(nn.Module):
    """
    Learn embeddings for entities (persons, organizations, locations).

    Enables entity resolution across sources and languages.
    """

    def __init__(self, config: SIGINTConfig):
        super().__init__()
        self.config = config

        # Text encoder for entity names/descriptions
        self.text_encoder = MultilingualTextEncoder(config)

        # Attribute encoder (for structured entity data)
        self.attribute_encoder = nn.Sequential(
            nn.Linear(100, config.hidden_dim),  # 100 attribute features
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )

    def forward(
        self,
        name_ids: torch.Tensor,
        name_mask: torch.Tensor,
        attributes: torch.Tensor,
        language_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode entity to embedding.

        Args:
            name_ids: Tokenized entity name
            name_mask: Attention mask for name
            attributes: Structured attribute features
            language_ids: Language of the name

        Returns:
            entity_embedding: [batch, embedding_dim]
        """
        # Encode name
        name_emb = self.text_encoder(name_ids, name_mask, language_ids)

        # Encode attributes
        attr_emb = self.attribute_encoder(attributes)
        attr_emb = F.normalize(attr_emb, dim=-1)

        # Fuse representations
        combined = torch.cat([name_emb, attr_emb], dim=-1)
        entity_emb = self.fusion(combined)

        return F.normalize(entity_emb, dim=-1)


class CommunicationPatternEncoder(nn.Module):
    """
    Encode patterns in communication networks.

    Captures behavioral signatures of communication patterns
    for entity profiling and anomaly detection.
    """

    def __init__(self, config: SIGINTConfig):
        super().__init__()
        self.config = config

        # Encode communication metadata features
        # (time, duration, frequency, network position, etc.)
        self.metadata_encoder = nn.Sequential(
            nn.Linear(50, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        # Temporal pattern encoder
        self.temporal_encoder = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.embedding_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Project bidirectional output
        self.projection = nn.Linear(config.embedding_dim * 2, config.embedding_dim)

    def forward(
        self,
        comm_features: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode communication pattern.

        Args:
            comm_features: [batch, seq_len, 50] communication metadata
            sequence_mask: [batch, seq_len] valid communication mask

        Returns:
            pattern_embedding: [batch, embedding_dim]
        """
        # Encode each communication
        x = self.metadata_encoder(comm_features)

        # Temporal pattern learning
        if sequence_mask is not None:
            lengths = sequence_mask.sum(dim=1).cpu()
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )

        _, hidden = self.temporal_encoder(x)

        # Combine bidirectional hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        pattern_emb = self.projection(hidden)

        return F.normalize(pattern_emb, dim=-1)


class NetworkGraphEncoder(nn.Module):
    """
    Encode communication network structure.

    Learns node embeddings that capture network position
    and relationship patterns.
    """

    def __init__(self, config: SIGINTConfig):
        super().__init__()
        self.config = config

        # Node feature encoder
        self.node_encoder = nn.Linear(config.embedding_dim, config.embedding_dim)

        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(config.embedding_dim, config.embedding_dim, n_heads=8)
            for _ in range(3)
        ])

        self.output_projection = nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode nodes in communication network.

        Args:
            node_features: [n_nodes, embedding_dim] initial node features
            edge_index: [2, n_edges] graph connectivity

        Returns:
            node_embeddings: [n_nodes, embedding_dim]
        """
        x = self.node_encoder(node_features)

        for gat in self.gat_layers:
            x = x + gat(x, edge_index)
            x = F.relu(x)

        embeddings = self.output_projection(x)
        return F.normalize(embeddings, dim=-1)


class GraphAttentionLayer(nn.Module):
    """Simple graph attention layer."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Graph attention forward pass."""
        n_nodes = x.shape[0]
        src, dst = edge_index

        # Compute attention
        q = self.query(x).view(n_nodes, self.n_heads, self.head_dim)
        k = self.key(x).view(n_nodes, self.n_heads, self.head_dim)
        v = self.value(x).view(n_nodes, self.n_heads, self.head_dim)

        # Attention scores for edges
        q_src = q[src]  # [n_edges, n_heads, head_dim]
        k_dst = k[dst]
        attn = (q_src * k_dst).sum(dim=-1) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=0)  # Simplified softmax

        # Aggregate
        v_src = v[src]
        messages = attn.unsqueeze(-1) * v_src

        # Sum messages per destination node
        out = torch.zeros_like(v)
        out.index_add_(0, dst, messages)

        return out.view(n_nodes, -1)


class EntityResolutionSystem:
    """
    Resolve entity identities across sources.
    """

    def __init__(self, config: SIGINTConfig):
        self.config = config
        self.entity_encoder = EntityEmbedding(config)

        # Known entity database
        self.entity_embeddings = None
        self.entity_metadata = None

    def resolve_entity(
        self,
        name_ids: torch.Tensor,
        name_mask: torch.Tensor,
        attributes: torch.Tensor,
        threshold: float = 0.85
    ) -> list[dict]:
        """
        Find matching entities in database.
        """
        query_emb = self.entity_encoder(name_ids, name_mask, attributes)

        if self.entity_embeddings is None:
            return []

        similarities = F.cosine_similarity(
            query_emb.unsqueeze(0), self.entity_embeddings
        )

        # Return matches above threshold
        matches = []
        for idx, sim in enumerate(similarities[0]):
            if sim.item() >= threshold:
                matches.append({
                    "entity_id": self.entity_metadata[idx]["id"],
                    "name": self.entity_metadata[idx]["name"],
                    "confidence": sim.item()
                })

        return sorted(matches, key=lambda x: x["confidence"], reverse=True)

    def cluster_identities(
        self,
        embeddings: torch.Tensor,
        threshold: float = 0.8
    ) -> list[list[int]]:
        """
        Cluster embeddings into identity groups.

        Returns list of clusters (each cluster = same identity).
        """
        n = embeddings.shape[0]
        similarities = F.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )

        # Simple greedy clustering
        assigned = set()
        clusters = []

        for i in range(n):
            if i in assigned:
                continue

            cluster = [i]
            assigned.add(i)

            for j in range(i + 1, n):
                if j not in assigned and similarities[i, j] >= threshold:
                    cluster.append(j)
                    assigned.add(j)

            clusters.append(cluster)

        return clusters
