import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class MaterialsConfig:
    """Configuration for materials science embedding models."""
    atom_features: int = 92  # One-hot for elements
    bond_features: int = 10
    hidden_dim: int = 256
    embedding_dim: int = 128
    n_conv_layers: int = 4
    n_attention_heads: int = 8


class AtomEmbedding(nn.Module):
    """
    Embed atoms based on their properties.

    Combines element identity with learned representations
    of atomic properties (electronegativity, radius, etc.)
    """

    def __init__(self, n_elements: int = 92, embedding_dim: int = 64):
        super().__init__()

        # Learnable element embeddings
        self.element_embedding = nn.Embedding(n_elements, embedding_dim)

        # Additional atomic properties (could load from periodic table)
        self.property_mlp = nn.Sequential(
            nn.Linear(8, 32),  # electronegativity, radius, etc.
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        atomic_properties: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Embed atoms.

        Args:
            atomic_numbers: [batch, n_atoms] atomic numbers (1-92)
            atomic_properties: [batch, n_atoms, n_props] optional properties

        Returns:
            atom_embeddings: [batch, n_atoms, embedding_dim]
        """
        # Element embeddings (subtract 1 for 0-indexing)
        elem_emb = self.element_embedding(atomic_numbers - 1)

        if atomic_properties is not None:
            prop_emb = self.property_mlp(atomic_properties)
            return elem_emb + prop_emb

        return elem_emb


class CrystalGraphConv(nn.Module):
    """
    Graph convolution layer for crystal structures.

    Performs message passing between atoms based on their
    connections in the crystal graph.
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()

        # Edge network
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform graph convolution.

        Args:
            node_features: [n_nodes, hidden_dim]
            edge_index: [2, n_edges] source and target node indices
            edge_features: [n_edges, edge_dim]

        Returns:
            updated_nodes: [n_nodes, hidden_dim]
        """
        src, dst = edge_index

        # Gather source and destination node features
        src_features = node_features[src]
        dst_features = node_features[dst]

        # Compute edge messages
        edge_input = torch.cat([src_features, dst_features, edge_features], dim=-1)
        edge_weights = self.edge_mlp(edge_input)
        messages = src_features * edge_weights

        # Aggregate messages per node
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, dst, messages)

        # Update nodes
        node_input = torch.cat([node_features, aggregated], dim=-1)
        updated = self.node_mlp(node_input)
        updated = self.bn(updated)

        return node_features + updated  # Residual connection


class CrystalGraphEncoder(nn.Module):
    """
    Encode crystal structures into embeddings.

    Based on CGCNN (Crystal Graph Convolutional Neural Networks),
    learns representations of periodic crystal structures for
    property prediction.
    """

    def __init__(self, config: MaterialsConfig):
        super().__init__()
        self.config = config

        # Atom embedding
        self.atom_embed = AtomEmbedding(
            n_elements=config.atom_features,
            embedding_dim=config.hidden_dim
        )

        # Edge feature projection
        self.edge_embed = nn.Linear(config.bond_features, config.hidden_dim)

        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            CrystalGraphConv(config.hidden_dim, config.hidden_dim)
            for _ in range(config.n_conv_layers)
        ])

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode crystal structure.

        Args:
            atomic_numbers: [n_atoms] atomic numbers
            edge_index: [2, n_edges] graph connectivity
            edge_features: [n_edges, edge_dim] bond distances, angles
            batch: [n_atoms] batch assignment for each atom

        Returns:
            embeddings: [batch_size, embedding_dim] crystal embeddings
        """
        # Embed atoms
        x = self.atom_embed(atomic_numbers)

        # Embed edges
        edge_attr = self.edge_embed(edge_features)

        # Graph convolutions
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)

        # Global mean pooling per crystal
        batch_size = batch.max().item() + 1
        pooled = torch.zeros(batch_size, x.shape[-1], device=x.device)
        counts = torch.zeros(batch_size, device=x.device)

        for i in range(x.shape[0]):
            pooled[batch[i]] += x[i]
            counts[batch[i]] += 1

        pooled = pooled / counts.unsqueeze(-1).clamp(min=1)

        # Project to embedding
        embeddings = self.readout(pooled)
        return F.normalize(embeddings, dim=-1)


class MolecularGraphEncoder(nn.Module):
    """
    Encode molecules as graphs for property prediction.

    Represents molecules as graphs where atoms are nodes
    and bonds are edges, learning structure-property relationships.
    """

    def __init__(self, config: MaterialsConfig):
        super().__init__()
        self.config = config

        # Atom and bond embeddings
        self.atom_embed = nn.Linear(config.atom_features, config.hidden_dim)
        self.bond_embed = nn.Linear(config.bond_features, config.hidden_dim)

        # Message passing layers with attention
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                config.hidden_dim,
                config.n_attention_heads,
                batch_first=True
            ) for _ in range(config.n_conv_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 4),
                nn.GELU(),
                nn.Linear(config.hidden_dim * 4, config.hidden_dim)
            ) for _ in range(config.n_conv_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim)
            for _ in range(config.n_conv_layers * 2)
        ])

        self.readout = nn.Linear(config.hidden_dim, config.embedding_dim)

    def forward(
        self,
        atom_features: torch.Tensor,
        adjacency: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode molecule.

        Args:
            atom_features: [batch, max_atoms, atom_features]
            adjacency: [batch, max_atoms, max_atoms] adjacency matrix
            mask: [batch, max_atoms] valid atom mask

        Returns:
            embeddings: [batch, embedding_dim] molecule embeddings
        """
        # Initial atom embeddings
        x = self.atom_embed(atom_features)

        # Create attention mask from adjacency (or use full attention)
        attn_mask = None
        if mask is not None:
            # Mask out padding atoms
            attn_mask = ~mask.unsqueeze(1).expand(-1, mask.shape[1], -1)

        # Message passing with attention
        for i, (attn, ffn) in enumerate(zip(self.attention_layers, self.ffn_layers)):
            # Self-attention
            residual = x
            x = self.layer_norms[2*i](x)
            x, _ = attn(x, x, x, key_padding_mask=~mask if mask is not None else None)
            x = residual + x

            # Feed-forward
            residual = x
            x = self.layer_norms[2*i + 1](x)
            x = residual + ffn(x)

        # Global pooling (mean over valid atoms)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            pooled = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        embeddings = self.readout(pooled)
        return F.normalize(embeddings, dim=-1)


class MaterialsPropertyPredictor:
    """
    Property prediction system for materials discovery.

    Uses crystal/molecular embeddings to predict properties
    like formation energy, band gap, and stability.
    """

    def __init__(self, config: MaterialsConfig):
        self.config = config
        self.encoder = CrystalGraphEncoder(config)

        # Property prediction heads
        self.formation_energy_head = nn.Linear(config.embedding_dim, 1)
        self.band_gap_head = nn.Linear(config.embedding_dim, 1)
        self.stability_head = nn.Linear(config.embedding_dim, 1)

    def predict_properties(
        self,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Predict material properties from structure.

        Returns:
            Dictionary of predicted properties
        """
        embedding = self.encoder(atomic_numbers, edge_index, edge_features, batch)

        return {
            "formation_energy": self.formation_energy_head(embedding),
            "band_gap": self.band_gap_head(embedding),
            "stability": torch.sigmoid(self.stability_head(embedding))
        }

    def find_similar_materials(
        self,
        query_embedding: torch.Tensor,
        database_embeddings: torch.Tensor,
        database_properties: dict,
        k: int = 10
    ) -> list[dict]:
        """
        Find materials with similar structure/properties.

        Useful for identifying known materials similar to
        a computationally designed candidate.
        """
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            database_embeddings
        )

        top_k = torch.topk(similarities, k)

        results = []
        for idx, sim in zip(top_k.indices, top_k.values):
            results.append({
                "index": idx.item(),
                "similarity": sim.item(),
                "formation_energy": database_properties["formation_energy"][idx].item(),
                "band_gap": database_properties["band_gap"][idx].item()
            })

        return results
