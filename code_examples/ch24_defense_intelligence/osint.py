from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OSINTConfig:
    """Configuration for open-source intelligence embedding models."""

    vocab_size: int = 50000
    max_seq_length: int = 512
    embedding_dim: int = 768
    hidden_dim: int = 3072
    n_heads: int = 12
    n_layers: int = 6
    image_dim: int = 2048  # From pretrained vision model


class MultiModalDocumentEncoder(nn.Module):
    """
    Encode documents with text and images into unified embeddings.

    Handles news articles, social media posts, and reports
    with mixed text and visual content.
    """

    def __init__(self, config: OSINTConfig):
        super().__init__()
        self.config = config

        # Text encoder (simplified transformer)
        self.token_embed = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embed = nn.Embedding(config.max_seq_length, config.embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim,
            batch_first=True,
        )
        self.text_transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Image projection (from pretrained features)
        self.image_projection = nn.Linear(config.image_dim, config.embedding_dim)

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            config.embedding_dim, config.n_heads, batch_first=True
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode document with text and optional images.

        Args:
            text_ids: [batch, seq_len] token indices
            text_mask: [batch, seq_len] attention mask
            image_features: [batch, n_images, image_dim] pretrained image features

        Returns:
            document_embedding: [batch, embedding_dim]
        """
        batch_size, seq_len = text_ids.shape

        # Encode text
        positions = torch.arange(seq_len, device=text_ids.device).unsqueeze(0)
        text_emb = self.token_embed(text_ids) + self.position_embed(positions)
        text_features = self.text_transformer(text_emb, src_key_padding_mask=~text_mask.bool())

        # Pool text
        text_emb = text_features * text_mask.unsqueeze(-1)
        text_pooled = text_emb.sum(dim=1) / text_mask.sum(dim=1, keepdim=True).clamp(min=1)

        if image_features is None:
            return F.normalize(text_pooled, dim=-1)

        # Project images
        img_emb = self.image_projection(image_features)

        # Cross-modal attention (text attends to images)
        attended, _ = self.cross_attention(text_features, img_emb, img_emb)
        img_context = attended.mean(dim=1)

        # Fuse modalities
        combined = torch.cat([text_pooled, img_context], dim=-1)
        doc_embedding = self.fusion(combined)

        return F.normalize(doc_embedding, dim=-1)


class SourceCredibilityEncoder(nn.Module):
    """
    Encode source credibility features.

    Assesses reliability of information sources based on
    historical accuracy, bias patterns, and corroboration.
    """

    def __init__(self, config: OSINTConfig):
        super().__init__()
        self.config = config

        # Source feature encoder
        self.source_encoder = nn.Sequential(
            nn.Linear(50, config.hidden_dim),  # Source metadata features
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

        # Historical content encoder (past articles from source)
        self.history_encoder = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.embedding_dim,
            num_layers=2,
            batch_first=True,
        )

        # Credibility scorer
        self.credibility_head = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, source_features: torch.Tensor, historical_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode source and assess credibility.

        Args:
            source_features: [batch, 50] source metadata
            historical_embeddings: [batch, n_articles, embedding_dim] past content

        Returns:
            source_embedding: Source representation
            credibility_score: Reliability score (0-1)
        """
        # Encode source metadata
        source_emb = self.source_encoder(source_features)
        source_emb = F.normalize(source_emb, dim=-1)

        # Encode historical pattern
        _, history_hidden = self.history_encoder(historical_embeddings)
        history_emb = history_hidden[-1]

        # Combine for credibility assessment
        combined = torch.cat([source_emb, history_emb], dim=-1)
        credibility = self.credibility_head(combined)

        return source_emb, credibility


class NarrativeTracker(nn.Module):
    """
    Track narrative evolution across sources and time.

    Identifies how stories develop, spread, and mutate
    across the information environment.
    """

    def __init__(self, config: OSINTConfig):
        super().__init__()
        self.config = config

        # Document encoder
        self.doc_encoder = MultiModalDocumentEncoder(config)

        # Temporal evolution model
        self.temporal_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.n_heads,
                dim_feedforward=config.hidden_dim,
                batch_first=True,
            ),
            num_layers=4,
        )

        # Narrative state embedding
        self.narrative_projection = nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(self, document_embeddings: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Track narrative from document sequence.

        Args:
            document_embeddings: [batch, n_docs, embedding_dim] chronological docs
            timestamps: [batch, n_docs] publication times

        Returns:
            narrative_embedding: [batch, embedding_dim] current narrative state
        """
        # Add temporal encoding (simplified)
        time_encoding = torch.sin(timestamps.unsqueeze(-1) * 0.001)
        time_encoding = time_encoding.expand(-1, -1, self.config.embedding_dim)
        x = document_embeddings + time_encoding * 0.1

        # Model temporal evolution
        narrative_features = self.temporal_model(x)

        # Use final state as narrative embedding
        narrative_emb = self.narrative_projection(narrative_features[:, -1])

        return F.normalize(narrative_emb, dim=-1)


class InfluenceOperationDetector(nn.Module):
    """
    Detect coordinated influence operations.

    Identifies patterns of coordinated inauthentic behavior
    through behavioral and content similarity analysis.
    """

    def __init__(self, config: OSINTConfig):
        super().__init__()
        self.config = config

        # Account behavior encoder
        self.behavior_encoder = nn.Sequential(
            nn.Linear(100, config.hidden_dim),  # Behavioral features
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

        # Content pattern encoder
        self.content_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

        # Coordination detector
        self.coordination_head = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, behavior_features: torch.Tensor, content_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Detect coordinated behavior.

        Args:
            behavior_features: [batch, 100] account behavior features
            content_embeddings: [batch, n_posts, embedding_dim] post embeddings

        Returns:
            account_embedding: Account representation
            coordination_score: Probability of coordinated behavior
        """
        # Encode behavior
        behavior_emb = self.behavior_encoder(behavior_features)
        behavior_emb = F.normalize(behavior_emb, dim=-1)

        # Encode content pattern
        content_pattern = content_embeddings.mean(dim=1)
        content_emb = self.content_encoder(content_pattern)
        content_emb = F.normalize(content_emb, dim=-1)

        # Assess coordination
        combined = torch.cat([behavior_emb, content_emb], dim=-1)
        coordination_score = self.coordination_head(combined)

        # Account embedding combines both
        account_emb = (behavior_emb + content_emb) / 2

        return account_emb, coordination_score

    def find_coordinated_clusters(
        self, account_embeddings: torch.Tensor, threshold: float = 0.9
    ) -> list[list[int]]:
        """
        Cluster accounts that appear coordinated.
        """
        _n = account_embeddings.shape[0]  # noqa: F841
        similarities = F.cosine_similarity(
            account_embeddings.unsqueeze(1), account_embeddings.unsqueeze(0), dim=2
        )

        # Find highly similar account pairs
        coordinated_pairs = (similarities > threshold).nonzero()

        # Build clusters
        clusters: list[set[int]] = []
        _assigned: set[int] = set()  # noqa: F841

        for i, j in coordinated_pairs:
            i, j = i.item(), j.item()
            if i >= j:  # Skip diagonal and duplicates
                continue

            # Find or create cluster
            found_cluster = None
            for cluster in clusters:
                if i in cluster or j in cluster:
                    found_cluster = cluster
                    break

            if found_cluster:
                found_cluster.add(i)
                found_cluster.add(j)
            else:
                clusters.append({i, j})

        return [list(c) for c in clusters if len(c) >= 3]


class OSINTSearchSystem:
    """
    Search and analysis system for open-source intelligence.
    """

    def __init__(self, config: OSINTConfig):
        self.config = config
        self.doc_encoder = MultiModalDocumentEncoder(config)
        self.credibility_encoder = SourceCredibilityEncoder(config)
        self.influence_detector = InfluenceOperationDetector(config)

        # Document index
        self.doc_embeddings = None
        self.doc_metadata = None

    def search(
        self, query_embedding: torch.Tensor, k: int = 20, min_credibility: float = 0.5
    ) -> list[dict]:
        """
        Search documents with credibility filtering.
        """
        if self.doc_embeddings is None:
            raise ValueError("Index not built")

        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), self.doc_embeddings)

        # Sort by similarity
        sorted_indices = torch.argsort(similarities, descending=True)

        results = []
        for idx in sorted_indices:
            metadata = self.doc_metadata[idx.item()]
            if metadata.get("credibility", 1.0) >= min_credibility:
                results.append(
                    {
                        "doc_id": metadata["id"],
                        "title": metadata["title"],
                        "source": metadata["source"],
                        "credibility": metadata.get("credibility", 1.0),
                        "similarity": similarities[idx].item(),
                    }
                )
                if len(results) >= k:
                    break

        return results

    def summarize_topic(self, topic_embedding: torch.Tensor, n_docs: int = 50) -> dict:
        """
        Summarize a topic from relevant documents.
        """
        # Find relevant documents
        results = self.search(topic_embedding, k=n_docs)

        # Aggregate metadata
        sources = {}
        for r in results:
            source = r["source"]
            if source not in sources:
                sources[source] = {"count": 0, "avg_credibility": 0}
            sources[source]["count"] += 1
            sources[source]["avg_credibility"] += r["credibility"]

        for s in sources:
            sources[s]["avg_credibility"] /= sources[s]["count"]

        return {
            "n_documents": len(results),
            "sources": sources,
            "avg_credibility": sum(r["credibility"] for r in results) / len(results),
            "top_documents": results[:10],
        }
