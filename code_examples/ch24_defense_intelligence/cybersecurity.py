import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class CyberConfig:
    """Configuration for cybersecurity embedding models."""
    n_syscall_types: int = 500
    n_api_types: int = 1000
    max_sequence_length: int = 1000
    embedding_dim: int = 256
    hidden_dim: int = 512
    n_attack_types: int = 50  # MITRE ATT&CK techniques


class BehavioralEncoder(nn.Module):
    """
    Encode user/system behavior for anomaly detection.

    Learns representations of normal behavior patterns
    to identify deviations indicating compromise.
    """

    def __init__(self, config: CyberConfig):
        super().__init__()
        self.config = config

        # Action embedding (commands, API calls, etc.)
        self.action_embed = nn.Embedding(config.n_syscall_types, config.embedding_dim)

        # Context features (time, user, system)
        self.context_encoder = nn.Linear(50, config.embedding_dim)

        # Sequence model
        self.sequence_encoder = nn.LSTM(
            input_size=config.embedding_dim * 2,
            hidden_size=config.embedding_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Project to embedding
        self.projection = nn.Linear(config.embedding_dim * 2, config.embedding_dim)

    def forward(
        self,
        action_ids: torch.Tensor,
        context_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode behavioral sequence.

        Args:
            action_ids: [batch, seq_len] action type indices
            context_features: [batch, seq_len, 50] contextual features
            mask: [batch, seq_len] valid action mask

        Returns:
            behavior_embedding: [batch, embedding_dim]
        """
        # Embed actions and context
        action_emb = self.action_embed(action_ids)
        context_emb = self.context_encoder(context_features)

        # Combine
        x = torch.cat([action_emb, context_emb], dim=-1)

        # Encode sequence
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )

        _, (hidden, _) = self.sequence_encoder(x)

        # Combine bidirectional hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        embedding = self.projection(hidden)

        return F.normalize(embedding, dim=-1)


class MalwareEncoder(nn.Module):
    """
    Encode malware samples for classification and clustering.

    Combines static (code structure) and dynamic (behavior)
    features for robust malware analysis.
    """

    def __init__(self, config: CyberConfig):
        super().__init__()
        self.config = config

        # Static feature encoder (byte sequences, imports, strings)
        self.static_encoder = nn.Sequential(
            nn.Linear(1000, config.hidden_dim),  # Static features
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        # Dynamic feature encoder (API call sequences)
        self.api_embed = nn.Embedding(config.n_api_types, config.embedding_dim)

        self.dynamic_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim,
                batch_first=True
            ),
            num_layers=4
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        # Family classifier
        self.family_classifier = nn.Linear(config.embedding_dim, 100)  # Malware families

    def forward(
        self,
        static_features: torch.Tensor,
        api_sequence: torch.Tensor,
        api_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode malware sample.

        Args:
            static_features: [batch, 1000] static analysis features
            api_sequence: [batch, seq_len] API call sequence
            api_mask: [batch, seq_len] valid call mask

        Returns:
            malware_embedding: Sample embedding
            family_logits: Family classification logits
        """
        # Encode static features
        static_emb = self.static_encoder(static_features)
        static_emb = F.normalize(static_emb, dim=-1)

        # Encode dynamic behavior
        api_emb = self.api_embed(api_sequence)
        if api_mask is not None:
            dynamic_features = self.dynamic_encoder(
                api_emb, src_key_padding_mask=~api_mask.bool()
            )
            # Masked mean pooling
            dynamic_features = dynamic_features * api_mask.unsqueeze(-1)
            dynamic_emb = dynamic_features.sum(dim=1) / api_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            dynamic_features = self.dynamic_encoder(api_emb)
            dynamic_emb = dynamic_features.mean(dim=1)

        dynamic_emb = F.normalize(dynamic_emb, dim=-1)

        # Fuse representations
        combined = torch.cat([static_emb, dynamic_emb], dim=-1)
        malware_emb = self.fusion(combined)
        malware_emb = F.normalize(malware_emb, dim=-1)

        # Classify family
        family_logits = self.family_classifier(malware_emb)

        return malware_emb, family_logits


class NetworkTrafficEncoder(nn.Module):
    """
    Encode network traffic patterns for intrusion detection.

    Learns representations of network flows to identify
    anomalous patterns indicating attacks.
    """

    def __init__(self, config: CyberConfig):
        super().__init__()
        self.config = config

        # Flow feature encoder
        self.flow_encoder = nn.Sequential(
            nn.Linear(40, config.hidden_dim),  # Packet/flow features
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        # Temporal pattern encoder
        self.temporal_encoder = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.embedding_dim,
            num_layers=2,
            batch_first=True
        )

        # Anomaly scorer
        self.anomaly_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

        # Attack type classifier
        self.attack_classifier = nn.Linear(config.embedding_dim, config.n_attack_types)

    def forward(
        self,
        flow_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode network traffic.

        Args:
            flow_features: [batch, n_flows, 40] network flow features

        Returns:
            traffic_embedding: Traffic pattern embedding
            anomaly_score: Anomaly likelihood
            attack_logits: Attack type classification
        """
        # Encode individual flows
        flow_emb = self.flow_encoder(flow_features)

        # Temporal pattern
        _, hidden = self.temporal_encoder(flow_emb)
        traffic_emb = hidden[-1]
        traffic_emb = F.normalize(traffic_emb, dim=-1)

        # Anomaly score
        anomaly_score = self.anomaly_head(traffic_emb)

        # Attack classification
        attack_logits = self.attack_classifier(traffic_emb)

        return traffic_emb, anomaly_score, attack_logits


class ThreatActorProfiler(nn.Module):
    """
    Profile threat actors based on TTPs.

    Learns actor embeddings from observed techniques
    for attribution and prediction.
    """

    def __init__(self, config: CyberConfig):
        super().__init__()
        self.config = config

        # TTP embedding (MITRE ATT&CK techniques)
        self.ttp_embed = nn.Embedding(config.n_attack_types, config.embedding_dim)

        # Infrastructure embedding (C2, domains, IPs)
        self.infra_encoder = nn.Sequential(
            nn.Linear(200, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        # Temporal pattern (attack timing)
        self.temporal_encoder = nn.Linear(50, config.embedding_dim)

        # Actor profiler
        self.actor_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim,
                batch_first=True
            ),
            num_layers=4
        )

        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(
        self,
        ttp_ids: torch.Tensor,
        infra_features: torch.Tensor,
        temporal_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Create threat actor profile.

        Args:
            ttp_ids: [batch, n_ttps] observed technique IDs
            infra_features: [batch, 200] infrastructure features
            temporal_features: [batch, 50] timing patterns

        Returns:
            actor_embedding: [batch, embedding_dim] actor profile
        """
        # Embed TTPs
        ttp_emb = self.ttp_embed(ttp_ids)  # [batch, n_ttps, dim]

        # Embed infrastructure and temporal as additional tokens
        infra_emb = self.infra_encoder(infra_features).unsqueeze(1)
        temporal_emb = self.temporal_encoder(temporal_features).unsqueeze(1)

        # Combine all features
        features = torch.cat([ttp_emb, infra_emb, temporal_emb], dim=1)

        # Profile actor
        actor_features = self.actor_encoder(features)
        actor_emb = actor_features.mean(dim=1)
        actor_emb = self.projection(actor_emb)

        return F.normalize(actor_emb, dim=-1)


class ThreatIntelligenceSystem:
    """
    Threat intelligence and detection system.
    """

    def __init__(self, config: CyberConfig):
        self.config = config
        self.behavior_encoder = BehavioralEncoder(config)
        self.malware_encoder = MalwareEncoder(config)
        self.traffic_encoder = NetworkTrafficEncoder(config)
        self.actor_profiler = ThreatActorProfiler(config)

        # Baseline embeddings for anomaly detection
        self.baseline_embeddings = None

        # Known threat actor profiles
        self.actor_embeddings = None
        self.actor_metadata = None

    def detect_anomaly(
        self,
        behavior_embedding: torch.Tensor,
        threshold: float = 0.7
    ) -> dict:
        """
        Detect behavioral anomaly.
        """
        if self.baseline_embeddings is None:
            raise ValueError("Baseline not established")

        # Find nearest baseline behavior
        similarities = F.cosine_similarity(
            behavior_embedding.unsqueeze(0),
            self.baseline_embeddings
        )

        max_similarity = similarities.max().item()
        is_anomaly = max_similarity < threshold

        return {
            "is_anomaly": is_anomaly,
            "similarity_to_baseline": max_similarity,
            "confidence": 1 - max_similarity if is_anomaly else max_similarity
        }

    def classify_malware(
        self,
        static_features: torch.Tensor,
        api_sequence: torch.Tensor
    ) -> dict:
        """
        Classify malware sample.
        """
        embedding, family_logits = self.malware_encoder(static_features, api_sequence)

        family_probs = F.softmax(family_logits, dim=-1)
        predicted_family = torch.argmax(family_probs, dim=-1)
        confidence = family_probs.max(dim=-1).values

        return {
            "embedding": embedding,
            "predicted_family": predicted_family.item(),
            "confidence": confidence.item(),
            "family_probabilities": family_probs
        }

    def attribute_attack(
        self,
        ttp_ids: torch.Tensor,
        infra_features: torch.Tensor,
        temporal_features: torch.Tensor,
        k: int = 5
    ) -> list[dict]:
        """
        Attribute attack to known threat actors.
        """
        attack_profile = self.actor_profiler(ttp_ids, infra_features, temporal_features)

        if self.actor_embeddings is None:
            return []

        similarities = F.cosine_similarity(
            attack_profile, self.actor_embeddings
        )

        top_k = torch.topk(similarities, k)

        attributions = []
        for idx, sim in zip(top_k.indices, top_k.values):
            attributions.append({
                "actor_id": self.actor_metadata[idx]["id"],
                "actor_name": self.actor_metadata[idx]["name"],
                "confidence": sim.item(),
                "known_ttps": self.actor_metadata[idx]["ttps"]
            })

        return attributions
