from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecisionSupportConfig:
    """Configuration for decision support embedding models."""

    embedding_dim: int = 768
    hidden_dim: int = 3072
    n_heads: int = 12
    n_layers: int = 6
    n_source_types: int = 10  # GEOINT, SIGINT, HUMINT, OSINT, etc.


class SituationEncoder(nn.Module):
    """
    Encode tactical/strategic situations from multi-source intelligence.

    Fuses information from diverse sources into unified
    situation representation for decision support.
    """

    def __init__(self, config: DecisionSupportConfig):
        super().__init__()
        self.config = config

        # Source-specific encoders
        self.source_encoders = nn.ModuleDict(
            {
                "geoint": nn.Linear(512, config.embedding_dim),
                "sigint": nn.Linear(512, config.embedding_dim),
                "humint": nn.Linear(256, config.embedding_dim),
                "osint": nn.Linear(512, config.embedding_dim),
                "cyber": nn.Linear(256, config.embedding_dim),
            }
        )

        # Source type embedding
        self.source_type_embed = nn.Embedding(config.n_source_types, config.embedding_dim)

        # Cross-source fusion via transformer
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.n_heads,
                dim_feedforward=config.hidden_dim,
                batch_first=True,
            ),
            num_layers=config.n_layers,
        )

        # Situation classification
        self.situation_classifier = nn.Linear(config.embedding_dim, 20)  # Situation types

        # Urgency assessment
        self.urgency_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, source_embeddings: dict[str, torch.Tensor], source_types: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode situation from multi-source intelligence.

        Args:
            source_embeddings: Dict of source type -> embeddings
            source_types: [batch, n_sources] source type indices

        Returns:
            situation_embedding: Fused situation representation
            situation_logits: Situation type classification
            urgency: Urgency score (0-1)
        """
        # Encode each source type
        encoded_sources = []
        for source_type, emb in source_embeddings.items():
            if source_type in self.source_encoders:
                encoded = self.source_encoders[source_type](emb)
                encoded = F.normalize(encoded, dim=-1)
                encoded_sources.append(encoded)

        if not encoded_sources:
            raise ValueError("No valid source embeddings provided")

        # Stack sources
        stacked = torch.stack(encoded_sources, dim=1)

        # Add source type embeddings
        source_type_emb = self.source_type_embed(source_types)
        stacked = stacked + source_type_emb[:, : stacked.shape[1]]

        # Fuse across sources
        fused = self.fusion_transformer(stacked)

        # Pool to situation embedding
        situation_emb = fused.mean(dim=1)
        situation_emb = F.normalize(situation_emb, dim=-1)

        # Classifications
        situation_logits = self.situation_classifier(situation_emb)
        urgency = self.urgency_head(situation_emb)

        return situation_emb, situation_logits, urgency


class CourseOfActionEncoder(nn.Module):
    """
    Encode courses of action for comparison and selection.

    Represents potential actions with their expected outcomes
    and resource requirements.
    """

    def __init__(self, config: DecisionSupportConfig):
        super().__init__()
        self.config = config

        # Action parameters encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(200, config.hidden_dim),  # Action parameters
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

        # Resource requirements encoder
        self.resource_encoder = nn.Sequential(
            nn.Linear(50, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.embedding_dim),
        )

        # Expected outcome encoder
        self.outcome_encoder = nn.Sequential(
            nn.Linear(100, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.embedding_dim),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.embedding_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

        # Risk assessment
        self.risk_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        action_params: torch.Tensor,
        resource_reqs: torch.Tensor,
        expected_outcomes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode course of action.

        Args:
            action_params: Action specification
            resource_reqs: Resource requirements
            expected_outcomes: Projected outcomes

        Returns:
            coa_embedding: Course of action representation
            risk_score: Assessed risk level
        """
        action_emb = F.normalize(self.action_encoder(action_params), dim=-1)
        resource_emb = F.normalize(self.resource_encoder(resource_reqs), dim=-1)
        outcome_emb = F.normalize(self.outcome_encoder(expected_outcomes), dim=-1)

        combined = torch.cat([action_emb, resource_emb, outcome_emb], dim=-1)
        coa_emb = self.fusion(combined)
        coa_emb = F.normalize(coa_emb, dim=-1)

        risk = self.risk_head(coa_emb)

        return coa_emb, risk


class PrecedentRetrieval(nn.Module):
    """
    Retrieve relevant historical precedents.

    Finds similar past situations and their outcomes
    to inform current decisions.
    """

    def __init__(self, config: DecisionSupportConfig):
        super().__init__()
        self.config = config

        # Situation comparison
        self.situation_projection = nn.Linear(config.embedding_dim, config.embedding_dim)

        # Outcome encoder
        self.outcome_encoder = nn.Sequential(
            nn.Linear(100, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

        # Relevance scoring
        self.relevance_head = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        current_situation: torch.Tensor,
        historical_situations: torch.Tensor,
        historical_outcomes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Score relevance of historical precedents.

        Args:
            current_situation: Current situation embedding
            historical_situations: [n_precedents, embedding_dim]
            historical_outcomes: [n_precedents, 100] outcome features

        Returns:
            relevance_scores: Relevance of each precedent
            outcome_embeddings: Encoded outcomes
        """
        # Project current situation
        current_proj = self.situation_projection(current_situation)

        # Compute relevance scores
        # Expand current for comparison
        current_expanded = current_proj.unsqueeze(0).expand(historical_situations.shape[0], -1)

        combined = torch.cat([current_expanded, historical_situations], dim=-1)
        relevance = self.relevance_head(combined).squeeze(-1)

        # Encode outcomes
        outcome_embs = self.outcome_encoder(historical_outcomes)

        return relevance, outcome_embs


class RiskAssessment(nn.Module):
    """
    Assess risks associated with situations and actions.

    Quantifies uncertainty and potential negative outcomes.
    """

    def __init__(self, config: DecisionSupportConfig):
        super().__init__()
        self.config = config

        # Situation risk factors
        self.situation_risk = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 50),  # 50 risk factors
        )

        # Action risk factors
        self.action_risk = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 50),
        )

        # Combined risk assessment
        self.risk_fusion = nn.Sequential(
            nn.Linear(100, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 5),  # 5 risk categories
        )

    def forward(
        self, situation_embedding: torch.Tensor, action_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assess risks.

        Args:
            situation_embedding: Current situation
            action_embedding: Proposed action

        Returns:
            risk_factors: Individual risk factor scores
            risk_categories: Category-level risk assessments
        """
        sit_risk = torch.sigmoid(self.situation_risk(situation_embedding))
        act_risk = torch.sigmoid(self.action_risk(action_embedding))

        combined_risk = torch.cat([sit_risk, act_risk], dim=-1)
        risk_categories = torch.sigmoid(self.risk_fusion(combined_risk))

        return combined_risk, risk_categories


class DecisionSupportSystem:
    """
    Decision support system for command and control.
    """

    def __init__(self, config: DecisionSupportConfig):
        self.config = config
        self.situation_encoder = SituationEncoder(config)
        self.coa_encoder = CourseOfActionEncoder(config)
        self.precedent_retrieval = PrecedentRetrieval(config)
        self.risk_assessment = RiskAssessment(config)

        # Historical database
        self.precedent_situations = None
        self.precedent_outcomes = None
        self.precedent_metadata = None

    def analyze_situation(
        self, source_embeddings: dict[str, torch.Tensor], source_types: torch.Tensor
    ) -> dict:
        """
        Analyze current situation.
        """
        situation_emb, situation_logits, urgency = self.situation_encoder(
            source_embeddings, source_types
        )

        situation_probs = F.softmax(situation_logits, dim=-1)
        predicted_type = torch.argmax(situation_probs, dim=-1)

        return {
            "embedding": situation_emb,
            "situation_type": predicted_type.item(),
            "type_confidence": situation_probs.max().item(),
            "urgency": urgency.item(),
        }

    def retrieve_precedents(self, situation_embedding: torch.Tensor, k: int = 10) -> list[dict]:
        """
        Retrieve relevant historical precedents.
        """
        if self.precedent_situations is None:
            return []

        relevance, outcome_embs = self.precedent_retrieval(
            situation_embedding, self.precedent_situations, self.precedent_outcomes
        )

        top_k = torch.topk(relevance, min(k, len(relevance)))

        precedents = []
        for idx, score in zip(top_k.indices, top_k.values):
            precedents.append(
                {
                    "precedent_id": self.precedent_metadata[idx]["id"],
                    "description": self.precedent_metadata[idx]["description"],
                    "outcome": self.precedent_metadata[idx]["outcome"],
                    "relevance": score.item(),
                }
            )

        return precedents

    def evaluate_coa(
        self,
        situation_embedding: torch.Tensor,
        action_params: torch.Tensor,
        resource_reqs: torch.Tensor,
        expected_outcomes: torch.Tensor,
    ) -> dict:
        """
        Evaluate a course of action.
        """
        coa_emb, coa_risk = self.coa_encoder(action_params, resource_reqs, expected_outcomes)

        risk_factors, risk_categories = self.risk_assessment(situation_embedding, coa_emb)

        return {
            "embedding": coa_emb,
            "overall_risk": coa_risk.item(),
            "risk_categories": risk_categories.tolist(),
            "top_risk_factors": torch.topk(risk_factors.squeeze(), 5),
        }

    def compare_coas(
        self, coa_embeddings: list[torch.Tensor], situation_embedding: torch.Tensor
    ) -> list[dict]:
        """
        Compare multiple courses of action.
        """
        comparisons = []

        # Stack COA embeddings
        coa_stack = torch.stack(coa_embeddings)

        # Compute situation alignment
        alignments = F.cosine_similarity(situation_embedding.unsqueeze(0), coa_stack)

        for i, (emb, align) in enumerate(zip(coa_embeddings, alignments)):
            comparisons.append(
                {"coa_index": i, "situation_alignment": align.item(), "embedding": emb}
            )

        # Sort by alignment
        comparisons.sort(key=lambda x: x["situation_alignment"], reverse=True)

        return comparisons

    def generate_recommendation(
        self, situation_analysis: dict, precedents: list[dict], coa_evaluations: list[dict]
    ) -> dict:
        """
        Generate decision recommendation with justification.
        """
        # Simple recommendation logic
        # Real system would use more sophisticated reasoning

        # Filter to acceptable risk
        acceptable_coas = [coa for coa in coa_evaluations if coa["overall_risk"] < 0.7]

        if not acceptable_coas:
            return {
                "recommendation": None,
                "reasoning": "No courses of action meet acceptable risk thresholds",
                "precedents_considered": len(precedents),
                "urgency": situation_analysis["urgency"],
            }

        # Select highest alignment COA
        best_coa = max(acceptable_coas, key=lambda x: x.get("situation_alignment", 0))

        return {
            "recommendation": best_coa,
            "reasoning": "Selected based on situation alignment and acceptable risk",
            "precedents_considered": len(precedents),
            "similar_precedent": precedents[0] if precedents else None,
            "urgency": situation_analysis["urgency"],
            "confidence": 1 - best_coa["overall_risk"],
        }
