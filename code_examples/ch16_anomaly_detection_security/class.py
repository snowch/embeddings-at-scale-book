from typing import Dict, Tuple

# Code from Chapter 16
# Book: Embeddings at Scale

"""
Financial Risk Assessment with Embeddings

Architecture:
1. Entity encoder: Company/person features → embedding
2. Transaction network: Graph of financial relationships
3. Risk propagation: Risk flows through network
4. Risk scoring: Distance from low-risk cluster

Applications:
- Credit risk: Predict loan default probability
- Investment risk: Identify high-risk securities
- Insurance risk: Price policies based on risk profile
- Counterparty risk: Assess risk in financial networks
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FinancialEntity:
    """
    Financial entity (company, person)

    Attributes:
        entity_id: Unique identifier
        entity_type: Type (company, person)
        features: Financial features (income, debt, assets, etc.)
        risk_level: Risk category (low, medium, high)
        embedding: Learned embedding
    """

    entity_id: str
    entity_type: str
    features: Dict[str, float]
    risk_level: Optional[str] = None
    embedding: Optional[np.ndarray] = None


class FinancialEntityEncoder(nn.Module):
    """
    Encode financial entities to embeddings

    Features:
    - Financial ratios (debt-to-income, asset-to-liability, etc.)
    - Behavioral features (payment history, transaction patterns)
    - Network features (relationships, supply chain position)
    """

    def __init__(self, feature_dim: int = 50, embedding_dim: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, embedding_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode entity

        Args:
            features: Entity features (batch, feature_dim)

        Returns:
            Entity embeddings (batch, embedding_dim)
        """
        emb = self.encoder(features)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


class RiskAssessmentSystem:
    """
    Financial risk assessment system

    Components:
    1. Entity encoder: Features → embedding
    2. Risk clusters: Low/medium/high risk clusters
    3. Network analysis: Risk propagation through relationships
    4. Risk scoring: Probability of default/loss

    Features:
    - Credit risk scoring
    - Portfolio risk assessment
    - Network risk analysis (contagion)
    """

    def __init__(self, embedding_dim: int = 128):
        """
        Args:
            embedding_dim: Embedding dimension
        """
        self.embedding_dim = embedding_dim

        # Entity encoder
        self.encoder = FinancialEntityEncoder(embedding_dim=embedding_dim)
        self.encoder.eval()

        # Risk cluster centroids
        self.risk_centroids: Dict[str, np.ndarray] = {}

        print("Initialized Risk Assessment System")
        print(f"  Embedding dimension: {embedding_dim}")

    def build_risk_clusters(self, entities: List[FinancialEntity]):
        """
        Build risk clusters from labeled entities

        Args:
            entities: Entities with known risk levels
        """
        print(f"Building risk clusters from {len(entities)} entities...")

        # Group by risk level
        risk_groups = {"low": [], "medium": [], "high": []}

        for entity in entities:
            if entity.risk_level in risk_groups:
                # Extract features (simplified)
                features = np.array(
                    [
                        entity.features.get("debt_to_income", 0.5),
                        entity.features.get("payment_history_score", 0.7),
                        entity.features.get("assets", 50000) / 100000,
                        # ... more features
                    ]
                    + [0.0] * 47,
                    dtype=np.float32,
                )  # Pad to 50 dimensions

                features_tensor = torch.from_numpy(features).unsqueeze(0)

                with torch.no_grad():
                    emb = self.encoder(features_tensor).numpy()[0]

                risk_groups[entity.risk_level].append(emb)

        # Compute cluster centroids
        for risk_level, embeddings in risk_groups.items():
            if embeddings:
                self.risk_centroids[risk_level] = np.mean(embeddings, axis=0)
                print(f"  {risk_level.capitalize()} risk: {len(embeddings)} entities")

        print("✓ Built risk clusters")

    def assess_risk(self, entity: FinancialEntity) -> Tuple[str, Dict[str, float]]:
        """
        Assess risk level for entity

        Args:
            entity: Entity to assess

        Returns:
            (risk_level, distances_to_clusters)
        """
        # Extract features
        features = np.array(
            [
                entity.features.get("debt_to_income", 0.5),
                entity.features.get("payment_history_score", 0.7),
                entity.features.get("assets", 50000) / 100000,
            ]
            + [0.0] * 47,
            dtype=np.float32,
        )

        features_tensor = torch.from_numpy(features).unsqueeze(0)

        with torch.no_grad():
            emb = self.encoder(features_tensor).numpy()[0]

        # Compute distance to each risk cluster
        distances = {}
        for risk_level, centroid in self.risk_centroids.items():
            distance = np.linalg.norm(emb - centroid)
            distances[risk_level] = float(distance)

        # Assign to nearest cluster
        risk_level = min(distances.keys(), key=lambda k: distances[k])

        return risk_level, distances


# Example: Credit risk assessment
def risk_assessment_example():
    """
    Credit risk assessment for loan applicants

    Use case:
    - 1M loan applications/year
    - Predict default probability
    - Set interest rates based on risk

    Features: Income, debt, credit history, employment
    """

    # Initialize system
    system = RiskAssessmentSystem(embedding_dim=64)

    # Create training entities with known risk
    training_entities = []

    # Low risk entities
    for i in range(100):
        entity = FinancialEntity(
            entity_id=f"low_risk_{i}",
            entity_type="person",
            features={
                "debt_to_income": 0.2 + np.random.rand() * 0.1,  # 20-30%
                "payment_history_score": 0.9 + np.random.rand() * 0.1,  # 90-100%
                "assets": 100000 + np.random.rand() * 50000,  # $100K-$150K
            },
            risk_level="low",
        )
        training_entities.append(entity)

    # Medium risk entities
    for i in range(100):
        entity = FinancialEntity(
            entity_id=f"medium_risk_{i}",
            entity_type="person",
            features={
                "debt_to_income": 0.35 + np.random.rand() * 0.15,  # 35-50%
                "payment_history_score": 0.7 + np.random.rand() * 0.15,  # 70-85%
                "assets": 30000 + np.random.rand() * 40000,  # $30K-$70K
            },
            risk_level="medium",
        )
        training_entities.append(entity)

    # High risk entities
    for i in range(100):
        entity = FinancialEntity(
            entity_id=f"high_risk_{i}",
            entity_type="person",
            features={
                "debt_to_income": 0.6 + np.random.rand() * 0.3,  # 60-90%
                "payment_history_score": 0.3 + np.random.rand() * 0.3,  # 30-60%
                "assets": 5000 + np.random.rand() * 15000,  # $5K-$20K
            },
            risk_level="high",
        )
        training_entities.append(entity)

    print("=== Building Risk Clusters ===")
    system.build_risk_clusters(training_entities)

    # Assess new applicants
    print("\n=== Assessing New Applicants ===")

    # Test: Low risk applicant
    test_low = FinancialEntity(
        entity_id="applicant_1",
        entity_type="person",
        features={"debt_to_income": 0.25, "payment_history_score": 0.95, "assets": 120000},
    )

    risk_level, distances = system.assess_risk(test_low)
    print("\nApplicant 1:")
    print(f"  Debt-to-income: {test_low.features['debt_to_income']:.1%}")
    print(f"  Payment history: {test_low.features['payment_history_score']:.1%}")
    print(f"  Assets: ${test_low.features['assets']:,.0f}")
    print(f"  Risk level: {risk_level.upper()}")
    print(f"  Distances: {', '.join([f'{k}={v:.3f}' for k, v in distances.items()])}")

    # Test: High risk applicant
    test_high = FinancialEntity(
        entity_id="applicant_2",
        entity_type="person",
        features={"debt_to_income": 0.75, "payment_history_score": 0.45, "assets": 8000},
    )

    risk_level, distances = system.assess_risk(test_high)
    print("\nApplicant 2:")
    print(f"  Debt-to-income: {test_high.features['debt_to_income']:.1%}")
    print(f"  Payment history: {test_high.features['payment_history_score']:.1%}")
    print(f"  Assets: ${test_high.features['assets']:,.0f}")
    print(f"  Risk level: {risk_level.upper()}")
    print(f"  Distances: {', '.join([f'{k}={v:.3f}' for k, v in distances.items()])}")


# Uncomment to run:
# risk_assessment_example()
