from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Code from Chapter 17
# Book: Embeddings at Scale

"""
Risk Scoring with Embeddings

Architecture:
1. Entity encoder: Customer/property/transaction → embedding
2. Risk model: Embedding → risk score
3. Multi-modal: Text (loan applications), images (property photos), graphs (social networks)

Applications:
- Credit underwriting: Default probability
- Insurance: Claims probability, loss severity
- Fraud detection: Transaction fraud risk
- Cybersecurity: Breach risk assessment

Techniques:
- Multi-task learning: Predict multiple risk types jointly
- Temporal: Account for risk changes over time
- Causal: Distinguish correlation from causation
- Explainable: SHAP values for risk drivers
"""

@dataclass
class RiskEntity:
    """
    Entity to assess risk for
    
    Attributes:
        entity_id: Unique identifier
        entity_type: Type (customer, property, transaction)
        features: Risk-relevant features
        risk_score: Assessed risk score
        risk_factors: Explanation of risk score
        embedding: Learned entity embedding
    """
    entity_id: str
    entity_type: str
    features: Dict[str, Any]
    risk_score: Optional[float] = None
    risk_factors: Optional[List[str]] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.features is None:
            self.features = {}
        if self.risk_factors is None:
            self.risk_factors = []

class RiskEncoder(nn.Module):
    """
    Encode entities for risk assessment
    
    Architecture:
    - Demographic features
    - Financial features
    - Behavioral features
    - External data (social, location)
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(50, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode entities for risk assessment
        
        Args:
            features: Entity features (batch_size, num_features)
        
        Returns:
            Risk embeddings (batch_size, embedding_dim)
        """
        risk_emb = self.encoder(features)
        risk_emb = F.normalize(risk_emb, p=2, dim=1)
        return risk_emb

class RiskScoringModel(nn.Module):
    """
    Predict risk score from embedding
    
    Output: Probability of adverse event (default, claim, fraud)
    
    Training:
    - Classification: Binary (default / no default)
    - Survival analysis: Time until event
    - Multi-task: Predict multiple risk types
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        self.risk_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict risk scores
        
        Args:
            embeddings: Risk embeddings (batch_size, embedding_dim)
        
        Returns:
            Risk scores (batch_size, 1)
        """
        risk_scores = self.risk_head(embeddings)
        return risk_scores

# Example: Credit underwriting
def risk_scoring_example():
    """
    Credit underwriting using embedding-based risk scoring
    
    Traditional: FICO score + debt-to-income ratio
    - FICO > 700 and DTI < 40% → approve
    - Otherwise → reject
    
    Embedding approach:
    - Encode applicant from all available data
    - Predict default probability
    - Account for non-linear patterns
    - Continuously improve from outcomes
    """

    print("=== Credit Risk Scoring ===")
    print("\nTraditional approach: FICO score + DTI ratio")
    print("  FICO > 700 AND DTI < 40% → Approve")
    print("  Otherwise → Reject")

    print("\n--- Applicant 1: Traditional Accept ---")
    print("FICO: 750")
    print("Income: $80,000")
    print("Debt-to-income: 30%")
    print("Employment: 5 years")
    print("Traditional: APPROVE")
    print("Embedding risk score: 0.03 (3% default probability)")
    print("  → Consistent with traditional model")

    print("\n--- Applicant 2: False Negative (Traditional Rejects Good Customer) ---")
    print("FICO: 680 (thin file - recent immigrant)")
    print("Income: $95,000 (software engineer)")
    print("Debt-to-income: 15%")
    print("Employment: 2 years at Google")
    print("Rent payments: 100% on-time for 3 years")
    print("Traditional: REJECT (FICO < 700)")
    print("Embedding risk score: 0.04 (4% default probability)")
    print("  → Embedding learns that Google employees + on-time rent = low risk")
    print("  → Would approve, capturing good customer traditional model misses")

    print("\n--- Applicant 3: False Positive (Traditional Approves Risky Customer) ---")
    print("FICO: 720")
    print("Income: $60,000")
    print("Debt-to-income: 38%")
    print("Employment: 6 months (new job)")
    print("Gambling transactions: $500/week (credit card)")
    print("Recent inquiries: 8 in past 3 months")
    print("Traditional: APPROVE (FICO > 700, DTI < 40%)")
    print("Embedding risk score: 0.22 (22% default probability)")
    print("  → Embedding detects risky pattern: job instability + gambling + credit seeking")
    print("  → Would reject or price for risk")

    print("\n--- Applicant 4: Novel Pattern ---")
    print("FICO: 700")
    print("Income: $45,000 (gig economy - Uber driver)")
    print("Debt-to-income: 35%")
    print("Income volatility: High")
    print("Gig employment: 3 years")
    print("Traditional: APPROVE (marginally)")
    print("Embedding risk score: 0.12 (12% default probability)")
    print("  → Embedding learned that gig workers have higher volatility risk")
    print("  → Traditional model doesn't distinguish W2 vs gig income")

# Uncomment to run:
# risk_scoring_example()
