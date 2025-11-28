from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Code from Chapter 18
# Book: Embeddings at Scale

"""
Credit Risk Assessment with Embeddings

Architecture:
1. Borrower encoder: Maps borrowers to embeddings
2. Transaction encoder: Behavioral patterns from transaction history
3. Network encoder: Social and business relationships
4. Economic encoder: Macro conditions, regional factors
5. Risk scorer: Predicts default probability from embeddings

Techniques:
- Multi-modal fusion: Credit history + alternative data
- Graph embeddings: Capture relationship networks
- Sequential modeling: Transaction patterns over time
- Transfer learning: Pre-train on large datasets, fine-tune per segment

Production considerations:
- Explainability: SHAP values, adverse action requirements
- Fairness: Monitor for disparate impact on protected groups
- Compliance: FCRA, ECOA, state-specific regulations
- Online learning: Update models as borrowers repay/default
"""


@dataclass
class Borrower:
    """
    Loan applicant or existing borrower

    Attributes:
        borrower_id: Unique identifier
        credit_score: Traditional credit score (if available)
        income: Annual income
        employment: Employment history
        credit_history: Payment history, utilization, etc.
        transaction_history: Bank transaction patterns
        alternative_data: Rent, utilities, etc.
        relationships: Known relationships (employer, landlord, etc.)
        application: Current loan application details
    """

    borrower_id: str
    credit_score: Optional[int] = None
    income: Optional[float] = None
    employment: Optional[Dict[str, Any]] = None
    credit_history: Optional[Dict[str, Any]] = None
    transaction_history: Optional[List[Dict[str, Any]]] = None
    alternative_data: Optional[Dict[str, Any]] = None
    relationships: Optional[List[str]] = None
    application: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.employment is None:
            self.employment = {}
        if self.credit_history is None:
            self.credit_history = {}
        if self.transaction_history is None:
            self.transaction_history = []
        if self.alternative_data is None:
            self.alternative_data = {}
        if self.relationships is None:
            self.relationships = []
        if self.application is None:
            self.application = {}


@dataclass
class CreditDecision:
    """
    Credit decision output

    Attributes:
        borrower_id: Applicant identifier
        decision: Approve, reject, or manual review
        interest_rate: Approved interest rate (if approved)
        credit_limit: Credit limit (if approved)
        default_probability: Predicted default probability
        confidence: Decision confidence
        explanation: Explanation for decision
        adverse_action_reasons: Reasons for rejection (if applicable)
    """

    borrower_id: str
    decision: str  # approve, reject, review
    interest_rate: Optional[float] = None
    credit_limit: Optional[float] = None
    default_probability: float = 0.0
    confidence: float = 0.0
    explanation: str = ""
    adverse_action_reasons: Optional[List[str]] = None


class BorrowerEncoder(nn.Module):
    """
    Encode borrowers to embeddings

    Architecture:
    - Credit history encoder: Payment patterns, utilization, age of accounts
    - Transaction encoder: LSTM over bank transactions
    - Alternative data encoder: Rent, utilities, employment stability
    - Network encoder: Graph neural network over relationships
    - Fusion: Attention-based combination

    Training:
    - Default prediction: Embedding predicts default probability
    - Contrastive: Good borrowers close, bad borrowers far
    - Multi-task: Predict default, prepayment, utilization
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_credit_features: int = 30,
        num_alternative_features: int = 20,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Credit history encoder
        self.credit_encoder = nn.Sequential(
            nn.Linear(num_credit_features, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 64)
        )

        # Transaction pattern encoder
        self.transaction_encoder = nn.LSTM(
            input_size=10,  # transaction features
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        # Alternative data encoder
        self.alternative_encoder = nn.Sequential(
            nn.Linear(num_alternative_features, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 64)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(192, 128),  # 64 * 3
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
        )

    def forward(
        self,
        credit_features: torch.Tensor,
        transaction_history: torch.Tensor,
        alternative_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode borrowers

        Args:
            credit_features: Credit history (batch_size, num_credit_features)
            transaction_history: Transactions (batch_size, seq_len, 10)
            alternative_features: Alternative data (batch_size, num_alternative_features)

        Returns:
            Borrower embeddings (batch_size, embedding_dim)
        """
        # Encode credit history
        credit_emb = self.credit_encoder(credit_features)

        # Encode transaction history
        _, (transaction_hidden, _) = self.transaction_encoder(transaction_history)
        transaction_emb = transaction_hidden[-1]

        # Encode alternative data
        alternative_emb = self.alternative_encoder(alternative_features)

        # Fuse
        combined = torch.cat([credit_emb, transaction_emb, alternative_emb], dim=1)
        borrower_emb = self.fusion(combined)

        # Normalize
        borrower_emb = F.normalize(borrower_emb, p=2, dim=1)

        return borrower_emb


class CreditRiskScorer(nn.Module):
    """
    Score credit risk from borrower embeddings

    Outputs:
    - Default probability
    - Expected loss (probability × loss given default)
    - Confidence score

    Calibrated to produce well-calibrated probabilities
    for regulatory compliance and pricing.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        # Risk scoring network
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim + 10, 128),  # +10 for loan features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),  # default_prob, expected_loss, confidence
        )

    def forward(
        self, borrower_emb: torch.Tensor, loan_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Score credit risk

        Args:
            borrower_emb: Borrower embeddings (batch_size, embedding_dim)
            loan_features: Loan characteristics (batch_size, 10)

        Returns:
            Tuple of (default_prob, expected_loss, confidence)
        """
        # Combine borrower and loan features
        combined = torch.cat([borrower_emb, loan_features], dim=1)

        # Score risk
        outputs = self.scorer(combined)

        # Split outputs
        default_prob = torch.sigmoid(outputs[:, 0])  # 0-1
        expected_loss = torch.sigmoid(outputs[:, 1])  # 0-1
        confidence = torch.sigmoid(outputs[:, 2])  # 0-1

        return default_prob, expected_loss, confidence


# Example: Credit risk assessment
def credit_risk_example():
    """
    Credit risk assessment pipeline

    Demonstrates:
    1. Traditional credit scoring limitations
    2. Alternative data integration
    3. Embedding-based risk assessment
    4. Explainability for adverse actions
    """

    print("=== Credit Risk Assessment System ===")
    print("\nObjective: Expand access to credit while managing risk")
    print("Challenge: Credit invisibles lack traditional credit history")
    print("Solution: Alternative data + embeddings")

    print("\n--- Applicant 1: Traditional Credit User ---")
    print("Credit score: 750")
    print("Income: $75,000/year")
    print("Credit history: 10 years, no missed payments")
    print("Utilization: 15%")
    print("Loan request: $10,000 personal loan")

    print("\nAssessment:")
    print("  Traditional model: APPROVE (strong credit history)")
    print("  Default probability: 2.5%")
    print("  Interest rate: 8.5%")
    print("  Credit limit: $10,000")
    print("  Confidence: 0.92")
    print("\n→ Easy approval, standard process")

    print("\n--- Applicant 2: Credit Invisible ---")
    print("Credit score: None (no credit history)")
    print("Income: $55,000/year (verified)")
    print("Rent payments: 3 years, always on time")
    print("Utility payments: Never late")
    print("Transaction history: Stable income deposits, responsible spending")
    print("Employment: 4 years same employer")
    print("Loan request: $5,000 personal loan")

    print("\nTraditional assessment:")
    print("  REJECTED - No credit score")
    print("  → Lost customer, perpetuates credit invisibility")

    print("\nEmbedding-based assessment:")
    print("  Borrower embedding analysis:")
    print("    - Similar to: Prime borrowers (based on alternative data)")
    print("    - Cluster: Responsible credit invisible segment")
    print("    - Distance from high-risk cluster: 0.78 (far)")
    print("  ")
    print("  Decision: APPROVE")
    print("  Default probability: 4.2% (slightly higher due to uncertainty)")
    print("  Interest rate: 11.5% (risk-adjusted)")
    print("  Credit limit: $5,000")
    print("  Confidence: 0.78")
    print("  ")
    print("  Explanation:")
    print("    ✓ Consistent rent/utility payments (3 years)")
    print("    ✓ Stable employment (4 years)")
    print("    ✓ Responsible transaction patterns")
    print("    ! Limited by lack of traditional credit history")
    print("\n→ Expanded access while managing risk")

    print("\n--- Applicant 3: High Risk ---")
    print("Credit score: 620 (fair)")
    print("Income: $45,000/year")
    print("Credit history: 2 late payments last year")
    print("Utilization: 85% (high)")
    print("Transaction history: Frequent overdrafts, gambling transactions")
    print("Recent applications: 5 credit cards last 3 months")
    print("Loan request: $15,000 personal loan")

    print("\nEmbedding-based assessment:")
    print("  Borrower embedding analysis:")
    print("    - Similar to: High-default borrowers")
    print("    - Cluster: Financial stress segment")
    print("    - Red flags: High utilization, credit seeking, gambling")
    print("  ")
    print("  Decision: REJECT")
    print("  Default probability: 18.5%")
    print("  Confidence: 0.85")
    print("  ")
    print("  Adverse action reasons:")
    print("    1. High credit utilization (85%)")
    print("    2. Recent late payments")
    print("    3. Multiple recent credit applications")
    print("    4. Transaction patterns indicate financial stress")
    print("\n→ Responsible rejection with explanation")

    print("\n--- Results Across Portfolio ---")
    print("Approval rate: 72% (vs 60% traditional)")
    print("Default rate: 3.8% (vs 4.5% traditional)")
    print("Credit invisibles served: 15,000 new customers")
    print("Average interest rate: 10.2%")
    print("Portfolio ROI: 8.5% (vs 7.2% traditional)")
    print("\n→ Expanded access + better risk management")


# Uncomment to run:
# credit_risk_example()
