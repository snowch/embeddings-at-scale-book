from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Code from Chapter 18
# Book: Embeddings at Scale

"""
Regulatory Compliance Automation

Architecture:
1. Document encoder: Embed regulations, policies, procedures
2. Transaction encoder: Embed financial transactions
3. Communication encoder: Embed emails, chats, calls
4. Violation detector: Identify actions similar to known violations
5. Report generator: Automated regulatory reporting

Use cases:
- AML: Detect suspicious transaction patterns
- Trading surveillance: Identify market manipulation, insider trading
- Privacy: Monitor GDPR, CCPA compliance
- KYC: Verify customer identities, screen against sanctions lists
- Recordkeeping: Ensure complete audit trails

Production considerations:
- Accuracy: Minimize false positives (costly manual review)
- Coverage: Monitor all transactions, communications
- Latency: Real-time blocking for high-risk transactions
- Auditability: Explain why violations were flagged
"""

@dataclass
class ComplianceRule:
    """
    Regulatory or internal compliance rule

    Attributes:
        rule_id: Unique identifier
        rule_type: Type (AML, trading, privacy, etc.)
        description: Human-readable description
        examples: Example violations
        severity: Low, medium, high, critical
        actions: Actions to take when triggered
        embedding: Learned rule embedding
    """
    rule_id: str
    rule_type: str
    description: str
    examples: List[str]
    severity: str
    actions: List[str]
    embedding: Optional[np.ndarray] = None

@dataclass
class ComplianceEvent:
    """
    Event requiring compliance review

    Attributes:
        event_id: Unique identifier
        event_type: Type (transaction, communication, etc.)
        timestamp: When event occurred
        entities: Involved entities (customers, employees, etc.)
        content: Event content (transaction details, message text, etc.)
        matched_rules: Rules potentially violated
        risk_score: Compliance risk score (0-1)
        requires_review: Whether manual review needed
    """
    event_id: str
    event_type: str
    timestamp: float
    entities: List[str]
    content: Dict[str, Any]
    matched_rules: List[str]
    risk_score: float
    requires_review: bool

class ComplianceEncoder(nn.Module):
    """
    Encode compliance rules and events

    Uses same embedding space for rules and events,
    enabling semantic similarity matching.

    Training:
    - Contrastive: Violations close to violated rules
    - Classification: Predict rule type from content
    - Few-shot: Learn from limited violation examples
    """

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Text encoder (for rules and event descriptions)
        self.text_encoder = nn.LSTM(
            input_size=768,  # BERT embeddings
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Structured data encoder (for transaction features)
        self.structured_encoder = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )

    def forward(
        self,
        text_embeddings: torch.Tensor,
        structured_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode compliance rules or events

        Args:
            text_embeddings: Text embeddings (batch_size, seq_len, 768)
            structured_features: Structured features (batch_size, 50)

        Returns:
            Compliance embeddings (batch_size, embedding_dim)
        """
        # Encode text
        _, (text_hidden, _) = self.text_encoder(text_embeddings)
        text_emb = text_hidden[-1]

        # Encode structured features
        structured_emb = self.structured_encoder(structured_features)

        # Fuse
        combined = torch.cat([text_emb, structured_emb], dim=1)
        compliance_emb = self.fusion(combined)

        # Normalize
        compliance_emb = F.normalize(compliance_emb, p=2, dim=1)

        return compliance_emb

# Example: AML transaction monitoring
def aml_monitoring_example():
    """
    Anti-money laundering transaction monitoring

    Detects suspicious patterns:
    - Structuring (splitting large transactions to avoid reporting)
    - Rapid movement (funds in and out quickly)
    - Round-tripping (circular money flows)
    - Shell company usage
    - Geographic anomalies (high-risk jurisdictions)
    """

    print("=== AML Transaction Monitoring ===")
    print("\nObjective: Detect money laundering patterns")
    print("Challenge: Criminals constantly evolve tactics")
    print("Solution: Learn embeddings of suspicious behavior")

    print("\n--- Normal Transaction Pattern ---")
    print("Customer: John Smith")
    print("Account age: 5 years")
    print("Transactions:")
    print("  - Weekly paycheck deposits: $2,500")
    print("  - Monthly rent: $1,800")
    print("  - Utilities, groceries, entertainment")
    print("  - Occasional savings transfers: $500")

    print("\nAssessment:")
    print("  Embedding analysis: Normal consumer banking cluster")
    print("  Risk score: 0.02 (low)")
    print("  Action: No review required")

    print("\n--- Structuring Pattern (Money Laundering) ---")
    print("Customer: Jane Doe")
    print("Account age: 3 months")
    print("Recent activity (last 7 days):")
    print("  - Day 1: Cash deposit $9,500 (just under $10K reporting threshold)")
    print("  - Day 2: Cash deposit $9,800")
    print("  - Day 3: Cash deposit $9,700")
    print("  - Day 4: Wire transfer out $28,000 to offshore account")

    print("\nEmbedding analysis:")
    print("  Similar to: Known structuring cases")
    print("  Pattern: Multiple sub-threshold deposits → large outbound transfer")
    print("  Red flags:")
    print("    - Deposits just under reporting threshold")
    print("    - Rapid sequence of cash deposits")
    print("    - Immediate outbound transfer")
    print("    - Offshore destination")

    print("\nAssessment:")
    print("  Risk score: 0.94 (very high)")
    print("  Matched rules: Structuring, rapid movement")
    print("  Action: File SAR (Suspicious Activity Report)")
    print("  Freeze account pending investigation")

    print("\n--- Round-Tripping Pattern ---")
    print("Network of accounts:")
    print("  Company A → Company B ($500K)")
    print("  Company B → Company C ($480K)")
    print("  Company C → Company D ($460K)")
    print("  Company D → Company A ($440K)")
    print("All within 48 hours, no economic purpose")

    print("\nGraph embedding analysis:")
    print("  Pattern: Circular money flow")
    print("  Similar to: Known round-tripping schemes")
    print("  Red flags:")
    print("    - Circular transaction graph")
    print("    - Rapid timing (no genuine business delay)")
    print("    - Decreasing amounts (fees laundering)")
    print("    - Shell companies (minimal operations)")

    print("\nAssessment:")
    print("  Risk score: 0.89 (high)")
    print("  Matched rules: Round-tripping, shell company usage")
    print("  Action: File SAR for all accounts")
    print("  Investigate beneficial ownership")

    print("\n--- System Performance ---")
    print("Transactions monitored: 10M per day")
    print("Alerts generated: 2,500 per day (0.025%)")
    print("True positives: 1,800 per day (72% precision)")
    print("False positive reduction: 85% vs rule-based")
    print("SAR filings: 600 per day")
    print("Regulatory compliance: 100%")
    print("\n→ Effective detection with manageable false positives")

# Uncomment to run:
# aml_monitoring_example()
