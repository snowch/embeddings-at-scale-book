# Code from Chapter 17
# Book: Embeddings at Scale

"""
Embedding-Driven Business Rules

Architecture:
1. Entity encoders: Map customers/products/transactions to embeddings
2. Historical decision database: Past decisions with outcomes
3. Decision retrieval: Find similar past cases via ANN search
4. Decision synthesis: Aggregate outcomes from similar cases
5. Explainability: Surface which past cases influenced decision

Techniques:
- Case-based reasoning: Retrieve similar cases, apply their outcomes
- Decision boundary learning: Train classifier in embedding space
- Meta-learning: Learn to learn decision rules from few examples
- Hybrid: Embeddings + explicit rules for regulatory compliance

Production considerations:
- Latency: <100ms for real-time decisions
- Explainability: SHAP values, similar case explanations
- Override mechanisms: Human review for edge cases
- Fairness: Monitor for demographic disparities
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BusinessCase:
    """
    Historical business decision case

    Attributes:
        case_id: Unique identifier
        entity_type: Type (customer, product, transaction)
        entity_id: Entity identifier
        context: Decision context features
        decision: Decision made (approve, reject, price, etc.)
        outcome: Observed outcome (default, profit, churn, etc.)
        timestamp: When decision occurred
        embedding: Learned entity embedding
    """

    case_id: str
    entity_type: str
    entity_id: str
    context: Dict[str, Any]
    decision: Any
    outcome: Optional[Any] = None
    timestamp: Optional[float] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class DecisionRequest:
    """
    Request for automated decision

    Attributes:
        request_id: Unique identifier
        entity_type: Type (customer, product, transaction)
        entity_id: Entity identifier
        context: Current context features
        required_confidence: Minimum confidence for auto-decision
        human_review: Whether to force human review
    """

    request_id: str
    entity_type: str
    entity_id: str
    context: Dict[str, Any]
    required_confidence: float = 0.8
    human_review: bool = False

    def __post_init__(self):
        if self.context is None:
            self.context = {}


class EntityEncoder(nn.Module):
    """
    Encode entities to embeddings for decision making

    Architecture:
    - Demographic features: Age, location, income, etc.
    - Behavioral features: Transaction history, engagement patterns
    - Contextual features: Current situation, external factors
    - MLP: Combine features into entity embedding

    Training:
    - Metric learning: Entities with similar outcomes close
    - Outcome prediction: Embedding predicts decision outcome
    - Contrastive: Positive outcomes close, negative outcomes far
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_categorical_features: int = 10,
        num_numerical_features: int = 20,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Categorical feature embeddings
        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(1000, 16) for _ in range(num_categorical_features)]
        )

        # Numerical feature encoder
        self.numerical_encoder = nn.Sequential(
            nn.Linear(num_numerical_features, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 64)
        )

        # Combined encoder
        feature_dim = num_categorical_features * 16 + 64
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, embedding_dim)
        )

    def forward(
        self, categorical_features: torch.Tensor, numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode entities to embeddings

        Args:
            categorical_features: Categorical features (batch_size, num_categorical)
            numerical_features: Numerical features (batch_size, num_numerical)

        Returns:
            Entity embeddings (batch_size, embedding_dim)
        """
        # Embed categorical features
        cat_embs = []
        for i, emb_layer in enumerate(self.categorical_embeddings):
            cat_embs.append(emb_layer(categorical_features[:, i]))
        cat_emb = torch.cat(cat_embs, dim=1)

        # Encode numerical features
        num_emb = self.numerical_encoder(numerical_features)

        # Combine
        combined = torch.cat([cat_emb, num_emb], dim=1)

        # Encode to entity embedding
        entity_emb = self.feature_encoder(combined)

        # Normalize
        entity_emb = F.normalize(entity_emb, p=2, dim=1)

        return entity_emb


class DecisionModel(nn.Module):
    """
    Predict decision outcomes from embeddings

    Architecture:
    - Entity embedding input
    - MLP classifier/regressor
    - Output: Decision outcome prediction

    Training:
    - Classification: Binary (approve/reject) or multi-class
    - Regression: Continuous outcome (LTV, default probability)
    - Multi-task: Predict multiple outcomes jointly
    """

    def __init__(
        self, embedding_dim: int = 128, num_outcomes: int = 2, task: str = "classification"
    ):
        super().__init__()
        self.task = task

        self.decision_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_outcomes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict outcomes from embeddings

        Args:
            embeddings: Entity embeddings (batch_size, embedding_dim)

        Returns:
            Predictions (batch_size, num_outcomes)
        """
        logits = self.decision_head(embeddings)

        if self.task == "classification":
            return F.softmax(logits, dim=1)
        else:
            return logits


class CaseBasedReasoning:
    """
    Make decisions by retrieving similar historical cases

    Approach:
    1. New request arrives
    2. Encode request to embedding
    3. Retrieve k similar past cases (ANN search)
    4. Aggregate outcomes from similar cases
    5. Make decision based on historical outcomes
    6. Explain via similar cases

    Advantages:
    - Intuitive: Decisions based on "what happened before"
    - Explainable: Can show similar past cases
    - Adaptive: Automatically incorporates new cases
    - No retraining: Just add cases to database
    """

    def __init__(self, encoder: EntityEncoder, embedding_dim: int = 128, k_neighbors: int = 10):
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.k_neighbors = k_neighbors

        # Historical case database
        self.cases: List[BusinessCase] = []
        self.case_embeddings: Optional[np.ndarray] = None

    def add_case(self, case: BusinessCase, embedding: np.ndarray):
        """
        Add historical case to database

        Args:
            case: Business case with outcome
            embedding: Case embedding
        """
        case.embedding = embedding
        self.cases.append(case)

        # Rebuild embedding matrix
        if self.case_embeddings is None:
            self.case_embeddings = embedding.reshape(1, -1)
        else:
            self.case_embeddings = np.vstack([self.case_embeddings, embedding.reshape(1, -1)])

    def retrieve_similar_cases(
        self, query_embedding: np.ndarray, k: Optional[int] = None
    ) -> List[Tuple[BusinessCase, float]]:
        """
        Retrieve k most similar historical cases

        Args:
            query_embedding: Query embedding (embedding_dim,)
            k: Number of neighbors (defaults to self.k_neighbors)

        Returns:
            List of (case, similarity_score) tuples
        """
        if k is None:
            k = self.k_neighbors

        if len(self.cases) == 0:
            return []

        # Compute similarities
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(self.case_embeddings, query_embedding)

        # Get top k
        top_indices = np.argsort(similarities)[-k:][::-1]

        similar_cases = [(self.cases[idx], similarities[idx]) for idx in top_indices]

        return similar_cases

    def make_decision(
        self, request: DecisionRequest, request_embedding: np.ndarray
    ) -> Tuple[Any, float, List[BusinessCase]]:
        """
        Make decision based on similar cases

        Args:
            request: Decision request
            request_embedding: Request embedding

        Returns:
            (decision, confidence, supporting_cases)
        """
        # Retrieve similar cases
        similar_cases = self.retrieve_similar_cases(request_embedding)

        if len(similar_cases) == 0:
            # No historical cases - force human review
            return None, 0.0, []

        # Aggregate outcomes
        # For binary decisions: weighted vote
        # For continuous: weighted average

        total_weight = 0.0
        weighted_outcome = 0.0
        supporting_cases = []

        for case, similarity in similar_cases:
            if case.outcome is None:
                continue

            weight = similarity  # Could use exponential: exp(10 * similarity)
            weighted_outcome += weight * case.outcome
            total_weight += weight
            supporting_cases.append(case)

        if total_weight == 0:
            return None, 0.0, []

        # Compute decision and confidence
        avg_outcome = weighted_outcome / total_weight

        # For binary: outcome is 0/1, avg_outcome is probability
        if isinstance(supporting_cases[0].outcome, bool):
            decision = avg_outcome > 0.5
            confidence = abs(avg_outcome - 0.5) * 2  # Scale to [0, 1]
        else:
            decision = avg_outcome
            # Confidence based on similarity of top cases
            top_similarities = [s for _, s in similar_cases[:3]]
            confidence = np.mean(top_similarities)

        return decision, confidence, supporting_cases


class HybridDecisionSystem:
    """
    Combine embedding-based decisions with rule-based constraints

    Architecture:
    1. Embedding model makes initial decision
    2. Rule engine enforces hard constraints
    3. Hybrid score combines model + rules

    Use cases:
    - Regulatory compliance: Hard rules for legal requirements
    - Business constraints: Inventory, capacity limits
    - Risk limits: Maximum exposure per category
    - Fairness: Demographic parity constraints
    """

    def __init__(
        self,
        encoder: EntityEncoder,
        decision_model: DecisionModel,
        rules: Optional[Dict[str, Any]] = None,
    ):
        self.encoder = encoder
        self.decision_model = decision_model
        self.rules = rules or {}

    def check_rules(self, request: DecisionRequest) -> Tuple[bool, List[str]]:
        """
        Check hard constraint rules

        Args:
            request: Decision request

        Returns:
            (passed, violated_rules)
        """
        violated_rules = []

        # Example rules
        if "minimum_age" in self.rules:
            if request.context.get("age", 0) < self.rules["minimum_age"]:
                violated_rules.append(f"Age below minimum ({self.rules['minimum_age']})")

        if "maximum_amount" in self.rules:
            if request.context.get("amount", 0) > self.rules["maximum_amount"]:
                violated_rules.append(f"Amount exceeds maximum ({self.rules['maximum_amount']})")

        if "required_fields" in self.rules:
            for field in self.rules["required_fields"]:
                if field not in request.context:
                    violated_rules.append(f"Missing required field: {field}")

        return len(violated_rules) == 0, violated_rules

    def make_decision(
        self, request: DecisionRequest, request_embedding: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Make decision combining model and rules

        Args:
            request: Decision request
            request_embedding: Request embedding

        Returns:
            Decision result with explanation
        """
        # Check hard rules first
        rules_passed, violated_rules = self.check_rules(request)

        if not rules_passed:
            return {
                "decision": "reject",
                "confidence": 1.0,
                "reason": "rule_violation",
                "violated_rules": violated_rules,
                "model_prediction": None,
            }

        # Get model prediction
        with torch.no_grad():
            prediction = self.decision_model(request_embedding.unsqueeze(0))
            prediction = prediction.squeeze(0).cpu().numpy()

        # Interpret prediction
        if self.decision_model.task == "classification":
            decision = "approve" if prediction[1] > 0.5 else "reject"
            confidence = max(prediction)
        else:
            decision = float(prediction[0])
            confidence = 0.9  # Placeholder

        # Check confidence threshold
        if confidence < request.required_confidence:
            return {
                "decision": "human_review",
                "confidence": confidence,
                "reason": "low_confidence",
                "model_prediction": decision,
                "violated_rules": [],
            }

        return {
            "decision": decision,
            "confidence": confidence,
            "reason": "model_prediction",
            "model_prediction": decision,
            "violated_rules": [],
        }


# Example: Credit approval system
def credit_approval_example():
    """
    Credit approval using embedding-driven business rules

    Traditional approach:
    - Credit score > 700
    - Income > $50K
    - Debt-to-income < 40%
    - No recent delinquencies

    Embedding approach:
    - Encode applicant to embedding
    - Find similar past applicants
    - Approve if similar applicants had low default rates
    - Automatically adapts as new data arrives
    """

    print("=== Credit Approval System ===")
    print("\nTraditional rules:")
    print("  IF credit_score > 700")
    print("  AND income > $50,000")
    print("  AND debt_to_income < 40%")
    print("  AND no_recent_delinquencies")
    print("  THEN approve")

    print("\n--- Application 1: Clear Approve ---")
    print("Credit score: 750")
    print("Income: $80,000")
    print("Debt-to-income: 25%")
    print("Recent delinquencies: 0")
    print("Traditional: APPROVE (all rules passed)")
    print("Embedding: APPROVE (confidence: 0.95)")
    print("  Similar cases: 50 approved, 1 defaulted (2% default rate)")

    print("\n--- Application 2: Edge Case ---")
    print("Credit score: 680")
    print("Income: $65,000")
    print("Debt-to-income: 38%")
    print("Recent delinquencies: 0")
    print("Traditional: REJECT (credit score < 700)")
    print("Embedding: APPROVE (confidence: 0.78)")
    print("  Similar cases: 30 approved, 2 defaulted (6.7% default rate)")
    print("  Reason: Strong income and stable employment history")
    print("  → System learns that 680 score + high income often performs well")

    print("\n--- Application 3: Novel Scenario ---")
    print("Credit score: 720")
    print("Income: $45,000 (gig economy)")
    print("Debt-to-income: 35%")
    print("Recent delinquencies: 0")
    print("Traditional: REJECT (income < $50K)")
    print("Embedding: HUMAN REVIEW (confidence: 0.65)")
    print("  Similar cases: Only 5 gig economy workers in database")
    print("  → Insufficient data, flag for human underwriter")

    print("\n--- Application 4: Regulatory Violation ---")
    print("Credit score: 780")
    print("Income: $120,000")
    print("Debt-to-income: 20%")
    print("Age: 17")
    print("Traditional: REJECT (age < 18)")
    print("Embedding: REJECT (rule violation)")
    print("  Model prediction: Approve")
    print("  Hard rule: Minimum age 18")
    print("  → Rules override model for regulatory compliance")


# Uncomment to run:
# credit_approval_example()
