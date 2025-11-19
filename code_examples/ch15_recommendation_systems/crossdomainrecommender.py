# Code from Chapter 15
# Book: Embeddings at Scale

"""
Cross-Domain Recommendation Transfer

Approaches:
1. Shared user embeddings: Learn single user embedding across domains
2. Domain-specific item embeddings: Separate embeddings per domain
3. Transfer learning: Pre-train on rich domain, fine-tune on sparse domain
4. Multi-task learning: Joint optimization across domains

Techniques:
- Domain adaptation: Align feature distributions across domains
- Meta-learning: Learn to adapt quickly to new domains
- Knowledge distillation: Transfer from complex to simple models
- Auxiliary tasks: Use rich domain as auxiliary signal

Benefits:
- Better cold start in sparse domains
- Improved recommendations from shared preferences
- Reduced training cost (transfer instead of train from scratch)
"""

# Placeholder classes - see from.py for full implementation
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CollaborativeFilteringModel(nn.Module):
    """Placeholder for CollaborativeFilteringModel."""
    def __init__(self):
        super().__init__()

    def forward(self, user_ids, item_ids):
        import torch
        return torch.randn(len(user_ids))

@dataclass
class Interaction:
    """Placeholder for Interaction."""
    user_id: str
    item_id: str
    rating: float = 0.0

class CrossDomainRecommender(nn.Module):
    """
    Multi-domain recommender with shared user embeddings

    Architecture:
    - Shared user encoder: Maps users to shared embedding space
    - Domain-specific item encoders: Separate embeddings per domain
    - Domain-specific scoring: Dot product within each domain

    Training:
    - Multi-task learning: Joint optimization across domains
    - Weighted loss: Balance domains by interaction volume
    - Hard parameter sharing: User encoder shared across all domains

    Inference:
    - Encode user once (shared encoder)
    - Retrieve from any domain using domain-specific item embeddings
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_users: int = 1000000,
        num_items_per_domain: Dict[str, int] = None
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Shared user encoder
        self.user_encoder = nn.Embedding(num_users, embedding_dim)

        # Domain-specific item encoders
        self.item_encoders = nn.ModuleDict()
        for domain, num_items in num_items_per_domain.items():
            self.item_encoders[domain] = nn.Embedding(num_items, embedding_dim)

        self.domains = list(num_items_per_domain.keys())

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        domain: str
    ) -> torch.Tensor:
        """
        Predict scores for user-item pairs in given domain

        Args:
            user_ids: User IDs (batch_size,)
            item_ids: Item IDs (batch_size,)
            domain: Domain name

        Returns:
            Scores (batch_size,)
        """
        # Encode users (shared across domains)
        user_emb = self.user_encoder(user_ids)
        user_emb = F.normalize(user_emb, p=2, dim=1)

        # Encode items (domain-specific)
        item_emb = self.item_encoders[domain](item_ids)
        item_emb = F.normalize(item_emb, p=2, dim=1)

        # Dot product scoring
        scores = (user_emb * item_emb).sum(dim=1)

        return scores

    def encode_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Encode users to shared embedding"""
        user_emb = self.user_encoder(user_ids)
        return F.normalize(user_emb, p=2, dim=1)

    def encode_items(
        self,
        item_ids: torch.Tensor,
        domain: str
    ) -> torch.Tensor:
        """Encode items in specific domain"""
        item_emb = self.item_encoders[domain](item_ids)
        return F.normalize(item_emb, p=2, dim=1)

class TransferLearningRecommender:
    """
    Transfer learning from rich domain to sparse domain

    Strategy:
    1. Pre-train on rich domain (movies: 1B interactions)
    2. Transfer user encoder to sparse domain (books: 10M interactions)
    3. Fine-tune on sparse domain with regularization

    Benefits:
    - Sparse domain benefits from rich domain patterns
    - Faster convergence in sparse domain
    - Better cold start performance
    """

    def __init__(
        self,
        embedding_dim: int = 128
    ):
        self.embedding_dim = embedding_dim

        # Source domain model (rich domain)
        self.source_model: Optional[CollaborativeFilteringModel] = None

        # Target domain model (sparse domain)
        self.target_model: Optional[CollaborativeFilteringModel] = None

        print("Initialized Transfer Learning Recommender")

    def pretrain_source(
        self,
        source_interactions: List[Interaction],
        num_epochs: int = 10
    ):
        """
        Pre-train on source domain (rich domain)

        Args:
            source_interactions: Interactions in source domain
            num_epochs: Training epochs
        """
        print(f"\nPre-training on source domain ({len(source_interactions)} interactions)...")

        # Extract unique users/items
        user_ids = set(i.user_id for i in source_interactions)
        item_ids = set(i.item_id for i in source_interactions)

        # Initialize source model
        self.source_model = CollaborativeFilteringModel(
            embedding_dim=self.embedding_dim,
            num_users=len(user_ids),
            num_items=len(item_ids)
        )

        # Train (simplified - in production: full training loop)
        print("✓ Pre-trained source model")

    def transfer_to_target(
        self,
        target_interactions: List[Interaction],
        num_epochs: int = 5,
        freeze_user_encoder: bool = False
    ):
        """
        Transfer to target domain (sparse domain)

        Args:
            target_interactions: Interactions in target domain
            num_epochs: Fine-tuning epochs
            freeze_user_encoder: Whether to freeze user encoder (transfer only)
        """
        if self.source_model is None:
            raise ValueError("Must pre-train source model first")

        print(f"\nTransferring to target domain ({len(target_interactions)} interactions)...")

        # Extract unique users/items
        user_ids = set(i.user_id for i in target_interactions)
        item_ids = set(i.item_id for i in target_interactions)

        # Initialize target model
        self.target_model = CollaborativeFilteringModel(
            embedding_dim=self.embedding_dim,
            num_users=len(user_ids),
            num_items=len(item_ids)
        )

        # Transfer user encoder weights
        self.target_model.user_encoder.load_state_dict(
            self.source_model.user_encoder.state_dict()
        )

        # Optionally freeze user encoder
        if freeze_user_encoder:
            for param in self.target_model.user_encoder.parameters():
                param.requires_grad = False

        # Fine-tune on target domain
        # (simplified - in production: full training loop)
        print("✓ Transferred and fine-tuned on target domain")

# Example: Movies → Books transfer
def cross_domain_transfer_example():
    """
    Transfer learning from movies (rich) to books (sparse)

    Scenario:
    - Movies: 1B interactions, well-trained model
    - Books: 10M interactions, sparse data
    - Transfer user preferences from movies to books

    Hypothesis: Users with similar movie tastes have similar book tastes
    """

    print("=== Cross-Domain Transfer Learning ===")
    print("\nScenario: Transfer from Movies to Books")

    # Initialize multi-domain recommender
    model = CrossDomainRecommender(
        embedding_dim=64,
        num_users=1000,
        num_items_per_domain={'movies': 10000, 'books': 5000}
    )

    print("\nModel architecture:")
    print(f"  Shared user embedding: {model.embedding_dim}-dim")
    print(f"  Movies: {10000} items")
    print(f"  Books: {5000} items")

    # Simulate user with movies history
    user_id = torch.tensor([0])

    # User watched movies
    print("\nUser watched movies: 5, 12, 23, 45, 67")

    # Encode user (from movie viewing history)
    user_emb = model.encode_user(user_id)

    # Recommend books (transfer to books domain)
    print("\nRecommending books based on movie preferences...")

    # Get book embeddings
    book_ids = torch.arange(100)  # Sample 100 books
    book_embs = model.encode_items(book_ids, domain='books')

    # Compute scores
    scores = torch.matmul(book_embs, user_emb.T).squeeze()

    # Top-5 books
    top_scores, top_indices = torch.topk(scores, k=5)

    print("\nTop 5 book recommendations:")
    for idx, score in zip(top_indices, top_scores):
        print(f"  Book {idx.item()}: {score.item():.3f}")

    print("\n✓ Cross-domain transfer successful")

# Uncomment to run:
# cross_domain_transfer_example()
