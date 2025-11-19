# Code from Chapter 15
# Book: Embeddings at Scale

"""
Cold Start Solutions

Strategies:
1. Content-based initialization: Use item features to estimate embeddings
2. Meta-learning: Learn to adapt quickly from few interactions
3. Transfer learning: Leverage embeddings from similar domains
4. Hybrid models: Combine collaborative + content signals

Techniques:
- MAML (Model-Agnostic Meta-Learning): Learn initialization
- Few-shot learning: Adapt from 1-5 examples
- Knowledge distillation: Transfer from large model to production model
- Multi-task learning: Learn across multiple domains simultaneously
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentBasedItemEmbedding(nn.Module):
    """
    Generate item embeddings from content features

    Addresses: Cold start for new items

    Architecture:
    - Feature encoder (for text, images, metadata)
    - Projection to embedding space
    - Trained to match collaborative embeddings

    Usage:
    1. Train on existing items (content → collaborative embedding)
    2. For new items: Generate embedding from content
    3. Insert into recommendation index immediately
    """

    def __init__(
        self,
        content_dim: int = 512,
        embedding_dim: int = 128
    ):
        super().__init__()

        # Content encoder
        self.content_encoder = nn.Sequential(
            nn.Linear(content_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, content_features: torch.Tensor) -> torch.Tensor:
        """
        Generate embedding from content

        Args:
            content_features: Content features (batch, content_dim)

        Returns:
            Item embeddings (batch, embedding_dim)
        """
        emb = self.content_encoder(content_features)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

class MetaLearningRecommender(nn.Module):
    """
    Meta-learning for cold start users

    Addresses: Cold start for new users

    Approach: MAML (Model-Agnostic Meta-Learning)
    - Learn model initialization that adapts quickly from few examples
    - Given 1-5 user interactions, fine-tune to estimate preferences
    - Generalize to new users with minimal data

    Training:
    - Sample user tasks (each user = task)
    - For each task: Split interactions into support (1-5) and query (rest)
    - Meta-train to minimize loss on query set after adapting on support set

    Inference:
    - New user: Collect 1-5 interactions (support set)
    - Fine-tune model on support set (few gradient steps)
    - Generate recommendations
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_items: int = 10000
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Item embeddings
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # User preference model (learns from support set)
        self.preference_model = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        user_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict preferences given user context

        Args:
            item_ids: Item IDs (batch,)
            user_context: User preference context (embedding_dim,)

        Returns:
            Preference scores (batch,)
        """
        # Get item embeddings
        item_embs = self.item_embeddings(item_ids)  # (batch, embedding_dim)

        # Combine with user context
        # In MAML: user_context is learned from support set
        combined = item_embs * user_context.unsqueeze(0)  # (batch, embedding_dim)

        # Predict preference
        scores = self.preference_model(combined).squeeze(-1)  # (batch,)

        return scores

    def adapt(
        self,
        support_items: torch.Tensor,
        support_labels: torch.Tensor,
        num_steps: int = 5,
        learning_rate: float = 0.01
    ) -> torch.Tensor:
        """
        Adapt to new user from support set

        Args:
            support_items: Items in support set (num_support,)
            support_labels: Labels (1=positive, 0=negative) (num_support,)
            num_steps: Number of adaptation steps
            learning_rate: Adaptation learning rate

        Returns:
            User context embedding (embedding_dim,)
        """
        # Initialize user context (learnable)
        user_context = torch.zeros(self.embedding_dim, requires_grad=True)

        # Optimizer for user context
        optimizer = torch.optim.SGD([user_context], lr=learning_rate)

        # Adapt on support set
        for _ in range(num_steps):
            # Predict on support set
            scores = self.forward(support_items, user_context)

            # Compute loss
            loss = F.binary_cross_entropy_with_logits(scores, support_labels)

            # Update user context
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return user_context.detach()

class HybridRecommender:
    """
    Hybrid collaborative + content-based recommender

    Addresses: Both user and item cold start

    Strategy:
    - Collaborative signal when available (existing users/items)
    - Content signal for cold start (new users/items)
    - Smooth transition as interactions accumulate

    Components:
    1. Collaborative model: User/item embeddings from interactions
    2. Content model: Item embeddings from features
    3. Blending: Weight collaborative vs content based on data availability

    Blending formula:
    score = α * collaborative_score + (1-α) * content_score
    where α = min(num_interactions / threshold, 1.0)
    """

    def __init__(
        self,
        collaborative_model: CollaborativeFilteringModel,
        content_model: ContentBasedItemEmbedding,
        cold_start_threshold: int = 10
    ):
        """
        Args:
            collaborative_model: CF model for existing users/items
            content_model: Content model for cold start
            cold_start_threshold: Interactions needed for full collaborative weight
        """
        self.collaborative_model = collaborative_model
        self.content_model = content_model
        self.cold_start_threshold = cold_start_threshold

        # Track interaction counts
        self.user_interaction_counts: Dict[str, int] = {}
        self.item_interaction_counts: Dict[str, int] = {}

    def get_blending_weight(self, num_interactions: int) -> float:
        """
        Compute blending weight for collaborative signal

        Args:
            num_interactions: Number of interactions

        Returns:
            Weight in [0, 1] for collaborative model
        """
        return min(num_interactions / self.cold_start_threshold, 1.0)

    def recommend_hybrid(
        self,
        user_id: str,
        item_features: Dict[str, np.ndarray],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations with hybrid approach

        Args:
            user_id: User ID
            item_features: Content features for each item {item_id: features}
            top_k: Number of recommendations

        Returns:
            List of (item_id, score) tuples
        """
        # Get user interaction count
        user_interactions = self.user_interaction_counts.get(user_id, 0)
        user_weight = self.get_blending_weight(user_interactions)

        recommendations = {}

        for item_id, features in item_features.items():
            # Get item interaction count
            item_interactions = self.item_interaction_counts.get(item_id, 0)
            item_weight = self.get_blending_weight(item_interactions)

            # Overall collaborative weight (min of user and item weights)
            collab_weight = min(user_weight, item_weight)
            content_weight = 1.0 - collab_weight

            # Collaborative score (if available)
            collab_score = 0.0
            if collab_weight > 0:
                # Get from collaborative model
                # In production: Actual model inference
                collab_score = 0.5  # Placeholder

            # Content score
            content_score = 0.0
            if content_weight > 0:
                # Get from content model
                # In production: Actual model inference
                content_score = 0.5  # Placeholder

            # Blended score
            final_score = collab_weight * collab_score + content_weight * content_score
            recommendations[item_id] = final_score

        # Sort and return top-k
        sorted_items = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]

# Example: Cold start for new user
def cold_start_example():
    """
    Handle cold start for new movie streaming user

    Scenario:
    - New user signs up
    - Watches 2 movies (support set)
    - Generate personalized recommendations

    Approach: Meta-learning
    - Adapt quickly from 2 examples
    - Leverage learned initialization
    """

    # Initialize meta-learning model
    model = MetaLearningRecommender(embedding_dim=64, num_items=100)

    # Simulate new user
    print("=== New User Cold Start ===")
    print("User watches 2 movies:")

    # Support set: User watched movies 5 and 12
    support_items = torch.tensor([5, 12], dtype=torch.long)
    support_labels = torch.tensor([1.0, 1.0], dtype=torch.float32)  # Both positive

    print("  Movie 5 (Action)")
    print("  Movie 12 (Action)")

    # Adapt to user preferences
    user_context = model.adapt(
        support_items,
        support_labels,
        num_steps=10,
        learning_rate=0.01
    )

    print("\n✓ Adapted user context from 2 examples")

    # Generate recommendations
    print("\nRecommendations:")
    all_items = torch.arange(100, dtype=torch.long)
    with torch.no_grad():
        scores = model.forward(all_items, user_context)

    # Get top-5
    top_scores, top_indices = torch.topk(scores, k=5)

    for idx, score in zip(top_indices, top_scores):
        print(f"  Movie {idx.item()}: score = {score.item():.3f}")

# Uncomment to run:
# cold_start_example()
