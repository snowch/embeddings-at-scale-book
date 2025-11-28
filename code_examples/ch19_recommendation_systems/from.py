# Code from Chapter 15
# Book: Embeddings at Scale

"""
Embedding-Based Collaborative Filtering

Architecture:
1. User encoder: Maps user features → user embedding
2. Item encoder: Maps item features → item embedding
3. Interaction prediction: score(user, item) = user_emb · item_emb
4. Training: Optimize embeddings to predict observed interactions

Techniques:
- Two-tower architecture: Separate encoders for users and items
- Negative sampling: Sample items user didn't interact with
- Hard negative mining: Focus on plausible but incorrect recommendations
- Batch training: Process millions of interactions per batch

Production optimizations:
- Pre-compute item embeddings (items change slowly)
- Online user embedding computation (users change frequently)
- ANN search for retrieval (Faiss, ScaNN)
- A/B testing framework for evaluation
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class User:
    """
    User with interaction history

    Attributes:
        user_id: Unique identifier
        features: User features (age, location, etc.)
        interactions: List of item IDs user interacted with
        embedding: Learned user embedding
    """

    user_id: str
    features: Dict[str, any] = None
    interactions: List[str] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.features is None:
            self.features = {}
        if self.interactions is None:
            self.interactions = []


@dataclass
class Item:
    """
    Item available for recommendation

    Attributes:
        item_id: Unique identifier
        features: Item features (category, price, etc.)
        content: Item content (text, image, etc.)
        embedding: Learned item embedding
        popularity: Interaction count (for popularity bias)
    """

    item_id: str
    features: Dict[str, any] = None
    content: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    popularity: int = 0

    def __post_init__(self):
        if self.features is None:
            self.features = {}


@dataclass
class Interaction:
    """
    User-item interaction

    Attributes:
        user_id: User who interacted
        item_id: Item that was interacted with
        interaction_type: Type (click, purchase, rating, etc.)
        rating: Explicit rating (1-5) or implicit (1 for positive)
        timestamp: When interaction occurred
    """

    user_id: str
    item_id: str
    interaction_type: str = "click"
    rating: float = 1.0
    timestamp: Optional[float] = None


class UserEncoder(nn.Module):
    """
    Encode user features to embedding

    Architecture:
    - Feature embedding layers (categorical features)
    - MLP to combine features
    - Projection to user embedding space

    Features:
    - User demographics (age, gender, location)
    - Interaction statistics (total interactions, recency)
    - Preferences (favorite categories)
    """

    def __init__(self, embedding_dim: int = 128, num_users: int = 1000000):
        super().__init__()
        self.embedding_dim = embedding_dim

        # User ID embedding (for collaborative signal)
        self.user_id_embedding = nn.Embedding(num_users, embedding_dim)

        # Feature projection
        # In production: Embed categorical features, normalize numerical
        self.feature_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode users to embeddings

        Args:
            user_ids: User IDs (batch_size,)

        Returns:
            User embeddings (batch_size, embedding_dim)
        """
        # Embed user IDs
        user_emb = self.user_id_embedding(user_ids)

        # Process features
        user_emb = self.feature_mlp(user_emb)

        # Normalize (for dot product scoring)
        user_emb = F.normalize(user_emb, p=2, dim=1)

        return user_emb


class ItemEncoder(nn.Module):
    """
    Encode item features to embedding

    Architecture:
    - Feature embedding layers (category, brand, etc.)
    - Content encoder (for text, images)
    - Projection to item embedding space

    Features:
    - Item metadata (category, brand, price)
    - Content embeddings (from text/image encoders)
    - Popularity signals
    """

    def __init__(self, embedding_dim: int = 128, num_items: int = 10000000):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Item ID embedding
        self.item_id_embedding = nn.Embedding(num_items, embedding_dim)

        # Feature projection
        self.feature_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode items to embeddings

        Args:
            item_ids: Item IDs (batch_size,)

        Returns:
            Item embeddings (batch_size, embedding_dim)
        """
        # Embed item IDs
        item_emb = self.item_id_embedding(item_ids)

        # Process features
        item_emb = self.feature_mlp(item_emb)

        # Normalize
        item_emb = F.normalize(item_emb, p=2, dim=1)

        return item_emb


class CollaborativeFilteringModel(nn.Module):
    """
    Two-tower embedding model for collaborative filtering

    Architecture:
    - User tower: Encodes users to embeddings
    - Item tower: Encodes items to embeddings
    - Scoring: Dot product of user and item embeddings

    Training:
    - Positive examples: Observed interactions
    - Negative examples: Sampled non-interactions
    - Loss: Binary cross-entropy or ranking loss

    Inference:
    - Encode user once
    - ANN search over item embeddings
    - Return top-k items
    """

    def __init__(
        self, embedding_dim: int = 128, num_users: int = 1000000, num_items: int = 10000000
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Encoders
        self.user_encoder = UserEncoder(embedding_dim, num_users)
        self.item_encoder = ItemEncoder(embedding_dim, num_items)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict relevance scores

        Args:
            user_ids: User IDs (batch_size,)
            item_ids: Item IDs (batch_size,)

        Returns:
            Relevance scores (batch_size,)
        """
        # Encode users and items
        user_emb = self.user_encoder(user_ids)
        item_emb = self.item_encoder(item_ids)

        # Dot product scoring
        scores = (user_emb * item_emb).sum(dim=1)

        return scores

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Encode users to embeddings"""
        return self.user_encoder(user_ids)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Encode items to embeddings"""
        return self.item_encoder(item_ids)


class RecommendationEngine:
    """
    Production recommendation system

    Components:
    1. Collaborative filtering model (user/item embeddings)
    2. Item embedding index (for fast retrieval)
    3. Negative sampler (for training)
    4. Evaluation metrics (precision, recall, NDCG)

    Features:
    - Batch training on interaction logs
    - Online serving via ANN search
    - A/B testing framework
    - Diversity and fairness controls
    """

    def __init__(self, embedding_dim: int = 128, device: str = "cuda"):
        """
        Args:
            embedding_dim: Embedding dimension
            device: Device for computation
        """
        self.embedding_dim = embedding_dim
        self.device = device if torch.cuda.is_available() else "cpu"

        # Data stores
        self.users: Dict[str, User] = {}
        self.items: Dict[str, Item] = {}
        self.interactions: List[Interaction] = []

        # Mappings
        self.user_id_to_idx: Dict[str, int] = {}
        self.item_id_to_idx: Dict[str, int] = {}

        # Model (initialized when data is loaded)
        self.model: Optional[CollaborativeFilteringModel] = None

        # Item embeddings cache (for fast serving)
        self.item_embeddings: Optional[np.ndarray] = None
        self.item_ids_list: List[str] = []

        print("Initialized Recommendation Engine")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Device: {self.device}")

    def add_user(self, user: User):
        """Add user to system"""
        self.users[user.user_id] = user

    def add_item(self, item: Item):
        """Add item to system"""
        self.items[item.item_id] = item

    def add_interaction(self, interaction: Interaction):
        """Record user-item interaction"""
        self.interactions.append(interaction)

        # Update user interaction history
        if interaction.user_id in self.users:
            self.users[interaction.user_id].interactions.append(interaction.item_id)

        # Update item popularity
        if interaction.item_id in self.items:
            self.items[interaction.item_id].popularity += 1

    def build_mappings(self):
        """Build user/item ID to index mappings"""
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.users.keys())}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(self.items.keys())}

        print("Built mappings:")
        print(f"  Users: {len(self.user_id_to_idx)}")
        print(f"  Items: {len(self.item_id_to_idx)}")

    def train(
        self,
        num_epochs: int = 10,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        neg_samples: int = 4,
    ):
        """
        Train collaborative filtering model

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            neg_samples: Number of negative samples per positive
        """
        # Build mappings if not already built
        if not self.user_id_to_idx:
            self.build_mappings()

        # Initialize model
        self.model = CollaborativeFilteringModel(
            embedding_dim=self.embedding_dim, num_users=len(self.users), num_items=len(self.items)
        ).to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Loss function
        criterion = nn.BCEWithLogitsLoss()

        print(f"\nTraining on {len(self.interactions)} interactions...")

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            # Shuffle interactions
            random.shuffle(self.interactions)

            # Mini-batch training
            for i in range(0, len(self.interactions), batch_size):
                batch_interactions = self.interactions[i : i + batch_size]

                # Prepare batch
                user_ids = []
                item_ids = []
                labels = []

                for interaction in batch_interactions:
                    user_idx = self.user_id_to_idx[interaction.user_id]
                    item_idx = self.item_id_to_idx[interaction.item_id]

                    # Positive example
                    user_ids.append(user_idx)
                    item_ids.append(item_idx)
                    labels.append(1.0)

                    # Negative examples
                    for _ in range(neg_samples):
                        # Sample random item user didn't interact with
                        neg_item_id = random.choice(list(self.items.keys()))
                        while neg_item_id in self.users[interaction.user_id].interactions:
                            neg_item_id = random.choice(list(self.items.keys()))

                        neg_item_idx = self.item_id_to_idx[neg_item_id]

                        user_ids.append(user_idx)
                        item_ids.append(neg_item_idx)
                        labels.append(0.0)

                # Convert to tensors
                user_ids = torch.tensor(user_ids, dtype=torch.long).to(self.device)
                item_ids = torch.tensor(item_ids, dtype=torch.long).to(self.device)
                labels = torch.tensor(labels, dtype=torch.float32).to(self.device)

                # Forward pass
                scores = self.model(user_ids, item_ids)

                # Compute loss
                loss = criterion(scores, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        print("✓ Training complete")

        # Cache item embeddings for serving
        self._cache_item_embeddings()

    def _cache_item_embeddings(self):
        """Cache item embeddings for fast serving"""
        self.model.eval()

        self.item_ids_list = list(self.items.keys())
        item_indices = [self.item_id_to_idx[iid] for iid in self.item_ids_list]

        with torch.no_grad():
            item_ids_tensor = torch.tensor(item_indices, dtype=torch.long).to(self.device)
            item_embs = self.model.encode_items(item_ids_tensor)
            self.item_embeddings = item_embs.cpu().numpy()

        print(f"✓ Cached {len(self.item_ids_list)} item embeddings")

    def recommend(
        self, user_id: str, top_k: int = 10, exclude_interacted: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations for user

        Args:
            user_id: User ID
            top_k: Number of recommendations
            exclude_interacted: Exclude items user already interacted with

        Returns:
            List of (item_id, score) tuples
        """
        if user_id not in self.users:
            return []

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()

        # Encode user
        user_idx = self.user_id_to_idx[user_id]
        user_idx_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)

        with torch.no_grad():
            user_emb = self.model.encode_users(user_idx_tensor)
            user_emb_np = user_emb.cpu().numpy()[0]

        # Compute scores for all items (dot product)
        scores = np.dot(self.item_embeddings, user_emb_np)

        # Get top-k
        top_indices = np.argsort(scores)[::-1]

        # Filter and collect recommendations
        recommendations = []
        user_interactions = set(self.users[user_id].interactions) if exclude_interacted else set()

        for idx in top_indices:
            item_id = self.item_ids_list[idx]

            # Skip if already interacted
            if exclude_interacted and item_id in user_interactions:
                continue

            recommendations.append((item_id, float(scores[idx])))

            if len(recommendations) >= top_k:
                break

        return recommendations


# Example: Movie recommendation system
def collaborative_filtering_example():
    """
    Collaborative filtering for movie recommendations

    Use case:
    - 1M users, 10K movies
    - 100M interactions (ratings, views)
    - Recommend movies user will like

    Scale: Netflix has 200M+ users, 10K+ titles
    """

    # Initialize engine
    engine = RecommendationEngine(embedding_dim=64)

    # Add users
    users = [User(f"user_{i}", features={"age": 20 + i % 40}) for i in range(100)]
    for user in users:
        engine.add_user(user)

    # Add movies
    movies = [
        Item(f"movie_{i}", features={"genre": ["action", "comedy", "drama"][i % 3]})
        for i in range(50)
    ]
    for movie in movies:
        engine.add_item(movie)

    # Generate synthetic interactions
    # Users tend to watch movies in same genre
    for user in users:
        # User prefers specific genre
        preferred_genre_idx = int(user.user_id.split("_")[1]) % 3

        # Watch 5-10 movies
        num_interactions = 5 + (int(user.user_id.split("_")[1]) % 6)

        for _ in range(num_interactions):
            # 70% chance of preferred genre
            if random.random() < 0.7:
                movie_idx = preferred_genre_idx + (random.randint(0, 15) * 3)
            else:
                movie_idx = random.randint(0, 49)

            movie_id = f"movie_{movie_idx}"

            interaction = Interaction(
                user_id=user.user_id,
                item_id=movie_id,
                interaction_type="watch",
                rating=4.0 + random.random(),
            )
            engine.add_interaction(interaction)

    print("\n=== Dataset Statistics ===")
    print(f"Users: {len(engine.users)}")
    print(f"Movies: {len(engine.items)}")
    print(f"Interactions: {len(engine.interactions)}")

    # Train model
    engine.train(num_epochs=5, batch_size=64, neg_samples=4)

    # Generate recommendations
    test_user = "user_0"
    print(f"\n=== Recommendations for {test_user} ===")
    print("User's watch history:")
    for item_id in engine.users[test_user].interactions[:5]:
        item = engine.items[item_id]
        print(f"  {item_id}: {item.features['genre']}")

    recommendations = engine.recommend(test_user, top_k=5)
    print("\nTop 5 recommendations:")
    for item_id, score in recommendations:
        item = engine.items[item_id]
        print(f"  {item_id}: {item.features['genre']} (score: {score:.3f})")


# Uncomment to run:
# collaborative_filtering_example()
