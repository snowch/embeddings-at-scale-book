# Code from Chapter 15
# Book: Embeddings at Scale

"""
Real-Time Personalization with Streaming Embeddings

Architecture:
1. Base user embedding: Learned from historical interactions
2. Session embedding: Computed from current session behavior
3. Context embedding: Time, device, location signals
4. Combined embedding: Fusion of base, session, and context

Techniques:
- Incremental embedding updates (online learning)
- Attention over recent interactions (recency weighting)
- Session-based RNN/Transformer (sequential modeling)
- Context-aware fusion (time-of-day, device, location)

Production:
- Stream processing: Kafka, Kinesis for event ingestion
- Online inference: Sub-100ms embedding computation
- Cache invalidation: Update user cache on interactions
- Fallback: Base embedding if session too short
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Deque
from dataclasses import dataclass
from collections import deque
import time

@dataclass
class SessionEvent:
    """
    User interaction event in current session

    Attributes:
        item_id: Item interacted with
        event_type: Type (view, click, add_to_cart, purchase)
        timestamp: When event occurred
        context: Additional context (device, location, etc.)
    """
    item_id: str
    event_type: str
    timestamp: float
    context: Dict[str, any] = None

class SessionEncoder(nn.Module):
    """
    Encode user session to embedding

    Architecture:
    - RNN/Transformer over session events
    - Attention mechanism (recent events matter more)
    - Output: Session embedding

    Training:
    - Predict next item from session history
    - Self-supervised on session logs
    """

    def __init__(
        self,
        item_embedding_dim: int = 128,
        session_embedding_dim: int = 128,
        hidden_dim: int = 256
    ):
        super().__init__()

        # RNN for sequential modeling
        self.rnn = nn.GRU(
            input_size=item_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, session_embedding_dim)

    def forward(
        self,
        item_embeddings: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode session to embedding

        Args:
            item_embeddings: Item embeddings in session (batch, max_len, item_dim)
            lengths: Actual sequence lengths (batch,)

        Returns:
            Session embeddings (batch, session_dim)
        """
        batch_size = item_embeddings.size(0)

        # RNN encoding
        rnn_out, _ = self.rnn(item_embeddings)  # (batch, max_len, hidden_dim)

        # Attention weights
        attn_weights = self.attention(rnn_out)  # (batch, max_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum
        session_emb = (rnn_out * attn_weights).sum(dim=1)  # (batch, hidden_dim)

        # Project to session embedding space
        session_emb = self.output_projection(session_emb)  # (batch, session_dim)

        # Normalize
        session_emb = F.normalize(session_emb, p=2, dim=1)

        return session_emb

class RealTimePersonalizer:
    """
    Real-time recommendation personalization

    Components:
    1. Base embeddings: User/item embeddings from batch training
    2. Session tracker: Maintains current session state
    3. Session encoder: Computes session embedding
    4. Embedding fusion: Combines base + session + context

    Features:
    - Sub-second latency (100ms p95)
    - Automatic cache invalidation
    - Recency weighting (recent interactions matter more)
    - Context awareness (time, device, location)
    """

    def __init__(
        self,
        base_user_embeddings: Dict[str, np.ndarray],
        base_item_embeddings: Dict[str, np.ndarray],
        session_window: int = 30  # minutes
    ):
        """
        Args:
            base_user_embeddings: Pre-computed user embeddings
            base_item_embeddings: Pre-computed item embeddings
            session_window: Session timeout in minutes
        """
        self.base_user_embeddings = base_user_embeddings
        self.base_item_embeddings = base_item_embeddings
        self.session_window = session_window * 60  # Convert to seconds

        # Session state: user_id -> deque of recent events
        self.user_sessions: Dict[str, Deque[SessionEvent]] = {}

        # Session encoder
        self.session_encoder = SessionEncoder(
            item_embedding_dim=128,
            session_embedding_dim=128
        )
        self.session_encoder.eval()

        print(f"Initialized Real-Time Personalizer")
        print(f"  Session window: {session_window} minutes")

    def track_event(self, user_id: str, event: SessionEvent):
        """
        Track user event in current session

        Args:
            user_id: User ID
            event: Session event
        """
        # Initialize session if new user
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = deque(maxlen=50)  # Keep last 50 events

        # Add event
        self.user_sessions[user_id].append(event)

        # Clean old events (outside session window)
        current_time = time.time()
        while (self.user_sessions[user_id] and
               current_time - self.user_sessions[user_id][0].timestamp > self.session_window):
            self.user_sessions[user_id].popleft()

    def get_personalized_embedding(
        self,
        user_id: str,
        context: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Get real-time personalized user embedding

        Combines:
        1. Base user embedding (long-term preferences)
        2. Session embedding (current intent)
        3. Context embedding (time, device, location)

        Args:
            user_id: User ID
            context: Current context (time, device, etc.)

        Returns:
            Personalized user embedding
        """
        # Get base embedding
        base_emb = self.base_user_embeddings.get(user_id)
        if base_emb is None:
            # New user: Use average embedding
            base_emb = np.mean(list(self.base_user_embeddings.values()), axis=0)

        # Get session embedding
        session_emb = self._compute_session_embedding(user_id)

        # Compute blending weights
        num_session_events = len(self.user_sessions.get(user_id, []))

        # More session events = higher session weight
        session_weight = min(num_session_events / 10.0, 0.5)  # Max 50% from session
        base_weight = 1.0 - session_weight

        # Combine embeddings
        if session_emb is not None:
            combined_emb = base_weight * base_emb + session_weight * session_emb
        else:
            combined_emb = base_emb

        # Normalize
        combined_emb = combined_emb / np.linalg.norm(combined_emb)

        return combined_emb

    def _compute_session_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """
        Compute embedding from current session

        Args:
            user_id: User ID

        Returns:
            Session embedding or None if no session
        """
        if user_id not in self.user_sessions or len(self.user_sessions[user_id]) == 0:
            return None

        # Get item embeddings for session events
        session_events = list(self.user_sessions[user_id])
        item_embs = []

        for event in session_events:
            if event.item_id in self.base_item_embeddings:
                item_embs.append(self.base_item_embeddings[event.item_id])

        if not item_embs:
            return None

        # Simple averaging (in production: use session encoder)
        session_emb = np.mean(item_embs, axis=0)

        # Apply recency weighting (recent items matter more)
        current_time = time.time()
        weights = []
        for event in session_events:
            # Exponential decay: more recent = higher weight
            time_diff = current_time - event.timestamp
            weight = np.exp(-time_diff / 600)  # 10-minute half-life
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        # Weighted average
        session_emb = np.average(item_embs, axis=0, weights=weights[:len(item_embs)])

        return session_emb

    def recommend_realtime(
        self,
        user_id: str,
        top_k: int = 10,
        context: Optional[Dict] = None
    ) -> List[Tuple[str, float]]:
        """
        Generate real-time personalized recommendations

        Args:
            user_id: User ID
            top_k: Number of recommendations
            context: Current context

        Returns:
            List of (item_id, score) tuples
        """
        # Get personalized embedding
        user_emb = self.get_personalized_embedding(user_id, context)

        # Compute scores for all items
        scores = {}
        for item_id, item_emb in self.base_item_embeddings.items():
            score = np.dot(user_emb, item_emb)
            scores[item_id] = score

        # Sort and return top-k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]

# Example: Real-time e-commerce personalization
def realtime_personalization_example():
    """
    Real-time personalization for e-commerce

    Scenario:
    - User browsing products
    - Each view/click updates recommendations
    - Adapt to session context in real-time

    Scale: 10K updates/second per user
    """

    # Initialize with base embeddings
    base_user_embs = {
        'user_123': np.random.randn(128).astype(np.float32)
    }
    base_user_embs['user_123'] /= np.linalg.norm(base_user_embs['user_123'])

    base_item_embs = {
        f'item_{i}': np.random.randn(128).astype(np.float32)
        for i in range(20)
    }
    for item_id in base_item_embs:
        base_item_embs[item_id] /= np.linalg.norm(base_item_embs[item_id])

    # Initialize personalizer
    personalizer = RealTimePersonalizer(
        base_user_embeddings=base_user_embs,
        base_item_embeddings=base_item_embs,
        session_window=30
    )

    user_id = 'user_123'

    # Initial recommendations (based on base embedding)
    print("=== Initial Recommendations ===")
    recs = personalizer.recommend_realtime(user_id, top_k=5)
    for item_id, score in recs:
        print(f"{item_id}: {score:.3f}")

    # Simulate user session
    print("\n=== User Session ===")
    session_items = ['item_5', 'item_7', 'item_12']

    for i, item_id in enumerate(session_items):
        print(f"\nUser views {item_id}")

        # Track event
        event = SessionEvent(
            item_id=item_id,
            event_type='view',
            timestamp=time.time()
        )
        personalizer.track_event(user_id, event)

        # Generate updated recommendations
        print("Updated recommendations:")
        recs = personalizer.recommend_realtime(user_id, top_k=5)
        for rec_item_id, score in recs[:3]:
            print(f"  {rec_item_id}: {score:.3f}")

# Uncomment to run:
# realtime_personalization_example()
