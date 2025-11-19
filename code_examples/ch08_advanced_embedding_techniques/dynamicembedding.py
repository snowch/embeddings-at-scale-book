# Code from Chapter 08
# Book: Embeddings at Scale

import torch
import torch.nn as nn
from datetime import datetime, timedelta

class DynamicEmbedding(nn.Module):
    """
    Time-aware embeddings that evolve with data

    Three modes:
    1. Discrete: Separate embeddings per time window (daily, weekly, monthly)
    2. Continuous: Embedding = f(base_embedding, time)
    3. Streaming: Incrementally update embeddings from data stream

    Applications:
    - User preferences (evolve as users interact)
    - Document relevance (decay over time, spike on events)
    - Product popularity (seasonal cycles, trends)
    - Stock market sentiment (rapid intraday changes)
    - Medical patient state (disease progression)
    """

    def __init__(
        self,
        num_items,
        embedding_dim=256,
        mode='continuous',
        num_time_slices=None,
        decay_rate=0.01
    ):
        """
        Args:
            num_items: Number of entities to embed
            embedding_dim: Dimension of embeddings
            mode: 'discrete', 'continuous', or 'streaming'
            num_time_slices: Number of discrete time windows (for discrete mode)
            decay_rate: Rate of temporal decay (for continuous mode)
        """
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.decay_rate = decay_rate

        if mode == 'discrete':
            # Separate embedding matrix for each time slice
            assert num_time_slices is not None
            self.num_time_slices = num_time_slices
            self.embeddings = nn.Parameter(
                torch.randn(num_time_slices, num_items, embedding_dim) * 0.01
            )

        elif mode == 'continuous':
            # Base embedding + temporal transformation
            self.base_embeddings = nn.Parameter(
                torch.randn(num_items, embedding_dim) * 0.01
            )

            # Temporal transformation network
            self.temporal_network = nn.Sequential(
                nn.Linear(1, 64),  # Time as input
                nn.ReLU(),
                nn.Linear(64, embedding_dim),
                nn.Tanh()  # Bounded transformation
            )

        elif mode == 'streaming':
            # Incrementally updated embeddings
            self.embeddings = nn.Parameter(
                torch.randn(num_items, embedding_dim) * 0.01
            )
            # Track last update time for each item
            self.register_buffer(
                'last_update',
                torch.zeros(num_items)
            )
            # Exponential moving average momentum
            self.ema_momentum = 0.9

    def forward(self, indices, timestamps=None):
        """
        Get time-aware embeddings

        Args:
            indices: Item indices (batch_size,)
            timestamps: Time information (batch_size,) or (batch_size, 1)
                       Format depends on mode:
                       - discrete: time slice index (0 to num_time_slices-1)
                       - continuous: normalized time value (0.0 to 1.0)
                       - streaming: actual timestamps (for decay calculation)

        Returns:
            embeddings: Time-aware embeddings (batch_size, embedding_dim)
        """
        if self.mode == 'discrete':
            # Index into specific time slice
            time_slice = timestamps.long()
            batch_embeddings = self.embeddings[time_slice, indices]
            return batch_embeddings

        elif self.mode == 'continuous':
            # Base embedding + temporal transformation
            base_emb = self.base_embeddings[indices]

            # Apply temporal transformation
            if timestamps is None:
                timestamps = torch.zeros(len(indices), 1)
            if len(timestamps.shape) == 1:
                timestamps = timestamps.unsqueeze(1)

            temporal_shift = self.temporal_network(timestamps.float())

            # Combine base and temporal components
            dynamic_emb = base_emb + temporal_shift

            return dynamic_emb

        elif self.mode == 'streaming':
            # Get current embeddings
            current_emb = self.embeddings[indices]

            # Apply temporal decay if timestamps provided
            if timestamps is not None:
                last_update_time = self.last_update[indices]
                time_delta = timestamps - last_update_time

                # Exponential decay factor
                decay = torch.exp(-self.decay_rate * time_delta).unsqueeze(1)
                current_emb = current_emb * decay

            return current_emb

    def update_streaming(self, indices, new_observations, timestamps):
        """
        Update embeddings based on new observations (streaming mode only)

        Uses exponential moving average to incorporate new information
        while preserving historical signal

        Args:
            indices: Items to update
            new_observations: New embedding values from recent data
            timestamps: Current time for decay calculation
        """
        assert self.mode == 'streaming'

        with torch.no_grad():
            # Compute time-based decay
            last_update_time = self.last_update[indices]
            time_delta = timestamps - last_update_time
            decay = torch.exp(-self.decay_rate * time_delta).unsqueeze(1)

            # Exponential moving average update
            old_emb = self.embeddings[indices] * decay
            new_emb = (
                self.ema_momentum * old_emb +
                (1 - self.ema_momentum) * new_observations
            )

            self.embeddings[indices] = new_emb
            self.last_update[indices] = timestamps

class TemporalUserEmbedding:
    """
    User embeddings that evolve based on interaction history

    Critical for:
    - E-commerce: Capture seasonal preferences, life events
    - Content platforms: Track evolving interests
    - Finance: Monitor changing risk profiles
    - Healthcare: Model disease progression and treatment response

    Handles:
    - Short-term interests (what user engaged with today)
    - Long-term preferences (persistent tastes)
    - Periodic patterns (weekly/monthly cycles)
    - Trend adaptation (gradual interest shifts)
    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=128,
        short_term_weight=0.3,
        device='cpu'
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.short_term_weight = short_term_weight
        self.device = device

        # Long-term user preferences (slowly evolving)
        self.long_term = nn.Embedding(num_users, embedding_dim).to(device)

        # Short-term user state (rapidly changing)
        self.short_term = nn.Embedding(num_users, embedding_dim).to(device)

        # Item embeddings (what users interact with)
        self.items = nn.Embedding(num_items, embedding_dim).to(device)

        # LSTM for modeling temporal sequences
        self.lstm = nn.LSTM(
            embedding_dim,
            embedding_dim,
            batch_first=True
        ).to(device)

    def get_user_embedding(self, user_id, current_time=None):
        """
        Get current user embedding combining long and short-term components

        Returns weighted combination that balances:
        - Stable long-term preferences
        - Recent short-term interests
        """
        long_term_emb = self.long_term(user_id)
        short_term_emb = self.short_term(user_id)

        # Weighted combination
        combined = (
            (1 - self.short_term_weight) * long_term_emb +
            self.short_term_weight * short_term_emb
        )

        return combined

    def update_from_interaction(
        self,
        user_id,
        item_id,
        interaction_type='view',
        timestamp=None
    ):
        """
        Update user embedding based on new interaction

        Args:
            user_id: User who interacted
            item_id: Item they interacted with
            interaction_type: 'view', 'click', 'purchase', etc.
            timestamp: When interaction occurred
        """
        # Get item embedding
        item_emb = self.items(item_id)

        # Update short-term state (immediate impact)
        with torch.no_grad():
            current_short_term = self.short_term(user_id)

            # Weight based on interaction type
            interaction_weights = {
                'view': 0.1,
                'click': 0.3,
                'add_to_cart': 0.5,
                'purchase': 1.0
            }
            weight = interaction_weights.get(interaction_type, 0.1)

            # Update with exponential moving average
            updated = 0.9 * current_short_term + weight * item_emb
            self.short_term.weight[user_id] = updated

        # Long-term updated more slowly (during training)
        # This happens in batch training, not per-interaction

    def predict_sequence(self, user_id, item_sequence, timestamps):
        """
        Predict next items based on interaction sequence

        Uses LSTM to model temporal dependencies in user behavior

        Args:
            user_id: User ID
            item_sequence: Sequence of item IDs (seq_len,)
            timestamps: Time of each interaction (seq_len,)

        Returns:
            next_item_logits: Scores for all possible next items
        """
        # Embed item sequence
        item_embs = self.items(item_sequence).unsqueeze(0)  # (1, seq_len, dim)

        # Run through LSTM
        lstm_out, (hidden, cell) = self.lstm(item_embs)

        # Final hidden state represents current user state
        current_state = hidden[-1]  # (1, dim)

        # Combine with long-term preferences
        long_term = self.long_term(user_id).unsqueeze(0)
        combined_state = 0.7 * current_state + 0.3 * long_term

        # Compute scores for all items
        all_item_embs = self.items.weight  # (num_items, dim)
        logits = torch.matmul(combined_state, all_item_embs.T)  # (1, num_items)

        return logits.squeeze(0)

# Example: E-commerce user with evolving preferences
def temporal_user_example():
    """
    Model user whose preferences evolve over 6 months

    Scenario:
    - Jan-Feb: Browse fitness equipment (New Year's resolution)
    - Mar-Apr: Shift to outdoor gear (spring)
    - May-Jun: Focus on camping equipment (summer vacation planning)

    Dynamic embeddings capture these shifts while maintaining
    long-term preferences (e.g., preference for eco-friendly products)
    """
    model = TemporalUserEmbedding(
        num_users=10000,
        num_items=50000,
        embedding_dim=128
    )

    user_id = torch.tensor([42])

    # Simulate 6 months of interactions
    print("Temporal user embedding evolution:")

    # January: Fitness equipment
    fitness_items = torch.randint(0, 100, (20,))  # Items 0-99 are fitness
    for item in fitness_items[:10]:
        model.update_from_interaction(user_id, torch.tensor([item]), 'view')

    jan_emb = model.get_user_embedding(user_id)
    print(f"January embedding norm: {torch.norm(jan_emb).item():.3f}")

    # March: Outdoor gear
    outdoor_items = torch.randint(100, 200, (20,))  # Items 100-199 outdoor
    for item in outdoor_items[:10]:
        model.update_from_interaction(user_id, torch.tensor([item]), 'view')

    mar_emb = model.get_user_embedding(user_id)
    print(f"March embedding norm: {torch.norm(mar_emb).item():.3f}")
    print(f"Embedding shift (Jan→Mar): {torch.norm(jan_emb - mar_emb).item():.3f}")

    # May: Camping equipment
    camping_items = torch.randint(200, 300, (20,))  # Items 200-299 camping
    for item in camping_items[:10]:
        model.update_from_interaction(user_id, torch.tensor([item]), 'purchase')

    may_emb = model.get_user_embedding(user_id)
    print(f"May embedding norm: {torch.norm(may_emb).item():.3f}")
    print(f"Embedding shift (Mar→May): {torch.norm(mar_emb - may_emb).item():.3f}")

# Uncomment to run:
# temporal_user_example()
