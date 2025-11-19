import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Code from Chapter 07
# Book: Embeddings at Scale

class TimeSeriesSelfSupervised(nn.Module):
    """
    Self-supervised learning for time-series data

    Pretext tasks:
    1. Future prediction: Predict next N steps
    2. Masked reconstruction: Predict masked time steps
    3. Contrastive temporal coding: Distinguish shuffled from real
    4. Transformation recognition: Identify applied transformation

    Use cases:
    - IoT sensor data: Learn patterns from millions of devices
    - Financial time-series: Capture market dynamics
    - User behavior logs: Model activity patterns
    - Healthcare monitoring: Learn from vital signs
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_layers=4,
        pretext_task='forecasting'
    ):
        """
        Args:
            input_dim: Dimension of time-series features
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            pretext_task: 'forecasting', 'masked', or 'contrastive'
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pretext_task = pretext_task

        # Encoder (Transformer or LSTM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Task-specific heads
        if pretext_task == 'forecasting':
            self.forecast_head = nn.Linear(hidden_dim, input_dim)
        elif pretext_task == 'masked':
            self.reconstruction_head = nn.Linear(hidden_dim, input_dim)
        elif pretext_task == 'contrastive':
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 128)
            )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)

        Returns:
            embeddings: (batch_size, seq_len, hidden_dim)
        """
        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Encode
        embeddings = self.encoder(x)

        return embeddings

    def forecasting_loss(self, x, forecast_steps=5):
        """
        Predict future time steps

        Args:
            x: (batch_size, seq_len, input_dim)
            forecast_steps: Number of steps to predict

        Returns:
            loss: Forecasting loss
        """
        # Use first (seq_len - forecast_steps) as context
        context = x[:, :-forecast_steps, :]
        target = x[:, -forecast_steps:, :]

        # Encode context
        embeddings = self.forward(context)

        # Predict future
        predictions = self.forecast_head(embeddings[:, -forecast_steps:, :])

        # MSE loss
        loss = F.mse_loss(predictions, target)

        return loss

    def masked_reconstruction_loss(self, x, mask_ratio=0.15):
        """
        Masked reconstruction pretext task

        Args:
            x: (batch_size, seq_len, input_dim)
            mask_ratio: Ratio of time steps to mask

        Returns:
            loss: Reconstruction loss
        """
        batch_size, seq_len, input_dim = x.shape

        # Create random mask
        mask = torch.rand(batch_size, seq_len) < mask_ratio
        mask = mask.unsqueeze(-1).expand_as(x)

        # Masked input
        x_masked = x.clone()
        x_masked[mask] = 0

        # Encode
        embeddings = self.forward(x_masked)

        # Reconstruct
        reconstructed = self.reconstruction_head(embeddings)

        # Loss only on masked positions
        loss = F.mse_loss(reconstructed[mask], x[mask])

        return loss

    def contrastive_temporal_loss(self, x):
        """
        Contrastive loss for time-series

        Strategy: Create positive pairs through augmentation,
        negative pairs from different time series
        """
        # Create two augmented views
        x1 = self.augment_time_series(x)
        x2 = self.augment_time_series(x)

        # Encode
        emb1 = self.forward(x1).mean(dim=1)  # Pool over time
        emb2 = self.forward(x2).mean(dim=1)

        # Project
        z1 = self.projection_head(emb1)
        z2 = self.projection_head(emb2)

        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Contrastive loss (NT-Xent)
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.mm(z, z.T) / 0.5

        labels = torch.arange(batch_size).to(z1.device)
        labels = torch.cat([labels + batch_size, labels])

        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
        similarity_matrix.masked_fill_(mask, float('-inf'))

        loss = F.cross_entropy(similarity_matrix, labels)

        return loss

    def augment_time_series(self, x):
        """
        Augment time-series data

        Augmentations:
        - Jittering: Add noise
        - Scaling: Multiply by constant
        - Time warping: Stretch/compress
        - Window slicing: Extract subsequence
        """
        # Jittering
        noise = torch.randn_like(x) * 0.05
        x_aug = x + noise

        # Scaling
        scale = torch.rand(x.shape[0], 1, 1).to(x.device) * 0.4 + 0.8
        x_aug = x_aug * scale

        return x_aug


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


# Example: IoT sensor self-supervised learning
def example_iot_sensor_ssl():
    """
    Example: Learn from unlabeled IoT sensor data
    """
    # Initialize model
    model = TimeSeriesSelfSupervised(
        input_dim=32,  # 32 sensor readings
        hidden_dim=256,
        num_layers=6,
        pretext_task='forecasting'
    ).cuda()

    # Dummy data (in production: load from time-series database)
    # Shape: (batch_size, seq_len, num_sensors)
    sensor_data = torch.randn(64, 100, 32).cuda()

    # Train with forecasting task
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(100):
        optimizer.zero_grad()
        loss = model.forecasting_loss(sensor_data, forecast_steps=10)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Extract embeddings for downstream tasks
    with torch.no_grad():
        embeddings = model.forward(sensor_data)
        # Use embeddings for anomaly detection, classification, etc.

    print(f"Embeddings shape: {embeddings.shape}")
