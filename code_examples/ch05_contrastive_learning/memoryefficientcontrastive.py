import torch.nn as nn

# Code from Chapter 05
# Book: Embeddings at Scale
from torch.utils.checkpoint import checkpoint


class MemoryEfficientContrastive(nn.Module):
    """
    Use gradient checkpointing to reduce memory consumption

    Trades computation for memory:
    - Forward pass: only store inputs and outputs of checkpointed layers
    - Backward pass: recompute forward pass for checkpointed layers

    Enables:
    - Larger models
    - Larger batch sizes
    - Deeper architectures

    Cost:
    - ~20-30% slower (recomputation overhead)
    """

    def __init__(self, base_model, projection_dim=128):
        super().__init__()

        self.encoder = base_model

        # Enable gradient checkpointing for encoder
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        self.projection = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward with gradient checkpointing
        """
        # Encoder with checkpointing (handled internally if enabled)
        outputs = self.encoder(input_ids, attention_mask)
        embeddings = outputs.last_hidden_state[:, 0]

        # Projection with manual checkpointing
        projected = checkpoint(self.projection, embeddings)

        return projected
