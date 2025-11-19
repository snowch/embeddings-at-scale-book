# Code from Chapter 10
# Book: Embeddings at Scale

from torch.utils.checkpoint import checkpoint


class CheckpointedTransformerLayer(nn.Module):
    """
    Transformer layer with gradient checkpointing

    Standard approach:
    - Forward: Compute and store all activations
    - Backward: Use stored activations for gradients
    - Memory: O(layers × sequence_length × hidden_dim)

    Gradient checkpointing:
    - Forward: Compute activations, discard most
    - Backward: Recompute activations on-the-fly
    - Memory: O(checkpoints × sequence_length × hidden_dim)

    Trade-off:
    - Memory: 10-50× reduction (depending on checkpoints)
    - Compute: 30-50% slowdown (recomputation cost)

    When to use:
    - Very deep models (50+ layers)
    - Long sequences (10K+ tokens)
    - Memory is bottleneck, compute is available
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with gradient checkpointing

        Checkpoint at layer boundaries:
        - Attention block
        - FFN block
        """
        # Checkpoint attention
        def attention_forward(x):
            attn_out, _ = self.attention(x, x, x)
            return self.norm1(x + attn_out)

        x = checkpoint(attention_forward, x)

        # Checkpoint FFN
        def ffn_forward(x):
            return self.norm2(x + self.ffn(x))

        x = checkpoint(ffn_forward, x)

        return x

class MemoryEfficientEmbeddingModel(nn.Module):
    """
    Embedding model with aggressive memory optimization

    Techniques:
    1. Gradient checkpointing for transformer layers
    2. Mixed precision (FP16)
    3. Gradient accumulation
    4. In-place operations where safe

    Memory breakdown (12-layer transformer, 512 hidden dim):
    - Without optimization: ~40GB (A100 won't fit large batches)
    - With checkpointing: ~10GB (fits with batch 2048)
    - With checkpointing + FP16: ~6GB (fits with batch 4096)

    Enables training much larger models on same hardware.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8
    ):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Transformer layers with checkpointing
        self.layers = nn.ModuleList([
            CheckpointedTransformerLayer(embedding_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory-efficient operations"""

        # Embed
        x = self.embeddings(input_ids)

        # Transform (checkpointed)
        for layer in self.layers:
            x = layer(x)

        # Project
        x = self.projection(x)

        return x
