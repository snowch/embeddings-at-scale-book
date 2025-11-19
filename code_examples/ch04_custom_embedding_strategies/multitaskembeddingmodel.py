# Code from Chapter 04
# Book: Embeddings at Scale
import torch
import torch.nn as nn
import torch.nn.functional as F


# Placeholder transformer encoder
class TransformerEncoder(nn.Module):
    """Placeholder transformer encoder. Replace with actual model."""
    def __init__(self, dim=512, depth=6, heads=8):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.linear = nn.Linear(768, dim)  # Dummy linear layer

    def forward(self, input_ids, attention_mask):
        # Return dummy hidden states
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1] if len(input_ids.shape) > 1 else 1
        return torch.randn(batch_size, seq_len, self.dim)


class MultiTaskEmbeddingModel(nn.Module):
    """
    Single encoder with multiple task-specific heads
    """

    def __init__(self, embedding_dim=512, num_categories=1000, num_brands=5000):
        super().__init__()

        # Shared encoder (e.g., transformer)
        self.shared_encoder = TransformerEncoder(
            dim=embedding_dim,
            depth=6,
            heads=8
        )

        # Task-specific heads
        self.similarity_head = nn.Linear(embedding_dim, embedding_dim)  # For similarity search
        self.category_head = nn.Linear(embedding_dim, num_categories)   # Category classification
        self.brand_head = nn.Linear(embedding_dim, num_brands)          # Brand classification
        self.price_head = nn.Linear(embedding_dim, 1)                   # Price regression

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through shared encoder
        """
        # Shared representation
        hidden_state = self.shared_encoder(input_ids, attention_mask)
        pooled = hidden_state.mean(dim=1)  # Average pooling

        # Task-specific outputs
        outputs = {
            'embedding': self.similarity_head(pooled),
            'category_logits': self.category_head(pooled),
            'brand_logits': self.brand_head(pooled),
            'price_pred': self.price_head(pooled)
        }

        return outputs

    def compute_loss(self, outputs, targets, task_weights):
        """
        Weighted multi-task loss
        """
        losses = {}

        # Similarity loss (contrastive or triplet)
        if 'positive' in targets and 'negative' in targets:
            pos_sim = F.cosine_similarity(outputs['embedding'], targets['positive'])
            neg_sim = F.cosine_similarity(outputs['embedding'], targets['negative'])
            losses['similarity'] = torch.clamp(1.0 - pos_sim + neg_sim, min=0.0).mean()

        # Category classification loss
        if 'category' in targets:
            losses['category'] = F.cross_entropy(
                outputs['category_logits'],
                targets['category']
            )

        # Brand classification loss
        if 'brand' in targets:
            losses['brand'] = F.cross_entropy(
                outputs['brand_logits'],
                targets['brand']
            )

        # Price regression loss
        if 'price' in targets:
            losses['price'] = F.mse_loss(
                outputs['price_pred'].squeeze(),
                targets['price']
            )

        # Weighted combination
        total_loss = sum(
            task_weights.get(task, 1.0) * loss
            for task, loss in losses.items()
        )

        return total_loss, losses


# Training with multi-task learning
model = MultiTaskEmbeddingModel(embedding_dim=512)

# Task weights (tune based on importance)
task_weights = {
    'similarity': 1.0,   # Core task
    'category': 0.3,     # Help preserve category info
    'brand': 0.2,        # Help preserve brand info
    'price': 0.1         # Weak signal for price tier
}

# Placeholder data loader
class PlaceholderDataLoader:
    """Placeholder data loader. Replace with actual DataLoader."""
    def __iter__(self):
        return iter([])  # Empty iterator for demonstration

train_loader = PlaceholderDataLoader()

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for batch in train_loader:
    outputs = model(batch['input_ids'], batch['attention_mask'])

    loss, task_losses = model.compute_loss(
        outputs,
        targets=batch['targets'],
        task_weights=task_weights
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
