# Code from Chapter 06
# Book: Embeddings at Scale

class EnterpriseOptimizedSiameseNetwork(nn.Module):
    """
    Production-optimized Siamese network with enterprise features

    Features:
    - Mixed precision training support
    - Gradient checkpointing for memory efficiency
    - Batch normalization for stability
    - Optional attention mechanisms
    - Multi-GPU training support
    """

    def __init__(
        self,
        base_model,
        embedding_dim=512,
        use_attention=True,
        use_gradient_checkpointing=False
    ):
        super().__init__()

        self.base_model = base_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Projection head for better embeddings
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Optional attention for focusing on important features
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        else:
            self.attention = None

    def forward(self, x1, x2):
        """Forward pass with optional gradient checkpointing"""

        if self.use_gradient_checkpointing and self.training:
            # Save memory during training by recomputing activations
            embedding1 = torch.utils.checkpoint.checkpoint(
                self._encode, x1
            )
            embedding2 = torch.utils.checkpoint.checkpoint(
                self._encode, x2
            )
        else:
            embedding1 = self._encode(x1)
            embedding2 = self._encode(x2)

        return embedding1, embedding2

    def _encode(self, x):
        """Encode input to embedding"""
        # Base encoding
        features = self.base_model(x)

        # Apply attention if configured
        if self.attention is not None:
            # Reshape for attention (batch, seq_len=1, dim)
            features_reshaped = features.unsqueeze(1)
            attended, _ = self.attention(
                features_reshaped,
                features_reshaped,
                features_reshaped
            )
            features = attended.squeeze(1)

        # Project to embedding space
        embedding = self.projection(features)

        # L2 normalize
        return F.normalize(embedding, p=2, dim=1)


# Training loop with mixed precision
def train_siamese_enterprise(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    use_amp=True
):
    """
    Enterprise training loop with automatic mixed precision

    Args:
        model: Siamese network
        train_loader: DataLoader yielding (x1, x2, labels)
        optimizer: PyTorch optimizer
        criterion: Loss function (ContrastiveLoss or TripletLoss)
        device: 'cuda' or 'cpu'
        use_amp: Use automatic mixed precision for faster training
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    total_loss = 0
    total_accuracy = 0

    for batch_idx, (x1, x2, labels) in enumerate(train_loader):
        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                embedding1, embedding2 = model(x1, x2)
                loss, metrics = criterion(embedding1, embedding2, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            embedding1, embedding2 = model(x1, x2)
            loss, metrics = criterion(embedding1, embedding2, labels)

            loss.backward()
            optimizer.step()

        total_loss += metrics['loss']
        total_accuracy += metrics['accuracy']

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)

    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy
    }
