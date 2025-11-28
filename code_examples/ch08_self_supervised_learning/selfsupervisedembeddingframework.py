# Code from Chapter 07
# Book: Embeddings at Scale

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfSupervisedEmbeddingFramework:
    """
    Framework for self-supervised learning on enterprise data

    Supports multiple pretext tasks:
    - Masked prediction (for text, tabular data)
    - Contrastive learning (for all data types)
    - Reconstruction (for images, time-series)

    Use cases:
    - Learn from millions of unlabeled documents
    - Train on industrial imagery without manual labeling
    - Build embeddings from sensor data streams
    - Leverage historical transaction logs
    """

    def __init__(
        self, encoder_model, pretext_task="masked", embedding_dim=768, mask_probability=0.15
    ):
        """
        Args:
            encoder_model: Neural network encoder (BERT, ResNet, custom)
            pretext_task: 'masked', 'contrastive', or 'reconstruction'
            embedding_dim: Dimension of learned embeddings
            mask_probability: Probability of masking for masked prediction
        """
        self.encoder = encoder_model
        self.pretext_task = pretext_task
        self.embedding_dim = embedding_dim
        self.mask_probability = mask_probability

        # Prediction head depends on pretext task
        if pretext_task == "masked":
            # For masked prediction: predict original tokens
            self.prediction_head = nn.Linear(embedding_dim, embedding_dim)
        elif pretext_task == "contrastive":
            # For contrastive learning: projection head
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, 128)
            )
        elif pretext_task == "reconstruction":
            # For reconstruction: decoder network
            self.decoder = self._build_decoder(embedding_dim)

    def _build_decoder(self, embedding_dim):
        """Build decoder for reconstruction tasks"""
        return nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

    def create_pretext_task(self, batch):
        """
        Create pretext task from unlabeled batch

        Args:
            batch: Unlabeled data (batch_size, seq_len, features)

        Returns:
            inputs: Modified inputs for pretext task
            targets: Targets for pretext task
            mask: Positions to predict (for masked tasks)
        """
        if self.pretext_task == "masked":
            return self._create_masked_task(batch)
        elif self.pretext_task == "contrastive":
            return self._create_contrastive_task(batch)
        elif self.pretext_task == "reconstruction":
            return self._create_reconstruction_task(batch)

    def _create_masked_task(self, batch):
        """
        Create masked prediction task

        Randomly mask tokens and predict them from context
        """
        batch_size, seq_len, features = batch.shape

        # Create mask (True = masked position)
        mask = torch.rand(batch_size, seq_len) < self.mask_probability

        # Clone batch for inputs
        inputs = batch.clone()

        # Replace masked positions with zeros or special token
        inputs[mask] = 0

        # Targets are original values at masked positions
        targets = batch.clone()

        return inputs, targets, mask

    def _create_contrastive_task(self, batch):
        """
        Create contrastive task with data augmentations

        Generate two augmented views of each sample
        """
        # Apply augmentations (specific to data type)
        view1 = self._augment(batch)
        view2 = self._augment(batch)

        return (view1, view2), None, None

    def _create_reconstruction_task(self, batch):
        """
        Create reconstruction task

        Add noise and predict clean version
        """
        # Add noise
        noise = torch.randn_like(batch) * 0.1
        noisy_batch = batch + noise

        return noisy_batch, batch, None

    def _augment(self, batch):
        """
        Apply data augmentation (override for specific data types)

        For text: dropout, span deletion, synonym replacement
        For images: cropping, color jitter, blur
        For time-series: masking, jittering, scaling
        """
        # Simple example: add small noise
        noise = torch.randn_like(batch) * 0.05
        return batch + noise

    def forward(self, inputs):
        """
        Forward pass through encoder

        Args:
            inputs: Batch of inputs

        Returns:
            embeddings: Learned embeddings
        """
        return self.encoder(inputs)

    def compute_loss(self, inputs, targets, mask=None):
        """
        Compute loss for pretext task

        Args:
            inputs: Input batch
            targets: Target batch (may be None for contrastive)
            mask: Mask for positions to predict (may be None)

        Returns:
            loss: Scalar loss
            metrics: Dict of training metrics
        """
        if self.pretext_task == "masked":
            return self._compute_masked_loss(inputs, targets, mask)
        elif self.pretext_task == "contrastive":
            return self._compute_contrastive_loss(inputs)
        elif self.pretext_task == "reconstruction":
            return self._compute_reconstruction_loss(inputs, targets)

    def _compute_masked_loss(self, inputs, targets, mask):
        """Compute loss for masked prediction"""
        # Encode inputs
        embeddings = self.encoder(inputs)

        # Predict masked tokens
        predictions = self.prediction_head(embeddings)

        # Compute loss only at masked positions
        loss = F.mse_loss(predictions[mask], targets[mask])

        with torch.no_grad():
            # Compute accuracy
            accuracy = ((predictions[mask] - targets[mask]).abs() < 0.1).float().mean()

        return loss, {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "masked_positions": mask.sum().item(),
        }

    def _compute_contrastive_loss(self, views):
        """Compute contrastive loss (NT-Xent)"""
        view1, view2 = views

        # Encode both views
        embeddings1 = self.encoder(view1)
        embeddings2 = self.encoder(view2)

        # Project to contrastive space
        z1 = self.projection_head(embeddings1)
        z2 = self.projection_head(embeddings2)

        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute similarity matrix
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.mm(z, z.T)

        # Temperature scaling
        temperature = 0.5
        similarity_matrix = similarity_matrix / temperature

        # Create labels: positive pairs are (i, i+batch_size)
        labels = torch.arange(batch_size).to(z1.device)
        labels = torch.cat([labels + batch_size, labels])

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
        similarity_matrix.masked_fill_(mask, float("-inf"))

        # Cross entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)

        with torch.no_grad():
            # Top-1 accuracy
            predictions = similarity_matrix.argmax(dim=1)
            accuracy = (predictions == labels).float().mean()

        return loss, {"loss": loss.item(), "accuracy": accuracy.item()}

    def _compute_reconstruction_loss(self, noisy_inputs, clean_targets):
        """Compute reconstruction loss"""
        # Encode noisy inputs
        embeddings = self.encoder(noisy_inputs)

        # Decode to reconstruct
        reconstructions = self.decoder(embeddings)

        # MSE loss
        loss = F.mse_loss(reconstructions, clean_targets)

        with torch.no_grad():
            # PSNR (Peak Signal-to-Noise Ratio)
            mse = ((reconstructions - clean_targets) ** 2).mean()
            psnr = 10 * torch.log10(1.0 / mse)

        return loss, {"loss": loss.item(), "psnr": psnr.item()}


def train_self_supervised(model, dataloader, optimizer, device, num_epochs=10):
    """
    Train self-supervised model

    Args:
        model: SelfSupervisedEmbeddingFramework
        dataloader: DataLoader with unlabeled data
        optimizer: PyTorch optimizer
        device: 'cuda' or 'cpu'
        num_epochs: Number of training epochs
    """
    model.encoder.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            batch = batch.to(device)

            # Create pretext task
            inputs, targets, mask = model.create_pretext_task(batch)

            # Forward pass
            optimizer.zero_grad()
            loss, metrics = model.compute_loss(inputs, targets, mask)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += metrics["loss"]
            if "accuracy" in metrics:
                total_accuracy += metrics["accuracy"]

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {metrics['loss']:.4f}")

        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)

        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
