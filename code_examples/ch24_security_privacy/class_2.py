# Code from Chapter 24
# Book: Embeddings at Scale

"""
Differential Privacy for Embedding Training

Architecture:
1. Gradient clipping: Bound per-example gradient norm
2. Noise injection: Add Gaussian noise to gradients
3. Privacy accounting: Track cumulative privacy loss
4. Batch sampling: Use Poisson sampling for stronger privacy
5. Model validation: Verify privacy-utility trade-off

Techniques:
- DP-SGD: Stochastic gradient descent with differential privacy
- PATE: Private aggregation of teacher ensembles
- DP-FTRL: Follow-the-regularized-leader with DP
- Private federated learning: Distributed DP training
- Post-processing: Additional privacy via output perturbation

Privacy guarantees:
- (ε,δ)-differential privacy for entire training process
- Privacy amplification via sampling
- Composition theorems for multiple releases
- Rényi differential privacy for tighter accounting

Performance targets:
- Privacy: ε ≤ 1.0, δ ≤ 1e-5 for typical applications
- Utility: >85% of non-private model performance
- Training time: 2-5× longer than standard training
- Memory: 1.5-2× due to per-example gradients
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DPTrainingConfig:
    """
    Configuration for differentially private training

    Attributes:
        target_epsilon: Target privacy parameter
        target_delta: Target failure probability
        max_grad_norm: Gradient clipping threshold
        noise_multiplier: Noise scale relative to clipping
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        sampling_probability: Probability of including example
        accounting_mode: Privacy accounting method (rdp, gdp, glw)
    """
    target_epsilon: float = 1.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0  # σ, computed from ε,δ
    batch_size: int = 256
    num_epochs: int = 10
    learning_rate: float = 0.001
    sampling_probability: Optional[float] = None  # Computed from batch_size
    accounting_mode: str = "rdp"  # "rdp", "gdp", "glw"

@dataclass
class PrivacyAccountant:
    """
    Track cumulative privacy loss during training

    Attributes:
        epsilon_spent: Cumulative ε used
        delta_spent: Cumulative δ used
        steps: Number of training steps
        composition_method: How to compose privacy guarantees
        history: History of privacy loss per step
    """
    epsilon_spent: float = 0.0
    delta_spent: float = 0.0
    steps: int = 0
    composition_method: str = "rdp"
    history: List[Tuple[float, float]] = field(default_factory=list)

class DPEmbeddingModel(nn.Module):
    """
    Embedding model with differential privacy support

    Standard embedding model with hooks for:
    - Per-example gradient computation
    - Gradient clipping
    - Noise injection
    - Privacy accounting
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Projection layers
        layers = []
        in_dim = embedding_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, embedding_dim))
        self.projection = nn.Sequential(*layers)

        # Output normalization
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: [batch_size, seq_len] token IDs

        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        # Embed tokens
        embedded = self.embedding(input_ids)  # [batch, seq_len, emb_dim]

        # Pool over sequence (mean pooling)
        pooled = embedded.mean(dim=1)  # [batch, emb_dim]

        # Project
        projected = self.projection(pooled)

        # Normalize
        output = self.output_norm(projected)

        return F.normalize(output, p=2, dim=1)

class DPSGDOptimizer:
    """
    Differentially Private SGD optimizer

    Implements DP-SGD algorithm:
    1. Compute per-example gradients
    2. Clip gradients to max_grad_norm
    3. Add Gaussian noise
    4. Average and apply update
    """

    def __init__(
        self,
        model: nn.Module,
        config: DPTrainingConfig
    ):
        self.model = model
        self.config = config

        # Standard optimizer for parameter updates
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )

        # Privacy accountant
        self.accountant = PrivacyAccountant(
            composition_method=config.accounting_mode
        )

        # Compute noise multiplier if not provided
        if config.noise_multiplier == 1.0:
            self.config.noise_multiplier = self._compute_noise_multiplier()

        print("DP-SGD initialized:")
        print(f"  Target privacy: ε={config.target_epsilon}, δ={config.target_delta}")
        print(f"  Noise multiplier: σ={self.config.noise_multiplier:.2f}")
        print(f"  Gradient clipping: {config.max_grad_norm}")

    def _compute_noise_multiplier(self) -> float:
        """
        Compute noise multiplier from target ε,δ

        Uses Gaussian mechanism calibration:
        σ = sqrt(2 * ln(1.25/δ)) * sensitivity / ε

        For DP-SGD with gradient clipping:
        sensitivity = max_grad_norm

        Returns:
            Noise multiplier σ
        """
        # Simplified computation
        # In production, use more accurate calibration (e.g., from Opacus)

        sensitivity = self.config.max_grad_norm
        delta = self.config.target_delta
        epsilon = self.config.target_epsilon

        sigma = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon

        return sigma

    def compute_per_example_gradients(
        self,
        loss: torch.Tensor,
        inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradients for each example in batch

        Standard PyTorch computes mean gradient across batch.
        DP-SGD requires per-example gradients for clipping.

        Args:
            loss: Loss for batch
            inputs: Input batch

        Returns:
            Dictionary of per-example gradients for each parameter
        """
        # Enable per-example gradient computation
        # In production, use torch.func.grad or Opacus GradSampleModule

        per_example_grads = {}

        # Compute gradients for each parameter
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Get per-example gradient
                # This is simplified; production uses proper per-sample gradient
                grad = torch.autograd.grad(
                    loss,
                    param,
                    retain_graph=True,
                    create_graph=False
                )[0]

                per_example_grads[name] = grad

        return per_example_grads

    def clip_gradients(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Clip per-example gradients to max_grad_norm

        Clipping bounds the influence of any single example:
        g_clipped = g * min(1, max_grad_norm / ||g||)

        Args:
            gradients: Per-example gradients

        Returns:
            Clipped gradients
        """
        clipped_grads = {}

        for name, grad in gradients.items():
            # Compute gradient norm
            grad_norm = torch.norm(grad)

            # Clip if needed
            if grad_norm > self.config.max_grad_norm:
                clipped_grads[name] = grad * (
                    self.config.max_grad_norm / grad_norm
                )
            else:
                clipped_grads[name] = grad

        return clipped_grads

    def add_noise(
        self,
        gradients: Dict[str, torch.Tensor],
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """
        Add Gaussian noise to gradients for privacy

        Noise scale: σ = noise_multiplier * max_grad_norm

        Args:
            gradients: Clipped gradients
            batch_size: Batch size for normalization

        Returns:
            Noisy gradients
        """
        noisy_grads = {}

        noise_scale = self.config.noise_multiplier * self.config.max_grad_norm

        for name, grad in gradients.items():
            # Sample Gaussian noise with same shape as gradient
            noise = torch.randn_like(grad) * noise_scale

            # Add noise and normalize by batch size
            noisy_grads[name] = (grad + noise) / batch_size

        return noisy_grads

    def step(
        self,
        loss: torch.Tensor,
        inputs: torch.Tensor
    ):
        """
        Perform one DP-SGD optimization step

        Steps:
        1. Compute per-example gradients
        2. Clip each gradient
        3. Add noise
        4. Average and apply update
        5. Update privacy accountant

        Args:
            loss: Batch loss
            inputs: Input batch
        """
        batch_size = inputs.shape[0]

        # Zero existing gradients
        self.optimizer.zero_grad()

        # Compute per-example gradients (simplified)
        # In production: use Opacus or torch.func.grad_and_value
        for _name, param in self.model.named_parameters():
            if param.grad is not None:
                # Clip gradient
                grad_norm = torch.norm(param.grad)
                if grad_norm > self.config.max_grad_norm:
                    param.grad = param.grad * (
                        self.config.max_grad_norm / grad_norm
                    )

                # Add noise
                noise_scale = (
                    self.config.noise_multiplier *
                    self.config.max_grad_norm /
                    batch_size
                )
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad = param.grad + noise

        # Apply update
        self.optimizer.step()

        # Update privacy accounting
        self._update_privacy_accountant()

    def _update_privacy_accountant(self):
        """
        Update privacy loss after one step

        Uses composition theorems to track cumulative (ε,δ)
        """
        self.accountant.steps += 1

        # Simplified privacy accounting
        # In production: use opacus.privacy_analysis or TF Privacy

        # Per-step privacy loss (simplified Gaussian mechanism)
        sampling_prob = self.config.batch_size / 60000  # Assume 60k dataset

        # RDP accounting (simplified)
        step_epsilon = (
            2 * sampling_prob * self.accountant.steps /
            (self.config.noise_multiplier ** 2)
        )

        self.accountant.epsilon_spent = step_epsilon
        self.accountant.delta_spent = self.config.target_delta

        self.accountant.history.append((
            self.accountant.epsilon_spent,
            self.accountant.delta_spent
        ))

    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get current privacy spent

        Returns:
            (epsilon, delta) tuple
        """
        return (
            self.accountant.epsilon_spent,
            self.accountant.delta_spent
        )

class PrivateAggregationOfTeacherEnsembles:
    """
    PATE: Private Aggregation of Teacher Ensembles

    Alternative to DP-SGD:
    1. Train multiple "teacher" models on disjoint data
    2. Label public data using noisy aggregation of teachers
    3. Train "student" model on public labeled data

    Benefits:
    - Student model has no direct privacy cost
    - Can achieve better utility than DP-SGD
    - Suitable when public unlabeled data available

    Limitations:
    - Requires partitionable private data
    - Needs public unlabeled data
    - Multiple model training overhead
    """

    def __init__(
        self,
        num_teachers: int,
        embedding_dim: int,
        privacy_config: DPTrainingConfig
    ):
        self.num_teachers = num_teachers
        self.embedding_dim = embedding_dim
        self.privacy_config = privacy_config

        # Teacher models (would be trained separately)
        self.teachers: List[nn.Module] = []

        # Privacy accountant for aggregation
        self.accountant = PrivacyAccountant()

        print(f"PATE initialized with {num_teachers} teachers")

    def noisy_aggregation(
        self,
        teacher_predictions: List[torch.Tensor],
        epsilon: float = 1.0
    ) -> torch.Tensor:
        """
        Aggregate teacher predictions with differential privacy

        Steps:
        1. Each teacher predicts on unlabeled example
        2. Take majority vote (for classification) or mean (for regression)
        3. Add Laplace noise to vote counts
        4. Return noisy consensus

        Args:
            teacher_predictions: List of teacher predictions
            epsilon: Privacy budget for this aggregation

        Returns:
            Noisy consensus prediction
        """
        # Stack predictions
        predictions = torch.stack(teacher_predictions)  # [num_teachers, ...]

        # Average predictions (for embeddings)
        consensus = predictions.mean(dim=0)

        # Add Laplace noise for privacy
        sensitivity = 2.0 / self.num_teachers  # Bounded by averaging
        noise_scale = sensitivity / epsilon

        noise = torch.from_numpy(
            np.random.laplace(0, noise_scale, size=consensus.shape)
        ).float()

        noisy_consensus = consensus + noise

        # Update privacy accounting
        self.accountant.epsilon_spent += epsilon
        self.accountant.steps += 1

        return noisy_consensus

    def train_student(
        self,
        public_data: torch.Tensor,
        student_model: nn.Module
    ) -> nn.Module:
        """
        Train student model on privately labeled public data

        Args:
            public_data: Unlabeled public data
            student_model: Student model to train

        Returns:
            Trained student model with privacy guarantee
        """
        # Get teacher predictions with noisy aggregation
        print("Labeling public data with private aggregation...")

        labels = []
        for example in public_data:
            # Each teacher predicts
            teacher_preds = [
                teacher(example.unsqueeze(0))
                for teacher in self.teachers
            ]

            # Noisy aggregation
            label = self.noisy_aggregation(
                teacher_preds,
                epsilon=self.privacy_config.target_epsilon / len(public_data)
            )
            labels.append(label)

        labels = torch.stack(labels)

        # Train student on labeled public data (no privacy cost!)
        print("Training student model...")
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

        for _epoch in range(10):
            optimizer.zero_grad()

            predictions = student_model(public_data)
            loss = F.mse_loss(predictions, labels)

            loss.backward()
            optimizer.step()

        print(f"Student trained with privacy: ε={self.accountant.epsilon_spent:.2f}")

        return student_model

# Example usage
def dp_embedding_training_example():
    """
    Demonstrate differentially private embedding training
    """
    print("=== Differentially Private Embedding Training ===")
    print()

    # Configuration
    dp_config = DPTrainingConfig(
        target_epsilon=1.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        batch_size=256,
        num_epochs=5,
        learning_rate=0.001
    )

    # Initialize model
    vocab_size = 10000
    embedding_dim = 256

    model = DPEmbeddingModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=512,
        num_layers=2
    )

    # Initialize DP optimizer
    dp_optimizer = DPSGDOptimizer(model, dp_config)
    print()

    # Simulate training data
    print("Training with DP-SGD...")
    num_samples = 10000

    for epoch in range(dp_config.num_epochs):
        epoch_loss = 0.0
        num_batches = num_samples // dp_config.batch_size

        for _batch_idx in range(num_batches):
            # Simulate batch
            batch_ids = torch.randint(
                0, vocab_size,
                (dp_config.batch_size, 20)
            )

            # Forward pass
            embeddings = model(batch_ids)

            # Contrastive loss (simplified)
            # Positive pairs: embeddings[i] with embeddings[i]
            # Negative pairs: embeddings[i] with embeddings[j≠i]
            similarity = torch.matmul(embeddings, embeddings.t())
            labels = torch.arange(dp_config.batch_size)
            loss = F.cross_entropy(similarity, labels)

            # Backward pass (compute gradients)
            loss.backward()

            # DP-SGD step
            dp_optimizer.step(loss, batch_ids)

            epoch_loss += loss.item()

        # Check privacy spent
        epsilon_spent, delta_spent = dp_optimizer.get_privacy_spent()

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{dp_config.num_epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Privacy spent: ε={epsilon_spent:.4f}, δ={delta_spent:.2e}")

        # Stop if privacy budget exhausted
        if epsilon_spent > dp_config.target_epsilon:
            print("  Privacy budget exhausted! Stopping training.")
            break

    print()
    print("Training complete!")
    final_eps, final_delta = dp_optimizer.get_privacy_spent()
    print(f"Final privacy guarantee: (ε={final_eps:.2f}, δ={final_delta:.2e})")

if __name__ == "__main__":
    dp_embedding_training_example()
