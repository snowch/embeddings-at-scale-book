import numpy as np
import torch
import torch.nn.functional as F

# Code from Chapter 08
# Book: Embeddings at Scale


class DifferentiallyPrivateEmbedding:
    """
    Add differential privacy guarantees to embeddings

    Ensures that embeddings don't leak information about
    individual training examples

    Privacy budget (ε):
    - ε < 1: Strong privacy (more noise, less accuracy)
    - ε = 1-10: Moderate privacy (balanced)
    - ε > 10: Weak privacy (less noise, more accuracy)
    """

    def __init__(self, embedding_model, epsilon=1.0, delta=1e-5):
        self.model = embedding_model
        self.epsilon = epsilon
        self.delta = delta

        # Track privacy budget consumption
        self.privacy_spent = 0.0

    def private_gradient_descent(
        self, data, labels, num_iterations=1000, batch_size=64, clip_norm=1.0
    ):
        """
        Train with differentially private SGD

        Steps:
        1. Clip gradients (bound sensitivity)
        2. Add Gaussian noise
        3. Update model
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        # Noise scale based on privacy budget
        noise_scale = clip_norm * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

        for _iteration in range(num_iterations):
            # Sample batch
            indices = torch.randint(0, len(data), (batch_size,))
            batch_data = data[indices]
            batch_labels = labels[indices]

            # Forward pass
            embeddings = self.model(batch_data)
            loss = F.cross_entropy(embeddings, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)

            # Add noise
            for param in self.model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * noise_scale
                    param.grad.add_(noise)

            # Update
            optimizer.step()

            # Track privacy consumption (simplified)
            self.privacy_spent += self.epsilon / num_iterations

        print(f"Privacy budget spent: {self.privacy_spent:.2f}")
