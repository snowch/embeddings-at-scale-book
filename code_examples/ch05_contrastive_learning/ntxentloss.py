import torch
import torch.nn.functional as F

# Code from Chapter 05
# Book: Embeddings at Scale


class NTXentLoss:
    """
    NT-Xent loss from SimCLR paper

    Key differences from InfoNCE:
    - Symmetric: both (i, j) and (j, i) are positives
    - Uses cosine similarity
    - Specific temperature scaling
    """

    def __init__(self, temperature=0.5):
        self.temperature = temperature

    def compute_loss(self, embeddings, labels=None):
        """
        Args:
            embeddings: (2*batch_size, embed_dim) - contains augmented pairs
            labels: Optional labels indicating which pairs are positive
                   If None, assumes pairs are (2i, 2i+1)

        Returns:
            loss: scalar
        """
        batch_size = embeddings.shape[0] // 2

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix for all pairs
        # Shape: (2*batch_size, 2*batch_size)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create mask for positive pairs
        # For each i, positive is the other augmentation of the same instance
        mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)

        for i in range(batch_size):
            # (2i, 2i+1) are a positive pair
            mask[2 * i, 2 * i + 1] = True
            mask[2 * i + 1, 2 * i] = True

        # Remove self-similarities (diagonal)
        mask.fill_diagonal_(False)

        # For each sample, compute loss against its positive and all negatives
        losses = []

        for i in range(2 * batch_size):
            # Positive: the one entry in mask that's True for row i
            positive_indices = mask[i]

            # Get similarity to positive
            positive_sim = similarity_matrix[i, positive_indices]

            # Get similarities to all except self
            all_similarities = similarity_matrix[i]
            all_similarities = all_similarities[
                ~torch.eye(2 * batch_size, dtype=torch.bool, device=embeddings.device)[i]
            ]

            # Numerator: exp(positive similarity)
            numerator = torch.exp(positive_sim)

            # Denominator: sum of exp(all similarities except self)
            denominator = torch.exp(all_similarities).sum()

            # Loss: -log(numerator / denominator)
            loss_i = -torch.log(numerator / denominator)
            losses.append(loss_i)

        return torch.stack(losses).mean()
