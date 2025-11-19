# Code from Chapter 05
# Book: Embeddings at Scale

import torch
import torch.nn.functional as F

class InfoNCELoss:
    """
    InfoNCE loss for contrastive learning

    Core idea: Given an anchor and one positive example, distinguish the
    positive from N-1 negative examples drawn from the distribution.
    """

    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: Controls the concentration of the distribution.
                        Lower = harder negatives, higher = softer.
                        Typical range: 0.01 - 0.5
        """
        self.temperature = temperature

    def compute_loss(self, anchor_embeddings, positive_embeddings,
                     negative_embeddings=None, all_embeddings=None):
        """
        Compute InfoNCE loss

        Args:
            anchor_embeddings: (batch_size, embed_dim)
            positive_embeddings: (batch_size, embed_dim)
            negative_embeddings: (batch_size, num_negatives, embed_dim)
                                or None if using all_embeddings
            all_embeddings: (total_size, embed_dim) - use all as negatives

        Returns:
            loss: scalar tensor
            metrics: dict with accuracy, positive/negative similarities
        """
        batch_size = anchor_embeddings.shape[0]

        # Normalize embeddings (critical for stable training)
        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_norm = F.normalize(positive_embeddings, p=2, dim=1)

        # Positive similarities: anchor · positive
        # Shape: (batch_size,)
        positive_sim = torch.sum(anchor_norm * positive_norm, dim=1) / self.temperature

        if all_embeddings is not None:
            # In-batch negatives: compare anchor to all embeddings
            # This is the efficient approach for large batches

            all_norm = F.normalize(all_embeddings, p=2, dim=1)

            # Similarity matrix: anchor × all
            # Shape: (batch_size, total_size)
            similarity_matrix = torch.matmul(
                anchor_norm,
                all_norm.T
            ) / self.temperature

            # Mask out the positive (assume positives at same index)
            # Create labels: positive is at index i for anchor i
            labels = torch.arange(batch_size, device=anchor_embeddings.device)

            # Cross-entropy: -log(exp(pos) / sum(exp(all)))
            loss = F.cross_entropy(similarity_matrix, labels)

            # Metrics
            with torch.no_grad():
                # Accuracy: how often is positive the highest similarity?
                predictions = similarity_matrix.argmax(dim=1)
                accuracy = (predictions == labels).float().mean()

                # Average positive/negative similarities
                positive_sim_mean = positive_sim.mean()

                # Negative similarities (excluding positive)
                mask = torch.ones_like(similarity_matrix, dtype=torch.bool)
                mask[torch.arange(batch_size), labels] = False
                negative_sim_mean = similarity_matrix[mask].mean()

        elif negative_embeddings is not None:
            # Explicit negatives provided
            negative_norm = F.normalize(negative_embeddings, p=2, dim=1)

            # Negative similarities: anchor · negatives
            # Shape: (batch_size, num_negatives)
            negative_sim = torch.matmul(
                anchor_norm.unsqueeze(1),  # (batch, 1, dim)
                negative_norm.transpose(1, 2)  # (batch, dim, num_neg)
            ).squeeze(1) / self.temperature

            # Concatenate positive and negative similarities
            # Shape: (batch_size, 1 + num_negatives)
            logits = torch.cat([
                positive_sim.unsqueeze(1),  # Positive is first
                negative_sim
            ], dim=1)

            # Labels: positive is always at index 0
            labels = torch.zeros(batch_size, dtype=torch.long,
                               device=anchor_embeddings.device)

            loss = F.cross_entropy(logits, labels)

            # Metrics
            with torch.no_grad():
                predictions = logits.argmax(dim=1)
                accuracy = (predictions == labels).float().mean()
                positive_sim_mean = positive_sim.mean()
                negative_sim_mean = negative_sim.mean()

        else:
            raise ValueError("Must provide either negative_embeddings or all_embeddings")

        metrics = {
            'accuracy': accuracy.item(),
            'positive_similarity': positive_sim_mean.item(),
            'negative_similarity': negative_sim_mean.item(),
            'similarity_gap': (positive_sim_mean - negative_sim_mean).item()
        }

        return loss, metrics


# Example usage
encoder = torch.nn.Sequential(
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128)
)

# Batch of data
anchors = torch.randn(64, 512)  # 64 examples
positives = torch.randn(64, 512)  # Corresponding positives
all_batch = torch.cat([anchors, positives], dim=0)  # Use full batch as negatives

# Encode
anchor_emb = encoder(anchors)
positive_emb = encoder(positives)
all_emb = encoder(all_batch)

# Compute loss
loss_fn = InfoNCELoss(temperature=0.07)
loss, metrics = loss_fn.compute_loss(
    anchor_emb,
    positive_emb,
    all_embeddings=all_emb
)

print(f"Loss: {loss.item():.4f}")
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Positive similarity: {metrics['positive_similarity']:.4f}")
print(f"Negative similarity: {metrics['negative_similarity']:.4f}")
