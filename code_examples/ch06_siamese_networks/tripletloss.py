import torch
import torch.nn as nn
import torch.nn.functional as F

# Code from Chapter 06
# Book: Embeddings at Scale


class TripletLoss(nn.Module):
    """
    Triplet loss for learning embeddings

    Introduced by Schroff et al. (2015) for FaceNet.
    More efficient than contrastive loss as each sample contributes
    to two comparisons (positive and negative).

    The goal: d(anchor, positive) + margin < d(anchor, negative)
    """

    def __init__(self, margin=1.0, distance_metric="euclidean"):
        """
        Args:
            margin: Minimum difference between positive and negative distances
            distance_metric: 'euclidean' or 'cosine'
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss

        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)

        Returns:
            loss: Scalar tensor
            metrics: Dict with distances and statistics
        """
        if self.distance_metric == "euclidean":
            pos_distance = F.pairwise_distance(anchor, positive, p=2)
            neg_distance = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance_metric == "cosine":
            pos_distance = 1 - F.cosine_similarity(anchor, positive)
            neg_distance = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        # Triplet loss: positive should be closer than negative by margin
        losses = F.relu(pos_distance - neg_distance + self.margin)
        loss = losses.mean()

        # Compute metrics
        with torch.no_grad():
            # Fraction of triplets that violate the margin
            hard_triplets = (losses > 0).float().mean()

            # Average distances
            avg_pos_distance = pos_distance.mean()
            avg_neg_distance = neg_distance.mean()

            # Accuracy: positive closer than negative
            accuracy = (pos_distance < neg_distance).float().mean()

            metrics = {
                "loss": loss.item(),
                "accuracy": accuracy.item(),
                "hard_triplets_fraction": hard_triplets.item(),
                "avg_pos_distance": avg_pos_distance.item(),
                "avg_neg_distance": avg_neg_distance.item(),
                "avg_margin": (avg_neg_distance - avg_pos_distance).item(),
            }

        return loss, metrics
