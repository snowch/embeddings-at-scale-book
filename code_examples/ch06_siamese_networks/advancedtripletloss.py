import torch
import torch.nn as nn
import torch.nn.functional as F

# Code from Chapter 06
# Book: Embeddings at Scale

class AdvancedTripletLoss(nn.Module):
    """
    Advanced triplet loss with multiple optimization techniques

    Improvements over basic triplet loss:
    1. Hard negative mining: Focus on difficult examples
    2. Semi-hard negative mining: Balance between hard and random
    3. Soft margin: Smooth gradients for better optimization
    4. Angular loss: Focus on angles rather than distances
    """

    def __init__(
        self,
        margin=1.0,
        mining_strategy='semi-hard',  # 'hard', 'semi-hard', 'all'
        use_soft_margin=False,
        distance_metric='euclidean'
    ):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        self.use_soft_margin = use_soft_margin
        self.distance_metric = distance_metric

    def forward(self, embeddings, labels):
        """
        Compute triplet loss with online mining

        Args:
            embeddings: All embeddings in batch (batch_size, embedding_dim)
            labels: Class labels (batch_size,)

        Returns:
            loss: Scalar tensor
            metrics: Dict with mining statistics
        """
        batch_size = embeddings.shape[0]

        # Compute pairwise distances
        if self.distance_metric == 'euclidean':
            distances = torch.cdist(embeddings, embeddings, p=2)
        else:  # cosine
            # Normalize embeddings
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            distances = 1 - torch.mm(embeddings_norm, embeddings_norm.T)

        # Find valid triplets
        triplets = self._mine_triplets(distances, labels)

        if len(triplets) == 0:
            # No valid triplets found
            return torch.tensor(0.0, device=embeddings.device), {
                'loss': 0.0,
                'num_triplets': 0,
                'hard_triplets_fraction': 0.0
            }

        # Extract distances for valid triplets
        anchor_idx, positive_idx, negative_idx = zip(*triplets)

        pos_distances = distances[anchor_idx, positive_idx]
        neg_distances = distances[anchor_idx, negative_idx]

        # Compute loss
        if self.use_soft_margin:
            # Soft margin: log(1 + exp(pos - neg))
            # Smoother gradients, better for optimization
            loss = torch.log1p(torch.exp(pos_distances - neg_distances)).mean()
        else:
            # Hard margin: max(pos - neg + margin, 0)
            loss = F.relu(pos_distances - neg_distances + self.margin).mean()

        # Compute metrics
        with torch.no_grad():
            hard_triplets = (pos_distances > neg_distances).float().mean()

            metrics = {
                'loss': loss.item(),
                'num_triplets': len(triplets),
                'hard_triplets_fraction': hard_triplets.item(),
                'avg_pos_distance': pos_distances.mean().item(),
                'avg_neg_distance': neg_distances.mean().item()
            }

        return loss, metrics

    def _mine_triplets(self, distances, labels):
        """
        Mine triplets based on strategy

        Hard negative mining: For each anchor-positive pair, select the
        negative that's closest to the anchor (hardest negative).

        Semi-hard negative mining: Select negatives that are farther than
        positive but still within the margin (challenging but not impossible).

        All: Use all valid triplets (computationally expensive).
        """
        batch_size = labels.shape[0]
        triplets = []

        for i in range(batch_size):
            anchor_label = labels[i]

            # Find all positives (same label, different index)
            positive_mask = (labels == anchor_label) & (torch.arange(batch_size, device=labels.device) != i)
            positive_indices = torch.where(positive_mask)[0]

            if len(positive_indices) == 0:
                continue

            # Find all negatives (different label)
            negative_mask = (labels != anchor_label)
            negative_indices = torch.where(negative_mask)[0]

            if len(negative_indices) == 0:
                continue

            # Get distances from anchor
            anchor_pos_distances = distances[i, positive_indices]
            anchor_neg_distances = distances[i, negative_indices]

            if self.mining_strategy == 'hard':
                # For each positive, find hardest negative
                for pos_idx in positive_indices:
                    pos_distance = distances[i, pos_idx]

                    # Hardest negative: closest negative to anchor
                    hardest_neg_idx = negative_indices[anchor_neg_distances.argmin()]

                    triplets.append((i, pos_idx.item(), hardest_neg_idx.item()))

            elif self.mining_strategy == 'semi-hard':
                # For each positive, find semi-hard negatives
                for j, pos_idx in enumerate(positive_indices):
                    pos_distance = anchor_pos_distances[j]

                    # Semi-hard: negatives farther than positive but within margin
                    semi_hard_mask = (anchor_neg_distances > pos_distance) & \
                                   (anchor_neg_distances < pos_distance + self.margin)

                    if semi_hard_mask.any():
                        semi_hard_negs = negative_indices[semi_hard_mask]
                        # Pick one semi-hard negative
                        neg_idx = semi_hard_negs[0]
                    else:
                        # Fall back to hardest negative
                        neg_idx = negative_indices[anchor_neg_distances.argmin()]

                    triplets.append((i, pos_idx.item(), neg_idx.item()))

            else:  # 'all'
                # Use all valid combinations (expensive!)
                for pos_idx in positive_indices:
                    for neg_idx in negative_indices:
                        triplets.append((i, pos_idx.item(), neg_idx.item()))

        return triplets


class AngularTripletLoss(nn.Module):
    """
    Angular triplet loss for better geometric properties

    Instead of distance margins, focuses on angles between embeddings.
    Provides better invariance to embedding magnitude.

    Reference: "Deep Metric Learning with Angular Loss" (Wang et al., 2017)
    """

    def __init__(self, alpha=45):
        """
        Args:
            alpha: Angular margin in degrees (typical: 30-50 degrees)
        """
        super().__init__()
        self.alpha = torch.tensor(alpha * torch.pi / 180)  # Convert to radians

    def forward(self, anchor, positive, negative):
        """
        Compute angular triplet loss

        The loss encourages the angle between (anchor, positive) to be smaller
        than the angle between (anchor, negative) by at least alpha.
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Compute angles using dot products
        anchor_positive = (anchor * positive).sum(dim=1)
        anchor_negative = (anchor * negative).sum(dim=1)

        # Convert to angles
        angle_ap = torch.acos(torch.clamp(anchor_positive, -1, 1))
        angle_an = torch.acos(torch.clamp(anchor_negative, -1, 1))

        # Angular margin loss
        loss = F.relu(angle_ap - angle_an + self.alpha).mean()

        with torch.no_grad():
            metrics = {
                'loss': loss.item(),
                'avg_angle_ap': (angle_ap * 180 / torch.pi).mean().item(),
                'avg_angle_an': (angle_an * 180 / torch.pi).mean().item(),
            }

        return loss, metrics
