# Code from Chapter 06
# Book: Embeddings at Scale

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese networks

    Introduced by Hadsell et al. (2006) for learning similarity metrics.

    Loss = (1 - Y) * 0.5 * D^2 + Y * 0.5 * max(margin - D, 0)^2

    Where:
    - Y = 1 if pair is dissimilar, 0 if similar
    - D = distance between embeddings
    - margin = how far apart dissimilar pairs should be
    """

    def __init__(self, margin=2.0):
        """
        Args:
            margin: Margin for dissimilar pairs. Dissimilar pairs with
                   distance > margin contribute 0 to loss.
                   Typical values: 1.0 - 2.0
        """
        super().__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        """
        Compute contrastive loss

        Args:
            embedding1: First embeddings (batch_size, embedding_dim)
            embedding2: Second embeddings (batch_size, embedding_dim)
            label: 0 if similar, 1 if dissimilar (batch_size,)

        Returns:
            loss: Scalar tensor
            metrics: Dict with distances and accuracy
        """
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)

        # Contrastive loss
        # For similar pairs (label=0): minimize distance
        # For dissimilar pairs (label=1): maximize distance up to margin
        loss_similar = (1 - label) * torch.pow(euclidean_distance, 2)
        loss_dissimilar = label * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )

        loss = torch.mean(loss_similar + loss_dissimilar) * 0.5

        # Compute metrics
        with torch.no_grad():
            # Accuracy: similar pairs have distance < margin/2
            threshold = self.margin / 2
            predictions = (euclidean_distance < threshold).long()
            accuracy = (predictions == (1 - label)).float().mean()

            similar_mask = (label == 0)
            dissimilar_mask = (label == 1)

            metrics = {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'mean_similar_distance': euclidean_distance[similar_mask].mean().item() if similar_mask.any() else 0,
                'mean_dissimilar_distance': euclidean_distance[dissimilar_mask].mean().item() if dissimilar_mask.any() else 0,
            }

        return loss, metrics
