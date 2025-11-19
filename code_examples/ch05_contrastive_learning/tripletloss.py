# Code from Chapter 05
# Book: Embeddings at Scale

class TripletLoss:
    """
    Triplet loss: minimize distance(anchor, positive) while maximizing
    distance(anchor, negative) with a margin
    """

    def __init__(self, margin=1.0, distance='euclidean'):
        """
        Args:
            margin: Minimum distance between positive and negative
            distance: 'euclidean' or 'cosine'
        """
        self.margin = margin
        self.distance = distance

    def compute_loss(self, anchor, positive, negative):
        """
        Args:
            anchor: (batch_size, embed_dim)
            positive: (batch_size, embed_dim)
            negative: (batch_size, embed_dim) or (batch_size, num_neg, embed_dim)
        """
        if self.distance == 'euclidean':
            # L2 distance
            pos_dist = torch.norm(anchor - positive, p=2, dim=-1)

            if negative.dim() == 3:
                # Multiple negatives per anchor
                # Compute distance to each negative
                neg_dist = torch.norm(
                    anchor.unsqueeze(1) - negative,  # (batch, 1, dim) - (batch, num_neg, dim)
                    p=2,
                    dim=-1
                )  # (batch, num_neg)

                # Use hardest negative (smallest distance)
                neg_dist = neg_dist.min(dim=1)[0]
            else:
                neg_dist = torch.norm(anchor - negative, p=2, dim=-1)

        elif self.distance == 'cosine':
            # Cosine distance = 1 - cosine similarity
            pos_sim = F.cosine_similarity(anchor, positive, dim=-1)
            pos_dist = 1 - pos_sim

            if negative.dim() == 3:
                # Multiple negatives
                neg_sim = F.cosine_similarity(
                    anchor.unsqueeze(1),
                    negative,
                    dim=-1
                )
                neg_dist = 1 - neg_sim.max(dim=1)[0]  # Hardest negative
            else:
                neg_sim = F.cosine_similarity(anchor, negative, dim=-1)
                neg_dist = 1 - neg_sim

        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()
