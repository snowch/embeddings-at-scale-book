import torch
import torch.nn.functional as F

# Code from Chapter 05
# Book: Embeddings at Scale


class InBatchHardNegativeMining:
    """
    Mine hard negatives from within batch

    Advantages:
    - Zero overhead (no additional computation)
    - Works with any batch size
    - Simple to implement

    Disadvantages:
    - Limited by batch diversity
    - May miss global hard negatives
    """

    def __init__(self, temperature=0.07, hardness_threshold=0.5):
        """
        Args:
            temperature: Contrastive loss temperature
            hardness_threshold: Minimum similarity to consider "hard"
        """
        self.temperature = temperature
        self.hardness_threshold = hardness_threshold

    def compute_loss_with_hard_negatives(
        self, anchor_emb, positive_emb, all_emb, num_hard_negatives=10
    ):
        """
        Compute contrastive loss using hard negatives from batch

        Args:
            anchor_emb: (batch_size, dim)
            positive_emb: (batch_size, dim)
            all_emb: (2*batch_size, dim) - all embeddings in batch
            num_hard_negatives: How many hard negatives to use per anchor

        Returns:
            loss, metrics
        """
        batch_size = anchor_emb.shape[0]

        # Normalize
        anchor_norm = F.normalize(anchor_emb, p=2, dim=1)
        positive_norm = F.normalize(positive_emb, p=2, dim=1)
        all_norm = F.normalize(all_emb, p=2, dim=1)

        # Compute all similarities: (batch_size, 2*batch_size)
        all_similarities = torch.matmul(anchor_norm, all_norm.T)

        # For each anchor, find hard negatives
        # Hard negative: high similarity but not the positive
        losses = []
        hard_negative_sims = []

        for i in range(batch_size):
            # Positive similarity
            pos_sim = F.cosine_similarity(anchor_norm[i : i + 1], positive_norm[i : i + 1])

            # All negative similarities (excluding positive)
            neg_sims = torch.cat([all_similarities[i, :i], all_similarities[i, i + 1 :]])

            # Find hard negatives: highest similarities that aren't positive
            # Filter to similarities above threshold (hard enough to be useful)
            hard_mask = neg_sims > self.hardness_threshold

            if hard_mask.sum() > 0:
                hard_neg_sims = neg_sims[hard_mask]

                # Take top-k hardest
                if len(hard_neg_sims) > num_hard_negatives:
                    hard_neg_sims = hard_neg_sims.topk(num_hard_negatives)[0]
            else:
                # Fallback: use hardest negatives even if below threshold
                hard_neg_sims = neg_sims.topk(min(num_hard_negatives, len(neg_sims)))[0]

            # Contrastive loss with hard negatives
            pos_exp = torch.exp(pos_sim / self.temperature)
            neg_exp = torch.exp(hard_neg_sims / self.temperature).sum()

            loss_i = -torch.log(pos_exp / (pos_exp + neg_exp))
            losses.append(loss_i)

            hard_negative_sims.append(hard_neg_sims.mean())

        loss = torch.stack(losses).mean()

        metrics = {
            "hard_negative_similarity": torch.stack(hard_negative_sims).mean().item(),
            "num_hard_negatives_used": num_hard_negatives,
        }

        return loss, metrics
