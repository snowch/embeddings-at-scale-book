import torch
import torch.nn.functional as F

# Code from Chapter 05
# Book: Embeddings at Scale

class QueueBasedHardNegativeMining:
    """
    Maintain queue of recent embeddings for hard negative mining

    Advantages:
    - Larger pool of candidates (10K-100K typical)
    - Can find harder negatives than single batch
    - Smoother training (less batch-dependent)

    Disadvantages:
    - Staleness: queue contains embeddings from old model
    - Memory overhead
    """

    def __init__(self, embedding_dim, queue_size=65536,
                 temperature=0.07, momentum=0.999):
        """
        Args:
            embedding_dim: Dimension of embeddings
            queue_size: Number of embeddings to store
            temperature: Contrastive temperature
            momentum: Momentum for queue encoder updates
        """
        self.embedding_dim = embedding_dim
        self.queue_size = queue_size
        self.temperature = temperature
        self.momentum = momentum

        # Queue of embeddings
        self.queue = torch.randn(queue_size, embedding_dim)
        self.queue = F.normalize(self.queue, dim=1)

        # Metadata for each embedding in queue (optional)
        self.queue_metadata = [None] * queue_size

        self.queue_ptr = 0

    def update_queue(self, new_embeddings, metadata=None):
        """
        Add new embeddings to queue, removing oldest

        Args:
            new_embeddings: (batch_size, embedding_dim)
            metadata: Optional metadata for each embedding
        """
        batch_size = new_embeddings.shape[0]

        # Add to queue
        end_ptr = (self.queue_ptr + batch_size) % self.queue_size

        if end_ptr > self.queue_ptr:
            self.queue[self.queue_ptr:end_ptr] = new_embeddings.detach()
            if metadata:
                self.queue_metadata[self.queue_ptr:end_ptr] = metadata
        else:
            # Wrap around
            first_part = self.queue_size - self.queue_ptr
            self.queue[self.queue_ptr:] = new_embeddings[:first_part].detach()
            self.queue[:end_ptr] = new_embeddings[first_part:].detach()

            if metadata:
                self.queue_metadata[self.queue_ptr:] = metadata[:first_part]
                self.queue_metadata[:end_ptr] = metadata[first_part:]

        self.queue_ptr = end_ptr

    def mine_hard_negatives(self, anchor_emb, k=100):
        """
        Find k hardest negatives from queue for each anchor

        Args:
            anchor_emb: (batch_size, embedding_dim)
            k: Number of hard negatives to return

        Returns:
            hard_negatives: (batch_size, k, embedding_dim)
            hard_negative_scores: (batch_size, k)
        """
        # Normalize
        anchor_norm = F.normalize(anchor_emb, dim=1)

        # Compute similarities to all queue entries
        # (batch_size, queue_size)
        similarities = torch.matmul(anchor_norm, self.queue.T)

        # For each anchor, get top-k hardest (highest similarity) negatives
        # (batch_size, k)
        hard_negative_scores, hard_negative_indices = similarities.topk(k, dim=1)

        # Gather hard negative embeddings
        # (batch_size, k, embedding_dim)
        hard_negatives = self.queue[hard_negative_indices]

        return hard_negatives, hard_negative_scores

    def compute_loss(self, anchor_emb, positive_emb, num_hard_negatives=100):
        """
        Compute contrastive loss using hard negatives from queue
        """
        anchor_emb.shape[0]

        # Get hard negatives
        hard_negs, hard_neg_sims = self.mine_hard_negatives(
            anchor_emb,
            k=num_hard_negatives
        )

        # Normalize
        anchor_norm = F.normalize(anchor_emb, dim=1)
        positive_norm = F.normalize(positive_emb, dim=1)

        # Positive similarities
        pos_sim = F.cosine_similarity(anchor_norm, positive_norm, dim=1)

        # Hard negative similarities (already computed in mining)
        # Shape: (batch_size, num_hard_negatives)

        # Compute loss
        pos_exp = torch.exp(pos_sim / self.temperature)
        neg_exp = torch.exp(hard_neg_sims / self.temperature).sum(dim=1)

        loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()

        # Update queue with current batch
        self.update_queue(anchor_emb)

        metrics = {
            'positive_similarity': pos_sim.mean().item(),
            'hard_negative_similarity': hard_neg_sims.mean().item(),
            'queue_utilization': min(self.queue_ptr, self.queue_size) / self.queue_size
        }

        return loss, metrics
