import torch

# Code from Chapter 02
# Book: Embeddings at Scale

class MultiModalTraining:
    """Training strategies for multi-modal embeddings"""

    def contrastive_loss(self, anchor_emb, positive_emb, negative_embs, temperature=0.07):
        """
        Contrastive learning: anchor close to positive, far from negatives
        Used by CLIP, ALIGN, and other multi-modal models
        """
        # Similarities
        pos_sim = torch.cosine_similarity(anchor_emb, positive_emb) / temperature
        neg_sims = torch.stack([
            torch.cosine_similarity(anchor_emb, neg_emb) / temperature
            for neg_emb in negative_embs
        ])

        # InfoNCE loss
        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.sum(torch.exp(neg_sims))
        loss = -torch.log(numerator / denominator)

        return loss

    def triplet_loss(self, anchor, positive, negative, margin=0.2):
        """
        Triplet loss: distance(anchor, positive) + margin < distance(anchor, negative)
        Classic approach for metric learning
        """
        pos_dist = torch.norm(anchor - positive)
        neg_dist = torch.norm(anchor - negative)
        loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        return loss

    def alignment_and_uniformity_loss(self, embeddings1, embeddings2, labels):
        """
        Two objectives:
        - Alignment: matched pairs should be close
        - Uniformity: embeddings should be uniformly distributed on hypersphere

        This prevents collapse while encouraging alignment
        """
        # Alignment loss: matched pairs close together
        matched_pairs = [(emb1, emb2) for emb1, emb2, label in zip(embeddings1, embeddings2, labels) if label == 1]
        alignment_loss = sum(
            torch.norm(emb1 - emb2) ** 2
            for emb1, emb2 in matched_pairs
        ) / len(matched_pairs)

        # Uniformity loss: embeddings uniformly distributed
        def uniformity(embeddings):
            # Pairwise distances on unit hypersphere
            normalized = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
            pairwise_dot = torch.matmul(normalized, normalized.T)
            loss = torch.log(torch.mean(torch.exp(pairwise_dot)))
            return loss

        uniformity_loss = uniformity(embeddings1) + uniformity(embeddings2)

        # Combined loss
        return alignment_loss + uniformity_loss
