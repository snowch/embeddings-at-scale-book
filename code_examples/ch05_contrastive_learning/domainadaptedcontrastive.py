import torch
import torch.nn.functional as F

# Code from Chapter 05
# Book: Embeddings at Scale


class DomainAdaptedContrastive:
    """
    Domain-specific adaptations for enterprise use cases
    """

    def hierarchical_contrastive_loss(self, embeddings, hierarchical_labels):
        """
        Hierarchical contrastive learning for taxonomies

        Use case: E-commerce categories, medical ontologies, document classification

        Args:
            embeddings: (batch_size, dim)
            hierarchical_labels: List of (parent_id, child_id, item_id)

        Returns:
            loss that respects hierarchy
        """
        # Items with same parent should be somewhat similar
        # Items with same grandparent should be slightly similar
        # Items in different hierarchies should be dissimilar

        # Implement soft contrastive loss with hierarchy weights
        batch_size = len(embeddings)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T)

        # Create weight matrix based on hierarchical distance
        weights = torch.zeros((batch_size, batch_size))

        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    weights[i, j] = 0  # Self
                elif hierarchical_labels[i]["child_id"] == hierarchical_labels[j]["child_id"]:
                    weights[i, j] = 1.0  # Same leaf node
                elif hierarchical_labels[i]["parent_id"] == hierarchical_labels[j]["parent_id"]:
                    weights[i, j] = 0.5  # Same parent, different child
                else:
                    weights[i, j] = -1.0  # Different hierarchy

        # Weighted contrastive loss
        # Positive examples: weight > 0
        # Negative examples: weight < 0
        positive_mask = weights > 0
        negative_mask = weights < 0

        loss = 0
        for i in range(batch_size):
            if positive_mask[i].sum() == 0:
                continue

            # Positive similarities (maximize)
            pos_sim = sim_matrix[i, positive_mask[i]] * weights[i, positive_mask[i]]

            # Negative similarities (minimize)
            neg_sim = sim_matrix[i, negative_mask[i]]

            # Contrastive loss for this example
            numerator = torch.exp(pos_sim).sum()
            denominator = numerator + torch.exp(neg_sim).sum()

            loss -= torch.log(numerator / denominator)

        return loss / batch_size

    def temporal_contrastive_loss(self, embeddings, timestamps, decay_halflife=30):
        """
        Temporal contrastive learning: recent items more similar

        Use case: News, social media, time-series data

        Args:
            embeddings: (batch_size, dim)
            timestamps: (batch_size,) - Unix timestamps
            decay_halflife: Days for similarity decay

        Returns:
            loss with temporal weighting
        """
        len(embeddings)

        # Compute temporal distances (in days)
        time_diff_matrix = torch.abs(timestamps.unsqueeze(1) - timestamps.unsqueeze(0)) / (
            60 * 60 * 24
        )  # Convert to days

        # Temporal similarity: exponential decay
        temporal_similarity = torch.exp(-time_diff_matrix / decay_halflife)

        # Compute embedding similarities
        emb_sim = torch.matmul(F.normalize(embeddings, dim=1), F.normalize(embeddings, dim=1).T)

        # Loss: embedding similarity should match temporal similarity
        # Recent items should have similar embeddings
        loss = F.mse_loss(emb_sim, temporal_similarity)

        return loss
