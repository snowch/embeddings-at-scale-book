import torch
import torch.nn as nn
import torch.nn.functional as F

# Code from Chapter 04
# Book: Embeddings at Scale

class ProgressiveDimensionReduction:
    """
    Start with high dimensions, progressively reduce while monitoring quality
    """

    def __init__(self, base_model, original_dim=768):
        self.base_model = base_model
        self.original_dim = original_dim

    def train_projection(self, embeddings, target_dim):
        """
        Learn projection from high-dim to low-dim
        """
        from sklearn.decomposition import PCA

        # Option 1: PCA (preserves variance)
        pca = PCA(n_components=target_dim)
        pca.fit(embeddings)

        # Option 2: Learned projection (preserves task performance)
        projection_net = nn.Linear(self.original_dim, target_dim)

        # Train projection to preserve similarities
        self.train_projection_network(projection_net, embeddings)

        return projection_net

    def train_projection_network(self, projection, embeddings, pairs=None):
        """
        Train projection to preserve pairwise similarities
        """
        optimizer = torch.optim.Adam(projection.parameters(), lr=1e-3)

        for epoch in range(10):
            # Sample pairs
            if pairs is None:
                idx1 = torch.randint(0, len(embeddings), (1000,))
                idx2 = torch.randint(0, len(embeddings), (1000,))

            # Original similarities
            orig_sim = F.cosine_similarity(
                embeddings[idx1],
                embeddings[idx2]
            )

            # Projected similarities
            proj_emb1 = projection(embeddings[idx1])
            proj_emb2 = projection(embeddings[idx2])
            proj_sim = F.cosine_similarity(proj_emb1, proj_emb2)

            # Loss: preserve similarities
            loss = F.mse_loss(proj_sim, orig_sim)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return projection

    def find_minimal_dimension(self, embeddings, test_data, quality_threshold=0.95):
        """
        Binary search for minimal dimension meeting quality threshold
        """
        original_quality = self.evaluate(self.base_model, test_data)
        target_quality = original_quality * quality_threshold

        # Binary search
        low, high = 64, self.original_dim
        best_dim = high

        while low <= high:
            mid = (low + high) // 2

            # Train projection to mid dimensions
            projection = self.train_projection(embeddings, target_dim=mid)

            # Evaluate
            quality = self.evaluate_with_projection(
                self.base_model,
                projection,
                test_data
            )

            if quality >= target_quality:
                # Can go lower
                best_dim = mid
                high = mid - 1
            else:
                # Need more dimensions
                low = mid + 1

        return best_dim
