import pandas as pd
import torch
import torch.nn.functional as F

# Code from Chapter 05
# Book: Embeddings at Scale

class AlignmentUniformityAnalysis:
    """
    Analyze embedding quality via alignment and uniformity metrics
    """

    def compute_alignment(self, embeddings1, embeddings2):
        """
        Compute alignment between positive pairs

        Args:
            embeddings1, embeddings2: (N, dim) paired embeddings

        Returns:
            alignment: lower is better (closer pairs)
        """
        # Normalize
        emb1 = F.normalize(embeddings1, p=2, dim=1)
        emb2 = F.normalize(embeddings2, p=2, dim=1)

        # Squared L2 distance
        alignment = torch.norm(emb1 - emb2, p=2, dim=1).pow(2).mean()

        return alignment.item()

    def compute_uniformity(self, embeddings, t=2):
        """
        Compute uniformity of embedding distribution

        Args:
            embeddings: (N, dim)
            t: temperature parameter (default 2 from paper)

        Returns:
            uniformity: lower is better (more uniform)
        """
        # Normalize
        emb = F.normalize(embeddings, p=2, dim=1)

        # Pairwise similarities
        # (N, N) matrix of similarities
        sim_matrix = torch.matmul(emb, emb.T)

        # Exclude diagonal (self-similarity)
        mask = ~torch.eye(len(emb), dtype=torch.bool, device=emb.device)
        similarities = sim_matrix[mask]

        # Uniformity: log of average exp(-t * squared_distance)
        # Since ||x - y||² = 2(1 - x·y) for normalized vectors:
        squared_distances = 2 * (1 - similarities)

        uniformity = torch.log(torch.exp(-t * squared_distances).mean())

        return uniformity.item()

    def analyze_training_progress(self, embeddings_history):
        """
        Track alignment and uniformity throughout training

        Args:
            embeddings_history: List of (epoch, positive_pairs) tuples

        Returns:
            DataFrame with alignment and uniformity over time
        """
        results = []

        for epoch, (emb1, emb2) in embeddings_history:
            # Combine for uniformity calculation
            all_emb = torch.cat([emb1, emb2], dim=0)

            alignment = self.compute_alignment(emb1, emb2)
            uniformity = self.compute_uniformity(all_emb)

            results.append({
                'epoch': epoch,
                'alignment': alignment,
                'uniformity': uniformity
            })

        return pd.DataFrame(results)


# Healthy training should show:
# - Alignment decreasing (positives getting closer)
# - Uniformity stable or slightly decreasing (not collapsing)
# - If uniformity increases significantly → embeddings collapsing
