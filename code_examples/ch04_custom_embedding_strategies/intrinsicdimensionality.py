# Code from Chapter 04
# Book: Embeddings at Scale

import numpy as np
from sklearn.decomposition import PCA


class IntrinsicDimensionality:
    """
    Estimate intrinsic dimensionality of embedding space
    """

    def estimate_via_pca(self, embeddings, variance_threshold=0.95):
        """
        Use PCA to find dimensions capturing X% of variance
        """
        pca = PCA()
        pca.fit(embeddings)

        # Cumulative explained variance
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

        # Find number of components needed for threshold
        n_components = np.argmax(cumsum_variance >= variance_threshold) + 1

        return {
            'intrinsic_dimension': n_components,
            'variance_captured': cumsum_variance[n_components - 1],
            'variance_ratio_by_component': pca.explained_variance_ratio_
        }

    def estimate_via_mle(self, embeddings, k=10):
        """
        Maximum Likelihood Estimation of intrinsic dimensionality

        Based on: Levina & Bickel (2004)
        """
        from sklearn.neighbors import NearestNeighbors

        # Find k nearest neighbors for each point
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)

        # Remove self (distance 0)
        distances = distances[:, 1:]

        # MLE estimator
        # d_i = (k / sum_{j=1}^{k} log(r_k / r_j))
        dimensions = []

        for dist_vec in distances:
            r_k = dist_vec[-1]  # Distance to k-th neighbor
            if r_k > 0:
                log_ratios = np.log(r_k / dist_vec[:-1])
                if log_ratios.sum() > 0:
                    d_i = (k - 1) / log_ratios.sum()
                    dimensions.append(d_i)

        intrinsic_dim = np.median(dimensions)

        return {
            'intrinsic_dimension': int(intrinsic_dim),
            'dimension_distribution': dimensions
        }


# Example usage
embeddings = load_embeddings()  # Your 768-dim embeddings

estimator = IntrinsicDimensionality()

# PCA-based estimate
pca_result = estimator.estimate_via_pca(embeddings, variance_threshold=0.95)
print(f"PCA estimate: {pca_result['intrinsic_dimension']} dimensions capture 95% variance")

# MLE estimate
mle_result = estimator.estimate_via_mle(embeddings, k=10)
print(f"MLE estimate: {mle_result['intrinsic_dimension']} dimensions")

# Recommendation: Use max of estimates as minimum dimension
recommended_dim = max(
    pca_result['intrinsic_dimension'],
    mle_result['intrinsic_dimension']
)
print(f"\nRecommended minimum: {recommended_dim} dimensions")
