# Code from Chapter 02
# Book: Embeddings at Scale

from sklearn.decomposition import PCA

class DimensionReducer:
    """Reduce embedding dimensionality to save costs"""

    def reduce_dimensions(self, embeddings, target_dim, method='pca'):
        """
        Reduce embedding dimensions
        768-dim → 256-dim = 66% storage savings
        """
        if method == 'pca':
            pca = PCA(n_components=target_dim)
            reduced = pca.fit_transform(embeddings)

            # Evaluate quality loss
            variance_retained = pca.explained_variance_ratio_.sum()

            return {
                'reduced_embeddings': reduced,
                'variance_retained': variance_retained,
                'storage_savings': 1 - (target_dim / embeddings.shape[1]),
                'quality_loss': 1 - variance_retained
            }
