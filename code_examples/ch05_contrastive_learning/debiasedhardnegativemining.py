import torch.nn.functional as F

# Code from Chapter 05
# Book: Embeddings at Scale

class DebiasedHardNegativeMining:
    """
    Filter out false negatives from hard negative candidates

    Techniques:
    1. Cross-encoder filtering
    2. Clustering-based filtering
    3. Human-in-the-loop verification
    4. Confidence thresholding
    """

    def __init__(self, embedding_dim, cross_encoder=None):
        """
        Args:
            embedding_dim: Dimension of embeddings
            cross_encoder: Optional cross-encoder model for verification
        """
        self.embedding_dim = embedding_dim
        self.cross_encoder = cross_encoder

    def filter_with_cross_encoder(self, anchor_texts, negative_candidates,
                                  threshold=0.3):
        """
        Use cross-encoder to filter out false negatives

        Cross-encoder: directly compares pair of texts (more accurate than embeddings)

        Args:
            anchor_texts: List of anchor texts
            negative_candidates: List of lists of candidate negatives
            threshold: Similarity threshold for false negative (0-1)

        Returns:
            filtered_negatives: List of lists with false negatives removed
        """
        if self.cross_encoder is None:
            raise ValueError("Cross-encoder model required")

        filtered_negatives = []

        for anchor, candidates in zip(anchor_texts, negative_candidates):
            # Score each candidate with cross-encoder
            pairs = [[anchor, cand] for cand in candidates]
            scores = self.cross_encoder.predict(pairs)

            # Filter out candidates with high similarity (likely false negatives)
            filtered = [
                cand for cand, score in zip(candidates, scores)
                if score < threshold
            ]

            filtered_negatives.append(filtered)

        return filtered_negatives

    def filter_with_clustering(self, embeddings, hard_negative_indices,
                              cluster_threshold=0.8):
        """
        Filter negatives that cluster with positives

        Logic: If hard negative is in same tight cluster as anchor,
               likely false negative

        Args:
            embeddings: (num_examples, embedding_dim)
            hard_negative_indices: (num_queries, num_candidates)
            cluster_threshold: Similarity threshold for same cluster

        Returns:
            filtered_indices: Hard negatives outside anchor's cluster
        """
        from sklearn.cluster import AgglomerativeClustering

        # Cluster embeddings
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - cluster_threshold,
            linkage='average'
        )

        labels = clustering.fit_predict(embeddings)

        # For each query, filter negatives in same cluster
        filtered_indices = []

        for query_idx, neg_indices in enumerate(hard_negative_indices):
            query_cluster = labels[query_idx]

            # Keep negatives from different clusters
            filtered = [
                neg_idx for neg_idx in neg_indices
                if labels[neg_idx] != query_cluster
            ]

            filtered_indices.append(filtered)

        return filtered_indices

    def confidence_based_filtering(self, anchor_emb, positive_emb,
                                   negative_candidates, min_margin=0.1):
        """
        Filter negatives too similar to positive

        Ensures: sim(anchor, negative) < sim(anchor, positive) - margin

        Args:
            anchor_emb: (batch_size, dim)
            positive_emb: (batch_size, dim)
            negative_candidates: (batch_size, num_candidates, dim)
            min_margin: Minimum margin between positive and negative

        Returns:
            filtered_negatives: Negatives with sufficient margin
        """
        # Normalize
        anchor_norm = F.normalize(anchor_emb, dim=1)
        positive_norm = F.normalize(positive_emb, dim=1)

        # Positive similarities
        pos_sim = F.cosine_similarity(anchor_norm, positive_norm, dim=1)

        filtered_negatives = []

        for i in range(len(anchor_emb)):
            # Negative similarities
            neg_sims = F.cosine_similarity(
                anchor_norm[i:i+1],
                negative_candidates[i],
                dim=1
            )

            # Filter: keep negatives with sim < pos_sim - margin
            threshold = pos_sim[i] - min_margin
            valid_mask = neg_sims < threshold

            filtered = negative_candidates[i][valid_mask]
            filtered_negatives.append(filtered)

        return filtered_negatives
