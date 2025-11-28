# Code from Chapter 25
# Book: Embeddings at Scale

"""
Comprehensive Embedding Quality Monitoring

Architecture:
1. Intrinsic metrics: Semantic coherence, cluster stability, dimension utilization
2. Extrinsic metrics: Downstream task performance, proxy metrics
3. User metrics: CTR, conversion, dwell time correlated with embedding changes
4. Drift detection: Distribution shifts, concept emergence, calibration drift
5. Automated alerting: Statistical tests trigger retraining or rollback

Metrics:
- Coherence: Intra-cluster similarity vs inter-cluster dissimilarity
- Stability: Embedding consistency across model versions, time periods
- Calibration: Similarity score distribution, threshold reliability
- Coverage: Uniform semantic space utilization, no dead dimensions
- Downstream performance: Search relevance, classification accuracy, clustering quality

Quality thresholds:
- Coherence: >0.8 intra-cluster similarity, <0.3 inter-cluster
- Stability: >0.95 correlation between consecutive embeddings
- Downstream: <5% accuracy drop vs baseline, >90% absolute performance
- User metrics: <10% CTR drop, <5% conversion drop
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


@dataclass
class EmbeddingBatch:
    """
    Batch of embeddings for quality analysis

    Attributes:
        embeddings: Array of embedding vectors (N, D)
        ids: Embedding identifiers
        labels: Optional ground truth labels for supervised metrics
        metadata: Optional metadata (timestamps, source, etc.)
        timestamp: When embeddings were generated
        model_version: Embedding model version
    """

    embeddings: np.ndarray
    ids: List[str]
    labels: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    model_version: str = "unknown"


@dataclass
class QualityMetrics:
    """
    Comprehensive embedding quality metrics

    Attributes:
        timestamp: When metrics were computed
        model_version: Embedding model version

        # Intrinsic metrics
        intra_cluster_similarity: Average similarity within clusters
        inter_cluster_similarity: Average similarity between clusters
        silhouette_score: Silhouette coefficient (-1 to 1, higher better)
        davies_bouldin_score: DB index (lower better)
        calinski_harabasz_score: CH score (higher better)

        # Dimension utilization
        dimension_variance: Per-dimension variance
        effective_dimensions: Number of dimensions with >1% variance
        dimension_entropy: Entropy of dimension importance

        # Calibration metrics
        similarity_distribution: Histogram of pairwise similarities
        score_calibration_error: Calibration error for similarity scores
        threshold_stability: Variance in optimal threshold across folds

        # Stability metrics
        temporal_stability: Correlation with previous batch
        cross_version_stability: Correlation with other model versions

        # Downstream metrics (if available)
        downstream_accuracy: Performance on labeled task
        proxy_metrics: Dict of proxy task metrics

        # Anomaly flags
        anomalies: List of detected quality anomalies
        quality_score: Overall quality score (0-100)
    """

    timestamp: datetime
    model_version: str

    # Intrinsic clustering metrics
    intra_cluster_similarity: float
    inter_cluster_similarity: float
    silhouette_score: float
    davies_bouldin_score: float
    calinski_harabasz_score: float

    # Dimension metrics
    dimension_variance: np.ndarray
    effective_dimensions: int
    dimension_entropy: float

    # Calibration
    similarity_distribution: Dict[str, float]
    score_calibration_error: float
    threshold_stability: float

    # Stability
    temporal_stability: Optional[float] = None
    cross_version_stability: Dict[str, float] = field(default_factory=dict)

    # Downstream
    downstream_accuracy: Optional[float] = None
    proxy_metrics: Dict[str, float] = field(default_factory=dict)

    # Anomalies and overall score
    anomalies: List[str] = field(default_factory=list)
    quality_score: float = 0.0


class EmbeddingQualityMonitor:
    """
    Comprehensive embedding quality monitoring system

    Tracks intrinsic and extrinsic quality metrics, detects anomalies,
    alerts on degradation, and enables continuous quality improvement.
    """

    def __init__(
        self,
        reference_embeddings: Optional[EmbeddingBatch] = None,
        quality_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
        alert_callback: Optional[callable] = None,
        history_window: int = 100,
    ):
        """
        Initialize quality monitoring system

        Args:
            reference_embeddings: Baseline embeddings for comparison
            quality_thresholds: (min, max) thresholds for each metric
            alert_callback: Function to call when anomalies detected
            history_window: Number of historical metrics to retain
        """
        self.reference_embeddings = reference_embeddings
        self.quality_thresholds = quality_thresholds or self._default_thresholds()
        self.alert_callback = alert_callback
        self.history_window = history_window

        # Historical metrics for drift detection
        self.metrics_history: List[QualityMetrics] = []
        self.baseline_metrics: Optional[QualityMetrics] = None

        # Compute baseline metrics if reference provided
        if reference_embeddings is not None:
            self.baseline_metrics = self.compute_quality_metrics(reference_embeddings)

    def _default_thresholds(self) -> Dict[str, Tuple[float, float]]:
        """Default quality thresholds (min, max)"""
        return {
            "intra_cluster_similarity": (0.7, 1.0),
            "inter_cluster_similarity": (0.0, 0.4),
            "silhouette_score": (0.3, 1.0),
            "davies_bouldin_score": (0.0, 2.0),  # Lower is better
            "effective_dimensions": (50, None),  # At least 50 dimensions used
            "dimension_entropy": (3.0, None),  # High entropy = good utilization
            "temporal_stability": (0.90, 1.0),
            "downstream_accuracy": (0.85, 1.0),
            "quality_score": (70, 100),
        }

    def compute_quality_metrics(
        self, batch: EmbeddingBatch, n_clusters: int = 10, sample_size: int = 10000
    ) -> QualityMetrics:
        """
        Compute comprehensive quality metrics for embedding batch

        Args:
            batch: Embedding batch to analyze
            n_clusters: Number of clusters for clustering metrics
            sample_size: Sample size for expensive computations

        Returns:
            QualityMetrics object with all computed metrics
        """
        embeddings = batch.embeddings
        n_samples, n_dims = embeddings.shape

        # Sample if too large
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            sampled_embeddings = embeddings[indices]
            sampled_labels = batch.labels[indices] if batch.labels is not None else None
        else:
            sampled_embeddings = embeddings
            sampled_labels = batch.labels

        # 1. Clustering metrics (if labels available, use them; otherwise cluster)
        if sampled_labels is not None:
            labels = sampled_labels
        else:
            from sklearn.cluster import MiniBatchKMeans

            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
            labels = kmeans.fit_predict(sampled_embeddings)

        # Compute cluster statistics
        intra_sim, inter_sim = self._compute_cluster_similarities(sampled_embeddings, labels)

        # Sklearn clustering metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            silhouette = float(
                silhouette_score(
                    sampled_embeddings, labels, sample_size=min(5000, len(sampled_embeddings))
                )
            )
            db_score = float(davies_bouldin_score(sampled_embeddings, labels))
            ch_score = float(calinski_harabasz_score(sampled_embeddings, labels))

        # 2. Dimension utilization metrics
        dim_variance = np.var(embeddings, axis=0)
        total_variance = np.sum(dim_variance)
        variance_ratio = dim_variance / total_variance if total_variance > 0 else dim_variance
        effective_dims = int(np.sum(variance_ratio > 0.01))  # Dimensions with >1% variance
        dim_entropy = float(
            entropy(variance_ratio + 1e-10)
        )  # Add small constant for numerical stability

        # 3. Similarity calibration metrics
        similarity_dist = self._compute_similarity_distribution(sampled_embeddings)
        calibration_error = self._compute_calibration_error(sampled_embeddings, sampled_labels)
        threshold_stability = self._compute_threshold_stability(sampled_embeddings, sampled_labels)

        # 4. Temporal stability (compare with previous batch)
        temporal_stab = None
        if len(self.metrics_history) > 0:
            prev_batch = self.metrics_history[-1]
            temporal_stab = self._compute_temporal_stability(embeddings, prev_batch)

        # 5. Cross-version stability (compare with reference)
        cross_version_stab = {}
        if self.reference_embeddings is not None:
            cross_version_stab[self.reference_embeddings.model_version] = (
                self._compute_cross_version_stability(batch, self.reference_embeddings)
            )

        # 6. Downstream metrics (if labels available)
        downstream_acc = None
        proxy_metrics_dict = {}
        if sampled_labels is not None:
            downstream_acc = self._compute_downstream_accuracy(sampled_embeddings, sampled_labels)
            proxy_metrics_dict = self._compute_proxy_metrics(sampled_embeddings, sampled_labels)

        # Create metrics object
        metrics = QualityMetrics(
            timestamp=batch.timestamp,
            model_version=batch.model_version,
            intra_cluster_similarity=intra_sim,
            inter_cluster_similarity=inter_sim,
            silhouette_score=silhouette,
            davies_bouldin_score=db_score,
            calinski_harabasz_score=ch_score,
            dimension_variance=dim_variance,
            effective_dimensions=effective_dims,
            dimension_entropy=dim_entropy,
            similarity_distribution=similarity_dist,
            score_calibration_error=calibration_error,
            threshold_stability=threshold_stability,
            temporal_stability=temporal_stab,
            cross_version_stability=cross_version_stab,
            downstream_accuracy=downstream_acc,
            proxy_metrics=proxy_metrics_dict,
        )

        # Detect anomalies and compute quality score
        metrics.anomalies = self._detect_anomalies(metrics)
        metrics.quality_score = self._compute_quality_score(metrics)

        # Alert if anomalies detected
        if metrics.anomalies and self.alert_callback:
            self.alert_callback(metrics)

        # Add to history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.history_window:
            self.metrics_history.pop(0)

        return metrics

    def _compute_cluster_similarities(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, float]:
        """Compute average intra-cluster and inter-cluster similarity"""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            return 0.0, 0.0

        # Compute cluster centroids
        centroids = np.array([embeddings[labels == label].mean(axis=0) for label in unique_labels])

        # Intra-cluster similarity: average similarity to cluster centroid
        intra_sims = []
        for label in unique_labels:
            cluster_embeddings = embeddings[labels == label]
            centroid = centroids[label]
            sims = 1 - np.array([cosine(emb, centroid) for emb in cluster_embeddings])
            intra_sims.extend(sims)

        intra_cluster_sim = float(np.mean(intra_sims))

        # Inter-cluster similarity: average similarity between cluster centroids
        inter_sims = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                sim = 1 - cosine(centroids[i], centroids[j])
                inter_sims.append(sim)

        inter_cluster_sim = float(np.mean(inter_sims)) if inter_sims else 0.0

        return intra_cluster_sim, inter_cluster_sim

    def _compute_similarity_distribution(
        self, embeddings: np.ndarray, n_samples: int = 1000
    ) -> Dict[str, float]:
        """Compute distribution statistics of pairwise similarities"""
        # Sample pairs to avoid O(N^2) complexity
        n = len(embeddings)
        pairs = min(n_samples, n * (n - 1) // 2)

        similarities = []
        for _ in range(pairs):
            i, j = np.random.choice(n, 2, replace=False)
            sim = 1 - cosine(embeddings[i], embeddings[j])
            similarities.append(sim)

        similarities = np.array(similarities)

        return {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "q25": float(np.percentile(similarities, 25)),
            "q50": float(np.percentile(similarities, 50)),
            "q75": float(np.percentile(similarities, 75)),
            "q95": float(np.percentile(similarities, 95)),
        }

    def _compute_calibration_error(
        self, embeddings: np.ndarray, labels: Optional[np.ndarray], n_bins: int = 10
    ) -> float:
        """
        Compute calibration error for similarity scores

        For labeled data: similarity score should correlate with label agreement
        For unlabeled: return 0 (cannot compute without ground truth)
        """
        if labels is None:
            return 0.0

        # Sample pairs with labels
        n = len(embeddings)
        n_samples = min(1000, n * (n - 1) // 2)

        similarities = []
        agreements = []

        for _ in range(n_samples):
            i, j = np.random.choice(n, 2, replace=False)
            sim = 1 - cosine(embeddings[i], embeddings[j])
            agree = 1.0 if labels[i] == labels[j] else 0.0
            similarities.append(sim)
            agreements.append(agree)

        similarities = np.array(similarities)
        agreements = np.array(agreements)

        # Bin similarities and compute calibration error
        calibration_error = 0.0
        for i in range(n_bins):
            lower = i / n_bins
            upper = (i + 1) / n_bins
            mask = (similarities >= lower) & (similarities < upper)

            if mask.sum() == 0:
                continue

            avg_sim = similarities[mask].mean()
            avg_agree = agreements[mask].mean()
            calibration_error += abs(avg_sim - avg_agree) * mask.sum()

        calibration_error /= n_samples
        return float(calibration_error)

    def _compute_threshold_stability(
        self, embeddings: np.ndarray, labels: Optional[np.ndarray], n_folds: int = 5
    ) -> float:
        """
        Compute stability of optimal similarity threshold across data splits

        For unlabeled data: return 0 (cannot compute without ground truth)
        """
        if labels is None:
            return 0.0

        from sklearn.metrics import f1_score
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        optimal_thresholds = []

        for train_idx, val_idx in kf.split(embeddings):
            _train_emb, val_emb = embeddings[train_idx], embeddings[val_idx]
            _train_labels, val_labels = labels[train_idx], labels[val_idx]

            # Find optimal threshold on validation set
            thresholds = np.linspace(0, 1, 50)
            best_threshold = 0.5
            best_f1 = 0.0

            for thresh in thresholds:
                # Sample pairs and compute F1
                n_samples = min(500, len(val_emb) * (len(val_emb) - 1) // 2)
                predictions = []
                ground_truth = []

                for _ in range(n_samples):
                    i, j = np.random.choice(len(val_emb), 2, replace=False)
                    sim = 1 - cosine(val_emb[i], val_emb[j])
                    pred = 1 if sim >= thresh else 0
                    truth = 1 if val_labels[i] == val_labels[j] else 0
                    predictions.append(pred)
                    ground_truth.append(truth)

                f1 = f1_score(ground_truth, predictions, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thresh

            optimal_thresholds.append(best_threshold)

        # Return variance of optimal thresholds
        return float(np.var(optimal_thresholds))

    def _compute_temporal_stability(
        self, current_embeddings: np.ndarray, previous_metrics: QualityMetrics
    ) -> float:
        """
        Compute stability between current and previous embeddings

        Uses dimension-wise correlation of variance patterns
        """
        current_variance = np.var(current_embeddings, axis=0)
        previous_variance = previous_metrics.dimension_variance

        # Ensure same dimensionality
        if len(current_variance) != len(previous_variance):
            return 0.0

        # Compute correlation of variance patterns
        correlation = np.corrcoef(current_variance, previous_variance)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

    def _compute_cross_version_stability(
        self, current_batch: EmbeddingBatch, reference_batch: EmbeddingBatch, n_samples: int = 1000
    ) -> float:
        """
        Compute stability between different model versions

        Compares embeddings for same IDs across versions
        """
        # Find common IDs
        current_ids = set(current_batch.ids)
        reference_ids = set(reference_batch.ids)
        common_ids = current_ids & reference_ids

        if len(common_ids) == 0:
            return 0.0

        # Sample common IDs
        sampled_ids = list(common_ids)[:n_samples]

        # Get embeddings for sampled IDs
        current_id_to_idx = {id_: idx for idx, id_ in enumerate(current_batch.ids)}
        reference_id_to_idx = {id_: idx for idx, id_ in enumerate(reference_batch.ids)}

        correlations = []
        for id_ in sampled_ids:
            curr_emb = current_batch.embeddings[current_id_to_idx[id_]]
            ref_emb = reference_batch.embeddings[reference_id_to_idx[id_]]

            # Compute cosine similarity
            sim = 1 - cosine(curr_emb, ref_emb)
            correlations.append(sim)

        return float(np.mean(correlations))

    def _compute_downstream_accuracy(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute downstream classification accuracy using embeddings"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, embeddings, labels, cv=5, scoring="accuracy")
        return float(np.mean(scores))

    def _compute_proxy_metrics(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute proxy task metrics (k-NN classification, clustering purity)"""
        from sklearn.model_selection import cross_val_score
        from sklearn.neighbors import KNeighborsClassifier

        metrics = {}

        # k-NN accuracy
        knn = KNeighborsClassifier(n_neighbors=5)
        knn_scores = cross_val_score(knn, embeddings, labels, cv=3, scoring="accuracy")
        metrics["knn_accuracy"] = float(np.mean(knn_scores))

        # Clustering purity (if enough samples)
        if len(embeddings) > 100:
            from sklearn.cluster import MiniBatchKMeans
            from sklearn.metrics import adjusted_rand_score

            n_clusters = len(np.unique(labels))
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
            pred_labels = kmeans.fit_predict(embeddings)
            metrics["clustering_ari"] = float(adjusted_rand_score(labels, pred_labels))

        return metrics

    def _detect_anomalies(self, metrics: QualityMetrics) -> List[str]:
        """Detect quality anomalies by comparing against thresholds"""
        anomalies = []

        # Check each metric against thresholds
        checks = [
            ("intra_cluster_similarity", metrics.intra_cluster_similarity),
            ("inter_cluster_similarity", metrics.inter_cluster_similarity),
            ("silhouette_score", metrics.silhouette_score),
            ("effective_dimensions", metrics.effective_dimensions),
            ("dimension_entropy", metrics.dimension_entropy),
        ]

        if metrics.temporal_stability is not None:
            checks.append(("temporal_stability", metrics.temporal_stability))

        if metrics.downstream_accuracy is not None:
            checks.append(("downstream_accuracy", metrics.downstream_accuracy))

        for metric_name, value in checks:
            if metric_name not in self.quality_thresholds:
                continue

            min_thresh, max_thresh = self.quality_thresholds[metric_name]

            if min_thresh is not None and value < min_thresh:
                anomalies.append(f"{metric_name} below threshold: {value:.3f} < {min_thresh}")

            if max_thresh is not None and value > max_thresh:
                anomalies.append(f"{metric_name} above threshold: {value:.3f} > {max_thresh}")

        # Davies-Bouldin score: lower is better, so invert the check
        db_min, db_max = self.quality_thresholds.get("davies_bouldin_score", (None, 2.0))
        if db_max is not None and metrics.davies_bouldin_score > db_max:
            anomalies.append(
                f"davies_bouldin_score too high: {metrics.davies_bouldin_score:.3f} > {db_max}"
            )

        # Check for significant drops from baseline
        if self.baseline_metrics is not None:
            if (
                metrics.downstream_accuracy is not None
                and self.baseline_metrics.downstream_accuracy is not None
            ):
                drop = self.baseline_metrics.downstream_accuracy - metrics.downstream_accuracy
                if drop > 0.05:  # >5% drop
                    anomalies.append(f"downstream_accuracy dropped {drop * 100:.1f}% from baseline")

            if metrics.temporal_stability is not None and metrics.temporal_stability < 0.90:
                anomalies.append(f"temporal_stability low: {metrics.temporal_stability:.3f} < 0.90")

        return anomalies

    def _compute_quality_score(self, metrics: QualityMetrics) -> float:
        """
        Compute overall quality score (0-100)

        Weighted combination of normalized metrics
        """
        scores = []
        weights = []

        # Intrinsic metrics (40% weight)
        if metrics.silhouette_score >= 0:
            scores.append(metrics.silhouette_score * 100)  # Already 0-1
            weights.append(0.15)

        if metrics.intra_cluster_similarity > metrics.inter_cluster_similarity:
            cluster_separation = metrics.intra_cluster_similarity - metrics.inter_cluster_similarity
            scores.append(cluster_separation * 100)
            weights.append(0.15)

        # Dimension utilization (20% weight)
        if metrics.effective_dimensions > 0:
            dim_score = min(100, (metrics.effective_dimensions / 100) * 100)  # Cap at 100 dims
            scores.append(dim_score)
            weights.append(0.20)

        # Stability (20% weight)
        if metrics.temporal_stability is not None:
            scores.append(metrics.temporal_stability * 100)
            weights.append(0.20)

        # Downstream performance (20% weight)
        if metrics.downstream_accuracy is not None:
            scores.append(metrics.downstream_accuracy * 100)
            weights.append(0.20)

        # If no weights added, return 50 (neutral)
        if not weights:
            return 50.0

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Compute weighted average
        quality_score = sum(s * w for s, w in zip(scores, normalized_weights))

        # Penalize anomalies
        if metrics.anomalies:
            quality_score *= 1 - 0.1 * len(metrics.anomalies)  # -10% per anomaly

        return max(0.0, min(100.0, quality_score))

    def generate_quality_report(self, metrics: QualityMetrics) -> str:
        """Generate human-readable quality report"""
        report = f"""
Embedding Quality Report
========================
Timestamp: {metrics.timestamp.isoformat()}
Model Version: {metrics.model_version}
Overall Quality Score: {metrics.quality_score:.1f}/100

Intrinsic Metrics:
------------------
Intra-cluster similarity: {metrics.intra_cluster_similarity:.3f}
Inter-cluster similarity: {metrics.inter_cluster_similarity:.3f}
Silhouette score: {metrics.silhouette_score:.3f}
Davies-Bouldin score: {metrics.davies_bouldin_score:.3f}
Calinski-Harabasz score: {metrics.calinski_harabasz_score:.1f}

Dimension Utilization:
---------------------
Effective dimensions: {metrics.effective_dimensions}
Dimension entropy: {metrics.dimension_entropy:.3f}

Similarity Distribution:
-----------------------
Mean: {metrics.similarity_distribution["mean"]:.3f}
Std: {metrics.similarity_distribution["std"]:.3f}
Range: [{metrics.similarity_distribution["min"]:.3f}, {metrics.similarity_distribution["max"]:.3f}]
Percentiles: Q25={metrics.similarity_distribution["q25"]:.3f}, Q50={metrics.similarity_distribution["q50"]:.3f}, Q75={metrics.similarity_distribution["q75"]:.3f}
"""

        if metrics.temporal_stability is not None:
            report += f"\nTemporal Stability: {metrics.temporal_stability:.3f}"

        if metrics.cross_version_stability:
            report += "\n\nCross-Version Stability:"
            for version, stability in metrics.cross_version_stability.items():
                report += f"\n  vs {version}: {stability:.3f}"

        if metrics.downstream_accuracy is not None:
            report += f"\n\nDownstream Accuracy: {metrics.downstream_accuracy:.3f}"

        if metrics.proxy_metrics:
            report += "\n\nProxy Metrics:"
            for metric, value in metrics.proxy_metrics.items():
                report += f"\n  {metric}: {value:.3f}"

        if metrics.anomalies:
            report += "\n\n⚠️ ANOMALIES DETECTED:"
            for anomaly in metrics.anomalies:
                report += f"\n  - {anomaly}"
        else:
            report += "\n\n✓ No anomalies detected"

        return report

    def plot_quality_trends(self, metrics_list: Optional[List[QualityMetrics]] = None):
        """
        Plot quality metrics over time

        Args:
            metrics_list: List of metrics to plot (defaults to history)
        """
        import matplotlib.pyplot as plt

        if metrics_list is None:
            metrics_list = self.metrics_history

        if not metrics_list:
            print("No metrics history available")
            return

        timestamps = [m.timestamp for m in metrics_list]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Overall quality score
        quality_scores = [m.quality_score for m in metrics_list]
        axes[0, 0].plot(timestamps, quality_scores, marker="o")
        axes[0, 0].set_title("Overall Quality Score")
        axes[0, 0].set_ylabel("Score (0-100)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=70, color="r", linestyle="--", alpha=0.5, label="Threshold")
        axes[0, 0].legend()

        # Plot 2: Clustering metrics
        silhouette_scores = [m.silhouette_score for m in metrics_list]
        intra_sims = [m.intra_cluster_similarity for m in metrics_list]
        inter_sims = [m.inter_cluster_similarity for m in metrics_list]
        axes[0, 1].plot(timestamps, silhouette_scores, marker="o", label="Silhouette")
        axes[0, 1].plot(timestamps, intra_sims, marker="s", label="Intra-cluster sim", alpha=0.7)
        axes[0, 1].plot(timestamps, inter_sims, marker="^", label="Inter-cluster sim", alpha=0.7)
        axes[0, 1].set_title("Clustering Metrics")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Stability metrics
        temporal_stabilities = [
            m.temporal_stability for m in metrics_list if m.temporal_stability is not None
        ]
        temporal_timestamps = [
            m.timestamp for m in metrics_list if m.temporal_stability is not None
        ]
        if temporal_stabilities:
            axes[1, 0].plot(temporal_timestamps, temporal_stabilities, marker="o", color="purple")
            axes[1, 0].set_title("Temporal Stability")
            axes[1, 0].set_ylabel("Correlation")
            axes[1, 0].axhline(y=0.90, color="r", linestyle="--", alpha=0.5, label="Threshold")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Downstream accuracy
        downstream_accs = [
            m.downstream_accuracy for m in metrics_list if m.downstream_accuracy is not None
        ]
        downstream_timestamps = [
            m.timestamp for m in metrics_list if m.downstream_accuracy is not None
        ]
        if downstream_accs:
            axes[1, 1].plot(downstream_timestamps, downstream_accs, marker="o", color="green")
            axes[1, 1].set_title("Downstream Accuracy")
            axes[1, 1].set_ylabel("Accuracy")
            axes[1, 1].axhline(y=0.85, color="r", linestyle="--", alpha=0.5, label="Threshold")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate synthetic embedding batches
    np.random.seed(42)

    # Reference embeddings (baseline)
    n_samples = 5000
    n_dims = 768
    n_classes = 10

    reference_embeddings_data = []
    reference_labels = []
    for i in range(n_classes):
        class_center = np.random.randn(n_dims) * 2
        class_samples = class_center + np.random.randn(n_samples // n_classes, n_dims) * 0.5
        reference_embeddings_data.append(class_samples)
        reference_labels.extend([i] * (n_samples // n_classes))

    reference_embeddings_data = np.vstack(reference_embeddings_data)
    reference_labels = np.array(reference_labels)

    reference_batch = EmbeddingBatch(
        embeddings=reference_embeddings_data,
        ids=[f"ref_{i}" for i in range(len(reference_embeddings_data))],
        labels=reference_labels,
        model_version="v1.0",
    )

    # Initialize monitor
    monitor = EmbeddingQualityMonitor(
        reference_embeddings=reference_batch,
        alert_callback=lambda m: print(f"⚠️ Alert: {len(m.anomalies)} anomalies detected!"),
    )

    # Simulate monitoring over time with gradual quality degradation
    for day in range(10):
        # Generate embeddings with increasing noise (simulating drift)
        noise_level = 0.5 + day * 0.1  # Increasing noise

        current_embeddings_data = []
        current_labels = []
        for i in range(n_classes):
            class_center = np.random.randn(n_dims) * 2
            class_samples = (
                class_center + np.random.randn(n_samples // n_classes, n_dims) * noise_level
            )
            current_embeddings_data.append(class_samples)
            current_labels.extend([i] * (n_samples // n_classes))

        current_embeddings_data = np.vstack(current_embeddings_data)
        current_labels = np.array(current_labels)

        current_batch = EmbeddingBatch(
            embeddings=current_embeddings_data,
            ids=[f"day{day}_{i}" for i in range(len(current_embeddings_data))],
            labels=current_labels,
            model_version=f"v1.{day}",
            timestamp=datetime.now() + timedelta(days=day),
        )

        # Compute quality metrics
        print(f"\n{'=' * 60}")
        print(f"Day {day}: Computing quality metrics")
        print(f"{'=' * 60}")

        metrics = monitor.compute_quality_metrics(current_batch)
        print(monitor.generate_quality_report(metrics))

    # Plot quality trends
    print("\nGenerating quality trends plot...")
    monitor.plot_quality_trends()
