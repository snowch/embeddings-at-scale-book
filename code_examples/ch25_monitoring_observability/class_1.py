# Code from Chapter 25
# Book: Embeddings at Scale

"""
Comprehensive Embedding Drift Detection and Alerting

Architecture:
1. Baseline establishment: Capture embedding statistics from known-good period
2. Continuous monitoring: Track embedding properties in production
3. Statistical tests: Detect distribution shifts (KS test, MMD, Chi-squared)
4. Semantic tests: Track cluster stability, similarity patterns
5. Business impact: Correlate with downstream metrics (CTR, conversion)
6. Automated alerts: Trigger when drift exceeds thresholds

Drift types:
- Statistical drift: Distribution moments, dimensionality changes
- Semantic drift: Cluster centroid shifts, similarity pattern changes
- Performance drift: Downstream task accuracy degradation
- Business drift: User engagement metrics decrease

Alert conditions:
- Multiple statistical tests show p < 0.01
- Cluster stability < 0.85 correlation with baseline
- Downstream accuracy drops >5% from baseline
- Business metrics drop >10% with embedding changes
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from scipy.stats import ks_2samp, chi2_contingency, entropy
from scipy.spatial.distance import euclidean, cosine, jensenshannon
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import warnings

@dataclass
class DriftSignal:
    """Single drift detection signal"""
    timestamp: datetime
    signal_type: str  # "statistical", "semantic", "performance", "business"
    metric_name: str
    baseline_value: float
    current_value: float
    drift_score: float  # 0-1, higher = more drift
    p_value: Optional[float] = None
    confidence: str = "medium"  # "low", "medium", "high"
    description: str = ""

@dataclass
class DriftAlert:
    """Drift alert with multiple supporting signals"""
    timestamp: datetime
    severity: str  # "warning", "critical"
    signals: List[DriftSignal]
    recommended_action: str
    drift_score: float  # Aggregate drift score
    
class EmbeddingDriftDetector:
    """
    Comprehensive embedding drift detection system
    
    Monitors statistical, semantic, performance, and business metrics
    to detect embedding quality degradation.
    """
    
    def __init__(
        self,
        baseline_embeddings: np.ndarray,
        baseline_labels: Optional[np.ndarray] = None,
        drift_thresholds: Optional[Dict[str, float]] = None,
        alert_callback: Optional[Callable] = None,
        history_window: int = 100
    ):
        """
        Initialize drift detector
        
        Args:
            baseline_embeddings: Reference embeddings from known-good period
            baseline_labels: Optional labels for supervised drift detection
            drift_thresholds: Custom thresholds for drift metrics
            alert_callback: Function to call when drift detected
            history_window: Number of historical checks to retain
        """
        self.baseline_embeddings = baseline_embeddings
        self.baseline_labels = baseline_labels
        self.drift_thresholds = drift_thresholds or self._default_thresholds()
        self.alert_callback = alert_callback
        self.history_window = history_window
        
        # Compute baseline statistics
        self.baseline_stats = self._compute_embedding_statistics(baseline_embeddings)
        self.baseline_clusters = self._compute_cluster_centroids(baseline_embeddings)
        
        # Historical drift signals
        self.drift_history: deque = deque(maxlen=history_window)
        self.alert_history: List[DriftAlert] = []
    
    def _default_thresholds(self) -> Dict[str, float]:
        """Default drift detection thresholds"""
        return {
            "ks_test_p_value": 0.01,  # p < 0.01 indicates drift
            "mmd_threshold": 0.05,     # MMD > 0.05 indicates drift
            "cluster_stability": 0.85,  # Correlation < 0.85 indicates drift
            "mean_shift": 0.1,          # L2 distance of means
            "variance_ratio": 0.8,      # Variance ratio <0.8 or >1.2 indicates drift
            "dimensionality_change": 0.1,  # >10% change in effective dims
            "downstream_accuracy_drop": 0.05,  # >5% accuracy drop
            "business_metric_drop": 0.10  # >10% business metric drop
        }
    
    def _compute_embedding_statistics(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive embedding statistics"""
        stats = {
            "mean": np.mean(embeddings, axis=0),
            "std": np.std(embeddings, axis=0),
            "min": np.min(embeddings, axis=0),
            "max": np.max(embeddings, axis=0),
            "median": np.median(embeddings, axis=0),
            "variance": np.var(embeddings, axis=0),
            "total_variance": np.sum(np.var(embeddings, axis=0)),
            "l2_norm_mean": np.mean(np.linalg.norm(embeddings, axis=1)),
            "l2_norm_std": np.std(np.linalg.norm(embeddings, axis=1))
        }
        
        # Dimension importance
        dim_variance = np.var(embeddings, axis=0)
        total_var = np.sum(dim_variance)
        stats["dim_importance"] = dim_variance / total_var if total_var > 0 else dim_variance
        stats["effective_dims"] = np.sum(stats["dim_importance"] > 0.01)
        
        # PCA for dimensionality analysis
        try:
            pca = PCA(n_components=min(50, embeddings.shape[1]))
            pca.fit(embeddings)
            stats["explained_variance_ratio"] = pca.explained_variance_ratio_
            stats["cumulative_variance"] = np.cumsum(pca.explained_variance_ratio_)
        except:
            pass
        
        return stats
    
    def _compute_cluster_centroids(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 20
    ) -> np.ndarray:
        """Compute cluster centroids for semantic drift detection"""
        n_samples = min(10000, len(embeddings))
        sample_indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sampled_embeddings = embeddings[sample_indices]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
            kmeans.fit(sampled_embeddings)
        
        return kmeans.cluster_centers_
    
    def detect_drift(
        self,
        current_embeddings: np.ndarray,
        current_labels: Optional[np.ndarray] = None,
        downstream_accuracy: Optional[float] = None,
        business_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, List[DriftSignal], Optional[DriftAlert]]:
        """
        Detect drift in current embeddings compared to baseline
        
        Returns:
            (has_drift, signals, alert)
        """
        signals = []
        
        # 1. Statistical drift tests
        signals.extend(self._detect_statistical_drift(current_embeddings))
        
        # 2. Semantic drift tests
        signals.extend(self._detect_semantic_drift(current_embeddings))
        
        # 3. Performance drift tests
        if downstream_accuracy is not None:
            signals.extend(self._detect_performance_drift(downstream_accuracy))
        
        # 4. Business metric drift tests
        if business_metrics is not None:
            signals.extend(self._detect_business_drift(business_metrics))
        
        # Add to history
        for signal in signals:
            self.drift_history.append(signal)
        
        # Determine if alert needed
        has_drift, alert = self._evaluate_drift_signals(signals)
        
        if has_drift and alert and self.alert_callback:
            self.alert_callback(alert)
            self.alert_history.append(alert)
        
        return has_drift, signals, alert
    
    def _detect_statistical_drift(self, current_embeddings: np.ndarray) -> List[DriftSignal]:
        """Detect statistical distribution drift"""
        signals = []
        current_stats = self._compute_embedding_statistics(current_embeddings)
        
        # 1. Kolmogorov-Smirnov test on dimension distributions
        n_dims = min(current_embeddings.shape[1], self.baseline_embeddings.shape[1])
        ks_p_values = []
        
        for dim in range(min(50, n_dims)):  # Sample dimensions to avoid excessive computation
            dim_idx = dim * (n_dims // 50) if n_dims > 50 else dim
            ks_stat, p_value = ks_2samp(
                self.baseline_embeddings[:, dim_idx],
                current_embeddings[:, dim_idx]
            )
            ks_p_values.append(p_value)
        
        mean_ks_p = np.mean(ks_p_values)
        min_ks_p = np.min(ks_p_values)
        
        if min_ks_p < self.drift_thresholds["ks_test_p_value"]:
            signals.append(DriftSignal(
                timestamp=datetime.now(),
                signal_type="statistical",
                metric_name="ks_test",
                baseline_value=1.0,  # p=1.0 means no drift
                current_value=min_ks_p,
                drift_score=1.0 - min_ks_p,
                p_value=min_ks_p,
                confidence="high",
                description=f"KS test detected distribution shift (p={min_ks_p:.4f})"
            ))
        
        # 2. Mean shift
        mean_distance = euclidean(
            self.baseline_stats["mean"],
            current_stats["mean"]
        )
        baseline_mean_norm = np.linalg.norm(self.baseline_stats["mean"])
        normalized_mean_shift = mean_distance / max(baseline_mean_norm, 1e-6)
        
        if normalized_mean_shift > self.drift_thresholds["mean_shift"]:
            signals.append(DriftSignal(
                timestamp=datetime.now(),
                signal_type="statistical",
                metric_name="mean_shift",
                baseline_value=0.0,
                current_value=normalized_mean_shift,
                drift_score=min(1.0, normalized_mean_shift / self.drift_thresholds["mean_shift"]),
                confidence="high",
                description=f"Embedding mean shifted by {normalized_mean_shift:.3f}"
            ))
        
        # 3. Variance ratio
        variance_ratio = current_stats["total_variance"] / self.baseline_stats["total_variance"]
        
        if variance_ratio < self.drift_thresholds["variance_ratio"] or variance_ratio > (1 / self.drift_thresholds["variance_ratio"]):
            signals.append(DriftSignal(
                timestamp=datetime.now(),
                signal_type="statistical",
                metric_name="variance_ratio",
                baseline_value=1.0,
                current_value=variance_ratio,
                drift_score=abs(1.0 - variance_ratio),
                confidence="medium",
                description=f"Embedding variance changed by {(variance_ratio - 1) * 100:.1f}%"
            ))
        
        # 4. Dimensionality change
        dim_change_ratio = abs(current_stats["effective_dims"] - self.baseline_stats["effective_dims"]) / self.baseline_stats["effective_dims"]
        
        if dim_change_ratio > self.drift_thresholds["dimensionality_change"]:
            signals.append(DriftSignal(
                timestamp=datetime.now(),
                signal_type="statistical",
                metric_name="dimensionality_change",
                baseline_value=self.baseline_stats["effective_dims"],
                current_value=current_stats["effective_dims"],
                drift_score=dim_change_ratio,
                confidence="medium",
                description=f"Effective dimensions changed from {self.baseline_stats['effective_dims']} to {current_stats['effective_dims']}"
            ))
        
        # 5. Jensen-Shannon divergence on dimension importance
        try:
            js_div = jensenshannon(
                self.baseline_stats["dim_importance"] + 1e-10,
                current_stats["dim_importance"] + 1e-10
            )
            
            if js_div > 0.1:  # Threshold for JS divergence
                signals.append(DriftSignal(
                    timestamp=datetime.now(),
                    signal_type="statistical",
                    metric_name="dimension_importance_shift",
                    baseline_value=0.0,
                    current_value=js_div,
                    drift_score=min(1.0, js_div / 0.3),  # Normalize to 0-1
                    confidence="medium",
                    description=f"Dimension importance distribution shifted (JS={js_div:.3f})"
                ))
        except:
            pass
        
        return signals
    
    def _detect_semantic_drift(self, current_embeddings: np.ndarray) -> List[DriftSignal]:
        """Detect semantic/cluster structure drift"""
        signals = []
        
        # Compute current cluster centroids
        current_clusters = self._compute_cluster_centroids(current_embeddings)
        
        # Align clusters using Hungarian algorithm (optimal matching)
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist
        
        # Compute distance matrix between baseline and current clusters
        dist_matrix = cdist(self.baseline_clusters, current_clusters, metric='cosine')
        
        # Find optimal alignment
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        # Compute cluster stability (average similarity of aligned clusters)
        aligned_similarities = [1 - dist_matrix[i, j] for i, j in zip(row_ind, col_ind)]
        cluster_stability = np.mean(aligned_similarities)
        
        if cluster_stability < self.drift_thresholds["cluster_stability"]:
            signals.append(DriftSignal(
                timestamp=datetime.now(),
                signal_type="semantic",
                metric_name="cluster_stability",
                baseline_value=1.0,
                current_value=cluster_stability,
                drift_score=1.0 - cluster_stability,
                confidence="high",
                description=f"Cluster structure shifted (stability={cluster_stability:.3f})"
            ))
        
        # Compute intra-cluster vs inter-cluster similarity
        # This detects if clusters are becoming more or less separated
        
        return signals
    
    def _detect_performance_drift(self, current_accuracy: float) -> List[DriftSignal]:
        """Detect performance drift on downstream tasks"""
        signals = []
        
        # Assume baseline accuracy is stored or provided
        # In practice, this should be tracked from baseline period
        baseline_accuracy = getattr(self, 'baseline_accuracy', 0.90)
        
        accuracy_drop = baseline_accuracy - current_accuracy
        
        if accuracy_drop > self.drift_thresholds["downstream_accuracy_drop"]:
            signals.append(DriftSignal(
                timestamp=datetime.now(),
                signal_type="performance",
                metric_name="downstream_accuracy",
                baseline_value=baseline_accuracy,
                current_value=current_accuracy,
                drift_score=accuracy_drop / self.drift_thresholds["downstream_accuracy_drop"],
                confidence="high",
                description=f"Downstream accuracy dropped {accuracy_drop*100:.1f}% ({baseline_accuracy:.3f} → {current_accuracy:.3f})"
            ))
        
        return signals
    
    def _detect_business_drift(self, business_metrics: Dict[str, float]) -> List[DriftSignal]:
        """Detect business metric degradation"""
        signals = []
        
        # Assume baseline metrics are stored
        # In practice, these should be tracked from baseline period
        baseline_metrics = getattr(self, 'baseline_business_metrics', {})
        
        for metric_name, current_value in business_metrics.items():
            if metric_name not in baseline_metrics:
                continue
            
            baseline_value = baseline_metrics[metric_name]
            relative_drop = (baseline_value - current_value) / baseline_value
            
            if relative_drop > self.drift_thresholds["business_metric_drop"]:
                signals.append(DriftSignal(
                    timestamp=datetime.now(),
                    signal_type="business",
                    metric_name=metric_name,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    drift_score=relative_drop / self.drift_thresholds["business_metric_drop"],
                    confidence="high",
                    description=f"{metric_name} dropped {relative_drop*100:.1f}% ({baseline_value:.3f} → {current_value:.3f})"
                ))
        
        return signals
    
    def _evaluate_drift_signals(
        self,
        signals: List[DriftSignal]
    ) -> Tuple[bool, Optional[DriftAlert]]:
        """
        Evaluate drift signals and determine if alert needed
        
        Returns:
            (has_drift, alert)
        """
        if not signals:
            return False, None
        
        # Count high-confidence signals by type
        signal_counts = defaultdict(int)
        high_confidence_signals = []
        
        for signal in signals:
            if signal.confidence == "high":
                signal_counts[signal.signal_type] += 1
                high_confidence_signals.append(signal)
        
        # Aggregate drift score
        aggregate_drift = np.mean([s.drift_score for s in high_confidence_signals]) if high_confidence_signals else 0.0
        
        # Alert conditions:
        # - Multiple statistical signals OR
        # - Any semantic signal with statistical support OR
        # - Performance/business signals
        has_drift = (
            signal_counts["statistical"] >= 2 or
            (signal_counts["semantic"] >= 1 and signal_counts["statistical"] >= 1) or
            signal_counts["performance"] >= 1 or
            signal_counts["business"] >= 1
        )
        
        if not has_drift:
            return False, None
        
        # Determine severity
        if signal_counts["performance"] >= 1 or signal_counts["business"] >= 1 or aggregate_drift > 0.5:
            severity = "critical"
            recommended_action = "Immediate model retraining or rollback recommended"
        else:
            severity = "warning"
            recommended_action = "Monitor closely, prepare for retraining"
        
        alert = DriftAlert(
            timestamp=datetime.now(),
            severity=severity,
            signals=high_confidence_signals,
            recommended_action=recommended_action,
            drift_score=aggregate_drift
        )
        
        return True, alert
    
    def generate_drift_report(self, signals: List[DriftSignal], alert: Optional[DriftAlert] = None) -> str:
        """Generate human-readable drift report"""
        report = f"""
Embedding Drift Detection Report
=================================
Timestamp: {datetime.now().isoformat()}

"""
        
        if alert:
            report += f"""
⚠️ DRIFT ALERT - {alert.severity.upper()}
Aggregate Drift Score: {alert.drift_score:.3f}
Recommended Action: {alert.recommended_action}

"""
        
        if signals:
            report += "Detected Drift Signals:\n"
            report += "-" * 60 + "\n"
            
            for signal in signals:
                report += f"""
{signal.signal_type.upper()}: {signal.metric_name}
  Baseline: {signal.baseline_value:.4f}
  Current: {signal.current_value:.4f}
  Drift Score: {signal.drift_score:.3f}
  Confidence: {signal.confidence}
  {signal.description}
"""
        else:
            report += "✓ No drift detected\n"
        
        return report


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate baseline embeddings
    n_samples = 5000
    n_dims = 768
    baseline = np.random.randn(n_samples, n_dims) * 0.5
    
    # Initialize detector
    def alert_handler(alert: DriftAlert):
        print(f"\n🚨 DRIFT ALERT - {alert.severity.upper()}")
        print(f"Drift Score: {alert.drift_score:.3f}")
        print(f"Action: {alert.recommended_action}")
        print(f"Signals: {len(alert.signals)}")
    
    detector = EmbeddingDriftDetector(
        baseline_embeddings=baseline,
        alert_callback=alert_handler
    )
    detector.baseline_accuracy = 0.90
    detector.baseline_business_metrics = {"ctr": 0.15, "conversion": 0.05}
    
    # Test 1: No drift
    print("\nTest 1: No drift (similar distribution)")
    current_no_drift = baseline + np.random.randn(n_samples, n_dims) * 0.1
    has_drift, signals, alert = detector.detect_drift(current_no_drift)
    print(detector.generate_drift_report(signals, alert))
    
    # Test 2: Mean shift
    print("\nTest 2: Mean shift drift")
    current_mean_shift = baseline + 1.0  # Significant mean shift
    has_drift, signals, alert = detector.detect_drift(current_mean_shift)
    print(detector.generate_drift_report(signals, alert))
    
    # Test 3: Variance change
    print("\nTest 3: Variance change drift")
    current_variance_change = baseline * 2.0  # Double the variance
    has_drift, signals, alert = detector.detect_drift(current_variance_change)
    print(detector.generate_drift_report(signals, alert))
    
    # Test 4: Performance degradation
    print("\nTest 4: Performance degradation")
    current_perf_drop = baseline + np.random.randn(n_samples, n_dims) * 0.1
    has_drift, signals, alert = detector.detect_drift(
        current_perf_drop,
        downstream_accuracy=0.82,  # 8% drop
        business_metrics={"ctr": 0.12, "conversion": 0.04}  # 20% drops
    )
    print(detector.generate_drift_report(signals, alert))
