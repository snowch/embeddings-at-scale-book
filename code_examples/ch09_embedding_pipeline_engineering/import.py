# Code from Chapter 09
# Book: Embeddings at Scale

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EmbeddingQualityMetrics:
    """
    Metrics for monitoring embedding quality

    Intrinsic metrics (computed from embeddings themselves):
    - Average embedding norm
    - Embedding variance
    - Nearest neighbor distance distribution
    - Cluster cohesion

    Extrinsic metrics (task performance):
    - Retrieval recall@k
    - Ranking metrics (NDCG, MRR)
    - Downstream task accuracy (if available)
    - User engagement metrics (CTR, conversion)
    """
    timestamp: datetime

    # Intrinsic metrics
    avg_norm: float
    norm_std: float
    avg_nn_distance: float  # Average distance to nearest neighbor
    embedding_variance: float

    # Extrinsic metrics
    retrieval_recall_at_10: float
    retrieval_recall_at_100: float
    ndcg_at_10: float
    mrr: float  # Mean reciprocal rank

    # System metrics
    inference_latency_p50_ms: float
    inference_latency_p99_ms: float
    index_size_gb: float
    queries_per_second: float

class EmbeddingMonitoringSystem:
    """
    Continuous monitoring system for embedding quality

    Responsibilities:
    - Periodic quality evaluation (hourly/daily)
    - Drift detection (statistical tests on metric distributions)
    - Alerting on quality degradation
    - Automatic retraining triggers
    - Historical metric tracking

    Alert conditions:
    - Recall drops >5% from baseline
    - Latency increases >20% from baseline
    - Embedding distribution shifts significantly (KL divergence)
    - User engagement metrics decline
    """

    def __init__(
        self,
        model_registry,
        test_dataset,
        alert_thresholds: Optional[Dict] = None
    ):
        """
        Args:
            model_registry: Access to embedding models
            test_dataset: Fixed test set for consistent evaluation
            alert_thresholds: Thresholds for triggering alerts
        """
        self.model_registry = model_registry
        self.test_dataset = test_dataset

        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'recall_at_10_drop': 0.05,  # Alert if recall drops >5%
            'latency_p99_increase': 0.20,  # Alert if p99 latency increases >20%
            'embedding_norm_change': 0.15,  # Alert if avg norm changes >15%
            'queries_per_second_drop': 0.30  # Alert if QPS drops >30%
        }

        # Historical metrics for baseline comparison
        self.historical_metrics: List[EmbeddingQualityMetrics] = []

        # Baseline metrics (from initial deployment)
        self.baseline_metrics: Optional[EmbeddingQualityMetrics] = None

    def evaluate_current_quality(
        self,
        model_id: str,
        sample_size: int = 10000
    ) -> EmbeddingQualityMetrics:
        """
        Evaluate current embedding quality

        Runs comprehensive evaluation:
        - Sample embeddings from index
        - Compute intrinsic metrics
        - Run retrieval evaluation on test set
        - Measure system performance
        """
        print(f"Evaluating embedding quality for {model_id}...")

        # Load model
        model, metadata = self.model_registry.load_model(model_id, 'cuda')
        model.eval()

        # Compute intrinsic metrics
        intrinsic = self._compute_intrinsic_metrics(model, sample_size)

        # Compute extrinsic metrics (retrieval evaluation)
        extrinsic = self._compute_extrinsic_metrics(model)

        # Measure system performance
        system = self._measure_system_performance(model)

        # Combine into quality metrics
        metrics = EmbeddingQualityMetrics(
            timestamp=datetime.now(),
            avg_norm=intrinsic['avg_norm'],
            norm_std=intrinsic['norm_std'],
            avg_nn_distance=intrinsic['avg_nn_distance'],
            embedding_variance=intrinsic['embedding_variance'],
            retrieval_recall_at_10=extrinsic['recall_at_10'],
            retrieval_recall_at_100=extrinsic['recall_at_100'],
            ndcg_at_10=extrinsic['ndcg_at_10'],
            mrr=extrinsic['mrr'],
            inference_latency_p50_ms=system['latency_p50_ms'],
            inference_latency_p99_ms=system['latency_p99_ms'],
            index_size_gb=system['index_size_gb'],
            queries_per_second=system['queries_per_second']
        )

        return metrics

    def _compute_intrinsic_metrics(
        self,
        model: nn.Module,
        sample_size: int
    ) -> Dict:
        """
        Compute intrinsic embedding metrics

        Intrinsic metrics measure properties of embedding space itself:
        - Norm statistics: Are embeddings well-scaled?
        - Variance: Is embedding space well-utilized?
        - Nearest neighbor distances: Are similar items close?
        """
        # Sample random inputs
        sample_data = torch.randn(sample_size, model.encoder[0].in_features).to('cuda')

        with torch.no_grad():
            embeddings = model(sample_data).cpu().numpy()

        # Compute norms
        norms = np.linalg.norm(embeddings, axis=1)
        avg_norm = np.mean(norms)
        norm_std = np.std(norms)

        # Compute variance (measure of space utilization)
        embedding_variance = np.mean(np.var(embeddings, axis=0))

        # Nearest neighbor distances (sample 1000 points)
        nn_distances = []
        sample_indices = np.random.choice(len(embeddings), size=min(1000, len(embeddings)), replace=False)

        for idx in sample_indices[:100]:  # Limit for efficiency
            query_emb = embeddings[idx]
            distances = np.linalg.norm(embeddings - query_emb, axis=1)
            distances = np.sort(distances)
            nn_distances.append(distances[1])  # Distance to nearest neighbor (exclude self)

        avg_nn_distance = np.mean(nn_distances)

        return {
            'avg_norm': float(avg_norm),
            'norm_std': float(norm_std),
            'embedding_variance': float(embedding_variance),
            'avg_nn_distance': float(avg_nn_distance)
        }

    def _compute_extrinsic_metrics(self, model: nn.Module) -> Dict:
        """
        Compute extrinsic metrics (task performance)

        Extrinsic metrics measure how well embeddings perform on real tasks:
        - Retrieval: Given query, find relevant documents
        - Ranking: Order results by relevance
        - Classification: If embeddings used for downstream task
        """
        # Run retrieval evaluation on test set
        # (Simplified - real implementation uses full evaluation framework)

        recall_at_10 = 0.89  # Placeholder
        recall_at_100 = 0.95  # Placeholder
        ndcg_at_10 = 0.85  # Placeholder
        mrr = 0.78  # Placeholder

        return {
            'recall_at_10': recall_at_10,
            'recall_at_100': recall_at_100,
            'ndcg_at_10': ndcg_at_10,
            'mrr': mrr
        }

    def _measure_system_performance(self, model: nn.Module) -> Dict:
        """
        Measure system performance metrics

        System metrics track operational health:
        - Latency: How fast are embeddings generated?
        - Throughput: How many queries per second?
        - Resource usage: Index size, memory, compute
        """
        # Run latency benchmark
        num_queries = 1000
        latencies_ms = []

        sample_data = torch.randn(1, model.encoder[0].in_features).to('cuda')

        for _ in range(num_queries):
            start = datetime.now()
            with torch.no_grad():
                _ = model(sample_data)
            latency = (datetime.now() - start).total_seconds() * 1000
            latencies_ms.append(latency)

        latency_p50 = np.percentile(latencies_ms, 50)
        latency_p99 = np.percentile(latencies_ms, 99)

        # Measure throughput
        queries_per_second = 1000 / np.mean(latencies_ms) if np.mean(latencies_ms) > 0 else 0

        # Estimate index size (simplified)
        index_size_gb = 10.5  # Placeholder

        return {
            'latency_p50_ms': latency_p50,
            'latency_p99_ms': latency_p99,
            'queries_per_second': queries_per_second,
            'index_size_gb': index_size_gb
        }

    def detect_drift(
        self,
        current_metrics: EmbeddingQualityMetrics
    ) -> Dict[str, bool]:
        """
        Detect drift in embedding quality

        Compares current metrics to:
        - Baseline (from initial deployment)
        - Recent history (rolling window)

        Returns alerts for each metric that exceeded threshold
        """
        if self.baseline_metrics is None:
            # First evaluation - set as baseline
            self.baseline_metrics = current_metrics
            print("✓ Baseline metrics established")
            return {}

        alerts = {}

        # Check recall drift
        recall_drop = (
            self.baseline_metrics.retrieval_recall_at_10 -
            current_metrics.retrieval_recall_at_10
        ) / self.baseline_metrics.retrieval_recall_at_10

        if recall_drop > self.alert_thresholds['recall_at_10_drop']:
            alerts['recall_degradation'] = True
            print(f"⚠️  ALERT: Recall dropped {recall_drop:.1%} from baseline")

        # Check latency drift
        latency_increase = (
            current_metrics.inference_latency_p99_ms -
            self.baseline_metrics.inference_latency_p99_ms
        ) / self.baseline_metrics.inference_latency_p99_ms

        if latency_increase > self.alert_thresholds['latency_p99_increase']:
            alerts['latency_increase'] = True
            print(f"⚠️  ALERT: P99 latency increased {latency_increase:.1%}")

        # Check embedding distribution drift
        norm_change = abs(
            current_metrics.avg_norm - self.baseline_metrics.avg_norm
        ) / self.baseline_metrics.avg_norm

        if norm_change > self.alert_thresholds['embedding_norm_change']:
            alerts['embedding_distribution_shift'] = True
            print(f"⚠️  ALERT: Embedding norm changed {norm_change:.1%}")

        # Check throughput degradation
        qps_drop = (
            self.baseline_metrics.queries_per_second -
            current_metrics.queries_per_second
        ) / self.baseline_metrics.queries_per_second

        if qps_drop > self.alert_thresholds['queries_per_second_drop']:
            alerts['throughput_degradation'] = True
            print(f"⚠️  ALERT: Throughput dropped {qps_drop:.1%}")

        if not alerts:
            print("✓ No drift detected - quality stable")

        return alerts

    def should_retrain(
        self,
        alerts: Dict[str, bool],
        days_since_training: int
    ) -> Tuple[bool, str]:
        """
        Decide whether to trigger model retraining

        Retraining triggers:
        - Quality degradation alerts
        - Staleness (>30 days since last training)
        - Significant corpus growth (>20% new items)

        Returns:
            (should_retrain, reason)
        """
        # Quality-based trigger
        critical_alerts = [
            'recall_degradation',
            'embedding_distribution_shift'
        ]

        if any(alerts.get(alert, False) for alert in critical_alerts):
            return True, "quality_degradation"

        # Time-based trigger
        if days_since_training > 30:
            return True, "model_staleness"

        # No retraining needed
        return False, ""

    def continuous_monitoring_loop(
        self,
        model_id: str,
        check_interval_hours: int = 24
    ):
        """
        Continuous monitoring loop (runs as background service)

        Args:
            model_id: Model to monitor
            check_interval_hours: How often to evaluate
        """
        print(f"Starting continuous monitoring for {model_id}")
        print(f"Check interval: every {check_interval_hours} hours")

        while True:
            # Evaluate quality
            current_metrics = self.evaluate_current_quality(model_id)

            # Store in history
            self.historical_metrics.append(current_metrics)

            # Detect drift
            alerts = self.detect_drift(current_metrics)

            # Check retraining trigger
            days_since_training = 15  # Placeholder (get from model metadata)
            should_retrain, reason = self.should_retrain(alerts, days_since_training)

            if should_retrain:
                print(f"\n🔄 Triggering model retraining: {reason}")
                # In production: Trigger retraining pipeline
                break

            # Wait for next check
            print(f"\nNext check in {check_interval_hours} hours...")
            # In production: time.sleep(check_interval_hours * 3600)
            break  # For example purposes

# Example: Monitor production embeddings
def production_monitoring_example():
    """
    Monitor product embeddings in production

    Checks:
    - Daily quality evaluation
    - Drift detection
    - Automatic retraining trigger
    """
    # Initialize monitoring system
    registry = EmbeddingModelRegistry()
    test_dataset = None  # In production: load test set

    monitor = EmbeddingMonitoringSystem(
        model_registry=registry,
        test_dataset=test_dataset
    )

    # Day 1: Initial deployment
    print("=== Day 1: Initial Deployment ===")
    model_id = "product-embeddings-v1.0.0"
    metrics_day1 = monitor.evaluate_current_quality(model_id)
    alerts_day1 = monitor.detect_drift(metrics_day1)

    # Day 30: Quality check
    print("\n=== Day 30: Regular Quality Check ===")
    metrics_day30 = monitor.evaluate_current_quality(model_id)

    # Simulate quality degradation
    metrics_day30.retrieval_recall_at_10 = 0.82  # Degraded from 0.89

    alerts_day30 = monitor.detect_drift(metrics_day30)

    # Check retraining trigger
    should_retrain, reason = monitor.should_retrain(alerts_day30, days_since_training=30)

    if should_retrain:
        print(f"\n✓ Retraining triggered: {reason}")
    else:
        print("\n✓ No retraining needed")

# Uncomment to run:
# production_monitoring_example()
