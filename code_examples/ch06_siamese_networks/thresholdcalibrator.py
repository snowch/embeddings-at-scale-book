import numpy as np
import torch
import torch.nn.functional as F

# Code from Chapter 06
# Book: Embeddings at Scale

class ThresholdCalibrator:
    """
    Calibrate similarity thresholds for production deployment

    Challenge: The optimal threshold depends on:
    - Distribution of true positives vs negatives
    - Business costs of false positives vs false negatives
    - Dataset characteristics (intra-class vs inter-class variance)

    This class provides multiple calibration strategies.
    """

    def __init__(self, siamese_model):
        self.model = siamese_model
        self.threshold = None
        self.calibration_metrics = {}

    def calibrate_on_validation_set(
        self,
        validation_pairs,
        validation_labels,
        metric='f1',
        plot=False
    ):
        """
        Calibrate threshold on validation set to optimize a metric

        Args:
            validation_pairs: List of (item1, item2) pairs
            validation_labels: 1 if similar, 0 if dissimilar
            metric: 'f1', 'precision', 'recall', or 'accuracy'
            plot: If True, plot threshold vs metric curve

        Returns:
            Optimal threshold value
        """
        # Compute distances for all pairs
        distances = []

        with torch.no_grad():
            self.model.eval()

            for item1, item2 in validation_pairs:
                embedding1 = self.model.get_embedding(item1.unsqueeze(0))
                embedding2 = self.model.get_embedding(item2.unsqueeze(0))

                distance = F.pairwise_distance(embedding1, embedding2).item()
                distances.append(distance)

        distances = np.array(distances)
        validation_labels = np.array(validation_labels)

        # Try different thresholds
        thresholds = np.linspace(distances.min(), distances.max(), 100)
        metrics_by_threshold = []

        for threshold in thresholds:
            # Predict: similar if distance < threshold
            predictions = (distances < threshold).astype(int)

            # Compute metrics
            tp = ((predictions == 1) & (validation_labels == 1)).sum()
            fp = ((predictions == 1) & (validation_labels == 0)).sum()
            tn = ((predictions == 0) & (validation_labels == 0)).sum()
            fn = ((predictions == 0) & (validation_labels == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(validation_labels)

            metrics_by_threshold.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            })

        # Find threshold that maximizes chosen metric
        best_idx = max(
            range(len(metrics_by_threshold)),
            key=lambda i: metrics_by_threshold[i][metric]
        )

        self.threshold = metrics_by_threshold[best_idx]['threshold']
        self.calibration_metrics = metrics_by_threshold[best_idx]

        if plot:
            self._plot_calibration_curve(metrics_by_threshold, metric)

        return self.threshold

    def calibrate_with_business_costs(
        self,
        validation_pairs,
        validation_labels,
        false_positive_cost=1.0,
        false_negative_cost=1.0
    ):
        """
        Calibrate threshold based on business costs

        Args:
            validation_pairs: List of (item1, item2) pairs
            validation_labels: 1 if similar, 0 if dissimilar
            false_positive_cost: Cost of incorrectly marking as similar
            false_negative_cost: Cost of missing a true match

        Returns:
            Cost-optimal threshold

        Example costs:
        - Fraud detection: FN cost >> FP cost (missing fraud is expensive)
        - Product matching: FP cost >> FN cost (wrong matches annoy users)
        """
        # Compute distances
        distances = []

        with torch.no_grad():
            self.model.eval()

            for item1, item2 in validation_pairs:
                embedding1 = self.model.get_embedding(item1.unsqueeze(0))
                embedding2 = self.model.get_embedding(item2.unsqueeze(0))

                distance = F.pairwise_distance(embedding1, embedding2).item()
                distances.append(distance)

        distances = np.array(distances)
        validation_labels = np.array(validation_labels)

        # Try different thresholds
        thresholds = np.linspace(distances.min(), distances.max(), 100)
        costs = []

        for threshold in thresholds:
            predictions = (distances < threshold).astype(int)

            fp = ((predictions == 1) & (validation_labels == 0)).sum()
            fn = ((predictions == 0) & (validation_labels == 1)).sum()

            total_cost = fp * false_positive_cost + fn * false_negative_cost
            costs.append(total_cost)

        # Find threshold that minimizes cost
        best_idx = np.argmin(costs)
        self.threshold = thresholds[best_idx]

        self.calibration_metrics = {
            'threshold': self.threshold,
            'expected_cost': costs[best_idx],
            'false_positive_cost': false_positive_cost,
            'false_negative_cost': false_negative_cost
        }

        return self.threshold

    def calibrate_for_precision_target(
        self,
        validation_pairs,
        validation_labels,
        target_precision=0.95
    ):
        """
        Calibrate to achieve target precision

        Use when false positives are unacceptable (e.g., financial matching)

        Args:
            validation_pairs: List of (item1, item2) pairs
            validation_labels: 1 if similar, 0 if dissimilar
            target_precision: Desired precision (0-1)

        Returns:
            Threshold that achieves target precision (or closest possible)
        """
        # Compute distances
        distances = []

        with torch.no_grad():
            self.model.eval()

            for item1, item2 in validation_pairs:
                embedding1 = self.model.get_embedding(item1.unsqueeze(0))
                embedding2 = self.model.get_embedding(item2.unsqueeze(0))

                distance = F.pairwise_distance(embedding1, embedding2).item()
                distances.append(distance)

        distances = np.array(distances)
        validation_labels = np.array(validation_labels)

        # Try different thresholds
        thresholds = np.linspace(distances.min(), distances.max(), 100)

        best_threshold = None
        best_precision = 0
        best_recall = 0

        for threshold in thresholds:
            predictions = (distances < threshold).astype(int)

            tp = ((predictions == 1) & (validation_labels == 1)).sum()
            fp = ((predictions == 1) & (validation_labels == 0)).sum()
            fn = ((predictions == 0) & (validation_labels == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # Find threshold closest to target precision
            if precision >= target_precision:
                if best_threshold is None or recall > best_recall:
                    best_threshold = threshold
                    best_precision = precision
                    best_recall = recall

        if best_threshold is None:
            # Can't achieve target, return threshold with highest precision
            for threshold in thresholds:
                predictions = (distances < threshold).astype(int)
                tp = ((predictions == 1) & (validation_labels == 1)).sum()
                fp = ((predictions == 1) & (validation_labels == 0)).sum()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0

                if precision > best_precision:
                    best_precision = precision
                    best_threshold = threshold

        self.threshold = best_threshold
        self.calibration_metrics = {
            'threshold': best_threshold,
            'achieved_precision': best_precision,
            'achieved_recall': best_recall,
            'target_precision': target_precision
        }

        return self.threshold

    def _plot_calibration_curve(self, metrics_by_threshold, target_metric):
        """Plot threshold vs metric curve"""
        import matplotlib.pyplot as plt

        thresholds = [m['threshold'] for m in metrics_by_threshold]
        values = [m[target_metric] for m in metrics_by_threshold]

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, values)
        plt.axvline(self.threshold, color='r', linestyle='--',
                   label=f'Optimal: {self.threshold:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel(target_metric.capitalize())
        plt.title(f'Threshold Calibration: {target_metric.capitalize()}')
        plt.legend()
        plt.grid(True)
        plt.show()
