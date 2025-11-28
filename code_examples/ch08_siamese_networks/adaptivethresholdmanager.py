import numpy as np
import torch

# Code from Chapter 06
# Book: Embeddings at Scale


class AdaptiveThresholdManager:
    """
    Manage thresholds that adapt to changing data distributions

    Strategies:
    1. Per-category thresholds: Different thresholds for different data types
    2. Time-based adaptation: Adjust based on recent performance
    3. Confidence-based: Use prediction confidence to adjust threshold
    """

    def __init__(self, base_threshold=0.5):
        self.base_threshold = base_threshold
        self.category_thresholds = {}
        self.performance_history = []

    def get_threshold(self, category=None, confidence=None):
        """
        Get threshold, potentially adjusted for category or confidence

        Args:
            category: Optional category identifier
            confidence: Optional confidence score from model

        Returns:
            Adjusted threshold
        """
        threshold = self.base_threshold

        # Category-specific adjustment
        if category is not None and category in self.category_thresholds:
            threshold = self.category_thresholds[category]

        # Confidence-based adjustment
        # Higher confidence -> lower threshold (more lenient)
        # Lower confidence -> higher threshold (more strict)
        if confidence is not None:
            adjustment = (confidence - 0.5) * 0.2  # ±0.1 adjustment
            threshold = threshold - adjustment

        return threshold

    def update_category_threshold(self, category, new_threshold):
        """Update threshold for a specific category"""
        self.category_thresholds[category] = new_threshold

    def adapt_from_feedback(self, predictions, labels, learning_rate=0.1):
        """
        Adapt thresholds based on recent performance feedback

        Args:
            predictions: Recent prediction distances
            labels: Ground truth labels (1 = similar, 0 = dissimilar)
            learning_rate: How quickly to adapt (0-1)
        """
        # Compute current error rate
        current_predictions = (predictions < self.base_threshold).astype(int)
        error_rate = (current_predictions != labels).mean()

        # If error rate is high, adjust threshold
        if error_rate > 0.2:
            # Compute optimal threshold for recent data
            best_threshold = self._find_optimal_threshold(predictions, labels)

            # Move toward optimal threshold
            self.base_threshold = (
                1 - learning_rate
            ) * self.base_threshold + learning_rate * best_threshold

        # Track performance
        self.performance_history.append(
            {
                "threshold": self.base_threshold,
                "error_rate": error_rate,
                "timestamp": torch.Tensor([0]).item(),  # Use real timestamp in production
            }
        )

    def _find_optimal_threshold(self, distances, labels):
        """Find threshold that minimizes error rate"""
        thresholds = np.linspace(distances.min(), distances.max(), 50)
        errors = []

        for threshold in thresholds:
            predictions = (distances < threshold).astype(int)
            error = (predictions != labels).mean()
            errors.append(error)

        best_idx = np.argmin(errors)
        return thresholds[best_idx]
