# Code from Chapter 05
# Book: Embeddings at Scale

import matplotlib.pyplot as plt
import numpy as np


class TemperatureAnalysis:
    """
    Analyze impact of temperature on contrastive learning
    """

    def demonstrate_temperature_effect(self):
        """
        Show how temperature affects the probability distribution
        """
        # Simulate similarities: 1 positive, 9 negatives
        # Positive is much more similar (0.8) than negatives (0.1-0.3)
        similarities = np.array([0.8, 0.3, 0.25, 0.2, 0.15, 0.15, 0.12, 0.1, 0.1, 0.1])

        temperatures = [0.01, 0.05, 0.07, 0.1, 0.3, 0.5]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, tau in enumerate(temperatures):
            # Compute probabilities with temperature
            logits = similarities / tau
            probs = np.exp(logits) / np.exp(logits).sum()

            # Plot
            ax = axes[idx]
            colors = ["green"] + ["red"] * 9  # Positive green, negatives red
            ax.bar(range(10), probs, color=colors, alpha=0.6)
            ax.set_title(f"Temperature τ={tau}\nP(positive)={probs[0]:.3f}")
            ax.set_xlabel("Example index")
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)

        plt.tight_layout()
        return fig

    def recommend_temperature(self, batch_size, data_quality="high"):
        """
        Recommend temperature based on batch size and data quality

        Args:
            batch_size: Training batch size
            data_quality: 'high', 'medium', or 'low' (refers to negative quality)

        Returns:
            recommended temperature and rationale
        """
        if batch_size >= 4096:
            # Very large batches: many high-quality negatives available
            if data_quality == "high":
                return 0.03, "Large batch + high quality → very low temperature for hard negatives"
            else:
                return 0.05, "Large batch but lower quality → slightly higher temperature"

        elif batch_size >= 1024:
            # Large batches: good number of negatives
            if data_quality == "high":
                return 0.05, "Large batch + high quality → low temperature"
            else:
                return 0.07, "Standard setting for large batches"

        elif batch_size >= 256:
            # Medium batches: standard setting
            return 0.07, "Standard temperature for medium batches"

        elif batch_size >= 64:
            # Small batches: need softer distribution
            if data_quality == "low":
                return 0.15, "Small batch + noisy data → higher temperature for stability"
            else:
                return 0.1, "Small batch → moderately high temperature"

        else:
            # Very small batches: high temperature required
            return 0.2, "Very small batch → high temperature to utilize all negatives"


# Example usage
analyzer = TemperatureAnalysis()

# Get recommendation for your setup
batch_size = 512
data_quality = "high"

temp, reasoning = analyzer.recommend_temperature(batch_size, data_quality)
print(f"Recommended temperature: {temp}")
print(f"Reasoning: {reasoning}")
