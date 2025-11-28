import pandas as pd

# Code from Chapter 04
# Book: Embeddings at Scale


class DimensionalityExperiment:
    """
    Systematically evaluate different embedding dimensions
    """

    def run_dimensionality_sweep(self, train_data, test_data, dimensions=None):
        """
        Train models at different dimensions and evaluate
        """
        if dimensions is None:
            dimensions = [128, 256, 384, 512, 768]
        results = []

        for dim in dimensions:
            print(f"\nTraining {dim}-dimensional model...")

            # Train model
            model = self.train_model(train_data, embedding_dim=dim)

            # Evaluate
            metrics = self.evaluate_model(model, test_data)

            # Measure costs
            storage_gb = self.estimate_storage(dim, num_embeddings=100_000_000)
            latency_ms = self.measure_latency(model)

            results.append(
                {
                    "dimension": dim,
                    "recall@10": metrics["recall@10"],
                    "mrr": metrics["mrr"],
                    "storage_gb": storage_gb,
                    "p99_latency_ms": latency_ms,
                    "cost_per_1m_queries": self.estimate_query_cost(dim),
                }
            )

        return pd.DataFrame(results)

    def find_optimal_dimension(self, results, quality_threshold=0.95):
        """
        Find smallest dimension meeting quality threshold

        Args:
            results: DataFrame from dimensionality sweep
            quality_threshold: Minimum acceptable quality (relative to best)

        Returns:
            Optimal dimension
        """
        # Normalize quality metrics to [0, 1]
        max_recall = results["recall@10"].max()
        results["normalized_quality"] = results["recall@10"] / max_recall

        # Filter to dimensions meeting quality threshold
        acceptable = results[results["normalized_quality"] >= quality_threshold]

        if acceptable.empty:
            return results.loc[results["recall@10"].idxmax(), "dimension"]

        # Among acceptable dimensions, choose smallest (cheapest)
        optimal_dim = acceptable.loc[acceptable["dimension"].idxmin(), "dimension"]

        return optimal_dim


# Example results:
# | Dimension | Recall@10 | Storage | Latency | Quality (normalized) |
# |-----------|-----------|---------|---------|---------------------|
# | 128       | 0.834     | 48 GB   | 12 ms   | 0.909               |
# | 256       | 0.891     | 96 GB   | 18 ms   | 0.972               |
# | 384       | 0.908     | 144 GB  | 25 ms   | 0.991               |
# | 512       | 0.915     | 192 GB  | 32 ms   | 0.998               |
# | 768       | 0.917     | 288 GB  | 45 ms   | 1.000               |
#
# Conclusion: 384 dimensions optimal
# - Achieves 99.1% of maximum quality
# - 25% cheaper than 512-dim
# - 50% cheaper than 768-dim
