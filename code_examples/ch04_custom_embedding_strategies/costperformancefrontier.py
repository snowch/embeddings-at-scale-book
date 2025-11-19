# Code from Chapter 04
# Book: Embeddings at Scale

class CostPerformanceFrontier:
    """
    Explore cost-performance trade-offs
    """

    def generate_configuration_space(self, requirements):
        """
        Generate configurations spanning cost-performance space
        """
        configs = []

        # Vary key parameters
        dimensions = [128, 256, 384, 512, 768, 1024]
        quantizations = ['float32', 'float16', 'int8', 'binary']
        index_types = ['flat', 'ivf', 'hnsw', 'pq']

        for dim in dimensions:
            for quant in quantizations:
                for index in index_types:
                    config = {
                        'dimension': dim,
                        'quantization': quant,
                        'index_type': index,
                        'num_embeddings': requirements['num_embeddings']
                    }

                    # Estimate cost
                    cost = self.estimate_cost(config)

                    # Estimate performance (latency and quality)
                    performance = self.estimate_performance(config)

                    configs.append({
                        **config,
                        'annual_cost': cost,
                        'p99_latency_ms': performance['latency'],
                        'recall@10': performance['recall']
                    })

        return configs

    def plot_frontier(self, configs):
        """
        Visualize cost-performance Pareto frontier
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Cost vs. Quality
        scatter1 = ax1.scatter(
            [c['annual_cost'] for c in configs],
            [c['recall@10'] for c in configs],
            c=[c['dimension'] for c in configs],
            cmap='viridis',
            s=100,
            alpha=0.6
        )
        ax1.set_xlabel('Annual Cost ($)')
        ax1.set_ylabel('Recall@10')
        ax1.set_title('Cost vs. Quality Trade-off')
        plt.colorbar(scatter1, ax=ax1, label='Dimension')

        # Plot 2: Latency vs. Cost
        scatter2 = ax2.scatter(
            [c['p99_latency_ms'] for c in configs],
            [c['annual_cost'] for c in configs],
            c=[c['recall@10'] for c in configs],
            cmap='coolwarm',
            s=100,
            alpha=0.6
        )
        ax2.set_xlabel('P99 Latency (ms)')
        ax2.set_ylabel('Annual Cost ($)')
        ax2.set_title('Latency vs. Cost Trade-off')
        plt.colorbar(scatter2, ax=ax2, label='Recall@10')

        plt.tight_layout()
        return fig
