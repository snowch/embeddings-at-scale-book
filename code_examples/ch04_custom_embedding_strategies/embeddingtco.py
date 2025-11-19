# Code from Chapter 04
# Book: Embeddings at Scale

class EmbeddingTCO:
    """
    Comprehensive TCO model for embedding systems
    """

    def __init__(self):
        # Cloud pricing (approximate, as of 2024)
        self.storage_cost_per_gb_month = 0.023  # S3 standard
        self.compute_cost_per_hour = 3.0  # A100 GPU
        self.inference_cost_per_million = 10.0  # Vector DB queries

    def calculate_tco(self, config, duration_years=3):
        """
        Calculate total cost of ownership

        Args:
            config: {
                'num_embeddings': 100_000_000_000,
                'embedding_dim': 768,
                'qps': 10_000,
                'training_frequency_per_year': 4,
                'team_size': 10
            }
        """

        # Component 1: Storage
        storage_cost = self.compute_storage_cost(
            config['num_embeddings'],
            config['embedding_dim'],
            duration_years
        )

        # Component 2: Training
        training_cost = self.compute_training_cost(
            config['num_embeddings'],
            config['training_frequency_per_year'],
            duration_years
        )

        # Component 3: Inference
        inference_cost = self.compute_inference_cost(
            config['qps'],
            duration_years
        )

        # Component 4: Engineering team
        team_cost = self.compute_team_cost(
            config['team_size'],
            duration_years
        )

        # Total
        total_cost = (
            storage_cost +
            training_cost +
            inference_cost +
            team_cost
        )

        return {
            'total_cost_3_years': total_cost,
            'annual_cost': total_cost / duration_years,
            'breakdown': {
                'storage': storage_cost,
                'training': training_cost,
                'inference': inference_cost,
                'team': team_cost
            },
            'cost_per_embedding': total_cost / config['num_embeddings'],
            'cost_per_million_queries': inference_cost / (
                config['qps'] * 60 * 60 * 24 * 365 * duration_years / 1_000_000
            )
        }

    def compute_storage_cost(self, num_embeddings, dim, duration_years):
        """Storage cost with replication and indexing overhead"""
        bytes_per_embedding = dim * 4  # float32
        total_bytes = num_embeddings * bytes_per_embedding

        # Index overhead (HNSW adds ~50%)
        indexed_bytes = total_bytes * 1.5

        # Replication (3x for availability)
        replicated_bytes = indexed_bytes * 3

        # Convert to GB
        total_gb = replicated_bytes / (1024 ** 3)

        # Monthly cost
        monthly_cost = total_gb * self.storage_cost_per_gb_month

        # Total over duration
        return monthly_cost * 12 * duration_years

    def optimize_for_budget(self, requirements, budget_annual):
        """
        Given requirements and budget, find optimal configuration
        """
        # Requirements: {'num_embeddings', 'qps', 'min_quality'}
        # Budget: annual spending limit

        # Explore dimension options
        dimensions = [128, 256, 384, 512, 768]
        configs = []

        for dim in dimensions:
            config = {
                'num_embeddings': requirements['num_embeddings'],
                'embedding_dim': dim,
                'qps': requirements['qps'],
                'training_frequency_per_year': 4,
                'team_size': 10
            }

            tco = self.calculate_tco(config, duration_years=1)

            # Estimate quality (simplified)
            quality_score = self.estimate_quality(dim, requirements)

            configs.append({
                'dimension': dim,
                'annual_cost': tco['annual_cost'],
                'quality_score': quality_score,
                'within_budget': tco['annual_cost'] <= budget_annual
            })

        # Filter to budget
        viable = [c for c in configs if c['within_budget']]

        if not viable:
            return {
                'recommendation': 'INSUFFICIENT_BUDGET',
                'message': f"Minimum cost: ${min(c['annual_cost'] for c in configs):,.0f}/year"
            }

        # Choose highest quality within budget
        best = max(viable, key=lambda c: c['quality_score'])

        return {
            'recommendation': 'OPTIMAL_CONFIG',
            'dimension': best['dimension'],
            'annual_cost': best['annual_cost'],
            'quality_score': best['quality_score'],
            'configurations_evaluated': configs
        }
