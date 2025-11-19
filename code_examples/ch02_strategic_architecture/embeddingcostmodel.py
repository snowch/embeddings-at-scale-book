# Code from Chapter 02
# Book: Embeddings at Scale

class EmbeddingCostModel:
    """Model total cost of ownership for embeddings"""

    def calculate_tco(self, num_embeddings, embedding_dim, qps, duration_years=3):
        """Calculate total cost of ownership"""

        # 1. Storage costs
        storage_costs = self.calculate_storage_costs(num_embeddings, embedding_dim)

        # 2. Compute costs (training + inference)
        training_costs = self.calculate_training_costs(num_embeddings, embedding_dim)
        inference_costs = self.calculate_inference_costs(qps, duration_years)

        # 3. Data transfer costs
        transfer_costs = self.calculate_transfer_costs(qps, duration_years)

        # 4. Operations costs (monitoring, maintenance)
        ops_costs = self.calculate_ops_costs(num_embeddings, duration_years)

        # 5. Team costs
        team_costs = self.calculate_team_costs(num_embeddings, duration_years)

        total_cost = sum([
            storage_costs['total'],
            training_costs['total'],
            inference_costs['total'],
            transfer_costs['total'],
            ops_costs['total'],
            team_costs['total']
        ])

        return {
            'total_cost_3_years': total_cost,
            'annual_cost': total_cost / duration_years,
            'cost_per_embedding': total_cost / num_embeddings,
            'breakdown': {
                'storage': storage_costs,
                'training': training_costs,
                'inference': inference_costs,
                'transfer': transfer_costs,
                'ops': ops_costs,
                'team': team_costs
            }
        }

    def calculate_storage_costs(self, num_embeddings, embedding_dim):
        """Calculate storage costs"""
        bytes_per_embedding = embedding_dim * 4  # float32
        total_bytes = num_embeddings * bytes_per_embedding

        # Raw storage
        raw_storage_tb = total_bytes / (1024 ** 4)

        # Index overhead (HNSW adds ~50%)
        index_storage_tb = raw_storage_tb * 1.5

        # Replication (3x for availability)
        replicated_storage_tb = index_storage_tb * 3

        # Cost (S3-like object storage: $0.023/GB/month)
        monthly_cost = replicated_storage_tb * 1024 * 0.023

        return {
            'storage_tb': replicated_storage_tb,
            'monthly_cost': monthly_cost,
            'annual_cost': monthly_cost * 12,
            'total': monthly_cost * 12 * 3  # 3 years
        }

    def calculate_training_costs(self, num_embeddings, embedding_dim):
        """Calculate training costs"""
        # Training frequency
        retrains_per_year = 4  # Quarterly retraining

        # Compute hours per training run
        # Rough estimate: 1M embeddings = 10 GPU hours
        gpu_hours_per_run = (num_embeddings / 1_000_000) * 10

        # GPU cost (A100 on cloud: ~$3/hour)
        cost_per_run = gpu_hours_per_run * 3

        # Annual cost
        annual_cost = cost_per_run * retrains_per_year

        return {
            'cost_per_training_run': cost_per_run,
            'training_runs_per_year': retrains_per_year,
            'annual_cost': annual_cost,
            'total': annual_cost * 3  # 3 years
        }

    def calculate_inference_costs(self, qps, duration_years):
        """Calculate inference (query) costs"""
        # Queries per year
        queries_per_year = qps * 60 * 60 * 24 * 365

        # Compute cost per million queries
        # Vector DB on cloud: ~$10 per million queries
        cost_per_million = 10

        annual_cost = (queries_per_year / 1_000_000) * cost_per_million

        return {
            'qps': qps,
            'queries_per_year': queries_per_year,
            'annual_cost': annual_cost,
            'total': annual_cost * duration_years
        }

# Example: 100B embeddings, 768-dim, 10K QPS
model = EmbeddingCostModel()
tco = model.calculate_tco(
    num_embeddings=100_000_000_000,  # 100B
    embedding_dim=768,
    qps=10_000,
    duration_years=3
)

print(f"Total 3-year cost: ${tco['total_cost_3_years']:,.0f}")
print(f"Annual cost: ${tco['annual_cost']:,.0f}")
print(f"Cost per embedding: ${tco['cost_per_embedding']:.6f}")
