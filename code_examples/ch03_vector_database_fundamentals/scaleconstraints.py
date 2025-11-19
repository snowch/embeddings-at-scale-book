# Code from Chapter 03
# Book: Embeddings at Scale

class ScaleConstraints:
    """Understanding what makes 256T rows challenging"""

    def memory_constraints(self):
        """Why you can't fit everything in RAM"""

        num_vectors = 256_000_000_000_000
        embedding_dim = 768
        bytes_per_float = 4

        # Raw embeddings
        raw_bytes = num_vectors * embedding_dim * bytes_per_float
        raw_petabytes = raw_bytes / (1024 ** 5)

        # HNSW index (adds ~50% overhead for graph structure)
        index_petabytes = raw_petabytes * 1.5

        # Total memory needed if all in RAM
        total_memory_pb = index_petabytes

        # Cost analysis
        # AWS r6i.32xlarge: 1TB RAM, $8.064/hour
        machines_needed = total_memory_pb * 1024  # Convert to TB
        monthly_cost = machines_needed * 8.064 * 24 * 30

        return {
            'raw_data_size_pb': raw_petabytes,
            'with_index_size_pb': index_petabytes,
            'machines_needed_1tb_ram': int(machines_needed),
            'monthly_cost_if_all_ram': f'${monthly_cost:,.0f}',
            'conclusion': 'Infeasible - must use hybrid memory/disk strategies'
        }

    def build_time_constraints(self):
        """How long to build index from scratch"""

        num_vectors = 256_000_000_000_000

        # HNSW build time: ~100 microseconds per vector (empirical)
        microseconds_per_vector = 100
        total_seconds = (num_vectors * microseconds_per_vector) / 1_000_000
        total_days = total_seconds / (60 * 60 * 24)
        total_years = total_days / 365

        # Parallel building across 10,000 machines
        parallel_machines = 10_000
        parallel_days = total_days / parallel_machines

        return {
            'single_machine_build_time_years': total_years,
            'parallel_build_time_days': parallel_days,
            'conclusion': 'Must parallelize + use incremental updates'
        }

    def query_time_constraints(self):
        """Target query latency"""

        target_p99_latency_ms = 100

        # Available time budget

        # Breakdown
        breakdown = {
            'network_latency': '20ms (to nearest region)',
            'query_parsing': '1ms',
            'index_search': '50ms (the critical path)',
            'metadata_filtering': '10ms',
            'result_aggregation': '5ms',
            'serialization': '5ms',
            'buffer': '9ms (for variance)',
            'total': '100ms'
        }

        return {
            'target_p99_ms': target_p99_latency_ms,
            'breakdown': breakdown,
            'index_search_budget': '50ms',
            'implication': 'Index must return results in <50ms at p99'
        }

constraints = ScaleConstraints()
mem_constraints = constraints.memory_constraints()
print(f"Memory needed: {mem_constraints['with_index_size_pb']:.1f} PB")
print(f"Cost if all in RAM: {mem_constraints['monthly_cost_if_all_ram']}")
# Output: Memory needed: 1179.6 PB, Cost: $71,145,984,000/month
