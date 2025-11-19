# Code from Chapter 03
# Book: Embeddings at Scale

class ShardingPatterns:
    """Patterns for distributing embeddings across machines"""

    def random_sharding(self, num_vectors, num_shards):
        """Random assignment to shards"""

        return {
            'strategy': 'Hash vector ID % num_shards',
            'pros': [
                'Even distribution (no hot shards)',
                'Simple to implement',
                'Easy to add/remove shards'
            ],
            'cons': [
                'Must query all shards',
                'No data locality'
            ],
            'search_pattern': 'Fan-out to all shards, merge top-k',
            'latency': 'p99 = max(shard_p99) ← limited by slowest shard',
            'use_case': 'Default choice for most applications'
        }

    def learned_sharding(self, num_vectors, num_shards):
        """Cluster vectors, assign clusters to shards"""

        return {
            'strategy': 'K-means cluster, route by nearest centroid',
            'pros': [
                'Data locality (similar vectors on same shard)',
                'Can query subset of shards (only relevant ones)',
                '5-10x faster search (fewer shards queried)'
            ],
            'cons': [
                'Uneven load (some shards more popular)',
                'Requires training and updating cluster assignments',
                'Difficult rebalancing'
            ],
            'search_pattern': [
                '1. Find nearest K cluster centroids',
                '2. Query only shards containing those clusters',
                '3. Merge results'
            ],
            'latency': 'Lower than random (fewer shards)',
            'use_case': 'When queries have locality (e.g., domain-specific searches)'
        }

    def geo_sharding(self, num_vectors, regions):
        """Shard by geography for multi-region deployments"""

        return {
            'strategy': 'Full copy in each region, shard within region',
            'pros': [
                'Low latency (query nearest region)',
                'High availability (region failures isolated)',
                'Regulatory compliance (data residency)'
            ],
            'cons': [
                'Higher storage cost (replication)',
                'Update complexity (sync across regions)',
                'Consistency challenges'
            ],
            'search_pattern': 'Route to nearest region, query within region',
            'latency': 'Lowest (no cross-region latency)',
            'use_case': 'Global applications with regional users'
        }

    def hybrid_sharding(self):
        """Combine multiple strategies"""

        return {
            'strategy': 'Geo-sharding + learned sharding within region',
            'example': [
                'Tier 1: Geographic regions (US, EU, APAC)',
                'Tier 2: Learned clusters within region (64 clusters)',
                'Tier 3: Random sharding within cluster (16 shards per cluster)'
            ],
            'benefit': 'Best of all worlds',
            'search_flow': [
                '1. Route to nearest region (geo)',
                '2. Find relevant clusters (learned)',
                '3. Query shards in parallel (random)',
                '4. Merge and return'
            ],
            'used_by': 'Large-scale production systems (Google, Meta, etc.)'
        }

    def calculate_query_fanout(self, total_shards, strategy):
        """How many shards must be queried?"""

        if strategy == 'random':
            return {
                'shards_queried': total_shards,
                'query_parallelization': 'High',
                'latency_bound': 'Slowest shard (tail latency problem)'
            }

        elif strategy == 'learned':
            # Typically query top-K clusters, K << total clusters
            typical_clusters_queried = min(10, total_shards // 10)
            return {
                'shards_queried': typical_clusters_queried,
                'query_parallelization': 'Medium',
                'latency_bound': 'Slowest of queried shards',
                'speedup_vs_random': f'{total_shards / typical_clusters_queried:.1f}x'
            }

        elif strategy == 'geo':
            shards_per_region = total_shards // 3  # Assume 3 regions
            return {
                'shards_queried': shards_per_region,
                'query_parallelization': 'High within region',
                'latency_bound': 'Regional p99',
                'cross_region': 'No (unless failover)'
            }

# Example: 1000 shards
patterns = ShardingPatterns()
fanout_random = patterns.calculate_query_fanout(1000, 'random')
fanout_learned = patterns.calculate_query_fanout(1000, 'learned')

print(f"Random sharding: Query {fanout_random['shards_queried']} shards")
print(f"Learned sharding: Query {fanout_learned['shards_queried']} shards")
print(f"Speedup: {fanout_learned['speedup_vs_random']}")
