# Code from Chapter 03
# Book: Embeddings at Scale

class GlobalDistributionArchitecture:
    """Patterns for deploying vector databases globally"""

    def architecture_patterns(self):
        """Different global deployment patterns"""

        return {
            'full_replication': {
                'description': 'Complete copy of all embeddings in each region',

                'architecture': {
                    'regions': ['US-West', 'US-East', 'EU-West', 'Asia-Pacific', 'Latin America'],
                    'data': 'Full 256T embeddings in each region',
                    'sync': 'Async replication across regions'
                },

                'pros': [
                    'Lowest query latency (serve from nearest region)',
                    'High availability (region failures isolated)',
                    'No cross-region queries',
                    'Simple routing (geographic DNS)'
                ],

                'cons': [
                    'Highest storage cost (5x replication)',
                    'Write complexity (must sync all regions)',
                    'Consistency challenges (eventual consistency across regions)',
                    'Bandwidth cost for cross-region sync'
                ],

                'cost_calculation': {
                    'regions': 5,
                    'storage_per_region_pb': 512,  # 256T vectors × 768 dims × 4 bytes
                    'total_storage_pb': 512 * 5,
                    'monthly_storage_cost': 512 * 5 * 1000 * 1000 * 0.023,  # S3 pricing
                    'cross_region_bandwidth_tb_per_day': 100,
                    'monthly_bandwidth_cost': 100 * 30 * 0.02 * 1000  # $0.02/GB
                },

                'when_to_use': 'Global consumer applications with <100ms latency SLA'
            },

            'regional_sharding': {
                'description': 'Partition data by region, each region has subset',

                'architecture': {
                    'us_data': 'Embeddings for US users/products',
                    'eu_data': 'Embeddings for EU users/products',
                    'apac_data': 'Embeddings for APAC users/products',
                    'routing': 'User queries route to their home region'
                },

                'pros': [
                    'Lower storage cost (no full replication)',
                    'Data sovereignty (EU data stays in EU)',
                    'Simpler sync (no cross-region writes)',
                    'Scales better (data sharded)'
                ],

                'cons': [
                    'Cross-region queries expensive (rare use case)',
                    'Load imbalance (some regions busier)',
                    'Harder to serve global catalog (e.g., e-commerce)',
                    'Partitioning logic complexity'
                ],

                'when_to_use': 'Inherently regional data (local businesses, regional content)',

                'example_e_commerce': {
                    'challenge': 'Products sold globally, users everywhere',
                    'solution': 'Full replication for product catalog + regional sharding for user data'
                }
            },

            'tiered_distribution': {
                'description': 'Hot data in all regions, cold data in primary region',

                'architecture': {
                    'tier_1_hot': 'Top 10% most-queried embeddings in all regions',
                    'tier_2_warm': 'Next 30% in 2-3 regions',
                    'tier_3_cold': 'Remaining 60% in primary region only',
                    'routing': 'Check local tier first, fall back to remote'
                },

                'pros': [
                    'Balance cost and latency',
                    'Hot queries served locally (fast)',
                    'Cold queries tolerate higher latency (acceptable)',
                    'Much lower storage cost than full replication'
                ],

                'cons': [
                    'Complexity in tier management',
                    'Inconsistent latency (hot vs cold)',
                    'Must track access patterns to tier correctly'
                ],

                'cost_savings': '60-80% vs full replication',

                'when_to_use': 'Zipfian access patterns (power law distribution)',

                'implementation': {
                    'tracking': 'Count queries per embedding',
                    'promotion': 'Move to higher tier if access frequency > threshold',
                    'demotion': 'Move to lower tier if idle for 30 days',
                    'periodic_rebalancing': 'Weekly tier reassignment'
                }
            },

            'edge_caching': {
                'description': 'CDN-style caching at edge locations',

                'architecture': {
                    'origin': 'Authoritative data in regional datacenters',
                    'edge_pops': '100+ edge locations (CloudFlare, Fastly, etc.)',
                    'cache': 'Top 1% most-queried embeddings at edge',
                    'cache_size': '~2.5T embeddings (1% of 256T)'
                },

                'pros': [
                    'Extremely low latency (<20ms globally)',
                    'Handles traffic spikes',
                    'Reduces origin load by 80-90%',
                    'Leverages existing CDN infrastructure'
                ],

                'cons': [
                    'Only helps for popular queries (cold queries still slow)',
                    'Cache invalidation complexity',
                    'CDN costs',
                    'Limited cache size at edge'
                ],

                'use_cases': [
                    'Product search (same popular products queried repeatedly)',
                    'Content recommendation (trending items)',
                    'Fraud detection (common fraud patterns)'
                ],

                'cache_hit_rate': '85-95% for typical workloads'
            }
        }

    def data_residency_compliance(self):
        """Handle data sovereignty requirements (GDPR, etc.)"""

        return {
            'gdpr_eu': {
                'requirement': 'EU citizen data must stay in EU',

                'architecture': {
                    'eu_region': 'Primary storage for EU users in EU datacenter',
                    'replication': 'Replicate within EU only (Paris, Frankfurt, Ireland)',
                    'cross_region': 'No replication to US/Asia',
                    'user_routing': 'EU users always routed to EU region'
                },

                'implementation': [
                    'Tag each embedding with region constraint',
                    'Enforce at ingestion (reject non-compliant writes)',
                    'Audit trail of data location',
                    'Encrypt at rest with EU-only keys'
                ]
            },

            'ccpa_california': {
                'requirement': 'California residents can request deletion',

                'implementation': [
                    'Maintain user_id → embedding_ids mapping',
                    'On deletion request, identify all user embeddings',
                    'Delete from all replicas within 30 days',
                    'Provide deletion confirmation'
                ]
            },

            'china_cybersecurity_law': {
                'requirement': 'China citizen data must stay in China',

                'architecture': 'Completely separate China region, no cross-border data transfer',

                'challenges': [
                    'Cannot replicate to/from China easily',
                    'Must operate independently',
                    'Higher operational complexity'
                ]
            }
        }

    def latency_optimization_strategies(self):
        """Minimize latency for global users"""

        return {
            'geo_dns': {
                'description': 'Route users to nearest datacenter',
                'implementation': 'AWS Route 53 geolocation routing, CloudFlare',
                'latency_reduction': '100-200ms (cross-continent → in-region)'
            },

            'anycast': {
                'description': 'Single IP that routes to nearest PoP',
                'implementation': 'CloudFlare, Fastly',
                'benefit': 'Automatic routing without DNS'
            },

            'prefetching': {
                'description': 'Predict likely queries, prefetch results',
                'implementation': 'Model user behavior, precompute popular queries',
                'latency_reduction': '50-100ms (cache hit)'
            },

            'query_result_caching': {
                'description': 'Cache query results, not just embeddings',
                'implementation': 'Redis/Memcached with query hash as key',
                'ttl': '5-60 minutes (depending on freshness needs)',
                'hit_rate': '60-80% for read-heavy workloads'
            },

            'compression': {
                'description': 'Compress results before transmitting',
                'implementation': 'gzip, brotli',
                'latency_reduction': '10-50ms (less data over network)'
            }
        }

# Example: Calculate global deployment costs
class GlobalDeploymentCostModel:
    """Model costs for global deployment"""

    def calculate_total_cost(self, strategy, num_vectors, embedding_dim, qps):
        """Calculate monthly costs"""

        # Storage costs
        bytes_per_vector = embedding_dim * 4  # float32
        total_bytes = num_vectors * bytes_per_vector
        total_tb = total_bytes / (1024 ** 4)

        if strategy == 'full_replication':
            regions = 5
            storage_tb = total_tb * regions * 1.5  # 1.5x for index overhead
            storage_cost = storage_tb * 1000 * 0.023  # $0.023/GB/month

            # Compute costs (query serving)
            machines_per_region = 200
            cost_per_machine = 1.50 * 24 * 30  # $1.50/hour
            compute_cost = machines_per_region * regions * cost_per_machine

            # Bandwidth (cross-region sync)
            bandwidth_tb_per_day = 50  # New embeddings + updates
            bandwidth_cost = bandwidth_tb_per_day * 30 * 0.02 * 1000

            total_cost = storage_cost + compute_cost + bandwidth_cost

            return {
                'strategy': 'Full Replication',
                'storage_cost': storage_cost,
                'compute_cost': compute_cost,
                'bandwidth_cost': bandwidth_cost,
                'total_monthly_cost': total_cost,
                'cost_per_million_queries': total_cost / (qps * 60 * 60 * 24 * 30 / 1_000_000)
            }

        elif strategy == 'tiered':
            # Hot tier (10%) in all 5 regions
            # Cold tier (90%) in 1 region
            hot_storage_tb = total_tb * 0.1 * 5 * 1.5
            cold_storage_tb = total_tb * 0.9 * 1 * 1.5
            storage_cost = (hot_storage_tb + cold_storage_tb) * 1000 * 0.023

            # Compute (less than full replication)
            machines_per_region = 100  # Fewer machines needed
            compute_cost = machines_per_region * 5 * 1.50 * 24 * 30

            # Bandwidth (less cross-region traffic)
            bandwidth_cost = 20 * 30 * 0.02 * 1000

            total_cost = storage_cost + compute_cost + bandwidth_cost

            return {
                'strategy': 'Tiered Distribution',
                'storage_cost': storage_cost,
                'compute_cost': compute_cost,
                'bandwidth_cost': bandwidth_cost,
                'total_monthly_cost': total_cost,
                'savings_vs_full_replication': '60-70%'
            }

# Example
cost_model = GlobalDeploymentCostModel()
full_rep_cost = cost_model.calculate_total_cost(
    strategy='full_replication',
    num_vectors=256_000_000_000_000,
    embedding_dim=768,
    qps=1_000_000
)

tiered_cost = cost_model.calculate_total_cost(
    strategy='tiered',
    num_vectors=256_000_000_000_000,
    embedding_dim=768,
    qps=1_000_000
)

print(f"Full Replication: ${full_rep_cost['total_monthly_cost']:,.0f}/month")
print(f"Tiered: ${tiered_cost['total_monthly_cost']:,.0f}/month")
print(f"Savings: {(1 - tiered_cost['total_monthly_cost']/full_rep_cost['total_monthly_cost'])*100:.1f}%")
