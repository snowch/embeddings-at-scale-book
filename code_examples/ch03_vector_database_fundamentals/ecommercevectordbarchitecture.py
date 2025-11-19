# Code from Chapter 03
# Book: Embeddings at Scale


class EcommerceVectorDBArchitecture:
    """Example production architecture for 50B products"""

    def __init__(self):
        self.scale = {
            "total_products": 50_000_000_000,
            "embedding_dim": 512,
            "queries_per_second": 500_000,
            "inserts_per_second": 50_000,  # New products + updates
            "regions": 5,  # US-West, US-East, EU, Asia-Pacific, Latin America
        }

    def architecture(self):
        """Multi-region, sharded architecture"""

        # Per-region deployment
        self.scale["total_products"]  # Full catalog in each region

        # Sharding within region
        shards_per_region = 200  # 250M vectors per shard

        # Machines per shard (including replicas)
        replicas_per_shard = 3

        machines_per_region = shards_per_region * replicas_per_shard
        total_machines = machines_per_region * self.scale["regions"]

        return {
            "deployment_strategy": "Multi-region active-active",
            "regions": self.scale["regions"],
            "shards_per_region": shards_per_region,
            "replicas_per_shard": replicas_per_shard,
            "machines_per_region": machines_per_region,
            "total_machines": total_machines,
            "machine_spec": {
                "cpu": "32 cores",
                "memory": "256 GB",
                "disk": "4TB NVMe SSD",
                "network": "25 Gbps",
                "cloud_instance": "AWS r5.8xlarge equivalent",
            },
            "index_configuration": {
                "type": "HNSW",
                "M": 32,
                "ef_construction": 400,
                "ef_search": 100,  # Tuned for p99 < 50ms
                "recall_at_10": 0.97,
            },
            "query_routing": {
                "strategy": "Geographic routing to nearest region",
                "fallback": "Route to next-nearest on failure",
                "load_balancing": "Consistent hashing across shards",
                "cache": "Regional L1 cache (10% of data)",
            },
            "update_strategy": {
                "new_products": "Async write to all regions within 5 minutes",
                "product_updates": "Lazy update (only on next query)",
                "deletions": "Soft delete with async cleanup",
                "index_rebuild": "Rolling rebuild every 7 days",
            },
            "cost_breakdown": {
                "compute": f"${total_machines * 1.5 * 24 * 30:,.0f}/month",  # $1.50/hour per machine
                "storage": f"${(total_machines * 4 * 150) / 1000:,.0f}/month",  # $150/TB/month
                "network": "$50,000/month",  # Data transfer
                "total_monthly": f"${total_machines * 1.5 * 24 * 30 + (total_machines * 4 * 150) / 1000 + 50000:,.0f}",
            },
        }

    def sla_design_example(self):
        """How architecture design meets SLA targets"""

        return {
            "p50_latency": {
                "target": "<20ms",
                "design_approach": [
                    "In-region routing (no cross-region latency)",
                    "HNSW with tuned ef_search=100",
                    "Hot index pages cached in memory",
                    "SSD-backed index storage",
                ],
            },
            "p99_latency": {
                "target": "<100ms",
                "design_approach": [
                    "Adaptive ef_search (increase on cache miss)",
                    "Circuit breaker (fail fast on overload)",
                    "Dedicated query serving machines (no interference from writes)",
                    "Pre-warming index after deployment",
                ],
            },
            "availability": {
                "target": "99.99%",
                "design_approach": [
                    "3x replication within region",
                    "Multi-region failover",
                    "Health checks with automatic traffic routing",
                    "Rolling deployments (no downtime)",
                ],
            },
            "throughput": {
                "target": "500K QPS",
                "design_approach": [
                    "Horizontal scaling (200 shards)",
                    "Query parallelization across replicas",
                    "Caching (target 85% cache hit rate)",
                    "Batching (process multiple queries together)",
                ],
            },
        }
