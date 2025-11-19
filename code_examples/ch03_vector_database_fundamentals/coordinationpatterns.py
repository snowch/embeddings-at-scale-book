# Code from Chapter 03
# Book: Embeddings at Scale


class CoordinationPatterns:
    """Coordination in distributed vector databases"""

    def what_needs_coordination(self):
        """Operations requiring coordination"""

        return {
            "shard_assignment": {
                "what": "Which vectors belong to which shards",
                "coordinator": "ZooKeeper, etcd, or Consul",
                "update_frequency": "Infrequent (shard rebalancing)",
                "consistency_requirement": "Strong (avoid double-assignment)",
            },
            "leader_election": {
                "what": "Which node is primary for each shard",
                "coordinator": "Raft/Paxos consensus",
                "update_frequency": "Only on failures",
                "consistency_requirement": "Strong (avoid split-brain)",
            },
            "schema_changes": {
                "what": "Embedding dimension changes, index upgrades",
                "coordinator": "Centralized config service",
                "update_frequency": "Rare (major version changes)",
                "consistency_requirement": "Strong (all nodes must agree)",
            },
            "global_counters": {
                "what": "Total vector count, global statistics",
                "coordinator": "Eventually consistent aggregation",
                "update_frequency": "Continuous",
                "consistency_requirement": "Weak (approximate is fine)",
            },
        }

    def avoid_coordination_where_possible(self):
        """Coordination is expensive at scale - minimize it"""

        return {
            "principle": "Coordination limits scalability - avoid when possible",
            "techniques": [
                "Deterministic shard assignment (hash function, no coordination)",
                "Immutable assignments (decide once at creation)",
                "Eventual consistency for non-critical paths",
                "Local decisions without global coordination",
                "Conflict-free replicated data types (CRDTs)",
            ],
            "example_shard_assignment": {
                "bad": "Query coordinator to find which shard for each vector",
                "good": "Hash(vector_id) % num_shards - no coordinator needed",
            },
            "example_counters": {
                "bad": "Coordinate across all shards to get exact count",
                "good": "Periodically aggregate counts, accept staleness",
            },
        }
