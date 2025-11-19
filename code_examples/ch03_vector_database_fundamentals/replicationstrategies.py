# Code from Chapter 03
# Book: Embeddings at Scale

class ReplicationStrategies:
    """How to replicate embeddings and indices"""

    def primary_replica_pattern(self):
        """Primary-replica (leader-follower) replication"""

        return {
            'architecture': {
                'primary': 'Handles all writes',
                'replicas': '2+ replicas handle reads',
                'replication': 'Async from primary to replicas'
            },

            'write_path': [
                '1. Client writes to primary',
                '2. Primary updates its index',
                '3. Primary ACKs write',
                '4. Primary async replicates to replicas',
                '5. Replicas update their indices'
            ],

            'read_path': [
                '1. Client queries any replica',
                '2. Replica searches its local index',
                '3. Replica returns results'
            ],

            'failure_handling': {
                'primary_fails': 'Promote replica to primary',
                'replica_fails': 'Remove from load balancer',
                'network_partition': 'Continue serving from available nodes'
            },

            'pros': [
                'Simple to understand and implement',
                'Read scalability (add more replicas)',
                'Write consistency (single write path)'
            ],

            'cons': [
                'Write bottleneck (single primary)',
                'Failover time (30-60 seconds)',
                'Replica lag (eventual consistency)'
            ],

            'use_case': 'Default choice for most vector databases'
        }

    def multi_primary_pattern(self):
        """Multi-primary (multi-leader) replication"""

        return {
            'architecture': {
                'primaries': 'Multiple nodes accept writes',
                'replication': 'Bidirectional between primaries',
                'conflict_resolution': 'Last-write-wins or custom logic'
            },

            'write_path': [
                '1. Client writes to nearest primary',
                '2. Primary updates local index',
                '3. Primary ACKs write',
                '4. Primary replicates to other primaries',
                '5. Handle conflicts if simultaneous writes'
            ],

            'conflict_resolution': {
                'vector_inserts': 'No conflict (different IDs)',
                'vector_updates': 'Last-write-wins (timestamp)',
                'vector_deletes': 'Tombstone approach',
                'metadata_updates': 'Custom merge logic'
            },

            'pros': [
                'Write scalability (multiple primaries)',
                'Low latency (write to nearest)',
                'No single point of failure'
            ],

            'cons': [
                'Complex conflict resolution',
                'Potential for inconsistency',
                'Harder to reason about'
            ],

            'use_case': 'Global deployments, high write throughput'
        }

    def leaderless_pattern(self):
        """Leaderless replication (quorum-based)"""

        return {
            'architecture': {
                'structure': 'No designated leader, all nodes equal',
                'writes': 'Write to W nodes, succeed if W/2+1 ACK',
                'reads': 'Read from R nodes, take majority'
            },

            'quorum_configuration': {
                'replication_factor_n': 3,
                'write_quorum_w': 2,  # Must write to 2/3 nodes
                'read_quorum_r': 2,   # Must read from 2/3 nodes
                'guarantee': 'W + R > N ensures read sees latest write'
            },

            'pros': [
                'High availability (tolerates node failures)',
                'No leader election delay',
                'Flexible consistency tuning (adjust W and R)'
            ],

            'cons': [
                'Higher latency (must contact multiple nodes)',
                'More complex client logic',
                'Read repair overhead'
            ],

            'use_case': 'Rarely used for vector databases - complexity not worth it'
        }

# Recommendation
def recommend_replication_strategy(scale, write_workload, read_workload, geo_distribution):
    """Recommend replication strategy"""

    if geo_distribution == 'multi_region' and write_workload == 'high':
        return {
            'strategy': 'Multi-primary (leader per region)',
            'rationale': 'Need low-latency writes in each region'
        }
    elif write_workload == 'low' and read_workload == 'high':
        return {
            'strategy': 'Primary-replica with many replicas',
            'rationale': 'Read-heavy workload benefits from read replicas'
        }
    else:
        return {
            'strategy': 'Primary-replica (default)',
            'rationale': 'Simple, reliable, covers most use cases'
        }
