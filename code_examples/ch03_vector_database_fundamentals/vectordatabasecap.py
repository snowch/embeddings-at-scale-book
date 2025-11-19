# Code from Chapter 03
# Book: Embeddings at Scale

class VectorDatabaseCAP:
    """CAP theorem as applied to vector databases"""

    def cap_tradeoffs(self):
        """Where vector DBs fall on CAP spectrum"""

        return {
            'consistency': {
                'requirement': 'Low - embeddings are already approximate',
                'acceptable_model': 'Eventual consistency',
                'reason': 'If one replica has slightly outdated embeddings, query results are still useful',
                'strong_consistency_when': 'Critical metadata (user permissions, deletion flags)'
            },

            'availability': {
                'requirement': 'High - user-facing queries must succeed',
                'target': '99.99% availability',
                'techniques': [
                    'Multi-replica (3x typical)',
                    'Read from any replica',
                    'Automatic failover',
                    'Circuit breakers'
                ]
            },

            'partition_tolerance': {
                'requirement': 'High - network issues are inevitable',
                'behavior': 'Gracefully degrade (serve stale data, reduce recall)',
                'techniques': [
                    'Fallback to cached results',
                    'Serve partial results from available shards',
                    'Multi-region replication'
                ]
            },

            'chosen_tradeoff': 'AP (Availability + Partition Tolerance)',
            'sacrifice': 'Strong consistency',
            'rationale': 'Slight staleness is acceptable for embedding search'
        }

    def consistency_models(self):
        """Consistency models for different operations"""

        return {
            'writes_inserts': {
                'model': 'Eventual consistency',
                'implementation': 'Write to primary, async replicate',
                'visibility': 'New embedding visible within 5 seconds',
                'acceptable_because': 'Rare that user immediately queries for just-inserted embedding'
            },

            'updates_deletions': {
                'model': 'Eventual consistency with tombstones',
                'implementation': 'Mark deleted immediately, propagate async',
                'visibility': 'Deletion effective within 1 second',
                'safety': 'Deleted items filtered at query time'
            },

            'reads_queries': {
                'model': 'Read-your-writes for same session',
                'implementation': 'Session affinity to replica that handled write',
                'consistency': 'User sees their own changes immediately',
                'cross_user': 'May see stale data from other users (acceptable)'
            },

            'metadata_filters': {
                'model': 'Strong consistency',
                'implementation': 'Sync write to metadata DB',
                'reason': 'Security filters (user access) must be immediate',
                'example': 'User loses access → must be blocked immediately'
            }
        }

# Concrete example
class EventualConsistencyExample:
    """How eventual consistency works in practice"""

    def user_journey(self):
        """User adds product, then searches for it"""

        timeline = [
            {
                'time': '00:00.000',
                'event': 'User uploads new product image',
                'action': 'Generate embedding, write to primary shard',
                'state': 'Embedding on primary only'
            },
            {
                'time': '00:00.100',
                'event': 'Primary acknowledges write',
                'action': 'Return success to user',
                'state': 'Primary has embedding, replicas replicating'
            },
            {
                'time': '00:01.000',
                'event': 'User searches for similar products',
                'action': 'Query routed to secondary replica',
                'state': 'Secondary may not have embedding yet',
                'result': 'New product not in results - acceptable'
            },
            {
                'time': '00:05.000',
                'event': 'Replication completes',
                'action': 'Embedding now on all replicas',
                'state': 'Eventual consistency achieved'
            },
            {
                'time': '00:10.000',
                'event': 'User searches again',
                'action': 'Query any replica',
                'result': 'New product now appears - user happy'
            }
        ]

        return timeline
