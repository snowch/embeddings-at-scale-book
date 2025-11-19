# Code from Chapter 03
# Book: Embeddings at Scale

class FailureRecovery:
    """Handling failures at trillion-scale"""

    def failure_modes(self):
        """Catalog of failures and mitigations"""

        return {
            'node_failure': {
                'probability': 'High (MTBF ~3-5 years per machine)',
                'impact': 'Loss of one shard or replica',
                'detection': 'Heartbeat timeout (10-30 seconds)',
                'recovery': [
                    'Remove from load balancer immediately',
                    'Serve from replica (if available)',
                    'Spawn replacement node',
                    'Rebuild index from backup or re-replicate'
                ],
                'recovery_time': '5-30 minutes',
                'data_loss': 'None (replicas available)'
            },

            'disk_failure': {
                'probability': 'Medium (MTBF ~4 years per disk)',
                'impact': 'Loss of shard data on disk',
                'detection': 'I/O errors, SMART metrics',
                'recovery': [
                    'Switch to replica',
                    'Replace disk',
                    'Restore from backup or peer',
                    'Rebuild index'
                ],
                'recovery_time': '1-4 hours',
                'data_loss': 'None if replicated'
            },

            'network_partition': {
                'probability': 'Medium (datacenter network issues)',
                'impact': 'Subset of nodes unreachable',
                'detection': 'Heartbeat failures from multiple nodes',
                'recovery': [
                    'Serve from available partition',
                    'Gradeful degradation (partial results)',
                    'Wait for network healing',
                    'Resync after partition resolves'
                ],
                'recovery_time': 'Minutes to hours (depends on network)',
                'data_loss': 'None, but stale data possible'
            },

            'index_corruption': {
                'probability': 'Low but critical',
                'impact': 'Index returns incorrect results',
                'detection': [
                    'Checksums on index files',
                    'Recall monitoring (sudden drop)',
                    'User reports'
                ],
                'recovery': [
                    'Rollback to previous index version',
                    'Rebuild index from raw embeddings',
                    'Post-mortem to identify root cause'
                ],
                'recovery_time': '1-8 hours (rebuild time)',
                'data_loss': 'Queries may have returned wrong results'
            },

            'query_of_death': {
                'description': 'Pathological query causes crashes',
                'probability': 'Low but high impact',
                'impact': 'Node crashes, cascading failures',
                'detection': 'Spike in error rates, node restarts',
                'recovery': [
                    'Identify query pattern',
                    'Add input validation/sanitization',
                    'Rate limit specific patterns',
                    'Circuit breaker to prevent cascades'
                ],
                'recovery_time': 'Minutes (once identified)',
                'prevention': 'Query timeouts, input validation, load shedding'
            },

            'replication_lag': {
                'description': 'Replicas fall behind primary',
                'probability': 'Medium during high write load',
                'impact': 'Stale query results',
                'detection': 'Monitor replication lag metric',
                'recovery': [
                    'Throttle writes temporarily',
                    'Add more replicas',
                    'Increase replication bandwidth',
                    'Batch updates more efficiently'
                ],
                'acceptable_lag': '<5 seconds',
                'alert_threshold': '>60 seconds'
            }
        }

    def disaster_recovery(self):
        """Region-level failures and recovery"""

        return {
            'scenario': 'Entire datacenter/region fails',

            'preparation': [
                'Multi-region replication (full copy in 2+ regions)',
                'Automated failover to backup region',
                'Regular DR drills (quarterly)',
                'RPO (Recovery Point Objective): <5 minutes',
                'RTO (Recovery Time Objective): <15 minutes'
            ],

            'failover_process': [
                '1. Detect region failure (load balancer health checks)',
                '2. Update DNS/routing to direct traffic to backup region',
                '3. Promote backup region to primary',
                '4. Monitor backup region for capacity',
                '5. Scale up if needed',
                '6. Investigate primary region failure',
                '7. Restore primary region when possible',
                '8. Resync and failback'
            ],

            'cost': 'Double infrastructure (active-active in 2 regions)',

            'alternative_active_passive': {
                'cost': 'Lower (passive region smaller)',
                'rto': 'Higher (need to scale up passive region)',
                'tradeoff': 'Cost vs recovery time'
            }
        }

    def chaos_engineering(self):
        """Proactive failure testing"""

        return {
            'philosophy': 'Test failures in production to build confidence',

            'experiments': [
                'Terminate random nodes (10% of fleet)',
                'Inject network latency (100-500ms)',
                'Induce CPU/memory saturation',
                'Simulate datacenter partition',
                'Corrupt index files',
                'Fill up disks'
            ],

            'metrics_to_monitor': [
                'Query success rate (should stay >99.9%)',
                'Query latency (should stay within SLA)',
                'Error logs (no crashes)',
                'Automatic recovery (no human intervention needed)'
            ],

            'frequency': 'Weekly in staging, monthly in production',

            'tools': ['Chaos Monkey', 'Gremlin', 'LitmusChaos']
        }
