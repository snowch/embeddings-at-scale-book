# Code from Chapter 03
# Book: Embeddings at Scale

class VectorDatabaseSLAs:
    """Service Level Agreements for vector databases"""

    def core_metrics(self):
        """Metrics that matter for SLAs"""

        return {
            'query_latency': {
                'metric': 'Time from query submission to results returned',
                'measurements': {
                    'p50': 'Median latency',
                    'p95': '95th percentile (1 in 20 queries slower)',
                    'p99': '99th percentile (1 in 100 queries slower)',
                    'p99.9': '99.9th percentile (1 in 1000 queries slower)'
                },
                'why_percentiles': 'Average hides outliers; users notice tail latency',
                'typical_targets': {
                    'p50': '<20ms',
                    'p95': '<50ms',
                    'p99': '<100ms',
                    'p99.9': '<500ms'
                },
                'business_impact': 'Every 100ms latency → 1% conversion loss (empirical)'
            },

            'recall_at_k': {
                'metric': 'Fraction of true top-K items returned',
                'formula': 'recall@k = |returned ∩ true_top_k| / k',
                'typical_targets': {
                    'recall@10': '>0.95',
                    'recall@100': '>0.98'
                },
                'measurement': 'Offline evaluation on test set',
                'business_impact': 'Low recall → users don\'t find relevant items → poor experience'
            },

            'throughput': {
                'metric': 'Queries per second (QPS)',
                'typical_targets': {
                    'per_shard': '1K-10K QPS',
                    'global': '100K-1M QPS'
                },
                'measurement': 'Monitor at load balancer and per-shard',
                'business_impact': 'Insufficient throughput → requests queued or dropped'
            },

            'availability': {
                'metric': 'Percentage of time system is operational',
                'typical_target': '99.99% (52 minutes downtime/year)',
                'measurement': 'Success rate of health checks',
                'business_impact': 'Downtime → lost revenue, user frustration'
            },

            'index_freshness': {
                'metric': 'Time from data ingestion to queryable',
                'typical_target': '<5 minutes for eventual consistency',
                'measurement': 'Monitor insertion timestamp vs query visibility',
                'business_impact': 'Stale indices → users don\'t see new items'
            },

            'resource_utilization': {
                'metrics': ['CPU', 'Memory', 'Disk I/O', 'Network'],
                'typical_targets': {
                    'cpu': '<70% average (headroom for spikes)',
                    'memory': '<85% (avoid swapping)',
                    'disk_io': '<80% (avoid saturation)',
                    'network': '<60% (avoid congestion)'
                },
                'business_impact': 'Over-utilization → increased latency, failures'
            }
        }

    def calculate_availability_budget(self, target_availability):
        """Calculate allowed downtime"""

        availability_to_downtime = {
            0.9: ('90%', '36.5 days/year'),
            0.99: ('99%', '3.65 days/year'),
            0.999: ('99.9%', '8.76 hours/year'),
            0.9999: ('99.99%', '52.6 minutes/year'),
            0.99999: ('99.999%', '5.26 minutes/year')
        }

        return availability_to_downtime.get(target_availability, 'Unknown target')

    def sla_vs_slo_vs_sli(self):
        """Clarify terminology"""

        return {
            'sli': {
                'name': 'Service Level Indicator',
                'definition': 'Quantitative measure of service level',
                'examples': [
                    'p99 query latency: 78ms',
                    'recall@10: 0.96',
                    'availability: 99.97%'
                ]
            },

            'slo': {
                'name': 'Service Level Objective',
                'definition': 'Target value for an SLI',
                'examples': [
                    'p99 latency < 100ms',
                    'recall@10 > 0.95',
                    'availability > 99.99%'
                ],
                'internal': 'Internal goals for engineering team'
            },

            'sla': {
                'name': 'Service Level Agreement',
                'definition': 'Contract with users specifying SLOs and consequences',
                'examples': [
                    'p99 < 100ms or 10% service credit',
                    'Availability > 99.99% or 25% refund'
                ],
                'external': 'Legal commitment to customers'
            },

            'relationship': 'SLI (measurement) → SLO (target) → SLA (contract)'
        }

# Example SLA document
class ExampleVectorDatabaseSLA:
    """Reference SLA for production vector database"""

    def __init__(self):
        self.service_name = "Production Vector Search API"
        self.version = "v1"

    def sla_terms(self):
        """Actual SLA commitments"""

        return {
            'performance_sla': {
                'query_latency_p99': {
                    'target': '<100ms',
                    'measurement_window': '5-minute rolling window',
                    'measurement_method': 'Server-side timing, excluding network',
                    'breach_threshold': 'p99 >100ms for >3 consecutive windows',
                    'consequence': '10% service credit for affected period'
                },

                'query_latency_p50': {
                    'target': '<20ms',
                    'measurement_window': '5-minute rolling window',
                    'breach_threshold': 'p50 >20ms for >5 consecutive windows',
                    'consequence': 'Informational only (no penalty)'
                }
            },

            'quality_sla': {
                'recall_at_10': {
                    'target': '>0.95',
                    'measurement_method': 'Weekly offline evaluation on test set',
                    'breach_threshold': 'recall <0.95 for 2 consecutive weeks',
                    'consequence': 'Must provide root cause analysis + fix plan'
                }
            },

            'availability_sla': {
                'uptime': {
                    'target': '99.99%',
                    'measurement_window': 'Monthly',
                    'measurement_method': 'Health check success rate',
                    'breach_threshold': 'Availability <99.99% in any month',
                    'consequence': '25% monthly service credit'
                },

                'scheduled_maintenance': {
                    'allowed': '4 hours/quarter',
                    'notice': '7 days advance notice',
                    'window': 'Sunday 2-6am PST'
                }
            },

            'capacity_sla': {
                'throughput': {
                    'target': '>100,000 QPS',
                    'measurement_method': 'Load balancer metrics',
                    'breach_threshold': 'Cannot sustain 100K QPS',
                    'consequence': 'Must scale within 4 hours'
                }
            },

            'exclusions': [
                'Customer misuse (sending malformed queries)',
                'DDoS attacks',
                'Force majeure (natural disasters, etc.)',
                'Issues caused by customer\'s infrastructure'
            ],

            'measurement_transparency': {
                'dashboard': 'Public status page with real-time SLI metrics',
                'reports': 'Monthly SLA compliance reports',
                'alerts': 'Proactive notification of SLA breaches'
            }
        }
