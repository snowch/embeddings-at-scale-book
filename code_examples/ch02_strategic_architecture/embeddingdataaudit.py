# Code from Chapter 02
# Book: Embeddings at Scale

class EmbeddingDataAudit:
    """Audit data readiness for embedding strategy"""

    def audit_data_readiness(self, data_sources):
        """
        Assess data sources for embedding suitability

        data_sources: List of available data sources
        Returns: Readiness assessment and recommendations
        """
        assessment = {
            'data_sources': [],
            'gaps': [],
            'quality_issues': [],
            'recommendations': []
        }

        for source in data_sources:
            source_assessment = {
                'name': source['name'],
                'volume': source['volume'],
                'quality_score': self.assess_quality(source),
                'coverage_score': self.assess_coverage(source),
                'freshness_score': self.assess_freshness(source),
                'labeling_status': self.assess_labeling(source),
                'readiness': 'ready' if all([
                    self.assess_quality(source) > 0.7,
                    self.assess_coverage(source) > 0.6,
                    source['volume'] > 10000
                ]) else 'needs_work'
            }

            assessment['data_sources'].append(source_assessment)

            # Identify gaps
            if source_assessment['quality_score'] < 0.7:
                assessment['gaps'].append(f"{source['name']}: quality below threshold")
            if source_assessment['coverage_score'] < 0.6:
                assessment['gaps'].append(f"{source['name']}: coverage insufficient")
            if source['volume'] < 10000:
                assessment['gaps'].append(f"{source['name']}: insufficient volume for quality embeddings")

        return assessment

    def assess_quality(self, source):
        """Score data quality 0-1"""
        quality_factors = {
            'completeness': source.get('completeness', 0.5),  # % of required fields populated
            'accuracy': source.get('accuracy', 0.7),  # Validation pass rate
            'consistency': source.get('consistency', 0.8),  # Format/schema compliance
            'deduplication': 1 - source.get('duplicate_rate', 0.1)  # 1 - duplicate %
        }
        return sum(quality_factors.values()) / len(quality_factors)

    def assess_coverage(self, source):
        """Score how well data covers the problem space 0-1"""
        # Domain coverage (does data span all important categories?)
        # Temporal coverage (sufficient historical depth?)
        # Edge case coverage (rare but important cases present?)
        return source.get('domain_coverage', 0.5)

    def assess_freshness(self, source):
        """Score data recency 0-1"""
        days_since_update = source.get('days_since_update', 365)
        # Exponential decay: fresh data (1.0) → stale data (0.0)
        import math
        return math.exp(-days_since_update / 90)  # 90-day half-life

    def assess_labeling(self, source):
        """Assess labeling status for supervised learning"""
        labeled_fraction = source.get('labeled_fraction', 0.0)
        label_quality = source.get('label_quality', 0.0)

        if labeled_fraction > 0.8 and label_quality > 0.9:
            return 'excellent'
        elif labeled_fraction > 0.5 and label_quality > 0.7:
            return 'good'
        elif labeled_fraction > 0.2:
            return 'partial'
        else:
            return 'unlabeled'

# Example usage
auditor = EmbeddingDataAudit()

data_sources = [
    {
        'name': 'product_catalog',
        'volume': 2_500_000,
        'completeness': 0.92,
        'accuracy': 0.88,
        'consistency': 0.95,
        'duplicate_rate': 0.03,
        'domain_coverage': 0.85,
        'days_since_update': 1,
        'labeled_fraction': 0.0,  # No labels needed for products
        'label_quality': 0.0
    },
    {
        'name': 'customer_reviews',
        'volume': 15_000_000,
        'completeness': 0.78,
        'accuracy': 0.65,  # Spam/fake reviews
        'consistency': 0.88,
        'duplicate_rate': 0.12,
        'domain_coverage': 0.75,
        'days_since_update': 1,
        'labeled_fraction': 0.15,  # Some sentiment labels
        'label_quality': 0.82
    },
    {
        'name': 'user_behavior_logs',
        'volume': 500_000_000,
        'completeness': 0.98,
        'accuracy': 0.99,
        'consistency': 0.97,
        'duplicate_rate': 0.001,
        'domain_coverage': 0.95,
        'days_since_update': 0,  # Real-time
        'labeled_fraction': 0.0,  # Behavioral, no labels
        'label_quality': 0.0
    }
]

assessment = auditor.audit_data_readiness(data_sources)
