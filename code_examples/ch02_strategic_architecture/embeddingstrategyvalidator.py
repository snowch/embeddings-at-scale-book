# Code from Chapter 02
# Book: Embeddings at Scale

class EmbeddingStrategyValidator:
    """Validate embedding strategy before large-scale investment"""

    def validate_strategy(self, strategy):
        """
        Score strategy across key dimensions
        Returns validation report with go/no-go recommendation
        """

        validation = {
            'strategic_fit': self.assess_strategic_fit(strategy),
            'data_readiness': self.assess_data_readiness(strategy),
            'technical_feasibility': self.assess_technical_feasibility(strategy),
            'organizational_readiness': self.assess_organizational_readiness(strategy),
            'financial_viability': self.assess_financial_viability(strategy),
            'risk_assessment': self.assess_risks(strategy)
        }

        # Overall score (weighted average)
        weights = {
            'strategic_fit': 0.25,
            'data_readiness': 0.20,
            'technical_feasibility': 0.15,
            'organizational_readiness': 0.15,
            'financial_viability': 0.15,
            'risk_assessment': 0.10
        }

        overall_score = sum(
            validation[dim]['score'] * weights[dim]
            for dim in weights.keys()
        )

        validation['overall_score'] = overall_score
        validation['recommendation'] = self.get_recommendation(overall_score, validation)

        return validation

    def assess_strategic_fit(self, strategy):
        """Does this strategy align with business objectives?"""
        # Scoring criteria:
        # - Clear connection to business metrics (0-0.3)
        # - Alignment with company strategy (0-0.3)
        # - Defensibility / competitive moat potential (0-0.4)

        score = 0.0
        issues = []

        if strategy.get('business_metrics_defined'):
            score += 0.3
        else:
            issues.append("Business metrics not clearly defined")

        if strategy.get('aligns_with_company_strategy'):
            score += 0.3
        else:
            issues.append("Unclear alignment with overall company strategy")

        moat_potential = strategy.get('moat_potential', 'low')
        if moat_potential == 'high':
            score += 0.4
        elif moat_potential == 'medium':
            score += 0.2
        else:
            issues.append("Limited competitive moat potential")

        return {'score': score, 'issues': issues}

    def get_recommendation(self, overall_score, validation):
        """Generate go/no-go recommendation"""

        if overall_score >= 0.8:
            return {
                'decision': 'GO',
                'confidence': 'high',
                'rationale': 'Strategy scores highly across all dimensions. Proceed with full investment.',
                'next_steps': [
                    'Secure executive sponsorship',
                    'Allocate budget',
                    'Begin Phase 1 hiring',
                    'Initiate infrastructure setup'
                ]
            }
        elif overall_score >= 0.6:
            return {
                'decision': 'GO (with conditions)',
                'confidence': 'medium',
                'rationale': 'Strategy is viable but has gaps. Address issues before full commitment.',
                'next_steps': [
                    'Address identified gaps',
                    'Run pilot project to validate assumptions',
                    'Secure contingent budget approval',
                    'Re-validate after pilot'
                ]
            }
        else:
            return {
                'decision': 'NO-GO',
                'confidence': 'high',
                'rationale': 'Strategy has fundamental issues. Do not proceed without major revisions.',
                'next_steps': [
                    'Revise strategy to address critical gaps',
                    'Consider smaller pilot to test assumptions',
                    'Re-validate revised strategy',
                    'Consider alternative approaches'
                ]
            }
