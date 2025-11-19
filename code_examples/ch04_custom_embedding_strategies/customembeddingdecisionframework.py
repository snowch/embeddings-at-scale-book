# Code from Chapter 04
# Book: Embeddings at Scale

class CustomEmbeddingDecisionFramework:
    """
    Systematic framework for custom vs. fine-tune decision
    """

    def __init__(self):
        self.score_custom = 0
        self.score_finetune = 0
        self.factors = []

    def evaluate(self, context):
        """
        Evaluate whether to build custom or fine-tune

        Args:
            context: Dictionary with decision factors

        Returns:
            Recommendation with rationale
        """

        # Factor 1: Data availability
        if context['training_examples'] > 10_000_000:
            self.score_custom += 3
            self.factors.append("Massive data enables custom training")
        elif context['training_examples'] > 1_000_000:
            self.score_custom += 1
            self.score_finetune += 2
            self.factors.append("Substantial data supports both approaches")
        elif context['training_examples'] > 100_000:
            self.score_finetune += 3
            self.factors.append("Limited data favors fine-tuning")
        else:
            self.score_finetune += 5
            self.factors.append("Insufficient data for custom training")

        # Factor 2: Domain uniqueness
        domain_gap = context.get('domain_gap', 'medium')  # low, medium, high
        if domain_gap == 'high':
            # Highly specialized domain (e.g., genomics, specialized legal)
            self.score_custom += 3
            self.factors.append("High domain gap benefits from custom architecture")
        elif domain_gap == 'medium':
            # Some domain shift (e.g., medical, financial)
            self.score_finetune += 2
            self.factors.append("Medium domain gap well-suited for fine-tuning")
        else:
            # Close to pre-training domain
            self.score_finetune += 3
            self.factors.append("Low domain gap - fine-tuning sufficient")

        # Factor 3: Performance requirements
        perf_req = context.get('performance_requirement', 'medium')
        if perf_req == 'world_class':
            # Need state-of-the-art, no compromises
            self.score_custom += 3
            self.factors.append("World-class performance requires custom approach")
        elif perf_req == 'high':
            self.score_custom += 1
            self.score_finetune += 1
            self.factors.append("High performance achievable with either approach")
        else:
            self.score_finetune += 2
            self.factors.append("Standard performance met by fine-tuning")

        # Factor 4: Specialized requirements
        special_reqs = context.get('specialized_requirements', [])
        # Examples: 'multilingual', 'multi-modal', 'low-latency', 'tiny-model', 'interpretable'

        if 'multi-modal' in special_reqs and not context.get('multimodal_pretrained_available', True):
            self.score_custom += 2
            self.factors.append("Custom multi-modal architecture needed")

        if 'tiny-model' in special_reqs:
            # Need very small models (e.g., edge deployment)
            self.score_custom += 2
            self.factors.append("Model size constraints favor custom architecture")

        if 'interpretable' in special_reqs:
            self.score_custom += 1
            self.factors.append("Interpretability easier with custom design")

        # Factor 5: Budget and timeline
        budget = context.get('annual_budget', 0)
        timeline_months = context.get('timeline_months', 12)

        if budget < 50_000 or timeline_months < 3:
            self.score_finetune += 4
            self.factors.append("Budget/timeline constraints favor fine-tuning")
        elif budget > 1_000_000 and timeline_months > 12:
            self.score_custom += 2
            self.factors.append("Sufficient resources for custom development")

        # Factor 6: Team capability
        team_capability = context.get('team_capability', 'medium')
        if team_capability == 'high':
            # Team has published papers, trained large models before
            self.score_custom += 1
        elif team_capability == 'low':
            self.score_finetune += 2
            self.factors.append("Limited ML expertise favors fine-tuning")

        # Factor 7: Competitive advantage
        competitive_impact = context.get('competitive_advantage', 'medium')
        if competitive_impact == 'critical':
            # Embeddings ARE your competitive moat
            self.score_custom += 3
            self.factors.append("Embeddings as competitive moat justify custom investment")
        elif competitive_impact == 'high':
            self.score_custom += 1
            self.score_finetune += 1
        else:
            self.score_finetune += 1

        # Make recommendation
        return self._generate_recommendation()

    def _generate_recommendation(self):
        """Generate final recommendation"""

        if self.score_custom > self.score_finetune + 5:
            return {
                'recommendation': 'BUILD_CUSTOM',
                'confidence': 'high',
                'approach': 'Train custom embedding model from scratch (Level 4)',
                'rationale': self.factors,
                'estimated_effort': '6-18 months',
                'estimated_cost': '$500K-$5M',
                'next_steps': [
                    'Assemble ML research team (5-10 people)',
                    'Conduct architecture exploration',
                    'Prepare large-scale training infrastructure',
                    'Plan 12-18 month roadmap'
                ]
            }
        elif self.score_custom > self.score_finetune + 2:
            return {
                'recommendation': 'BUILD_CUSTOM',
                'confidence': 'medium',
                'approach': 'Custom model, but consider hybrid approach',
                'rationale': self.factors,
                'caveat': 'Start with fine-tuning to establish baseline, then build custom if needed',
                'estimated_effort': '3-12 months',
                'estimated_cost': '$100K-$1M'
            }
        elif self.score_finetune > self.score_custom + 5:
            return {
                'recommendation': 'FINE_TUNE',
                'confidence': 'high',
                'approach': 'Full model fine-tuning (Level 3)',
                'rationale': self.factors,
                'estimated_effort': '1-3 months',
                'estimated_cost': '$25K-$150K',
                'next_steps': [
                    'Select base model (BERT, RoBERTa, Sentence-BERT)',
                    'Prepare labeled training data (100K+ examples)',
                    'Set up fine-tuning pipeline',
                    'Benchmark against frozen pre-trained baseline'
                ]
            }
        else:
            return {
                'recommendation': 'FINE_TUNE',
                'confidence': 'medium',
                'approach': 'Start with fine-tuning, keep custom as option',
                'rationale': self.factors,
                'caveat': 'Marginal difference - fine-tune first, measure gaps, then decide',
                'estimated_effort': '1-6 months',
                'estimated_cost': '$50K-$500K'
            }


# Example usage: E-commerce product search
ecommerce_context = {
    'training_examples': 5_000_000,  # 5M product-query pairs
    'domain_gap': 'medium',  # E-commerce is somewhat specialized
    'performance_requirement': 'high',  # Directly impacts revenue
    'specialized_requirements': ['multi-modal'],  # Products have images
    'annual_budget': 500_000,
    'timeline_months': 6,
    'team_capability': 'medium',
    'competitive_advantage': 'high'  # Search quality is competitive differentiator
}

framework = CustomEmbeddingDecisionFramework()
recommendation = framework.evaluate(ecommerce_context)

print(f"Recommendation: {recommendation['recommendation']}")
print(f"Approach: {recommendation['approach']}")
print(f"\nRationale:")
for factor in recommendation['rationale']:
    print(f"  - {factor}")
