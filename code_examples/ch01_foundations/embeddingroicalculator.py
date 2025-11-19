# Code from Chapter 01
# Book: Embeddings at Scale

class EmbeddingROICalculator:
    """Complete ROI framework for embedding projects"""

    def __init__(self, project_name):
        self.project_name = project_name
        self.benefits = {}
        self.costs = {}

    def add_search_benefit(self, **kwargs):
        """Add search improvement benefits"""
        benefit = calculate_search_roi(**kwargs)
        self.benefits['search'] = benefit

    def add_efficiency_benefit(self, **kwargs):
        """Add operational efficiency benefits"""
        benefit = calculate_efficiency_roi(**kwargs)
        self.benefits['efficiency'] = benefit

    def add_fraud_benefit(self, **kwargs):
        """Add fraud/risk reduction benefits"""
        benefit = calculate_fraud_roi(**kwargs)
        self.benefits['fraud'] = benefit

    def add_ltv_benefit(self, **kwargs):
        """Add customer LTV improvement benefits"""
        benefit = calculate_ltv_improvement(**kwargs)
        self.benefits['ltv'] = benefit

    def add_costs(self, implementation, annual_operating, annual_data_costs=0):
        """Add project costs"""
        self.costs = {
            'implementation': implementation,
            'annual_operating': annual_operating,
            'annual_data': annual_data_costs
        }

    def calculate_total_roi(self, years=5, probability_of_success=0.8):
        """Calculate complete ROI"""
        # Sum all benefits
        total_annual_benefit = sum(
            b.get('total_annual_benefit', b.get('annual_value', 0))
            for b in self.benefits.values()
        )

        # Calculate risk-adjusted ROI
        roi = risk_adjusted_roi(
            potential_benefit=total_annual_benefit,
            probability_of_success=probability_of_success,
            implementation_cost=self.costs['implementation'],
            annual_operating_cost=(
                self.costs['annual_operating'] +
                self.costs['annual_data']
            ),
            years=years
        )

        return {
            'project_name': self.project_name,
            'total_annual_benefit': total_annual_benefit,
            'risk_adjusted_annual_benefit': roi['expected_annual_benefit'],
            'implementation_cost': self.costs['implementation'],
            'annual_operating_cost': self.costs['annual_operating'],
            'npv': roi['npv'],
            'roi_percent': roi['roi_percent'],
            'payback_period_years': roi['payback_period_years'],
            'benefit_breakdown': self.benefits
        }
