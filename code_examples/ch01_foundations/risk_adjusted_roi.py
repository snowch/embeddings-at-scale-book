# Code from Chapter 01
# Book: Embeddings at Scale

def risk_adjusted_roi(potential_benefit, probability_of_success,
                     implementation_cost, annual_operating_cost,
                     years=5):
    """
    Calculate risk-adjusted ROI

    probability_of_success:
        - High certainty (proven use case, good data): 0.8-0.9
        - Medium certainty (proven use case, decent data): 0.6-0.7
        - Low certainty (novel use case or poor data): 0.3-0.5
    """
    expected_annual_benefit = potential_benefit * probability_of_success

    # NPV calculation
    discount_rate = 0.15  # 15% discount rate
    npv = -implementation_cost
    for year in range(1, years + 1):
        npv += (expected_annual_benefit - annual_operating_cost) / (1 + discount_rate) ** year

    roi = (npv / implementation_cost) * 100
    payback_period = implementation_cost / (expected_annual_benefit - annual_operating_cost)

    return {
        'expected_annual_benefit': expected_annual_benefit,
        'npv': npv,
        'roi_percent': roi,
        'payback_period_years': payback_period
    }
