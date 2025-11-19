# Code from Chapter 01
# Book: Embeddings at Scale

def calculate_ltv_improvement(current_ltv, churn_reduction, customer_base):
    """
    Embedding improvements often reduce churn

    Example:
    - Better search → customers find products → higher satisfaction → lower churn
    - Better recommendations → more value → stickier product
    """
    # Simplified: churn reduction directly impacts LTV
    improved_ltv = current_ltv * (1 + churn_reduction)
    ltv_increase = improved_ltv - current_ltv

    # Apply to customer base (existing + new customers)
    annual_value = ltv_increase * customer_base * 0.25  # 25% annual customer base turnover

    return {
        'ltv_improvement': ltv_increase,
        'annual_value': annual_value
    }
