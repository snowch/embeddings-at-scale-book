# Code from Chapter 01
# Book: Embeddings at Scale


def calculate_search_roi(current_metrics, target_metrics, users, avg_transaction_value):
    """
    Calculate ROI from search improvements

    current_metrics: {
        'conversion_rate': 0.08,  # 8% of searches convert
        'avg_time_to_find': 3.5,  # minutes
        'zero_result_rate': 0.15  # 15% of searches find nothing
    }

    target_metrics: {
        'conversion_rate': 0.12,  # Conservative 50% improvement
        'avg_time_to_find': 1.5,  # 60% reduction
        'zero_result_rate': 0.03  # 80% reduction
    }
    """
    # Annual searches
    annual_searches = users * 50  # 50 searches per user per year

    # Revenue impact from improved conversion
    current_conversions = annual_searches * current_metrics["conversion_rate"]
    target_conversions = annual_searches * target_metrics["conversion_rate"]
    additional_conversions = target_conversions - current_conversions
    additional_revenue = additional_conversions * avg_transaction_value

    # Time saved (user satisfaction + efficiency)
    time_saved_per_search = current_metrics["avg_time_to_find"] - target_metrics["avg_time_to_find"]
    total_time_saved = annual_searches * time_saved_per_search / 60  # hours

    # Reduced abandonment
    current_abandonments = annual_searches * current_metrics["zero_result_rate"]
    target_abandonments = annual_searches * target_metrics["zero_result_rate"]
    saved_abandonments = current_abandonments - target_abandonments
    recovered_revenue = saved_abandonments * 0.3 * avg_transaction_value  # 30% recovery rate

    return {
        "additional_revenue": additional_revenue,
        "time_saved_hours": total_time_saved,
        "recovered_revenue": recovered_revenue,
        "total_annual_benefit": additional_revenue + recovered_revenue,
    }
