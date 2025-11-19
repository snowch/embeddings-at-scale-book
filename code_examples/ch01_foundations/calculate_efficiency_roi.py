# Code from Chapter 01
# Book: Embeddings at Scale

def calculate_efficiency_roi(process_name, current_time, target_time,
                            annual_volume, hourly_cost):
    """
    Calculate ROI from process automation/augmentation

    Example: Document review
    - current_time: 4 hours per document
    - target_time: 1 hour per document (AI-augmented)
    - annual_volume: 10,000 documents
    - hourly_cost: $500
    """
    time_saved_per_unit = current_time - target_time
    annual_hours_saved = time_saved_per_unit * annual_volume
    annual_savings = annual_hours_saved * hourly_cost

    # Quality improvements (fewer errors, rework)
    # Conservative estimate: 5% reduction in rework
    rework_savings = annual_volume * current_time * 0.05 * hourly_cost

    return {
        'annual_hours_saved': annual_hours_saved,
        'direct_savings': annual_savings,
        'quality_savings': rework_savings,
        'total_annual_benefit': annual_savings + rework_savings
    }
