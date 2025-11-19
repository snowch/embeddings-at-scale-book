# Code from Chapter 01
# Book: Embeddings at Scale


def calculate_fraud_roi(
    current_loss_rate,
    target_loss_rate,
    annual_transaction_volume,
    false_positive_rate_current,
    false_positive_rate_target,
    cost_per_false_positive,
):
    """
    Calculate ROI from improved fraud detection
    """
    # Direct fraud loss reduction
    current_losses = annual_transaction_volume * current_loss_rate
    target_losses = annual_transaction_volume * target_loss_rate
    fraud_savings = current_losses - target_losses

    # False positive reduction
    annual_transactions = annual_transaction_volume / 100  # Assume $100 avg transaction
    current_fp = annual_transactions * false_positive_rate_current
    target_fp = annual_transactions * false_positive_rate_target
    fp_reduction = current_fp - target_fp
    fp_savings = fp_reduction * cost_per_false_positive

    return {
        "fraud_loss_reduction": fraud_savings,
        "false_positive_savings": fp_savings,
        "total_annual_benefit": fraud_savings + fp_savings,
    }
