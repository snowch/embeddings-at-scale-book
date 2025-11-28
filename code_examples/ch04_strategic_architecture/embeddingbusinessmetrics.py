# Code from Chapter 02
# Book: Embeddings at Scale


class EmbeddingBusinessMetrics:
    """Map embedding capabilities to business outcomes"""

    def __init__(self, business_context):
        self.context = business_context
        self.metric_map = {}

    def define_success_metrics(self):
        """Define measurable business outcomes"""

        if self.context == "ecommerce":
            return {
                "primary_metrics": [
                    {
                        "metric": "conversion_rate",
                        "baseline": 0.08,
                        "target": 0.12,
                        "timeline": "12mo",
                    },
                    {
                        "metric": "revenue_per_user",
                        "baseline": 420,
                        "target": 550,
                        "timeline": "18mo",
                    },
                    {"metric": "customer_ltv", "baseline": 850, "target": 1200, "timeline": "24mo"},
                ],
                "operational_metrics": [
                    {
                        "metric": "search_satisfaction",
                        "baseline": 3.2,
                        "target": 4.3,
                        "timeline": "6mo",
                    },
                    {
                        "metric": "zero_result_rate",
                        "baseline": 0.15,
                        "target": 0.03,
                        "timeline": "9mo",
                    },
                ],
            }

        elif self.context == "fraud_detection":
            return {
                "primary_metrics": [
                    {
                        "metric": "fraud_loss_rate",
                        "baseline": 0.0006,
                        "target": 0.00025,
                        "timeline": "18mo",
                    },
                    {
                        "metric": "false_positive_rate",
                        "baseline": 0.023,
                        "target": 0.004,
                        "timeline": "12mo",
                    },
                ],
                "operational_metrics": [
                    {
                        "metric": "detection_latency_ms",
                        "baseline": 250,
                        "target": 50,
                        "timeline": "6mo",
                    },
                    {
                        "metric": "new_pattern_adaptation_hours",
                        "baseline": 72,
                        "target": 2,
                        "timeline": "12mo",
                    },
                ],
            }

        elif self.context == "healthcare":
            return {
                "primary_metrics": [
                    {
                        "metric": "physician_research_hours_per_week",
                        "baseline": 4.3,
                        "target": 0.8,
                        "timeline": "18mo",
                    },
                    {
                        "metric": "diagnostic_accuracy_rare_diseases",
                        "baseline": 0.45,
                        "target": 0.75,
                        "timeline": "24mo",
                    },
                    {
                        "metric": "readmission_rate",
                        "baseline": 0.147,
                        "target": 0.134,
                        "timeline": "24mo",
                    },
                ],
                "operational_metrics": [
                    {
                        "metric": "time_to_correct_diagnosis_hours",
                        "baseline": 18.5,
                        "target": 14.2,
                        "timeline": "18mo",
                    }
                ],
            }
