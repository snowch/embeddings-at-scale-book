# Code from Chapter 02
# Book: Embeddings at Scale


class VectorDBEvaluation:
    """Evaluate commercial vector DB vendors"""

    def evaluate_vendor(self, vendor_name):
        """Comprehensive vendor evaluation"""
        criteria = {
            "scale": {
                "max_vectors": None,  # How many vectors supported?
                "max_qps": None,  # Query throughput?
                "score": 0,  # 0-10
            },
            "performance": {"p50_latency_ms": None, "p99_latency_ms": None, "score": 0},
            "cost": {
                "storage_cost_per_gb": None,
                "query_cost_per_million": None,
                "total_annual_cost_estimate": None,
                "score": 0,
            },
            "features": {
                "multi_vector_support": False,
                "hybrid_search": False,  # Vector + keyword
                "filtering": False,  # Metadata filtering
                "multi_tenancy": False,
                "real_time_updates": False,
                "score": 0,
            },
            "operations": {
                "uptime_sla": None,  # e.g., 99.9%
                "backup_restore": False,
                "monitoring_tools": False,
                "multi_region": False,
                "score": 0,
            },
            "vendor_risk": {
                "years_in_business": None,
                "funding": None,
                "customer_count": None,
                "open_source": False,
                "score": 0,
            },
        }

        # Calculate overall score
        overall_score = sum(c["score"] for c in criteria.values()) / len(criteria)

        return {
            "vendor": vendor_name,
            "overall_score": overall_score,
            "criteria": criteria,
            "recommendation": "recommended" if overall_score > 7 else "not recommended",
        }
