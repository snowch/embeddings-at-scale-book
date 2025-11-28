import numpy as np

# Code from Chapter 03
# Book: Embeddings at Scale


class LoadTestingStrategy:
    """Plan and execute load tests"""

    def load_test_scenarios(self):
        """Different load patterns to test"""

        return {
            "steady_state": {
                "description": "Constant QPS at target load",
                "pattern": "Maintain 100K QPS for 1 hour",
                "goal": "Verify system handles normal load",
                "success_criteria": [
                    "p99 latency < 100ms",
                    "No errors",
                    "CPU/memory stable",
                    "No memory leaks",
                ],
            },
            "ramp_up": {
                "description": "Gradually increase load",
                "pattern": "0 → 200K QPS over 30 minutes",
                "goal": "Find breaking point",
                "success_criteria": [
                    "Identify max sustainable QPS",
                    "Graceful degradation (not crash)",
                    "Circuit breakers engage correctly",
                ],
            },
            "spike": {
                "description": "Sudden traffic burst",
                "pattern": "50K → 500K QPS for 5 minutes → back to 50K",
                "goal": "Test autoscaling and elasticity",
                "success_criteria": [
                    "System scales up within 2 minutes",
                    "Temporary latency spike acceptable",
                    "Recovery to normal after spike",
                ],
            },
            "sustained_peak": {
                "description": "Extended period at peak load",
                "pattern": "150K QPS for 8 hours",
                "goal": "Test for memory leaks, resource exhaustion",
                "success_criteria": [
                    "No degradation over time",
                    "Memory usage stable",
                    "Disk space not growing unbounded",
                ],
            },
            "thundering_herd": {
                "description": "Coordinated simultaneous requests",
                "pattern": "1M clients all query at same time",
                "goal": "Test queueing and overload handling",
                "success_criteria": [
                    "Queue depth controlled",
                    "Load shedding prevents cascade",
                    "Graceful degradation",
                ],
            },
            "geographic_distribution": {
                "description": "Load from multiple regions",
                "pattern": "Queries from US, EU, APAC simultaneously",
                "goal": "Test multi-region routing",
                "success_criteria": [
                    "Queries route to nearest region",
                    "Cross-region failover works",
                    "Latency within SLA per region",
                ],
            },
        }

    def capacity_planning_model(self, expected_qps, growth_rate_per_year):
        """Model future capacity needs"""

        # Current capacity
        qps_per_shard = 10_000
        current_shards = int(np.ceil(expected_qps / qps_per_shard))

        # Headroom for spikes (2x)
        current_shards_with_headroom = current_shards * 2

        # Future projections
        projections = []
        for year in range(1, 4):  # 3-year plan
            future_qps = expected_qps * ((1 + growth_rate_per_year) ** year)
            future_shards = int(np.ceil(future_qps / qps_per_shard)) * 2

            projections.append(
                {
                    "year": year,
                    "expected_qps": future_qps,
                    "required_shards": future_shards,
                    "new_shards_needed": future_shards - current_shards_with_headroom,
                }
            )

        return {
            "current_capacity": {"qps": expected_qps, "shards": current_shards_with_headroom},
            "projections": projections,
            "recommendation": f"Plan for {projections[-1]['required_shards']} shards by year 3",
        }


# Example capacity plan
planner = LoadTestingStrategy()
capacity_plan = planner.capacity_planning_model(
    expected_qps=100_000,
    growth_rate_per_year=0.5,  # 50% YoY growth
)

print("Capacity Planning:")
for projection in capacity_plan["projections"]:
    print(f"  Year {projection['year']}: {projection['expected_qps']:,.0f} QPS")
    print(f"    → {projection['required_shards']} shards needed")
