# Code from Chapter 25
# Book: Embeddings at Scale

"""
Comprehensive Cost Tracking and Optimization for Embedding Systems

Architecture:
1. Resource metering: Track compute, storage, network, memory usage
2. Cost attribution: Assign costs to teams, projects, users
3. Cost analysis: Identify top cost drivers and optimization opportunities
4. Budget management: Set budgets, alert on overruns, forecast spending
5. Optimization: Automated recommendations for cost reduction

Cost categories:
- Training: Model training compute (GPU-hours × $per-hour)
- Inference: Embedding generation (CPU/GPU-hours, API calls)
- Storage: Vector storage, indexes, backups
- Query: Similarity search compute (CPU/GPU-hours)
- Network: Data transfer, replication bandwidth
- Cache: In-memory storage costs

Optimization strategies:
- Cache hot embeddings to reduce database queries
- Use quantization/compression to reduce storage
- Batch operations to amortize overhead
- Right-size infrastructure to actual usage
- Implement tiered storage (hot/warm/cold)
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ResourceUsage:
    """Resource usage for a single operation"""
    timestamp: datetime
    operation_type: str  # "training", "inference", "query", "storage"

    # Compute resources
    cpu_hours: float = 0.0
    gpu_hours: float = 0.0
    memory_gb_hours: float = 0.0

    # Storage resources
    storage_gb_hours: float = 0.0

    # Network resources
    network_gb_ingress: float = 0.0
    network_gb_egress: float = 0.0

    # Operation details
    num_operations: int = 1
    embeddings_generated: int = 0
    queries_executed: int = 0

    # Attribution
    team: str = "unknown"
    project: str = "unknown"
    user: str = "unknown"
    environment: str = "production"  # "production", "staging", "development"

@dataclass
class CostRates:
    """Cost rates per resource unit (USD)"""
    # Compute costs (per hour)
    cpu_hour: float = 0.10          # $0.10/CPU-hour (typical cloud instance)
    gpu_hour: float = 2.50          # $2.50/GPU-hour (A100 GPU)
    memory_gb_hour: float = 0.01    # $0.01/GB-hour of memory

    # Storage costs (per GB-month, converted to per hour)
    storage_standard_gb_hour: float = 0.023 / 730  # $0.023/GB-month ÷ 730 hours/month
    storage_ssd_gb_hour: float = 0.10 / 730        # $0.10/GB-month for SSD
    storage_memory_gb_hour: float = 0.15 / 730     # $0.15/GB-month for in-memory

    # Network costs (per GB)
    network_ingress_gb: float = 0.0   # Usually free
    network_egress_gb: float = 0.09   # $0.09/GB egress (typical cloud)

    # API costs (for managed services)
    embedding_api_call: float = 0.0001  # $0.0001 per embedding generated
    search_api_call: float = 0.00001    # $0.00001 per search query

@dataclass
class CostSummary:
    """Cost summary for a time period"""
    start_time: datetime
    end_time: datetime

    # Total costs by category
    training_cost: float = 0.0
    inference_cost: float = 0.0
    query_cost: float = 0.0
    storage_cost: float = 0.0
    network_cost: float = 0.0
    total_cost: float = 0.0

    # Cost breakdown by attribution
    cost_by_team: Dict[str, float] = field(default_factory=dict)
    cost_by_project: Dict[str, float] = field(default_factory=dict)
    cost_by_environment: Dict[str, float] = field(default_factory=dict)

    # Usage statistics
    total_embeddings_generated: int = 0
    total_queries_executed: int = 0
    total_gpu_hours: float = 0.0
    total_cpu_hours: float = 0.0
    total_storage_gb_hours: float = 0.0

    # Per-operation costs
    cost_per_embedding: float = 0.0
    cost_per_query: float = 0.0

class CostTracker:
    """
    Comprehensive cost tracking and optimization system
    
    Tracks resource usage, calculates costs, attributes to teams/projects,
    and provides optimization recommendations.
    """

    def __init__(
        self,
        cost_rates: Optional[CostRates] = None,
        budget_limits: Optional[Dict[str, float]] = None,
        alert_callback: Optional[callable] = None
    ):
        """
        Initialize cost tracker
        
        Args:
            cost_rates: Cost rates per resource unit
            budget_limits: Budget limits by team/project
            alert_callback: Function to call when budget exceeded
        """
        self.cost_rates = cost_rates or CostRates()
        self.budget_limits = budget_limits or {}
        self.alert_callback = alert_callback

        # Usage history
        self.usage_history: List[ResourceUsage] = []

        # Aggregated costs
        self.current_period_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        self.cost_cache: Dict[str, CostSummary] = {}

    def record_usage(self, usage: ResourceUsage):
        """Record resource usage"""
        self.usage_history.append(usage)

        # Check budget limits
        self._check_budgets()

    def record_training(
        self,
        duration_hours: float,
        num_gpus: int,
        memory_gb: float,
        team: str,
        project: str,
        embeddings_generated: int = 0
    ):
        """Convenience method to record model training"""
        usage = ResourceUsage(
            timestamp=datetime.now(),
            operation_type="training",
            gpu_hours=duration_hours * num_gpus,
            cpu_hours=duration_hours * 4,  # Assume 4 CPUs per GPU
            memory_gb_hours=memory_gb * duration_hours,
            embeddings_generated=embeddings_generated,
            team=team,
            project=project
        )
        self.record_usage(usage)

    def record_inference(
        self,
        num_embeddings: int,
        use_gpu: bool,
        duration_hours: float,
        team: str,
        project: str,
        user: str = "unknown"
    ):
        """Convenience method to record embedding generation"""
        usage = ResourceUsage(
            timestamp=datetime.now(),
            operation_type="inference",
            gpu_hours=duration_hours if use_gpu else 0.0,
            cpu_hours=duration_hours if not use_gpu else 0.0,
            embeddings_generated=num_embeddings,
            team=team,
            project=project,
            user=user
        )
        self.record_usage(usage)

    def record_query(
        self,
        num_queries: int,
        duration_hours: float,
        cache_hit_rate: float,
        team: str,
        project: str,
        user: str = "unknown"
    ):
        """Convenience method to record query execution"""
        # Adjust CPU hours by cache hit rate (cache hits are cheaper)
        effective_duration = duration_hours * (1 - cache_hit_rate * 0.9)  # Cache hits save 90% compute

        usage = ResourceUsage(
            timestamp=datetime.now(),
            operation_type="query",
            cpu_hours=effective_duration,
            queries_executed=num_queries,
            team=team,
            project=project,
            user=user
        )
        self.record_usage(usage)

    def record_storage(
        self,
        storage_gb: float,
        duration_hours: float,
        storage_type: str,  # "standard", "ssd", "memory"
        team: str,
        project: str
    ):
        """Convenience method to record storage costs"""
        usage = ResourceUsage(
            timestamp=datetime.now(),
            operation_type="storage",
            storage_gb_hours=storage_gb * duration_hours,
            team=team,
            project=project
        )
        # Adjust rate based on storage type
        if storage_type == "ssd":
            usage.storage_gb_hours *= (self.cost_rates.storage_ssd_gb_hour / self.cost_rates.storage_standard_gb_hour)
        elif storage_type == "memory":
            usage.storage_gb_hours *= (self.cost_rates.storage_memory_gb_hour / self.cost_rates.storage_standard_gb_hour)

        self.record_usage(usage)

    def calculate_cost(self, usage: ResourceUsage) -> float:
        """Calculate cost for a single usage record"""
        cost = 0.0

        # Compute costs
        cost += usage.cpu_hours * self.cost_rates.cpu_hour
        cost += usage.gpu_hours * self.cost_rates.gpu_hour
        cost += usage.memory_gb_hours * self.cost_rates.memory_gb_hour

        # Storage costs
        cost += usage.storage_gb_hours * self.cost_rates.storage_standard_gb_hour

        # Network costs
        cost += usage.network_gb_ingress * self.cost_rates.network_ingress_gb
        cost += usage.network_gb_egress * self.cost_rates.network_egress_gb

        # API costs (if using managed services)
        if usage.embeddings_generated > 0:
            cost += usage.embeddings_generated * self.cost_rates.embedding_api_call
        if usage.queries_executed > 0:
            cost += usage.queries_executed * self.cost_rates.search_api_call

        return cost

    def get_cost_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> CostSummary:
        """
        Get cost summary for time period
        
        Args:
            start_time: Start of period (default: start of current month)
            end_time: End of period (default: now)
            
        Returns:
            CostSummary with aggregated costs
        """
        if start_time is None:
            start_time = self.current_period_start
        if end_time is None:
            end_time = datetime.now()

        # Filter usage records to time period
        period_usage = [u for u in self.usage_history
                       if start_time <= u.timestamp <= end_time]

        # Initialize summary
        summary = CostSummary(start_time=start_time, end_time=end_time)

        # Aggregate costs
        for usage in period_usage:
            cost = self.calculate_cost(usage)

            # Categorize by operation type
            if usage.operation_type == "training":
                summary.training_cost += cost
            elif usage.operation_type == "inference":
                summary.inference_cost += cost
            elif usage.operation_type == "query":
                summary.query_cost += cost
            elif usage.operation_type == "storage":
                summary.storage_cost += cost

            # Add network costs separately
            network_cost = (usage.network_gb_ingress * self.cost_rates.network_ingress_gb +
                           usage.network_gb_egress * self.cost_rates.network_egress_gb)
            summary.network_cost += network_cost

            summary.total_cost += cost

            # Attribution
            summary.cost_by_team[usage.team] = summary.cost_by_team.get(usage.team, 0) + cost
            summary.cost_by_project[usage.project] = summary.cost_by_project.get(usage.project, 0) + cost
            summary.cost_by_environment[usage.environment] = summary.cost_by_environment.get(usage.environment, 0) + cost

            # Usage statistics
            summary.total_embeddings_generated += usage.embeddings_generated
            summary.total_queries_executed += usage.queries_executed
            summary.total_gpu_hours += usage.gpu_hours
            summary.total_cpu_hours += usage.cpu_hours
            summary.total_storage_gb_hours += usage.storage_gb_hours

        # Per-operation costs
        if summary.total_embeddings_generated > 0:
            summary.cost_per_embedding = summary.inference_cost / summary.total_embeddings_generated
        if summary.total_queries_executed > 0:
            summary.cost_per_query = summary.query_cost / summary.total_queries_executed

        return summary

    def _check_budgets(self):
        """Check if any budget limits exceeded"""
        current_month_summary = self.get_cost_summary()

        alerts = []

        # Check team budgets
        for team, budget in self.budget_limits.items():
            if team.startswith("team:"):
                team_name = team[5:]
                team_cost = current_month_summary.cost_by_team.get(team_name, 0)
                if team_cost > budget:
                    alerts.append(f"Team '{team_name}' exceeded budget: ${team_cost:.2f} > ${budget:.2f}")

        # Check project budgets
        for project, budget in self.budget_limits.items():
            if project.startswith("project:"):
                project_name = project[8:]
                project_cost = current_month_summary.cost_by_project.get(project_name, 0)
                if project_cost > budget:
                    alerts.append(f"Project '{project_name}' exceeded budget: ${project_cost:.2f} > ${budget:.2f}")

        # Check total budget
        if "total" in self.budget_limits:
            if current_month_summary.total_cost > self.budget_limits["total"]:
                alerts.append(f"Total budget exceeded: ${current_month_summary.total_cost:.2f} > ${self.budget_limits['total']:.2f}")

        # Trigger alerts
        if alerts and self.alert_callback:
            self.alert_callback(alerts, current_month_summary)

    def generate_cost_report(self, summary: CostSummary) -> str:
        """Generate human-readable cost report"""
        report = f"""
Embedding System Cost Report
=============================
Period: {summary.start_time.strftime('%Y-%m-%d')} to {summary.end_time.strftime('%Y-%m-%d')}

Total Cost: ${summary.total_cost:.2f}

Cost Breakdown by Category:
---------------------------
Training:     ${summary.training_cost:>10.2f} ({summary.training_cost/max(summary.total_cost, 1)*100:>5.1f}%)
Inference:    ${summary.inference_cost:>10.2f} ({summary.inference_cost/max(summary.total_cost, 1)*100:>5.1f}%)
Query:        ${summary.query_cost:>10.2f} ({summary.query_cost/max(summary.total_cost, 1)*100:>5.1f}%)
Storage:      ${summary.storage_cost:>10.2f} ({summary.storage_cost/max(summary.total_cost, 1)*100:>5.1f}%)
Network:      ${summary.network_cost:>10.2f} ({summary.network_cost/max(summary.total_cost, 1)*100:>5.1f}%)

Resource Usage:
--------------
Embeddings Generated: {summary.total_embeddings_generated:,}
Queries Executed:     {summary.total_queries_executed:,}
GPU Hours:            {summary.total_gpu_hours:.1f}
CPU Hours:            {summary.total_cpu_hours:.1f}
Storage GB-Hours:     {summary.total_storage_gb_hours:.1f}

Per-Operation Costs:
-------------------
Cost per Embedding:   ${summary.cost_per_embedding:.6f}
Cost per Query:       ${summary.cost_per_query:.6f}

Cost by Team:
------------
"""
        for team, cost in sorted(summary.cost_by_team.items(), key=lambda x: x[1], reverse=True):
            report += f"{team:>20s}: ${cost:>10.2f} ({cost/max(summary.total_cost, 1)*100:>5.1f}%)\n"

        report += "\nCost by Project:\n"
        report += "----------------\n"
        for project, cost in sorted(summary.cost_by_project.items(), key=lambda x: x[1], reverse=True):
            report += f"{project:>20s}: ${cost:>10.2f} ({cost/max(summary.total_cost, 1)*100:>5.1f}%)\n"

        return report

    def get_optimization_recommendations(self, summary: CostSummary) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []

        # High GPU costs
        if summary.training_cost > summary.total_cost * 0.5:
            recommendations.append(
                f"Training costs are {summary.training_cost/max(summary.total_cost, 1)*100:.1f}% of total. "
                "Consider: (1) Using spot instances for training (60-90% savings), "
                "(2) Optimizing hyperparameters to reduce training time, "
                "(3) Using smaller models or transfer learning"
            )

        # High query costs
        if summary.query_cost > summary.total_cost * 0.3:
            recommendations.append(
                f"Query costs are {summary.query_cost/max(summary.total_cost, 1)*100:.1f}% of total. "
                "Consider: (1) Increasing cache hit rate through better caching strategies, "
                "(2) Using approximate nearest neighbor (ANN) algorithms, "
                "(3) Implementing query result caching"
            )

        # High storage costs
        if summary.storage_cost > summary.total_cost * 0.3:
            recommendations.append(
                f"Storage costs are {summary.storage_cost/max(summary.total_cost, 1)*100:.1f}% of total. "
                "Consider: (1) Implementing vector quantization (4-8× compression), "
                "(2) Using tiered storage (hot/warm/cold), "
                "(3) Reducing embedding dimensionality through PCA"
            )

        # Expensive per-query costs
        if summary.cost_per_query > 0.0001:
            recommendations.append(
                f"Cost per query (${summary.cost_per_query:.6f}) is high. "
                "Consider: (1) Batching queries to amortize overhead, "
                "(2) Optimizing index structure (HNSW parameters), "
                "(3) Using cheaper compute instances for queries"
            )

        # Expensive per-embedding costs
        if summary.cost_per_embedding > 0.001:
            recommendations.append(
                f"Cost per embedding (${summary.cost_per_embedding:.6f}) is high. "
                "Consider: (1) Batch embedding generation, "
                "(2) Using distilled or smaller models, "
                "(3) Caching embeddings for frequently accessed items"
            )

        if not recommendations:
            recommendations.append("Cost structure looks efficient. Continue monitoring for optimization opportunities.")

        return recommendations

    def forecast_monthly_cost(self) -> float:
        """Forecast end-of-month cost based on current usage"""
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        days_elapsed = (now - month_start).days + 1

        # Get cost for elapsed days
        current_summary = self.get_cost_summary(start_time=month_start, end_time=now)

        # Extrapolate to full month
        days_in_month = 30  # Approximate
        forecasted_cost = current_summary.total_cost * (days_in_month / days_elapsed)

        return forecasted_cost


# Example usage
if __name__ == "__main__":
    # Initialize cost tracker with budgets
    tracker = CostTracker(
        budget_limits={
            "total": 10000.0,           # $10k/month total
            "team:ml-platform": 5000.0,  # $5k/month for ML platform team
            "team:search": 3000.0,       # $3k/month for search team
            "project:recommendation": 2000.0  # $2k/month for recommendation project
        },
        alert_callback=lambda alerts, summary: print("\n🚨 BUDGET ALERT:\n" + "\n".join(alerts))
    )

    # Simulate various operations
    print("Simulating embedding system operations...")

    # Training
    tracker.record_training(
        duration_hours=24.0,
        num_gpus=8,
        memory_gb=256,
        team="ml-platform",
        project="recommendation",
        embeddings_generated=0
    )
    print("Recorded training: 24 hours × 8 GPUs")

    # Inference
    tracker.record_inference(
        num_embeddings=1000000,
        use_gpu=True,
        duration_hours=2.0,
        team="ml-platform",
        project="recommendation"
    )
    print("Recorded inference: 1M embeddings generated")

    # Queries
    tracker.record_query(
        num_queries=10000000,
        duration_hours=5.0,
        cache_hit_rate=0.75,
        team="search",
        project="semantic-search"
    )
    print("Recorded queries: 10M queries with 75% cache hit rate")

    # Storage
    tracker.record_storage(
        storage_gb=5000,
        duration_hours=24 * 30,  # 1 month
        storage_type="ssd",
        team="ml-platform",
        project="recommendation"
    )
    print("Recorded storage: 5TB SSD for 1 month")

    # Generate cost report
    print("\n" + "="*70)
    summary = tracker.get_cost_summary()
    print(tracker.generate_cost_report(summary))

    # Optimization recommendations
    print("\nCost Optimization Recommendations:")
    print("="*70)
    recommendations = tracker.get_optimization_recommendations(summary)
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")

    # Forecast
    forecasted = tracker.forecast_monthly_cost()
    print(f"\n\nForecasted Monthly Cost: ${forecasted:.2f}")
    if "total" in tracker.budget_limits:
        budget = tracker.budget_limits["total"]
        if forecasted > budget:
            print(f"⚠️ WARNING: Forecasted cost exceeds budget by ${forecasted - budget:.2f}")
        else:
            print(f"✓ Forecasted cost within budget (${budget - forecasted:.2f} remaining)")
