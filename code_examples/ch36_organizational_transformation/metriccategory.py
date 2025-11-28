# Code from Chapter 27
# Book: Embeddings at Scale

"""
Comprehensive Embedding System Metrics Framework

Architecture:
1. Technical metrics: Model quality, performance, scale
2. Operational metrics: Availability, efficiency, cost
3. User metrics: Satisfaction, adoption, engagement
4. Business metrics: Revenue, efficiency, strategic value
5. Metric relationships: Leading → lagging indicator chains
6. Dashboards: Role-specific views (engineer, PM, exec)
7. Alerting: Automated detection of anomalies

Metric categories:
- Model quality: Accuracy, embedding coherence, drift
- Performance: Latency (p50/p99), throughput, error rate
- Infrastructure: CPU/GPU utilization, memory, cost
- User experience: CTR, search success, dwell time
- Business: Revenue impact, cost savings, competitive advantage

Success criteria:
- Technical: Meet SLAs consistently
- Operational: High availability, controlled costs
- User: Improved satisfaction and engagement
- Business: Positive ROI within target timeframe
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional


class MetricCategory(Enum):
    """Metric categories"""

    TECHNICAL = "technical"
    OPERATIONAL = "operational"
    USER = "user"
    BUSINESS = "business"


class MetricType(Enum):
    """Leading vs lagging indicators"""

    LEADING = "leading"  # Predicts future outcomes
    LAGGING = "lagging"  # Measures outcomes
    COINCIDENT = "coincident"  # Real-time indicator


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Metric:
    """
    Performance metric definition

    Attributes:
        name: Metric identifier
        category: Category (technical, operational, user, business)
        metric_type: Leading, lagging, or coincident
        description: What this metric measures
        unit: Unit of measurement
        target: Target value
        threshold_warning: Warning threshold
        threshold_critical: Critical threshold
        calculation: How to compute this metric
        review_frequency: How often to review (daily, weekly, etc.)
    """

    name: str
    category: MetricCategory
    metric_type: MetricType
    description: str
    unit: str
    target: float
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    calculation: str = ""
    review_frequency: str = "daily"
    related_metrics: List[str] = field(default_factory=list)


@dataclass
class MetricValue:
    """
    Metric measurement

    Attributes:
        metric_name: Which metric this measures
        value: Measured value
        timestamp: When measured
        dimensions: Additional context (application, region, etc.)
        alert_level: Alert severity if thresholds breached
    """

    metric_name: str
    value: float
    timestamp: datetime
    dimensions: Dict[str, str] = field(default_factory=dict)
    alert_level: Optional[AlertSeverity] = None


@dataclass
class Dashboard:
    """
    Metrics dashboard for specific audience

    Attributes:
        name: Dashboard name
        audience: Target audience (engineers, PMs, execs)
        metrics: Metrics to display
        refresh_rate: How often to update
        alert_routing: Where to send alerts
    """

    name: str
    audience: str
    metrics: List[str]
    refresh_rate: str
    alert_routing: List[str] = field(default_factory=list)


class MetricsFramework:
    """
    Comprehensive metrics framework for embedding systems

    Manages metric definitions, measurements, alerting,
    and role-specific dashboards
    """

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.metrics: Dict[str, Metric] = {}
        self.measurements: List[MetricValue] = []
        self.dashboards: Dict[str, Dashboard] = {}

    def define_metric(self, metric: Metric):
        """Define a tracked metric"""
        self.metrics[metric.name] = metric

    def record_measurement(self, measurement: MetricValue):
        """Record metric measurement"""

        # Check thresholds and set alert level
        metric = self.metrics.get(measurement.metric_name)
        if metric:
            if metric.threshold_critical and measurement.value >= metric.threshold_critical:
                measurement.alert_level = AlertSeverity.CRITICAL
            elif metric.threshold_warning and measurement.value >= metric.threshold_warning:
                measurement.alert_level = AlertSeverity.WARNING

        self.measurements.append(measurement)

    def create_dashboard(self, dashboard: Dashboard):
        """Create role-specific dashboard"""
        self.dashboards[dashboard.name] = dashboard

    def get_metric_summary(
        self, metric_name: str, start_time: datetime, end_time: datetime
    ) -> Dict[str, any]:
        """
        Get summary statistics for metric over time period

        Args:
            metric_name: Metric to summarize
            start_time: Period start
            end_time: Period end

        Returns:
            Summary statistics
        """
        metric = self.metrics.get(metric_name)
        if not metric:
            return {}

        # Filter measurements
        relevant_measurements = [
            m
            for m in self.measurements
            if m.metric_name == metric_name and start_time <= m.timestamp <= end_time
        ]

        if not relevant_measurements:
            return {
                "metric": metric_name,
                "period": f"{start_time} to {end_time}",
                "measurements": 0,
            }

        values = [m.value for m in relevant_measurements]

        return {
            "metric": metric_name,
            "category": metric.category.value,
            "period": f"{start_time} to {end_time}",
            "measurements": len(values),
            "current": values[-1],
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "target": metric.target,
            "meets_target": values[-1] <= metric.target if metric.target else None,
            "trend": "improving" if len(values) > 1 and values[-1] < values[0] else "degrading",
            "alerts": len([m for m in relevant_measurements if m.alert_level]),
        }

    def identify_issues(self) -> List[Dict[str, any]]:
        """
        Identify metrics not meeting targets

        Returns:
            List of issues requiring attention
        """
        issues = []

        # Get latest measurement for each metric
        latest_by_metric = {}
        for measurement in sorted(self.measurements, key=lambda m: m.timestamp):
            latest_by_metric[measurement.metric_name] = measurement

        for metric_name, latest in latest_by_metric.items():
            metric = self.metrics[metric_name]

            # Check if meeting target
            if metric.target and latest.value > metric.target * 1.1:  # 10% tolerance
                severity = "high" if latest.value > metric.target * 1.5 else "medium"

                issues.append(
                    {
                        "metric": metric_name,
                        "category": metric.category.value,
                        "severity": severity,
                        "current": latest.value,
                        "target": metric.target,
                        "gap": latest.value - metric.target,
                        "gap_percent": ((latest.value - metric.target) / metric.target * 100),
                        "recommendation": self._generate_recommendation(metric, latest),
                    }
                )

        # Sort by severity and gap
        issues.sort(key=lambda i: (0 if i["severity"] == "high" else 1, -i["gap_percent"]))

        return issues

    def _generate_recommendation(self, metric: Metric, measurement: MetricValue) -> str:
        """Generate recommendation for metric issue"""

        if metric.category == MetricCategory.TECHNICAL:
            if "latency" in metric.name.lower():
                return "Investigate query performance, check index efficiency, consider caching"
            elif "accuracy" in metric.name.lower():
                return "Review model quality, check for concept drift, consider retraining"
            elif "error" in metric.name.lower():
                return "Check logs for error patterns, review recent deployments"

        elif metric.category == MetricCategory.OPERATIONAL:
            if "availability" in metric.name.lower():
                return "Review incident logs, check infrastructure health, improve monitoring"
            elif "cost" in metric.name.lower():
                return "Analyze cost drivers, optimize queries, review pricing tiers"

        elif metric.category == MetricCategory.USER:
            if "ctr" in metric.name.lower():
                return "Review search relevance, analyze failed queries, improve ranking"
            elif "satisfaction" in metric.name.lower():
                return "Collect user feedback, identify pain points, run usability studies"

        elif metric.category == MetricCategory.BUSINESS:
            return "Analyze business impact, review attribution model, align with stakeholders"

        return "Investigate root cause and develop action plan"

    def create_metric_relationships(self) -> Dict[str, List[str]]:
        """
        Map relationships between metrics (leading → lagging)

        Returns:
            Metric dependency graph
        """
        relationships = {}

        for metric in self.metrics.values():
            if metric.related_metrics:
                relationships[metric.name] = metric.related_metrics

        return relationships

    def generate_dashboard_config(self, dashboard_name: str) -> str:
        """Generate dashboard configuration"""

        dashboard = self.dashboards.get(dashboard_name)
        if not dashboard:
            return f"Dashboard '{dashboard_name}' not found"

        config = f"# {dashboard.name}\n\n"
        config += f"**Audience:** {dashboard.audience}\n"
        config += f"**Refresh Rate:** {dashboard.refresh_rate}\n\n"

        config += "## Metrics\n\n"

        for metric_name in dashboard.metrics:
            metric = self.metrics.get(metric_name)
            if metric:
                config += f"### {metric_name}\n"
                config += f"- **Category:** {metric.category.value}\n"
                config += f"- **Type:** {metric.metric_type.value}\n"
                config += f"- **Target:** {metric.target} {metric.unit}\n"
                config += f"- **Description:** {metric.description}\n\n"

        if dashboard.alert_routing:
            config += "## Alert Routing\n\n"
            for route in dashboard.alert_routing:
                config += f"- {route}\n"

        return config

    def generate_executive_summary(self, start_time: datetime, end_time: datetime) -> str:
        """Generate executive summary of key metrics"""

        summary = "# Embedding System Performance Summary\n"
        summary += f"## Period: {start_time.date()} to {end_time.date()}\n\n"

        # Group metrics by category
        by_category = {}
        for metric in self.metrics.values():
            if metric.category not in by_category:
                by_category[metric.category] = []
            by_category[metric.category].append(metric.name)

        # Summarize each category
        for category in [
            MetricCategory.BUSINESS,
            MetricCategory.USER,
            MetricCategory.OPERATIONAL,
            MetricCategory.TECHNICAL,
        ]:
            if category not in by_category:
                continue

            summary += f"### {category.value.title()} Metrics\n\n"

            for metric_name in by_category[category]:
                metric_summary = self.get_metric_summary(metric_name, start_time, end_time)

                if metric_summary.get("measurements", 0) == 0:
                    continue

                status = "✓" if metric_summary.get("meets_target") else "⚠"
                summary += f"{status} **{metric_name}:** {metric_summary['current']:.2f} {self.metrics[metric_name].unit}"
                summary += (
                    f" (target: {metric_summary['target']:.2f}, trend: {metric_summary['trend']})\n"
                )

            summary += "\n"

        # Highlight issues
        issues = self.identify_issues()
        if issues:
            summary += "### Issues Requiring Attention\n\n"
            for issue in issues[:5]:  # Top 5
                summary += f"- **{issue['metric']}** ({issue['severity']} priority): "
                summary += f"{issue['gap_percent']:.1f}% above target\n"
                summary += f"  - Recommendation: {issue['recommendation']}\n"

        return summary


# Example: Define comprehensive metrics framework
def create_embedding_metrics_framework():
    """
    Example: Create metrics framework for embedding system
    """

    framework = MetricsFramework("Production Embedding System")

    # Technical metrics
    technical_metrics = [
        Metric(
            name="Query Latency p99",
            category=MetricCategory.TECHNICAL,
            metric_type=MetricType.COINCIDENT,
            description="99th percentile query latency",
            unit="ms",
            target=50.0,
            threshold_warning=75.0,
            threshold_critical=100.0,
            calculation="99th percentile of query execution time",
            review_frequency="hourly",
            related_metrics=["User Satisfaction", "Search Success Rate"],
        ),
        Metric(
            name="Embedding Quality Score",
            category=MetricCategory.TECHNICAL,
            metric_type=MetricType.LEADING,
            description="Semantic coherence of embeddings",
            unit="score",
            target=0.85,
            threshold_warning=0.80,
            threshold_critical=0.75,
            calculation="Intra-cluster similarity minus inter-cluster similarity",
            review_frequency="daily",
            related_metrics=["Search Relevance", "Recommendation CTR"],
        ),
        Metric(
            name="Model Drift Score",
            category=MetricCategory.TECHNICAL,
            metric_type=MetricType.LEADING,
            description="Distribution shift from training data",
            unit="score",
            target=0.05,
            threshold_warning=0.10,
            threshold_critical=0.15,
            calculation="KL divergence between current and baseline distributions",
            review_frequency="daily",
            related_metrics=["Embedding Quality Score"],
        ),
    ]

    # Operational metrics
    operational_metrics = [
        Metric(
            name="System Availability",
            category=MetricCategory.OPERATIONAL,
            metric_type=MetricType.LAGGING,
            description="Percentage uptime",
            unit="%",
            target=99.9,
            threshold_warning=99.5,
            threshold_critical=99.0,
            calculation="(Total time - downtime) / total time * 100",
            review_frequency="daily",
            related_metrics=["User Satisfaction"],
        ),
        Metric(
            name="Cost per 1M Queries",
            category=MetricCategory.OPERATIONAL,
            metric_type=MetricType.LAGGING,
            description="Infrastructure cost efficiency",
            unit="USD",
            target=10.0,
            threshold_warning=15.0,
            threshold_critical=20.0,
            calculation="Total infrastructure cost / query volume * 1M",
            review_frequency="weekly",
            related_metrics=["ROI"],
        ),
    ]

    # User metrics
    user_metrics = [
        Metric(
            name="Search Success Rate",
            category=MetricCategory.USER,
            metric_type=MetricType.COINCIDENT,
            description="Percentage of searches with successful outcome",
            unit="%",
            target=85.0,
            threshold_warning=80.0,
            threshold_critical=75.0,
            calculation="(Searches with click or conversion) / total searches * 100",
            review_frequency="daily",
            related_metrics=["User Satisfaction", "Conversion Rate"],
        ),
        Metric(
            name="User Satisfaction Score",
            category=MetricCategory.USER,
            metric_type=MetricType.LAGGING,
            description="User-reported satisfaction with search/recommendations",
            unit="score (1-5)",
            target=4.2,
            threshold_warning=4.0,
            threshold_critical=3.8,
            calculation="Average of user survey responses",
            review_frequency="weekly",
            related_metrics=["Customer Retention"],
        ),
        Metric(
            name="Feature Adoption Rate",
            category=MetricCategory.USER,
            metric_type=MetricType.LEADING,
            description="Percentage of users using embedding-powered features",
            unit="%",
            target=80.0,
            threshold_warning=70.0,
            threshold_critical=60.0,
            calculation="Active users of feature / total active users * 100",
            review_frequency="weekly",
            related_metrics=["User Engagement"],
        ),
    ]

    # Business metrics
    business_metrics = [
        Metric(
            name="Revenue Impact",
            category=MetricCategory.BUSINESS,
            metric_type=MetricType.LAGGING,
            description="Incremental revenue from embedding features",
            unit="USD/month",
            target=500000.0,
            threshold_warning=400000.0,
            threshold_critical=300000.0,
            calculation="A/B test measured revenue lift * user base",
            review_frequency="monthly",
            related_metrics=["Conversion Rate", "Average Order Value"],
        ),
        Metric(
            name="Cost Savings",
            category=MetricCategory.BUSINESS,
            metric_type=MetricType.LAGGING,
            description="Operational cost reduction from automation",
            unit="USD/month",
            target=200000.0,
            calculation="Previous manual process cost - current automated cost",
            review_frequency="monthly",
            related_metrics=["Efficiency Gain"],
        ),
        Metric(
            name="ROI",
            category=MetricCategory.BUSINESS,
            metric_type=MetricType.LAGGING,
            description="Return on investment",
            unit="ratio",
            target=3.0,
            threshold_warning=2.0,
            threshold_critical=1.0,
            calculation="(Revenue impact + cost savings) / total investment",
            review_frequency="quarterly",
            related_metrics=["Revenue Impact", "Cost Savings"],
        ),
    ]

    # Define all metrics
    for metric in technical_metrics + operational_metrics + user_metrics + business_metrics:
        framework.define_metric(metric)

    # Create role-specific dashboards
    dashboards = [
        Dashboard(
            name="Engineering Dashboard",
            audience="ML Engineers, SREs",
            metrics=[
                "Query Latency p99",
                "Embedding Quality Score",
                "Model Drift Score",
                "System Availability",
                "Cost per 1M Queries",
            ],
            refresh_rate="Real-time",
            alert_routing=["#eng-alerts", "oncall@company.com"],
        ),
        Dashboard(
            name="Product Dashboard",
            audience="Product Managers, Designers",
            metrics=[
                "Search Success Rate",
                "User Satisfaction Score",
                "Feature Adoption Rate",
                "Query Latency p99",
                "System Availability",
            ],
            refresh_rate="Hourly",
            alert_routing=["#product-alerts"],
        ),
        Dashboard(
            name="Executive Dashboard",
            audience="C-suite, VPs",
            metrics=[
                "Revenue Impact",
                "ROI",
                "Cost Savings",
                "User Satisfaction Score",
                "System Availability",
            ],
            refresh_rate="Daily",
            alert_routing=["exec-reports@company.com"],
        ),
    ]

    for dashboard in dashboards:
        framework.create_dashboard(dashboard)

    # Simulate some measurements
    base_time = datetime.now() - timedelta(days=7)

    # Good performance
    framework.record_measurement(
        MetricValue(
            metric_name="Query Latency p99",
            value=45.0,
            timestamp=base_time,
            dimensions={"region": "us-east-1", "application": "search"},
        )
    )

    framework.record_measurement(
        MetricValue(
            metric_name="Embedding Quality Score",
            value=0.87,
            timestamp=base_time,
            dimensions={"model_version": "v2.3"},
        )
    )

    # Issue: Cost above target
    framework.record_measurement(
        MetricValue(
            metric_name="Cost per 1M Queries",
            value=18.0,
            timestamp=base_time,
            dimensions={"month": "November"},
        )
    )

    # Good user metrics
    framework.record_measurement(
        MetricValue(metric_name="Search Success Rate", value=87.5, timestamp=base_time)
    )

    framework.record_measurement(
        MetricValue(metric_name="User Satisfaction Score", value=4.3, timestamp=base_time)
    )

    # Strong business impact
    framework.record_measurement(
        MetricValue(
            metric_name="Revenue Impact",
            value=550000.0,
            timestamp=base_time,
            dimensions={"quarter": "Q4"},
        )
    )

    framework.record_measurement(MetricValue(metric_name="ROI", value=3.8, timestamp=base_time))

    # Display executive summary
    print(framework.generate_executive_summary(base_time - timedelta(days=7), base_time))

    # Show issues
    print("\n" + "=" * 60)
    print("\n=== Issues Requiring Attention ===\n")

    issues = framework.identify_issues()
    for issue in issues:
        print(f"**{issue['metric']}** ({issue['severity']} priority)")
        print(f"  Current: {issue['current']:.2f}, Target: {issue['target']:.2f}")
        print(f"  Gap: {issue['gap_percent']:.1f}% above target")
        print(f"  Recommendation: {issue['recommendation']}\n")

    # Show dashboard configs
    print("\n" + "=" * 60)
    print("\n=== Engineering Dashboard Config ===\n")
    print(framework.generate_dashboard_config("Engineering Dashboard"))


if __name__ == "__main__":
    create_embedding_metrics_framework()
