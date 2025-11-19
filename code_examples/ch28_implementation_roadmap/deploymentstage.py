from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

# Code from Chapter 28
# Book: Embeddings at Scale

"""
Phase 2: Production Pilot Architecture

Architecture:
1. Production-grade infrastructure: HA, security, observability
2. Scalable serving: Load balancing, caching, rate limiting
3. Continuous deployment: CI/CD, feature flags, canary releases
4. Monitoring and alerting: Metrics, SLOs, incident response
5. User feedback integration: Analytics, A/B testing, iteration

Production requirements:
- Availability: 99.9%+ uptime (SLO)
- Performance: p95 < 50ms, p99 < 100ms (SLO)
- Scalability: Handle 10x traffic spikes gracefully
- Security: Authentication, encryption, audit logs
- Observability: Real-time metrics, distributed tracing
- Cost efficiency: <$0.01 per query at scale

Key components:
- Vector database cluster (HA, replicated)
- Embedding service (async, scaled)
- API gateway (rate limiting, auth)
- Cache layer (Redis cluster)
- Monitoring stack (Prometheus, Grafana)
- CI/CD pipeline (GitHub Actions, ArgoCD)
"""


class DeploymentStage(Enum):
    """Deployment stages for pilot"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"


class PerformanceMetric(Enum):
    """Key performance metrics"""

    QUERY_LATENCY_P50 = "query_latency_p50"
    QUERY_LATENCY_P95 = "query_latency_p95"
    QUERY_LATENCY_P99 = "query_latency_p99"
    QUERY_THROUGHPUT = "query_throughput"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    EMBEDDING_QUALITY = "embedding_quality"
    CACHE_HIT_RATE = "cache_hit_rate"


@dataclass
class ServiceLevelObjective:
    """Service Level Objective (SLO) definition"""

    name: str
    metric: PerformanceMetric
    target_value: float
    measurement_window: timedelta

    # Alerting
    warning_threshold: float  # Alert if approaching target
    critical_threshold: float  # Page if violated

    current_value: Optional[float] = None
    last_updated: Optional[datetime] = None

    def is_met(self) -> bool:
        """Check if SLO is currently being met"""
        if self.current_value is None:
            return False
        return self.current_value <= self.target_value

    def alert_level(self) -> Optional[str]:
        """Determine if alert should fire"""
        if self.current_value is None:
            return None

        if self.current_value >= self.critical_threshold:
            return "CRITICAL"
        elif self.current_value >= self.warning_threshold:
            return "WARNING"
        return None


@dataclass
class PilotConfiguration:
    """Configuration for pilot deployment"""

    pilot_name: str
    start_date: datetime
    target_duration_weeks: int

    # User cohorts
    cohort_definitions: List[Dict[str, any]]  # Segments for rollout
    initial_user_percentage: float  # Start with small %
    max_user_percentage: float  # Maximum during pilot
    ramp_up_schedule: List[Dict[str, any]]  # Planned increases

    # Feature flags
    features_enabled: Dict[str, bool]
    experiment_variants: List[str]

    # SLOs
    slos: List[ServiceLevelObjective] = field(default_factory=list)

    # Success criteria
    success_metrics: Dict[str, float]  # metric -> target
    go_live_criteria: List[str]  # Must meet before full rollout

    # Risk mitigation
    rollback_triggers: List[str]
    escalation_contacts: List[Dict[str, str]]


class PilotMonitor:
    """
    Monitor pilot deployment performance and health.

    Track SLOs, user metrics, incidents, and determine
    rollout readiness.
    """

    def __init__(self, config: PilotConfiguration):
        self.config = config
        self.metrics_history: Dict[PerformanceMetric, List[Tuple[datetime, float]]] = {}
        self.incidents: List[Dict[str, any]] = []
        self.user_feedback: List[Dict[str, any]] = []

    def record_metric(
        self, metric: PerformanceMetric, value: float, timestamp: Optional[datetime] = None
    ) -> None:
        """Record metric value"""
        if timestamp is None:
            timestamp = datetime.now()

        if metric not in self.metrics_history:
            self.metrics_history[metric] = []
        self.metrics_history[metric].append((timestamp, value))

        # Update SLOs
        for slo in self.config.slos:
            if slo.metric == metric:
                slo.current_value = value
                slo.last_updated = timestamp

                # Check for alerts
                alert = slo.alert_level()
                if alert:
                    self._trigger_alert(slo, alert)

    def _trigger_alert(self, slo: ServiceLevelObjective, level: str) -> None:
        """Trigger alert for SLO violation"""
        alert = {
            "timestamp": datetime.now(),
            "level": level,
            "slo": slo.name,
            "current": slo.current_value,
            "target": slo.target_value,
            "message": f"SLO {slo.name} {level}: {slo.current_value} vs target {slo.target_value}",
        }
        print(f"ALERT [{level}]: {alert['message']}")
        # In production: Send to PagerDuty, Slack, etc.

    def record_incident(
        self, title: str, severity: str, description: str, resolution: Optional[str] = None
    ) -> None:
        """Record incident during pilot"""
        incident = {
            "timestamp": datetime.now(),
            "title": title,
            "severity": severity,
            "description": description,
            "resolution": resolution,
            "resolved": resolution is not None,
        }
        self.incidents.append(incident)

    def record_user_feedback(
        self,
        user_id: str,
        rating: int,  # 1-5
        feedback: str,
        context: Optional[Dict[str, any]] = None,
    ) -> None:
        """Record user feedback"""
        feedback_record = {
            "timestamp": datetime.now(),
            "user_id": user_id,
            "rating": rating,
            "feedback": feedback,
            "context": context or {},
        }
        self.user_feedback.append(feedback_record)

    def check_slo_compliance(self) -> Dict[str, bool]:
        """Check if all SLOs are being met"""
        return {slo.name: slo.is_met() for slo in self.config.slos}

    def calculate_user_satisfaction(self) -> Optional[float]:
        """Calculate average user satisfaction score"""
        if not self.user_feedback:
            return None
        return sum(f["rating"] for f in self.user_feedback) / len(self.user_feedback)

    def assess_rollout_readiness(self) -> Dict[str, any]:
        """
        Assess readiness for broader rollout.

        Returns assessment with recommendations.
        """
        assessment = {
            "timestamp": datetime.now(),
            "ready": True,
            "blockers": [],
            "warnings": [],
            "metrics": {},
        }

        # Check SLO compliance
        slo_compliance = self.check_slo_compliance()
        assessment["metrics"]["slo_compliance"] = slo_compliance

        if not all(slo_compliance.values()):
            assessment["ready"] = False
            failed_slos = [name for name, met in slo_compliance.items() if not met]
            assessment["blockers"].append(f"SLOs not met: {failed_slos}")

        # Check incident rate
        recent_incidents = [
            i for i in self.incidents if (datetime.now() - i["timestamp"]) < timedelta(days=7)
        ]
        critical_incidents = [
            i for i in recent_incidents if i["severity"] == "CRITICAL" and not i["resolved"]
        ]

        assessment["metrics"]["incidents_7d"] = len(recent_incidents)
        assessment["metrics"]["critical_unresolved"] = len(critical_incidents)

        if critical_incidents:
            assessment["ready"] = False
            assessment["blockers"].append(
                f"{len(critical_incidents)} unresolved critical incidents"
            )
        elif len(recent_incidents) > 5:
            assessment["warnings"].append(f"High incident rate: {len(recent_incidents)} in 7 days")

        # Check user satisfaction
        satisfaction = self.calculate_user_satisfaction()
        assessment["metrics"]["user_satisfaction"] = satisfaction

        if satisfaction and satisfaction < 3.5:
            assessment["ready"] = False
            assessment["blockers"].append(f"User satisfaction too low: {satisfaction:.2f}/5.0")
        elif satisfaction and satisfaction < 4.0:
            assessment["warnings"].append(
                f"User satisfaction below target: {satisfaction:.2f}/5.0 (target: 4.0+)"
            )

        # Check success metrics
        for metric_name, _target in self.config.success_metrics.items():
            # In real implementation, fetch actual metric values
            assessment["metrics"][metric_name] = "Not implemented"

        return assessment

    def generate_pilot_report(self) -> str:
        """Generate comprehensive pilot report"""
        report = []
        report.append(f"# Pilot Report: {self.config.pilot_name}\n\n")
        report.append(f"Generated: {datetime.now().isoformat()}\n\n")

        # Overview
        duration = (datetime.now() - self.config.start_date).days
        report.append("## Pilot Overview\n\n")
        report.append(f"- Start date: {self.config.start_date.date()}\n")
        report.append(f"- Duration: {duration} days\n")
        report.append(
            f"- User percentage: {self.config.initial_user_percentage}% → {self.config.max_user_percentage}%\n\n"
        )

        # SLO compliance
        report.append("## SLO Compliance\n\n")
        slo_compliance = self.check_slo_compliance()
        for slo in self.config.slos:
            status = "✓" if slo_compliance[slo.name] else "✗"
            report.append(
                f"- {status} **{slo.name}**: {slo.current_value} (target: {slo.target_value})\n"
            )
        report.append("\n")

        # Incidents
        report.append(f"## Incidents ({len(self.incidents)} total)\n\n")
        if self.incidents:
            for incident in self.incidents[-10:]:  # Last 10
                status = "Resolved" if incident["resolved"] else "Open"
                report.append(f"- [{incident['severity']}] {incident['title']} - {status}\n")
                report.append(f"  {incident['description']}\n")
        else:
            report.append("No incidents recorded.\n")
        report.append("\n")

        # User feedback
        satisfaction = self.calculate_user_satisfaction()
        report.append(f"## User Feedback ({len(self.user_feedback)} responses)\n\n")
        report.append(f"Average satisfaction: {satisfaction:.2f}/5.0\n\n")

        if self.user_feedback:
            report.append("### Recent Feedback:\n\n")
            for feedback in self.user_feedback[-5:]:  # Last 5
                report.append(f"- ({feedback['rating']}/5) {feedback['feedback']}\n")
        report.append("\n")

        # Readiness assessment
        assessment = self.assess_rollout_readiness()
        report.append("## Rollout Readiness Assessment\n\n")
        report.append(f"**Status:** {'READY ✓' if assessment['ready'] else 'NOT READY ✗'}\n\n")

        if assessment["blockers"]:
            report.append("### Blockers:\n\n")
            for blocker in assessment["blockers"]:
                report.append(f"- ✗ {blocker}\n")
            report.append("\n")

        if assessment["warnings"]:
            report.append("### Warnings:\n\n")
            for warning in assessment["warnings"]:
                report.append(f"- ⚠ {warning}\n")
            report.append("\n")

        return "".join(report)


# Example: E-commerce search pilot
def example_pilot_deployment():
    """Example pilot deployment workflow"""

    # Configure pilot
    config = PilotConfiguration(
        pilot_name="E-commerce Semantic Search Pilot",
        start_date=datetime.now() - timedelta(days=30),
        target_duration_weeks=8,
        cohort_definitions=[
            {"name": "power_users", "criteria": "orders > 10"},
            {"name": "mobile_users", "criteria": "device == 'mobile'"},
        ],
        initial_user_percentage=5.0,
        max_user_percentage=20.0,
        ramp_up_schedule=[
            {"week": 1, "percentage": 5},
            {"week": 2, "percentage": 10},
            {"week": 4, "percentage": 15},
            {"week": 6, "percentage": 20},
        ],
        features_enabled={
            "semantic_search": True,
            "visual_search": False,  # Phase 3
            "personalization": False,  # Phase 3
        },
        experiment_variants=["control", "treatment"],
        success_metrics={
            "search_success_rate": 0.80,  # 80% of searches lead to engagement
            "zero_result_rate": 0.15,  # <15% zero results
            "conversion_lift": 0.15,  # 15% lift over baseline
            "user_satisfaction": 4.0,  # 4.0/5.0 rating
        },
        go_live_criteria=[
            "All SLOs met for 2+ weeks",
            "Zero critical incidents in last week",
            "User satisfaction > 4.0",
            "Conversion lift > 10% (significant)",
        ],
        rollback_triggers=[
            "Availability < 99.5%",
            "p99 latency > 200ms",
            "Error rate > 1%",
            "User satisfaction < 3.0",
        ],
    )

    # Define SLOs
    config.slos = [
        ServiceLevelObjective(
            name="Query Latency p95",
            metric=PerformanceMetric.QUERY_LATENCY_P95,
            target_value=50.0,  # ms
            warning_threshold=45.0,
            critical_threshold=60.0,
            measurement_window=timedelta(minutes=5),
        ),
        ServiceLevelObjective(
            name="Query Latency p99",
            metric=PerformanceMetric.QUERY_LATENCY_P99,
            target_value=100.0,  # ms
            warning_threshold=90.0,
            critical_threshold=150.0,
            measurement_window=timedelta(minutes=5),
        ),
        ServiceLevelObjective(
            name="Availability",
            metric=PerformanceMetric.AVAILABILITY,
            target_value=99.9,  # %
            warning_threshold=99.8,
            critical_threshold=99.5,
            measurement_window=timedelta(hours=1),
        ),
        ServiceLevelObjective(
            name="Error Rate",
            metric=PerformanceMetric.ERROR_RATE,
            target_value=0.1,  # %
            warning_threshold=0.5,
            critical_threshold=1.0,
            measurement_window=timedelta(minutes=5),
        ),
    ]

    # Create monitor
    monitor = PilotMonitor(config)

    # Simulate some metrics (in production, these come from actual system)
    monitor.record_metric(PerformanceMetric.QUERY_LATENCY_P95, 42.0)
    monitor.record_metric(PerformanceMetric.QUERY_LATENCY_P99, 95.0)
    monitor.record_metric(PerformanceMetric.AVAILABILITY, 99.95)
    monitor.record_metric(PerformanceMetric.ERROR_RATE, 0.08)

    # Record some incidents
    monitor.record_incident(
        title="Vector DB high latency spike",
        severity="WARNING",
        description="p99 latency spiked to 180ms for 5 minutes",
        resolution="Auto-scaled vector DB cluster, added cache warming",
    )

    # Record user feedback
    monitor.record_user_feedback(
        user_id="user_123",
        rating=5,
        feedback="Much better search results! Finally found what I needed.",
        context={"query": "wireless headphones for running"},
    )
    monitor.record_user_feedback(
        user_id="user_456",
        rating=4,
        feedback="Good improvement, but still some irrelevant results",
        context={"query": "laptop case 15 inch"},
    )
    monitor.record_user_feedback(
        user_id="user_789", rating=3, feedback="Slower than before", context={"latency_ms": 120}
    )

    # Generate report
    print(monitor.generate_pilot_report())

    # Check readiness
    assessment = monitor.assess_rollout_readiness()
    print("\n" + "=" * 80 + "\n")
    print(f"Rollout Ready: {assessment['ready']}")
    if assessment["blockers"]:
        print("Blockers:")
        for blocker in assessment["blockers"]:
            print(f"  - {blocker}")


if __name__ == "__main__":
    example_pilot_deployment()
