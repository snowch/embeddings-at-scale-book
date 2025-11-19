# Code from Chapter 25
# Book: Embeddings at Scale

"""
User Experience Analytics for Embedding Systems

Architecture:
1. Event tracking: Log user interactions with embedding-powered features
2. Metric calculation: Compute engagement, satisfaction, business metrics
3. A/B testing: Compare embedding models through controlled experiments
4. Causal inference: Isolate embedding quality impact from other factors
5. Feedback loops: Use UX signals to improve embeddings

Metrics:
- Engagement: CTR, dwell time, scroll depth, interactions
- Session quality: Bounce rate, pages per session, session duration
- Satisfaction: User ratings, feedback, repeat usage
- Business outcomes: Conversion rate, revenue, LTV
- Long-term: Retention, churn, DAU/MAU ratio

Analysis:
- Correlation: Embedding quality vs user metrics
- A/B testing: Statistical significance of improvements
- Cohort analysis: Different user segments respond differently
- Attribution: Isolate embedding contribution to outcomes
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class UserEvent:
    """Single user interaction event"""

    timestamp: datetime
    user_id: str
    session_id: str
    event_type: str  # "query", "click", "view", "purchase", "rating"

    # Query information
    query: Optional[str] = None
    query_embeddings: Optional[np.ndarray] = None

    # Result information
    result_id: Optional[str] = None
    result_position: Optional[int] = None
    result_score: Optional[float] = None

    # Interaction information
    clicked: bool = False
    dwell_time_seconds: float = 0.0
    rating: Optional[float] = None

    # Business outcomes
    converted: bool = False
    revenue: float = 0.0

    # Metadata
    experiment_group: Optional[str] = None  # For A/B testing
    model_version: str = "unknown"


@dataclass
class SessionMetrics:
    """Metrics for a user session"""

    session_id: str
    user_id: str
    start_time: datetime
    end_time: datetime

    # Engagement metrics
    num_queries: int = 0
    num_clicks: int = 0
    num_views: int = 0
    total_dwell_time: float = 0.0

    # Quality metrics
    avg_click_position: float = 0.0
    bounce_rate: float = 0.0  # Single-page sessions
    pages_per_session: int = 0

    # Business metrics
    converted: bool = False
    revenue: float = 0.0

    # Satisfaction
    satisfaction_rating: Optional[float] = None

    def click_through_rate(self) -> float:
        """Calculate session CTR"""
        return self.num_clicks / max(self.num_queries, 1)

    def duration_minutes(self) -> float:
        """Calculate session duration in minutes"""
        return (self.end_time - self.start_time).total_seconds() / 60.0


@dataclass
class ExperimentResults:
    """A/B test experiment results"""

    experiment_name: str
    start_date: datetime
    end_date: datetime

    # Experiment groups
    control_group: str
    treatment_group: str

    # Sample sizes
    control_users: int
    treatment_users: int

    # Metrics
    control_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]

    # Statistical significance
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

    # Decision
    winner: Optional[str] = None
    lift: Dict[str, float] = field(default_factory=dict)


class UserExperienceAnalytics:
    """
    Comprehensive user experience analytics for embedding systems

    Tracks user interactions, computes engagement/satisfaction metrics,
    runs A/B tests, and provides insights for optimization.
    """

    def __init__(self):
        """Initialize UX analytics system"""
        self.events: List[UserEvent] = []
        self.sessions: Dict[str, SessionMetrics] = {}
        self.experiments: Dict[str, ExperimentResults] = {}

        # Cached aggregations
        self.daily_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)

    def track_event(self, event: UserEvent):
        """Track a user interaction event"""
        self.events.append(event)

        # Update session metrics
        if event.session_id not in self.sessions:
            self.sessions[event.session_id] = SessionMetrics(
                session_id=event.session_id,
                user_id=event.user_id,
                start_time=event.timestamp,
                end_time=event.timestamp,
            )

        session = self.sessions[event.session_id]
        session.end_time = event.timestamp

        if event.event_type == "query":
            session.num_queries += 1
        elif event.event_type == "click":
            session.num_clicks += 1
            if event.result_position is not None:
                # Update average click position
                if session.num_clicks == 1:
                    session.avg_click_position = event.result_position
                else:
                    session.avg_click_position = (
                        session.avg_click_position * (session.num_clicks - 1)
                        + event.result_position
                    ) / session.num_clicks
        elif event.event_type == "view":
            session.num_views += 1
            session.total_dwell_time += event.dwell_time_seconds
        elif event.event_type == "purchase":
            session.converted = True
            session.revenue += event.revenue
        elif event.event_type == "rating" and event.rating is not None:
            session.satisfaction_rating = event.rating

        # Update daily metrics
        date_key = event.timestamp.strftime("%Y-%m-%d")
        if date_key not in self.daily_metrics:
            self.daily_metrics[date_key] = defaultdict(float)

        self.daily_metrics[date_key]["events"] += 1
        if event.clicked:
            self.daily_metrics[date_key]["clicks"] += 1
        if event.converted:
            self.daily_metrics[date_key]["conversions"] += 1
        self.daily_metrics[date_key]["revenue"] += event.revenue

    def get_engagement_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        experiment_group: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Calculate engagement metrics for time period

        Returns metrics like CTR, dwell time, clicks per session, etc.
        """
        # Filter events
        filtered_events = self.events
        if start_date:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_date]
        if end_date:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_date]
        if experiment_group:
            filtered_events = [e for e in filtered_events if e.experiment_group == experiment_group]

        if not filtered_events:
            return {}

        # Calculate metrics
        total_queries = sum(1 for e in filtered_events if e.event_type == "query")
        total_clicks = sum(1 for e in filtered_events if e.clicked)
        total_views = sum(1 for e in filtered_events if e.event_type == "view")
        total_dwell_time = sum(
            e.dwell_time_seconds for e in filtered_events if e.event_type == "view"
        )

        # Click positions
        click_positions = [
            e.result_position
            for e in filtered_events
            if e.clicked and e.result_position is not None
        ]

        metrics = {
            "total_events": len(filtered_events),
            "total_queries": total_queries,
            "total_clicks": total_clicks,
            "total_views": total_views,
            "click_through_rate": total_clicks / max(total_queries, 1),
            "avg_dwell_time_seconds": total_dwell_time / max(total_views, 1),
            "avg_click_position": np.mean(click_positions) if click_positions else 0.0,
            "clicks_per_query": total_clicks / max(total_queries, 1),
        }

        return metrics

    def get_business_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        experiment_group: Optional[str] = None,
    ) -> Dict[str, float]:
        """Calculate business metrics like conversion rate, revenue, etc."""
        # Filter sessions
        filtered_sessions = list(self.sessions.values())
        if start_date:
            filtered_sessions = [s for s in filtered_sessions if s.start_time >= start_date]
        if end_date:
            filtered_sessions = [s for s in filtered_sessions if s.end_time <= end_date]
        if experiment_group:
            # Need to check events in session for experiment group
            session_ids = {
                e.session_id for e in self.events if e.experiment_group == experiment_group
            }
            filtered_sessions = [s for s in filtered_sessions if s.session_id in session_ids]

        if not filtered_sessions:
            return {}

        total_sessions = len(filtered_sessions)
        converted_sessions = sum(1 for s in filtered_sessions if s.converted)
        total_revenue = sum(s.revenue for s in filtered_sessions)

        metrics = {
            "total_sessions": total_sessions,
            "conversion_rate": converted_sessions / total_sessions,
            "total_revenue": total_revenue,
            "revenue_per_session": total_revenue / total_sessions,
            "revenue_per_converted_session": total_revenue / max(converted_sessions, 1),
        }

        return metrics

    def run_ab_test(
        self,
        experiment_name: str,
        control_group: str,
        treatment_group: str,
        start_date: datetime,
        end_date: datetime,
        metrics_to_test: List[str],
    ) -> ExperimentResults:
        """
        Run A/B test comparing two embedding models

        Args:
            experiment_name: Name of experiment
            control_group: Control group identifier
            treatment_group: Treatment group identifier
            start_date: Experiment start date
            end_date: Experiment end date
            metrics_to_test: List of metrics to compare

        Returns:
            ExperimentResults with statistical analysis
        """

        # Get metrics for both groups
        control_engagement = self.get_engagement_metrics(start_date, end_date, control_group)
        treatment_engagement = self.get_engagement_metrics(start_date, end_date, treatment_group)

        control_business = self.get_business_metrics(start_date, end_date, control_group)
        treatment_business = self.get_business_metrics(start_date, end_date, treatment_group)

        control_metrics = {**control_engagement, **control_business}
        treatment_metrics = {**treatment_engagement, **treatment_business}

        # Calculate statistical significance for each metric
        p_values = {}
        confidence_intervals = {}
        lift = {}

        for metric in metrics_to_test:
            if metric not in control_metrics or metric not in treatment_metrics:
                continue

            control_value = control_metrics[metric]
            treatment_value = treatment_metrics[metric]

            # Calculate lift
            if control_value > 0:
                lift[metric] = (treatment_value - control_value) / control_value

            # For simplicity, use z-test approximation
            # In production, would use proper statistical tests with per-user data
            # This is a simplified example
            p_values[metric] = 0.05  # Placeholder
            confidence_intervals[metric] = (
                treatment_value * 0.95,
                treatment_value * 1.05,
            )  # Placeholder

        # Determine winner (simplified)
        winner = None
        primary_metric = metrics_to_test[0] if metrics_to_test else "click_through_rate"

        if primary_metric in lift:
            if lift[primary_metric] > 0.05 and p_values.get(primary_metric, 1.0) < 0.05:
                winner = treatment_group
            elif lift[primary_metric] < -0.05 and p_values.get(primary_metric, 1.0) < 0.05:
                winner = control_group

        results = ExperimentResults(
            experiment_name=experiment_name,
            start_date=start_date,
            end_date=end_date,
            control_group=control_group,
            treatment_group=treatment_group,
            control_users=int(control_metrics.get("total_sessions", 0)),
            treatment_users=int(treatment_metrics.get("total_sessions", 0)),
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            winner=winner,
            lift=lift,
        )

        self.experiments[experiment_name] = results
        return results

    def generate_ux_report(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> str:
        """Generate comprehensive UX report"""
        engagement = self.get_engagement_metrics(start_date, end_date)
        business = self.get_business_metrics(start_date, end_date)

        report = f"""
User Experience Analytics Report
=================================
Period: {start_date.strftime("%Y-%m-%d") if start_date else "All time"} to {end_date.strftime("%Y-%m-%d") if end_date else "Present"}

Engagement Metrics:
------------------
Total Events:        {engagement.get("total_events", 0):>12,}
Total Queries:       {engagement.get("total_queries", 0):>12,}
Total Clicks:        {engagement.get("total_clicks", 0):>12,}
Click-Through Rate:  {engagement.get("click_through_rate", 0) * 100:>11.2f}%
Avg Dwell Time:      {engagement.get("avg_dwell_time_seconds", 0):>11.1f}s
Avg Click Position:  {engagement.get("avg_click_position", 0):>11.1f}
Clicks per Query:    {engagement.get("clicks_per_query", 0):>11.2f}

Business Metrics:
----------------
Total Sessions:      {business.get("total_sessions", 0):>12,}
Conversion Rate:     {business.get("conversion_rate", 0) * 100:>11.2f}%
Total Revenue:       ${business.get("total_revenue", 0):>11.2f}
Revenue/Session:     ${business.get("revenue_per_session", 0):>11.2f}
Revenue/Conversion:  ${business.get("revenue_per_converted_session", 0):>11.2f}
"""

        # Add experiment results if any
        if self.experiments:
            report += "\n\nA/B Test Results:\n"
            report += "=================\n"
            for name, results in self.experiments.items():
                report += f"\nExperiment: {name}\n"
                report += f"Winner: {results.winner or 'Inconclusive'}\n"
                report += (
                    f"Primary Metric Lift: {results.lift.get('click_through_rate', 0) * 100:.2f}%\n"
                )

        return report


# Example usage
if __name__ == "__main__":
    analytics = UserExperienceAnalytics()

    # Simulate user events
    print("Simulating user interactions...")

    base_time = datetime.now() - timedelta(days=7)

    for day in range(7):
        for session_num in range(100):  # 100 sessions per day
            session_id = f"session_{day}_{session_num}"
            user_id = f"user_{session_num % 50}"  # 50 unique users

            # Randomly assign to control or treatment group
            experiment_group = "control" if session_num % 2 == 0 else "treatment"

            # Treatment group has slightly better metrics
            ctr_boost = 1.1 if experiment_group == "treatment" else 1.0

            # Generate queries for this session
            num_queries = np.random.randint(1, 5)
            for query_num in range(num_queries):
                timestamp = base_time + timedelta(
                    days=day, minutes=session_num * 10 + query_num * 2
                )

                # Query event
                analytics.track_event(
                    UserEvent(
                        timestamp=timestamp,
                        user_id=user_id,
                        session_id=session_id,
                        event_type="query",
                        query=f"sample query {query_num}",
                        experiment_group=experiment_group,
                    )
                )

                # Maybe click on result
                if np.random.random() < 0.3 * ctr_boost:  # 30% CTR for control, 33% for treatment
                    click_position = np.random.choice(
                        [1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.10, 0.05]
                    )
                    analytics.track_event(
                        UserEvent(
                            timestamp=timestamp + timedelta(seconds=2),
                            user_id=user_id,
                            session_id=session_id,
                            event_type="click",
                            result_id=f"result_{click_position}",
                            result_position=click_position,
                            clicked=True,
                            experiment_group=experiment_group,
                        )
                    )

                    # View the result
                    dwell_time = np.random.exponential(30)  # Average 30 seconds
                    analytics.track_event(
                        UserEvent(
                            timestamp=timestamp + timedelta(seconds=3),
                            user_id=user_id,
                            session_id=session_id,
                            event_type="view",
                            result_id=f"result_{click_position}",
                            dwell_time_seconds=dwell_time,
                            experiment_group=experiment_group,
                        )
                    )

            # Maybe convert
            if (
                np.random.random() < 0.1 * ctr_boost
            ):  # 10% conversion for control, 11% for treatment
                analytics.track_event(
                    UserEvent(
                        timestamp=timestamp + timedelta(minutes=5),
                        user_id=user_id,
                        session_id=session_id,
                        event_type="purchase",
                        converted=True,
                        revenue=np.random.uniform(20, 200),
                        experiment_group=experiment_group,
                    )
                )

    # Generate overall UX report
    print("\n" + "=" * 70)
    report = analytics.generate_ux_report(start_date=base_time, end_date=datetime.now())
    print(report)

    # Run A/B test
    print("\n\nRunning A/B Test...")
    print("=" * 70)
    experiment = analytics.run_ab_test(
        experiment_name="embedding_model_v2",
        control_group="control",
        treatment_group="treatment",
        start_date=base_time,
        end_date=datetime.now(),
        metrics_to_test=["click_through_rate", "conversion_rate", "revenue_per_session"],
    )

    print(f"\nExperiment: {experiment.experiment_name}")
    print(f"Winner: {experiment.winner or 'Inconclusive'}")
    print("\nControl Group Metrics:")
    for metric, value in experiment.control_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("\nTreatment Group Metrics:")
    for metric, value in experiment.treatment_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("\nLift:")
    for metric, lift_value in experiment.lift.items():
        print(f"  {metric}: {lift_value * 100:+.2f}%")
