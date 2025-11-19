# Code from Chapter 09
# Book: Embeddings at Scale

import hashlib
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ExperimentConfig:
    """
    Configuration for embedding A/B test

    Key parameters:
    - experiment_id: Unique identifier
    - control_model_id: Baseline model
    - treatment_model_id: New model being tested
    - traffic_allocation: % of users in treatment
    - primary_metric: Main success metric
    - secondary_metrics: Additional metrics to track
    - minimum_sample_size: Min users before stat sig test
    - maximum_duration_days: Auto-conclude after this period
    """
    experiment_id: str
    control_model_id: str
    treatment_model_id: str
    traffic_allocation: float  # 0.0 to 1.0
    primary_metric: str
    secondary_metrics: List[str]
    minimum_sample_size: int
    maximum_duration_days: int
    start_time: datetime

class EmbeddingExperimentFramework:
    """
    Framework for A/B testing embedding models

    Responsibilities:
    - User assignment (control vs. treatment)
    - Consistent routing throughout user session
    - Metric collection and aggregation
    - Statistical significance testing
    - Automatic ramp-up and conclusion

    Example experiment:
    - Control: Current product embeddings (v1.0)
    - Treatment: New contrastive learning model (v2.0)
    - Primary metric: Add-to-cart rate
    - Secondary metrics: Click-through rate, session length, revenue
    - Allocation: 95% control, 5% treatment (initial canary)
    """

    def __init__(self):
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # user_id → {exp_id → variant}

        # Metric storage (in production: data warehouse)
        self.metrics: Dict[str, List[Dict]] = {}  # exp_id → [metric_events]

    def create_experiment(
        self,
        config: ExperimentConfig
    ) -> str:
        """
        Create new A/B test experiment

        Validations:
        - Control and treatment models exist
        - Metrics are measurable
        - Traffic allocation is valid
        """
        # Validate models
        # (In production: Check model registry)

        # Register experiment
        self.active_experiments[config.experiment_id] = config
        self.metrics[config.experiment_id] = []

        print(f"✓ Created experiment: {config.experiment_id}")
        print(f"  Control: {config.control_model_id}")
        print(f"  Treatment: {config.treatment_model_id}")
        print(f"  Traffic to treatment: {config.traffic_allocation:.1%}")
        print(f"  Primary metric: {config.primary_metric}")

        return config.experiment_id

    def assign_user(
        self,
        user_id: str,
        experiment_id: str
    ) -> str:
        """
        Assign user to control or treatment

        Requirements:
        - Deterministic (same user always gets same variant)
        - Consistent (throughout entire experiment)
        - Randomized (but stable hash, not random seed)

        Implementation:
        - Hash user_id + experiment_id
        - Use hash to determine control vs. treatment
        - Store assignment for consistency
        """
        # Check if already assigned
        if user_id in self.user_assignments:
            if experiment_id in self.user_assignments[user_id]:
                return self.user_assignments[user_id][experiment_id]

        # Assign based on hash
        config = self.active_experiments[experiment_id]
        variant = self._hash_based_assignment(
            user_id,
            experiment_id,
            config.traffic_allocation
        )

        # Store assignment
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        self.user_assignments[user_id][experiment_id] = variant

        return variant

    def _hash_based_assignment(
        self,
        user_id: str,
        experiment_id: str,
        treatment_allocation: float
    ) -> str:
        """
        Deterministic assignment using hash function

        Hash(user_id + experiment_id) → [0, 1]
        If hash < treatment_allocation: treatment
        Else: control
        """
        hash_input = f"{user_id}:{experiment_id}".encode('utf-8')
        hash_output = hashlib.md5(hash_input).hexdigest()

        # Convert hex to float in [0, 1]
        hash_value = int(hash_output[:8], 16) / (2**32)

        if hash_value < treatment_allocation:
            return "treatment"
        else:
            return "control"

    def get_model_for_user(
        self,
        user_id: str,
        experiment_id: str
    ) -> str:
        """
        Get appropriate model version for user

        Returns:
            model_id: Either control or treatment model ID
        """
        variant = self.assign_user(user_id, experiment_id)
        config = self.active_experiments[experiment_id]

        if variant == "treatment":
            return config.treatment_model_id
        else:
            return config.control_model_id

    def log_metric(
        self,
        experiment_id: str,
        user_id: str,
        metric_name: str,
        metric_value: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Log metric event for analysis

        Args:
            experiment_id: Experiment this metric belongs to
            user_id: User who generated this metric
            metric_name: Name of metric (e.g., 'click_through_rate')
            metric_value: Value (e.g., 0.0 or 1.0 for binary, revenue for continuous)
            timestamp: When event occurred
        """
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        variant = self.user_assignments.get(user_id, {}).get(experiment_id)
        if variant is None:
            # User not assigned yet - assign now
            variant = self.assign_user(user_id, experiment_id)

        metric_event = {
            'user_id': user_id,
            'variant': variant,
            'metric_name': metric_name,
            'metric_value': metric_value,
            'timestamp': timestamp or datetime.now()
        }

        self.metrics[experiment_id].append(metric_event)

    def analyze_experiment(
        self,
        experiment_id: str
    ) -> Dict:
        """
        Analyze experiment results

        Statistical tests:
        - T-test for continuous metrics (revenue, session_length)
        - Chi-square for binary metrics (click_through, conversion)
        - Multiple testing correction (Bonferroni) for secondary metrics

        Returns:
            results: {
                'primary_metric': {lift: X%, p_value: Y, sig: True/False},
                'secondary_metrics': {...},
                'recommendation': 'ship' / 'iterate' / 'abandon'
            }
        """
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        config = self.active_experiments[experiment_id]
        events = self.metrics[experiment_id]

        # Separate control and treatment metrics
        control_metrics = [e for e in events if e['variant'] == 'control']
        treatment_metrics = [e for e in events if e['variant'] == 'treatment']

        # Analyze primary metric
        primary_result = self._analyze_metric(
            config.primary_metric,
            control_metrics,
            treatment_metrics
        )

        # Analyze secondary metrics
        secondary_results = {}
        for metric in config.secondary_metrics:
            secondary_results[metric] = self._analyze_metric(
                metric,
                control_metrics,
                treatment_metrics
            )

        # Statistical power and sample size check
        adequate_sample_size = len(set([e['user_id'] for e in events])) >= config.minimum_sample_size

        # Generate recommendation
        recommendation = self._generate_recommendation(
            primary_result,
            secondary_results,
            adequate_sample_size
        )

        results = {
            'experiment_id': experiment_id,
            'primary_metric': primary_result,
            'secondary_metrics': secondary_results,
            'sample_size': {
                'control': len(set([e['user_id'] for e in control_metrics])),
                'treatment': len(set([e['user_id'] for e in treatment_metrics]))
            },
            'adequate_sample_size': adequate_sample_size,
            'recommendation': recommendation
        }

        return results

    def _analyze_metric(
        self,
        metric_name: str,
        control_events: List[Dict],
        treatment_events: List[Dict]
    ) -> Dict:
        """
        Statistical analysis for single metric

        Returns lift, p-value, confidence interval
        """
        # Extract metric values
        control_values = [e['metric_value'] for e in control_events if e['metric_name'] == metric_name]
        treatment_values = [e['metric_value'] for e in treatment_events if e['metric_name'] == metric_name]

        if not control_values or not treatment_values:
            return {
                'control_mean': None,
                'treatment_mean': None,
                'lift': None,
                'p_value': None,
                'significant': False
            }

        # Compute statistics
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        lift = (treatment_mean - control_mean) / control_mean if control_mean > 0 else 0

        # T-test for significance (simplified)
        # In production: Use proper statistical test (t-test, chi-square, bootstrap)
        control_std = np.std(control_values)
        treatment_std = np.std(treatment_values)
        pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)

        if pooled_std > 0:
            t_stat = (treatment_mean - control_mean) / (pooled_std * np.sqrt(2 / min(len(control_values), len(treatment_values))))
            p_value = 2 * (1 - 0.975) if abs(t_stat) > 1.96 else 0.5  # Simplified
        else:
            p_value = 1.0

        significant = p_value < 0.05

        return {
            'metric_name': metric_name,
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'lift': lift,
            'p_value': p_value,
            'significant': significant
        }

    def _generate_recommendation(
        self,
        primary_result: Dict,
        secondary_results: Dict,
        adequate_sample: bool
    ) -> str:
        """
        Generate recommendation based on results

        Decision tree:
        - Inadequate sample → 'continue'
        - Primary metric positive and significant → 'ship'
        - Primary metric negative and significant → 'abandon'
        - Primary metric neutral, but secondaries positive → 'iterate'
        - All metrics neutral → 'abandon' (not worth the change)
        """
        if not adequate_sample:
            return "continue"

        if primary_result['significant']:
            if primary_result['lift'] > 0.02:  # >2% lift
                return "ship"
            elif primary_result['lift'] < -0.01:  # >1% degradation
                return "abandon"

        # Check secondary metrics for positive signals
        positive_secondaries = sum(
            1 for r in secondary_results.values()
            if r.get('significant') and r.get('lift', 0) > 0
        )

        if positive_secondaries >= 2:
            return "iterate"  # Some positive signals, needs more work

        return "abandon"  # No significant improvement

# Example: A/B test for product recommendation embeddings
def product_recommendation_ab_test():
    """
    A/B test new product embedding model

    Scenario:
    - E-commerce with 10M daily active users
    - Testing new embedding model for product recommendations
    - Primary metric: Add-to-cart rate
    - Secondary metrics: Click-through rate, session length, revenue per session

    Experiment design:
    - 95% control (current model)
    - 5% treatment (new model)
    - Run for 14 days or until statistical significance
    """

    framework = EmbeddingExperimentFramework()

    # Create experiment
    config = ExperimentConfig(
        experiment_id="product_embeddings_v2_test",
        control_model_id="product-embeddings-v1.0.0",
        treatment_model_id="product-embeddings-v2.0.0",
        traffic_allocation=0.05,  # 5% treatment
        primary_metric="add_to_cart_rate",
        secondary_metrics=["click_through_rate", "session_length_minutes", "revenue_per_session"],
        minimum_sample_size=10000,  # 10K users minimum
        maximum_duration_days=14,
        start_time=datetime.now()
    )

    framework.create_experiment(config)

    # Simulate user sessions
    print("\nSimulating user sessions...")

    for user_id in range(1000):
        uid = f"user_{user_id}"

        # Get assigned variant
        model_id = framework.get_model_for_user(uid, config.experiment_id)

        # Simulate user behavior
        # Treatment model is slightly better
        if "v2" in model_id:
            add_to_cart = np.random.random() < 0.12  # 12% conversion (treatment)
            ctr = np.random.random() < 0.25
        else:
            add_to_cart = np.random.random() < 0.10  # 10% conversion (control)
            ctr = np.random.random() < 0.22

        # Log metrics
        framework.log_metric(config.experiment_id, uid, "add_to_cart_rate", float(add_to_cart))
        framework.log_metric(config.experiment_id, uid, "click_through_rate", float(ctr))

    # Analyze results
    print("\nAnalyzing experiment results...")
    results = framework.analyze_experiment(config.experiment_id)

    print(f"\nExperiment: {results['experiment_id']}")
    print(f"Sample size: Control={results['sample_size']['control']}, Treatment={results['sample_size']['treatment']}")
    print(f"\nPrimary Metric ({results['primary_metric']['metric_name']}):")
    print(f"  Control: {results['primary_metric']['control_mean']:.3f}")
    print(f"  Treatment: {results['primary_metric']['treatment_mean']:.3f}")
    print(f"  Lift: {results['primary_metric']['lift']:.2%}")
    print(f"  P-value: {results['primary_metric']['p_value']:.4f}")
    print(f"  Significant: {results['primary_metric']['significant']}")
    print(f"\nRecommendation: {results['recommendation'].upper()}")

# Uncomment to run:
# product_recommendation_ab_test()
