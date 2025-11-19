import numpy as np

# Code from Chapter 30
# Book: Embeddings at Scale

"""
Disruption Scenario Planning and Preparedness

Architecture:
1. Horizon scanning: Monitor signals of potential disruption
2. Scenario development: Envision multiple plausible futures
3. Impact assessment: Analyze implications for business and technology
4. Response planning: Develop strategies for each scenario
5. Trigger monitoring: Track indicators signaling which scenario emerging
6. Adaptive execution: Adjust strategy as situation clarifies

Scenario categories:
- Technology disruptions: New embedding techniques, architectures
- Competitive disruptions: New entrants, business models
- Regulatory disruptions: Privacy laws, AI regulation, data sovereignty
- Market disruptions: Customer needs, use case evolution
- Economic disruptions: Recession, funding environment, cost pressures

Preparedness dimensions:
- Technical flexibility: Architecture supporting multiple approaches
- Organizational agility: Rapid decision-making and pivoting
- Financial resilience: Reserves for adaptation investments
- Talent adaptability: Team capable of learning new techniques
- Strategic optionality: Multiple paths forward preserved
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Set, Tuple


class DisruptionCategory(Enum):
    """Types of potential disruptions"""
    TECHNOLOGY = "technology"  # New techniques, architectures
    COMPETITIVE = "competitive"  # New entrants, business models
    REGULATORY = "regulatory"  # Laws, compliance requirements
    MARKET = "market"  # Customer needs, use cases
    ECONOMIC = "economic"  # Recession, funding, costs

class DisruptionLikelihood(Enum):
    """Probability assessment"""
    LOW = "low"  # <20% probability in 3 years
    MEDIUM = "medium"  # 20-50% probability
    HIGH = "high"  # >50% probability
    IMMINENT = "imminent"  # Already beginning

class DisruptionImpact(Enum):
    """Severity of impact"""
    MINOR = "minor"  # <10% business impact
    MODERATE = "moderate"  # 10-30% impact
    MAJOR = "major"  # 30-70% impact
    EXISTENTIAL = "existential"  # >70% impact, survival threat

@dataclass
class DisruptionScenario:
    """Potential disruption scenario"""
    scenario_name: str
    category: DisruptionCategory
    description: str

    # Probability and impact
    likelihood: DisruptionLikelihood
    impact: DisruptionImpact
    time_horizon: str  # "1-2 years", "2-3 years", etc.

    # Detailed analysis
    key_assumptions: List[str]
    triggering_events: List[str]  # What would indicate this happening
    early_warning_signals: List[str]  # Indicators to monitor

    # Business implications
    affected_capabilities: Set[str]
    affected_revenue_streams: Set[str]
    required_adaptations: List[str]
    adaptation_cost: float
    adaptation_timeline: timedelta

    # Response strategy
    response_plan: str
    contingency_actions: List[str]
    required_investments: Dict[str, float]
    success_metrics: Dict[str, float]

    # Tracking
    signals_observed: List[Tuple[datetime, str]] = field(default_factory=list)
    confidence_level: float = 0.5  # 0-1, how confident in this scenario
    last_reviewed: datetime = field(default_factory=datetime.now)

    metadata: Dict[str, any] = field(default_factory=dict)

@dataclass
class DisruptionIndicator:
    """Signal that could indicate emerging disruption"""
    indicator_name: str
    category: DisruptionCategory
    data_source: str  # Where we track this

    # Measurement
    current_value: float
    threshold_warning: float  # Value triggering attention
    threshold_critical: float  # Value triggering action

    # Context
    historical_values: List[Tuple[datetime, float]] = field(default_factory=list)
    trend_direction: str = "stable"  # "increasing", "decreasing", "stable"
    rate_of_change: float = 0.0

    # Response
    related_scenarios: List[str] = field(default_factory=list)
    monitoring_frequency: str = "monthly"
    owner: str = ""

    def update_value(self, new_value: float, observation_date: datetime):
        """Update indicator and assess trend"""
        self.historical_values.append((observation_date, new_value))

        # Calculate trend
        if len(self.historical_values) >= 2:
            old_value = self.historical_values[-2][1]
            self.rate_of_change = (new_value - old_value) / old_value if old_value != 0 else 0

            if self.rate_of_change > 0.1:
                self.trend_direction = "increasing"
            elif self.rate_of_change < -0.1:
                self.trend_direction = "decreasing"
            else:
                self.trend_direction = "stable"

        self.current_value = new_value

    def assess_status(self) -> str:
        """Assess current status relative to thresholds"""
        if self.current_value >= self.threshold_critical:
            return "CRITICAL: Immediate action required"
        elif self.current_value >= self.threshold_warning:
            return "WARNING: Monitor closely and prepare response"
        else:
            return "NORMAL: Continue monitoring"

class DisruptionPreparedness:
    """System for tracking and preparing for disruptions"""

    def __init__(self):
        self.scenarios: Dict[str, DisruptionScenario] = {}
        self.indicators: Dict[str, DisruptionIndicator] = {}
        self.response_playbooks: Dict[str, Dict] = {}

    def develop_scenarios(
        self,
        current_position: Dict[str, any],
        time_horizon_years: int = 3
    ) -> List[DisruptionScenario]:
        """Generate comprehensive disruption scenarios"""

        scenarios = []

        # Technology disruption scenarios
        scenarios.append(DisruptionScenario(
            scenario_name="Quantum Embeddings",
            category=DisruptionCategory.TECHNOLOGY,
            description="Quantum computing enables 1000× larger embedding dimensions with exponentially better similarity search",
            likelihood=DisruptionLikelihood.LOW,
            impact=DisruptionImpact.MAJOR,
            time_horizon="3-5 years",
            key_assumptions=[
                "Quantum computers achieve sufficient stability",
                "Quantum algorithms for similarity search mature",
                "Cost becomes competitive with classical"
            ],
            triggering_events=[
                "Major quantum computer achieving 1000+ stable qubits",
                "Published quantum algorithm with proven advantage",
                "Tech giant announces quantum embedding service"
            ],
            early_warning_signals=[
                "Quantum computing research papers on similarity search",
                "Startups in quantum ML raising significant funding",
                "Patents filed for quantum embedding techniques"
            ],
            affected_capabilities={"vector_search", "embedding_generation"},
            affected_revenue_streams={"search", "recommendation"},
            required_adaptations=[
                "Develop quantum-ready architecture",
                "Partner with quantum computing providers",
                "Hire quantum ML expertise"
            ],
            adaptation_cost=5_000_000,
            adaptation_timeline=timedelta(days=365),
            response_plan="Monitor quantum developments, maintain architecture flexibility, build partnerships",
            contingency_actions=[
                "Evaluate quantum cloud providers",
                "Prototype quantum embedding algorithms",
                "Design hybrid classical-quantum systems"
            ],
            required_investments={
                "quantum_research": 500_000,
                "quantum_partnerships": 1_000_000,
                "architecture_refactoring": 2_000_000
            },
            success_metrics={
                "quantum_readiness_score": 0.7,
                "adaptation_speed_months": 6
            }
        ))

        scenarios.append(DisruptionScenario(
            scenario_name="Embedding Commoditization",
            category=DisruptionCategory.COMPETITIVE,
            description="Major cloud providers offer free or near-free embedding APIs with excellent quality, commoditizing basic embeddings",
            likelihood=DisruptionLikelihood.HIGH,
            impact=DisruptionImpact.MODERATE,
            time_horizon="1-2 years",
            key_assumptions=[
                "Cloud providers see embeddings as customer acquisition",
                "Costs of serving embeddings drop 10×",
                "Open source models match commercial quality"
            ],
            triggering_events=[
                "AWS/Google/Azure announce free embedding tier",
                "Open source model achieves SOTA on benchmarks",
                "Pricing war between embedding providers"
            ],
            early_warning_signals=[
                "Cloud providers lowering embedding prices",
                "Open source models improving rapidly",
                "Startups pivoting from embeddings to applications"
            ],
            affected_capabilities={"basic_embeddings"},
            affected_revenue_streams={"embedding_api_revenue"},
            required_adaptations=[
                "Shift to specialized/domain-specific embeddings",
                "Focus on proprietary data advantages",
                "Move up stack to applications"
            ],
            adaptation_cost=2_000_000,
            adaptation_timeline=timedelta(days=180),
            response_plan="Accelerate specialization, deepen domain expertise, strengthen data moats",
            contingency_actions=[
                "Develop proprietary training approaches",
                "Build domain-specific evaluation",
                "Create application layer differentiation"
            ],
            required_investments={
                "domain_specialization": 1_000_000,
                "data_collection": 500_000,
                "application_development": 500_000
            },
            success_metrics={
                "specialized_model_advantage": 0.3,
                "application_revenue_percentage": 0.6
            }
        ))

        scenarios.append(DisruptionScenario(
            scenario_name="Privacy Regulation",
            category=DisruptionCategory.REGULATORY,
            description="Strict data privacy laws require on-device embeddings, prohibit centralized vector databases",
            likelihood=DisruptionLikelihood.MEDIUM,
            impact=DisruptionImpact.MAJOR,
            time_horizon="2-3 years",
            key_assumptions=[
                "Privacy concerns reach critical political mass",
                "Technology enables efficient on-device inference",
                "Enforcement is strict and global"
            ],
            triggering_events=[
                "Major data breach involving embeddings",
                "EU/US pass strict embedding data laws",
                "High-profile lawsuits over embedding privacy"
            ],
            early_warning_signals=[
                "Privacy advocacy groups targeting embeddings",
                "Regulatory consultations on AI data",
                "Court cases on embedding data ownership"
            ],
            affected_capabilities={"centralized_storage", "cross_user_learning"},
            affected_revenue_streams={"all_privacy_sensitive"},
            required_adaptations=[
                "Develop federated learning systems",
                "Build on-device embedding generation",
                "Implement differential privacy"
            ],
            adaptation_cost=10_000_000,
            adaptation_timeline=timedelta(days=545),
            response_plan="Proactive privacy engineering, influence standards, build compliant architecture",
            contingency_actions=[
                "Architect privacy-first systems",
                "Develop edge deployment capabilities",
                "Engage in regulatory discussions"
            ],
            required_investments={
                "privacy_engineering": 5_000_000,
                "federated_learning": 3_000_000,
                "compliance_infrastructure": 2_000_000
            },
            success_metrics={
                "privacy_compliance_score": 0.95,
                "on_device_capability": 0.8
            }
        ))

        scenarios.append(DisruptionScenario(
            scenario_name="Multimodal Convergence",
            category=DisruptionCategory.TECHNOLOGY,
            description="Single unified embedding space for text, images, video, audio, code becomes standard, replacing specialized embeddings",
            likelihood=DisruptionLikelihood.HIGH,
            impact=DisruptionImpact.MODERATE,
            time_horizon="1-2 years",
            key_assumptions=[
                "Multimodal training scales effectively",
                "Unified embeddings match specialized quality",
                "Computational costs remain acceptable"
            ],
            triggering_events=[
                "OpenAI/Google release production multimodal embeddings",
                "Research shows unified > specialized embeddings",
                "Major applications adopt multimodal"
            ],
            early_warning_signals=[
                "Multimodal papers showing strong results",
                "Embedding providers announcing multimodal",
                "Customers requesting multimodal support"
            ],
            affected_capabilities={"specialized_embeddings"},
            affected_revenue_streams={"modality_specific_products"},
            required_adaptations=[
                "Develop multimodal training capabilities",
                "Refactor pipeline for unified embeddings",
                "Retrain applications for multimodal"
            ],
            adaptation_cost=3_000_000,
            adaptation_timeline=timedelta(days=270),
            response_plan="Early experimentation, flexible architecture, gradual migration",
            contingency_actions=[
                "Prototype multimodal embeddings",
                "Design migration path",
                "Test application compatibility"
            ],
            required_investments={
                "multimodal_research": 1_000_000,
                "training_infrastructure": 1_500_000,
                "application_migration": 500_000
            },
            success_metrics={
                "multimodal_quality_ratio": 1.1,
                "migration_completion": 0.8
            }
        ))

        return scenarios

    def prioritize_scenarios(
        self,
        scenarios: List[DisruptionScenario]
    ) -> List[Tuple[DisruptionScenario, float]]:
        """Prioritize scenarios by urgency and impact"""

        scored_scenarios = []

        for scenario in scenarios:
            # Calculate urgency score
            likelihood_scores = {
                DisruptionLikelihood.LOW: 0.2,
                DisruptionLikelihood.MEDIUM: 0.5,
                DisruptionLikelihood.HIGH: 0.8,
                DisruptionLikelihood.IMMINENT: 1.0
            }

            impact_scores = {
                DisruptionImpact.MINOR: 0.2,
                DisruptionImpact.MODERATE: 0.5,
                DisruptionImpact.MAJOR: 0.8,
                DisruptionImpact.EXISTENTIAL: 1.0
            }

            likelihood_score = likelihood_scores[scenario.likelihood]
            impact_score = impact_scores[scenario.impact]

            # Priority = likelihood × impact × (1 / time_horizon)
            time_factor = 1.0 if "1-2" in scenario.time_horizon else 0.7 if "2-3" in scenario.time_horizon else 0.4

            priority = likelihood_score * impact_score * time_factor

            scored_scenarios.append((scenario, priority))

        # Sort by priority
        scored_scenarios.sort(key=lambda x: x[1], reverse=True)

        return scored_scenarios

    def design_indicators(
        self,
        scenario: DisruptionScenario
    ) -> List[DisruptionIndicator]:
        """Design monitoring indicators for scenario"""

        indicators = []

        # Create indicator for each warning signal
        for i, signal in enumerate(scenario.early_warning_signals):
            indicators.append(DisruptionIndicator(
                indicator_name=f"{scenario.scenario_name}_signal_{i+1}",
                category=scenario.category,
                data_source=self._infer_data_source(signal),
                current_value=0.0,
                threshold_warning=0.5,
                threshold_critical=0.8,
                related_scenarios=[scenario.scenario_name],
                monitoring_frequency="monthly",
                owner="strategy_team"
            ))

        return indicators

    def _infer_data_source(self, signal: str) -> str:
        """Infer appropriate data source for monitoring signal"""
        if "paper" in signal.lower() or "research" in signal.lower():
            return "arxiv_monitor"
        elif "funding" in signal.lower() or "startup" in signal.lower():
            return "crunchbase_tracker"
        elif "customer" in signal.lower():
            return "sales_feedback"
        elif "patent" in signal.lower():
            return "patent_database"
        else:
            return "news_aggregator"

    def assess_preparedness(
        self,
        scenario: DisruptionScenario,
        current_capabilities: Dict[str, float]
    ) -> Dict[str, any]:
        """Assess how prepared organization is for scenario"""

        # Calculate gap in required capabilities
        capability_gaps = {}
        for capability in scenario.affected_capabilities:
            current = current_capabilities.get(capability, 0.0)
            required = 0.8  # Assume need 80% capability
            gap = max(0, required - current)
            capability_gaps[capability] = gap

        # Calculate adaptation feasibility
        avg_gap = np.mean(list(capability_gaps.values())) if capability_gaps else 0
        time_available = 365  # days, simplified
        time_required = scenario.adaptation_timeline.days

        time_pressure = time_required / time_available if time_available > 0 else 999

        # Determine readiness level
        if avg_gap < 0.2 and time_pressure < 0.5:
            readiness = "READY: Well positioned for this scenario"
        elif avg_gap < 0.4 and time_pressure < 1.0:
            readiness = "PREPARED: Can adapt with moderate effort"
        elif avg_gap < 0.6 and time_pressure < 1.5:
            readiness = "VULNERABLE: Significant adaptation required"
        else:
            readiness = "AT RISK: May not adapt in time"

        return {
            "scenario": scenario.scenario_name,
            "capability_gaps": capability_gaps,
            "average_gap": avg_gap,
            "adaptation_cost": scenario.adaptation_cost,
            "adaptation_time_months": scenario.adaptation_timeline.days / 30,
            "time_pressure": time_pressure,
            "readiness_level": readiness,
            "recommended_actions": scenario.contingency_actions[:3],
            "investment_priority": "HIGH" if time_pressure > 1.0 and avg_gap > 0.4 else "MEDIUM" if avg_gap > 0.3 else "LOW"
        }

# Example usage for disruption planning
def build_disruption_response_strategy(
    organization_profile: Dict[str, any],
    risk_tolerance: str,
    planning_horizon_years: int
) -> Dict[str, any]:
    """Develop comprehensive disruption response strategy"""

    preparedness = DisruptionPreparedness()

    # Generate scenarios
    scenarios = preparedness.develop_scenarios(
        current_position=organization_profile,
        time_horizon_years=planning_horizon_years
    )

    # Prioritize
    prioritized = preparedness.prioritize_scenarios(scenarios)

    # Assess preparedness for each
    assessments = []
    for scenario, priority in prioritized[:5]:  # Top 5
        assessment = preparedness.assess_preparedness(
            scenario,
            organization_profile.get("capabilities", {})
        )
        assessment["priority_score"] = priority
        assessments.append(assessment)

    # Design monitoring indicators
    all_indicators = []
    for scenario, _ in prioritized[:5]:
        indicators = preparedness.design_indicators(scenario)
        all_indicators.extend(indicators)

    # Calculate total investment needed
    total_investment = sum(
        s[0].adaptation_cost * s[1]  # Cost weighted by priority
        for s in prioritized[:5]
    )

    # Determine investment allocation
    if risk_tolerance == "conservative":
        allocation_factor = 0.5  # Prepare for top 2-3 scenarios
    elif risk_tolerance == "moderate":
        allocation_factor = 0.3  # Hedge on top 4-5 scenarios
    else:  # aggressive
        allocation_factor = 0.2  # Minimal preparation, rapid response

    recommended_investment = total_investment * allocation_factor

    return {
        "planning_horizon_years": planning_horizon_years,
        "scenarios_evaluated": len(scenarios),
        "top_scenarios": [
            {
                "name": s[0].scenario_name,
                "priority": s[1],
                "likelihood": s[0].likelihood.value,
                "impact": s[0].impact.value,
                "time_horizon": s[0].time_horizon
            }
            for s in prioritized[:5]
        ],
        "preparedness_assessments": assessments,
        "monitoring_indicators": len(all_indicators),
        "total_adaptation_cost": total_investment,
        "recommended_investment": recommended_investment,
        "risk_tolerance": risk_tolerance,
        "key_recommendations": [
            "Maintain architectural flexibility for rapid pivoting",
            "Invest in top 3-5 most likely/impactful scenarios",
            "Monitor indicators monthly, review scenarios quarterly",
            "Build organizational agility for fast decision-making",
            "Preserve strategic optionality through multi-vendor approaches"
        ]
    }
