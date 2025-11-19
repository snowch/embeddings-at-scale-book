# Code from Chapter 30
# Book: Embeddings at Scale

"""
Strategic Data Collection for Embedding Advantage

Architecture:
1. Usage capture: Every system interaction generates training signal
2. Behavioral logging: User actions provide ground truth for relevance
3. Expert annotation: Domain experts label critical examples
4. Synthetic generation: Automated systems create training data at scale
5. Partnership data: External data sources through API integrations

Compounding mechanisms:
- More users → More behavioral signals → Better embeddings → More users
- More data → Better rare pattern detection → Higher quality → More usage
- Larger scale → More edge cases discovered → Improved handling → Better reliability

Investment priorities:
- Maximize data collection rate: Capture every possible signal
- Improve data quality: Clean, accurate, representative samples
- Build feedback loops: Automatically convert usage into training data
- Create unique sources: Data competitors cannot easily obtain
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Set, Tuple

import numpy as np


class DataAssetType(Enum):
    """Types of proprietary data assets"""

    USER_BEHAVIOR = "user_behavior"  # Clicks, dwell, conversions
    EXPERT_LABELS = "expert_labels"  # Domain expert annotations
    INTERACTION_FEEDBACK = "interaction_feedback"  # Explicit user feedback
    SYNTHETIC_DATA = "synthetic_data"  # Simulation or generation
    PARTNERSHIP_DATA = "partnership_data"  # Third-party integrations
    OPERATIONAL_LOGS = "operational_logs"  # System telemetry
    DOMAIN_CORPUS = "domain_corpus"  # Specialized text/media collection


@dataclass
class DataMoat:
    """Proprietary data asset tracking"""

    asset_type: DataAssetType
    collection_start: datetime
    total_examples: int
    monthly_growth_rate: float  # Examples per month
    uniqueness_score: float  # 0-1, how hard for competitors to replicate
    quality_score: float  # 0-1, accuracy and completeness
    business_value: float  # Estimated revenue impact
    collection_cost: float  # Monthly cost to maintain

    # Coverage and representation
    domain_coverage: float  # 0-1, fraction of domain represented
    rare_pattern_count: int  # Long-tail phenomena captured
    temporal_recency: timedelta  # How fresh the data is

    # Competitive advantage metrics
    time_to_replicate: timedelta  # Estimated competitor catch-up time
    replication_cost: float  # Estimated competitor investment needed
    network_effect_strength: float  # 0-1, how much advantage compounds

    # Integration and usage
    active_models: Set[str]  # Models trained on this data
    downstream_applications: Set[str]  # Applications depending on it
    feedback_loop_strength: float  # 0-1, how well usage improves data

    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class AdvantageCompounding:
    """Tracking how advantages compound over time"""

    advantage_source: str
    initial_value: float  # Starting competitive advantage
    compound_rate: float  # Monthly growth in advantage
    investment_required: float  # Monthly spend to maintain

    # Compounding mechanisms
    scale_effects: float  # Advantage from data/user scale
    learning_effects: float  # Advantage from accumulated knowledge
    network_effects: float  # Advantage from user/partner ecosystem
    expertise_effects: float  # Advantage from team capability

    # Defensive moat
    switching_costs: float  # Cost for users to switch to competitor
    integration_complexity: float  # Difficulty replicating integrations
    regulatory_barriers: float  # Compliance/legal advantage

    # Time dynamics
    months_to_neutralize: float  # Time for competitor to match
    erosion_rate: float  # Monthly decay without investment

    # Strategic value
    revenue_contribution: float  # Monthly revenue attributable
    margin_improvement: float  # Profit margin versus competitors
    market_share_impact: float  # Market share gain from advantage

    measurements: List[Tuple[datetime, float]] = field(default_factory=list)

    def project_advantage(self, months: int) -> float:
        """Project advantage value over time"""
        # Compound growth with diminishing returns
        value = self.initial_value
        for _month in range(months):
            growth = self.compound_rate * (1 - value / 100)  # Diminishing returns
            erosion = self.erosion_rate
            value = value * (1 + growth - erosion)
        return value

    def calculate_roi(self, months: int) -> float:
        """Calculate return on moat investment"""
        total_investment = self.investment_required * months
        total_revenue = sum(
            self.revenue_contribution * (1 + self.compound_rate) ** m for m in range(months)
        )
        return (total_revenue - total_investment) / total_investment if total_investment > 0 else 0


class SustainableAdvantageBuilder:
    """System for building and tracking sustainable advantages"""

    def __init__(self):
        self.data_moats: Dict[str, DataMoat] = {}
        self.advantages: Dict[str, AdvantageCompounding] = {}
        self.start_date = datetime.now()

    def assess_data_moat(
        self, asset_type: DataAssetType, current_size: int, growth_rate: float, uniqueness: float
    ) -> Dict[str, float]:
        """Assess strength and sustainability of data moat"""

        # Calculate time for competitor to reach parity
        competitor_catch_up_months = (
            (
                current_size / (growth_rate * 2)  # Assume competitor grows 2× faster
            )
            if growth_rate > 0
            else float("inf")
        )

        # Estimate replication cost based on data collection challenges
        collection_cost_per_example = 0.01 * (1 + uniqueness * 10)  # Higher for unique data
        replication_cost = current_size * collection_cost_per_example

        # Network effect strength - how much having more data attracts more users
        network_strength = min(uniqueness * 0.8, 1.0)  # Unique data creates stronger effects

        # Compound value over time
        months_ahead = 36  # 3-year horizon
        advantage_multiplier = (1 + growth_rate / current_size) ** months_ahead

        return {
            "competitive_lead_months": competitor_catch_up_months,
            "replication_cost_millions": replication_cost / 1_000_000,
            "network_effect_strength": network_strength,
            "advantage_multiplier_3y": advantage_multiplier,
            "sustainability_score": min(
                (competitor_catch_up_months / 36) * uniqueness * network_strength, 1.0
            ),
        }

    def optimize_investment_allocation(
        self,
        total_budget: float,
        investment_options: List[Tuple[str, float, float]],  # name, cost, compound_rate
    ) -> Dict[str, float]:
        """Optimize budget allocation across advantage-building activities"""

        # Simple greedy allocation prioritizing highest ROI
        sorted_options = sorted(
            investment_options,
            key=lambda x: x[2] / x[1],  # Compound rate per dollar
            reverse=True,
        )

        allocation = {}
        remaining_budget = total_budget

        for name, cost, _compound_rate in sorted_options:
            if remaining_budget >= cost:
                allocation[name] = cost
                remaining_budget -= cost
            else:
                allocation[name] = remaining_budget
                break

        return allocation

    def identify_moat_opportunities(
        self,
        current_capabilities: Dict[str, float],
        market_gaps: Dict[str, float],
        resources_available: Dict[str, float],
    ) -> List[Dict[str, any]]:
        """Identify highest-value opportunities for building moats"""

        opportunities = []

        # Data moat opportunities
        if resources_available.get("data_collection_capacity", 0) > 0:
            opportunities.append(
                {
                    "type": "data_moat",
                    "focus": "behavioral_signal_collection",
                    "estimated_value": market_gaps.get("user_understanding", 0) * 10,
                    "time_to_value": 6,  # months
                    "sustainability": 0.85,
                    "investment_required": 500_000,  # Initial setup
                }
            )

        # Domain expertise opportunities
        if resources_available.get("expert_hiring_capacity", 0) > 0:
            opportunities.append(
                {
                    "type": "expertise_moat",
                    "focus": "domain_specialist_team",
                    "estimated_value": market_gaps.get("domain_understanding", 0) * 8,
                    "time_to_value": 12,
                    "sustainability": 0.75,
                    "investment_required": 2_000_000,  # Annual comp for team
                }
            )

        # Learning system opportunities
        if current_capabilities.get("ml_platform_maturity", 0) > 0.7:
            opportunities.append(
                {
                    "type": "learning_moat",
                    "focus": "continuous_improvement_loops",
                    "estimated_value": market_gaps.get("adaptation_speed", 0) * 12,
                    "time_to_value": 9,
                    "sustainability": 0.90,
                    "investment_required": 1_000_000,  # Platform development
                }
            )

        # Sort by value/investment ratio
        opportunities.sort(
            key=lambda x: x["estimated_value"] / x["investment_required"], reverse=True
        )

        return opportunities


# Example usage for strategic planning
def develop_moat_strategy(
    current_position: Dict[str, any], available_budget: float, time_horizon_months: int
) -> Dict[str, any]:
    """Develop comprehensive moat-building strategy"""

    builder = SustainableAdvantageBuilder()

    # Assess current data assets
    data_strength = builder.assess_data_moat(
        DataAssetType.USER_BEHAVIOR,
        current_size=current_position.get("behavioral_examples", 0),
        growth_rate=current_position.get("monthly_growth", 0),
        uniqueness=current_position.get("data_uniqueness", 0.5),
    )

    # Identify investment opportunities
    opportunities = builder.identify_moat_opportunities(
        current_capabilities=current_position.get("capabilities", {}),
        market_gaps=current_position.get("market_gaps", {}),
        resources_available=current_position.get("resources", {}),
    )

    # Allocate budget
    investment_options = [
        (opp["focus"], opp["investment_required"], opp["estimated_value"]) for opp in opportunities
    ]

    allocation = builder.optimize_investment_allocation(available_budget, investment_options)

    # Project outcomes
    projected_advantages = []
    for opp in opportunities:
        if opp["focus"] in allocation:
            advantage = AdvantageCompounding(
                advantage_source=opp["focus"],
                initial_value=5.0,  # Starting % advantage
                compound_rate=opp["estimated_value"] / 100,
                investment_required=allocation[opp["focus"]] / time_horizon_months,
                scale_effects=0.3,
                learning_effects=0.2,
                network_effects=0.4,
                expertise_effects=0.1,
                months_to_neutralize=opp["time_to_value"] * 2,
                erosion_rate=0.02,
            )

            projected_value = advantage.project_advantage(time_horizon_months)
            roi = advantage.calculate_roi(time_horizon_months)

            projected_advantages.append(
                {
                    "source": opp["focus"],
                    "type": opp["type"],
                    "projected_advantage": projected_value,
                    "roi": roi,
                    "sustainability": opp["sustainability"],
                    "investment": allocation[opp["focus"]],
                }
            )

    return {
        "current_strength": data_strength,
        "opportunities": opportunities,
        "allocation": allocation,
        "projected_advantages": projected_advantages,
        "total_investment": sum(allocation.values()),
        "expected_3y_lead": sum(p["projected_advantage"] for p in projected_advantages),
        "blended_roi": np.mean([p["roi"] for p in projected_advantages]),
    }
