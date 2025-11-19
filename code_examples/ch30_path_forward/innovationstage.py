import numpy as np

# Code from Chapter 30
# Book: Embeddings at Scale

"""
Research Integration and Innovation Pipeline

Architecture:
1. Research monitoring: Track papers, preprints, conference proceedings
2. Relevance filtering: Identify high-potential advances for organization
3. Feasibility assessment: Evaluate technical viability and resource requirements
4. Prototyping: Build minimal implementations testing key hypotheses
5. Production adaptation: Engineer research prototypes for scale and reliability
6. Deployment: Integrate into production systems with monitoring
7. Impact measurement: Quantify business value and technical improvement

Innovation pipeline stages:
- Discovery: Find promising research directions (100+ papers reviewed monthly)
- Evaluation: Assess applicability and value (20-30 assessed deeply)
- Prototyping: Build working implementations (5-10 prototyped)
- Production: Deploy to real systems (2-3 reach production)
- Scale: Expand across organization (1-2 scale widely)

Success metrics:
- Research-to-production time: 3-6 months for validated innovations
- Production success rate: 40-60% of prototypes reach production
- Business impact: 20%+ improvement in key metrics
- Knowledge accumulation: Learnings captured even from failed experiments
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set


class InnovationStage(Enum):
    """Stages in innovation pipeline"""
    DISCOVERED = "discovered"  # Identified as potentially valuable
    EVALUATING = "evaluating"  # Deep assessment underway
    PROTOTYPING = "prototyping"  # Building minimal implementation
    VALIDATING = "validating"  # Testing in realistic conditions
    PRODUCTIONIZING = "productionizing"  # Engineering for scale
    DEPLOYED = "deployed"  # In production systems
    SCALED = "scaled"  # Expanded across organization
    RETIRED = "retired"  # Removed or superseded

class InnovationType(Enum):
    """Types of innovations"""
    INCREMENTAL = "incremental"  # 10-30% improvement to existing
    ADJACENT = "adjacent"  # New capability related to existing
    BREAKTHROUGH = "breakthrough"  # Fundamentally new approach
    PLATFORM = "platform"  # Enabling technology for many applications

@dataclass
class ResearchItem:
    """Research paper or technique being evaluated"""
    title: str
    authors: List[str]
    publication_venue: str
    publication_date: datetime
    arxiv_id: Optional[str]

    # Categorization
    innovation_type: InnovationType
    technical_areas: Set[str]  # e.g., "contrastive learning", "quantization"
    potential_applications: Set[str]

    # Assessment
    relevance_score: float  # 0-1, how applicable to our problems
    novelty_score: float  # 0-1, how new vs incremental
    feasibility_score: float  # 0-1, how practical to implement
    impact_potential: float  # 0-1, expected business value

    # Resource requirements
    estimated_engineering_months: float
    estimated_compute_cost: float
    required_expertise: Set[str]
    data_requirements: str

    # Status tracking
    stage: InnovationStage
    assigned_team: Optional[str]
    prototype_repo: Optional[str]
    evaluation_results: Dict[str, float] = field(default_factory=dict)

    # Decision tracking
    go_no_go_decision: Optional[bool] = None
    decision_rationale: Optional[str] = None
    decision_date: Optional[datetime] = None

    notes: str = ""
    metadata: Dict[str, any] = field(default_factory=dict)

@dataclass
class InnovationExperiment:
    """Specific experiment testing innovation hypothesis"""
    hypothesis: str
    research_basis: Optional[str]  # Link to ResearchItem
    experiment_owner: str
    start_date: datetime

    # Experiment design
    baseline: str  # Current approach being compared against
    innovation: str  # New approach being tested
    success_metrics: Dict[str, float]  # Target improvements

    # Resource allocation
    team_size: int
    duration_weeks: int
    compute_budget: float

    # Results
    actual_results: Dict[str, float] = field(default_factory=dict)
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    qualitative_learnings: List[str] = field(default_factory=list)

    # Outcomes
    success: Optional[bool] = None
    production_decision: bool = False
    production_timeline: Optional[timedelta] = None
    expected_roi: Optional[float] = None

    completion_date: Optional[datetime] = None

@dataclass
class InnovationPortfolio:
    """Managing balanced portfolio of innovations"""

    # Portfolio allocation targets
    incremental_allocation: float = 0.70  # 70% on incremental improvements
    adjacent_allocation: float = 0.20  # 20% on adjacent innovations
    breakthrough_allocation: float = 0.10  # 10% on breakthrough experiments

    # Active innovations
    active_items: Dict[str, ResearchItem] = field(default_factory=dict)
    active_experiments: Dict[str, InnovationExperiment] = field(default_factory=dict)

    # Historical tracking
    completed_experiments: List[InnovationExperiment] = field(default_factory=list)
    production_deployments: List[Dict[str, any]] = field(default_factory=list)

    # Resource tracking
    total_engineering_capacity: float = 100.0  # Engineering months per quarter
    allocated_capacity: Dict[InnovationType, float] = field(default_factory=dict)

    def check_portfolio_balance(self) -> Dict[str, any]:
        """Verify portfolio allocation matches targets"""

        current_allocation = defaultdict(float)
        for item in self.active_items.values():
            if item.stage in [InnovationStage.PROTOTYPING, InnovationStage.VALIDATING]:
                current_allocation[item.innovation_type] += item.estimated_engineering_months

        total_allocated = sum(current_allocation.values())

        if total_allocated == 0:
            return {"balanced": True, "message": "No active innovations"}

        actual_percentages = {
            itype: current_allocation[itype] / total_allocated
            for itype in InnovationType
        }

        targets = {
            InnovationType.INCREMENTAL: self.incremental_allocation,
            InnovationType.ADJACENT: self.adjacent_allocation,
            InnovationType.BREAKTHROUGH: self.breakthrough_allocation,
        }

        imbalances = {
            itype: actual_percentages.get(itype, 0) - targets.get(itype, 0)
            for itype in targets
        }

        max_imbalance = max(abs(v) for v in imbalances.values())
        balanced = max_imbalance < 0.15  # Within 15% is acceptable

        return {
            "balanced": balanced,
            "current_allocation": actual_percentages,
            "target_allocation": targets,
            "imbalances": imbalances,
            "max_imbalance": max_imbalance,
            "recommendation": self._get_rebalancing_recommendation(imbalances)
        }

    def _get_rebalancing_recommendation(self, imbalances: Dict) -> str:
        """Suggest actions to rebalance portfolio"""
        recommendations = []

        for itype, imbalance in imbalances.items():
            if imbalance > 0.15:
                recommendations.append(
                    f"Reduce {itype.value} investments by {imbalance*100:.0f}%"
                )
            elif imbalance < -0.15:
                recommendations.append(
                    f"Increase {itype.value} investments by {-imbalance*100:.0f}%"
                )

        return "; ".join(recommendations) if recommendations else "Portfolio balanced"

class InnovationPipeline:
    """System for managing research integration and innovation"""

    def __init__(
        self,
        monthly_research_review_capacity: int = 100,
        quarterly_innovation_budget: float = 1_000_000
    ):
        self.portfolio = InnovationPortfolio()
        self.research_sources: List[str] = []
        self.review_capacity = monthly_research_review_capacity
        self.innovation_budget = quarterly_innovation_budget

    def evaluate_research_item(
        self,
        item: ResearchItem,
        evaluation_team: List[str]
    ) -> Dict[str, any]:
        """Systematic evaluation of research for potential adoption"""

        # Technical feasibility assessment
        technical_score = self._assess_technical_feasibility(item)

        # Business value assessment
        business_score = self._assess_business_value(item)

        # Resource requirement assessment
        resource_score = self._assess_resource_requirements(item)

        # Risk assessment
        risk_score = self._assess_implementation_risks(item)

        # Overall priority score
        priority = (
            technical_score * 0.3 +
            business_score * 0.4 +
            resource_score * 0.2 +
            risk_score * 0.1
        )

        # Recommendation
        if priority > 0.7:
            recommendation = "PRIORITY: Fast-track to prototyping"
        elif priority > 0.5:
            recommendation = "CONSIDER: Prototype when resources available"
        elif priority > 0.3:
            recommendation = "MONITOR: Track developments, revisit in 3-6 months"
        else:
            recommendation = "PASS: Not aligned with current priorities"

        return {
            "item_id": f"{item.arxiv_id or item.title}",
            "technical_feasibility": technical_score,
            "business_value": business_score,
            "resource_efficiency": resource_score,
            "risk_level": 1 - risk_score,
            "priority_score": priority,
            "recommendation": recommendation,
            "evaluation_team": evaluation_team,
            "evaluation_date": datetime.now()
        }

    def _assess_technical_feasibility(self, item: ResearchItem) -> float:
        """Assess whether we can actually implement this"""

        # Check if we have required expertise
        expertise_available = len(item.required_expertise) * 0.2  # Simplified

        # Check data requirements
        data_feasible = 0.8 if "proprietary data" not in item.data_requirements.lower() else 0.5

        # Check computational requirements
        compute_feasible = min(item.estimated_compute_cost / 100_000, 1.0)

        # Combine factors
        return (expertise_available + data_feasible + compute_feasible) / 3

    def _assess_business_value(self, item: ResearchItem) -> float:
        """Assess potential business impact"""

        # Map potential applications to business value
        application_value = len(item.potential_applications) * 0.15

        # Consider impact potential
        impact_factor = item.impact_potential

        # Consider innovation type (breakthroughs more valuable but risky)
        if item.innovation_type == InnovationType.BREAKTHROUGH:
            type_multiplier = 1.5
        elif item.innovation_type == InnovationType.PLATFORM:
            type_multiplier = 1.3
        elif item.innovation_type == InnovationType.ADJACENT:
            type_multiplier = 1.1
        else:
            type_multiplier = 1.0

        return min(application_value * impact_factor * type_multiplier, 1.0)

    def _assess_resource_requirements(self, item: ResearchItem) -> float:
        """Assess resource efficiency (inverse of requirements)"""

        # Engineering time (6 months is threshold for acceptable)
        time_score = max(0, 1 - item.estimated_engineering_months / 6)

        # Cost (100K is threshold)
        cost_score = max(0, 1 - item.estimated_compute_cost / 100_000)

        # Combine (higher is better = more resource efficient)
        return (time_score + cost_score) / 2

    def _assess_implementation_risks(self, item: ResearchItem) -> float:
        """Assess risks in implementation (higher = lower risk)"""

        # Novelty risk (very novel = higher risk)
        novelty_risk = 1 - item.novelty_score * 0.5

        # Feasibility risk
        feasibility_risk = item.feasibility_score

        # Combine
        return (novelty_risk + feasibility_risk) / 2

    def design_experiment(
        self,
        research_item: ResearchItem,
        baseline_system: str,
        success_threshold: float = 0.15  # 15% improvement
    ) -> InnovationExperiment:
        """Design rigorous experiment to validate innovation"""

        # Define success metrics based on innovation type
        if "search" in research_item.potential_applications:
            metrics = {
                "ndcg@10": success_threshold,
                "mrr": success_threshold,
                "user_satisfaction": 0.10
            }
        elif "recommendation" in research_item.potential_applications:
            metrics = {
                "click_through_rate": success_threshold,
                "conversion_rate": success_threshold * 0.5,
                "user_engagement": 0.10
            }
        else:
            metrics = {
                "task_accuracy": success_threshold,
                "latency_improvement": 0.20,
                "cost_reduction": 0.15
            }

        # Estimate experiment duration
        if research_item.innovation_type == InnovationType.INCREMENTAL:
            duration = 4  # weeks
            team_size = 2
        elif research_item.innovation_type == InnovationType.ADJACENT:
            duration = 8
            team_size = 3
        else:  # BREAKTHROUGH
            duration = 12
            team_size = 4

        return InnovationExperiment(
            hypothesis=f"Implementing {research_item.title} will improve {baseline_system}",
            research_basis=research_item.title,
            experiment_owner="innovation_team",
            start_date=datetime.now(),
            baseline=baseline_system,
            innovation=research_item.title,
            success_metrics=metrics,
            team_size=team_size,
            duration_weeks=duration,
            compute_budget=research_item.estimated_compute_cost
        )

    def track_experiment_results(
        self,
        experiment: InnovationExperiment,
        results: Dict[str, float]
    ) -> Dict[str, any]:
        """Analyze experiment results and make go/no-go decision"""

        experiment.actual_results = results

        # Check if success criteria met
        success_count = 0
        total_metrics = len(experiment.success_metrics)

        improvements = {}
        for metric, target_improvement in experiment.success_metrics.items():
            if metric in results:
                actual_improvement = results[metric]
                improvements[metric] = actual_improvement
                if actual_improvement >= target_improvement:
                    success_count += 1

        # Declare success if majority of metrics improved
        experiment.success = success_count >= (total_metrics / 2)

        # Production decision based on success and strategic fit
        if experiment.success:
            # Calculate expected ROI
            avg_improvement = np.mean(list(improvements.values()))
            estimated_annual_value = avg_improvement * 1_000_000  # Simplified
            implementation_cost = experiment.compute_budget + (
                experiment.team_size * experiment.duration_weeks / 4 * 50_000
            )
            experiment.expected_roi = estimated_annual_value / implementation_cost

            # Decide on production
            if experiment.expected_roi > 3.0:
                experiment.production_decision = True
                experiment.production_timeline = timedelta(days=90)
            elif experiment.expected_roi > 1.5:
                experiment.production_decision = True
                experiment.production_timeline = timedelta(days=180)

        return {
            "experiment": experiment.hypothesis,
            "success": experiment.success,
            "improvements": improvements,
            "production_ready": experiment.production_decision,
            "expected_roi": experiment.expected_roi,
            "recommendation": self._get_experiment_recommendation(experiment)
        }

    def _get_experiment_recommendation(self, experiment: InnovationExperiment) -> str:
        """Generate recommendation based on experiment results"""

        if experiment.success and experiment.production_decision:
            return f"DEPLOY: Move to production within {experiment.production_timeline.days} days"
        elif experiment.success:
            return "ITERATE: Promising results but needs optimization before production"
        else:
            learnings = ", ".join(experiment.qualitative_learnings[:3])
            return f"ARCHIVE: Did not meet success criteria. Learnings: {learnings}"

    def generate_innovation_roadmap(
        self,
        quarters: int = 4
    ) -> Dict[str, any]:
        """Generate innovation roadmap for next N quarters"""

        roadmap = {
            f"Q{i+1}": {
                "incremental_initiatives": [],
                "adjacent_innovations": [],
                "breakthrough_experiments": [],
                "expected_outcomes": []
            }
            for i in range(quarters)
        }

        # Allocate initiatives across quarters
        for item_id, item in self.portfolio.active_items.items():
            if item.stage in [InnovationStage.EVALUATING, InnovationStage.PROTOTYPING]:
                # Estimate which quarter this will deploy
                quarters_ahead = int(item.estimated_engineering_months / 3)
                if quarters_ahead < quarters:
                    quarter_key = f"Q{quarters_ahead + 1}"

                    if item.innovation_type == InnovationType.INCREMENTAL:
                        roadmap[quarter_key]["incremental_initiatives"].append(item.title)
                    elif item.innovation_type == InnovationType.ADJACENT:
                        roadmap[quarter_key]["adjacent_innovations"].append(item.title)
                    else:
                        roadmap[quarter_key]["breakthrough_experiments"].append(item.title)

                    roadmap[quarter_key]["expected_outcomes"].append({
                        "innovation": item.title,
                        "impact": item.impact_potential,
                        "applications": list(item.potential_applications)
                    })

        return roadmap

# Example usage
def build_innovation_program(
    organization_size: str,
    innovation_maturity: str,
    annual_budget: float
) -> Dict[str, any]:
    """Design innovation program appropriate for organization"""

    pipeline = InnovationPipeline(
        monthly_research_review_capacity=100,
        quarterly_innovation_budget=annual_budget / 4
    )

    # Set portfolio allocation based on maturity
    if innovation_maturity == "early":
        # More conservative, focus on incremental
        pipeline.portfolio.incremental_allocation = 0.80
        pipeline.portfolio.adjacent_allocation = 0.15
        pipeline.portfolio.breakthrough_allocation = 0.05
    elif innovation_maturity == "mature":
        # More aggressive, seek breakthroughs
        pipeline.portfolio.incremental_allocation = 0.60
        pipeline.portfolio.adjacent_allocation = 0.25
        pipeline.portfolio.breakthrough_allocation = 0.15

    # Generate initial roadmap
    roadmap = pipeline.generate_innovation_roadmap(quarters=4)

    # Check portfolio balance
    balance = pipeline.portfolio.check_portfolio_balance()

    return {
        "pipeline_capacity": pipeline.review_capacity,
        "quarterly_budget": pipeline.innovation_budget,
        "portfolio_allocation": {
            "incremental": pipeline.portfolio.incremental_allocation,
            "adjacent": pipeline.portfolio.adjacent_allocation,
            "breakthrough": pipeline.portfolio.breakthrough_allocation
        },
        "roadmap": roadmap,
        "portfolio_balance": balance,
        "organization_size": organization_size,
        "maturity_level": innovation_maturity
    }
