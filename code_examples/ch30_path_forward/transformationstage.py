from typing import Tuple

import numpy as np

# Code from Chapter 30
# Book: Embeddings at Scale

"""
Organization-Specific Embedding Future Planning

Architecture:
1. Current state assessment: Capabilities, data, culture, competitive position
2. Vision development: Where embeddings can create transformative value
3. Gap analysis: Differences between current and desired state
4. Transformation roadmap: Phased journey to embedding-native organization
5. Success metrics: Measuring progress toward vision

Vision dimensions:
- Technical capabilities: Infrastructure, models, applications
- Data assets: Proprietary datasets, learning systems
- Organizational capabilities: Teams, processes, culture
- Business outcomes: Revenue, efficiency, market position
- Strategic positioning: Competitive differentiation

Future scenarios by industry:
- Financial services: Real-time risk assessment, personalized advice, fraud detection
- Healthcare: Clinical decision support, drug discovery, personalized treatment
- Retail: Hyper-personalization, dynamic inventory, autonomous supply chain
- Manufacturing: Predictive maintenance, quality prediction, process optimization
- Media: Content recommendation, automated creation, audience understanding
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class TransformationStage(Enum):
    """Stages in embedding-native transformation"""
    EXPLORING = "exploring"  # Learning and experimenting
    PILOTING = "piloting"  # First production applications
    SCALING = "scaling"  # Expanding across organization
    OPTIMIZING = "optimizing"  # Continuous improvement
    LEADING = "leading"  # Industry thought leadership

class CapabilityDomain(Enum):
    """Areas of capability development"""
    TECHNICAL_INFRASTRUCTURE = "technical_infrastructure"
    DATA_ASSETS = "data_assets"
    ORGANIZATIONAL_CAPABILITY = "organizational_capability"
    BUSINESS_APPLICATIONS = "business_applications"
    STRATEGIC_POSITIONING = "strategic_positioning"

@dataclass
class FutureVision:
    """Vision for organization's embedding-powered future"""
    organization_name: str
    industry: str
    current_stage: TransformationStage
    target_stage: TransformationStage
    time_horizon: int  # years

    # Vision statement
    vision_statement: str
    strategic_imperatives: List[str]
    success_definition: str

    # Target capabilities
    technical_targets: Dict[str, float]  # Capability -> target level (0-1)
    data_targets: Dict[str, float]
    organizational_targets: Dict[str, float]

    # Business outcomes
    revenue_impact_target: float  # % increase
    efficiency_impact_target: float  # % improvement
    customer_satisfaction_target: float  # NPS or similar
    market_position_target: str  # "leader", "fast follower", etc.

    # Key applications
    transformative_applications: List[Dict[str, any]]

    # Investment required
    total_investment: float
    annual_run_rate: float

    # Risks and mitigation
    key_risks: List[str]
    mitigation_strategies: List[str]

    metadata: Dict[str, any] = field(default_factory=dict)

@dataclass
class TransformationJourney:
    """Roadmap from current to future state"""
    vision: FutureVision

    # Current state
    current_capabilities: Dict[CapabilityDomain, float]  # 0-1
    current_investments: Dict[str, float]
    current_challenges: List[str]

    # Gap analysis
    capability_gaps: Dict[CapabilityDomain, float]
    priority_gaps: List[Tuple[CapabilityDomain, float]]

    # Transformation phases
    phases: List[Dict[str, any]] = field(default_factory=list)

    # Milestones
    key_milestones: List[Dict[str, any]] = field(default_factory=list)

    # Resources
    team_scaling_plan: Dict[int, int]  # year -> team size
    budget_allocation: Dict[int, float]  # year -> investment

    def calculate_transformation_progress(self) -> Dict[str, any]:
        """Calculate progress toward vision"""

        # Assess current vs target capabilities
        progress_by_domain = {}
        for domain in CapabilityDomain:
            current = self.current_capabilities.get(domain, 0.0)
            target = self.vision.technical_targets.get(domain.value, 0.8)
            gap = target - current
            progress = 1 - (gap / target) if target > 0 else 0
            progress_by_domain[domain.value] = {
                "current": current,
                "target": target,
                "gap": gap,
                "progress_percent": progress * 100
            }

        # Overall progress
        avg_progress = np.mean([
            p["progress_percent"]
            for p in progress_by_domain.values()
        ])

        # Time remaining
        phases_remaining = len([p for p in self.phases if not p.get("completed", False)])
        estimated_months_remaining = phases_remaining * 6  # Assume 6 months per phase

        return {
            "overall_progress_percent": avg_progress,
            "progress_by_domain": progress_by_domain,
            "phases_completed": len(self.phases) - phases_remaining,
            "phases_remaining": phases_remaining,
            "estimated_completion_months": estimated_months_remaining,
            "on_track": avg_progress >= 50 if self.vision.time_horizon <= 2 else avg_progress >= 30
        }

class FuturePlanning:
    """System for envisioning and planning embedding-powered future"""

    def develop_vision(
        self,
        organization_profile: Dict[str, any],
        strategic_goals: List[str],
        constraints: Dict[str, any]
    ) -> FutureVision:
        """Develop comprehensive vision for organization"""

        industry = organization_profile["industry"]
        current_stage = self._assess_current_stage(organization_profile)

        # Determine realistic target stage based on time and investment
        if constraints.get("time_horizon", 3) >= 5 and constraints.get("investment", 0) >= 10_000_000:
            target_stage = TransformationStage.LEADING
        elif constraints.get("time_horizon", 3) >= 3:
            target_stage = TransformationStage.OPTIMIZING
        else:
            target_stage = TransformationStage.SCALING

        # Generate industry-specific vision
        vision_statement = self._generate_vision_statement(industry, target_stage, strategic_goals)

        # Define target capabilities
        technical_targets = self._define_technical_targets(target_stage)
        data_targets = self._define_data_targets(target_stage, industry)
        organizational_targets = self._define_organizational_targets(target_stage)

        # Identify transformative applications
        applications = self._identify_transformative_applications(industry, strategic_goals)

        # Estimate investment required
        investment = self._estimate_transformation_investment(
            current_stage,
            target_stage,
            len(applications),
            constraints.get("time_horizon", 3)
        )

        return FutureVision(
            organization_name=organization_profile.get("name", "Organization"),
            industry=industry,
            current_stage=current_stage,
            target_stage=target_stage,
            time_horizon=constraints.get("time_horizon", 3),
            vision_statement=vision_statement,
            strategic_imperatives=self._generate_imperatives(target_stage),
            success_definition=self._define_success(target_stage, strategic_goals),
            technical_targets=technical_targets,
            data_targets=data_targets,
            organizational_targets=organizational_targets,
            revenue_impact_target=self._estimate_revenue_impact(target_stage),
            efficiency_impact_target=self._estimate_efficiency_impact(target_stage),
            customer_satisfaction_target=self._estimate_satisfaction_impact(target_stage),
            market_position_target=self._determine_market_position(target_stage),
            transformative_applications=applications,
            total_investment=investment["total"],
            annual_run_rate=investment["annual"],
            key_risks=self._identify_risks(target_stage, industry),
            mitigation_strategies=self._develop_mitigations(target_stage)
        )

    def _assess_current_stage(self, profile: Dict) -> TransformationStage:
        """Assess organization's current transformation stage"""

        maturity_score = profile.get("ai_maturity_score", 0.3)

        if maturity_score < 0.3:
            return TransformationStage.EXPLORING
        elif maturity_score < 0.5:
            return TransformationStage.PILOTING
        elif maturity_score < 0.7:
            return TransformationStage.SCALING
        elif maturity_score < 0.9:
            return TransformationStage.OPTIMIZING
        else:
            return TransformationStage.LEADING

    def _generate_vision_statement(
        self,
        industry: str,
        target_stage: TransformationStage,
        goals: List[str]
    ) -> str:
        """Generate compelling vision statement"""

        industry_visions = {
            "financial_services": "Transform into the most intelligent financial institution, where every decision—from risk assessment to customer advice—is powered by real-time semantic understanding of global financial patterns, delivering superior outcomes while reducing risk.",

            "healthcare": "Revolutionize patient care through AI-powered clinical intelligence, where embeddings enable personalized treatment recommendations, drug discovery acceleration, and early disease detection that saves lives at scale.",

            "retail": "Create the most personalized shopping experience in the industry, where every customer interaction is powered by deep understanding of individual preferences, real-time inventory intelligence, and predictive demand management.",

            "manufacturing": "Build the autonomous factory of the future, where embeddings enable predictive maintenance preventing downtime, quality prediction catching defects before they occur, and process optimization maximizing efficiency.",

            "media": "Deliver unprecedented content discovery and engagement, where semantic understanding of viewer preferences and content enables hyper-personalization that keeps audiences engaged while enabling efficient content creation."
        }

        return industry_visions.get(industry,
            "Become embedding-native organization where AI-powered decision making creates sustainable competitive advantage")

    def _define_technical_targets(self, stage: TransformationStage) -> Dict[str, float]:
        """Define target technical capability levels"""

        stage_targets = {
            TransformationStage.PILOTING: {
                "vector_database_scale": 0.3,
                "embedding_quality": 0.6,
                "latency_performance": 0.5,
                "infrastructure_automation": 0.4
            },
            TransformationStage.SCALING: {
                "vector_database_scale": 0.7,
                "embedding_quality": 0.8,
                "latency_performance": 0.8,
                "infrastructure_automation": 0.7
            },
            TransformationStage.OPTIMIZING: {
                "vector_database_scale": 0.9,
                "embedding_quality": 0.9,
                "latency_performance": 0.9,
                "infrastructure_automation": 0.9
            },
            TransformationStage.LEADING: {
                "vector_database_scale": 0.95,
                "embedding_quality": 0.95,
                "latency_performance": 0.95,
                "infrastructure_automation": 0.95
            }
        }

        return stage_targets.get(stage, stage_targets[TransformationStage.SCALING])

    def _define_data_targets(self, stage: TransformationStage, industry: str) -> Dict[str, float]:
        """Define target data capability levels"""

        base_targets = {
            "proprietary_data_scale": 0.7,
            "data_quality": 0.8,
            "feedback_loop_strength": 0.6,
            "domain_coverage": 0.7
        }

        # Adjust based on stage
        if stage == TransformationStage.LEADING:
            return {k: min(v * 1.3, 1.0) for k, v in base_targets.items()}
        elif stage == TransformationStage.OPTIMIZING:
            return {k: min(v * 1.15, 1.0) for k, v in base_targets.items()}
        else:
            return base_targets

    def _define_organizational_targets(self, stage: TransformationStage) -> Dict[str, float]:
        """Define target organizational capability levels"""

        return {
            "team_expertise": 0.8 if stage in [TransformationStage.LEADING, TransformationStage.OPTIMIZING] else 0.6,
            "experimentation_velocity": 0.9 if stage == TransformationStage.LEADING else 0.7,
            "cross_functional_integration": 0.8,
            "learning_culture": 0.85,
            "decision_speed": 0.8
        }

    def _identify_transformative_applications(
        self,
        industry: str,
        goals: List[str]
    ) -> List[Dict[str, any]]:
        """Identify high-impact applications for industry"""

        industry_applications = {
            "financial_services": [
                {
                    "name": "Real-time Risk Intelligence",
                    "description": "Embedding-powered risk assessment updating in real-time with market conditions",
                    "impact": "30-50% improvement in risk-adjusted returns",
                    "timeline": "12-18 months"
                },
                {
                    "name": "Personalized Financial Advice",
                    "description": "AI advisor understanding complete financial situation and goals",
                    "impact": "3-5× increase in customer engagement",
                    "timeline": "18-24 months"
                }
            ],
            "healthcare": [
                {
                    "name": "Clinical Decision Support",
                    "description": "Embedding-based system suggesting diagnoses and treatments",
                    "impact": "20-30% improvement in diagnostic accuracy",
                    "timeline": "24-36 months"
                },
                {
                    "name": "Drug Discovery Acceleration",
                    "description": "Molecular embeddings identifying promising compounds",
                    "impact": "5-10× faster candidate identification",
                    "timeline": "36-48 months"
                }
            ],
            "retail": [
                {
                    "name": "Hyper-Personalized Discovery",
                    "description": "Product recommendations understanding individual style and preferences",
                    "impact": "40-60% increase in conversion rates",
                    "timeline": "9-12 months"
                },
                {
                    "name": "Autonomous Inventory Management",
                    "description": "Predictive system optimizing stock levels and allocation",
                    "impact": "30-50% reduction in excess inventory",
                    "timeline": "15-18 months"
                }
            ],
            "manufacturing": [
                {
                    "name": "Predictive Maintenance",
                    "description": "Equipment failure prediction from sensor embeddings",
                    "impact": "60-80% reduction in unplanned downtime",
                    "timeline": "12-18 months"
                },
                {
                    "name": "Quality Prediction",
                    "description": "Real-time defect prediction from process embeddings",
                    "impact": "50-70% reduction in defects",
                    "timeline": "15-24 months"
                }
            ]
        }

        return industry_applications.get(industry, [
            {
                "name": "Intelligent Search",
                "description": "Semantic search across all organizational knowledge",
                "impact": "30-50% improvement in information discovery",
                "timeline": "6-12 months"
            }
        ])

    def _estimate_transformation_investment(
        self,
        current: TransformationStage,
        target: TransformationStage,
        num_applications: int,
        years: int
    ) -> Dict[str, float]:
        """Estimate investment required for transformation"""

        # Base investment per stage progression
        stage_costs = {
            (TransformationStage.EXPLORING, TransformationStage.PILOTING): 2_000_000,
            (TransformationStage.PILOTING, TransformationStage.SCALING): 5_000_000,
            (TransformationStage.SCALING, TransformationStage.OPTIMIZING): 10_000_000,
            (TransformationStage.OPTIMIZING, TransformationStage.LEADING): 20_000_000
        }

        # Calculate stages to traverse
        stage_order = [
            TransformationStage.EXPLORING,
            TransformationStage.PILOTING,
            TransformationStage.SCALING,
            TransformationStage.OPTIMIZING,
            TransformationStage.LEADING
        ]

        current_idx = stage_order.index(current)
        target_idx = stage_order.index(target)

        total = 0
        for i in range(current_idx, target_idx):
            stage_pair = (stage_order[i], stage_order[i+1])
            total += stage_costs.get(stage_pair, 5_000_000)

        # Add application-specific costs
        total += num_applications * 1_500_000

        # Annual run rate (30% of total)
        annual = total * 0.3

        return {
            "total": total,
            "annual": annual,
            "per_application": total / num_applications if num_applications > 0 else 0
        }

    def _estimate_revenue_impact(self, stage: TransformationStage) -> float:
        """Estimate revenue impact percentage"""
        impacts = {
            TransformationStage.PILOTING: 0.05,  # 5%
            TransformationStage.SCALING: 0.15,  # 15%
            TransformationStage.OPTIMIZING: 0.30,  # 30%
            TransformationStage.LEADING: 0.50  # 50%
        }
        return impacts.get(stage, 0.10)

    def _estimate_efficiency_impact(self, stage: TransformationStage) -> float:
        """Estimate operational efficiency improvement"""
        impacts = {
            TransformationStage.PILOTING: 0.10,  # 10%
            TransformationStage.SCALING: 0.25,  # 25%
            TransformationStage.OPTIMIZING: 0.40,  # 40%
            TransformationStage.LEADING: 0.60  # 60%
        }
        return impacts.get(stage, 0.20)

    def _estimate_satisfaction_impact(self, stage: TransformationStage) -> float:
        """Estimate customer satisfaction improvement (NPS points)"""
        impacts = {
            TransformationStage.PILOTING: 5,
            TransformationStage.SCALING: 15,
            TransformationStage.OPTIMIZING: 25,
            TransformationStage.LEADING: 40
        }
        return impacts.get(stage, 10)

    def _determine_market_position(self, stage: TransformationStage) -> str:
        """Determine expected market position"""
        positions = {
            TransformationStage.EXPLORING: "experimenter",
            TransformationStage.PILOTING: "fast follower",
            TransformationStage.SCALING: "industry standard",
            TransformationStage.OPTIMIZING: "market leader",
            TransformationStage.LEADING: "industry innovator"
        }
        return positions.get(stage, "fast follower")

    def _generate_imperatives(self, stage: TransformationStage) -> List[str]:
        """Generate strategic imperatives"""
        imperatives = {
            TransformationStage.PILOTING: [
                "Prove value with initial production applications",
                "Build foundational technical capabilities",
                "Develop organizational learning culture"
            ],
            TransformationStage.SCALING: [
                "Expand embeddings across all key applications",
                "Establish embedding platform and standards",
                "Build specialized domain expertise"
            ],
            TransformationStage.OPTIMIZING: [
                "Achieve operational excellence in embedding systems",
                "Build proprietary data and model advantages",
                "Develop continuous innovation capabilities"
            ],
            TransformationStage.LEADING: [
                "Shape industry standards and best practices",
                "Build ecosystem partnerships and platforms",
                "Pioneer next-generation embedding applications"
            ]
        }
        return imperatives.get(stage, imperatives[TransformationStage.SCALING])

    def _define_success(self, stage: TransformationStage, goals: List[str]) -> str:
        """Define what success looks like"""
        return f"Successfully progress to {stage.value} stage, delivering measurable business impact through embedding-powered applications while building sustainable competitive advantages"

    def _identify_risks(self, stage: TransformationStage, industry: str) -> List[str]:
        """Identify key risks to transformation"""
        return [
            "Technology commoditization reducing competitive advantages",
            "Talent acquisition and retention challenges",
            "Data privacy regulations limiting capabilities",
            "Competitive pressure from larger players",
            "Organizational resistance to change"
        ]

    def _develop_mitigations(self, stage: TransformationStage) -> List[str]:
        """Develop risk mitigation strategies"""
        return [
            "Focus on proprietary data and domain expertise moats",
            "Build strong engineering culture and competitive compensation",
            "Proactive privacy-first architecture and compliance",
            "Rapid innovation and specialization in key areas",
            "Executive sponsorship and change management investment"
        ]

# Example: Planning your embedding-powered future
def plan_your_future(
    industry: str,
    current_maturity: float,
    strategic_goals: List[str],
    investment_capacity: float,
    time_horizon: int
) -> Dict[str, any]:
    """Complete future planning for organization"""

    planner = FuturePlanning()

    # Organization profile
    profile = {
        "name": "YourOrganization",
        "industry": industry,
        "ai_maturity_score": current_maturity,
        "size": "enterprise",
        "data_assets": "moderate"
    }

    # Constraints
    constraints = {
        "investment": investment_capacity,
        "time_horizon": time_horizon
    }

    # Develop vision
    vision = planner.develop_vision(profile, strategic_goals, constraints)

    # Create transformation journey
    journey = TransformationJourney(
        vision=vision,
        current_capabilities={
            CapabilityDomain.TECHNICAL_INFRASTRUCTURE: current_maturity * 0.8,
            CapabilityDomain.DATA_ASSETS: current_maturity * 0.6,
            CapabilityDomain.ORGANIZATIONAL_CAPABILITY: current_maturity * 0.7,
            CapabilityDomain.BUSINESS_APPLICATIONS: current_maturity * 0.5,
            CapabilityDomain.STRATEGIC_POSITIONING: current_maturity * 0.4
        },
        current_investments={"embedding_systems": investment_capacity * 0.1},
        current_challenges=[
            "Limited embedding expertise",
            "Legacy infrastructure constraints",
            "Organizational change resistance"
        ]
    )

    # Calculate capability gaps
    journey.capability_gaps = {
        domain: vision.technical_targets.get(domain.value, 0.8) -
                journey.current_capabilities.get(domain, 0.0)
        for domain in CapabilityDomain
    }

    # Prioritize gaps
    journey.priority_gaps = sorted(
        journey.capability_gaps.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Define transformation phases
    journey.phases = [
        {
            "phase": 1,
            "name": "Foundation",
            "duration_months": 6,
            "objectives": ["Build team", "Deploy first application", "Establish infrastructure"],
            "investment": vision.total_investment * 0.2,
            "completed": False
        },
        {
            "phase": 2,
            "name": "Expansion",
            "duration_months": 12,
            "objectives": ["Scale to 3-5 applications", "Build platform", "Develop expertise"],
            "investment": vision.total_investment * 0.3,
            "completed": False
        },
        {
            "phase": 3,
            "name": "Optimization",
            "duration_months": 12,
            "objectives": ["Enterprise-wide deployment", "Continuous improvement", "Market leadership"],
            "investment": vision.total_investment * 0.5,
            "completed": False
        }
    ]

    # Key milestones
    journey.key_milestones = [
        {"milestone": "First production application", "target_month": 6, "achieved": False},
        {"milestone": "Platform launch", "target_month": 12, "achieved": False},
        {"milestone": "10+ applications deployed", "target_month": 24, "achieved": False},
        {"milestone": "Industry recognition", "target_month": 30, "achieved": False}
    ]

    # Team scaling
    journey.team_scaling_plan = {
        0: 5,  # Start with 5
        1: 15,  # Year 1: grow to 15
        2: 30,  # Year 2: grow to 30
        3: 50   # Year 3: mature at 50
    }

    # Budget allocation
    journey.budget_allocation = {
        year: vision.annual_run_rate * (1 + 0.2 * year)
        for year in range(time_horizon)
    }

    # Calculate progress
    progress = journey.calculate_transformation_progress()

    return {
        "vision": {
            "statement": vision.vision_statement,
            "target_stage": vision.target_stage.value,
            "time_horizon_years": vision.time_horizon,
            "total_investment": vision.total_investment,
            "expected_revenue_impact": f"{vision.revenue_impact_target*100}%",
            "expected_efficiency_impact": f"{vision.efficiency_impact_target*100}%"
        },
        "transformative_applications": vision.transformative_applications,
        "transformation_journey": {
            "current_stage": vision.current_stage.value,
            "target_stage": vision.target_stage.value,
            "phases": len(journey.phases),
            "total_duration_months": sum(p["duration_months"] for p in journey.phases)
        },
        "capability_gaps": {k.value: v for k, v in journey.priority_gaps},
        "progress": progress,
        "key_milestones": journey.key_milestones,
        "team_plan": journey.team_scaling_plan,
        "budget_plan": journey.budget_allocation,
        "strategic_imperatives": vision.strategic_imperatives,
        "success_metrics": {
            "revenue_impact": vision.revenue_impact_target,
            "efficiency_improvement": vision.efficiency_impact_target,
            "customer_satisfaction": vision.customer_satisfaction_target,
            "market_position": vision.market_position_target
        },
        "next_steps": [
            "Secure executive sponsorship and commitment",
            "Allocate initial budget and recruit core team",
            "Define first pilot application and success criteria",
            "Establish measurement framework",
            "Begin foundation phase execution"
        ]
    }
