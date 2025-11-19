# Code from Chapter 30
# Book: Embeddings at Scale

"""
Ecosystem Partnership Strategy and Management

Architecture:
1. Partnership identification: Find high-value collaboration opportunities
2. Value assessment: Evaluate potential benefits and risks
3. Negotiation: Structure win-win agreements with clear boundaries
4. Integration: Connect partner capabilities with internal systems
5. Governance: Manage ongoing relationship and value delivery
6. Evolution: Adapt partnerships as strategy and market evolve

Partnership types:
- Vendor partnerships: Technology providers and platform companies
- Academic partnerships: Universities and research institutions  
- Open source partnerships: Community projects and foundations
- Industry consortiums: Standards bodies and collaborative initiatives
- Customer partnerships: Design partners and early adopters
- Startup partnerships: Emerging technology and innovation access

Strategic principles:
- Collaborate on infrastructure, compete on applications
- Share non-differentiating, protect advantages
- Maintain strategic optionality through multi-vendor approaches
- Build genuine win-win rather than extractive relationships
- Invest in relationships proportional to strategic value
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
from datetime import datetime, timedelta

class PartnershipType(Enum):
    """Types of strategic partnerships"""
    VENDOR = "vendor"  # Technology provider
    ACADEMIC = "academic"  # Research institution
    OPEN_SOURCE = "open_source"  # Community project
    INDUSTRY_CONSORTIUM = "industry_consortium"  # Standards body
    CUSTOMER = "customer"  # Design partner
    STARTUP = "startup"  # Emerging technology
    INTEGRATION = "integration"  # Complementary product

class PartnershipValue(Enum):
    """Value contribution categories"""
    TECHNOLOGY_ACCESS = "technology_access"
    COST_REDUCTION = "cost_reduction"
    TIME_TO_MARKET = "time_to_market"
    RISK_MITIGATION = "risk_mitigation"
    MARKET_ACCESS = "market_access"
    TALENT_ACCESS = "talent_access"
    INNOVATION_ACCELERATION = "innovation_acceleration"

@dataclass
class Partnership:
    """Strategic partnership tracking"""
    partner_name: str
    partnership_type: PartnershipType
    start_date: datetime
    
    # Strategic alignment
    strategic_value: Set[PartnershipValue]
    business_objectives: List[str]
    success_metrics: Dict[str, float]
    
    # Scope definition
    collaboration_areas: Set[str]  # What we collaborate on
    protected_areas: Set[str]  # What we keep proprietary
    shared_ip: bool
    data_sharing: bool
    
    # Commercial terms
    financial_commitment: float  # Annual spend or investment
    duration_years: int
    renewal_terms: str
    exit_clauses: List[str]
    
    # Governance
    executive_sponsors: Dict[str, str]  # Internal and partner
    working_team: List[str]
    meeting_cadence: str  # e.g., "monthly", "quarterly"
    escalation_path: List[str]
    
    # Value tracking
    benefits_realized: Dict[str, float] = field(default_factory=dict)
    issues_encountered: List[str] = field(default_factory=list)
    relationship_health: float = 1.0  # 0-1 score
    
    # Risk management
    dependency_level: float = 0.0  # 0-1, how dependent we are
    switching_cost: float = 0.0  # Cost to move to alternative
    competitive_risk: float = 0.0  # Risk partner becomes competitor
    
    notes: str = ""
    metadata: Dict[str, any] = field(default_factory=dict)

@dataclass
class OpenSourceContribution:
    """Open source project contributions tracking"""
    project_name: str
    project_url: str
    contribution_type: str  # "code", "documentation", "maintenance", "funding"
    
    # Contribution details
    engineering_months_invested: float
    financial_contribution: float
    start_date: datetime
    
    # Strategic rationale
    business_value: str  # Why contributing
    alternatives_cost: float  # Cost of building ourselves
    ecosystem_benefit: str  # How others benefit
    
    # Recognition and influence
    contributor_status: str  # "user", "contributor", "committer", "steering"
    influence_level: float  # 0-1, ability to shape direction
    
    # Value received
    capabilities_gained: List[str]
    cost_savings: float  # From using vs building
    time_savings: timedelta  # Faster than building
    
    maintenance: Dict[str, any] = field(default_factory=dict)

@dataclass
class AcademicPartnership:
    """Academic research partnership"""
    institution_name: str
    research_group: str
    principal_investigators: List[str]
    
    # Partnership structure
    partnership_type: str  # "sponsored research", "joint lab", "internship program"
    funding_amount: float
    duration_years: int
    start_date: datetime
    
    # Research focus
    research_areas: Set[str]
    expected_outcomes: List[str]
    publication_rights: str
    ip_ownership: str
    
    # Talent pipeline
    interns_hosted: int = 0
    hires_from_program: int = 0
    
    # Value realization
    papers_published: List[str] = field(default_factory=list)
    techniques_adopted: List[str] = field(default_factory=list)
    estimated_research_acceleration: float = 0.0  # Months saved
    
    relationship_status: str = "active"

class PartnershipPortfolio:
    """Managing portfolio of strategic partnerships"""
    
    def __init__(self):
        self.partnerships: Dict[str, Partnership] = {}
        self.open_source: Dict[str, OpenSourceContribution] = {}
        self.academic: Dict[str, AcademicPartnership] = {}
    
    def assess_vendor_partnership(
        self,
        vendor: str,
        capabilities: List[str],
        cost_annual: float,
        lock_in_risk: float
    ) -> Dict[str, any]:
        """Evaluate potential vendor partnership"""
        
        # Calculate value factors
        build_cost = sum(self._estimate_build_cost(cap) for cap in capabilities)
        time_to_market_value = self._estimate_time_value(capabilities)
        
        # Calculate risks
        dependency_risk = lock_in_risk * cost_annual * 5  # 5-year exposure
        competitive_risk = self._assess_competitive_risk(vendor, capabilities)
        
        # Net value
        total_value = build_cost + time_to_market_value
        total_risk = dependency_risk + competitive_risk
        net_value = total_value - total_risk - (cost_annual * 3)  # 3-year TCO
        
        # Decision
        if net_value > 1_000_000 and lock_in_risk < 0.5:
            recommendation = "PARTNER: Strong value with manageable risk"
        elif net_value > 0:
            recommendation = "CONSIDER: Positive value but monitor risks carefully"
        else:
            recommendation = "BUILD: Better to develop internally"
        
        return {
            "vendor": vendor,
            "build_cost_avoided": build_cost,
            "time_to_market_value": time_to_market_value,
            "3y_cost": cost_annual * 3,
            "dependency_risk": dependency_risk,
            "competitive_risk": competitive_risk,
            "net_value": net_value,
            "lock_in_risk": lock_in_risk,
            "recommendation": recommendation
        }
    
    def _estimate_build_cost(self, capability: str) -> float:
        """Estimate cost to build capability internally"""
        # Simplified estimation
        capability_costs = {
            "vector_database": 2_000_000,  # 2 years, 4 engineers
            "embedding_api": 500_000,  # 6 months, 3 engineers  
            "monitoring_platform": 1_000_000,  # 1 year, 3 engineers
            "ml_platform": 3_000_000,  # 2 years, 5 engineers
        }
        return capability_costs.get(capability, 1_000_000)
    
    def _estimate_time_value(self, capabilities: List[str]) -> float:
        """Estimate value of faster time to market"""
        # Each quarter faster worth ~$200K in opportunity cost
        months_saved = len(capabilities) * 6  # 6 months per capability
        quarters_saved = months_saved / 3
        return quarters_saved * 200_000
    
    def _assess_competitive_risk(self, vendor: str, capabilities: List[str]) -> float:
        """Assess risk vendor could leverage relationship competitively"""
        # Higher risk if vendor could use learnings to compete
        if "embedding_api" in capabilities:
            return 500_000  # High risk - vendor sees our use cases
        elif "vector_database" in capabilities:
            return 200_000  # Medium risk - vendor sees our scale
        else:
            return 50_000  # Low risk - commodity infrastructure
    
    def optimize_partnership_portfolio(
        self,
        budget: float,
        strategic_priorities: Dict[str, float]
    ) -> Dict[str, any]:
        """Optimize allocation across partnership opportunities"""
        
        # Score each partnership by strategic alignment
        scored_partnerships = []
        for pid, partnership in self.partnerships.items():
            alignment_score = sum(
                strategic_priorities.get(area, 0)
                for area in partnership.collaboration_areas
            )
            
            value_score = (
                sum(partnership.benefits_realized.values()) /
                max(partnership.financial_commitment, 1)
            )
            
            risk_score = 1 - (
                partnership.dependency_level * 0.5 +
                partnership.competitive_risk * 0.5
            )
            
            overall_score = alignment_score * value_score * risk_score
            
            scored_partnerships.append({
                "partner": partnership.partner_name,
                "type": partnership.partnership_type,
                "current_investment": partnership.financial_commitment,
                "score": overall_score,
                "alignment": alignment_score,
                "value": value_score,
                "risk": risk_score
            })
        
        # Sort by score
        scored_partnerships.sort(key=lambda x: x["score"], reverse=True)
        
        # Allocate budget prioritizing highest value
        allocation = {}
        remaining_budget = budget
        
        for partnership in scored_partnerships:
            if remaining_budget >= partnership["current_investment"]:
                allocation[partnership["partner"]] = partnership["current_investment"]
                remaining_budget -= partnership["current_investment"]
            else:
                # Partial allocation to top partners
                if partnership["score"] > 0.7:
                    allocation[partnership["partner"]] = remaining_budget
                    remaining_budget = 0
                break
        
        return {
            "total_budget": budget,
            "allocated": sum(allocation.values()),
            "partnerships_funded": len(allocation),
            "allocation": allocation,
            "prioritized_list": scored_partnerships
        }
    
    def assess_open_source_strategy(
        self,
        internal_capability: float,
        community_maturity: float,
        strategic_importance: float
    ) -> str:
        """Determine appropriate open source engagement level"""
        
        if strategic_importance > 0.8:
            # Critical to business - need strong influence
            if internal_capability > 0.7:
                return "LEAD: Become maintainer/steering committee member"
            else:
                return "PARTNER: Major contributor to gain influence"
        
        elif strategic_importance > 0.5:
            # Important but not critical
            if community_maturity > 0.7:
                return "CONTRIBUTE: Active contributor with some influence"
            else:
                return "MONITOR: Watch developments, evaluate stability"
        
        else:
            # Nice to have
            if community_maturity > 0.8:
                return "USE: Consume with minimal contribution"
            else:
                return "EVALUATE: Consider alternatives or building internally"

# Example strategic partnership design
def design_partnership_strategy(
    organization_maturity: str,
    annual_partnership_budget: float,
    strategic_focus: List[str]
) -> Dict[str, any]:
    """Design comprehensive partnership strategy"""
    
    portfolio = PartnershipPortfolio()
    
    # Vendor partnership allocation (50-60% of budget)
    vendor_budget = annual_partnership_budget * 0.55
    key_vendors = {
        "vector_db_provider": {
            "cost": 300_000,
            "value": "Managed infrastructure, faster scaling",
            "priority": 0.9 if "scale" in strategic_focus else 0.6
        },
        "embedding_api": {
            "cost": 200_000,
            "value": "Pre-trained models, faster development",
            "priority": 0.8 if "speed" in strategic_focus else 0.5
        },
        "ml_platform": {
            "cost": 500_000,
            "value": "Training infrastructure, experiment management",
            "priority": 0.9 if "innovation" in strategic_focus else 0.7
        }
    }
    
    # Academic partnership allocation (15-20% of budget)
    academic_budget = annual_partnership_budget * 0.175
    academic_programs = {
        "research_sponsorship": {
            "cost": 100_000,
            "value": "Access to cutting-edge research",
            "priority": 0.8 if "innovation" in strategic_focus else 0.4
        },
        "internship_program": {
            "cost": 150_000,
            "value": "Talent pipeline and fresh perspectives",
            "priority": 0.7
        }
    }
    
    # Open source contribution (10-15% of budget)
    open_source_budget = annual_partnership_budget * 0.125
    open_source_strategy = {
        "core_infrastructure": {
            "projects": ["vector databases", "ml frameworks"],
            "engagement": "active contributor",
            "allocation": open_source_budget * 0.6
        },
        "specialized_tools": {
            "projects": ["embedding libraries", "evaluation tools"],
            "engagement": "occasional contributor",
            "allocation": open_source_budget * 0.4
        }
    }
    
    # Industry consortium participation (10-15% of budget)
    consortium_budget = annual_partnership_budget * 0.125
    consortiums = {
        "ml_standards": {
            "cost": 50_000,
            "value": "Shape industry standards",
            "priority": 0.6
        },
        "benchmark_initiatives": {
            "cost": 75_000,
            "value": "Fair competition and credibility",
            "priority": 0.7
        }
    }
    
    return {
        "total_budget": annual_partnership_budget,
        "allocation": {
            "vendor_partnerships": vendor_budget,
            "academic_programs": academic_budget,
            "open_source": open_source_budget,
            "industry_consortiums": consortium_budget
        },
        "vendor_strategy": key_vendors,
        "academic_strategy": academic_programs,
        "open_source_strategy": open_source_strategy,
        "consortium_strategy": consortiums,
        "maturity_level": organization_maturity,
        "strategic_focus": strategic_focus,
        "principles": [
            "Collaborate on infrastructure, compete on applications",
            "Maintain multi-vendor optionality",
            "Build strategic relationships, not just transactions",
            "Contribute proportionally to value received",
            "Protect proprietary advantages while embracing openness"
        ]
    }
