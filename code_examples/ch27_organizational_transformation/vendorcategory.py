# Code from Chapter 27
# Book: Embeddings at Scale

"""
Vendor Evaluation Framework

Architecture:
1. Requirements definition: Technical, operational, business needs
2. Vendor discovery: Identify candidate vendors
3. Capability assessment: Evaluate against requirements
4. Cost modeling: Total cost of ownership analysis
5. Risk evaluation: Technical, business, strategic risks
6. POC/pilot: Hands-on validation with real workloads
7. Negotiation: Pricing, terms, SLAs, exit clauses
8. Decision: Weighted scoring across criteria

Evaluation dimensions:
- Technical capabilities: Features, performance, scale, integrations
- Operational maturity: Reliability, support, documentation, community
- Business factors: Pricing, contract terms, vendor stability
- Strategic fit: Roadmap alignment, partnership potential, lock-in risk

Vendor types:
- Vector databases: Pinecone, Weaviate, Milvus, Qdrant
- Embedding models: OpenAI, Cohere, Anthropic, open source
- MLOps platforms: Databricks, AWS SageMaker, Google Vertex
- Observability: Datadog, New Relic, Grafana, custom
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
from datetime import datetime

class VendorCategory(Enum):
    """Vendor categories"""
    VECTOR_DATABASE = "vector_database"
    EMBEDDING_MODEL = "embedding_model"
    MLOPS_PLATFORM = "mlops_platform"
    SERVING_INFRASTRUCTURE = "serving_infrastructure"
    MONITORING = "monitoring"
    DATA_PIPELINE = "data_pipeline"

class RequirementPriority(Enum):
    """Requirement priorities"""
    MUST_HAVE = "must_have"  # Non-negotiable
    IMPORTANT = "important"  # Strong preference
    NICE_TO_HAVE = "nice_to_have"  # Bonus but not required

class RiskLevel(Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Requirement:
    """
    Vendor requirement
    
    Attributes:
        name: Requirement identifier
        category: Technical, operational, or business
        priority: How critical this requirement is
        description: Detailed requirement description
        evaluation_criteria: How to assess if vendor meets this
        weight: Scoring weight (1-10)
    """
    name: str
    category: str
    priority: RequirementPriority
    description: str
    evaluation_criteria: str
    weight: int

@dataclass
class Vendor:
    """
    Vendor profile
    
    Attributes:
        name: Vendor name
        category: Primary vendor category
        description: What vendor provides
        founded_year: When vendor was founded
        funding_raised: Total funding (for stability assessment)
        customer_count: Approximate customer base
        key_customers: Notable reference customers
        pricing_model: How vendor charges
        strengths: Key advantages
        weaknesses: Known limitations
    """
    name: str
    category: VendorCategory
    description: str
    founded_year: int
    funding_raised: Optional[float] = None  # Millions USD
    customer_count: Optional[int] = None
    key_customers: List[str] = field(default_factory=list)
    pricing_model: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)

@dataclass
class VendorScore:
    """
    Vendor evaluation score
    
    Attributes:
        vendor_name: Vendor being evaluated
        requirement_scores: Scores for each requirement (0-10)
        weighted_score: Overall weighted score
        total_cost_5yr: 5-year TCO estimate
        risks: Identified risks
        recommendation: Buy/pass/pilot recommendation
        reasoning: Explanation of recommendation
    """
    vendor_name: str
    requirement_scores: Dict[str, float]
    weighted_score: float
    total_cost_5yr: float
    risks: List[Dict[str, any]]
    recommendation: str
    reasoning: str

@dataclass
class BuildVsBuyAnalysis:
    """
    Analysis for build vs buy decision
    
    Attributes:
        component: What component is being considered
        build_cost_5yr: 5-year cost to build and maintain
        build_timeline_months: Time to production-ready
        buy_cost_5yr: 5-year cost to buy from vendor
        buy_timeline_months: Time to production with vendor
        strategic_value: Strategic importance (1-10)
        team_capability: Internal capability to build (1-10)
        recommendation: Build/buy/hybrid
    """
    component: str
    build_cost_5yr: float
    build_timeline_months: int
    buy_cost_5yr: float
    buy_timeline_months: int
    strategic_value: int
    team_capability: int
    recommendation: str
    reasoning: str

class VendorEvaluationFramework:
    """
    Framework for evaluating vendors and build-vs-buy decisions
    
    Manages requirements, vendor scoring, cost analysis, and recommendations
    """
    
    def __init__(self):
        self.requirements: Dict[str, Requirement] = {}
        self.vendors: Dict[str, Vendor] = {}
        
    def add_requirement(self, requirement: Requirement):
        """Add evaluation requirement"""
        self.requirements[requirement.name] = requirement
        
    def add_vendor(self, vendor: Vendor):
        """Add vendor to evaluation"""
        self.vendors[vendor.name] = vendor
        
    def score_vendor(
        self,
        vendor_name: str,
        requirement_scores: Dict[str, float],
        cost_model: Dict[str, float]
    ) -> VendorScore:
        """
        Score vendor against requirements
        
        Args:
            vendor_name: Vendor to score
            requirement_scores: Scores for each requirement (0-10)
            cost_model: Cost breakdown (setup, annual, per_query, etc.)
            
        Returns:
            Vendor evaluation score
        """
        vendor = self.vendors[vendor_name]
        
        # Calculate weighted score
        total_weight = 0
        weighted_sum = 0
        
        for req_name, score in requirement_scores.items():
            req = self.requirements[req_name]
            
            # Apply priority multiplier
            priority_multiplier = {
                RequirementPriority.MUST_HAVE: 3.0,
                RequirementPriority.IMPORTANT: 2.0,
                RequirementPriority.NICE_TO_HAVE: 1.0
            }[req.priority]
            
            weight = req.weight * priority_multiplier
            weighted_sum += score * weight
            total_weight += weight
        
        weighted_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Calculate 5-year TCO
        setup_cost = cost_model.get('setup', 0)
        annual_cost = cost_model.get('annual', 0)
        total_cost_5yr = setup_cost + (annual_cost * 5)
        
        # Identify risks
        risks = self._identify_vendor_risks(vendor, requirement_scores, cost_model)
        
        # Generate recommendation
        recommendation, reasoning = self._generate_recommendation(
            weighted_score,
            total_cost_5yr,
            risks
        )
        
        return VendorScore(
            vendor_name=vendor_name,
            requirement_scores=requirement_scores,
            weighted_score=weighted_score,
            total_cost_5yr=total_cost_5yr,
            risks=risks,
            recommendation=recommendation,
            reasoning=reasoning
        )
    
    def _identify_vendor_risks(
        self,
        vendor: Vendor,
        requirement_scores: Dict[str, float],
        cost_model: Dict[str, float]
    ) -> List[Dict[str, any]]:
        """Identify vendor-specific risks"""
        
        risks = []
        
        # Vendor stability risk
        if vendor.founded_year and vendor.founded_year > datetime.now().year - 3:
            risks.append({
                'type': 'vendor_stability',
                'level': RiskLevel.MEDIUM,
                'description': f'Young company (founded {vendor.founded_year}), potential acquisition or shutdown risk',
                'mitigation': 'Negotiate data portability, maintain exit strategy'
            })
        
        # Funding risk
        if vendor.funding_raised and vendor.funding_raised < 10:
            risks.append({
                'type': 'funding',
                'level': RiskLevel.MEDIUM,
                'description': 'Limited funding may impact long-term viability',
                'mitigation': 'Monitor financial health, diversify vendors'
            })
        
        # Must-have requirement gaps
        for req_name, score in requirement_scores.items():
            req = self.requirements[req_name]
            if req.priority == RequirementPriority.MUST_HAVE and score < 7:
                risks.append({
                    'type': 'capability_gap',
                    'level': RiskLevel.HIGH,
                    'description': f'Does not fully meet must-have requirement: {req_name}',
                    'mitigation': f'Negotiate roadmap commitment or find alternative'
                })
        
        # Cost risk
        variable_cost = cost_model.get('per_query', 0)
        if variable_cost > 0:
            risks.append({
                'type': 'cost_scaling',
                'level': RiskLevel.MEDIUM,
                'description': 'Variable pricing creates cost unpredictability at scale',
                'mitigation': 'Negotiate volume discounts, implement cost controls'
            })
        
        # Lock-in risk
        if any('proprietary' in w.lower() or 'closed' in w.lower() for w in vendor.weaknesses):
            risks.append({
                'type': 'vendor_lock_in',
                'level': RiskLevel.HIGH,
                'description': 'Proprietary technology increases switching costs',
                'mitigation': 'Use open standards where possible, maintain abstraction layer'
            })
        
        return risks
    
    def _generate_recommendation(
        self,
        weighted_score: float,
        total_cost_5yr: float,
        risks: List[Dict[str, any]]
    ) -> tuple[str, str]:
        """Generate buy/pass/pilot recommendation"""
        
        # Count high/critical risks
        high_risks = len([
            r for r in risks 
            if r['level'] in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ])
        
        # Decision logic
        if weighted_score >= 8.0 and high_risks == 0:
            return (
                "BUY",
                f"Strong match (score: {weighted_score:.1f}/10) with manageable risks. Proceed to negotiation."
            )
        elif weighted_score >= 7.0 and high_risks <= 1:
            return (
                "PILOT",
                f"Good match (score: {weighted_score:.1f}/10) but some concerns. Run pilot to validate."
            )
        elif weighted_score >= 6.0:
            return (
                "PILOT",
                f"Moderate match (score: {weighted_score:.1f}/10). Pilot alongside alternatives."
            )
        else:
            return (
                "PASS",
                f"Insufficient match (score: {weighted_score:.1f}/10). Look for better alternatives."
            )
    
    def analyze_build_vs_buy(
        self,
        component: str,
        build_estimate: Dict[str, any],
        buy_estimate: Dict[str, any],
        strategic_factors: Dict[str, int]
    ) -> BuildVsBuyAnalysis:
        """
        Analyze whether to build or buy component
        
        Args:
            component: Component name
            build_estimate: Build cost and timeline
            buy_estimate: Buy cost and timeline
            strategic_factors: Strategic importance, team capability
            
        Returns:
            Build vs buy recommendation
        """
        build_cost_5yr = (
            build_estimate['dev_cost'] +
            build_estimate['annual_maintenance'] * 5
        )
        
        buy_cost_5yr = (
            buy_estimate['setup_cost'] +
            buy_estimate['annual_cost'] * 5
        )
        
        strategic_value = strategic_factors.get('strategic_value', 5)
        team_capability = strategic_factors.get('team_capability', 5)
        
        # Decision logic
        cost_ratio = buy_cost_5yr / build_cost_5yr if build_cost_5yr > 0 else float('inf')
        time_advantage_buy = build_estimate['timeline_months'] - buy_estimate['timeline_months']
        
        # Scoring
        build_score = (
            team_capability * 2 +  # Can we build it?
            strategic_value * 3 +  # Should we build it?
            (10 if cost_ratio > 1.5 else 0)  # Cost advantage?
        )
        
        buy_score = (
            (10 - team_capability) * 2 +  # Lacking capability?
            (10 - strategic_value) * 2 +  # Not strategic?
            (min(time_advantage_buy, 12) * 0.8) +  # Time advantage?
            (10 if cost_ratio < 0.7 else 0)  # Cost advantage?
        )
        
        if build_score > buy_score + 10:
            recommendation = "BUILD"
            reasoning = f"Strong internal capability (score: {build_score:.0f} vs {buy_score:.0f})"
            if strategic_value >= 8:
                reasoning += ", strategically important"
            if cost_ratio > 1.5:
                reasoning += f", significant cost advantage ({cost_ratio:.1f}x cheaper)"
        elif buy_score > build_score + 10:
            recommendation = "BUY"
            reasoning = f"Better to buy (score: {buy_score:.0f} vs {build_score:.0f})"
            if time_advantage_buy >= 6:
                reasoning += f", much faster ({time_advantage_buy} months faster)"
            if cost_ratio < 0.7:
                reasoning += f", cost effective ({cost_ratio:.1f}x of build cost)"
        else:
            recommendation = "HYBRID"
            reasoning = f"Close call (build: {build_score:.0f}, buy: {buy_score:.0f}). "
            reasoning += "Consider hybrid approach: buy infrastructure, build customization"
        
        return BuildVsBuyAnalysis(
            component=component,
            build_cost_5yr=build_cost_5yr,
            build_timeline_months=build_estimate['timeline_months'],
            buy_cost_5yr=buy_cost_5yr,
            buy_timeline_months=buy_estimate['timeline_months'],
            strategic_value=strategic_value,
            team_capability=team_capability,
            recommendation=recommendation,
            reasoning=reasoning
        )
    
    def create_vendor_comparison(
        self,
        vendor_scores: List[VendorScore]
    ) -> str:
        """Create vendor comparison matrix"""
        
        comparison = "# Vendor Comparison\n\n"
        
        # Overall ranking
        comparison += "## Overall Ranking\n\n"
        sorted_vendors = sorted(
            vendor_scores,
            key=lambda v: v.weighted_score,
            reverse=True
        )
        
        comparison += "| Rank | Vendor | Score | 5Y TCO | Recommendation |\n"
        comparison += "|------|--------|-------|--------|----------------|\n"
        
        for i, vendor_score in enumerate(sorted_vendors, 1):
            comparison += f"| {i} | {vendor_score.vendor_name} | "
            comparison += f"{vendor_score.weighted_score:.1f}/10 | "
            comparison += f"${vendor_score.total_cost_5yr:,.0f} | "
            comparison += f"{vendor_score.recommendation} |\n"
        
        # Detailed scores
        comparison += "\n## Detailed Requirement Scores\n\n"
        
        # Get all requirements
        req_names = list(self.requirements.keys())
        
        comparison += "| Requirement | "
        comparison += " | ".join(v.vendor_name for v in sorted_vendors) + " |\n"
        comparison += "|" + "|".join(["---"] * (len(sorted_vendors) + 1)) + "|\n"
        
        for req_name in req_names:
            comparison += f"| {req_name} | "
            scores = [
                f"{v.requirement_scores.get(req_name, 0):.1f}" 
                for v in sorted_vendors
            ]
            comparison += " | ".join(scores) + " |\n"
        
        # Risk summary
        comparison += "\n## Risk Summary\n\n"
        
        for vendor_score in sorted_vendors:
            comparison += f"### {vendor_score.vendor_name}\n\n"
            
            if not vendor_score.risks:
                comparison += "*No significant risks identified*\n\n"
            else:
                high_risks = [r for r in vendor_score.risks if r['level'] == RiskLevel.HIGH]
                medium_risks = [r for r in vendor_score.risks if r['level'] == RiskLevel.MEDIUM]
                
                if high_risks:
                    comparison += "**High Risks:**\n"
                    for risk in high_risks:
                        comparison += f"- {risk['description']}\n"
                        comparison += f"  - Mitigation: {risk['mitigation']}\n"
                    comparison += "\n"
                
                if medium_risks:
                    comparison += "**Medium Risks:**\n"
                    for risk in medium_risks:
                        comparison += f"- {risk['description']}\n"
                    comparison += "\n"
        
        return comparison


# Example: Evaluate vector database vendors
def evaluate_vector_database_vendors():
    """
    Example: Evaluate vector database vendors
    """
    
    framework = VendorEvaluationFramework()
    
    # Define requirements
    requirements = [
        Requirement(
            name="Scale to 10B+ vectors",
            category="technical",
            priority=RequirementPriority.MUST_HAVE,
            description="Support at least 10 billion vectors with acceptable performance",
            evaluation_criteria="Documented large deployments, benchmark results",
            weight=10
        ),
        Requirement(
            name="Sub-50ms p99 latency",
            category="technical",
            priority=RequirementPriority.MUST_HAVE,
            description="p99 query latency under 50ms at scale",
            evaluation_criteria="Load testing, customer references",
            weight=9
        ),
        Requirement(
            name="High availability (99.9%+)",
            category="operational",
            priority=RequirementPriority.IMPORTANT,
            description="Production-grade reliability with replication",
            evaluation_criteria="SLA, uptime history, architecture",
            weight=8
        ),
        Requirement(
            name="Managed service option",
            category="operational",
            priority=RequirementPriority.IMPORTANT,
            description="Fully managed cloud deployment available",
            evaluation_criteria="Service offering, pricing",
            weight=7
        ),
        Requirement(
            name="Open source / portable",
            category="strategic",
            priority=RequirementPriority.NICE_TO_HAVE,
            description="Open source or standard APIs to avoid lock-in",
            evaluation_criteria="License, export capabilities",
            weight=6
        )
    ]
    
    for req in requirements:
        framework.add_requirement(req)
    
    # Add vendors
    vendors = [
        Vendor(
            name="Pinecone",
            category=VendorCategory.VECTOR_DATABASE,
            description="Managed vector database service",
            founded_year=2019,
            funding_raised=138.0,
            customer_count=1000,
            key_customers=["Shopify", "Gong", "Hubspot"],
            pricing_model="Usage-based (per query, per GB storage)",
            strengths=[
                "Fully managed, zero ops",
                "Excellent performance at scale",
                "Strong support and documentation"
            ],
            weaknesses=[
                "Proprietary / closed source",
                "Pricing can be expensive at scale",
                "Less control than self-hosted"
            ]
        ),
        Vendor(
            name="Weaviate",
            category=VendorCategory.VECTOR_DATABASE,
            description="Open source vector database",
            founded_year=2019,
            funding_raised=67.0,
            customer_count=500,
            key_customers=["Instabase", "Hugging Face"],
            pricing_model="Open source free, managed service available",
            strengths=[
                "Open source / portable",
                "Rich feature set (hybrid search, ML integration)",
                "Active community"
            ],
            weaknesses=[
                "Self-hosting complexity",
                "Smaller reference customer base",
                "Some performance limitations"
            ]
        ),
        Vendor(
            name="Milvus",
            category=VendorCategory.VECTOR_DATABASE,
            description="Open source vector database (CNCF)",
            founded_year=2019,
            funding_raised=43.0,
            customer_count=1000,
            key_customers=["Nvidia", "Walmart", "IKEA"],
            pricing_model="Open source free, Zilliz cloud managed service",
            strengths=[
                "CNCF project / open source",
                "Proven at extreme scale",
                "Strong GPU acceleration"
            ],
            weaknesses=[
                "Complex setup and tuning",
                "Documentation can be challenging",
                "Managed service less mature"
            ]
        )
    ]
    
    for vendor in vendors:
        framework.add_vendor(vendor)
    
    # Score vendors
    vendor_scores = []
    
    # Pinecone scores
    pinecone_scores = {
        "Scale to 10B+ vectors": 9.0,
        "Sub-50ms p99 latency": 9.0,
        "High availability (99.9%+)": 9.5,
        "Managed service option": 10.0,
        "Open source / portable": 2.0
    }
    pinecone_costs = {
        'setup': 0,
        'annual': 120000  # Example: $10k/month managed
    }
    vendor_scores.append(
        framework.score_vendor("Pinecone", pinecone_scores, pinecone_costs)
    )
    
    # Weaviate scores
    weaviate_scores = {
        "Scale to 10B+ vectors": 7.0,
        "Sub-50ms p99 latency": 7.5,
        "High availability (99.9%+)": 7.0,
        "Managed service option": 8.0,
        "Open source / portable": 10.0
    }
    weaviate_costs = {
        'setup': 50000,  # Implementation effort
        'annual': 80000  # Self-hosted infrastructure
    }
    vendor_scores.append(
        framework.score_vendor("Weaviate", weaviate_scores, weaviate_costs)
    )
    
    # Milvus scores
    milvus_scores = {
        "Scale to 10B+ vectors": 9.0,
        "Sub-50ms p99 latency": 8.0,
        "High availability (99.9%+)": 7.5,
        "Managed service option": 7.0,
        "Open source / portable": 10.0
    }
    milvus_costs = {
        'setup': 75000,  # More complex setup
        'annual': 100000  # Infrastructure + maintenance
    }
    vendor_scores.append(
        framework.score_vendor("Milvus", milvus_scores, milvus_costs)
    )
    
    # Display comparison
    print(framework.create_vendor_comparison(vendor_scores))
    
    # Build vs buy analysis for custom solution
    print("\n" + "="*60)
    print("\n=== Build vs Buy Analysis: Vector Database ===\n")
    
    analysis = framework.analyze_build_vs_buy(
        component="Vector Database",
        build_estimate={
            'dev_cost': 500000,  # 2 engineers x 12 months
            'annual_maintenance': 300000,  # 1.5 engineers ongoing
            'timeline_months': 12
        },
        buy_estimate={
            'setup_cost': 50000,  # Integration and setup
            'annual_cost': 120000,  # Managed service
            'timeline_months': 2
        },
        strategic_factors={
            'strategic_value': 4,  # Not core differentiation
            'team_capability': 6  # Some expertise but not specialized
        }
    )
    
    print(f"Component: {analysis.component}")
    print(f"Recommendation: {analysis.recommendation}")
    print(f"Reasoning: {analysis.reasoning}")
    print(f"\nBuild option:")
    print(f"  5-year cost: ${analysis.build_cost_5yr:,.0f}")
    print(f"  Time to production: {analysis.build_timeline_months} months")
    print(f"\nBuy option:")
    print(f"  5-year cost: ${analysis.buy_cost_5yr:,.0f}")
    print(f"  Time to production: {analysis.buy_timeline_months} months")

if __name__ == "__main__":
    evaluate_vector_database_vendors()
