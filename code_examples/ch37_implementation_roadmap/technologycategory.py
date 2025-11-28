"""
Phase 1: Technology Selection and Architecture Baseline

Architecture:
1. Requirements analysis: Define constraints and priorities
2. Technology evaluation: Assess options against requirements
3. Proof of concept: Build minimal working system
4. Performance validation: Test quality, latency, cost at small scale
5. Architecture documentation: Document decisions and rationale

Technology decisions:
- Embedding model: Pre-trained vs custom, size, modality
- Vector database: Managed vs self-hosted, scale, features
- Infrastructure: Cloud provider, compute, storage
- Data pipeline: Batch vs streaming, ETL, quality
- Integration: APIs, SDKs, existing systems

Success criteria:
- Technical feasibility: System works end-to-end
- Performance targets: Meets latency/quality requirements
- Cost projections: Acceptable at target scale
- Integration viability: Fits existing architecture
- Scalability assessment: Clear path to production scale
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


class TechnologyCategory(Enum):
    """Technology categories for evaluation"""

    EMBEDDING_MODEL = "embedding_model"
    VECTOR_DATABASE = "vector_database"
    CLOUD_PLATFORM = "cloud_platform"
    DATA_PIPELINE = "data_pipeline"
    MONITORING = "monitoring"
    INTEGRATION = "integration"


class EvaluationCriteria(Enum):
    """Evaluation criteria for technology selection"""

    PERFORMANCE = "performance"  # Quality, latency, throughput
    SCALABILITY = "scalability"  # Target scale support
    COST = "cost"  # Total cost of ownership
    EASE_OF_USE = "ease_of_use"  # Development velocity
    MATURITY = "maturity"  # Production readiness
    ECOSYSTEM = "ecosystem"  # Integration, support
    VENDOR_LOCK_IN = "vendor_lock_in"  # Strategic flexibility


@dataclass
class TechnologyOption:
    """Technology option for evaluation"""

    name: str
    category: TechnologyCategory
    description: str

    # Evaluation scores (1-10)
    performance_score: float
    scalability_score: float
    cost_score: float
    ease_of_use_score: float
    maturity_score: float
    ecosystem_score: float
    lock_in_score: float  # 10 = no lock-in, 1 = high lock-in

    # Details
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    pricing_model: str = ""
    deployment_options: List[str] = field(default_factory=list)

    # Experience data
    community_size: str = ""  # small, medium, large
    documentation_quality: str = ""  # poor, good, excellent
    support_options: List[str] = field(default_factory=list)

    def overall_score(self, weights: Dict[EvaluationCriteria, float]) -> float:
        """Calculate weighted overall score"""
        score_map = {
            EvaluationCriteria.PERFORMANCE: self.performance_score,
            EvaluationCriteria.SCALABILITY: self.scalability_score,
            EvaluationCriteria.COST: self.cost_score,
            EvaluationCriteria.EASE_OF_USE: self.ease_of_use_score,
            EvaluationCriteria.MATURITY: self.maturity_score,
            EvaluationCriteria.ECOSYSTEM: self.ecosystem_score,
            EvaluationCriteria.VENDOR_LOCK_IN: self.lock_in_score,
        }

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(score_map[criterion] * weight for criterion, weight in weights.items())

        return weighted_sum / total_weight


@dataclass
class RequirementProfile:
    """Requirements profile for technology selection"""

    project_name: str
    target_scale: int  # Number of vectors
    target_qps: int  # Queries per second
    target_latency_ms: float  # Maximum acceptable latency
    budget_monthly: float  # Monthly budget in USD

    # Data characteristics
    vector_dimensionality: int
    data_modalities: List[str]  # text, image, audio, etc.
    update_frequency: str  # real-time, hourly, daily

    # Constraints
    data_sensitivity: str  # public, internal, confidential
    regulatory_requirements: List[str]  # GDPR, HIPAA, etc.
    existing_infrastructure: List[str]  # AWS, GCP, Azure, on-prem
    team_expertise: List[str]  # Python, Kubernetes, etc.

    # Priorities (weights for evaluation)
    criterion_weights: Dict[EvaluationCriteria, float] = field(default_factory=dict)

    def __post_init__(self):
        """Set default weights if not provided"""
        if not self.criterion_weights:
            # Default equal weights
            self.criterion_weights = dict.fromkeys(EvaluationCriteria, 1.0)


class TechnologyEvaluator:
    """
    Evaluate and select technologies for embedding system.

    Systematic evaluation across categories with configurable
    priorities and constraints.
    """

    def __init__(self, requirements: RequirementProfile):
        self.requirements = requirements
        self.options: Dict[TechnologyCategory, List[TechnologyOption]] = {
            category: [] for category in TechnologyCategory
        }
        self.selected: Dict[TechnologyCategory, Optional[TechnologyOption]] = dict.fromkeys(
            TechnologyCategory
        )

    def add_option(self, option: TechnologyOption) -> None:
        """Add technology option for evaluation"""
        self.options[option.category].append(option)

    def evaluate_category(
        self, category: TechnologyCategory, top_n: int = 3
    ) -> List[Tuple[TechnologyOption, float]]:
        """
        Evaluate options in category and return top N.

        Returns list of (option, score) tuples sorted by score.
        """
        if category not in self.options:
            return []

        # Score all options
        scored_options = [
            (option, option.overall_score(self.requirements.criterion_weights))
            for option in self.options[category]
        ]

        # Sort by score descending
        scored_options.sort(key=lambda x: x[1], reverse=True)

        return scored_options[:top_n]

    def select_option(self, category: TechnologyCategory, option: TechnologyOption) -> None:
        """Select specific option for category"""
        if option not in self.options[category]:
            raise ValueError(f"Option {option.name} not in category {category}")
        self.selected[category] = option

    def generate_recommendation_report(self) -> str:
        """Generate technology selection recommendation report"""
        report = []
        report.append("# Technology Selection Recommendations\n")
        report.append(f"Project: {self.requirements.project_name}\n")
        report.append(f"Generated: {datetime.now().isoformat()}\n\n")

        report.append("## Requirements Summary\n")
        report.append(f"- Target scale: {self.requirements.target_scale:,} vectors\n")
        report.append(f"- Target QPS: {self.requirements.target_qps:,}\n")
        report.append(f"- Target latency: {self.requirements.target_latency_ms}ms\n")
        report.append(f"- Monthly budget: ${self.requirements.budget_monthly:,}\n")
        report.append(f"- Vector dimension: {self.requirements.vector_dimensionality}\n")
        report.append(f"- Update frequency: {self.requirements.update_frequency}\n\n")

        report.append("## Evaluation Priorities\n")
        for criterion, weight in self.requirements.criterion_weights.items():
            if weight > 0:
                report.append(f"- {criterion.value}: {weight}\n")
        report.append("\n")

        # Recommendations by category
        for category in TechnologyCategory:
            report.append(f"## {category.value.replace('_', ' ').title()}\n\n")

            top_options = self.evaluate_category(category, top_n=3)

            if not top_options:
                report.append("No options evaluated for this category.\n\n")
                continue

            for i, (option, score) in enumerate(top_options, 1):
                report.append(f"### {i}. {option.name} (Score: {score:.2f}/10)\n\n")
                report.append(f"{option.description}\n\n")

                if option.strengths:
                    report.append("**Strengths:**\n")
                    for strength in option.strengths:
                        report.append(f"- {strength}\n")
                    report.append("\n")

                if option.weaknesses:
                    report.append("**Weaknesses:**\n")
                    for weakness in option.weaknesses:
                        report.append(f"- {weakness}\n")
                    report.append("\n")

                report.append(f"**Pricing:** {option.pricing_model}\n\n")

                if i == 1:
                    report.append("**RECOMMENDED**\n\n")

        # Selected stack
        report.append("## Recommended Technology Stack\n\n")
        for category, option in self.selected.items():
            if option:
                report.append(f"- **{category.value.replace('_', ' ').title()}:** {option.name}\n")
        report.append("\n")

        return "".join(report)


@dataclass
class ProofOfConceptPlan:
    """Plan for proof of concept implementation"""

    project_name: str
    objectives: List[str]
    success_criteria: List[str]

    # Timeline
    duration_weeks: int
    milestones: List[Dict[str, str]]  # name, week, deliverable

    # Resources
    team_members: List[Dict[str, str]]  # name, role, allocation
    compute_resources: Dict[str, str]
    data_sources: List[Dict[str, str]]

    # Scope
    data_volume: int  # Number of records
    user_count: int  # Number of test users
    use_cases: List[str]
    out_of_scope: List[str]

    # Risk mitigation
    risks: List[Dict[str, str]]  # risk, impact, mitigation
    dependencies: List[str]
    assumptions: List[str]

    def generate_plan_document(self) -> str:
        """Generate POC plan document"""
        doc = []
        doc.append(f"# Proof of Concept Plan: {self.project_name}\n\n")

        doc.append("## Objectives\n\n")
        for obj in self.objectives:
            doc.append(f"- {obj}\n")
        doc.append("\n")

        doc.append("## Success Criteria\n\n")
        for criterion in self.success_criteria:
            doc.append(f"- {criterion}\n")
        doc.append("\n")

        doc.append(f"## Timeline: {self.duration_weeks} weeks\n\n")
        for milestone in self.milestones:
            doc.append(f"### Week {milestone['week']}: {milestone['name']}\n")
            doc.append(f"{milestone['deliverable']}\n\n")

        doc.append("## Team\n\n")
        for member in self.team_members:
            doc.append(f"- **{member['name']}** ({member['role']}): {member['allocation']}\n")
        doc.append("\n")

        doc.append("## Scope\n\n")
        doc.append(f"- Data volume: {self.data_volume:,} records\n")
        doc.append(f"- Test users: {self.user_count}\n")
        doc.append("- Use cases:\n")
        for uc in self.use_cases:
            doc.append(f"  - {uc}\n")
        doc.append("\n")

        if self.out_of_scope:
            doc.append("### Out of Scope\n\n")
            for item in self.out_of_scope:
                doc.append(f"- {item}\n")
            doc.append("\n")

        doc.append("## Risks and Mitigation\n\n")
        for risk in self.risks:
            doc.append(f"- **Risk:** {risk['risk']}\n")
            doc.append(f"  - Impact: {risk['impact']}\n")
            doc.append(f"  - Mitigation: {risk['mitigation']}\n\n")

        return "".join(doc)


# Example: Technology evaluation for e-commerce search
def example_technology_evaluation():
    """Example technology evaluation workflow"""

    # Define requirements
    requirements = RequirementProfile(
        project_name="E-commerce Product Search",
        target_scale=10_000_000,  # 10M products
        target_qps=1000,
        target_latency_ms=50,
        budget_monthly=5000,
        vector_dimensionality=768,
        data_modalities=["text", "image"],
        update_frequency="hourly",
        data_sensitivity="internal",
        regulatory_requirements=["GDPR"],
        existing_infrastructure=["AWS"],
        team_expertise=["Python", "Docker", "PostgreSQL"],
        criterion_weights={
            EvaluationCriteria.PERFORMANCE: 2.0,  # High priority
            EvaluationCriteria.SCALABILITY: 2.0,  # High priority
            EvaluationCriteria.COST: 1.5,
            EvaluationCriteria.EASE_OF_USE: 1.0,
            EvaluationCriteria.MATURITY: 1.5,
            EvaluationCriteria.ECOSYSTEM: 1.0,
            EvaluationCriteria.VENDOR_LOCK_IN: 0.5,  # Less concerned
        },
    )

    evaluator = TechnologyEvaluator(requirements)

    # Add vector database options
    evaluator.add_option(
        TechnologyOption(
            name="Pinecone",
            category=TechnologyCategory.VECTOR_DATABASE,
            description="Managed vector database with excellent performance",
            performance_score=9.0,
            scalability_score=9.5,
            cost_score=7.0,
            ease_of_use_score=9.5,
            maturity_score=8.5,
            ecosystem_score=8.0,
            lock_in_score=6.0,
            strengths=[
                "Excellent query performance (<50ms p95)",
                "Fully managed, no ops overhead",
                "Great documentation and SDKs",
                "Strong metadata filtering",
            ],
            weaknesses=[
                "Higher cost at scale ($0.096/GB/month + query costs)",
                "Vendor lock-in concerns",
                "Less control over infrastructure",
            ],
            pricing_model="Storage + compute, ~$500-1000/month at 10M vectors",
            deployment_options=["Managed cloud (multi-region)"],
        )
    )

    evaluator.add_option(
        TechnologyOption(
            name="Weaviate (self-hosted)",
            category=TechnologyCategory.VECTOR_DATABASE,
            description="Open source vector database with hybrid search",
            performance_score=8.0,
            scalability_score=8.5,
            cost_score=8.5,
            ease_of_use_score=7.0,
            maturity_score=8.0,
            ecosystem_score=8.5,
            lock_in_score=9.5,
            strengths=[
                "Open source, full control",
                "Hybrid search (vector + keyword)",
                "Lower cost self-hosted (~$300/month)",
                "Strong community and documentation",
            ],
            weaknesses=[
                "Requires ops expertise (Kubernetes)",
                "More development time",
                "Responsibility for scaling and reliability",
            ],
            pricing_model="Infrastructure cost only, ~$300-500/month on AWS",
            deployment_options=["Self-hosted", "Managed cloud"],
        )
    )

    # Add embedding model options
    evaluator.add_option(
        TechnologyOption(
            name="OpenAI text-embedding-3-large",
            category=TechnologyCategory.EMBEDDING_MODEL,
            description="High-quality general-purpose embeddings",
            performance_score=9.0,
            scalability_score=8.0,
            cost_score=6.0,
            ease_of_use_score=10.0,
            maturity_score=9.0,
            ecosystem_score=9.0,
            lock_in_score=7.0,
            strengths=[
                "Excellent quality for general text",
                "Simple API, no training needed",
                "3072 dimensions, configurable",
                "Fast time to value",
            ],
            weaknesses=[
                "API costs add up ($0.13/1M tokens)",
                "Limited customization for domain",
                "API dependency and rate limits",
            ],
            pricing_model="$0.13/1M tokens, ~$1000/month for 10M products",
            deployment_options=["API only"],
        )
    )

    # Generate recommendations
    print(evaluator.generate_recommendation_report())

    # Create POC plan
    poc_plan = ProofOfConceptPlan(
        project_name="E-commerce Product Search POC",
        objectives=[
            "Validate embedding quality for product search",
            "Achieve <50ms p95 query latency",
            "Demonstrate 20%+ improvement in search quality",
            "Establish baseline architecture for scaling",
        ],
        success_criteria=[
            "Search relevance: NDCG@10 > 0.85",
            "Query latency: p95 < 50ms, p99 < 100ms",
            "User satisfaction: >80% find results relevant",
            "Cost projection: <$10K/month at full scale",
        ],
        duration_weeks=8,
        milestones=[
            {
                "name": "Infrastructure Setup",
                "week": "1-2",
                "deliverable": "Vector DB deployed, embedding pipeline running",
            },
            {
                "name": "Data Ingestion",
                "week": "3-4",
                "deliverable": "100K products indexed with embeddings",
            },
            {
                "name": "Search Implementation",
                "week": "5-6",
                "deliverable": "Search API with reranking, basic UI",
            },
            {
                "name": "Evaluation & Optimization",
                "week": "7-8",
                "deliverable": "Performance metrics, business case, go/no-go",
            },
        ],
        team_members=[
            {"name": "Alex Chen", "role": "ML Engineer", "allocation": "100%"},
            {"name": "Sam Rodriguez", "role": "Backend Engineer", "allocation": "75%"},
            {"name": "Jordan Lee", "role": "Product Manager", "allocation": "25%"},
        ],
        compute_resources={
            "Vector DB": "AWS r6g.xlarge (16GB RAM)",
            "Embedding service": "AWS Lambda + SQS",
            "Development": "AWS t3.large for testing",
        },
        data_sources=[
            {"name": "Product catalog", "records": "10M products"},
            {"name": "Search logs", "records": "Historical queries for evaluation"},
        ],
        data_volume=100_000,
        user_count=20,
        use_cases=[
            "Text search: User searches for products by description",
            "Visual search: User searches by product image",
        ],
        out_of_scope=[
            "Personalized recommendations (Phase 2)",
            "Multi-language support (Phase 2)",
            "Mobile app integration (Phase 3)",
        ],
        risks=[
            {
                "risk": "Embedding quality insufficient for domain",
                "impact": "Core value proposition fails",
                "mitigation": "Test multiple embedding models, plan for fine-tuning",
            },
            {
                "risk": "Latency targets not met",
                "impact": "User experience degraded",
                "mitigation": "Architecture optimization, caching strategy, ANN tuning",
            },
            {
                "risk": "Team lacks vector DB expertise",
                "impact": "Delayed timeline, suboptimal implementation",
                "mitigation": "Training, vendor support, community resources",
            },
        ],
        dependencies=[
            "Product catalog API access",
            "AWS account with sufficient quotas",
            "Search log data for evaluation",
        ],
        assumptions=[
            "Product data quality sufficient for embedding generation",
            "Search logs representative of user behavior",
            "Budget approved for Phase 2 if POC successful",
        ],
    )

    print("\n" + "=" * 80 + "\n")
    print(poc_plan.generate_plan_document())


if __name__ == "__main__":
    # Run example evaluation
    example_technology_evaluation()
