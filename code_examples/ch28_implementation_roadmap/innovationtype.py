# Code from Chapter 28
# Book: Embeddings at Scale

"""
Phase 4: Research Integration and Continuous Innovation

Architecture:
1. Research monitoring: Track advances in embeddings, vector search
2. Evaluation framework: Assess relevance, maturity, impact
3. Prototyping pipeline: Rapid experimentation with new techniques
4. Production integration: Harden and deploy validated innovations
5. Knowledge sharing: Document learnings, enable teams

Innovation areas:
- Model improvements: Better embeddings (quality, efficiency)
- Algorithm advances: Faster search, better compression
- Infrastructure optimization: Cost reduction, latency improvement
- New applications: Expand use cases leveraging platform
- Developer experience: Easier onboarding, better tooling

Success metrics:
- Time to production: <3 months from research to deployment
- Impact: >10% improvement in key metrics
- Adoption: >50% of applications use new capabilities
- ROI: 3-5× value from innovation investment
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional


class InnovationType(Enum):
    """Types of innovations"""
    MODEL_IMPROVEMENT = "model_improvement"
    ALGORITHM_ADVANCE = "algorithm_advance"
    INFRASTRUCTURE_OPTIMIZATION = "infrastructure_optimization"
    NEW_APPLICATION = "new_application"
    DEVELOPER_EXPERIENCE = "developer_experience"

class InnovationStage(Enum):
    """Stages of innovation pipeline"""
    RESEARCH_REVIEW = "research_review"
    PROTOTYPING = "prototyping"
    VALIDATION = "validation"
    PRODUCTION_ENGINEERING = "production_engineering"
    ROLLOUT = "rollout"
    COMPLETED = "completed"
    ABANDONED = "abandoned"

@dataclass
class Innovation:
    """Track innovation project"""
    id: str
    title: str
    description: str
    innovation_type: InnovationType

    # Evaluation
    relevance_score: float  # 1-10
    maturity_score: float  # 1-10
    expected_impact: str  # low, medium, high
    complexity: str  # low, medium, high
    risk: str  # low, medium, high

    # Execution
    stage: InnovationStage
    owner: str
    start_date: datetime
    target_completion: Optional[datetime] = None
    actual_completion: Optional[datetime] = None

    # Resources
    effort_weeks: float = 0.0
    cost_estimate: float = 0.0

    # Results
    achieved_impact: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)

    # Related
    research_papers: List[str] = field(default_factory=list)
    prototypes: List[str] = field(default_factory=list)

    def advance_stage(self, new_stage: InnovationStage) -> None:
        """Advance innovation to next stage"""
        self.stage = new_stage
        if new_stage == InnovationStage.COMPLETED:
            self.actual_completion = datetime.now()

class InnovationPipeline:
    """
    Manage research integration and continuous innovation.
    
    Track innovations from research review through production
    deployment, measure impact, and share learnings.
    """

    def __init__(self, platform_name: str):
        self.platform_name = platform_name
        self.innovations: Dict[str, Innovation] = {}

    def add_innovation(self, innovation: Innovation) -> None:
        """Add new innovation to pipeline"""
        if innovation.id in self.innovations:
            raise ValueError(f"Innovation {innovation.id} already exists")
        self.innovations[innovation.id] = innovation
        print(f"Added innovation: {innovation.title}")

    def update_stage(self, innovation_id: str, new_stage: InnovationStage) -> None:
        """Update innovation stage"""
        if innovation_id not in self.innovations:
            raise ValueError(f"Innovation {innovation_id} not found")

        innovation = self.innovations[innovation_id]
        old_stage = innovation.stage
        innovation.advance_stage(new_stage)

        print(f"Innovation '{innovation.title}' advanced:")
        print(f"  {old_stage.value} → {new_stage.value}")

    def record_impact(
        self,
        innovation_id: str,
        achieved_impact: str,
        lessons: List[str]
    ) -> None:
        """Record innovation impact and learnings"""
        if innovation_id not in self.innovations:
            raise ValueError(f"Innovation {innovation_id} not found")

        innovation = self.innovations[innovation_id]
        innovation.achieved_impact = achieved_impact
        innovation.lessons_learned = lessons

        print(f"Recorded impact for '{innovation.title}':")
        print(f"  Expected: {innovation.expected_impact}")
        print(f"  Achieved: {achieved_impact}")

    def get_active_innovations(self) -> List[Innovation]:
        """Get all active innovations"""
        return [
            inn for inn in self.innovations.values()
            if inn.stage not in [InnovationStage.COMPLETED, InnovationStage.ABANDONED]
        ]

    def get_innovations_by_stage(self, stage: InnovationStage) -> List[Innovation]:
        """Get innovations at specific stage"""
        return [
            inn for inn in self.innovations.values()
            if inn.stage == stage
        ]

    def calculate_roi(self) -> Dict[str, any]:
        """Calculate ROI of innovation program"""
        completed = [
            inn for inn in self.innovations.values()
            if inn.stage == InnovationStage.COMPLETED
        ]

        if not completed:
            return {"roi": 0, "details": "No completed innovations"}

        total_investment = sum(inn.cost_estimate for inn in completed)

        # Simplified value calculation
        # In production: Measure actual business impact
        impact_value = {
            "high": 10.0,  # 10× value
            "medium": 3.0,  # 3× value
            "low": 1.0  # 1× value
        }

        total_value = sum(
            inn.cost_estimate * impact_value.get(inn.achieved_impact or "low", 1.0)
            for inn in completed
        )

        roi = (total_value - total_investment) / total_investment if total_investment > 0 else 0

        return {
            "roi": roi,
            "investment": total_investment,
            "value": total_value,
            "completed_count": len(completed),
            "high_impact": sum(1 for inn in completed if inn.achieved_impact == "high"),
            "medium_impact": sum(1 for inn in completed if inn.achieved_impact == "medium"),
            "low_impact": sum(1 for inn in completed if inn.achieved_impact == "low")
        }

    def generate_innovation_report(self) -> str:
        """Generate innovation pipeline report"""
        report = []
        report.append(f"# Innovation Pipeline Report: {self.platform_name}\n\n")
        report.append(f"Generated: {datetime.now().isoformat()}\n\n")

        # Overview
        active = self.get_active_innovations()
        completed = self.get_innovations_by_stage(InnovationStage.COMPLETED)

        report.append("## Pipeline Overview\n\n")
        report.append(f"- Total innovations: {len(self.innovations)}\n")
        report.append(f"- Active: {len(active)}\n")
        report.append(f"- Completed: {len(completed)}\n\n")

        # By stage
        report.append("## Innovations by Stage\n\n")
        for stage in InnovationStage:
            if stage in [InnovationStage.COMPLETED, InnovationStage.ABANDONED]:
                continue
            innovations = self.get_innovations_by_stage(stage)
            report.append(f"### {stage.value.replace('_', ' ').title()} ({len(innovations)})\n\n")
            for inn in innovations:
                report.append(f"- **{inn.title}** ({inn.innovation_type.value})\n")
                report.append(f"  - Owner: {inn.owner}\n")
                report.append(f"  - Expected impact: {inn.expected_impact}\n")
                report.append(f"  - Effort: {inn.effort_weeks} weeks\n\n")

        # Completed innovations
        if completed:
            report.append("## Completed Innovations\n\n")
            for inn in completed:
                duration = (inn.actual_completion - inn.start_date).days if inn.actual_completion else 0
                report.append(f"### {inn.title}\n\n")
                report.append(f"- Type: {inn.innovation_type.value}\n")
                report.append(f"- Duration: {duration} days\n")
                report.append(f"- Expected impact: {inn.expected_impact}\n")
                report.append(f"- Achieved impact: {inn.achieved_impact}\n")
                if inn.lessons_learned:
                    report.append("- Lessons learned:\n")
                    for lesson in inn.lessons_learned:
                        report.append(f"  - {lesson}\n")
                report.append("\n")

        # ROI
        roi_metrics = self.calculate_roi()
        report.append("## Innovation ROI\n\n")
        report.append(f"- Total ROI: {roi_metrics['roi']:.1f}×\n")
        report.append(f"- Investment: ${roi_metrics['investment']:,.0f}\n")
        report.append(f"- Value delivered: ${roi_metrics['value']:,.0f}\n")
        report.append(f"- Completed projects: {roi_metrics['completed_count']}\n")
        report.append(f"- High impact: {roi_metrics.get('high_impact', 0)}\n")
        report.append(f"- Medium impact: {roi_metrics.get('medium_impact', 0)}\n")
        report.append(f"- Low impact: {roi_metrics.get('low_impact', 0)}\n\n")

        return "".join(report)


# Example: Innovation pipeline
def example_innovation_pipeline():
    """Example innovation pipeline management"""

    pipeline = InnovationPipeline("Enterprise Embedding Platform")

    # Add innovations
    pipeline.add_innovation(Innovation(
        id="inn-001",
        title="Binary Quantization for 4× Storage Reduction",
        description="Implement binary quantization reducing vector storage from 768×4 bytes to 768 bits",
        innovation_type=InnovationType.INFRASTRUCTURE_OPTIMIZATION,
        relevance_score=9.0,
        maturity_score=8.0,
        expected_impact="high",
        complexity="medium",
        risk="low",
        stage=InnovationStage.COMPLETED,
        owner="Alex Chen",
        start_date=datetime.now() - timedelta(days=120),
        target_completion=datetime.now() - timedelta(days=30),
        actual_completion=datetime.now() - timedelta(days=25),
        effort_weeks=12,
        cost_estimate=120000,
        achieved_impact="high",
        lessons_learned=[
            "Binary quantization works well for semantic search with <5% quality degradation",
            "Requires careful tuning of threshold for binarization",
            "Storage savings enable 4× scale increase within same budget"
        ],
        research_papers=["https://arxiv.org/abs/2106.09685"]
    ))

    pipeline.add_innovation(Innovation(
        id="inn-002",
        title="Multi-Vector Product Embeddings",
        description="Generate multiple embeddings per product (title, description, images) for better retrieval",
        innovation_type=InnovationType.MODEL_IMPROVEMENT,
        relevance_score=8.0,
        maturity_score=6.0,
        expected_impact="medium",
        complexity="high",
        risk="medium",
        stage=InnovationStage.VALIDATION,
        owner="Jordan Lee",
        start_date=datetime.now() - timedelta(days=60),
        target_completion=datetime.now() + timedelta(days=30),
        effort_weeks=16,
        cost_estimate=150000,
        research_papers=["https://arxiv.org/abs/2112.07768"]
    ))

    pipeline.add_innovation(Innovation(
        id="inn-003",
        title="Real-time Embedding Updates",
        description="Stream processing pipeline for <1 minute embedding freshness",
        innovation_type=InnovationType.INFRASTRUCTURE_OPTIMIZATION,
        relevance_score=7.0,
        maturity_score=7.0,
        expected_impact="medium",
        complexity="high",
        risk="medium",
        stage=InnovationStage.PROTOTYPING,
        owner="Sam Rodriguez",
        start_date=datetime.now() - timedelta(days=30),
        target_completion=datetime.now() + timedelta(days=60),
        effort_weeks=12,
        cost_estimate=100000
    ))

    # Generate report
    print(pipeline.generate_innovation_report())


if __name__ == "__main__":
    example_innovation_pipeline()
