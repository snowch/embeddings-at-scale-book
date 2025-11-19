# Code from Chapter 27
# Book: Embeddings at Scale

"""
Embedding-Native Team Structure and Capability Assessment

Architecture:
1. Team composition analysis: Map required capabilities to current team
2. Gap identification: Determine critical missing skills
3. Hiring vs training: Decide whether to recruit or develop
4. Cross-functional integration: Connect embedding team to organization
5. Growth planning: Scale team capabilities with system maturity

Team roles:
- Embedding ML Engineer: Model development, training, evaluation
- Vector Infrastructure Engineer: Database, indexing, serving
- Data Platform Engineer: Pipelines, quality, integration
- Domain Expert: Application design, metrics, validation
- Product Manager: Strategy, prioritization, adoption

Capability requirements:
- ML foundations: Deep learning, optimization, evaluation
- Embedding expertise: Contrastive learning, similarity, dimensionality
- Infrastructure: Distributed systems, databases, caching
- Data engineering: ETL, streaming, quality, governance
- Domain knowledge: Business problems, data semantics
- Product skills: User research, prioritization, adoption
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class Capability(Enum):
    """Team capability categories"""
    ML_FOUNDATIONS = "ml_foundations"
    EMBEDDING_EXPERTISE = "embedding_expertise"
    INFRASTRUCTURE = "infrastructure"
    DATA_ENGINEERING = "data_engineering"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    PRODUCT_MANAGEMENT = "product_management"
    RESEARCH = "research"
    OPERATIONS = "operations"

class ProficiencyLevel(Enum):
    """Proficiency levels for capabilities"""
    NOVICE = 1  # Learning, requires supervision
    COMPETENT = 2  # Can work independently on routine tasks
    PROFICIENT = 3  # Can handle complex problems, mentor others
    EXPERT = 4  # Domain authority, can architect systems

@dataclass
class TeamMember:
    """
    Individual team member profile
    
    Attributes:
        name: Team member identifier
        role: Primary role
        capabilities: Map of capabilities to proficiency levels
        experience_years: Years of relevant experience
        capacity: Available capacity (fraction of time)
        growth_trajectory: Capabilities being developed
    """
    name: str
    role: str
    capabilities: Dict[Capability, ProficiencyLevel]
    experience_years: float
    capacity: float = 1.0
    growth_trajectory: Set[Capability] = field(default_factory=set)

    def proficiency(self, capability: Capability) -> Optional[ProficiencyLevel]:
        """Get proficiency level for capability"""
        return self.capabilities.get(capability)

    def is_developing(self, capability: Capability) -> bool:
        """Check if actively developing capability"""
        return capability in self.growth_trajectory

@dataclass
class CapabilityRequirement:
    """
    Required capability for project success
    
    Attributes:
        capability: Capability type
        min_proficiency: Minimum required proficiency
        headcount: Number of people needed at this level
        criticality: How critical this capability is (1-10)
        current_gaps: Number of additional people needed
    """
    capability: Capability
    min_proficiency: ProficiencyLevel
    headcount: int
    criticality: int
    current_gaps: int = 0

class TeamCapabilityAssessment:
    """
    Assess team capabilities against requirements
    
    Analyzes current team composition, identifies gaps,
    and recommends hiring/training strategies
    """

    def __init__(self):
        self.team_members: List[TeamMember] = []
        self.requirements: List[CapabilityRequirement] = []

    def add_team_member(self, member: TeamMember):
        """Add team member"""
        self.team_members.append(member)

    def add_requirement(self, requirement: CapabilityRequirement):
        """Add capability requirement"""
        self.requirements.append(requirement)

    def assess_capability(
        self,
        capability: Capability,
        min_proficiency: ProficiencyLevel
    ) -> Dict[str, any]:
        """
        Assess team capability coverage
        
        Args:
            capability: Capability to assess
            min_proficiency: Minimum required proficiency
            
        Returns:
            Assessment results with gaps and recommendations
        """
        # Count team members meeting proficiency requirement
        qualified_members = [
            member for member in self.team_members
            if member.proficiency(capability)
            and member.proficiency(capability).value >= min_proficiency.value
        ]

        # Count members actively developing this capability
        developing_members = [
            member for member in self.team_members
            if member.is_developing(capability)
        ]

        # Calculate effective capacity
        effective_capacity = sum(
            member.capacity for member in qualified_members
        )

        return {
            'capability': capability.value,
            'min_proficiency': min_proficiency.name,
            'qualified_count': len(qualified_members),
            'qualified_members': [m.name for m in qualified_members],
            'effective_capacity': effective_capacity,
            'developing_count': len(developing_members),
            'developing_members': [m.name for m in developing_members],
            'has_expert': any(
                m.proficiency(capability) == ProficiencyLevel.EXPERT
                for m in qualified_members
            )
        }

    def identify_gaps(self) -> List[Dict[str, any]]:
        """
        Identify capability gaps across all requirements
        
        Returns:
            List of gaps with severity and recommendations
        """
        gaps = []

        for req in self.requirements:
            assessment = self.assess_capability(
                req.capability,
                req.min_proficiency
            )

            capacity_gap = req.headcount - assessment['effective_capacity']

            if capacity_gap > 0:
                gap = {
                    'capability': req.capability.value,
                    'required_proficiency': req.min_proficiency.name,
                    'required_headcount': req.headcount,
                    'current_capacity': assessment['effective_capacity'],
                    'capacity_gap': capacity_gap,
                    'criticality': req.criticality,
                    'severity': capacity_gap * req.criticality,  # Combined metric
                    'has_expert': assessment['has_expert'],
                    'developing_count': assessment['developing_count'],
                    'recommendation': self._generate_recommendation(
                        capacity_gap,
                        req.criticality,
                        assessment['has_expert'],
                        assessment['developing_count']
                    )
                }
                gaps.append(gap)

        # Sort by severity (most critical first)
        gaps.sort(key=lambda x: x['severity'], reverse=True)

        return gaps

    def _generate_recommendation(
        self,
        capacity_gap: float,
        criticality: int,
        has_expert: bool,
        developing_count: int
    ) -> Dict[str, any]:
        """Generate hiring/training recommendation"""

        # Critical gap with no expert: urgent external hire needed
        if criticality >= 8 and not has_expert and capacity_gap >= 0.5:
            return {
                'action': 'urgent_hire',
                'priority': 'P0',
                'description': 'Critical capability gap requires immediate external hire',
                'timeline': '1-2 months',
                'alternatives': []
            }

        # Large gap but has expert: can train internally
        if capacity_gap >= 1.0 and has_expert:
            return {
                'action': 'internal_training',
                'priority': 'P1',
                'description': 'Sufficient expertise available for internal training program',
                'timeline': '3-6 months',
                'alternatives': ['contract_hire', 'consulting_partnership']
            }

        # Small gap with people developing: monitor
        if capacity_gap < 0.5 and developing_count > 0:
            return {
                'action': 'monitor',
                'priority': 'P2',
                'description': 'Team members developing capability, monitor progress',
                'timeline': '6-12 months',
                'alternatives': ['accelerated_training', 'mentorship_program']
            }

        # Moderate gap: flexible approach
        return {
            'action': 'flexible',
            'priority': 'P1',
            'description': 'Can address through hiring, training, or consulting',
            'timeline': '3-6 months',
            'alternatives': ['hire', 'train', 'contract', 'partnership']
        }

    def generate_hiring_plan(self, gaps: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Generate hiring plan from identified gaps
        
        Args:
            gaps: Capability gaps from identify_gaps()
            
        Returns:
            Structured hiring plan with priorities and timelines
        """
        # Group gaps by recommendation
        urgent_hires = [g for g in gaps if g['recommendation']['action'] == 'urgent_hire']
        training_needs = [g for g in gaps if g['recommendation']['action'] == 'internal_training']
        flexible_needs = [g for g in gaps if g['recommendation']['action'] == 'flexible']

        return {
            'urgent_hires': {
                'count': len(urgent_hires),
                'capabilities': [g['capability'] for g in urgent_hires],
                'timeline': '1-2 months',
                'estimated_cost': len(urgent_hires) * 200000,  # Rough estimate
                'risks': [
                    'Market competition for specialized talent',
                    'Long ramp-up time for domain knowledge',
                    'Cultural fit challenges'
                ]
            },
            'training_programs': {
                'participants': sum(g['developing_count'] for g in training_needs),
                'capabilities': [g['capability'] for g in training_needs],
                'timeline': '3-6 months',
                'estimated_cost': len(training_needs) * 50000,
                'success_factors': [
                    'Expert availability for mentorship',
                    'Hands-on project assignments',
                    'Regular skill assessments'
                ]
            },
            'flexible_positions': {
                'count': len(flexible_needs),
                'capabilities': [g['capability'] for g in flexible_needs],
                'options': ['full-time hire', 'contract', 'consulting partnership'],
                'decision_factors': [
                    'Budget constraints',
                    'Project timeline urgency',
                    'Long-term strategic needs'
                ]
            },
            'total_investment': (
                len(urgent_hires) * 200000 +
                len(training_needs) * 50000 +
                len(flexible_needs) * 150000
            ),
            'timeline_summary': {
                'immediate (0-2 months)': len(urgent_hires),
                'short-term (3-6 months)': len(training_needs) + len(flexible_needs),
                'long-term (6-12 months)': 'Capability development and maturation'
            }
        }

    def create_team_dashboard(self) -> str:
        """Generate visual team capability dashboard"""

        dashboard = "# Embedding Team Capability Dashboard\n\n"

        # Team overview
        dashboard += "## Team Overview\n"
        dashboard += f"- Total team members: {len(self.team_members)}\n"
        dashboard += f"- Total FTE capacity: {sum(m.capacity for m in self.team_members):.1f}\n"
        dashboard += f"- Active growth initiatives: {sum(len(m.growth_trajectory) for m in self.team_members)}\n\n"

        # Capability coverage matrix
        dashboard += "## Capability Coverage\n\n"
        dashboard += "| Capability | Expert | Proficient | Competent | Novice | Gap |\n"
        dashboard += "|------------|--------|------------|-----------|--------|-----|\n"

        for capability in Capability:
            expert = sum(1 for m in self.team_members
                        if m.proficiency(capability) == ProficiencyLevel.EXPERT)
            proficient = sum(1 for m in self.team_members
                           if m.proficiency(capability) == ProficiencyLevel.PROFICIENT)
            competent = sum(1 for m in self.team_members
                          if m.proficiency(capability) == ProficiencyLevel.COMPETENT)
            novice = sum(1 for m in self.team_members
                        if m.proficiency(capability) == ProficiencyLevel.NOVICE)

            # Find if there's a requirement
            req = next((r for r in self.requirements if r.capability == capability), None)
            gap_indicator = "✓" if not req or expert > 0 else "⚠" if proficient > 0 else "✗"

            dashboard += f"| {capability.value} | {expert} | {proficient} | {competent} | {novice} | {gap_indicator} |\n"

        # Critical gaps
        gaps = self.identify_gaps()
        if gaps:
            dashboard += "\n## Critical Capability Gaps\n\n"
            for gap in gaps[:5]:  # Top 5 most critical
                dashboard += f"### {gap['capability']} (Severity: {gap['severity']:.1f})\n"
                dashboard += f"- Required: {gap['required_headcount']} at {gap['required_proficiency']} level\n"
                dashboard += f"- Current: {gap['current_capacity']:.1f} FTE\n"
                dashboard += f"- Gap: {gap['capacity_gap']:.1f} FTE\n"
                dashboard += f"- Recommendation: {gap['recommendation']['action']} ({gap['recommendation']['priority']})\n"
                dashboard += f"- Timeline: {gap['recommendation']['timeline']}\n\n"

        return dashboard


# Example: Building an embedding team for enterprise deployment
def build_enterprise_embedding_team():
    """
    Example: Assess and build enterprise embedding team
    """

    assessment = TeamCapabilityAssessment()

    # Define requirements for enterprise embedding system
    requirements = [
        CapabilityRequirement(
            capability=Capability.EMBEDDING_EXPERTISE,
            min_proficiency=ProficiencyLevel.PROFICIENT,
            headcount=2,
            criticality=10
        ),
        CapabilityRequirement(
            capability=Capability.INFRASTRUCTURE,
            min_proficiency=ProficiencyLevel.PROFICIENT,
            headcount=2,
            criticality=9
        ),
        CapabilityRequirement(
            capability=Capability.ML_FOUNDATIONS,
            min_proficiency=ProficiencyLevel.COMPETENT,
            headcount=3,
            criticality=8
        ),
        CapabilityRequirement(
            capability=Capability.DATA_ENGINEERING,
            min_proficiency=ProficiencyLevel.COMPETENT,
            headcount=2,
            criticality=8
        ),
        CapabilityRequirement(
            capability=Capability.DOMAIN_KNOWLEDGE,
            min_proficiency=ProficiencyLevel.PROFICIENT,
            headcount=1,
            criticality=7
        ),
        CapabilityRequirement(
            capability=Capability.PRODUCT_MANAGEMENT,
            min_proficiency=ProficiencyLevel.COMPETENT,
            headcount=1,
            criticality=6
        )
    ]

    for req in requirements:
        assessment.add_requirement(req)

    # Current team (small, underspecialized)
    team_members = [
        TeamMember(
            name="Alice (ML Engineer)",
            role="ML Engineer",
            capabilities={
                Capability.ML_FOUNDATIONS: ProficiencyLevel.PROFICIENT,
                Capability.EMBEDDING_EXPERTISE: ProficiencyLevel.COMPETENT,
                Capability.RESEARCH: ProficiencyLevel.COMPETENT
            },
            experience_years=4.0,
            growth_trajectory={Capability.EMBEDDING_EXPERTISE, Capability.INFRASTRUCTURE}
        ),
        TeamMember(
            name="Bob (Backend Engineer)",
            role="Backend Engineer",
            capabilities={
                Capability.INFRASTRUCTURE: ProficiencyLevel.COMPETENT,
                Capability.DATA_ENGINEERING: ProficiencyLevel.COMPETENT,
                Capability.OPERATIONS: ProficiencyLevel.PROFICIENT
            },
            experience_years=6.0,
            growth_trajectory={Capability.INFRASTRUCTURE, Capability.ML_FOUNDATIONS}
        ),
        TeamMember(
            name="Carol (Data Scientist)",
            role="Data Scientist",
            capabilities={
                Capability.ML_FOUNDATIONS: ProficiencyLevel.COMPETENT,
                Capability.DOMAIN_KNOWLEDGE: ProficiencyLevel.PROFICIENT,
                Capability.PRODUCT_MANAGEMENT: ProficiencyLevel.NOVICE
            },
            experience_years=3.0,
            capacity=0.5,  # Split across multiple projects
            growth_trajectory={Capability.EMBEDDING_EXPERTISE}
        )
    ]

    for member in team_members:
        assessment.add_team_member(member)

    # Assess gaps
    gaps = assessment.identify_gaps()

    print("=== Capability Gap Analysis ===\n")
    for gap in gaps:
        print(f"Capability: {gap['capability']}")
        print(f"  Severity: {gap['severity']:.1f} (Gap: {gap['capacity_gap']:.1f} FTE, Criticality: {gap['criticality']})")
        print(f"  Recommendation: {gap['recommendation']['action']} - {gap['recommendation']['description']}")
        print(f"  Timeline: {gap['recommendation']['timeline']}")
        print()

    # Generate hiring plan
    hiring_plan = assessment.generate_hiring_plan(gaps)

    print("\n=== Hiring Plan ===\n")
    print(f"Urgent hires needed: {hiring_plan['urgent_hires']['count']}")
    print(f"  Capabilities: {', '.join(hiring_plan['urgent_hires']['capabilities'])}")
    print(f"  Timeline: {hiring_plan['urgent_hires']['timeline']}")
    print(f"  Estimated cost: ${hiring_plan['urgent_hires']['estimated_cost']:,}")

    print(f"\nTraining programs: {len(hiring_plan['training_programs']['capabilities'])}")
    print(f"  Participants: {hiring_plan['training_programs']['participants']}")
    print(f"  Timeline: {hiring_plan['training_programs']['timeline']}")

    print(f"\nTotal investment: ${hiring_plan['total_investment']:,}")

    # Display dashboard
    print("\n" + "="*60)
    print(assessment.create_team_dashboard())

if __name__ == "__main__":
    build_enterprise_embedding_team()
