# Code from Chapter 27
# Book: Embeddings at Scale

"""
Change Management Framework for Embedding Adoption

Architecture:
1. Stakeholder mapping: Identify champions, resistors, influencers
2. Readiness assessment: Evaluate organizational capability for change
3. Communication strategy: Tailor messaging to different audiences
4. Pilot design: Demonstrate value with low-risk, high-impact projects
5. Training rollout: Build capability systematically across organization
6. Feedback loops: Iterate based on user experience and concerns
7. Success amplification: Publicize wins to build momentum

Change stages:
- Awareness: Education on embedding capabilities and limitations
- Desire: Build excitement through demos and pilot results
- Knowledge: Training on how to use embedding-powered systems
- Ability: Provide tools, support, and resources for adoption
- Reinforcement: Recognize early adopters, measure and share success

Success factors:
- Executive sponsorship with visible commitment
- Early wins demonstrating clear value
- Addressing concerns transparently rather than dismissing
- Gradual rollout minimizing disruption
- Champions in each affected team advocating for change
- Sustained communication throughout adoption journey
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set


class StakeholderRole(Enum):
    """Stakeholder roles in change process"""
    EXECUTIVE_SPONSOR = "executive_sponsor"
    CHAMPION = "champion"
    EARLY_ADOPTER = "early_adopter"
    NEUTRAL = "neutral"
    SKEPTIC = "skeptic"
    RESISTOR = "resistor"
    BLOCKER = "blocker"

class ChangeReadiness(Enum):
    """Organization readiness for change"""
    READY = "ready"  # Culture and capability support change
    SOMEWHAT_READY = "somewhat_ready"  # Some barriers exist
    NOT_READY = "not_ready"  # Significant barriers to overcome

class CommunicationChannel(Enum):
    """Communication channels for change management"""
    ALL_HANDS = "all_hands"
    TEAM_MEETINGS = "team_meetings"
    EMAIL_UPDATES = "email_updates"
    SLACK_CHANNELS = "slack_channels"
    DEMOS = "demos"
    WORKSHOPS = "workshops"
    ONE_ON_ONE = "one_on_one"
    DOCUMENTATION = "documentation"

@dataclass
class Stakeholder:
    """
    Stakeholder in embedding adoption
    
    Attributes:
        name: Stakeholder identifier
        department: Department or team
        role: Role in change process
        influence: Influence level (1-10)
        concerns: Specific concerns about embedding adoption
        interests: What motivates this stakeholder
        preferred_channels: Preferred communication channels
    """
    name: str
    department: str
    role: StakeholderRole
    influence: int
    concerns: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    preferred_channels: Set[CommunicationChannel] = field(default_factory=set)

    def engagement_priority(self) -> int:
        """Calculate engagement priority based on influence and role"""
        role_weights = {
            StakeholderRole.EXECUTIVE_SPONSOR: 10,
            StakeholderRole.CHAMPION: 8,
            StakeholderRole.BLOCKER: 9,
            StakeholderRole.RESISTOR: 7,
            StakeholderRole.EARLY_ADOPTER: 6,
            StakeholderRole.SKEPTIC: 5,
            StakeholderRole.NEUTRAL: 3
        }
        return self.influence * role_weights[self.role]

@dataclass
class ChangeBarrier:
    """
    Barrier to embedding adoption
    
    Attributes:
        name: Barrier identifier
        category: Type of barrier (technical, cultural, political)
        severity: Impact on adoption (1-10)
        affected_stakeholders: Stakeholders affected by this barrier
        mitigation_strategy: How to address this barrier
        timeline: Time needed to address
    """
    name: str
    category: str
    severity: int
    affected_stakeholders: List[str]
    mitigation_strategy: str
    timeline: str

@dataclass
class PilotProject:
    """
    Pilot project for demonstrating embedding value
    
    Attributes:
        name: Project name
        description: What the pilot will accomplish
        target_metrics: Success metrics
        stakeholders: Involved stakeholders
        timeline: Project timeline
        risk_level: Implementation risk (low/medium/high)
        business_impact: Expected business value
    """
    name: str
    description: str
    target_metrics: Dict[str, float]
    stakeholders: List[str]
    timeline: str
    risk_level: str
    business_impact: str
    status: str = "planned"
    actual_results: Optional[Dict[str, float]] = None

class ChangeManagementFramework:
    """
    Framework for managing embedding adoption change
    
    Manages stakeholder engagement, communication strategy,
    pilot projects, and progress tracking
    """

    def __init__(self, organization_name: str):
        self.organization_name = organization_name
        self.stakeholders: List[Stakeholder] = []
        self.barriers: List[ChangeBarrier] = []
        self.pilots: List[PilotProject] = []
        self.communication_log: List[Dict] = []

    def add_stakeholder(self, stakeholder: Stakeholder):
        """Add stakeholder to framework"""
        self.stakeholders.append(stakeholder)

    def add_barrier(self, barrier: ChangeBarrier):
        """Add adoption barrier"""
        self.barriers.append(barrier)

    def add_pilot(self, pilot: PilotProject):
        """Add pilot project"""
        self.pilots.append(pilot)

    def assess_readiness(self) -> Dict[str, any]:
        """
        Assess organizational readiness for embedding adoption
        
        Returns:
            Readiness assessment with recommendations
        """
        # Count stakeholders by role
        role_counts = {}
        for stakeholder in self.stakeholders:
            role = stakeholder.role
            role_counts[role] = role_counts.get(role, 0) + 1

        # Assess leadership support
        has_executive_sponsor = StakeholderRole.EXECUTIVE_SPONSOR in role_counts
        champion_count = role_counts.get(StakeholderRole.CHAMPION, 0)
        resistor_count = role_counts.get(StakeholderRole.RESISTOR, 0) + \
                        role_counts.get(StakeholderRole.BLOCKER, 0)

        # Assess barriers
        critical_barriers = [b for b in self.barriers if b.severity >= 8]
        moderate_barriers = [b for b in self.barriers if 5 <= b.severity < 8]

        # Calculate readiness score
        readiness_score = 0
        if has_executive_sponsor:
            readiness_score += 30
        readiness_score += min(champion_count * 10, 30)  # Up to 3 champions
        readiness_score -= resistor_count * 15
        readiness_score -= len(critical_barriers) * 10
        readiness_score -= len(moderate_barriers) * 5

        # Determine readiness level
        if readiness_score >= 60:
            readiness = ChangeReadiness.READY
            recommendation = "Proceed with phased rollout"
        elif readiness_score >= 30:
            readiness = ChangeReadiness.SOMEWHAT_READY
            recommendation = "Address critical barriers before full rollout"
        else:
            readiness = ChangeReadiness.NOT_READY
            recommendation = "Build foundation before attempting adoption"

        return {
            'readiness': readiness.value,
            'score': readiness_score,
            'has_executive_sponsor': has_executive_sponsor,
            'champion_count': champion_count,
            'resistor_count': resistor_count,
            'critical_barriers': len(critical_barriers),
            'moderate_barriers': len(moderate_barriers),
            'recommendation': recommendation,
            'next_steps': self._generate_next_steps(
                readiness,
                has_executive_sponsor,
                champion_count,
                critical_barriers
            )
        }

    def _generate_next_steps(
        self,
        readiness: ChangeReadiness,
        has_executive_sponsor: bool,
        champion_count: int,
        critical_barriers: List[ChangeBarrier]
    ) -> List[str]:
        """Generate recommended next steps based on readiness"""

        steps = []

        if not has_executive_sponsor:
            steps.append("Secure executive sponsorship through business case and demos")

        if champion_count < 2:
            steps.append("Identify and recruit 2-3 champions across key departments")

        if critical_barriers:
            steps.append(f"Address {len(critical_barriers)} critical barriers: " +
                        ", ".join(b.name for b in critical_barriers[:3]))

        if readiness == ChangeReadiness.READY:
            steps.extend([
                "Launch pilot project with early adopters",
                "Establish communication cadence for updates",
                "Begin training program for affected teams",
                "Set up feedback mechanisms for iterative improvement"
            ])
        elif readiness == ChangeReadiness.SOMEWHAT_READY:
            steps.extend([
                "Run small proof-of-concept with friendly team",
                "Document and address concerns from skeptics",
                "Build technical capability through training",
                "Create detailed rollout plan addressing barriers"
            ])
        else:  # NOT_READY
            steps.extend([
                "Build awareness through education sessions",
                "Demonstrate value through external case studies",
                "Assess technical and organizational gaps",
                "Develop 6-12 month readiness roadmap"
            ])

        return steps

    def design_communication_strategy(self) -> Dict[str, any]:
        """
        Design stakeholder communication strategy
        
        Returns:
            Communication plan tailored to different stakeholder groups
        """
        # Group stakeholders by role
        grouped_stakeholders = {}
        for stakeholder in self.stakeholders:
            role = stakeholder.role
            if role not in grouped_stakeholders:
                grouped_stakeholders[role] = []
            grouped_stakeholders[role].append(stakeholder)

        # Design messaging for each group
        messaging_strategy = {}

        # Executive sponsors: Business value, ROI, strategic advantage
        if StakeholderRole.EXECUTIVE_SPONSOR in grouped_stakeholders:
            messaging_strategy['executives'] = {
                'stakeholders': [s.name for s in grouped_stakeholders[StakeholderRole.EXECUTIVE_SPONSOR]],
                'key_messages': [
                    'Competitive advantage through AI-powered capabilities',
                    'ROI projections and success metrics',
                    'Risk mitigation and governance approach',
                    'Strategic roadmap and resource requirements'
                ],
                'channels': [CommunicationChannel.ONE_ON_ONE, CommunicationChannel.EMAIL_UPDATES],
                'frequency': 'Monthly',
                'content_type': 'Business case, success metrics, strategic updates'
            }

        # Champions: Technical details, implementation progress, how to advocate
        if StakeholderRole.CHAMPION in grouped_stakeholders:
            messaging_strategy['champions'] = {
                'stakeholders': [s.name for s in grouped_stakeholders[StakeholderRole.CHAMPION]],
                'key_messages': [
                    'Technical architecture and capabilities',
                    'Implementation progress and challenges',
                    'How to advocate to their teams',
                    'Resources and support available'
                ],
                'channels': [CommunicationChannel.WORKSHOPS, CommunicationChannel.SLACK_CHANNELS],
                'frequency': 'Weekly',
                'content_type': 'Technical deep dives, demos, Q&A sessions'
            }

        # Skeptics/resistors: Address concerns, demonstrate value, reduce risk
        skeptics_and_resistors = (
            grouped_stakeholders.get(StakeholderRole.SKEPTIC, []) +
            grouped_stakeholders.get(StakeholderRole.RESISTOR, [])
        )
        if skeptics_and_resistors:
            messaging_strategy['skeptics'] = {
                'stakeholders': [s.name for s in skeptics_and_resistors],
                'key_messages': [
                    'Transparent acknowledgment of limitations',
                    'How concerns are being addressed',
                    'Evidence from pilot projects and external success stories',
                    'Gradual rollout minimizing disruption'
                ],
                'channels': [CommunicationChannel.ONE_ON_ONE, CommunicationChannel.DEMOS],
                'frequency': 'As needed, minimum monthly',
                'content_type': 'Direct conversations addressing specific concerns'
            }

        # Blockers: Understand motivations, find common ground, escalate if needed
        if StakeholderRole.BLOCKER in grouped_stakeholders:
            messaging_strategy['blockers'] = {
                'stakeholders': [s.name for s in grouped_stakeholders[StakeholderRole.BLOCKER]],
                'key_messages': [
                    'Understanding their concerns and constraints',
                    'Finding mutually beneficial approaches',
                    'Executive alignment on strategic direction',
                    'Clear escalation path if blocking continues'
                ],
                'channels': [CommunicationChannel.ONE_ON_ONE],
                'frequency': 'Weekly until resolution',
                'content_type': 'Direct negotiation, executive involvement if needed'
            }

        # Broad organization: General awareness, training opportunities, success stories
        messaging_strategy['organization_wide'] = {
            'key_messages': [
                'What embeddings are and why they matter',
                'How they will improve daily work',
                'Training and support available',
                'Success stories from early adopters'
            ],
            'channels': [
                CommunicationChannel.ALL_HANDS,
                CommunicationChannel.EMAIL_UPDATES,
                CommunicationChannel.DOCUMENTATION
            ],
            'frequency': 'Monthly major updates, quarterly deep dives',
            'content_type': 'Accessible explanations, demos, case studies'
        }

        return {
            'strategy': messaging_strategy,
            'overall_principles': [
                'Transparency: Acknowledge limitations and challenges openly',
                'Evidence: Back claims with data from pilots and external examples',
                'Empathy: Address concerns rather than dismissing them',
                'Consistency: Regular communication prevents information vacuum',
                'Two-way: Solicit feedback and iterate based on input'
            ],
            'communication_calendar': self._create_communication_calendar(messaging_strategy)
        }

    def _create_communication_calendar(
        self,
        messaging_strategy: Dict[str, any]
    ) -> List[Dict[str, str]]:
        """Create month-by-month communication calendar"""

        calendar = []

        # Month 1: Launch and awareness
        calendar.append({
            'month': 1,
            'theme': 'Launch and Awareness',
            'activities': [
                'All-hands announcement of embedding initiative',
                'Executive sponsor blog post on strategic vision',
                'Technical webinar for champions and early adopters',
                'One-on-one meetings with key skeptics'
            ]
        })

        # Month 2-3: Education and pilot start
        calendar.append({
            'month': '2-3',
            'theme': 'Education and Pilot Launch',
            'activities': [
                'Weekly demos of embedding capabilities',
                'Training workshops for affected teams',
                'Pilot project kick-off with early adopters',
                'Monthly email updates on progress'
            ]
        })

        # Month 4-6: Pilot results and iteration
        calendar.append({
            'month': '4-6',
            'theme': 'Pilot Results and Learning',
            'activities': [
                'Pilot results presentation to executives',
                'Success stories shared in all-hands and internal communications',
                'Iteration on system based on feedback',
                'Expansion planning with additional teams'
            ]
        })

        # Month 7-12: Scale and reinforcement
        calendar.append({
            'month': '7-12',
            'theme': 'Scale and Reinforcement',
            'activities': [
                'Phased rollout to additional departments',
                'Recognition program for early adopters and champions',
                'Quarterly business review showing impact metrics',
                'Documentation and best practices dissemination'
            ]
        })

        return calendar

    def track_progress(self) -> Dict[str, any]:
        """
        Track change management progress
        
        Returns:
            Progress dashboard with metrics and status
        """
        # Stakeholder engagement metrics
        engagement_score = sum(
            1 for s in self.stakeholders
            if s.role in [StakeholderRole.CHAMPION, StakeholderRole.EARLY_ADOPTER]
        ) / max(len(self.stakeholders), 1)

        resistance_score = sum(
            1 for s in self.stakeholders
            if s.role in [StakeholderRole.RESISTOR, StakeholderRole.BLOCKER]
        ) / max(len(self.stakeholders), 1)

        # Pilot project status
        completed_pilots = [p for p in self.pilots if p.status == 'completed']
        successful_pilots = [
            p for p in completed_pilots
            if p.actual_results and all(
                p.actual_results.get(k, 0) >= v
                for k, v in p.target_metrics.items()
            )
        ]

        # Barrier resolution
        resolved_barriers = [b for b in self.barriers if b.severity < 3]  # Largely addressed

        return {
            'engagement_score': engagement_score,
            'resistance_score': resistance_score,
            'stakeholder_breakdown': {
                role.value: len([s for s in self.stakeholders if s.role == role])
                for role in StakeholderRole
            },
            'pilot_status': {
                'total': len(self.pilots),
                'completed': len(completed_pilots),
                'successful': len(successful_pilots),
                'success_rate': len(successful_pilots) / max(len(completed_pilots), 1)
            },
            'barrier_status': {
                'total': len(self.barriers),
                'resolved': len(resolved_barriers),
                'critical_remaining': len([b for b in self.barriers if b.severity >= 8])
            },
            'overall_health': self._assess_overall_health(
                engagement_score,
                resistance_score,
                len(successful_pilots),
                len([b for b in self.barriers if b.severity >= 8])
            )
        }

    def _assess_overall_health(
        self,
        engagement_score: float,
        resistance_score: float,
        successful_pilots: int,
        critical_barriers: int
    ) -> str:
        """Assess overall change management health"""

        if (engagement_score > 0.3 and resistance_score < 0.2 and
            successful_pilots >= 2 and critical_barriers == 0):
            return "Healthy - Change progressing well"
        elif (engagement_score > 0.2 and resistance_score < 0.3 and
              successful_pilots >= 1):
            return "Moderate - Some challenges but manageable"
        else:
            return "At Risk - Significant intervention needed"


# Example: Enterprise change management for embedding adoption
def manage_enterprise_embedding_change():
    """
    Example: Manage change for enterprise embedding adoption
    """

    framework = ChangeManagementFramework("TechCorp")

    # Add stakeholders
    stakeholders = [
        Stakeholder(
            name="CTO (Executive Sponsor)",
            department="Engineering",
            role=StakeholderRole.EXECUTIVE_SPONSOR,
            influence=10,
            concerns=["Budget", "Timeline", "Risk"],
            interests=["Innovation", "Competitive advantage", "Efficiency"],
            preferred_channels={CommunicationChannel.ONE_ON_ONE, CommunicationChannel.EMAIL_UPDATES}
        ),
        Stakeholder(
            name="Head of Search (Champion)",
            department="Product",
            role=StakeholderRole.CHAMPION,
            influence=8,
            concerns=["User experience", "Performance"],
            interests=["Better search results", "User satisfaction"],
            preferred_channels={CommunicationChannel.DEMOS, CommunicationChannel.WORKSHOPS}
        ),
        Stakeholder(
            name="VP Operations (Skeptic)",
            department="Operations",
            role=StakeholderRole.SKEPTIC,
            influence=7,
            concerns=["Operational complexity", "Support burden", "Reliability"],
            interests=["System stability", "Cost control"],
            preferred_channels={CommunicationChannel.ONE_ON_ONE}
        ),
        Stakeholder(
            name="Security Lead (Blocker)",
            department="Security",
            role=StakeholderRole.BLOCKER,
            influence=9,
            concerns=["Data leakage", "Compliance", "Auditability"],
            interests=["Security posture", "Regulatory compliance"],
            preferred_channels={CommunicationChannel.ONE_ON_ONE}
        )
    ]

    for stakeholder in stakeholders:
        framework.add_stakeholder(stakeholder)

    # Add barriers
    barriers = [
        ChangeBarrier(
            name="Security concerns about embedding data",
            category="compliance",
            severity=9,
            affected_stakeholders=["Security Lead"],
            mitigation_strategy="Implement encryption, access controls, compliance documentation",
            timeline="2 months"
        ),
        ChangeBarrier(
            name="Lack of embedding expertise in team",
            category="technical",
            severity=8,
            affected_stakeholders=["Head of Search", "VP Operations"],
            mitigation_strategy="Hire 2 embedding ML engineers, training program for existing team",
            timeline="3-6 months"
        ),
        ChangeBarrier(
            name="Integration complexity with existing systems",
            category="technical",
            severity=6,
            affected_stakeholders=["VP Operations"],
            mitigation_strategy="Gradual migration, maintain parallel systems during transition",
            timeline="6 months"
        )
    ]

    for barrier in barriers:
        framework.add_barrier(barrier)

    # Add pilot projects
    pilots = [
        PilotProject(
            name="Product Search Improvement",
            description="Replace keyword search with semantic search for product catalog",
            target_metrics={
                'click_through_rate': 0.15,  # 15% improvement
                'user_satisfaction': 0.10    # 10% improvement
            },
            stakeholders=["Head of Search"],
            timeline="3 months",
            risk_level="low",
            business_impact="High - directly affects customer experience and conversion"
        ),
        PilotProject(
            name="Internal Knowledge Base Search",
            description="Improve employee search for internal documentation",
            target_metrics={
                'search_success_rate': 0.25,  # 25% improvement
                'time_to_find_info': -0.30    # 30% reduction
            },
            stakeholders=["VP Operations"],
            timeline="2 months",
            risk_level="low",
            business_impact="Medium - improves employee productivity"
        )
    ]

    for pilot in pilots:
        framework.add_pilot(pilot)

    # Assess readiness
    readiness = framework.assess_readiness()

    print("=== Change Readiness Assessment ===\n")
    print(f"Readiness: {readiness['readiness']}")
    print(f"Score: {readiness['score']}")
    print(f"Recommendation: {readiness['recommendation']}\n")
    print("Next Steps:")
    for i, step in enumerate(readiness['next_steps'], 1):
        print(f"  {i}. {step}")

    # Design communication strategy
    comm_strategy = framework.design_communication_strategy()

    print("\n=== Communication Strategy ===\n")
    for audience, details in comm_strategy['strategy'].items():
        print(f"{audience.upper()}:")
        if 'stakeholders' in details:
            print(f"  Stakeholders: {', '.join(details['stakeholders'])}")
        print(f"  Frequency: {details['frequency']}")
        print("  Key Messages:")
        for msg in details['key_messages']:
            print(f"    - {msg}")
        print()

    # Track progress (simulated)
    progress = framework.track_progress()

    print("=== Progress Dashboard ===\n")
    print(f"Overall Health: {progress['overall_health']}")
    print(f"Engagement Score: {progress['engagement_score']:.1%}")
    print(f"Resistance Score: {progress['resistance_score']:.1%}")
    print("\nPilot Status:")
    print(f"  Completed: {progress['pilot_status']['completed']}/{progress['pilot_status']['total']}")
    print(f"  Success Rate: {progress['pilot_status']['success_rate']:.1%}")

if __name__ == "__main__":
    manage_enterprise_embedding_change()
