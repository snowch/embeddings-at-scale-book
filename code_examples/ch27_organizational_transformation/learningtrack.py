# Code from Chapter 27
# Book: Embeddings at Scale

"""
Embedding Training Program Framework

Architecture:
1. Learning tracks: Tailored curricula for different roles
2. Competency assessment: Measure baseline and progress
3. Project-based learning: Hands-on work on real problems
4. Mentorship pairing: Experts guide learners
5. Community building: Shared learning, Q&A, best practices
6. Certification: Validate competency at different levels

Learning tracks:
- ML Engineer track: Deep technical, model development
- Infrastructure track: Distributed systems, vector databases
- Product track: Application design, user experience
- Business track: Strategic understanding, ROI assessment

Learning methods:
- Self-paced modules: Video, documentation, quizzes
- Live workshops: Interactive sessions with experts
- Hackathons: Team projects solving real problems
- Mentorship: 1-on-1 guidance from experienced practitioners
- Community: Slack channel, office hours, brown bags

Competency levels:
- Novice: Basic understanding, can consume embeddings
- Competent: Can implement standard applications
- Proficient: Can design custom solutions, optimize
- Expert: Can architect systems, research new techniques
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class LearningTrack(Enum):
    """Training tracks for different roles"""
    ML_ENGINEER = "ml_engineer"
    INFRASTRUCTURE = "infrastructure"
    DATA_ENGINEER = "data_engineer"
    PRODUCT_MANAGER = "product_manager"
    BUSINESS_STAKEHOLDER = "business_stakeholder"

class CompetencyLevel(Enum):
    """Competency levels"""
    NOVICE = 1
    COMPETENT = 2
    PROFICIENT = 3
    EXPERT = 4

class LearningMethod(Enum):
    """Learning delivery methods"""
    SELF_PACED_VIDEO = "self_paced_video"
    DOCUMENTATION = "documentation"
    LIVE_WORKSHOP = "live_workshop"
    HANDS_ON_PROJECT = "hands_on_project"
    MENTORSHIP = "mentorship"
    HACKATHON = "hackathon"
    CONFERENCE_TALK = "conference_talk"
    READING_GROUP = "reading_group"

@dataclass
class LearningModule:
    """
    Training module

    Attributes:
        name: Module name
        track: Which learning track this belongs to
        level: Target competency level
        duration_hours: Expected completion time
        prerequisites: Required prior knowledge
        learning_objectives: What learners will be able to do
        delivery_methods: How content is delivered
        assessment: How competency is validated
    """
    name: str
    track: LearningTrack
    level: CompetencyLevel
    duration_hours: float
    prerequisites: List[str]
    learning_objectives: List[str]
    delivery_methods: Set[LearningMethod]
    assessment: str
    completed_by: Set[str] = field(default_factory=set)

@dataclass
class Learner:
    """
    Individual learner profile

    Attributes:
        name: Learner identifier
        role: Job role
        primary_track: Primary learning track
        current_level: Current competency level
        completed_modules: Modules completed
        active_projects: Projects currently working on
        mentor: Assigned mentor (if any)
        learning_goals: Personal learning objectives
    """
    name: str
    role: str
    primary_track: LearningTrack
    current_level: CompetencyLevel
    completed_modules: List[str] = field(default_factory=list)
    active_projects: List[str] = field(default_factory=list)
    mentor: Optional[str] = None
    learning_goals: List[str] = field(default_factory=list)

    def complete_module(self, module_name: str):
        """Mark module as completed"""
        if module_name not in self.completed_modules:
            self.completed_modules.append(module_name)

@dataclass
class Project:
    """
    Hands-on learning project

    Attributes:
        name: Project name
        description: What the project involves
        track: Primary learning track
        difficulty: Difficulty level
        duration_weeks: Expected duration
        learning_outcomes: Skills developed
        real_world_application: Whether uses real company data
        mentors_available: Mentors who can guide
        participants: Current participants
    """
    name: str
    description: str
    track: LearningTrack
    difficulty: CompetencyLevel
    duration_weeks: int
    learning_outcomes: List[str]
    real_world_application: bool
    mentors_available: List[str]
    participants: List[str] = field(default_factory=list)

class TrainingProgram:
    """
    Comprehensive embedding training program

    Manages learning tracks, modules, projects, mentorship,
    and competency progression
    """

    def __init__(self, program_name: str):
        self.program_name = program_name
        self.modules: Dict[str, LearningModule] = {}
        self.learners: Dict[str, Learner] = {}
        self.projects: Dict[str, Project] = {}
        self.mentors: Set[str] = set()

    def add_module(self, module: LearningModule):
        """Add learning module"""
        self.modules[module.name] = module

    def add_learner(self, learner: Learner):
        """Add learner to program"""
        self.learners[learner.name] = learner

    def add_project(self, project: Project):
        """Add hands-on project"""
        self.projects[project.name] = project

    def add_mentor(self, mentor_name: str):
        """Register mentor"""
        self.mentors.add(mentor_name)

    def recommend_modules(
        self,
        learner_name: str,
        max_recommendations: int = 5
    ) -> List[Dict[str, any]]:
        """
        Recommend next modules for learner

        Args:
            learner_name: Learner identifier
            max_recommendations: Maximum modules to recommend

        Returns:
            Recommended modules with reasoning
        """
        learner = self.learners[learner_name]
        recommendations = []

        # Get modules for learner's track
        track_modules = [
            m for m in self.modules.values()
            if m.track == learner.primary_track
        ]

        for module in track_modules:
            # Skip if already completed
            if module.name in learner.completed_modules:
                continue

            # Check prerequisites
            missing_prereqs = [
                p for p in module.prerequisites
                if p not in learner.completed_modules
            ]

            if missing_prereqs:
                continue  # Can't take yet

            # Calculate relevance score
            level_match = abs(module.level.value - learner.current_level.value)
            relevance_score = 10 - level_match  # Higher is better

            # Prefer modules at or slightly above current level
            if module.level.value == learner.current_level.value + 1:
                relevance_score += 5  # Bonus for next level

            recommendations.append({
                'module': module.name,
                'track': module.track.value,
                'level': module.level.name,
                'duration_hours': module.duration_hours,
                'relevance_score': relevance_score,
                'reasoning': self._generate_recommendation_reasoning(
                    learner, module, missing_prereqs
                )
            })

        # Sort by relevance and return top recommendations
        recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
        return recommendations[:max_recommendations]

    def _generate_recommendation_reasoning(
        self,
        learner: Learner,
        module: LearningModule,
        missing_prereqs: List[str]
    ) -> str:
        """Generate explanation for recommendation"""

        if module.level.value == learner.current_level.value + 1:
            return f"Next step in your progression to {module.level.name} level"
        elif module.level.value == learner.current_level.value:
            return f"Reinforces your current {learner.current_level.name} level skills"
        elif module.level.value < learner.current_level.value:
            return "Foundation module to fill potential gaps"
        else:
            return "Advanced module for when you're ready to level up"

    def assign_mentor(self, learner_name: str, mentor_name: str):
        """Assign mentor to learner"""
        if mentor_name not in self.mentors:
            raise ValueError(f"Mentor {mentor_name} not registered")

        learner = self.learners[learner_name]
        learner.mentor = mentor_name

    def track_progress(self, learner_name: str) -> Dict[str, any]:
        """
        Track learner progress

        Args:
            learner_name: Learner identifier

        Returns:
            Progress summary and recommendations
        """
        learner = self.learners[learner_name]

        # Count modules by level
        track_modules = [
            m for m in self.modules.values()
            if m.track == learner.primary_track
        ]

        modules_by_level = {}
        completed_by_level = {}

        for level in CompetencyLevel:
            modules_at_level = [
                m for m in track_modules
                if m.level == level
            ]
            completed_at_level = [
                m for m in modules_at_level
                if m.name in learner.completed_modules
            ]

            modules_by_level[level] = len(modules_at_level)
            completed_by_level[level] = len(completed_at_level)

        # Calculate completion percentage
        total_modules = sum(modules_by_level.values())
        total_completed = len(learner.completed_modules)
        completion_percentage = (total_completed / total_modules * 100) if total_modules > 0 else 0

        # Assess readiness for level advancement
        current_level_modules = modules_by_level.get(learner.current_level, 0)
        current_level_completed = completed_by_level.get(learner.current_level, 0)

        ready_for_advancement = (
            current_level_modules > 0 and
            current_level_completed / current_level_modules >= 0.8
        )

        return {
            'learner': learner_name,
            'current_level': learner.current_level.name,
            'modules_completed': total_completed,
            'total_modules': total_modules,
            'completion_percentage': completion_percentage,
            'modules_by_level': {
                level.name: {
                    'total': modules_by_level.get(level, 0),
                    'completed': completed_by_level.get(level, 0)
                }
                for level in CompetencyLevel
            },
            'ready_for_advancement': ready_for_advancement,
            'active_projects': len(learner.active_projects),
            'has_mentor': learner.mentor is not None,
            'recommendations': self.recommend_modules(learner_name, max_recommendations=3)
        }

    def generate_curriculum(self, track: LearningTrack) -> str:
        """
        Generate curriculum overview for track

        Args:
            track: Learning track

        Returns:
            Formatted curriculum
        """
        curriculum = f"# {track.value.replace('_', ' ').title()} Curriculum\n\n"

        # Get modules for track
        track_modules = [
            m for m in self.modules.values()
            if m.track == track
        ]

        # Group by level
        by_level = {}
        for module in track_modules:
            if module.level not in by_level:
                by_level[module.level] = []
            by_level[module.level].append(module)

        # Format curriculum by level
        for level in sorted(by_level.keys(), key=lambda lvl: lvl.value):
            curriculum += f"## {level.name} Level\n\n"

            total_hours = sum(m.duration_hours for m in by_level[level])
            curriculum += f"*Total: {len(by_level[level])} modules, ~{total_hours:.0f} hours*\n\n"

            for module in by_level[level]:
                curriculum += f"### {module.name}\n"
                curriculum += f"- Duration: {module.duration_hours} hours\n"
                if module.prerequisites:
                    curriculum += f"- Prerequisites: {', '.join(module.prerequisites)}\n"
                curriculum += "- Learning Objectives:\n"
                for obj in module.learning_objectives:
                    curriculum += f"  - {obj}\n"
                curriculum += f"- Delivery: {', '.join(m.value for m in module.delivery_methods)}\n"
                curriculum += f"- Assessment: {module.assessment}\n\n"

        return curriculum

    def create_program_dashboard(self) -> str:
        """Generate program overview dashboard"""

        dashboard = f"# {self.program_name} Dashboard\n\n"

        # Overall statistics
        dashboard += "## Program Statistics\n\n"
        dashboard += f"- Total learners: {len(self.learners)}\n"
        dashboard += f"- Total modules: {len(self.modules)}\n"
        dashboard += f"- Active projects: {len(self.projects)}\n"
        dashboard += f"- Available mentors: {len(self.mentors)}\n\n"

        # Learner distribution by track
        dashboard += "## Learners by Track\n\n"
        by_track = {}
        for learner in self.learners.values():
            track = learner.primary_track
            by_track[track] = by_track.get(track, 0) + 1

        for track, count in sorted(by_track.items(), key=lambda x: x[1], reverse=True):
            dashboard += f"- {track.value}: {count} learners\n"

        # Learner distribution by level
        dashboard += "\n## Learners by Competency Level\n\n"
        by_level = {}
        for learner in self.learners.values():
            level = learner.current_level
            by_level[level] = by_level.get(level, 0) + 1

        for level in CompetencyLevel:
            count = by_level.get(level, 0)
            dashboard += f"- {level.name}: {count} learners\n"

        # Module completion statistics
        dashboard += "\n## Module Completion\n\n"
        total_completions = sum(len(learner.completed_modules) for learner in self.learners.values())
        avg_completions = total_completions / len(self.learners) if self.learners else 0
        dashboard += f"- Total completions: {total_completions}\n"
        dashboard += f"- Average per learner: {avg_completions:.1f}\n"

        # Most popular modules
        module_popularity = {}
        for learner in self.learners.values():
            for module_name in learner.completed_modules:
                module_popularity[module_name] = module_popularity.get(module_name, 0) + 1

        if module_popularity:
            dashboard += "\n### Most Completed Modules\n\n"
            top_modules = sorted(
                module_popularity.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            for module_name, count in top_modules:
                dashboard += f"- {module_name}: {count} completions\n"

        return dashboard


# Example: Enterprise embedding training program
def create_enterprise_training_program():
    """
    Example: Create comprehensive training program
    """

    program = TrainingProgram("Enterprise Embedding Excellence Program")

    # ML Engineer track modules
    ml_modules = [
        LearningModule(
            name="Embedding Fundamentals",
            track=LearningTrack.ML_ENGINEER,
            level=CompetencyLevel.NOVICE,
            duration_hours=8,
            prerequisites=[],
            learning_objectives=[
                "Understand vector representations and similarity",
                "Know common embedding models (Word2Vec, BERT, etc.)",
                "Compute and interpret embedding similarities"
            ],
            delivery_methods={
                LearningMethod.SELF_PACED_VIDEO,
                LearningMethod.DOCUMENTATION
            },
            assessment="Quiz and simple embedding generation exercise"
        ),
        LearningModule(
            name="Contrastive Learning Deep Dive",
            track=LearningTrack.ML_ENGINEER,
            level=CompetencyLevel.COMPETENT,
            duration_hours=16,
            prerequisites=["Embedding Fundamentals"],
            learning_objectives=[
                "Implement SimCLR and MoCo architectures",
                "Design effective data augmentation strategies",
                "Optimize contrastive learning hyperparameters",
                "Train custom embeddings on domain data"
            ],
            delivery_methods={
                LearningMethod.LIVE_WORKSHOP,
                LearningMethod.HANDS_ON_PROJECT
            },
            assessment="Train custom embedding model achieving target performance"
        ),
        LearningModule(
            name="Production Embedding Systems",
            track=LearningTrack.ML_ENGINEER,
            level=CompetencyLevel.PROFICIENT,
            duration_hours=20,
            prerequisites=["Contrastive Learning Deep Dive"],
            learning_objectives=[
                "Design end-to-end embedding pipelines",
                "Implement quality monitoring and drift detection",
                "Optimize for latency and cost at scale",
                "Handle embedding versioning and rollback"
            ],
            delivery_methods={
                LearningMethod.LIVE_WORKSHOP,
                LearningMethod.HANDS_ON_PROJECT,
                LearningMethod.MENTORSHIP
            },
            assessment="Deploy production embedding system with monitoring"
        )
    ]

    # Infrastructure track modules
    infra_modules = [
        LearningModule(
            name="Vector Database Fundamentals",
            track=LearningTrack.INFRASTRUCTURE,
            level=CompetencyLevel.NOVICE,
            duration_hours=8,
            prerequisites=[],
            learning_objectives=[
                "Understand vector indexing algorithms (HNSW, IVF)",
                "Set up and configure vector databases",
                "Optimize queries for latency and throughput"
            ],
            delivery_methods={
                LearningMethod.SELF_PACED_VIDEO,
                LearningMethod.LIVE_WORKSHOP
            },
            assessment="Deploy and benchmark vector database"
        ),
        LearningModule(
            name="Scaling to Trillions of Vectors",
            track=LearningTrack.INFRASTRUCTURE,
            level=CompetencyLevel.PROFICIENT,
            duration_hours=24,
            prerequisites=["Vector Database Fundamentals"],
            learning_objectives=[
                "Design distributed vector systems",
                "Implement sharding and replication strategies",
                "Optimize for global deployment",
                "Handle failure modes and disaster recovery"
            ],
            delivery_methods={
                LearningMethod.LIVE_WORKSHOP,
                LearningMethod.HANDS_ON_PROJECT,
                LearningMethod.MENTORSHIP
            },
            assessment="Architecture design for trillion-row system"
        )
    ]

    # Product Manager track modules
    product_modules = [
        LearningModule(
            name="Embedding Applications for Product Managers",
            track=LearningTrack.PRODUCT_MANAGER,
            level=CompetencyLevel.NOVICE,
            duration_hours=4,
            prerequisites=[],
            learning_objectives=[
                "Understand what embeddings enable",
                "Identify high-impact use cases",
                "Evaluate embedding system capabilities",
                "Define success metrics for embedding products"
            ],
            delivery_methods={
                LearningMethod.SELF_PACED_VIDEO,
                LearningMethod.LIVE_WORKSHOP
            },
            assessment="Use case proposal with metrics"
        ),
        LearningModule(
            name="Building Embedding-Powered Products",
            track=LearningTrack.PRODUCT_MANAGER,
            level=CompetencyLevel.COMPETENT,
            duration_hours=12,
            prerequisites=["Embedding Applications for Product Managers"],
            learning_objectives=[
                "Design user experiences leveraging embeddings",
                "Balance technical constraints with user needs",
                "Run A/B tests on embedding systems",
                "Measure and optimize product impact"
            ],
            delivery_methods={
                LearningMethod.LIVE_WORKSHOP,
                LearningMethod.HANDS_ON_PROJECT
            },
            assessment="Product spec and A/B test plan"
        )
    ]

    # Add all modules
    for module in ml_modules + infra_modules + product_modules:
        program.add_module(module)

    # Add mentors
    for mentor in ["Senior ML Engineer Alice", "Staff SRE Bob", "Principal PM Carol"]:
        program.add_mentor(mentor)

    # Add hands-on projects
    projects = [
        Project(
            name="Semantic Search for Product Catalog",
            description="Build semantic search replacing keyword search",
            track=LearningTrack.ML_ENGINEER,
            difficulty=CompetencyLevel.COMPETENT,
            duration_weeks=4,
            learning_outcomes=[
                "Train custom product embeddings",
                "Integrate with vector database",
                "Deploy and monitor in production"
            ],
            real_world_application=True,
            mentors_available=["Senior ML Engineer Alice"]
        ),
        Project(
            name="Scale Vector Database to 10B Embeddings",
            description="Architect and deploy distributed vector system",
            track=LearningTrack.INFRASTRUCTURE,
            difficulty=CompetencyLevel.PROFICIENT,
            duration_weeks=6,
            learning_outcomes=[
                "Design distributed architecture",
                "Implement sharding strategy",
                "Optimize query performance"
            ],
            real_world_application=True,
            mentors_available=["Staff SRE Bob"]
        )
    ]

    for project in projects:
        program.add_project(project)

    # Add sample learners
    learners = [
        Learner(
            name="Junior ML Engineer Dan",
            role="ML Engineer",
            primary_track=LearningTrack.ML_ENGINEER,
            current_level=CompetencyLevel.NOVICE,
            learning_goals=[
                "Master contrastive learning",
                "Deploy first production embedding system"
            ]
        ),
        Learner(
            name="Senior Backend Engineer Eve",
            role="Backend Engineer",
            primary_track=LearningTrack.INFRASTRUCTURE,
            current_level=CompetencyLevel.COMPETENT,
            completed_modules=["Vector Database Fundamentals"],
            learning_goals=[
                "Design trillion-scale systems",
                "Become vector database expert"
            ]
        ),
        Learner(
            name="Product Manager Frank",
            role="Product Manager",
            primary_track=LearningTrack.PRODUCT_MANAGER,
            current_level=CompetencyLevel.NOVICE,
            learning_goals=[
                "Identify embedding opportunities",
                "Launch embedding-powered feature"
            ]
        )
    ]

    for learner in learners:
        program.add_learner(learner)

    # Assign mentors
    program.assign_mentor("Junior ML Engineer Dan", "Senior ML Engineer Alice")
    program.assign_mentor("Senior Backend Engineer Eve", "Staff SRE Bob")

    # Display curriculum
    print("=== ML Engineer Curriculum ===\n")
    print(program.generate_curriculum(LearningTrack.ML_ENGINEER))

    # Track progress
    print("\n=== Learner Progress ===\n")
    for learner_name in program.learners:
        progress = program.track_progress(learner_name)
        print(f"{learner_name}:")
        print(f"  Level: {progress['current_level']}")
        print(f"  Progress: {progress['modules_completed']}/{progress['total_modules']} modules ({progress['completion_percentage']:.1f}%)")
        print(f"  Ready for advancement: {progress['ready_for_advancement']}")
        if progress['recommendations']:
            print("  Next recommended modules:")
            for rec in progress['recommendations']:
                print(f"    - {rec['module']} ({rec['reasoning']})")
        print()

    # Display dashboard
    print("=== Program Dashboard ===\n")
    print(program.create_program_dashboard())

if __name__ == "__main__":
    create_enterprise_training_program()
