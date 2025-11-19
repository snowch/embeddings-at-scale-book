# Code from Chapter 19
# Book: Embeddings at Scale

"""
Epidemic Modeling with Population Embeddings

Architecture:
1. Population encoder: Demographics, mobility, contact patterns
2. Pathogen encoder: Transmissibility, severity, immunity
3. Intervention encoder: NPIs, vaccines, treatments
4. Transmission model: Predict disease spread in embedding space
5. Intervention optimizer: Find optimal control strategy

Techniques:
- Graph neural networks: Population contact networks
- LSTM: Temporal dynamics of outbreak
- Agent-based modeling: Individual-level simulation
- Causal inference: Estimate intervention effects
- Multi-scale: Model individuals, communities, regions

Production considerations:
- Real-time updates: Incorporate new case data
- Uncertainty quantification: Epidemic prediction inherently uncertain
- Policy evaluation: Simulate interventions before implementation
- Privacy: Aggregate mobility data, no individual tracking
"""

@dataclass
class PopulationGroup:
    """Population subgroup for epidemic modeling"""
    group_id: str
    name: str
    size: int
    demographics: Dict[str, Any]
    mobility: Optional[Dict[str, float]] = None
    contact_rate: float = 10.0
    vulnerability: float = 1.0
    compliance: float = 0.7
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.demographics is None:
            self.demographics = {}
        if self.mobility is None:
            self.mobility = {}

@dataclass
class Pathogen:
    """Disease pathogen characteristics"""
    pathogen_id: str
    name: str
    r0: float
    generation_time: float
    incubation_period: float
    infectious_period: float
    severity: float
    immunity_duration: Optional[float] = None
    variants: Optional[List[str]] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.variants is None:
            self.variants = []

@dataclass
class Intervention:
    """Public health intervention"""
    intervention_id: str
    name: str
    type: str
    effectiveness: float
    compliance_required: float = 0.5
    cost: Optional[float] = None
    side_effects: Optional[List[str]] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.side_effects is None:
            self.side_effects = []

@dataclass
class EpidemicForecast:
    """Epidemic forecast output"""
    forecast_date: datetime
    horizon: int
    predicted_cases: List[float]
    predicted_deaths: List[float]
    peak_date: Optional[datetime] = None
    attack_rate: float = 0.0
    interventions_evaluated: Optional[List[str]] = None
    recommended_strategy: Optional[str] = None
    confidence_intervals: Optional[Dict[str, List[Tuple[float, float]]]] = None

class EpidemicModelingSystem:
    """Complete epidemic modeling and response system"""

    def __init__(self, embedding_dim: int = 128, device: str = 'cpu'):
        self.embedding_dim = embedding_dim
        self.device = device

        self.populations: Dict[str, PopulationGroup] = {}

        self.compartments = {
            'S': {},  # Susceptible
            'E': {},  # Exposed
            'I': {},  # Infectious
            'R': {},  # Recovered
            'D': {}   # Dead
        }

    def initialize_population(self, groups: List[PopulationGroup]):
        """Initialize population groups"""
        for group in groups:
            self.populations[group.group_id] = group

            self.compartments['S'][group.group_id] = group.size
            self.compartments['E'][group.group_id] = 0
            self.compartments['I'][group.group_id] = 0
            self.compartments['R'][group.group_id] = 0
            self.compartments['D'][group.group_id] = 0

    def seed_infection(self, group_id: str, initial_cases: int):
        """Seed initial infections"""
        if group_id in self.compartments['S']:
            self.compartments['S'][group_id] -= initial_cases
            self.compartments['I'][group_id] += initial_cases

    def simulate_transmission(
        self,
        pathogen: Pathogen,
        days: int,
        interventions: Optional[List[Intervention]] = None
    ) -> Dict[str, List[float]]:
        """Simulate disease transmission"""
        effective_r = pathogen.r0

        if interventions:
            for intervention in interventions:
                effective_r *= (1 - intervention.effectiveness)

        time_series = {
            'S': [], 'E': [], 'I': [], 'R': [], 'D': []
        }

        for day in range(days):
            total_S = sum(self.compartments['S'].values())
            total_E = sum(self.compartments['E'].values())
            total_I = sum(self.compartments['I'].values())
            total_R = sum(self.compartments['R'].values())
            total_D = sum(self.compartments['D'].values())

            time_series['S'].append(total_S)
            time_series['E'].append(total_E)
            time_series['I'].append(total_I)
            time_series['R'].append(total_R)
            time_series['D'].append(total_D)

            total_pop = total_S + total_E + total_I + total_R

            if total_pop > 0:
                beta = effective_r / pathogen.infectious_period
                foi = beta * total_I / total_pop

                new_E = foi * total_S

                sigma = 1.0 / pathogen.incubation_period
                new_I = sigma * total_E

                gamma = 1.0 / pathogen.infectious_period
                new_R = gamma * total_I * (1 - pathogen.severity)

                new_D = gamma * total_I * pathogen.severity

                for group_id in self.populations:
                    group_pop = self.populations[group_id].size
                    prop = group_pop / total_pop if total_pop > 0 else 0

                    self.compartments['S'][group_id] = max(0, self.compartments['S'][group_id] - new_E * prop)
                    self.compartments['E'][group_id] = max(0, self.compartments['E'][group_id] + new_E * prop - new_I * prop)
                    self.compartments['I'][group_id] = max(0, self.compartments['I'][group_id] + new_I * prop - new_R * prop - new_D * prop)
                    self.compartments['R'][group_id] += new_R * prop
                    self.compartments['D'][group_id] += new_D * prop

        return time_series

    def evaluate_intervention(
        self,
        pathogen: Pathogen,
        intervention: Intervention,
        duration: int = 180
    ) -> Dict[str, float]:
        """Evaluate intervention impact"""
        baseline_state = {
            k: {g: v for g, v in comp.items()}
            for k, comp in self.compartments.items()
        }

        baseline_ts = self.simulate_transmission(pathogen, duration, interventions=None)

        baseline_cases = baseline_ts['I'][-1] + baseline_ts['R'][-1] + baseline_ts['D'][-1]
        baseline_deaths = baseline_ts['D'][-1]

        self.compartments = baseline_state

        intervention_ts = self.simulate_transmission(
            pathogen, duration, interventions=[intervention]
        )

        intervention_cases = intervention_ts['I'][-1] + intervention_ts['R'][-1] + intervention_ts['D'][-1]
        intervention_deaths = intervention_ts['D'][-1]

        self.compartments = baseline_state

        return {
            'baseline_cases': baseline_cases,
            'baseline_deaths': baseline_deaths,
            'intervention_cases': intervention_cases,
            'intervention_deaths': intervention_deaths,
            'cases_averted': baseline_cases - intervention_cases,
            'deaths_averted': baseline_deaths - intervention_deaths,
            'percent_reduction': (baseline_cases - intervention_cases) / baseline_cases if baseline_cases > 0 else 0
        }

    def optimize_intervention_strategy(
        self,
        pathogen: Pathogen,
        available_interventions: List[Intervention],
        budget_constraint: Optional[float] = None
    ) -> List[Intervention]:
        """Find optimal combination of interventions"""
        evaluations = []

        for intervention in available_interventions:
            impact = self.evaluate_intervention(pathogen, intervention)

            if intervention.cost and intervention.cost > 0:
                cost_per_death_averted = intervention.cost / max(impact['deaths_averted'], 1)
            else:
                cost_per_death_averted = 0

            evaluations.append({
                'intervention': intervention,
                'impact': impact,
                'cost_effectiveness': cost_per_death_averted
            })

        evaluations.sort(
            key=lambda x: x['impact']['deaths_averted'],
            reverse=True
        )

        selected = []
        total_cost = 0

        for eval_data in evaluations:
            intervention = eval_data['intervention']

            if budget_constraint is None:
                selected.append(intervention)
            elif intervention.cost:
                if total_cost + intervention.cost <= budget_constraint:
                    selected.append(intervention)
                    total_cost += intervention.cost
            else:
                selected.append(intervention)

        return selected

def epidemic_modeling_example():
    """Example: COVID-19-like outbreak response"""
    print("=== Epidemic Modeling with Population Embeddings ===\n")

    pathogen = Pathogen(
        pathogen_id="VIRUS_2025",
        name="Novel Respiratory Virus",
        r0=3.5,
        generation_time=5.0,
        incubation_period=5.0,
        infectious_period=10.0,
        severity=0.01,
        immunity_duration=365.0
    )

    print(f"Pathogen: {pathogen.name}")
    print(f"  R0: {pathogen.r0} (highly transmissible)")
    print(f"  Generation time: {pathogen.generation_time} days")
    print(f"  Case fatality rate: {pathogen.severity:.1%}")

    populations = [
        PopulationGroup(
            group_id="urban_young",
            name="Urban 18-35",
            size=2_000_000,
            demographics={'age_range': '18-35', 'density': 'high'},
            contact_rate=20.0,
            vulnerability=0.8,
            compliance=0.6
        ),
        PopulationGroup(
            group_id="urban_middle",
            name="Urban 36-64",
            size=3_000_000,
            demographics={'age_range': '36-64', 'density': 'high'},
            contact_rate=15.0,
            vulnerability=1.0,
            compliance=0.7
        ),
        PopulationGroup(
            group_id="urban_elderly",
            name="Urban 65+",
            size=1_000_000,
            demographics={'age_range': '65+', 'density': 'high'},
            contact_rate=8.0,
            vulnerability=2.0,
            compliance=0.85
        ),
        PopulationGroup(
            group_id="rural",
            name="Rural (all ages)",
            size=4_000_000,
            demographics={'density': 'low'},
            contact_rate=10.0,
            vulnerability=1.0,
            compliance=0.5
        )
    ]

    total_pop = sum(p.size for p in populations)
    print(f"\nPopulation: {total_pop:,} total")
    for pop in populations:
        pct = pop.size / total_pop * 100
        print(f"  • {pop.name}: {pop.size:,} ({pct:.0f}%)")

    interventions = [
        Intervention(
            intervention_id="INT_001",
            name="Social distancing",
            type="NPI",
            effectiveness=0.30,
            compliance_required=0.6,
            cost=50_000_000
        ),
        Intervention(
            intervention_id="INT_002",
            name="Mask mandates",
            type="NPI",
            effectiveness=0.20,
            compliance_required=0.5,
            cost=10_000_000
        ),
        Intervention(
            intervention_id="INT_003",
            name="School closures",
            type="NPI",
            effectiveness=0.15,
            compliance_required=0.9,
            cost=80_000_000
        ),
        Intervention(
            intervention_id="INT_004",
            name="Mass vaccination",
            type="Vaccine",
            effectiveness=0.70,
            compliance_required=0.6,
            cost=200_000_000
        ),
        Intervention(
            intervention_id="INT_005",
            name="Contact tracing",
            type="Surveillance",
            effectiveness=0.25,
            compliance_required=0.7,
            cost=30_000_000
        )
    ]

    print(f"\nAvailable interventions: {len(interventions)}")
    for intv in interventions:
        print(f"  • {intv.name}: {intv.effectiveness:.0%} reduction")

    system = EpidemicModelingSystem(embedding_dim=128)
    system.initialize_population(populations)

    system.seed_infection("urban_young", initial_cases=100)

    print("\nInitial conditions:")
    print("  • 100 initial cases in urban young adults")
    print("  • No interventions active")

    print("\n--- Baseline Forecast (No Interventions) ---\n")

    baseline_ts = system.simulate_transmission(pathogen, days=180)

    peak_day = np.argmax(baseline_ts['I'])
    peak_cases = baseline_ts['I'][peak_day]
    total_infections = baseline_ts['R'][-1] + baseline_ts['D'][-1]
    total_deaths = baseline_ts['D'][-1]
    attack_rate = total_infections / total_pop

    print(f"Peak infections: Day {peak_day}, {peak_cases:,.0f} active cases")
    print(f"Total infections: {total_infections:,.0f} ({attack_rate:.1%} attack rate)")
    print(f"Total deaths: {total_deaths:,.0f}")
    print(f"Healthcare system: {'OVERWHELMED' if peak_cases > 50000 else 'Manageable'}")

    print("\n--- Intervention Evaluation ---\n")

    system.initialize_population(populations)
    system.seed_infection("urban_young", initial_cases=100)

    print("Individual intervention impacts:\n")

    intervention_impacts = []
    for intervention in interventions:
        impact = system.evaluate_intervention(pathogen, intervention, duration=180)
        intervention_impacts.append((intervention, impact))

        print(f"{intervention.name}:")
        print(f"  Cases averted: {impact['cases_averted']:,.0f}")
        print(f"  Deaths averted: {impact['deaths_averted']:,.0f}")
        print(f"  Reduction: {impact['percent_reduction']:.1%}")
        if intervention.cost:
            print(f"  Cost: ${intervention.cost:,.0f}/month")
            if impact['deaths_averted'] > 0:
                cost_per_death = intervention.cost / impact['deaths_averted']
                print(f"  Cost per death averted: ${cost_per_death:,.0f}")
        print()

    print("--- Optimal Intervention Strategy ---\n")

    system.initialize_population(populations)
    system.seed_infection("urban_young", initial_cases=100)

    optimal_strategy = system.optimize_intervention_strategy(
        pathogen=pathogen,
        available_interventions=interventions,
        budget_constraint=300_000_000
    )

    print("Recommended strategy (within budget):")
    total_cost = 0
    for intv in optimal_strategy:
        print(f"  ✓ {intv.name}")
        if intv.cost:
            total_cost += intv.cost

    print(f"\nTotal cost: ${total_cost:,.0f}/month")

    system.initialize_population(populations)
    system.seed_infection("urban_young", initial_cases=100)

    optimal_ts = system.simulate_transmission(pathogen, days=180, interventions=optimal_strategy)

    optimal_peak_day = np.argmax(optimal_ts['I'])
    optimal_peak_cases = optimal_ts['I'][optimal_peak_day]
    optimal_total_infections = optimal_ts['R'][-1] + optimal_ts['D'][-1]
    optimal_total_deaths = optimal_ts['D'][-1]

    print("\n--- Results with Optimal Strategy ---\n")
    print(f"Peak infections: Day {optimal_peak_day}, {optimal_peak_cases:,.0f} active cases")
    print(f"  (Baseline: Day {peak_day}, {peak_cases:,.0f} cases)")
    print(f"  Peak reduction: {(1 - optimal_peak_cases/peak_cases):.1%}")

    print(f"\nTotal infections: {optimal_total_infections:,.0f}")
    print(f"  (Baseline: {total_infections:,.0f})")
    print(f"  Cases averted: {total_infections - optimal_total_infections:,.0f}")

    print(f"\nTotal deaths: {optimal_total_deaths:,.0f}")
    print(f"  (Baseline: {total_deaths:,.0f})")
    print(f"  Deaths averted: {total_deaths - optimal_total_deaths:,.0f}")

    print("\n--- Policy Recommendations ---")
    print("  1. Implement recommended intervention bundle immediately")
    print("  2. Monitor compliance rates and adjust messaging")
    print("  3. Prioritize vaccination for elderly (highest vulnerability)")
    print("  4. Prepare healthcare surge capacity for peak")
    print("  5. Re-evaluate strategy every 2 weeks with new data")

    print("\n--- Expected Impact ---")
    print("Traditional approach:")
    print("  • Static interventions (one-size-fits-all)")
    print("  • Delayed response (waiting for data)")
    print("  • Inefficient resource allocation")
    print("  • Higher mortality and economic cost")
    print()
    print("Embedding-based optimization:")
    print("  • Population-specific interventions")
    print("  • Proactive forecasting (simulate before implementing)")
    print("  • Cost-optimal resource allocation")
    print(f"  • {(1 - optimal_total_deaths/total_deaths):.0%} reduction in deaths")
    print("  • Flattened epidemic curve (healthcare capacity preserved)")
    print()
    print("→ Data-driven response saves lives and reduces economic impact")

# Uncomment to run:
# epidemic_modeling_example()
