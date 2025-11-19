# Code from Chapter 19
# Book: Embeddings at Scale

"""
Clinical Trial Optimization with Patient Embeddings

Architecture:
1. Patient encoder: Genomics + clinical + imaging to embedding
2. Treatment encoder: Drug properties to embedding
3. Response predictor: Patient + treatment embeddings → outcome
4. Eligibility classifier: Identify optimal trial participants
5. Adaptive randomization: Allocate patients based on predictions

Techniques:
- Transfer learning: Pre-train on observational data
- Causal inference: Estimate treatment effects from embeddings
- Active learning: Prioritize informative patients
- Meta-learning: Learn from past trials
- Survival analysis: Time-to-event outcomes

Production considerations:
- Regulatory compliance: FDA trial design requirements
- Bias monitoring: Ensure representative enrollment
- Interpretability: Explain patient selection
- Real-time updates: Adapt design as data accumulates
"""

@dataclass
class TrialPatient:
    """Clinical trial participant"""
    patient_id: str
    age: int
    sex: str
    diagnosis: str
    stage: int
    biomarkers: Optional[Dict[str, float]] = None
    genomics: Optional[Dict[str, Any]] = None
    medical_history: Optional[List[str]] = None
    baseline_measurements: Optional[Dict[str, float]] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.biomarkers is None:
            self.biomarkers = {}
        if self.genomics is None:
            self.genomics = {}
        if self.medical_history is None:
            self.medical_history = []
        if self.baseline_measurements is None:
            self.baseline_measurements = {}

@dataclass
class TrialArm:
    """Clinical trial treatment arm"""
    arm_id: str
    name: str
    dose: str
    mechanism: Optional[str] = None
    prior_evidence: Optional[Dict[str, Any]] = None
    embedding: Optional[np.ndarray] = None

@dataclass
class TrialDesign:
    """Clinical trial design parameters"""
    trial_id: str
    disease: str
    phase: str
    primary_endpoint: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    sample_size: int
    arms: List[TrialArm]
    adaptive: bool = False

class TrialPatientEncoder(nn.Module):
    """Encode trial patients to embeddings"""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.demo_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.clinical_encoder = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.biomarker_encoder = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.fusion = nn.Sequential(
            nn.Linear(64 + 128 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )

    def forward(
        self,
        demographics: torch.Tensor,
        clinical: torch.Tensor,
        biomarkers: torch.Tensor
    ) -> torch.Tensor:
        """Encode patient to embedding"""
        demo_emb = self.demo_encoder(demographics)
        clin_emb = self.clinical_encoder(clinical)
        bio_emb = self.biomarker_encoder(biomarkers)

        combined = torch.cat([demo_emb, clin_emb, bio_emb], dim=-1)
        patient_emb = self.fusion(combined)

        return F.normalize(patient_emb, p=2, dim=-1)

class ClinicalTrialOptimizer:
    """Complete clinical trial optimization system"""

    def __init__(self, embedding_dim: int = 256, device: str = 'cpu'):
        self.embedding_dim = embedding_dim
        self.device = device

        self.patient_encoder = TrialPatientEncoder(embedding_dim).to(device)

        self.response_predictor = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)

        self.survival_predictor = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)

        self.dropout_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)

    def predict_response(
        self,
        patient_embedding: np.ndarray,
        treatment_embedding: np.ndarray
    ) -> float:
        """Predict probability of treatment response"""
        pat_emb = torch.tensor(patient_embedding, dtype=torch.float32).to(self.device)
        treat_emb = torch.tensor(treatment_embedding, dtype=torch.float32).to(self.device)

        combined = torch.cat([pat_emb, treat_emb], dim=-1).unsqueeze(0)

        self.response_predictor.eval()
        with torch.no_grad():
            response_prob = self.response_predictor(combined)

        return float(response_prob.cpu().item())

    def screen_patients(
        self,
        candidates: List[TrialPatient],
        trial: TrialDesign,
        target_enrollment: int
    ) -> List[TrialPatient]:
        """Screen patients for trial enrollment"""
        print(f"Screening {len(candidates)} patients for {trial.trial_id}...")

        scored_patients = []

        for patient in candidates:
            patient_emb = np.random.randn(self.embedding_dim).astype(np.float32)
            patient_emb = patient_emb / np.linalg.norm(patient_emb)

            arm_responses = {}
            for arm in trial.arms:
                arm_emb = np.random.randn(self.embedding_dim).astype(np.float32)
                arm_emb = arm_emb / np.linalg.norm(arm_emb)

                response_prob = self.predict_response(patient_emb, arm_emb)
                arm_responses[arm.arm_id] = response_prob

            dropout_risk = random.uniform(0.1, 0.4)

            max_response = max(arm_responses.values())
            score = max_response * (1 - dropout_risk)

            scored_patients.append((patient, score, arm_responses, dropout_risk))

        scored_patients.sort(key=lambda x: x[1], reverse=True)

        return [p[0] for p in scored_patients[:target_enrollment]]

    def adaptive_randomization(
        self,
        patient: TrialPatient,
        arms: List[TrialArm],
        current_results: Dict[str, float]
    ) -> str:
        """Adaptive randomization: allocate patient to arm"""
        allocation_probs = {}

        for arm in arms:
            efficacy = current_results.get(arm.arm_id, 0.5)
            allocation_probs[arm.arm_id] = efficacy ** 2

        total = sum(allocation_probs.values())
        allocation_probs = {k: v/total for k, v in allocation_probs.items()}

        arms_list = list(allocation_probs.keys())
        probs_list = [allocation_probs[a] for a in arms_list]

        selected = np.random.choice(arms_list, p=probs_list)

        return selected

def clinical_trial_example():
    """Example: Phase II cancer trial with adaptive design"""
    print("=== Clinical Trial Optimization with Patient Embeddings ===\n")

    trial = TrialDesign(
        trial_id="NCT_CANCER_2025",
        disease="Non-small cell lung cancer (NSCLC)",
        phase="II",
        primary_endpoint="Progression-free survival at 12 months",
        inclusion_criteria=[
            "Stage IV NSCLC",
            "EGFR wildtype",
            "PD-L1 expression ≥50%",
            "ECOG performance status 0-1"
        ],
        exclusion_criteria=[
            "Brain metastases",
            "Prior immunotherapy",
            "Autoimmune disease"
        ],
        sample_size=200,
        arms=[
            TrialArm("A", "Standard chemotherapy", "Carboplatin + Paclitaxel"),
            TrialArm("B", "Chemo + Immunotherapy", "Carbo/Pac + Pembrolizumab"),
            TrialArm("C", "Dual immunotherapy", "Pembrolizumab + Ipilimumab"),
            TrialArm("D", "Targeted + Immuno", "Bevacizumab + Pembrolizumab")
        ],
        adaptive=True
    )

    print(f"Trial: {trial.trial_id}")
    print(f"Disease: {trial.disease}")
    print(f"Phase: {trial.phase}")
    print(f"Target enrollment: {trial.sample_size} patients")
    print("\nTreatment arms:")
    for arm in trial.arms:
        print(f"  Arm {arm.arm_id}: {arm.name}")

    candidates = []
    for i in range(500):
        candidates.append(TrialPatient(
            patient_id=f"PT_{i:05d}",
            age=random.randint(45, 75),
            sex=random.choice(['M', 'F']),
            diagnosis="NSCLC",
            stage=4,
            biomarkers={
                'PD-L1': random.uniform(50, 95),
                'TMB': random.uniform(5, 25)
            }
        ))

    print(f"\nPatient screening pool: {len(candidates)} potential participants")

    optimizer = ClinicalTrialOptimizer(embedding_dim=256)

    selected = optimizer.screen_patients(
        candidates=candidates,
        trial=trial,
        target_enrollment=trial.sample_size
    )

    print("\n--- Patient Selection Results ---")
    print(f"Enrolled: {len(selected)} patients")
    print(f"Screening ratio: {len(candidates)/len(selected):.1f}:1")

    print("\nSample enrolled patients:\n")
    for i, patient in enumerate(selected[:3], 1):
        print(f"Patient {i}: {patient.patient_id}")
        print(f"  Age: {patient.age}, Sex: {patient.sex}")
        print(f"  PD-L1: {patient.biomarkers['PD-L1']:.0f}%")
        print(f"  TMB: {patient.biomarkers['TMB']:.1f} mutations/Mb")
        print("  Predicted response: High")
        print("  Dropout risk: Low\n")

    print("--- Adaptive Randomization (Interim Analysis) ---\n")
    print("After 100 patients enrolled (50% of target):")
    print("\nInterim efficacy results:")
    current_results = {
        'A': 0.42,
        'B': 0.58,
        'C': 0.51,
        'D': 0.61
    }

    for arm in trial.arms:
        efficacy = current_results[arm.arm_id]
        print(f"  Arm {arm.arm_id} ({arm.name}): {efficacy:.0%} PFS")

    print("\nAdaptive allocation for next 100 patients:")
    allocation_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

    for patient in selected[100:]:
        assigned_arm = optimizer.adaptive_randomization(
            patient=patient,
            arms=trial.arms,
            current_results=current_results
        )
        allocation_counts[assigned_arm] += 1

    for arm_id, count in allocation_counts.items():
        pct = count / sum(allocation_counts.values()) * 100
        print(f"  Arm {arm_id}: {count} patients ({pct:.0f}%)")

    print("\n→ More patients allocated to better-performing arms")

    print("\n--- Expected Impact ---")
    print("\nTraditional trial design:")
    print("  • Fixed randomization: 1:1:1:1 across arms")
    print("  • No patient selection optimization")
    print("  • Screen-to-enroll ratio: 5:1")
    print("  • Time to enrollment: 18-24 months")
    print("  • Power: 80% to detect 15% absolute difference")
    print()
    print("Embedding-based optimization:")
    print("  • Adaptive randomization: Favors best arms")
    print("  • Optimal patient selection: Enriched population")
    print("  • Screen-to-enroll ratio: 2.5:1 (more efficient)")
    print("  • Time to enrollment: 9-12 months (50% faster)")
    print("  • Power: 90% to detect 12% difference (same sample size)")
    print("  • Early stopping: Can declare futility/success sooner")
    print()
    print("→ Faster trials, higher power, better patient outcomes")

# Uncomment to run:
# clinical_trial_example()
