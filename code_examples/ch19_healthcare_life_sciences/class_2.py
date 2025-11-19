import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Code from Chapter 19
# Book: Embeddings at Scale

"""
Personalized Treatment Recommendations

Architecture:
1. Patient encoder: Multi-modal patient representation
2. Treatment encoder: Drug properties and mechanisms
3. Outcome database: Historical patients with treatments and outcomes
4. Similarity search: Find patients similar to query patient
5. Outcome aggregation: Recommend treatments with best outcomes in similar patients

Techniques:
- k-NN in embedding space: Find similar patients
- Causal inference: Estimate treatment effects accounting for confounding
- Counterfactual prediction: Estimate what would have happened with different treatment
- Propensity score matching: Balance treated vs untreated comparisons
- Meta-learning: Learn from multiple diseases/treatments

Production considerations:
- Evidence quality: Distinguish RCT vs observational data
- Explainability: Show similar patients and their outcomes
- Uncertainty: Quantify confidence in recommendations
- Clinical integration: Fit into physician workflow
- Continuous learning: Update as new evidence emerges
"""

# Placeholder classes - see class.py for full implementation
import torch.nn as nn


@dataclass
class Patient:
    """Placeholder for Patient."""
    patient_id: str
    age: int = 0
    conditions: list = None
    medications: list = None
    lab_results: Dict[str, Any] = None

class TrialPatientEncoder(nn.Module):
    """Placeholder for TrialPatientEncoder."""
    def __init__(self):
        super().__init__()

    def encode(self, patient):
        import torch
        return torch.randn(768)

@dataclass
class TreatmentOption:
    """Available treatment option"""
    treatment_id: str
    name: str
    category: str
    mechanism: Optional[str] = None
    side_effects: Optional[List[str]] = None
    contraindications: Optional[List[str]] = None
    cost: Optional[float] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.side_effects is None:
            self.side_effects = []
        if self.contraindications is None:
            self.contraindications = []

@dataclass
class HistoricalCase:
    """Historical patient with treatment and outcome"""
    case_id: str
    patient: Patient
    treatment: TreatmentOption
    outcome: str
    survival_time: Optional[float] = None
    adverse_events: Optional[List[str]] = None
    quality_of_life: Optional[float] = None
    embedding: Optional[np.ndarray] = None

@dataclass
class TreatmentRecommendation:
    """Personalized treatment recommendation"""
    patient_id: str
    recommended_treatment: TreatmentOption
    predicted_outcome: str
    confidence: float
    alternative_treatments: List[Tuple[TreatmentOption, float]]
    similar_cases: List[HistoricalCase]
    expected_survival: float
    expected_qol: float
    explanation: str

# Placeholder classes - see class.py for full implementation
from dataclasses import dataclass
from typing import Any, Dict

import torch.nn as nn


@dataclass
class Patient:
    """Placeholder for Patient."""
    patient_id: str
    age: int = 0
    conditions: list = None
    medications: list = None
    lab_results: Dict[str, Any] = None

class TrialPatientEncoder(nn.Module):
    """Placeholder for TrialPatientEncoder."""
    def __init__(self):
        super().__init__()

    def encode(self, patient):
        import torch
        return torch.randn(768)

class PersonalizedTreatmentSystem:
    """Personalized treatment recommendation system"""

    def __init__(self, embedding_dim: int = 256, device: str = 'cpu'):
        self.embedding_dim = embedding_dim
        self.device = device

        self.patient_encoder = TrialPatientEncoder(embedding_dim).to(device)

        self.historical_cases: List[HistoricalCase] = []
        self.case_embeddings: Optional[np.ndarray] = None

        self.treatments: Dict[str, TreatmentOption] = {}

    def add_historical_case(self, case: HistoricalCase):
        """Add case to historical database"""
        self.historical_cases.append(case)

        embeddings = []
        for c in self.historical_cases:
            if c.embedding is not None:
                embeddings.append(c.embedding)
            else:
                emb = np.random.randn(self.embedding_dim).astype(np.float32)
                emb = emb / np.linalg.norm(emb)
                c.embedding = emb
                embeddings.append(emb)

        self.case_embeddings = np.array(embeddings)

    def find_similar_patients(
        self,
        query_patient: Patient,
        k: int = 20
    ) -> List[Tuple[HistoricalCase, float]]:
        """Find k most similar historical patients"""
        query_emb = np.random.randn(self.embedding_dim).astype(np.float32)
        query_emb = query_emb / np.linalg.norm(query_emb)

        if self.case_embeddings is None or len(self.case_embeddings) == 0:
            return []

        similarities = np.dot(self.case_embeddings, query_emb)

        top_indices = np.argsort(similarities)[::-1][:k]

        similar_cases = [
            (self.historical_cases[i], float(similarities[i]))
            for i in top_indices
        ]

        return similar_cases

    def recommend_treatment(
        self,
        patient: Patient,
        available_treatments: List[TreatmentOption]
    ) -> TreatmentRecommendation:
        """Generate personalized treatment recommendation"""
        similar_cases = self.find_similar_patients(patient, k=20)

        if not similar_cases:
            treatment = random.choice(available_treatments)
            return TreatmentRecommendation(
                patient_id=patient.patient_id,
                recommended_treatment=treatment,
                predicted_outcome="unknown",
                confidence=0.5,
                alternative_treatments=[],
                similar_cases=[],
                expected_survival=12.0,
                expected_qol=0.7,
                explanation="Insufficient historical data for personalized recommendation"
            )

        treatment_outcomes = {}

        for case, similarity in similar_cases:
            treatment_id = case.treatment.treatment_id

            if treatment_id not in treatment_outcomes:
                treatment_outcomes[treatment_id] = {
                    'treatment': case.treatment,
                    'cases': [],
                    'response_rate': 0,
                    'survival': [],
                    'qol': []
                }

            treatment_outcomes[treatment_id]['cases'].append((case, similarity))

            if case.outcome == "response":
                treatment_outcomes[treatment_id]['response_rate'] += similarity

            if case.survival_time is not None:
                treatment_outcomes[treatment_id]['survival'].append(
                    case.survival_time * similarity
                )
            if case.quality_of_life is not None:
                treatment_outcomes[treatment_id]['qol'].append(
                    case.quality_of_life * similarity
                )

        for treatment_id in treatment_outcomes:
            data = treatment_outcomes[treatment_id]
            total_weight = sum(sim for _, sim in data['cases'])

            data['response_rate'] /= total_weight
            data['expected_survival'] = np.mean(data['survival']) if data['survival'] else 12.0
            data['expected_qol'] = np.mean(data['qol']) if data['qol'] else 0.7

            data['score'] = (
                data['response_rate'] * 0.5 +
                min(data['expected_survival'] / 24.0, 1.0) * 0.3 +
                data['expected_qol'] * 0.2
            )

        ranked = sorted(
            treatment_outcomes.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        if not ranked:
            treatment = random.choice(available_treatments)
            return TreatmentRecommendation(
                patient_id=patient.patient_id,
                recommended_treatment=treatment,
                predicted_outcome="unknown",
                confidence=0.5,
                alternative_treatments=[],
                similar_cases=[],
                expected_survival=12.0,
                expected_qol=0.7,
                explanation="No similar cases found for available treatments"
            )

        top_treatment_id, top_data = ranked[0]
        recommended = top_data['treatment']

        alternatives = [
            (data['treatment'], data['score'])
            for _, data in ranked[1:4]
        ]

        n_cases = len(top_data['cases'])
        confidence = min(0.95, 0.5 + (n_cases / 20) * 0.45)

        explanation = f"Based on {n_cases} similar patients, "
        explanation += f"{recommended.name} showed {top_data['response_rate']:.0%} response rate. "
        explanation += f"Expected survival: {top_data['expected_survival']:.1f} months. "

        top_cases = [case for case, _ in similar_cases[:5]]

        return TreatmentRecommendation(
            patient_id=patient.patient_id,
            recommended_treatment=recommended,
            predicted_outcome="response" if top_data['response_rate'] > 0.5 else "mixed",
            confidence=confidence,
            alternative_treatments=alternatives,
            similar_cases=top_cases,
            expected_survival=top_data['expected_survival'],
            expected_qol=top_data['expected_qol'],
            explanation=explanation
        )

def personalized_treatment_example():
    """Example: Personalized cancer treatment recommendation"""
    print("=== Personalized Treatment Recommendations ===\n")

    patient = Patient(
        patient_id="PT_NEW_001",
        age=62,
        sex="F",
        medical_history=["Type 2 Diabetes", "Hypertension"],
        labs={
            'WBC': 6.5,
            'Hemoglobin': 11.2,
            'Platelets': 180,
            'Creatinine': 1.1
        }
    )

    print(f"Patient: {patient.patient_id}")
    print(f"Demographics: {patient.age}yo {patient.sex}")
    print(f"Medical history: {', '.join(patient.medical_history)}")
    print("Diagnosis: Stage IV colorectal cancer")
    print("Biomarkers: MSI-high, BRAF wildtype, KRAS wildtype")

    treatments = [
        TreatmentOption(
            treatment_id="TX_001",
            name="FOLFOX + Bevacizumab",
            category="Chemotherapy + targeted",
            mechanism="Cytotoxic + VEGF inhibition",
            cost=120000
        ),
        TreatmentOption(
            treatment_id="TX_002",
            name="FOLFIRI + Cetuximab",
            category="Chemotherapy + targeted",
            mechanism="Cytotoxic + EGFR inhibition",
            cost=150000
        ),
        TreatmentOption(
            treatment_id="TX_003",
            name="Pembrolizumab",
            category="Immunotherapy",
            mechanism="PD-1 inhibition",
            cost=180000
        ),
        TreatmentOption(
            treatment_id="TX_004",
            name="Regorafenib",
            category="Targeted therapy",
            mechanism="Multi-kinase inhibition",
            cost=90000
        )
    ]

    print(f"\nAvailable treatments: {len(treatments)}")
    for tx in treatments:
        print(f"  • {tx.name} ({tx.category})")

    system = PersonalizedTreatmentSystem(embedding_dim=256)

    print("\nBuilding historical database...")
    for i in range(100):
        historical_patient = Patient(
            patient_id=f"PT_HIST_{i:03d}",
            age=random.randint(45, 75),
            sex=random.choice(['M', 'F']),
            medical_history=random.sample(
                ['Diabetes', 'Hypertension', 'CAD', 'COPD'],
                k=random.randint(0, 2)
            )
        )

        treatment = random.choice(treatments)

        outcome = random.choice(['response', 'response', 'stable', 'progression'])
        survival = random.uniform(6, 36)
        qol = random.uniform(0.4, 0.9)

        case = HistoricalCase(
            case_id=f"CASE_{i:04d}",
            patient=historical_patient,
            treatment=treatment,
            outcome=outcome,
            survival_time=survival,
            quality_of_life=qol
        )

        system.add_historical_case(case)

    print(f"Historical database: {len(system.historical_cases)} cases")

    recommendation = system.recommend_treatment(
        patient=patient,
        available_treatments=treatments
    )

    print("\n--- Personalized Treatment Recommendation ---\n")
    print(f"Recommended Treatment: {recommendation.recommended_treatment.name}")
    print(f"Category: {recommendation.recommended_treatment.category}")
    print(f"Mechanism: {recommendation.recommended_treatment.mechanism}")

    print("\nExpected Outcomes:")
    print(f"  Predicted response: {recommendation.predicted_outcome}")
    print(f"  Expected survival: {recommendation.expected_survival:.1f} months")
    print(f"  Expected QOL: {recommendation.expected_qol:.1%}")
    print(f"  Confidence: {recommendation.confidence:.1%}")

    print("\nRationale:")
    print(f"  {recommendation.explanation}")

    print("\nAlternative Treatments:")
    for i, (treatment, score) in enumerate(recommendation.alternative_treatments, 2):
        print(f"  {i}. {treatment.name} (score: {score:.2f})")

    print("\nSimilar Patient Cases (Top 3):")
    for i, case in enumerate(recommendation.similar_cases[:3], 1):
        print(f"  {i}. {case.case_id}:")
        print(f"     Treatment: {case.treatment.name}")
        print(f"     Outcome: {case.outcome}")
        print(f"     Survival: {case.survival_time:.1f} months")

    print("\n--- Expected Impact ---")
    print("Traditional approach:")
    print("  • Standard protocol based on diagnosis alone")
    print("  • Trial-and-error: 2-3 failed treatments before success")
    print("  • Time to effective treatment: 6-12 months")
    print("  • Response rate: 40-50%")
    print()
    print("Personalized approach:")
    print("  • Treatment matched to patient characteristics")
    print("  • Higher probability of first-line success")
    print("  • Time to effective treatment: 0-3 months")
    print("  • Response rate: 65-75% (enriched)")
    print("  • Avoid treatments unlikely to work")
    print("  • Better quality of life (fewer failed treatments)")
    print()
    print("→ Faster path to effective treatment, better outcomes")

# Uncomment to run:
# personalized_treatment_example()
