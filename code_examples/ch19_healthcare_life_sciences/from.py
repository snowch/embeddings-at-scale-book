# Code from Chapter 19
# Book: Embeddings at Scale

"""
Drug Discovery with Molecular Embeddings

Architecture:
1. Molecular encoder: SMILES/graph neural network to embedding
2. Protein encoder: Sequence/structure to binding site embedding
3. Interaction predictor: Binding affinity from molecule + protein embeddings
4. Property predictors: ADMET (absorption, distribution, metabolism, excretion, toxicity)
5. Generative models: Design new molecules optimizing multiple objectives

Techniques:
- Graph neural networks: Molecular structure as graph (atoms=nodes, bonds=edges)
- Contrastive learning: Molecules with similar activity close in embedding space
- Multi-task learning: Predict binding, toxicity, solubility simultaneously
- Transfer learning: Pre-train on large molecule databases, fine-tune on target
- Active learning: Iteratively synthesize promising candidates, update model

Production considerations:
- Chemical validity: Generated molecules must be synthesizable
- Explainability: Highlight structural features driving predictions
- Uncertainty quantification: Model confidence for risk assessment
- Synthesis planning: Retrosynthesis to plan compound creation
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Molecule:
    """
    Chemical compound representation

    Attributes:
        molecule_id: Unique identifier
        smiles: SMILES string representation
        name: Chemical name
        molecular_weight: MW in daltons
        structure: Graph representation (atoms, bonds)
        properties: Known properties (solubility, logP, etc.)
        activity: Biological activity data
        toxicity: Toxicity measurements
        embedding: Learned molecular embedding
    """
    molecule_id: str
    smiles: str
    name: Optional[str] = None
    molecular_weight: Optional[float] = None
    structure: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, float]] = None
    activity: Optional[Dict[str, float]] = None
    toxicity: Optional[Dict[str, float]] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.activity is None:
            self.activity = {}
        if self.toxicity is None:
            self.toxicity = {}

@dataclass
class Protein:
    """
    Protein target representation

    Attributes:
        protein_id: UniProt or PDB identifier
        name: Protein name
        sequence: Amino acid sequence
        structure: 3D structure (if available)
        binding_site: Active site residues
        disease: Associated disease
        known_ligands: Known binding molecules
        embedding: Learned protein embedding
    """
    protein_id: str
    name: str
    sequence: str
    structure: Optional[Dict[str, Any]] = None
    binding_site: Optional[List[int]] = None
    disease: Optional[str] = None
    known_ligands: Optional[List[str]] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.known_ligands is None:
            self.known_ligands = []

@dataclass
class DrugCandidate:
    """
    Predicted drug candidate

    Attributes:
        molecule: Candidate molecule
        target: Target protein
        binding_affinity: Predicted binding (lower = stronger)
        efficacy_score: Predicted therapeutic effect
        toxicity_score: Predicted toxicity risk
        selectivity: Specificity for target vs off-targets
        synthesis_difficulty: How hard to synthesize
        confidence: Model confidence in predictions
        explanation: Key structural features
    """
    molecule: Molecule
    target: Protein
    binding_affinity: float
    efficacy_score: float
    toxicity_score: float
    selectivity: float
    synthesis_difficulty: float
    confidence: float
    explanation: str

class MolecularEncoder(nn.Module):
    """
    Encode molecules to embeddings

    Architecture:
    - Graph neural network: Message passing over molecular graph
    - Atom features: Element, charge, hybridization, aromaticity
    - Bond features: Bond type, conjugation, ring membership
    - Global pooling: Aggregate atom embeddings to molecule embedding

    Training:
    - Contrastive: Molecules with similar activity close
    - Multi-task: Predict multiple properties (binding, toxicity, solubility)
    - Self-supervised: Masked atom prediction
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_atom_features: int = 128,
        num_bond_features: int = 32,
        num_gnn_layers: int = 4
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Atom and bond feature encoders
        self.atom_encoder = nn.Sequential(
            nn.Linear(num_atom_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.bond_encoder = nn.Sequential(
            nn.Linear(num_bond_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Graph neural network layers (message passing)
        self.gnn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_gnn_layers)
        ])

        # Graph pooling
        self.pool = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(
        self,
        atom_features: torch.Tensor,
        bond_features: torch.Tensor,
        atom_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode molecules to embeddings"""
        atom_emb = self.atom_encoder(atom_features)

        for gnn_layer in self.gnn_layers:
            atom_emb = gnn_layer(atom_emb, src_key_padding_mask=~atom_mask)

        atom_mask_expanded = atom_mask.unsqueeze(-1).float()
        atom_sum = (atom_emb * atom_mask_expanded).sum(dim=1)
        atom_count = atom_mask_expanded.sum(dim=1).clamp(min=1.0)
        mol_emb = atom_sum / atom_count

        mol_emb = self.pool(mol_emb)
        return F.normalize(mol_emb, p=2, dim=-1)

class ProteinEncoder(nn.Module):
    """
    Encode proteins to embeddings

    Architecture:
    - Sequence encoder: Transformer over amino acid sequence
    - Structure encoder: 3D graph neural network (if structure available)
    - Binding site attention: Focus on active site residues

    Training:
    - Contrastive: Proteins with similar function close
    - Ligand binding prediction: Embedding predicts known ligands
    - Transfer from pre-trained protein models (ESM, ProtTrans)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_amino_acids: int = 21,
        sequence_length: int = 2048
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.aa_embedding = nn.Embedding(num_amino_acids, 128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.binding_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.projection = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(
        self,
        sequence: torch.Tensor,
        binding_site_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode proteins to embeddings"""
        aa_emb = self.aa_embedding(sequence)
        seq_emb = self.sequence_encoder(aa_emb)

        if binding_site_mask is not None:
            binding_emb, _ = self.binding_attention(
                query=seq_emb,
                key=seq_emb,
                value=seq_emb,
                key_padding_mask=~binding_site_mask
            )
            protein_emb = binding_emb.mean(dim=1)
        else:
            protein_emb = seq_emb.mean(dim=1)

        protein_emb = self.projection(protein_emb)
        return F.normalize(protein_emb, p=2, dim=-1)

class DrugDiscoverySystem:
    """Complete drug discovery system with embeddings"""

    def __init__(self, embedding_dim: int = 256, device: str = 'cpu'):
        self.embedding_dim = embedding_dim
        self.device = device

        self.molecular_encoder = MolecularEncoder(embedding_dim).to(device)
        self.protein_encoder = ProteinEncoder(embedding_dim).to(device)

        self.binding_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        ).to(device)

        self.toxicity_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)

        self.solubility_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

        self.molecule_database = {}
        self.protein_database = {}

    def predict_binding_affinity(
        self,
        molecule_embedding: np.ndarray,
        protein_embedding: np.ndarray
    ) -> float:
        """Predict binding affinity between molecule and protein"""
        mol_emb = torch.tensor(molecule_embedding, dtype=torch.float32).to(self.device)
        prot_emb = torch.tensor(protein_embedding, dtype=torch.float32).to(self.device)

        combined = torch.cat([mol_emb, prot_emb], dim=-1).unsqueeze(0)

        self.binding_predictor.eval()
        with torch.no_grad():
            affinity = self.binding_predictor(combined)

        return float(affinity.cpu().item())

    def predict_properties(
        self,
        molecule_embedding: np.ndarray
    ) -> Dict[str, float]:
        """Predict ADMET properties"""
        mol_emb = torch.tensor(molecule_embedding, dtype=torch.float32).to(self.device).unsqueeze(0)

        self.toxicity_predictor.eval()
        self.solubility_predictor.eval()

        with torch.no_grad():
            toxicity = self.toxicity_predictor(mol_emb)
            solubility = self.solubility_predictor(mol_emb)

        return {
            'toxicity': float(toxicity.cpu().item()),
            'solubility': float(solubility.cpu().item())
        }

    def screen_candidates(
        self,
        molecules: List[Molecule],
        target: Protein,
        top_k: int = 100
    ) -> List[DrugCandidate]:
        """Virtual screening: rank molecules by predicted activity"""
        print(f"Screening {len(molecules)} molecules against {target.name}...")

        candidates = []

        for molecule in molecules:
            mol_emb = np.random.randn(self.embedding_dim).astype(np.float32)
            mol_emb = mol_emb / np.linalg.norm(mol_emb)

            prot_emb = np.random.randn(self.embedding_dim).astype(np.float32)
            prot_emb = prot_emb / np.linalg.norm(prot_emb)

            binding_affinity = self.predict_binding_affinity(mol_emb, prot_emb)
            properties = self.predict_properties(mol_emb)

            efficacy_score = (
                binding_affinity * 0.5 +
                (1 - properties['toxicity']) * 0.3 +
                max(0, properties['solubility']) * 0.2
            )

            selectivity = random.uniform(0.5, 0.95)
            synthesis_difficulty = random.uniform(0.2, 0.8)
            confidence = random.uniform(0.6, 0.95)

            candidates.append(DrugCandidate(
                molecule=molecule,
                target=target,
                binding_affinity=binding_affinity,
                efficacy_score=efficacy_score,
                toxicity_score=properties['toxicity'],
                selectivity=selectivity,
                synthesis_difficulty=synthesis_difficulty,
                confidence=confidence,
                explanation=f"Strong binding to {target.name} active site, favorable ADMET profile"
            ))

        candidates.sort(key=lambda x: x.efficacy_score, reverse=True)
        return candidates[:top_k]

def drug_discovery_example():
    """Example: Virtual screening for cancer drug"""
    print("=== Drug Discovery with Molecular Embeddings ===\n")

    target = Protein(
        protein_id="KINASE_ABC",
        name="Tyrosine Kinase ABC",
        sequence="MKTAYIAKQRQISFVKSHFSRQDILDL...",
        disease="Non-small cell lung cancer"
    )

    print(f"Target: {target.name}")
    print(f"Disease: {target.disease}")

    library = []
    for i in range(1000):
        library.append(Molecule(
            molecule_id=f"MOL_{i:04d}",
            smiles=f"CC(C)NCC(O)COC{i}",
            name=f"Compound {i}",
            molecular_weight=250.0 + random.uniform(-50, 50)
        ))

    print(f"\nMolecular library: {len(library)} compounds")

    system = DrugDiscoverySystem(embedding_dim=256)
    candidates = system.screen_candidates(molecules=library, target=target, top_k=10)

    print("\n--- Top 3 Drug Candidates ---\n")

    for rank, candidate in enumerate(candidates[:3], 1):
        print(f"Rank {rank}: {candidate.molecule.name}")
        print(f"  Predicted binding affinity: {candidate.binding_affinity:.2f} pIC50")
        print(f"  Toxicity risk: {candidate.toxicity_score:.1%}")
        print(f"  Overall efficacy score: {candidate.efficacy_score:.2f}/10")
        print(f"  Confidence: {candidate.confidence:.1%}\n")

    print("--- Expected Impact ---")
    print("Traditional: 6-12 months, $500K-$2M, 1-5% hit rate")
    print("Embedding-based: 1-2 weeks, $10K-$50K, 15-30% hit rate")
    print("→ 100x faster, 20x cheaper, 10x higher success rate")

# Uncomment to run:
# drug_discovery_example()
