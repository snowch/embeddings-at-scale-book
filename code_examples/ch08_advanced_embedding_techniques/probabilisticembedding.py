# Code from Chapter 08
# Book: Embeddings at Scale

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProbabilisticEmbedding(nn.Module):
    """
    Embeddings with uncertainty quantification

    Instead of point vector, output distribution over vectors
    Represented as Gaussian: N(μ, Σ)
    - μ (mean): Expected embedding
    - Σ (covariance): Uncertainty

    Applications:
    - High-stakes decisions (healthcare, finance, autonomous systems)
    - Out-of-distribution detection
    - Active learning (query most uncertain examples)
    - Confidence-aware retrieval
    - Trustworthy AI systems
    """

    def __init__(
        self,
        num_items,
        embedding_dim=256,
        uncertainty_type='diagonal'
    ):
        """
        Args:
            num_items: Number of items to embed
            embedding_dim: Dimension of embeddings
            uncertainty_type: 'diagonal' (efficient) or 'full' (expressive)
        """
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.uncertainty_type = uncertainty_type

        # Mean embeddings (expected value)
        self.mean_embeddings = nn.Parameter(
            torch.randn(num_items, embedding_dim) * 0.01
        )

        # Uncertainty embeddings (variance)
        if uncertainty_type == 'diagonal':
            # Diagonal covariance (independent dimensions)
            self.log_var_embeddings = nn.Parameter(
                torch.zeros(num_items, embedding_dim)  # Start with low uncertainty
            )
        elif uncertainty_type == 'full':
            # Full covariance matrix (correlated dimensions)
            # Parameterize as L @ L^T where L is lower triangular
            self.cholesky_embeddings = nn.Parameter(
                torch.randn(num_items, embedding_dim, embedding_dim) * 0.01
            )

    def forward(self, indices, return_uncertainty=True):
        """
        Get probabilistic embeddings

        Args:
            indices: Item indices (batch_size,)
            return_uncertainty: If True, return (mean, variance)

        Returns:
            If return_uncertainty=True: (mean, variance)
            If return_uncertainty=False: sampled embedding
        """
        mean = self.mean_embeddings[indices]

        if not return_uncertainty:
            # Sample from distribution
            if self.uncertainty_type == 'diagonal':
                log_var = self.log_var_embeddings[indices]
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(mean)
                return mean + eps * std
            else:
                # Full covariance sampling
                L = self.cholesky_embeddings[indices]
                eps = torch.randn_like(mean).unsqueeze(-1)
                sample = mean + (L @ eps).squeeze(-1)
                return sample
        else:
            # Return mean and variance
            if self.uncertainty_type == 'diagonal':
                variance = torch.exp(self.log_var_embeddings[indices])
                return mean, variance
            else:
                # Covariance = L @ L^T
                L = self.cholesky_embeddings[indices]
                covariance = L @ L.transpose(-2, -1)
                return mean, covariance

    def kl_divergence_loss(self, indices):
        """
        KL divergence from prior N(0, I)

        Regularizes uncertainty: prevents model from claiming
        infinite uncertainty to avoid being wrong
        """
        mean = self.mean_embeddings[indices]

        if self.uncertainty_type == 'diagonal':
            log_var = self.log_var_embeddings[indices]

            # KL(N(μ, σ²) || N(0, 1))
            kl = -0.5 * torch.sum(
                1 + log_var - mean.pow(2) - log_var.exp(),
                dim=-1
            )
        else:
            # Full covariance KL divergence
            L = self.cholesky_embeddings[indices]

            # More complex formula for full covariance
            # Simplified approximation here
            trace_term = torch.diagonal(L @ L.transpose(-2, -1), dim1=-2, dim2=-1).sum(-1)
            kl = 0.5 * (
                trace_term + (mean ** 2).sum(-1) - self.embedding_dim
                - torch.logdet(L @ L.transpose(-2, -1))
            )

        return kl.mean()

class UncertaintyAwareSimilarity:
    """
    Similarity search with uncertainty quantification

    Key insight: Similarity between uncertain embeddings should
    account for overlap of their distributions, not just distance
    between means

    Applications:
    - Medical image retrieval: Don't match if uncertain
    - Financial fraud: Flag high-uncertainty transactions
    - Autonomous vehicles: Slow down when perception is uncertain
    """

    def __init__(self, probabilistic_embedding):
        self.embedding_model = probabilistic_embedding

    def expected_similarity(self, idx1, idx2):
        """
        Expected cosine similarity between two distributions

        For distributions p₁ = N(μ₁, Σ₁) and p₂ = N(μ₂, Σ₂):
        E[sim(x₁, x₂)] where x₁ ~ p₁, x₂ ~ p₂
        """
        mean1, var1 = self.embedding_model(idx1)
        mean2, var2 = self.embedding_model(idx2)

        # Monte Carlo estimate: sample and average
        num_samples = 100
        similarities = []

        for _ in range(num_samples):
            sample1 = mean1 + torch.randn_like(mean1) * torch.sqrt(var1)
            sample2 = mean2 + torch.randn_like(mean2) * torch.sqrt(var2)

            sim = F.cosine_similarity(sample1, sample2, dim=-1)
            similarities.append(sim)

        expected_sim = torch.stack(similarities).mean(dim=0)
        uncertainty = torch.stack(similarities).std(dim=0)

        return expected_sim, uncertainty

    def wasserstein_distance(self, idx1, idx2):
        """
        Wasserstein distance between embedding distributions

        More principled than expected similarity:
        - Accounts for both mean and variance differences
        - Symmetric and satisfies triangle inequality
        - Natural notion of "distance between uncertainties"
        """
        mean1, var1 = self.embedding_model(idx1)
        mean2, var2 = self.embedding_model(idx2)

        # For Gaussians, Wasserstein-2 distance has closed form:
        # W²(p₁, p₂) = ||μ₁ - μ₂||² + trace(Σ₁ + Σ₂ - 2(Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2})

        # Simplified for diagonal covariance:
        mean_dist = torch.sum((mean1 - mean2) ** 2, dim=-1)
        var_dist = torch.sum((torch.sqrt(var1) - torch.sqrt(var2)) ** 2, dim=-1)

        wasserstein_dist = torch.sqrt(mean_dist + var_dist)

        return wasserstein_dist

    def uncertainty_aware_search(
        self,
        query_idx,
        candidate_indices,
        confidence_threshold=0.8
    ):
        """
        Retrieve similar items, filtering by confidence

        Only return matches where model is confident
        Better to return no match than wrong match in high-stakes scenarios

        Args:
            query_idx: Query item
            candidate_indices: Candidates to search
            confidence_threshold: Minimum confidence (0-1)

        Returns:
            filtered_results: Only high-confidence matches
        """
        results = []

        for candidate_idx in candidate_indices:
            # Compute expected similarity and uncertainty
            expected_sim, uncertainty = self.expected_similarity(
                query_idx,
                candidate_idx
            )

            # Confidence = 1 - normalized_uncertainty
            confidence = 1 - torch.clamp(uncertainty / expected_sim.abs(), 0, 1)

            if confidence >= confidence_threshold:
                results.append({
                    'candidate': candidate_idx,
                    'similarity': expected_sim.item(),
                    'confidence': confidence.item()
                })

        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)

        return results

class BayesianEmbeddingNetwork(nn.Module):
    """
    Deep learning approach: Bayesian neural network for embeddings

    Uses Monte Carlo Dropout or variational inference
    to estimate epistemic uncertainty (model uncertainty)

    Advantages:
    - Captures model's knowledge gaps
    - Detects out-of-distribution inputs
    - Enables active learning
    """

    def __init__(self, input_dim, embedding_dim=256, dropout_rate=0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, embedding_dim)
        )

        self.dropout_rate = dropout_rate

    def forward(self, x, num_samples=1):
        """
        Forward pass with uncertainty estimation

        Args:
            x: Input features
            num_samples: Number of forward passes for MC dropout

        Returns:
            If num_samples=1: single embedding
            If num_samples>1: (mean, variance) across samples
        """
        if num_samples == 1:
            return self.encoder(x)

        # Monte Carlo Dropout: multiple forward passes with dropout enabled
        self.train()  # Enable dropout even during inference

        samples = []
        for _ in range(num_samples):
            samples.append(self.encoder(x))

        samples = torch.stack(samples)  # (num_samples, batch, dim)

        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)

        return mean, variance

    def detect_ood(self, x, num_samples=20, threshold=0.5):
        """
        Out-of-distribution detection via uncertainty

        High uncertainty → likely OOD → reject or request human review

        Returns:
            is_ood: Boolean mask (True = out of distribution)
            uncertainty_score: Magnitude of uncertainty
        """
        mean, variance = self.forward(x, num_samples=num_samples)

        # Aggregate uncertainty across dimensions
        uncertainty_score = variance.mean(dim=-1)

        # Threshold for OOD detection
        is_ood = uncertainty_score > threshold

        return is_ood, uncertainty_score

# Example: Medical image retrieval with uncertainty
def medical_uncertainty_example():
    """
    Medical imaging scenario:
    - 100K training images (chest X-rays)
    - Need to find similar cases for diagnosis support
    - CRITICAL: Don't suggest matches if uncertain (patient safety)

    Uncertainty quantification:
    - High uncertainty on rare diseases → defer to specialist
    - High uncertainty on poor quality images → request better scan
    - Low uncertainty on common conditions → auto-suggest similar cases
    """
    # Initialize probabilistic embedding
    model = ProbabilisticEmbedding(
        num_items=100000,  # 100K medical images
        embedding_dim=512,
        uncertainty_type='diagonal'
    )

    # Similarity search
    search = UncertaintyAwareSimilarity(model)

    # Query: Patient X-ray
    query_idx = torch.tensor([42])

    # Candidate similar cases
    candidates = torch.arange(100)

    # Uncertainty-aware retrieval (only high-confidence matches)
    results = search.uncertainty_aware_search(
        query_idx,
        candidates,
        confidence_threshold=0.85  # High threshold for medical applications
    )

    print(f"Found {len(results)} high-confidence matches:")
    for result in results[:5]:
        print(f"  Case {result['candidate']}: "
              f"Similarity = {result['similarity']:.3f}, "
              f"Confidence = {result['confidence']:.3f}")

    if len(results) == 0:
        print("  No confident matches found → Defer to specialist")

# Uncomment to run:
# medical_uncertainty_example()
