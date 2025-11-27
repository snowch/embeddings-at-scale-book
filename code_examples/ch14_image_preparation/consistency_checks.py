"""Embedding consistency and quality checks."""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class ConsistencyResult:
    """Result of consistency check."""

    is_consistent: bool
    score: float
    details: dict = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Quality metrics for an embedding."""

    norm: float
    variance: float
    sparsity: float
    is_valid: bool
    issues: list[str] = field(default_factory=list)


class EmbeddingConsistencyChecker:
    """Check embedding quality and consistency."""

    def __init__(
        self,
        expected_dim: int = 768,
        min_norm: float = 0.1,
        max_norm: float = 100.0,
        min_variance: float = 1e-6,
        max_sparsity: float = 0.95,
    ):
        """
        Initialize consistency checker.

        Args:
            expected_dim: Expected embedding dimension
            min_norm: Minimum acceptable norm
            max_norm: Maximum acceptable norm
            min_variance: Minimum variance threshold
            max_sparsity: Maximum allowed sparsity (fraction of zeros)
        """
        self.expected_dim = expected_dim
        self.min_norm = min_norm
        self.max_norm = max_norm
        self.min_variance = min_variance
        self.max_sparsity = max_sparsity

    def check_embedding(self, embedding: np.ndarray) -> QualityMetrics:
        """
        Check quality of a single embedding.

        Args:
            embedding: Embedding vector

        Returns:
            QualityMetrics with validation results
        """
        issues = []

        # Check dimension
        if len(embedding.shape) != 1:
            issues.append(f"Expected 1D array, got shape {embedding.shape}")
            embedding = embedding.flatten()

        if embedding.shape[0] != self.expected_dim:
            issues.append(
                f"Dimension mismatch: expected {self.expected_dim}, got {embedding.shape[0]}"
            )

        # Calculate metrics
        norm = float(np.linalg.norm(embedding))
        variance = float(np.var(embedding))
        sparsity = float(np.mean(np.abs(embedding) < 1e-8))

        # Check norm
        if norm < self.min_norm:
            issues.append(f"Norm too small: {norm:.4f} < {self.min_norm}")
        elif norm > self.max_norm:
            issues.append(f"Norm too large: {norm:.4f} > {self.max_norm}")

        # Check variance
        if variance < self.min_variance:
            issues.append(f"Variance too low: {variance:.2e} (possible constant vector)")

        # Check sparsity
        if sparsity > self.max_sparsity:
            issues.append(f"Too sparse: {sparsity:.2%} zeros")

        # Check for NaN/Inf
        if np.any(np.isnan(embedding)):
            issues.append("Contains NaN values")
        if np.any(np.isinf(embedding)):
            issues.append("Contains Inf values")

        return QualityMetrics(
            norm=norm,
            variance=variance,
            sparsity=sparsity,
            is_valid=len(issues) == 0,
            issues=issues,
        )

    def check_consistency(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        similarity_threshold: float = 0.7,
    ) -> ConsistencyResult:
        """
        Check consistency between two embeddings of same/similar images.

        Args:
            embedding1: First embedding
            embedding2: Second embedding
            similarity_threshold: Expected minimum similarity

        Returns:
            ConsistencyResult with comparison details
        """
        # Cosine similarity
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return ConsistencyResult(
                is_consistent=False,
                score=0.0,
                details={"error": "Zero norm embedding"},
            )

        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Euclidean distance
        distance = np.linalg.norm(embedding1 - embedding2)

        is_consistent = similarity >= similarity_threshold

        return ConsistencyResult(
            is_consistent=is_consistent,
            score=float(similarity),
            details={
                "cosine_similarity": float(similarity),
                "euclidean_distance": float(distance),
                "norm_ratio": float(max(norm1, norm2) / min(norm1, norm2)),
            },
        )

    def check_augmentation_invariance(
        self,
        encoder,
        image: Image.Image,
        augmentations: list,
        expected_similarity: float = 0.8,
    ) -> dict:
        """
        Check if embeddings are stable under augmentations.

        Args:
            encoder: Embedding model
            image: Original image
            augmentations: List of augmentation functions
            expected_similarity: Expected minimum similarity

        Returns:
            Dict with invariance metrics
        """
        # Get original embedding
        original_emb = encoder.encode(image)

        results = []
        for aug_fn in augmentations:
            augmented = aug_fn(image)
            aug_emb = encoder.encode(augmented)

            consistency = self.check_consistency(original_emb, aug_emb, expected_similarity)

            results.append(
                {
                    "augmentation": aug_fn.__name__,
                    "similarity": consistency.score,
                    "is_consistent": consistency.is_consistent,
                }
            )

        avg_similarity = np.mean([r["similarity"] for r in results])
        all_consistent = all(r["is_consistent"] for r in results)

        return {
            "average_similarity": float(avg_similarity),
            "all_consistent": all_consistent,
            "individual_results": results,
        }


class BatchConsistencyChecker:
    """Check consistency across batches of embeddings."""

    def __init__(self, checker: EmbeddingConsistencyChecker | None = None):
        """
        Initialize batch checker.

        Args:
            checker: Single embedding checker
        """
        self.checker = checker or EmbeddingConsistencyChecker()

    def check_batch(self, embeddings: np.ndarray) -> dict:
        """
        Check quality of a batch of embeddings.

        Args:
            embeddings: Batch of embeddings (N x D)

        Returns:
            Dict with batch quality metrics
        """
        individual_results = []
        valid_count = 0

        for emb in embeddings:
            result = self.checker.check_embedding(emb)
            individual_results.append(result)
            if result.is_valid:
                valid_count += 1

        # Batch statistics
        norms = [r.norm for r in individual_results]
        variances = [r.variance for r in individual_results]

        return {
            "total": len(embeddings),
            "valid": valid_count,
            "invalid": len(embeddings) - valid_count,
            "valid_ratio": valid_count / len(embeddings),
            "norm_stats": {
                "mean": float(np.mean(norms)),
                "std": float(np.std(norms)),
                "min": float(np.min(norms)),
                "max": float(np.max(norms)),
            },
            "variance_stats": {
                "mean": float(np.mean(variances)),
                "std": float(np.std(variances)),
            },
            "invalid_indices": [i for i, r in enumerate(individual_results) if not r.is_valid],
        }

    def check_distribution(self, embeddings: np.ndarray) -> dict:
        """
        Check if embedding distribution is healthy.

        Args:
            embeddings: Batch of embeddings

        Returns:
            Dict with distribution metrics
        """
        # Inter-embedding similarities
        n = len(embeddings)
        similarities = []

        for i in range(min(n, 100)):  # Sample for efficiency
            for j in range(i + 1, min(n, 100)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)

        similarities = np.array(similarities)

        # Check for collapse (all embeddings similar)
        mean_sim = np.mean(similarities)
        is_collapsed = mean_sim > 0.95

        # Check for uniformity (embeddings well distributed)
        is_uniform = 0.1 < mean_sim < 0.5

        return {
            "mean_pairwise_similarity": float(mean_sim),
            "std_pairwise_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "is_collapsed": is_collapsed,
            "is_uniform": is_uniform,
            "health": "good" if is_uniform else ("collapsed" if is_collapsed else "sparse"),
        }
