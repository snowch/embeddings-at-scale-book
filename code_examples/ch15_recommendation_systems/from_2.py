# Code from Chapter 15
# Book: Embeddings at Scale

"""
Diversity and Fairness in Recommendations

Techniques:
1. Diversity constraints: Ensure recommendations span categories
2. MMR (Maximal Marginal Relevance): Balance relevance and novelty
3. Calibration: Match recommendation distribution to user preferences
4. Fairness metrics: Monitor exposure distribution across items

Metrics:
- Intra-list diversity: Average pairwise distance in recommendation list
- Coverage: % of items recommended at least once
- Gini coefficient: Exposure inequality (0=perfect equality, 1=max inequality)
- Calibration: KL divergence between user prefs and recommendations
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

class DiversityOptimizer:
    """
    Optimize recommendation diversity

    Strategies:
    1. MMR (Maximal Marginal Relevance): Iteratively select items that balance
       relevance and diversity
    2. Category constraints: Ensure min/max items per category
    3. Similarity penalty: Penalize items similar to already-selected
    4. Exploration bonus: Boost under-explored items
    """

    def __init__(
        self,
        lambda_diversity: float = 0.3,
        category_constraints: Optional[Dict[str, Tuple[int, int]]] = None
    ):
        """
        Args:
            lambda_diversity: Weight for diversity vs relevance (0=pure relevance, 1=pure diversity)
            category_constraints: Min/max items per category {category: (min, max)}
        """
        self.lambda_diversity = lambda_diversity
        self.category_constraints = category_constraints or {}

        print(f"Initialized Diversity Optimizer")
        print(f"  Lambda diversity: {lambda_diversity}")

    def mmr_rerank(
        self,
        candidates: List[Tuple[str, float]],
        item_embeddings: Dict[str, np.ndarray],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Rerank using Maximal Marginal Relevance

        MMR formula:
        score(item) = λ * relevance(item) - (1-λ) * max_similarity(item, selected_items)

        Iteratively select item with highest MMR score

        Args:
            candidates: Candidate items with relevance scores [(item_id, score), ...]
            item_embeddings: Item embeddings {item_id: embedding}
            top_k: Number of items to select

        Returns:
            Diversified top-k items
        """
        selected = []
        selected_embs = []
        remaining = list(candidates)

        while len(selected) < top_k and remaining:
            best_item = None
            best_score = -float('inf')
            best_idx = -1

            for idx, (item_id, relevance) in enumerate(remaining):
                if item_id not in item_embeddings:
                    continue

                item_emb = item_embeddings[item_id]

                # Compute diversity term (max similarity to selected items)
                diversity_penalty = 0.0
                if selected_embs:
                    similarities = [np.dot(item_emb, sel_emb) for sel_emb in selected_embs]
                    diversity_penalty = max(similarities)

                # MMR score
                mmr_score = (self.lambda_diversity * relevance -
                           (1 - self.lambda_diversity) * diversity_penalty)

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = (item_id, relevance)
                    best_idx = idx

            if best_item is None:
                break

            # Add to selected
            selected.append(best_item)
            selected_embs.append(item_embeddings[best_item[0]])

            # Remove from remaining
            remaining.pop(best_idx)

        return selected

    def calibrated_rerank(
        self,
        candidates: List[Tuple[str, float]],
        item_categories: Dict[str, str],
        user_category_preferences: Dict[str, float],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Calibrated recommendations: Match category distribution to user preferences

        Example:
        - User watches 70% action, 20% comedy, 10% drama
        - Recommendations should reflect this distribution

        Args:
            candidates: Candidate items with scores
            item_categories: Category for each item {item_id: category}
            user_category_preferences: User's category distribution {category: proportion}
            top_k: Number of items to select

        Returns:
            Calibrated top-k items
        """
        # Compute target counts per category
        target_counts = {
            category: int(proportion * top_k)
            for category, proportion in user_category_preferences.items()
        }

        # Ensure sum equals top_k
        total = sum(target_counts.values())
        if total < top_k:
            # Add remainder to largest category
            max_category = max(user_category_preferences, key=user_category_preferences.get)
            target_counts[max_category] += (top_k - total)

        # Select items per category
        selected = []
        category_items = defaultdict(list)

        # Group candidates by category
        for item_id, score in candidates:
            if item_id in item_categories:
                category = item_categories[item_id]
                category_items[category].append((item_id, score))

        # Sort items within each category by score
        for category in category_items:
            category_items[category].sort(key=lambda x: x[1], reverse=True)

        # Select from each category according to target counts
        for category, target_count in target_counts.items():
            if category in category_items:
                selected.extend(category_items[category][:target_count])

        # If still need items, add highest-scoring remaining
        if len(selected) < top_k:
            all_selected_ids = {item_id for item_id, _ in selected}
            remaining = [(item_id, score) for item_id, score in candidates
                        if item_id not in all_selected_ids]
            remaining.sort(key=lambda x: x[1], reverse=True)
            selected.extend(remaining[:top_k - len(selected)])

        return selected[:top_k]

class FairnessMonitor:
    """
    Monitor fairness metrics for recommendation system

    Metrics:
    1. Coverage: % of items recommended at least once
    2. Gini coefficient: Exposure inequality
    3. Category balance: Exposure distribution across categories
    4. Long-tail boost: Recommendations for rare items
    """

    def __init__(self):
        """Initialize fairness monitor"""
        # Track recommendations
        self.item_recommendation_counts: Dict[str, int] = defaultdict(int)
        self.total_recommendations = 0

        print("Initialized Fairness Monitor")

    def record_recommendation(self, item_ids: List[str]):
        """
        Record items that were recommended

        Args:
            item_ids: List of recommended item IDs
        """
        for item_id in item_ids:
            self.item_recommendation_counts[item_id] += 1
        self.total_recommendations += len(item_ids)

    def compute_coverage(self, total_items: int) -> float:
        """
        Compute catalog coverage

        Coverage = (# items recommended) / (total # items)

        Args:
            total_items: Total number of items in catalog

        Returns:
            Coverage ratio [0, 1]
        """
        items_recommended = len(self.item_recommendation_counts)
        coverage = items_recommended / total_items if total_items > 0 else 0.0
        return coverage

    def compute_gini_coefficient(self) -> float:
        """
        Compute Gini coefficient for exposure inequality

        Gini = 0: Perfect equality (all items exposed equally)
        Gini = 1: Perfect inequality (one item gets all exposure)

        Returns:
            Gini coefficient [0, 1]
        """
        if not self.item_recommendation_counts:
            return 0.0

        # Get exposure counts sorted
        exposures = sorted(self.item_recommendation_counts.values())
        n = len(exposures)

        # Compute Gini
        cumsum = np.cumsum(exposures)
        gini = (2 * sum((i + 1) * exp for i, exp in enumerate(exposures)) /
                (n * sum(exposures))) - (n + 1) / n

        return gini

    def get_fairness_report(self, total_items: int) -> Dict:
        """
        Generate comprehensive fairness report

        Args:
            total_items: Total number of items in catalog

        Returns:
            Fairness metrics dictionary
        """
        coverage = self.compute_coverage(total_items)
        gini = self.compute_gini_coefficient()

        # Exposure distribution
        exposures = list(self.item_recommendation_counts.values())

        report = {
            'coverage': coverage,
            'gini_coefficient': gini,
            'total_recommendations': self.total_recommendations,
            'unique_items_recommended': len(self.item_recommendation_counts),
            'mean_exposure': np.mean(exposures) if exposures else 0.0,
            'median_exposure': np.median(exposures) if exposures else 0.0,
            'max_exposure': max(exposures) if exposures else 0,
            'min_exposure': min(exposures) if exposures else 0
        }

        return report

# Example: Diversity optimization
def diversity_fairness_example():
    """
    Optimize recommendation diversity and monitor fairness

    Scenario:
    - E-commerce with 1000 products
    - Optimize for diversity (avoid homogeneous recommendations)
    - Monitor fairness (ensure long-tail exposure)
    """

    # Generate candidate items with relevance scores
    candidates = [(f'item_{i}', np.random.rand()) for i in range(50)]
    candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by relevance

    # Generate item embeddings
    item_embeddings = {
        f'item_{i}': np.random.randn(128).astype(np.float32)
        for i in range(50)
    }
    for item_id in item_embeddings:
        item_embeddings[item_id] /= np.linalg.norm(item_embeddings[item_id])

    # Diversity optimizer
    optimizer = DiversityOptimizer(lambda_diversity=0.3)

    print("=== Pure Relevance (Top 10) ===")
    top_relevance = candidates[:10]
    for item_id, score in top_relevance:
        print(f"{item_id}: {score:.3f}")

    # Compute average pairwise similarity (measure of homogeneity)
    embs = [item_embeddings[item_id] for item_id, _ in top_relevance]
    similarities = []
    for i in range(len(embs)):
        for j in range(i+1, len(embs)):
            sim = np.dot(embs[i], embs[j])
            similarities.append(sim)
    avg_sim = np.mean(similarities)
    print(f"\nAverage pairwise similarity: {avg_sim:.3f}")

    # MMR reranking
    print("\n=== MMR Diversified (Top 10) ===")
    diversified = optimizer.mmr_rerank(candidates, item_embeddings, top_k=10)
    for item_id, score in diversified:
        print(f"{item_id}: {score:.3f}")

    # Compute diversity
    embs_div = [item_embeddings[item_id] for item_id, _ in diversified]
    similarities_div = []
    for i in range(len(embs_div)):
        for j in range(i+1, len(embs_div)):
            sim = np.dot(embs_div[i], embs_div[j])
            similarities_div.append(sim)
    avg_sim_div = np.mean(similarities_div)
    print(f"\nAverage pairwise similarity: {avg_sim_div:.3f}")
    print(f"Diversity improvement: {(avg_sim - avg_sim_div):.3f}")

    # Fairness monitoring
    print("\n=== Fairness Monitoring ===")
    monitor = FairnessMonitor()

    # Simulate recommendations for 100 users
    for _ in range(100):
        # Generate recommendations (biased towards popular items)
        if np.random.rand() < 0.7:
            # Popular items (top 10)
            recs = [f'item_{i}' for i in range(10)]
        else:
            # Random items
            recs = [f'item_{np.random.randint(0, 50)}' for _ in range(10)]

        monitor.record_recommendation(recs)

    # Fairness report
    report = monitor.get_fairness_report(total_items=50)
    print(f"Coverage: {report['coverage']:.2%}")
    print(f"Gini coefficient: {report['gini_coefficient']:.3f}")
    print(f"Mean exposure: {report['mean_exposure']:.1f}")
    print(f"Max exposure: {report['max_exposure']}")

# Uncomment to run:
# diversity_fairness_example()
