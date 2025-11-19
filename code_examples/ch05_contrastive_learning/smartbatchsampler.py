# Code from Chapter 05
# Book: Embeddings at Scale


class SmartBatchSampler:
    """
    Intelligent batch composition for contrastive learning

    Strategies:
    1. Diversity maximization: ensure batches contain diverse examples
    2. Difficulty balancing: mix easy and hard examples
    3. Domain balancing: mix examples from different domains/topics
    """

    def __init__(self, dataset, batch_size=512):
        self.dataset = dataset
        self.batch_size = batch_size

        # Pre-compute embeddings or clusters for sampling
        self.example_embeddings = None
        self.example_clusters = None

    def diversity_maximizing_batch(self, embeddings, batch_size):
        """
        Sample batch that maximizes embedding diversity

        Ensures batch contains diverse examples → better negatives

        Args:
            embeddings: (num_examples, dim) pre-computed embeddings
            batch_size: Desired batch size

        Returns:
            indices: Selected example indices
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        num_examples = len(embeddings)

        # Greedy diversity sampling
        # 1. Start with random example
        selected_indices = [np.random.randint(0, num_examples)]
        remaining_indices = list(range(num_examples))
        remaining_indices.remove(selected_indices[0])

        # 2. Iteratively add example most dissimilar to selected set
        for _ in range(batch_size - 1):
            if not remaining_indices:
                break

            # Compute minimum similarity to any selected example
            min_similarities = []

            for idx in remaining_indices:
                # Similarities to all selected examples
                sims = cosine_similarity(embeddings[idx : idx + 1], embeddings[selected_indices])[0]

                # Minimum similarity (most dissimilar from closest)
                min_sim = sims.min()
                min_similarities.append(min_sim)

            # Choose example with minimum similarity (most diverse)
            most_diverse_idx = np.argmin(min_similarities)
            selected_idx = remaining_indices[most_diverse_idx]

            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)

        return selected_indices

    def difficulty_balanced_batch(
        self, difficulty_scores, batch_size, easy_ratio=0.3, medium_ratio=0.5
    ):
        """
        Sample batch with balanced difficulty

        Mix of easy, medium, and hard examples ensures stable training

        Args:
            difficulty_scores: (num_examples,) difficulty score per example
            batch_size: Desired batch size
            easy_ratio: Proportion of easy examples
            medium_ratio: Proportion of medium examples

        Returns:
            indices: Selected example indices
        """
        import numpy as np

        # Sort by difficulty
        sorted_indices = np.argsort(difficulty_scores)

        # Split into easy, medium, hard
        num_examples = len(difficulty_scores)
        split1 = num_examples // 3
        split2 = 2 * num_examples // 3

        easy_indices = sorted_indices[:split1]
        medium_indices = sorted_indices[split1:split2]
        hard_indices = sorted_indices[split2:]

        # Sample proportionally
        num_easy = int(batch_size * easy_ratio)
        num_medium = int(batch_size * medium_ratio)
        num_hard = batch_size - num_easy - num_medium

        selected = []
        selected.extend(np.random.choice(easy_indices, num_easy, replace=False))
        selected.extend(np.random.choice(medium_indices, num_medium, replace=False))
        selected.extend(np.random.choice(hard_indices, num_hard, replace=False))

        # Shuffle
        np.random.shuffle(selected)

        return selected

    def domain_balanced_batch(self, domain_labels, batch_size):
        """
        Sample batch with balanced domain representation

        Ensures model learns from all domains, not just dominant ones

        Args:
            domain_labels: (num_examples,) domain ID for each example
            batch_size: Desired batch size

        Returns:
            indices: Selected example indices
        """
        from collections import Counter

        import numpy as np

        # Count examples per domain
        domain_counts = Counter(domain_labels)
        num_domains = len(domain_counts)

        # Samples per domain (balanced)
        samples_per_domain = batch_size // num_domains

        # Sample from each domain
        selected = []

        for domain_id in domain_counts:
            # Indices for this domain
            domain_indices = np.where(domain_labels == domain_id)[0]

            # Sample
            num_samples = min(samples_per_domain, len(domain_indices))
            sampled = np.random.choice(domain_indices, num_samples, replace=False)

            selected.extend(sampled)

        # Fill remaining slots randomly
        if len(selected) < batch_size:
            remaining = batch_size - len(selected)
            all_indices = set(range(len(domain_labels)))
            available = list(all_indices - set(selected))

            additional = np.random.choice(available, remaining, replace=False)
            selected.extend(additional)

        # Shuffle
        np.random.shuffle(selected)

        return selected[:batch_size]
