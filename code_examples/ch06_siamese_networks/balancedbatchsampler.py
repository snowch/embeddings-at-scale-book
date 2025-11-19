# Code from Chapter 06
# Book: Embeddings at Scale

import numpy as np
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    """
    Sampler that ensures each batch contains multiple examples per class

    Critical for triplet loss: need positives and negatives in each batch.

    Strategy: Sample P classes, then sample K examples from each class.
    Batch size = P * K
    """

    def __init__(self, labels, n_classes_per_batch=10, n_samples_per_class=5):
        """
        Args:
            labels: List or array of labels for all samples
            n_classes_per_batch: Number of classes per batch (P)
            n_samples_per_class: Number of samples per class (K)
        """
        self.labels = np.array(labels)
        self.n_classes_per_batch = n_classes_per_batch
        self.n_samples_per_class = n_samples_per_class

        # Build index mapping: class_id -> [sample_indices]
        self.class_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

        # Remove classes with too few samples
        self.valid_classes = [
            c for c, indices in self.class_to_indices.items()
            if len(indices) >= self.n_samples_per_class
        ]

        self.batch_size = n_classes_per_batch * n_samples_per_class

    def __iter__(self):
        """Generate batches"""
        # Shuffle classes
        classes = np.random.permutation(self.valid_classes)

        for i in range(0, len(classes), self.n_classes_per_batch):
            batch_classes = classes[i:i + self.n_classes_per_batch]

            batch_indices = []
            for class_id in batch_classes:
                # Sample K examples from this class
                class_indices = self.class_to_indices[class_id]
                sampled = np.random.choice(
                    class_indices,
                    size=self.n_samples_per_class,
                    replace=len(class_indices) < self.n_samples_per_class
                )
                batch_indices.extend(sampled)

            yield batch_indices

    def __len__(self):
        """Number of batches per epoch"""
        return len(self.valid_classes) // self.n_classes_per_batch


# Example usage
def create_triplet_dataloader(dataset, batch_size=50, num_workers=4):
    """
    Create dataloader optimized for triplet loss training

    Args:
        dataset: PyTorch dataset with (data, label) pairs
        batch_size: Total batch size (should be P * K)
        num_workers: Number of data loading workers
    """
    # Extract all labels
    labels = [dataset[i][1] for i in range(len(dataset))]

    # Configure sampler: 10 classes × 5 samples = 50 per batch
    sampler = BalancedBatchSampler(
        labels=labels,
        n_classes_per_batch=10,
        n_samples_per_class=5
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
