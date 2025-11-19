import torch

# Code from Chapter 05
# Book: Embeddings at Scale
import torch.utils.data as data


class ContrastiveDataset(data.Dataset):
    """
    Efficient dataset for contrastive learning at scale
    """

    def __init__(self, data_source, augmenter, cache_size=10000):
        """
        Args:
            data_source: Iterator or dataset
            augmenter: TextAugmentation instance
            cache_size: Number of items to cache in memory
        """
        self.data_source = data_source
        self.augmenter = augmenter
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        """
        Return two augmented views of the same item
        """
        # Check cache first
        if idx in self.cache:
            text = self.cache[idx]
        else:
            text = self.data_source[idx]

            # Update cache (LRU-style)
            if len(self.cache) >= self.cache_size:
                # Remove random item
                self.cache.pop(next(iter(self.cache)))
            self.cache[idx] = text

        # Generate two augmented views
        view1 = self.augmenter.augment_simple(text, method='random_deletion')
        view2 = self.augmenter.augment_simple(text, method='synonym_replacement')

        return {
            'view1': view1,
            'view2': view2,
            'original': text,
            'idx': idx
        }


class DistributedContrastiveDataLoader:
    """
    Data loader for distributed contrastive training

    Ensures each GPU sees different augmentations of the same base data
    """

    def __init__(self, dataset, batch_size, world_size, rank):
        """
        Args:
            dataset: ContrastiveDataset
            batch_size: Per-GPU batch size
            world_size: Number of GPUs
            rank: Current GPU rank
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank

        # Sampler ensures each GPU gets different data
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True  # Important for contrastive learning
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)
