import torch

# Code from Chapter 05
# Book: Embeddings at Scale
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# Placeholder classes
class SimCLRTextEmbedding(nn.Module):
    """Placeholder for SimCLRTextEmbedding."""
    def __init__(self, base_model='bert-base-uncased', projection_dim=128):
        super().__init__()
        self.base_model = base_model
        self.projection_dim = projection_dim

    def forward(self, x):
        return torch.randn(self.projection_dim)

class ContrastiveDataset(Dataset):
    """Placeholder for ContrastiveDataset."""
    def __init__(self, *args, **kwargs):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.randn(768)

# Placeholder variable
num_epochs = 10


class DistributedContrastiveLearning:
    """
    Distributed contrastive learning across multiple GPUs

    Key challenge: Gathering embeddings from all GPUs for negative mining
    """

    def __init__(self, model, world_size, rank, backend='nccl'):
        """
        Args:
            model: Embedding model
            world_size: Number of GPUs/processes
            rank: Current process rank
            backend: 'nccl' for GPU, 'gloo' for CPU
        """
        self.model = model
        self.world_size = world_size
        self.rank = rank

        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size
            )

        # Wrap model in DistributedDataParallel
        self.model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank
        )

    def gather_embeddings(self, local_embeddings):
        """
        Gather embeddings from all GPUs

        Args:
            local_embeddings: (local_batch_size, dim) on this GPU

        Returns:
            all_embeddings: (world_size × local_batch_size, dim)
            all_gathered on all GPUs
        """
        # Prepare tensors for gathering
        world_size = dist.get_world_size()
        local_batch_size = local_embeddings.shape[0]
        embedding_dim = local_embeddings.shape[1]

        # Create list of tensors to receive gathered embeddings
        gathered_embeddings = [
            torch.zeros_like(local_embeddings)
            for _ in range(world_size)
        ]

        # All-gather: each GPU gets embeddings from all GPUs
        dist.all_gather(gathered_embeddings, local_embeddings)

        # Concatenate all embeddings
        # Shape: (world_size × local_batch_size, embedding_dim)
        all_embeddings = torch.cat(gathered_embeddings, dim=0)

        return all_embeddings

    def compute_distributed_contrastive_loss(self, anchor_emb, positive_emb,
                                            temperature=0.07):
        """
        Compute contrastive loss using embeddings from all GPUs

        Each GPU computes loss for its local batch using negatives from all GPUs

        Args:
            anchor_emb: (local_batch, dim) anchors on this GPU
            positive_emb: (local_batch, dim) positives on this GPU
            temperature: Contrastive temperature

        Returns:
            loss: scalar loss for this GPU
            metrics: dict with metrics
        """
        local_batch_size = anchor_emb.shape[0]

        # Gather embeddings from all GPUs
        # all_anchors: (world_size × local_batch, dim)
        # all_positives: (world_size × local_batch, dim)
        all_anchors = self.gather_embeddings(anchor_emb)
        all_positives = self.gather_embeddings(positive_emb)

        # Total batch size across all GPUs
        global_batch_size = all_anchors.shape[0]

        # Normalize
        all_anchors = F.normalize(all_anchors, dim=1)
        all_positives = F.normalize(all_positives, dim=1)

        # Compute similarity matrix for LOCAL anchors vs ALL embeddings
        # This is memory-efficient: only compute for local batch
        local_anchors_norm = F.normalize(anchor_emb, dim=1)

        # Concatenate all positives and anchors as potential negatives
        all_embeddings = torch.cat([all_anchors, all_positives], dim=0)

        # Similarity: (local_batch, 2 × global_batch)
        similarity_matrix = torch.matmul(
            local_anchors_norm,
            all_embeddings.T
        ) / temperature

        # Create labels: positive for each anchor is at specific global index
        # Local anchor i corresponds to global anchor (rank × local_batch + i)
        # Its positive is at the same global index in all_positives
        global_indices = torch.arange(
            self.rank * local_batch_size,
            (self.rank + 1) * local_batch_size,
            device=anchor_emb.device
        )

        # Positive is in second half of all_embeddings (all_positives)
        labels = global_indices + global_batch_size

        # Cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)

        # Metrics
        with torch.no_grad():
            predictions = similarity_matrix.argmax(dim=1)
            accuracy = (predictions == labels).float().mean()

            # Positive similarities
            positive_sim = similarity_matrix[
                torch.arange(local_batch_size),
                labels
            ].mean()

            # Negative similarities (excluding positive)
            mask = torch.ones_like(similarity_matrix, dtype=torch.bool)
            mask[torch.arange(local_batch_size), labels] = False
            negative_sim = similarity_matrix[mask].mean()

        metrics = {
            'accuracy': accuracy.item(),
            'positive_similarity': positive_sim.item(),
            'negative_similarity': negative_sim.item(),
            'effective_batch_size': global_batch_size
        }

        return loss, metrics


def launch_distributed_training(world_size, backend='nccl'):
    """
    Launch distributed training across multiple GPUs

    Args:
        world_size: Number of GPUs to use
        backend: 'nccl' for GPU, 'gloo' for CPU
    """
    import torch.multiprocessing as mp

    def train_worker(rank, world_size):
        """
        Training function for each GPU
        """
        # Set device
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')

        # Initialize distributed backend
        dist.init_process_group(
            backend=backend,
            init_method='tcp://localhost:23456',
            world_size=world_size,
            rank=rank
        )

        # Create model
        model = SimCLRTextEmbedding(
            base_model='bert-base-uncased',
            projection_dim=128
        ).to(device)

        # Wrap in distributed trainer
        distributed_trainer = DistributedContrastiveLearning(
            model, world_size, rank
        )

        # Create distributed dataset and loader
        dataset = ContrastiveDataset(...)  # Your dataset
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=512,  # Per-GPU batch size
            sampler=sampler,
            num_workers=4
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        # Training loop
        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)  # Important: shuffle differently each epoch

            for batch in dataloader:
                # Encode
                anchor_emb, _ = model(
                    batch['anchor_ids'].to(device),
                    batch['anchor_mask'].to(device)
                )
                positive_emb, _ = model(
                    batch['positive_ids'].to(device),
                    batch['positive_mask'].to(device)
                )

                # Compute distributed loss
                loss, metrics = distributed_trainer.compute_distributed_contrastive_loss(
                    anchor_emb, positive_emb
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if rank == 0:  # Only print from master process
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}, "
                          f"Effective batch: {metrics['effective_batch_size']}")

        # Cleanup
        dist.destroy_process_group()

    # Spawn processes for each GPU
    mp.spawn(
        train_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


# Launch training on 8 GPUs
# launch_distributed_training(world_size=8)
# Effective batch size: 8 GPUs × 512 per-GPU = 4096
