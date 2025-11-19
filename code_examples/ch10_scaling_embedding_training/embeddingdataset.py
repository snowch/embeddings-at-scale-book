# Code from Chapter 10
# Book: Embeddings at Scale

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# Placeholder class for distributed contrastive embedding
# See distributedembeddingtable.py for full implementation
class DistributedContrastiveEmbedding(nn.Module):
    """Placeholder for DistributedContrastiveEmbedding. Replace with actual implementation."""
    def __init__(self, vocab_size=100000, embedding_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, anchor_ids, positive_ids):
        anchor_emb = self.embedding(anchor_ids)
        positive_emb = self.embedding(positive_ids)
        # Return dummy loss
        return torch.tensor(0.5, requires_grad=True)


class EmbeddingDataset(Dataset):
    """
    Simple dataset for embedding training

    In production: Load from files, databases, or data lakes
    """

    def __init__(self, num_samples: int, vocab_size: int):
        self.num_samples = num_samples
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random anchor-positive pairs
        # In production: Load real pairs from storage
        return {
            'anchor_id': torch.randint(0, self.vocab_size, (1,)).item(),
            'positive_id': torch.randint(0, self.vocab_size, (1,)).item()
        }

def setup_distributed(rank: int, world_size: int):
    """
    Initialize distributed training process group

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
    """
    # Set device
    torch.cuda.set_device(rank)

    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # NCCL for GPU communication
        init_method='tcp://localhost:12355',
        world_size=world_size,
        rank=rank
    )

def cleanup_distributed():
    """Cleanup distributed resources"""
    dist.destroy_process_group()

def train_worker(
    rank: int,
    world_size: int,
    epochs: int = 10,
    batch_size: int = 512
):
    """
    Training worker for single GPU

    Each GPU runs this function independently.
    PyTorch coordinates gradient synchronization.

    Args:
        rank: GPU rank (0 to world_size-1)
        world_size: Total GPUs
        epochs: Number of epochs
        batch_size: Batch size per GPU
    """

    # Setup distributed
    setup_distributed(rank, world_size)

    # Create model and move to GPU
    model = DistributedContrastiveEmbedding(
        vocab_size=100000,
        embedding_dim=512
    )
    model = model.to(rank)

    # Wrap in DDP
    model = DDP(model, device_ids=[rank])

    # Create dataset
    dataset = EmbeddingDataset(num_samples=100000, vocab_size=100000)

    # Distributed sampler (each GPU sees different data)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    # Data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # Optimizer (scale learning rate by world size)
    base_lr = 0.001
    scaled_lr = base_lr * world_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr)

    # Training loop
    for epoch in range(epochs):
        # Set epoch for shuffling
        sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            # Move to GPU
            anchor_ids = torch.tensor([batch['anchor_id']], device=rank)
            positive_ids = torch.tensor([batch['positive_id']], device=rank)

            # Forward
            loss = model(anchor_ids, positive_ids)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            total_loss += loss.item()

            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")

        # Log epoch stats (only rank 0)
        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch} complete: Avg Loss = {avg_loss:.4f}")

    # Cleanup
    cleanup_distributed()

def launch_multi_gpu_training(world_size: int = 8):
    """
    Launch multi-GPU training

    Spawns one process per GPU, each running train_worker()

    Args:
        world_size: Number of GPUs to use
    """

    # Spawn processes (one per GPU)
    mp.spawn(
        train_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

# Uncomment to run:
# launch_multi_gpu_training(world_size=torch.cuda.device_count())
