# Code from Chapter 10
# Book: Embeddings at Scale

"""
Parallelism Strategies for Embedding Training

1. Data Parallelism: Replicate model, split data across devices
   - Standard approach for most layers
   - Each device processes different batch
   - Gradients synchronized across devices

2. Model Parallelism: Split model across devices
   - Necessary for large embedding tables
   - Different devices hold different vocabulary ranges
   - Activations transferred between devices

3. Pipeline Parallelism: Split model into stages
   - Each stage on different device
   - Micro-batches flow through pipeline
   - Reduces bubble time (idle GPU time)

4. Tensor Parallelism: Split individual layers across devices
   - Useful for very large transformer layers
   - Intra-layer parallelism
   - High communication overhead

For embedding training, typically combine:
- Data Parallelism: For encoder/projection layers
- Model Parallelism: For large embedding tables
- Pipeline Parallelism: For deep transformer stacks
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict, List, Tuple
import numpy as np

class DistributedEmbeddingTable(nn.Module):
    """
    Model-parallel embedding table for large vocabularies

    Splits vocabulary across multiple GPUs:
    - GPU 0: vocab indices 0 to N/4
    - GPU 1: vocab indices N/4 to N/2
    - GPU 2: vocab indices N/2 to 3N/4
    - GPU 3: vocab indices 3N/4 to N

    Critical for:
    - Large vocabularies (100M+ tokens)
    - Product catalogs (10M+ items)
    - User tables (100M+ users)

    Without model parallelism:
    - 100M vocab × 512 dims × 4 bytes = 200GB (exceeds single GPU memory)

    With 8-way model parallelism:
    - 12.5M vocab per GPU × 512 dims × 4 bytes = 25GB per GPU (fits on A100)
    """

    def __init__(
        self,
        total_vocab_size: int,
        embedding_dim: int,
        world_size: int,
        rank: int
    ):
        """
        Args:
            total_vocab_size: Full vocabulary size
            embedding_dim: Embedding dimension
            world_size: Number of GPUs
            rank: Current GPU rank (0 to world_size-1)
        """
        super().__init__()
        self.total_vocab_size = total_vocab_size
        self.embedding_dim = embedding_dim
        self.world_size = world_size
        self.rank = rank

        # Each GPU holds a slice of the vocabulary
        self.vocab_per_gpu = total_vocab_size // world_size
        self.vocab_start = rank * self.vocab_per_gpu
        self.vocab_end = (rank + 1) * self.vocab_per_gpu

        # Local embedding table (subset of vocabulary)
        self.embeddings = nn.Embedding(
            self.vocab_per_gpu,
            embedding_dim
        )

        print(f"Rank {rank}: Vocabulary range [{self.vocab_start}, {self.vocab_end})")
        print(f"  Local embedding size: {self.vocab_per_gpu * embedding_dim * 4 / 1e9:.2f} GB")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embeddings across distributed vocabulary

        Process:
        1. Determine which GPUs hold embeddings for input_ids
        2. Send lookup requests to appropriate GPUs
        3. Gather results and return

        Args:
            input_ids: Token IDs (batch_size, seq_len)

        Returns:
            embeddings: (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize output
        output = torch.zeros(
            batch_size, seq_len, self.embedding_dim,
            device=device
        )

        # Mask for tokens this GPU is responsible for
        local_mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)

        if local_mask.any():
            # Get local token IDs (offset by vocab_start)
            local_ids = input_ids[local_mask] - self.vocab_start

            # Lookup local embeddings
            local_embeddings = self.embeddings(local_ids)

            # Place in output
            output[local_mask] = local_embeddings

        # All-reduce: Sum embeddings from all GPUs
        # Each GPU contributes embeddings for its vocabulary range
        dist.all_reduce(output, op=dist.ReduceOp.SUM)

        return output

class DistributedContrastiveEmbedding(nn.Module):
    """
    Distributed contrastive learning for embeddings

    Challenges at scale:
    1. Large batch sizes (8K-32K) for effective negative sampling
    2. All-to-all similarity computation (N² complexity)
    3. Gradient synchronization across devices
    4. Memory constraints for large batches

    Architecture:
    - Each GPU processes batch_size/world_size samples
    - Embeddings gathered across all GPUs for contrastive loss
    - Gradients computed and synchronized
    - Optimized for high-throughput, large-batch training

    Typical setup:
    - 8 GPUs × 1024 batch per GPU = 8192 total batch size
    - Contrastive loss computed on full 8192 samples
    - Enables strong negative sampling
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        hidden_dim: int = 2048,
        temperature: float = 0.07
    ):
        super().__init__()

        # Embedding table (potentially distributed)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Projection head (data parallel)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self.temperature = temperature

    def forward(
        self,
        anchor_ids: torch.Tensor,
        positive_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Distributed contrastive loss

        Args:
            anchor_ids: Anchor samples (local_batch_size,)
            positive_ids: Positive samples (local_batch_size,)

        Returns:
            loss: Contrastive loss
        """
        # Embed locally
        anchor_emb = self.embeddings(anchor_ids)
        positive_emb = self.embeddings(positive_ids)

        # Project
        anchor_proj = self.projection(anchor_emb)
        positive_proj = self.projection(positive_emb)

        # Normalize
        anchor_proj = F.normalize(anchor_proj, dim=-1)
        positive_proj = F.normalize(positive_proj, dim=-1)

        # Gather embeddings from all GPUs
        anchor_proj_all = self._gather_from_all_gpus(anchor_proj)
        positive_proj_all = self._gather_from_all_gpus(positive_proj)

        # Compute similarity matrix
        # (global_batch_size, global_batch_size)
        logits_aa = torch.matmul(anchor_proj_all, anchor_proj_all.T) / self.temperature
        logits_ap = torch.matmul(anchor_proj_all, positive_proj_all.T) / self.temperature

        # Contrastive loss (InfoNCE)
        # Positive pairs: (anchor_i, positive_i)
        # Negatives: All other samples in global batch
        global_batch_size = anchor_proj_all.shape[0]
        labels = torch.arange(global_batch_size, device=anchor_proj.device)

        # Loss for anchor-positive pairs
        loss_ap = F.cross_entropy(logits_ap, labels)

        return loss_ap

    def _gather_from_all_gpus(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather tensor from all GPUs

        Each GPU has (local_batch_size, embedding_dim)
        Result: (global_batch_size, embedding_dim) where
                global_batch_size = local_batch_size × world_size
        """
        if not dist.is_initialized():
            return tensor

        world_size = dist.get_world_size()
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]

        # All-gather: Each GPU gets tensors from all GPUs
        dist.all_gather(tensor_list, tensor)

        # Concatenate along batch dimension
        return torch.cat(tensor_list, dim=0)

class DistributedTrainer:
    """
    Orchestrates distributed training across multiple GPUs/nodes

    Features:
    - Automatic device placement
    - Gradient synchronization
    - Learning rate scaling
    - Checkpoint aggregation
    - Fault tolerance

    Scaling laws:
    - Linear speedup: 8 GPUs → 8x throughput (ideal)
    - Reality: 6-7x speedup (communication overhead)
    - 64 GPUs → 40-50x speedup (diminishing returns)

    Bottlenecks:
    - Communication bandwidth (gradient synchronization)
    - Load imbalance (uneven batch sizes)
    - Optimizer state synchronization
    """

    def __init__(
        self,
        model: nn.Module,
        local_rank: int,
        world_size: int,
        backend: str = 'nccl'
    ):
        """
        Args:
            model: Model to train
            local_rank: GPU rank on this node
            world_size: Total number of GPUs across all nodes
            backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
        """
        self.local_rank = local_rank
        self.world_size = world_size

        # Initialize distributed process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=local_rank
            )

        # Move model to GPU
        self.device = torch.device(f'cuda:{local_rank}')
        model = model.to(self.device)

        # Wrap in DistributedDataParallel
        self.model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # Optimization: skip if all params used
        )

        print(f"Initialized DDP on rank {local_rank}/{world_size}")

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Single distributed training step

        Process:
        1. Each GPU processes local batch
        2. Forward pass (communication for model-parallel layers)
        3. Backward pass (gradient synchronization)
        4. Optimizer step (local, uses synchronized gradients)

        Args:
            batch: Training batch (already on correct device)
            optimizer: Optimizer instance

        Returns:
            loss: Scalar loss value
        """
        self.model.train()

        # Forward pass
        loss = self.model(
            batch['anchor_ids'],
            batch['positive_ids']
        )

        # Backward pass (DDP automatically synchronizes gradients)
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (before optimizer step)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        return loss.item()

    def save_checkpoint(self, path: str, epoch: int, optimizer: torch.optim.Optimizer):
        """
        Save checkpoint (only from rank 0 to avoid duplicate writes)

        Checkpoint includes:
        - Model state dict
        - Optimizer state dict
        - Training metadata (epoch, step, etc.)
        """
        if self.local_rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, path)
            print(f"Checkpoint saved: {path}")

        # Synchronize to ensure checkpoint written before proceeding
        dist.barrier()

    def cleanup(self):
        """Cleanup distributed training"""
        dist.destroy_process_group()

# Example: Training script for distributed embedding model
def train_distributed_embedding_model(
    rank: int,
    world_size: int,
    epochs: int = 10
):
    """
    Distributed training script

    Launch with:
