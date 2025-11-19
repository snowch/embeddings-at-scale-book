# Code from Chapter 05
# Book: Embeddings at Scale

import torch.distributed as dist

class MultiNodeContrastiveLearning:
    """
    Multi-node distributed contrastive learning

    Architecture:
    - N nodes (machines)
    - M GPUs per node
    - Total: N × M GPUs

    Communication:
    - All-reduce for gradients (handled by DDP)
    - All-gather for embeddings (manual)
    """

    def __init__(self, model, rank, world_size, local_rank,
                 master_addr='localhost', master_port='12355'):
        """
        Args:
            model: Embedding model
            rank: Global rank (0 to world_size-1)
            world_size: Total number of GPUs across all nodes
            local_rank: GPU rank within this node
            master_addr: IP of master node
            master_port: Port for communication
        """
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        # Initialize process group
        self._init_distributed(master_addr, master_port)

        # Wrap model
        torch.cuda.set_device(local_rank)
        self.model = self.model.to(local_rank)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank
        )

    def _init_distributed(self, master_addr, master_port):
        """Initialize distributed backend"""
        import os

        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port

        dist.init_process_group(
            backend='nccl',  # NCCL for multi-node GPU communication
            rank=self.rank,
            world_size=self.world_size
        )

        print(f"Initialized rank {self.rank}/{self.world_size}")

    def gather_embeddings_multi_node(self, local_embeddings):
        """
        Gather embeddings from all GPUs across all nodes

        More complex than single-node: network communication between nodes
        """
        # Create buffer for gathered embeddings
        gathered_embeddings = [
            torch.zeros_like(local_embeddings)
            for _ in range(self.world_size)
        ]

        # All-gather across all GPUs
        dist.all_gather(gathered_embeddings, local_embeddings)

        # Concatenate
        all_embeddings = torch.cat(gathered_embeddings, dim=0)

        return all_embeddings

    def train_step_multi_node(self, batch):
        """
        Training step with multi-node distribution
        """
        # Forward pass
        anchor_emb = self.model(
            batch['anchor_ids'].cuda(self.local_rank),
            batch['anchor_mask'].cuda(self.local_rank)
        )
        positive_emb = self.model(
            batch['positive_ids'].cuda(self.local_rank),
            batch['positive_mask'].cuda(self.local_rank)
        )

        # Gather embeddings from all GPUs (across nodes)
        all_anchors = self.gather_embeddings_multi_node(anchor_emb)
        all_positives = self.gather_embeddings_multi_node(positive_emb)

        # Compute loss using all embeddings
        loss = self.compute_contrastive_loss(
            anchor_emb,  # Local anchors
            all_anchors,  # Global negatives
            all_positives,  # Global positives
            rank=self.rank
        )

        return loss


def launch_multi_node_training(num_nodes, gpus_per_node):
    """
    Launch multi-node distributed training

    Typically launched via SLURM or similar cluster manager:
