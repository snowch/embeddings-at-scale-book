# Code from Chapter 10
# Book: Embeddings at Scale

import os

def setup_multi_node(rank: int, world_size: int, master_addr: str, master_port: str):
    """
    Initialize multi-node distributed training

    Environment variables (set by job scheduler like Slurm):
    - MASTER_ADDR: IP address of rank 0 node
    - MASTER_PORT: Port for communication
    - WORLD_SIZE: Total number of processes across all nodes
    - RANK: Global rank of this process

    Typical setup:
    - Node 0 (master): Ranks 0-7 (8 GPUs)
    - Node 1: Ranks 8-15 (8 GPUs)
    - Node 2: Ranks 16-23 (8 GPUs)
    - Node 3: Ranks 24-31 (8 GPUs)
    Total: 32 GPUs across 4 nodes

    Args:
        rank: Global rank (0 to world_size-1)
        world_size: Total GPUs across all nodes
        master_addr: IP of master node
        master_port: Communication port
    """

    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # Local rank (GPU on this node)
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank
    )

    print(f"Initialized rank {rank}/{world_size} on device {local_rank}")

def train_multi_node():
    """
    Multi-node training script

    Launch with Slurm or similar cluster scheduler.
    """
    pass
