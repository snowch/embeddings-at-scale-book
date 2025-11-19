import torch.nn as nn

# Code from Chapter 10
# Book: Embeddings at Scale
import torch.optim as optim


class MemoryEfficientOptimizer:
    """
    Optimize memory usage for large models

    Approaches:
    1. CPU offloading: Store optimizer state on CPU, transfer on-demand
    2. Mixed precision optimizer state: FP16 for momentum/variance
    3. 8-bit optimizers: Quantized optimizer states (bitsandbytes)
    4. Sparse updates: Only update frequently changing parameters

    Memory comparison (100M parameter model):
    - Adam (FP32): 100M params × 3 states × 4 bytes = 1.2GB
    - Adam (FP16): 100M params × 3 states × 2 bytes = 600MB
    - 8-bit Adam: 100M params × 3 states × 1 byte = 300MB
    - SGD: 100M params × 1 state × 4 bytes = 400MB (no momentum stored)
    """

    @staticmethod
    def get_optimizer(
        parameters, optimizer_type: str = "adamw", lr: float = 0.001, memory_efficient: bool = True
    ):
        """
        Get memory-efficient optimizer

        Args:
            parameters: Model parameters
            optimizer_type: 'adamw', 'sgd', '8bit_adam'
            lr: Learning rate
            memory_efficient: Enable memory optimizations

        Returns:
            optimizer: Configured optimizer
        """

        if optimizer_type == "adamw":
            if memory_efficient:
                # Use fused AdamW (faster, lower memory)
                return optim.AdamW(
                    parameters,
                    lr=lr,
                    fused=True,  # Fused kernel (A100+)
                )
            else:
                return optim.AdamW(parameters, lr=lr)

        elif optimizer_type == "sgd":
            # SGD with momentum uses less memory than Adam
            return optim.SGD(parameters, lr=lr, momentum=0.9, nesterov=True)

        elif optimizer_type == "8bit_adam":
            # Requires: pip install bitsandbytes
            try:
                import bitsandbytes as bnb

                return bnb.optim.Adam8bit(parameters, lr=lr)
            except ImportError:
                print("bitsandbytes not installed, falling back to AdamW")
                return optim.AdamW(parameters, lr=lr)

        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")


def memory_usage_comparison():
    """
    Compare memory usage across optimization techniques

    Model: 100M parameters (typical BERT-Base scale)
    """

    # Create dummy model
    model = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 1024))

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.1f}M")

    # Calculate memory requirements
    param_memory_mb = num_params * 4 / 1e6  # FP32

    # Optimizer state memory
    adam_fp32_mb = num_params * 3 * 4 / 1e6  # 3 states (param, momentum, variance)
    adam_fp16_mb = num_params * 3 * 2 / 1e6
    adam_8bit_mb = num_params * 3 * 1 / 1e6
    sgd_mb = num_params * 1 * 4 / 1e6  # Only momentum

    print("\nMemory usage:")
    print(f"  Parameters (FP32): {param_memory_mb:.1f} MB")
    print(f"  Adam (FP32): {adam_fp32_mb:.1f} MB optimizer state")
    print(f"  Adam (FP16): {adam_fp16_mb:.1f} MB optimizer state")
    print(f"  Adam (8-bit): {adam_8bit_mb:.1f} MB optimizer state")
    print(f"  SGD: {sgd_mb:.1f} MB optimizer state")

    print("\nTotal memory (params + optimizer):")
    print(f"  Adam (FP32): {param_memory_mb + adam_fp32_mb:.1f} MB")
    print(f"  Adam (FP16): {param_memory_mb + adam_fp16_mb:.1f} MB")
    print(f"  Adam (8-bit): {param_memory_mb + adam_8bit_mb:.1f} MB")
    print(f"  SGD: {param_memory_mb + sgd_mb:.1f} MB")


# Uncomment to run:
# memory_usage_comparison()
