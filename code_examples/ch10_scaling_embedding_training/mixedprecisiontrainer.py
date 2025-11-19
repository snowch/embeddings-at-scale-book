# Code from Chapter 10
# Book: Embeddings at Scale

import torch
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    """
    Automatic mixed precision (AMP) training

    FP16 (half precision):
    - 2 bytes per parameter (vs 4 bytes FP32)
    - 2-8× faster computation on Tensor Cores
    - 2× less memory usage

    Challenges:
    - Gradient underflow (very small gradients round to zero)
    - Reduced numerical range (FP16: ±65,504 vs FP32: ±3.4×10³⁸)

    Solution (Automatic Mixed Precision):
    1. Forward pass in FP16 (fast)
    2. Loss in FP32 (accuracy)
    3. Gradient scaling (prevent underflow)
    4. Backward pass in FP16 (fast)
    5. Unscale gradients to FP32
    6. Optimizer step in FP32 (stability)
    7. Copy updated FP32 parameters to FP16 for next iteration

    Speedup:
    - A100: 2-3× faster than FP32
    - V100: 2-5× faster than FP32
    - Memory: 2× reduction (enables larger batches)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device

        # Gradient scaler (prevents underflow)
        self.scaler = GradScaler()

    def train_step(
        self,
        batch: dict,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Training step with automatic mixed precision

        Args:
            batch: Training batch
            optimizer: Optimizer

        Returns:
            loss: Scalar loss
        """
        self.model.train()

        anchor_ids = batch['anchor_ids'].to(self.device)
        positive_ids = batch['positive_ids'].to(self.device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass in FP16 (autocast context)
        with autocast():
            loss = self.model(anchor_ids, positive_ids)

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        # Unscale gradients and clip
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )

        # Optimizer step (with gradient scaling)
        self.scaler.step(optimizer)

        # Update scaler for next iteration
        self.scaler.update()

        return loss.item()

# Combining Gradient Accumulation + Mixed Precision
class ScalableTrainer:
    """
    Combine gradient accumulation and mixed precision

    Enables:
    - Large effective batch sizes (gradient accumulation)
    - Fast computation (mixed precision)
    - Low memory usage (both techniques reduce memory)

    Example configuration:
    - Hardware: 8× A100 GPUs
    - Micro-batch per GPU: 512
    - Accumulation steps: 8
    - Effective global batch: 8 GPUs × 512 × 8 = 32,768

    Speedup vs baseline (single GPU, FP32, batch 128):
    - 8× from multi-GPU data parallelism
    - 2.5× from mixed precision
    - 256× larger effective batch size
    Total: ~20× faster training with 256× larger batch
    """

    def __init__(
        self,
        model: nn.Module,
        accumulation_steps: int = 4,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.accumulation_steps = accumulation_steps
        self.device = device
        self.scaler = GradScaler()

    def train_step(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Training step combining both techniques

        Args:
            dataloader: Micro-batch data loader
            optimizer: Optimizer

        Returns:
            avg_loss: Average loss over accumulated gradients
        """
        self.model.train()
        optimizer.zero_grad()

        total_loss = 0.0

        for i, batch in enumerate(dataloader):
            if i >= self.accumulation_steps:
                break

            anchor_ids = batch['anchor_ids'].to(self.device)
            positive_ids = batch['positive_ids'].to(self.device)

            # Forward in FP16
            with autocast():
                loss = self.model(anchor_ids, positive_ids)
                loss = loss / self.accumulation_steps

            # Backward with scaling
            self.scaler.scale(loss).backward()

            total_loss += loss.item()

        # Unscale and clip
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )

        # Step and update scaler
        self.scaler.step(optimizer)
        self.scaler.update()

        optimizer.zero_grad()

        return total_loss

def benchmark_mixed_precision():
    """
    Benchmark FP32 vs FP16 training speed

    Expected results (A100):
    - FP32: 100 samples/sec, 20GB memory
    - FP16: 250 samples/sec, 12GB memory
    """
    import time

    model = DistributedContrastiveEmbedding(vocab_size=100000)

    # FP32 baseline
    model_fp32 = model.to('cuda')
    optimizer = torch.optim.AdamW(model_fp32.parameters())

    batch = {
        'anchor_ids': torch.randint(0, 100000, (512,)).cuda(),
        'positive_ids': torch.randint(0, 100000, (512,)).cuda()
    }

    # Warmup
    for _ in range(10):
        loss = model_fp32(batch['anchor_ids'], batch['positive_ids'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Benchmark FP32
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        loss = model_fp32(batch['anchor_ids'], batch['positive_ids'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    fp32_time = time.time() - start

    # FP16 with AMP
    model_fp16 = model.to('cuda')
    optimizer = torch.optim.AdamW(model_fp16.parameters())
    trainer = MixedPrecisionTrainer(model_fp16)

    # Warmup
    for _ in range(10):
        trainer.train_step(batch, optimizer)

    # Benchmark FP16
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        trainer.train_step(batch, optimizer)
    torch.cuda.synchronize()
    fp16_time = time.time() - start

    print(f"FP32 time: {fp32_time:.2f}s")
    print(f"FP16 time: {fp16_time:.2f}s")
    print(f"Speedup: {fp32_time / fp16_time:.2f}×")

# Uncomment to run:
# benchmark_mixed_precision()
