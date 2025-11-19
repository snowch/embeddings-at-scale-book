# Code from Chapter 10
# Book: Embeddings at Scale


import torch
import torch.nn as nn


class GradientAccumulationTrainer:
    """
    Enable large effective batch sizes through gradient accumulation

    Problem:
    - Contrastive learning needs 16K-32K batch size for quality
    - Single GPU: 1024 max batch size (memory constraint)
    - Solution: Accumulate gradients over 32 micro-batches of 1024
      → Effective batch size: 32 × 1024 = 32,768

    Process:
    1. Forward pass on micro-batch
    2. Backward pass (accumulate gradients, don't update yet)
    3. Repeat for N micro-batches
    4. Optimizer step (update parameters with accumulated gradients)
    5. Zero gradients

    Benefits:
    - Train with batch sizes exceeding GPU memory
    - Identical results to full-batch training
    - Trade-off: More iterations per update (slower wall-clock time)

    Drawbacks:
    - Batch normalization statistics incorrect (computed per micro-batch)
    - Longer time per effective batch
    - May require learning rate adjustments
    """

    def __init__(
        self,
        model: nn.Module,
        accumulation_steps: int = 4
    ):
        """
        Args:
            model: Model to train
            accumulation_steps: Number of micro-batches to accumulate
        """
        self.model = model
        self.accumulation_steps = accumulation_steps

    def train_step(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda'
    ) -> float:
        """
        Training step with gradient accumulation

        Args:
            dataloader: Data loader providing micro-batches
            optimizer: Optimizer instance
            device: Device to train on

        Returns:
            avg_loss: Average loss across micro-batches
        """
        self.model.train()
        optimizer.zero_grad()

        total_loss = 0.0

        for i, batch in enumerate(dataloader):
            if i >= self.accumulation_steps:
                break

            # Move batch to device
            anchor_ids = batch['anchor_ids'].to(device)
            positive_ids = batch['positive_ids'].to(device)

            # Forward pass
            loss = self.model(anchor_ids, positive_ids)

            # Scale loss by accumulation steps
            # (since we're summing gradients, not averaging)
            loss = loss / self.accumulation_steps

            # Backward pass (accumulates gradients)
            loss.backward()

            total_loss += loss.item()

        # Gradient clipping (on accumulated gradients)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )

        # Optimizer step (updates parameters)
        optimizer.step()

        # Zero gradients for next accumulation cycle
        optimizer.zero_grad()

        return total_loss

# Example: Effective 32K batch with 8GB GPU
def gradient_accumulation_example():
    """
    Train with 32K effective batch size on limited memory

    Hardware: 1× A100 (80GB)
    Micro-batch: 1024 samples
    Accumulation steps: 32
    Effective batch: 32 × 1024 = 32,768 samples

    Memory usage:
    - Micro-batch: 1024 × 512 × 4 bytes = 2MB (embeddings)
    - Model: ~500MB (encoder parameters)
    - Optimizer state: ~1GB (Adam momentum)
    - Activations: ~10GB (forward/backward)
    Total: ~12GB per micro-batch (fits on A100)
    """

    model = DistributedContrastiveEmbedding(vocab_size=100000)
    model = model.to('cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    trainer = GradientAccumulationTrainer(
        model=model,
        accumulation_steps=32  # 32× micro-batches
    )

    # Dummy dataloader
    dataloader = [
        {
            'anchor_ids': torch.randint(0, 100000, (1024,)),
            'positive_ids': torch.randint(0, 100000, (1024,))
        }
        for _ in range(32)
    ]

    loss = trainer.train_step(dataloader, optimizer)
    print(f"Loss with 32K effective batch: {loss:.4f}")

# Uncomment to run:
# gradient_accumulation_example()
