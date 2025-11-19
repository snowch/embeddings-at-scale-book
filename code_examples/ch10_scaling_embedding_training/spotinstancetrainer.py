# Code from Chapter 10
# Book: Embeddings at Scale

import signal
import time
from pathlib import Path


class SpotInstanceTrainer:
    """
    Training on spot instances with automatic checkpointing

    Challenges:
    - Instance can be preempted with 2-minute warning
    - Need to checkpoint frequently
    - Resume from last checkpoint on new instance

    Strategies:
    1. Checkpoint every N minutes (e.g., every 15 minutes)
    2. Listen for SIGTERM (preemption signal)
    3. Save checkpoint on SIGTERM
    4. Resume from latest checkpoint on restart
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_dir: str = './checkpoints',
        checkpoint_interval_minutes: int = 15
    ):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_interval = checkpoint_interval_minutes * 60  # Convert to seconds

        self.last_checkpoint_time = time.time()

        # Register SIGTERM handler (spot instance preemption warning)
        signal.signal(signal.SIGTERM, self._handle_preemption)

    def _handle_preemption(self, signum, frame):
        """
        Handle spot instance preemption

        Called ~2 minutes before termination.
        Save checkpoint and gracefully exit.
        """
        print("⚠️  Spot instance preemption detected!")
        print("Saving emergency checkpoint...")

        self.save_checkpoint(emergency=True)

        print("✓ Emergency checkpoint saved. Exiting gracefully.")
        exit(0)

    def save_checkpoint(
        self,
        epoch: int = 0,
        step: int = 0,
        optimizer: torch.optim.Optimizer = None,
        emergency: bool = False
    ):
        """
        Save training checkpoint

        Args:
            epoch: Current epoch
            step: Current step
            optimizer: Optimizer state
            emergency: Emergency checkpoint (spot preemption)
        """
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'timestamp': time.time()
        }

        if emergency:
            checkpoint_path = self.checkpoint_dir / 'emergency_checkpoint.pt'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_step_{step}.pt'

        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")

        # Update last checkpoint time
        self.last_checkpoint_time = time.time()

    def load_latest_checkpoint(self, optimizer: torch.optim.Optimizer = None) -> Dict:
        """
        Load latest checkpoint (for resuming training)

        Returns:
            checkpoint: Checkpoint data (or None if no checkpoint exists)
        """
        # Check for emergency checkpoint first
        emergency_path = self.checkpoint_dir / 'emergency_checkpoint.pt'
        if emergency_path.exists():
            print(f"Loading emergency checkpoint: {emergency_path}")
            checkpoint = torch.load(emergency_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and checkpoint['optimizer_state_dict']:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint

        # Find latest regular checkpoint
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_*.pt'))
        if not checkpoints:
            print("No checkpoints found, starting from scratch")
            return None

        latest = checkpoints[-1]
        print(f"Loading checkpoint: {latest}")
        checkpoint = torch.load(latest)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint

    def should_checkpoint(self) -> bool:
        """Check if it's time to checkpoint based on interval"""
        elapsed = time.time() - self.last_checkpoint_time
        return elapsed >= self.checkpoint_interval

    def train_with_checkpointing(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        start_epoch: int = 0,
        start_step: int = 0
    ):
        """
        Training loop with automatic checkpointing

        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            epochs: Total epochs
            start_epoch: Resume from this epoch
            start_step: Resume from this step
        """

        for epoch in range(start_epoch, epochs):
            for step, batch in enumerate(dataloader):
                if epoch == start_epoch and step < start_step:
                    continue  # Skip to resume point

                # Training step
                loss = self.model(batch['anchor_ids'], batch['positive_ids'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Periodic checkpointing
                if self.should_checkpoint():
                    self.save_checkpoint(epoch, step, optimizer)

            # End-of-epoch checkpoint
            self.save_checkpoint(epoch, step, optimizer)

        print("Training complete!")
