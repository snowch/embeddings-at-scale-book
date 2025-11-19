import torch

# Code from Chapter 07
# Book: Embeddings at Scale
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler


class EnterpriseSelfsupervisedPipeline:
    """
    Production self-supervised learning pipeline

    Features:
    - Distributed training across multiple GPUs/nodes
    - Checkpointing and recovery
    - Monitoring and logging
    - Efficient data loading from data lake
    - Model versioning
    """

    def __init__(
        self,
        model,
        data_source,
        batch_size=256,
        num_workers=8,
        checkpoint_dir='./checkpoints',
        log_dir='./logs'
    ):
        self.model = model
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        # Setup distributed training if available
        self.world_size = torch.cuda.device_count()
        self.is_distributed = self.world_size > 1

    def setup_distributed(self):
        """Initialize distributed training"""
        if self.is_distributed:
            dist.init_process_group(backend='nccl')
            local_rank = dist.get_rank()
            torch.cuda.set_device(local_rank)

            # Wrap model in DDP
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[local_rank]
            )

    def create_dataloader(self):
        """Create efficient dataloader for unlabeled data"""
        # In production, this would load from S3, GCS, or data lake
        dataset = UnlabeledEnterpriseDataset(self.data_source)

        if self.is_distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = None

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2
        )

        return dataloader

    def train(self, num_epochs=100, learning_rate=1e-4):
        """
        Train self-supervised model

        Args:
            num_epochs: Number of epochs
            learning_rate: Learning rate
        """
        # Setup
        self.setup_distributed()
        dataloader = self.create_dataloader()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs
        )

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()

            epoch_loss = 0
            epoch_samples = 0

            for _batch_idx, batch in enumerate(dataloader):
                batch = batch.cuda()

                # Create pretext task
                inputs, targets, mask = self.model.module.create_pretext_task(batch)

                # Forward and backward
                optimizer.zero_grad()
                loss, metrics = self.model.module.compute_loss(inputs, targets, mask)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

                epoch_loss += loss.item() * batch.size(0)
                epoch_samples += batch.size(0)

            # Update learning rate
            scheduler.step()

            # Log metrics
            avg_loss = epoch_loss / epoch_samples
            self._log_metrics(epoch, avg_loss, scheduler.get_last_lr()[0])

            # Save checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, avg_loss)

    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss
        }

        path = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch'], checkpoint['loss']

    def _log_metrics(self, epoch, loss, lr):
        """Log training metrics"""
        print(f"Epoch {epoch}: Loss={loss:.4f}, LR={lr:.6f}")

        # In production, log to MLflow, Weights & Biases, etc.


class UnlabeledEnterpriseDataset(torch.utils.data.Dataset):
    """
    Dataset for loading unlabeled enterprise data

    In production, this would:
    - Stream from S3/GCS/Azure Blob
    - Handle multiple data formats (parquet, JSON, CSV)
    - Apply preprocessing on-the-fly
    - Cache frequently accessed data
    """

    def __init__(self, data_source):
        self.data_source = data_source
        # Load metadata about available data
        self.data_files = self._discover_data()

    def _discover_data(self):
        """Discover available data files"""
        # In production: list files from data lake
        # For now, return dummy list
        return ['file1.parquet', 'file2.parquet']

    def __len__(self):
        # Return total number of samples
        return 1000000  # Dummy value

    def __getitem__(self, idx):
        # Load and preprocess sample
        # In production: load from S3, apply preprocessing
        return torch.randn(512, 768)  # Dummy data
