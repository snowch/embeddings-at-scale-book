import torch

# Code from Chapter 05
# Book: Embeddings at Scale


class GradientAccumulationContrastive:
    """
    Simulate large batch sizes through gradient accumulation

    Idea: Accumulate gradients over multiple small batches,
          then update weights once.

    Effective batch size = micro_batch_size × accumulation_steps
    """

    def __init__(self, model, optimizer, micro_batch_size=256, effective_batch_size=4096):
        """
        Args:
            model: Embedding model
            optimizer: Optimizer
            micro_batch_size: Batch size that fits in memory
            effective_batch_size: Target effective batch size
        """
        self.model = model
        self.optimizer = optimizer
        self.micro_batch_size = micro_batch_size
        self.effective_batch_size = effective_batch_size

        self.accumulation_steps = effective_batch_size // micro_batch_size

        print(f"Gradient accumulation: {self.accumulation_steps} steps")
        print(f"Micro batch size: {micro_batch_size}")
        print(f"Effective batch size: {effective_batch_size}")

    def train_step(self, dataloader, device):
        """
        Training step with gradient accumulation
        """
        self.model.train()
        self.optimizer.zero_grad()

        accumulated_loss = 0
        accumulated_metrics = {"accuracy": 0, "positive_sim": 0, "negative_sim": 0}

        # Accumulate gradients over multiple micro-batches
        for step, batch in enumerate(dataloader):
            # Forward pass
            embeddings = self.model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )

            loss, metrics = self.model.compute_loss(embeddings)

            # Scale loss by accumulation steps
            # This ensures gradient magnitude is correct
            loss = loss / self.accumulation_steps

            # Backward pass (accumulates gradients)
            loss.backward()

            accumulated_loss += loss.item()
            accumulated_metrics["accuracy"] += metrics["accuracy"]
            accumulated_metrics["positive_sim"] += metrics["positive_sim"]
            accumulated_metrics["negative_sim"] += metrics["negative_sim"]

            # Update weights after accumulation_steps
            if (step + 1) % self.accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Average metrics
                for key in accumulated_metrics:
                    accumulated_metrics[key] /= self.accumulation_steps

                return accumulated_loss, accumulated_metrics
