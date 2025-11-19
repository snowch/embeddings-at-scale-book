# Code from Chapter 05
# Book: Embeddings at Scale

from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionContrastive:
    """
    Mixed precision training for contrastive learning

    Benefits:
    - 2x memory savings (FP16 vs FP32)
    - 2-3x faster training on modern GPUs (Tensor Cores)
    - Enables larger batch sizes

    Challenges:
    - Numerical stability for contrastive loss (exp operations)
    - Gradient scaling required
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        # Gradient scaler for FP16 training
        self.scaler = GradScaler()

    def train_step(self, batch, device):
        """
        Training step with mixed precision
        """
        # Move batch to device
        anchor_ids = batch['anchor_ids'].to(device)
        anchor_mask = batch['anchor_mask'].to(device)
        positive_ids = batch['positive_ids'].to(device)
        positive_mask = batch['positive_mask'].to(device)

        # Forward pass in FP16
        with autocast():
            # Encode
            anchor_emb, _ = self.model(anchor_ids, anchor_mask)
            positive_emb, _ = self.model(positive_ids, positive_mask)

            # Compute loss
            # Note: loss computation must be numerically stable in FP16
            loss, metrics = self.model.compute_loss(anchor_emb, positive_emb)

        # Backward pass with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # Unscale gradients and clip
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step with scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), metrics


# Numerical stability considerations for FP16 contrastive learning
class StableInfoNCELoss:
    """
    Numerically stable InfoNCE loss for mixed precision training
    """

    def __init__(self, temperature=0.07):
        self.temperature = temperature

    def compute_loss(self, anchor_emb, positive_emb, all_emb):
        """
        Compute InfoNCE with numerical stability tricks
        """
        # Normalize in FP32 for stability
        anchor_norm = F.normalize(anchor_emb.float(), dim=1)
        positive_norm = F.normalize(positive_emb.float(), dim=1)
        all_norm = F.normalize(all_emb.float(), dim=1)

        # Similarities in FP32
        similarity_matrix = torch.matmul(anchor_norm, all_norm.T) / self.temperature

        # Log-sum-exp trick for numerical stability
        # Instead of: -log(exp(x) / sum(exp(x_i)))
        # Use: -x + log(sum(exp(x_i)))
        # Which equals: log(sum(exp(x_i))) - x
        # This is: torch.logsumexp(logits) - logits[positive]

        labels = torch.arange(len(anchor_emb), device=anchor_emb.device)

        # Log-sum-exp of all logits (denominator)
        log_denominator = torch.logsumexp(similarity_matrix, dim=1)

        # Positive logits (numerator)
        positive_logits = similarity_matrix[torch.arange(len(anchor_emb)), labels]

        # Loss: -log(exp(pos) / sum(exp(all))) = log(sum) - pos
        loss = (log_denominator - positive_logits).mean()

        return loss
