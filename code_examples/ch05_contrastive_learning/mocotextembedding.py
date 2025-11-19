import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Code from Chapter 05
# Book: Embeddings at Scale


# Placeholder class
class TextAugmentation:
    """Placeholder for TextAugmentation."""

    def __init__(self):
        pass

    def augment(self, text):
        return text  # Return unchanged text


class MoCoTextEmbedding(nn.Module):
    """
    MoCo (Momentum Contrast) for text embeddings

    Advantages over SimCLR:
    - Small batch sizes work well (256 vs. 4096+)
    - Queue provides large, consistent set of negatives (65K typical)
    - Momentum encoder provides stable keys

    Trade-offs:
    - More complex implementation
    - Queue introduces staleness (keys encoded by older model)
    - Requires careful momentum tuning
    """

    def __init__(
        self,
        base_model="bert-base-uncased",
        projection_dim=128,
        hidden_dim=512,
        queue_size=65536,
        momentum=0.999,
        temperature=0.07,
    ):
        super().__init__()

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # Query encoder (actively trained)
        self.encoder_q = AutoModel.from_pretrained(base_model)

        # Key encoder (momentum updated)
        self.encoder_k = AutoModel.from_pretrained(base_model)

        # Initialize key encoder with same weights as query encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # Key encoder not trained by backprop

        encoder_dim = self.encoder_q.config.hidden_size

        # Projection heads
        self.projection_q = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim)
        )

        self.projection_k = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim)
        )

        # Initialize projection_k with same weights
        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Queue: stores negative keys
        # Shape: (projection_dim, queue_size)
        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update: key_encoder = m * key_encoder + (1 - m) * query_encoder

        This creates a slowly evolving key encoder, providing stable keys
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update queue: remove oldest keys, add new keys

        Args:
            keys: (batch_size, projection_dim)
        """
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # Replace oldest keys with new keys
        # Queue is circular buffer
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr : ptr + batch_size] = keys.T
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, : batch_size - remaining] = keys[remaining:].T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def encode_text(self, input_ids, attention_mask, encoder, projection):
        """
        Encode text through encoder and projection
        """
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size())
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        representations = sum_embeddings / sum_mask

        # Project
        embeddings = projection(representations)
        embeddings = F.normalize(embeddings, dim=1)

        return embeddings

    def forward(self, query_input_ids, query_attention_mask, key_input_ids, key_attention_mask):
        """
        Forward pass for MoCo

        Args:
            query_input_ids, query_attention_mask: Query batch
            key_input_ids, key_attention_mask: Key batch (augmented version)

        Returns:
            loss, metrics
        """
        # Encode queries (through actively trained encoder)
        q = self.encode_text(
            query_input_ids, query_attention_mask, self.encoder_q, self.projection_q
        )

        # Encode keys (through momentum encoder)
        with torch.no_grad():
            # Update key encoder
            self._momentum_update_key_encoder()

            # Encode keys
            k = self.encode_text(
                key_input_ids, key_attention_mask, self.encoder_k, self.projection_k
            )

        # Positive logits: (batch_size, 1)
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # Negative logits: (batch_size, queue_size)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # Concatenate positive and negative logits
        # Shape: (batch_size, 1 + queue_size)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

        # Labels: positive is always first (index 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)

        # Compute loss
        loss = F.cross_entropy(logits, labels)

        # Update queue
        self._dequeue_and_enqueue(k)

        # Metrics
        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == labels).float().mean()

            positive_sim = l_pos.mean()
            negative_sim = l_neg.mean()

        metrics = {
            "accuracy": accuracy.item(),
            "positive_similarity": positive_sim.item(),
            "negative_similarity": negative_sim.item(),
            "queue_ptr": int(self.queue_ptr[0]),
        }

        return loss, metrics


# Training function for MoCo
def train_moco_epoch(model, dataloader, optimizer, device):
    """
    Train MoCo for one epoch

    Key difference from SimCLR: small batch sizes work!
    """
    model.train()
    total_loss = 0
    total_accuracy = 0
    augmenter = TextAugmentation()

    for batch in dataloader:
        texts = batch["text"]

        # Generate query and key augmentations
        queries = [augmenter.augment_simple(text, method="random_deletion") for text in texts]
        keys = [augmenter.augment_simple(text, method="synonym_replacement") for text in texts]

        # Tokenize
        query_encoded = model.tokenizer(
            queries, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)

        key_encoded = model.tokenizer(
            keys, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)

        # Forward pass
        loss, metrics = model(
            query_encoded["input_ids"],
            query_encoded["attention_mask"],
            key_encoded["input_ids"],
            key_encoded["attention_mask"],
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += metrics["accuracy"]

    return total_loss / len(dataloader), total_accuracy / len(dataloader)


# Example: MoCo works well with smaller batches
model_moco = MoCoTextEmbedding(
    base_model="bert-base-uncased",
    projection_dim=128,
    queue_size=65536,  # Large queue of negatives
    momentum=0.999,  # Slow momentum update
    temperature=0.07,
).to("cuda")

optimizer = torch.optim.AdamW([p for p in model_moco.parameters() if p.requires_grad], lr=2e-5)

# Can use batch size 256 instead of 4096!
# Queue provides large set of high-quality negatives
