# Code from Chapter 05
# Book: Embeddings at Scale

class CommunicationEfficientDistributed:
    """
    Reduce communication overhead in distributed training

    Techniques:
    1. Gradient compression
    2. Local SGD (sync less frequently)
    3. Overlap communication with computation
    """

    def __init__(self, model, world_size, rank):
        self.model = model
        self.world_size = world_size
        self.rank = rank

        # Local update counter
        self.local_updates = 0
        self.sync_frequency = 10  # Sync every N steps

    def local_sgd_step(self, loss, optimizer):
        """
        Local SGD: update locally, sync periodically

        Reduces communication by 10x (if sync_frequency=10)
        """
        # Local backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.local_updates += 1

        # Sync periodically
        if self.local_updates % self.sync_frequency == 0:
            self.sync_models()

    def sync_models(self):
        """
        Average model parameters across all GPUs
        """
        for param in self.model.parameters():
            # All-reduce: sum then divide by world_size
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.world_size

    def compressed_all_gather(self, embeddings, compression_ratio=0.1):
        """
        Compress embeddings before gathering to reduce bandwidth

        Args:
            embeddings: (batch_size, dim)
            compression_ratio: Fraction of dimensions to keep

        Returns:
            all_embeddings: Gathered from all GPUs (compressed)
        """
        # Top-k compression: keep only largest magnitude values
        k = int(embeddings.shape[1] * compression_ratio)

        # Find top-k values and indices
        values, indices = embeddings.topk(k, dim=1)

        # Gather compressed representations
        all_values = [torch.zeros_like(values) for _ in range(self.world_size)]
        all_indices = [torch.zeros_like(indices) for _ in range(self.world_size)]

        dist.all_gather(all_values, values)
        dist.all_gather(all_indices, indices)

        # Reconstruct (sparse)
        batch_size = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]

        all_embeddings_sparse = []

        for vals, inds in zip(all_values, all_indices):
            # Reconstruct dense tensor (mostly zeros)
            dense = torch.zeros(batch_size, embedding_dim, device=embeddings.device)
            dense.scatter_(1, inds, vals)
            all_embeddings_sparse.append(dense)

        return torch.cat(all_embeddings_sparse, dim=0)
