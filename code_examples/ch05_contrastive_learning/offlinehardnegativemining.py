# Code from Chapter 05
# Book: Embeddings at Scale

import faiss
import numpy as np


class OfflineHardNegativeMining:
    """
    Offline hard negative mining using approximate nearest neighbors

    Process:
    1. Encode entire dataset with current model
    2. Build ANN index
    3. For each training example, find K nearest neighbors (hard negatives)
    4. Store hard negatives for next epoch
    5. Repeat periodically (e.g., every epoch or every N steps)

    Advantages:
    - Access to global hard negatives (best quality)
    - Can mine very hard negatives
    - Control over negative distribution

    Disadvantages:
    - Expensive: requires full dataset encoding
    - Storage: need to store mined negatives
    - Staleness: negatives from old model checkpoints
    """

    def __init__(self, embedding_dim, num_hard_negatives=10):
        """
        Args:
            embedding_dim: Dimension of embeddings
            num_hard_negatives: How many hard negatives to mine per example
        """
        self.embedding_dim = embedding_dim
        self.num_hard_negatives = num_hard_negatives

        # FAISS index for approximate nearest neighbor search
        self.index = None
        self.dataset_embeddings = None
        self.dataset_ids = None

    def encode_dataset(self, model, dataloader, device):
        """
        Encode entire dataset with current model

        Args:
            model: Current embedding model
            dataloader: DataLoader for full dataset
            device: torch device

        Returns:
            embeddings: (num_examples, embedding_dim)
            ids: (num_examples,) - example IDs
        """
        model.eval()

        all_embeddings = []
        all_ids = []

        with torch.no_grad():
            for batch in dataloader:
                # Encode batch
                embeddings = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device)
                )

                all_embeddings.append(embeddings.cpu())
                all_ids.extend(batch['id'])

        # Concatenate all
        embeddings = torch.cat(all_embeddings, dim=0).numpy()

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings, all_ids

    def build_index(self, embeddings):
        """
        Build FAISS index for efficient nearest neighbor search

        Args:
            embeddings: (num_examples, embedding_dim) numpy array
        """
        num_examples = embeddings.shape[0]

        # For large datasets (>1M), use IVF index for speed
        # For smaller datasets, use flat index for accuracy
        if num_examples > 1_000_000:
            # IVF index: partitions space into cells
            nlist = int(np.sqrt(num_examples))  # Number of cells
            quantizer = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                nlist,
                faiss.METRIC_INNER_PRODUCT
            )

            # Train index (required for IVF)
            print("Training FAISS index...")
            self.index.train(embeddings)
        else:
            # Flat index: exact search
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Add embeddings to index
        print(f"Adding {num_examples} embeddings to index...")
        self.index.add(embeddings)

        self.dataset_embeddings = embeddings

        print(f"Index built with {self.index.ntotal} vectors")

    def mine_hard_negatives(self, query_ids, positive_ids=None, k=None):
        """
        Mine hard negatives for given queries

        Args:
            query_ids: List of query example IDs
            positive_ids: List of lists of positive IDs (to exclude)
            k: Number of hard negatives to mine (default: self.num_hard_negatives)

        Returns:
            hard_negatives: Dict mapping query_id to list of hard negative IDs
        """
        if k is None:
            k = self.num_hard_negatives

        hard_negatives = {}

        for idx, query_id in enumerate(query_ids):
            # Get query embedding
            query_embedding = self.dataset_embeddings[query_id:query_id+1]

            # Search for k+N nearest neighbors (some may be positives to exclude)
            # Get extra to account for self and positives
            search_k = k + 10

            scores, indices = self.index.search(query_embedding, search_k)

            # Filter out self and known positives
            hard_neg_ids = []

            for score, idx_result in zip(scores[0], indices[0]):
                # Skip self
                if idx_result == query_id:
                    continue

                # Skip known positives
                if positive_ids and idx_result in positive_ids[idx]:
                    continue

                hard_neg_ids.append(idx_result)

                if len(hard_neg_ids) >= k:
                    break

            hard_negatives[query_id] = hard_neg_ids

        return hard_negatives

    def refresh_hard_negatives(self, model, dataloader, device,
                               positive_pairs=None):
        """
        Full refresh: encode dataset, build index, mine negatives

        Args:
            model: Current embedding model
            dataloader: DataLoader for full dataset
            device: torch device
            positive_pairs: Dict mapping query_id to list of positive_ids

        Returns:
            hard_negatives: Dict mapping each example to hard negative IDs
        """
        print("Encoding dataset with current model...")
        embeddings, ids = self.encode_dataset(model, dataloader, device)

        print("Building FAISS index...")
        self.build_index(embeddings)

        print("Mining hard negatives...")
        query_ids = list(range(len(ids)))
        positive_ids_list = [positive_pairs.get(i, []) for i in query_ids] if positive_pairs else None

        hard_negatives = self.mine_hard_negatives(
            query_ids,
            positive_ids=positive_ids_list
        )

        print(f"Mined {self.num_hard_negatives} hard negatives for {len(hard_negatives)} examples")

        return hard_negatives


# Training with offline hard negative mining
class HardNegativeDataset(data.Dataset):
    """
    Dataset that uses pre-mined hard negatives
    """

    def __init__(self, base_dataset, hard_negatives_map):
        """
        Args:
            base_dataset: Original dataset
            hard_negatives_map: Dict mapping example_id to list of hard negative IDs
        """
        self.base_dataset = base_dataset
        self.hard_negatives_map = hard_negatives_map

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get anchor and positive
        item = self.base_dataset[idx]

        # Get hard negatives
        hard_neg_ids = self.hard_negatives_map.get(idx, [])

        if hard_neg_ids:
            # Sample one hard negative
            neg_id = np.random.choice(hard_neg_ids)
            negative = self.base_dataset[neg_id]
        else:
            # Fallback: random negative
            neg_id = np.random.randint(0, len(self.base_dataset))
            while neg_id == idx:
                neg_id = np.random.randint(0, len(self.base_dataset))
            negative = self.base_dataset[neg_id]

        return {
            'anchor': item['text'],
            'positive': item['positive'],  # Assume dataset provides positives
            'negative': negative['text']
        }


# Usage in training loop
def train_with_offline_hard_negatives(model, base_dataset, device, num_epochs=10):
    """
    Training loop with periodic hard negative mining
    """
    hard_negative_miner = OfflineHardNegativeMining(
        embedding_dim=model.embedding_dim,
        num_hard_negatives=10
    )

    # Mine hard negatives every epoch
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")

        # Refresh hard negatives
        dataloader = torch.utils.data.DataLoader(
            base_dataset,
            batch_size=256,
            shuffle=False
        )

        hard_negatives = hard_negative_miner.refresh_hard_negatives(
            model, dataloader, device
        )

        # Create dataset with hard negatives
        train_dataset = HardNegativeDataset(base_dataset, hard_negatives)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True
        )

        # Train epoch
        model.train()
        for batch in train_loader:
            # Training step with anchor, positive, hard negative
            # ... (standard training loop)
            pass
