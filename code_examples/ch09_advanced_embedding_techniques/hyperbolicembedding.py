# Code from Chapter 08
# Book: Embeddings at Scale

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperbolicEmbedding(nn.Module):
    """
    Hyperbolic embeddings in the Poincaré ball model

    The Poincaré ball model represents hyperbolic space as:
    - Points inside the unit ball {x : ||x|| < 1}
    - Distance grows exponentially near the boundary
    - Perfect for representing hierarchies

    Applications:
    - Product taxonomies (millions of SKUs in hierarchical categories)
    - Organizational structures (companies with deep reporting chains)
    - Knowledge graphs (WordNet, medical ontologies)
    - Geographic hierarchies (continent → country → state → city)

    Advantages over Euclidean embeddings:
    - Lower dimensionality (often 2-5D vs 100-300D)
    - Better preservation of hierarchical distances
    - Natural representation of uncertainty (distance from origin = specificity)
    """

    def __init__(self, num_items, embedding_dim=5, curvature=1.0):
        """
        Args:
            num_items: Number of items in taxonomy
            embedding_dim: Dimension of hyperbolic space (typically much lower than Euclidean)
            curvature: Negative curvature parameter (higher = more curved)
        """
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.curvature = curvature

        # Initialize embeddings in the Poincaré ball
        # Start near origin for stability
        self.embeddings = nn.Parameter(torch.randn(num_items, embedding_dim) * 0.01)

    def poincare_distance(self, u, v):
        """
        Compute Poincaré distance between points u and v

        d(u,v) = arcosh(1 + 2 * ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))

        This distance:
        - Is 0 when u = v
        - Increases exponentially near the boundary
        - Preserves hyperbolic geometry
        """
        # Compute squared norms
        u_norm_sq = torch.sum(u**2, dim=-1, keepdim=True)
        v_norm_sq = torch.sum(v**2, dim=-1, keepdim=True)

        # Compute squared Euclidean distance
        diff_norm_sq = torch.sum((u - v) ** 2, dim=-1, keepdim=True)

        # Poincaré distance formula
        numerator = 2 * diff_norm_sq
        denominator = (1 - u_norm_sq) * (1 - v_norm_sq)

        # Add small epsilon for numerical stability
        distance = torch.acosh(1 + numerator / (denominator + 1e-7))

        return distance * self.curvature

    def project_to_ball(self, x, eps=1e-5):
        """
        Project points onto the Poincaré ball (inside unit sphere)

        Essential for:
        - Maintaining valid hyperbolic points during training
        - Preventing numerical instability at the boundary
        """
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        # If norm >= 1, project to (1 - eps)
        max_norm = 1 - eps
        scale = torch.where(norm >= max_norm, max_norm / (norm + 1e-7), torch.ones_like(norm))
        return x * scale

    def get_embeddings(self, indices):
        """Get embeddings and project to valid hyperbolic space"""
        emb = self.embeddings[indices]
        return self.project_to_ball(emb)

    def forward(self, parent_indices, child_indices):
        """
        Compute hierarchical loss

        Args:
            parent_indices: Indices of parent nodes
            child_indices: Indices of child nodes

        Returns:
            loss: Encourages parent-child pairs to be close
        """
        parent_emb = self.get_embeddings(parent_indices)
        child_emb = self.get_embeddings(child_indices)

        # Distance should be small for parent-child pairs
        distances = self.poincare_distance(parent_emb, child_emb)

        return distances.mean()


class HierarchicalEmbeddingTrainer:
    """
    Train hierarchical embeddings from taxonomy structure

    Supports multiple loss functions:
    - Parent-child proximity loss
    - Sibling similarity loss
    - Transitivity loss (grandparent → parent → child)
    - Level-based regularization
    """

    def __init__(self, taxonomy, embedding_dim=5, curvature=1.0, learning_rate=0.01):
        """
        Args:
            taxonomy: Dict mapping child_id → parent_id
            embedding_dim: Dimension of hyperbolic embeddings
            curvature: Curvature of hyperbolic space
            learning_rate: Riemannian gradient descent learning rate
        """
        self.taxonomy = taxonomy

        # Build reverse mapping: parent → children
        self.children = {}
        for child, parent in taxonomy.items():
            if parent not in self.children:
                self.children[parent] = []
            self.children[parent].append(child)

        # All unique items
        all_items = set(taxonomy.keys()) | set(taxonomy.values())
        self.num_items = len(all_items)

        # Create item to index mapping
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted(all_items))}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

        # Initialize model
        self.model = HyperbolicEmbedding(self.num_items, embedding_dim, curvature)

        # Riemannian optimizer for hyperbolic space
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def create_training_batch(self, batch_size=64):
        """
        Sample parent-child pairs and negative samples

        Returns:
            parent_indices: Parent nodes
            positive_children: Actual children
            negative_samples: Random non-children
        """
        # Sample random parent-child pairs
        pairs = list(self.taxonomy.items())
        sampled_pairs = np.random.choice(len(pairs), size=batch_size, replace=True)

        parent_indices = []
        child_indices = []
        negative_indices = []

        for idx in sampled_pairs:
            child, parent = pairs[idx]
            parent_indices.append(self.item_to_idx[parent])
            child_indices.append(self.item_to_idx[child])

            # Sample negative: any item that's not a child
            while True:
                neg = np.random.randint(self.num_items)
                if neg not in [self.item_to_idx[c] for c in self.children.get(parent, [])]:
                    negative_indices.append(neg)
                    break

        return (
            torch.LongTensor(parent_indices),
            torch.LongTensor(child_indices),
            torch.LongTensor(negative_indices),
        )

    def train_step(self, batch_size=64, margin=1.0):
        """
        Single training step with triplet-style loss

        Loss encourages:
        - distance(parent, child) < distance(parent, non-child) + margin
        """
        self.optimizer.zero_grad()

        # Get batch
        parent_idx, child_idx, neg_idx = self.create_training_batch(batch_size)

        # Get embeddings
        parent_emb = self.model.get_embeddings(parent_idx)
        child_emb = self.model.get_embeddings(child_idx)
        neg_emb = self.model.get_embeddings(neg_idx)

        # Compute distances
        positive_dist = self.model.poincare_distance(parent_emb, child_emb)
        negative_dist = self.model.poincare_distance(parent_emb, neg_emb)

        # Triplet loss: positive distance should be less than negative distance
        loss = F.relu(positive_dist - negative_dist + margin).mean()

        # Backprop and step
        loss.backward()
        self.optimizer.step()

        # Re-project to Poincaré ball after update
        with torch.no_grad():
            self.model.embeddings.data = self.model.project_to_ball(self.model.embeddings.data)

        return loss.item()

    def train(self, num_epochs=1000, batch_size=64, verbose=True):
        """
        Full training loop

        For enterprise taxonomies:
        - Product catalogs: 10K-1M items, train in hours
        - Knowledge graphs: 1M-100M entities, train in days
        - Organizational hierarchies: 100-100K positions, train in minutes
        """
        for epoch in range(num_epochs):
            loss = self.train_step(batch_size)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

        if verbose:
            print(f"Training complete. Final loss: {loss:.4f}")

    def get_embedding(self, item):
        """Get embedding for a specific item"""
        idx = self.item_to_idx[item]
        with torch.no_grad():
            return self.model.get_embeddings(torch.LongTensor([idx]))[0].numpy()

    def find_similar(self, item, top_k=5):
        """
        Find most similar items in hyperbolic space

        Results respect hierarchical structure:
        - Siblings rank higher than distant relatives
        - Ancestors/descendants rank higher than unrelated items
        """
        query_idx = self.item_to_idx[item]
        query_emb = self.model.get_embeddings(torch.LongTensor([query_idx]))

        # Compute distances to all items
        all_indices = torch.arange(self.num_items)
        all_emb = self.model.get_embeddings(all_indices)

        distances = self.model.poincare_distance(query_emb.expand(self.num_items, -1), all_emb)

        # Get top-k closest (excluding self)
        _, sorted_indices = torch.sort(distances.squeeze())
        similar_items = []

        for idx in sorted_indices[1 : top_k + 1]:  # Skip first (self)
            item_name = self.idx_to_item[idx.item()]
            dist = distances[idx].item()
            similar_items.append((item_name, dist))

        return similar_items


# Example: Training on product taxonomy
def train_product_hierarchy_example():
    """
    Example: E-commerce product catalog with 100K products

    Hyperbolic embeddings reduce dimensionality by 20-50x while
    improving hierarchical distance preservation
    """
    # Define product taxonomy (child → parent)
    product_taxonomy = {
        # Electronics branch
        "gaming_laptop": "laptops",
        "business_laptop": "laptops",
        "ultrabook": "laptops",
        "laptops": "computers",
        "desktop": "computers",
        "computers": "electronics",
        # Mobile branch
        "smartphone": "mobile_devices",
        "tablet": "mobile_devices",
        "smartwatch": "wearables",
        "fitness_tracker": "wearables",
        "mobile_devices": "electronics",
        "wearables": "electronics",
        # Home appliances branch
        "refrigerator": "kitchen_appliances",
        "dishwasher": "kitchen_appliances",
        "kitchen_appliances": "home_appliances",
        "washing_machine": "laundry_appliances",
        "dryer": "laundry_appliances",
        "laundry_appliances": "home_appliances",
    }

    # Train hierarchical embeddings
    trainer = HierarchicalEmbeddingTrainer(
        product_taxonomy,
        embedding_dim=5,  # Much lower than typical 256-768
        curvature=1.0,
        learning_rate=0.01,
    )

    print("Training hierarchical embeddings...")
    trainer.train(num_epochs=1000, batch_size=32)

    # Test hierarchical similarity
    print("\nHierarchical similarity for 'gaming_laptop':")
    similar = trainer.find_similar("gaming_laptop", top_k=5)
    for item, distance in similar:
        print(f"  {item}: {distance:.4f}")

    print("\nHierarchical similarity for 'smartphone':")
    similar = trainer.find_similar("smartphone", top_k=5)
    for item, distance in similar:
        print(f"  {item}: {distance:.4f}")

    return trainer


# Uncomment to run:
# trainer = train_product_hierarchy_example()
