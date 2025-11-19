import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Code from Chapter 06
# Book: Embeddings at Scale


# Placeholder function for creating Siamese network
def create_enterprise_siamese_network(input_type="tabular", input_dim=50):
    """Create enterprise Siamese network. Placeholder implementation."""

    class PlaceholderSiameseNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64))

        def forward(self, x):
            return self.encoder(x)

    return PlaceholderSiameseNetwork()


class OneShotClassifier:
    """
    One-shot classifier using Siamese networks

    Approach:
    1. Train Siamese network on abundant classes
    2. At inference, store one example per new class (support set)
    3. Classify by finding most similar support example

    Use cases:
    - Fraud detection: New fraud patterns from single examples
    - Manufacturing QA: Rare defect types
    - Security: Zero-day threat detection
    - Customer service: New issue categories
    """

    def __init__(self, siamese_model, distance_metric="euclidean"):
        """
        Args:
            siamese_model: Trained Siamese network
            distance_metric: 'euclidean' or 'cosine'
        """
        self.model = siamese_model
        self.distance_metric = distance_metric
        self.support_set = {}  # class_id -> embedding

    def add_support_example(self, class_id, example):
        """
        Add a single example for a new class

        Args:
            class_id: Identifier for the class
            example: Input data (will be embedded)
        """
        with torch.no_grad():
            self.model.eval()
            embedding = self.model.get_embedding(example)
            self.support_set[class_id] = embedding.cpu()

    def add_support_examples_batch(self, class_ids, examples):
        """
        Add multiple examples (potentially multiple per class)

        Args:
            class_ids: List of class identifiers
            examples: Batch of input data
        """
        with torch.no_grad():
            self.model.eval()
            embeddings = self.model.get_embedding(examples)

            # Average embeddings for classes with multiple examples
            for class_id, embedding in zip(class_ids, embeddings):
                if class_id in self.support_set:
                    # Average with existing embedding
                    self.support_set[class_id] = (self.support_set[class_id] + embedding.cpu()) / 2
                else:
                    self.support_set[class_id] = embedding.cpu()

    def predict(self, query, return_distances=False, top_k=1):
        """
        Predict class for query by finding nearest support example

        Args:
            query: Input to classify
            return_distances: If True, return distances along with predictions
            top_k: Return top-k most similar classes

        Returns:
            If return_distances=False: class_id or list of class_ids
            If return_distances=True: (class_ids, distances)
        """
        with torch.no_grad():
            self.model.eval()
            query_embedding = self.model.get_embedding(query)

            if len(self.support_set) == 0:
                raise ValueError("No support examples added. Call add_support_example first.")

            # Compute distances to all support examples
            distances = {}
            for class_id, support_embedding in self.support_set.items():
                support_embedding = support_embedding.to(query_embedding.device)

                if self.distance_metric == "euclidean":
                    dist = F.pairwise_distance(
                        query_embedding, support_embedding.unsqueeze(0)
                    ).item()
                else:  # cosine
                    dist = (
                        1 - F.cosine_similarity(query_embedding, support_embedding.unsqueeze(0))
                    ).item()

                distances[class_id] = dist

            # Sort by distance (ascending)
            sorted_classes = sorted(distances.items(), key=lambda x: x[1])

            if top_k == 1:
                result = sorted_classes[0]
                if return_distances:
                    return result[0], result[1]
                else:
                    return result[0]
            else:
                results = sorted_classes[:top_k]
                if return_distances:
                    class_ids, dists = zip(*results)
                    return list(class_ids), list(dists)
                else:
                    return [c for c, _ in results]

    def predict_proba(self, query, temperature=1.0):
        """
        Predict class probabilities using softmax over distances

        Args:
            query: Input to classify
            temperature: Controls distribution sharpness (lower = sharper)

        Returns:
            Dict mapping class_id to probability
        """
        class_ids, distances = self.predict(
            query, return_distances=True, top_k=len(self.support_set)
        )

        # Convert distances to similarities (negative distance)
        similarities = [-d / temperature for d in distances]

        # Softmax
        exp_sims = np.exp(similarities - np.max(similarities))  # Numerical stability
        probabilities = exp_sims / exp_sims.sum()

        return dict(zip(class_ids, probabilities))


# Example: Fraud detection with one-shot learning
class FraudDetectionOneShot:
    """
    Fraud detection system using one-shot learning

    Scenario: New fraud patterns emerge daily. We can't wait to collect
    thousands of examples. One-shot learning enables immediate detection.
    """

    def __init__(self, siamese_model):
        self.classifier = OneShotClassifier(siamese_model)
        self.normal_behavior_embedding = None

    def set_normal_behavior(self, normal_transactions):
        """
        Learn embedding for normal transactions

        Args:
            normal_transactions: Batch of normal transaction features
        """
        with torch.no_grad():
            embeddings = self.classifier.model.get_embedding(normal_transactions)
            # Average embedding represents "normal"
            self.normal_behavior_embedding = embeddings.mean(dim=0)

    def add_fraud_pattern(self, fraud_id, fraud_transaction):
        """
        Add a new fraud pattern from a single example

        Args:
            fraud_id: Identifier for this fraud type
            fraud_transaction: Features of the fraud transaction
        """
        self.classifier.add_support_example(fraud_id, fraud_transaction)

    def check_transaction(self, transaction, fraud_threshold=0.7):
        """
        Check if transaction is fraudulent

        Args:
            transaction: Transaction features to check
            fraud_threshold: Similarity threshold for fraud detection

        Returns:
            Dict with:
            - is_fraud: Boolean
            - fraud_type: ID of fraud type if detected
            - confidence: Probability
            - normal_similarity: How similar to normal behavior
        """
        # Get probabilities for all known fraud types
        fraud_probs = self.classifier.predict_proba(transaction)

        # Get distance to normal behavior
        with torch.no_grad():
            transaction_embedding = self.classifier.model.get_embedding(transaction)
            normal_similarity = F.cosine_similarity(
                transaction_embedding, self.normal_behavior_embedding.unsqueeze(0)
            ).item()

        # If very similar to normal, not fraud
        if normal_similarity > 0.9:
            return {
                "is_fraud": False,
                "fraud_type": None,
                "confidence": 0.0,
                "normal_similarity": normal_similarity,
            }

        # Check if similar to any fraud pattern
        max_fraud_type = max(fraud_probs.items(), key=lambda x: x[1])
        fraud_type, confidence = max_fraud_type

        is_fraud = confidence > fraud_threshold

        return {
            "is_fraud": is_fraud,
            "fraud_type": fraud_type if is_fraud else None,
            "confidence": confidence,
            "normal_similarity": normal_similarity,
        }


# Example usage
def example_fraud_detection():
    """Complete example of fraud detection with one-shot learning"""

    # 1. Train Siamese network on historical fraud patterns
    # (Assume this is already done)
    siamese_model = create_enterprise_siamese_network(
        input_type="tabular",
        input_dim=50,  # 50 transaction features
    )

    # 2. Create fraud detection system
    fraud_detector = FraudDetectionOneShot(siamese_model)

    # 3. Learn normal behavior from legitimate transactions
    normal_transactions = torch.randn(1000, 50)  # 1000 normal examples
    fraud_detector.set_normal_behavior(normal_transactions)

    # 4. Add known fraud patterns (one example each!)
    fraud_detector.add_fraud_pattern(fraud_id="card_testing", fraud_transaction=torch.randn(1, 50))
    fraud_detector.add_fraud_pattern(
        fraud_id="account_takeover", fraud_transaction=torch.randn(1, 50)
    )

    # 5. Check new transactions
    new_transaction = torch.randn(1, 50)
    result = fraud_detector.check_transaction(new_transaction)

    print(f"Is fraud: {result['is_fraud']}")
    print(f"Fraud type: {result['fraud_type']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Normal similarity: {result['normal_similarity']:.2f}")
