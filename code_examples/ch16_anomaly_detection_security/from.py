# Code from Chapter 16
# Book: Embeddings at Scale

"""
Embedding-Based Fraud Detection

Architecture:
1. Transaction encoder: Maps transactions to embeddings
2. User/merchant encoders: Embeddings for entities
3. Graph embeddings: Capture transaction network
4. Anomaly scoring: Distance-based or density-based

Techniques:
- Autoencoder: Reconstruct transactions, high reconstruction error = anomaly
- Isolation Forest: Embeddings as features for isolation
- LSTM: Sequential transaction patterns, flag deviations
- Graph Neural Networks: Network structure for money laundering

Production considerations:
- Online learning: Update embeddings as new transactions arrive
- Low latency: <50ms per transaction for real-time blocking
- Explainability: Surface features causing high anomaly score
- False positive management: Balance detection vs user friction
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Transaction:
    """
    Financial transaction

    Attributes:
        transaction_id: Unique identifier
        user_id: User making transaction
        merchant_id: Merchant receiving payment
        amount: Transaction amount
        timestamp: When transaction occurred
        location: Transaction location (lat, lon)
        device_id: Device used
        features: Additional features (category, etc.)
        is_fraud: Ground truth label (if available)
    """
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    timestamp: float
    location: Optional[Tuple[float, float]] = None
    device_id: Optional[str] = None
    features: Dict[str, any] = None
    is_fraud: Optional[bool] = None

    def __post_init__(self):
        if self.features is None:
            self.features = {}

class TransactionEncoder(nn.Module):
    """
    Encode transactions to embeddings

    Architecture:
    - Numerical features: Amount, time of day, day of week
    - Categorical features: Merchant category, location, device
    - User/merchant embeddings: Entity representations
    - MLP: Combine features into transaction embedding

    Training:
    - Autoencoder: Reconstruct transaction features
    - Contrastive learning: Similar transactions close, different transactions far
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_users: int = 1000000,
        num_merchants: int = 100000,
        num_devices: int = 10000
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Entity embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim // 4)
        self.merchant_embedding = nn.Embedding(num_merchants, embedding_dim // 4)
        self.device_embedding = nn.Embedding(num_devices, embedding_dim // 4)

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(embedding_dim // 4 + 10, 128),  # +10 for numerical features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        merchant_ids: torch.Tensor,
        device_ids: torch.Tensor,
        numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode transactions

        Args:
            user_ids: User IDs (batch_size,)
            merchant_ids: Merchant IDs (batch_size,)
            device_ids: Device IDs (batch_size,)
            numerical_features: Numerical features (batch_size, num_features)

        Returns:
            Transaction embeddings (batch_size, embedding_dim)
        """
        # Embed entities
        user_emb = self.user_embedding(user_ids)
        merchant_emb = self.merchant_embedding(merchant_ids)
        device_emb = self.device_embedding(device_ids)

        # Combine entity embeddings
        entity_emb = (user_emb + merchant_emb + device_emb) / 3.0

        # Concatenate with numerical features
        combined = torch.cat([entity_emb, numerical_features], dim=1)

        # Encode
        transaction_emb = self.feature_encoder(combined)

        # Normalize
        transaction_emb = F.normalize(transaction_emb, p=2, dim=1)

        return transaction_emb

class TransactionAutoencoder(nn.Module):
    """
    Autoencoder for fraud detection

    Architecture:
    - Encoder: Transaction → low-dim embedding
    - Decoder: Embedding → reconstructed transaction
    - Training: Minimize reconstruction error on normal transactions

    Inference:
    - Encode transaction
    - Compute reconstruction error
    - High error = anomaly (fraud)

    Why it works:
    - Normal transactions have low reconstruction error (seen during training)
    - Fraud transactions have high error (novel patterns)
    """

    def __init__(
        self,
        input_dim: int = 128,
        latent_dim: int = 32
    ):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode and decode

        Args:
            x: Input embeddings (batch_size, input_dim)

        Returns:
            (latent, reconstructed) embeddings
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score (reconstruction error)

        Args:
            x: Input embeddings (batch_size, input_dim)

        Returns:
            Anomaly scores (batch_size,)
        """
        _, reconstructed = self.forward(x)
        # MSE reconstruction error
        scores = ((x - reconstructed) ** 2).mean(dim=1)
        return scores

class FraudDetectionSystem:
    """
    Production fraud detection system

    Components:
    1. Transaction encoder: Transaction → embedding
    2. Anomaly detector: Autoencoder or distance-based
    3. Threshold calibration: Set score threshold for alerts
    4. Online learning: Update model with new transactions

    Features:
    - Real-time scoring (<50ms)
    - Online model updates
    - Explainability: Feature attribution
    - Feedback loop: Incorporate fraud labels
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        anomaly_threshold: float = 0.95,  # 95th percentile
        device: str = 'cuda'
    ):
        """
        Args:
            embedding_dim: Embedding dimension
            anomaly_threshold: Percentile for anomaly cutoff
            device: Device for computation
        """
        self.embedding_dim = embedding_dim
        self.anomaly_threshold = anomaly_threshold
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Transaction encoder
        self.transaction_encoder = TransactionEncoder(
            embedding_dim=embedding_dim
        ).to(self.device)

        # Autoencoder for anomaly detection
        self.autoencoder = TransactionAutoencoder(
            input_dim=embedding_dim,
            latent_dim=32
        ).to(self.device)

        # Threshold (learned from normal transactions)
        self.score_threshold: Optional[float] = None

        # Entity mappings
        self.user_id_to_idx: Dict[str, int] = {}
        self.merchant_id_to_idx: Dict[str, int] = {}
        self.device_id_to_idx: Dict[str, int] = {}

        # Statistics for online updates
        self.transaction_count = 0
        self.fraud_count = 0

        print("Initialized Fraud Detection System")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Anomaly threshold: {anomaly_threshold}")

    def build_entity_mappings(self, transactions: List[Transaction]):
        """Build entity ID to index mappings"""
        users = set(t.user_id for t in transactions)
        merchants = set(t.merchant_id for t in transactions)
        devices = set(t.device_id for t in transactions if t.device_id)

        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(users)}
        self.merchant_id_to_idx = {mid: idx for idx, mid in enumerate(merchants)}
        self.device_id_to_idx = {did: idx for idx, did in enumerate(devices)}

        print("Built entity mappings:")
        print(f"  Users: {len(self.user_id_to_idx)}")
        print(f"  Merchants: {len(self.merchant_id_to_idx)}")
        print(f"  Devices: {len(self.device_id_to_idx)}")

    def extract_features(self, transaction: Transaction) -> np.ndarray:
        """
        Extract numerical features from transaction

        Features:
        - Log amount
        - Hour of day (normalized)
        - Day of week (normalized)
        - Days since user's first transaction
        - Transaction velocity (# transactions in last hour)
        - Amount deviation from user's average
        """
        # Simple feature extraction (in production: more features)
        hour = (transaction.timestamp % 86400) / 3600  # Hour of day
        day_of_week = ((transaction.timestamp // 86400) % 7) / 7  # Day of week
        log_amount = np.log1p(transaction.amount)

        # Placeholder for additional features
        features = np.array([
            log_amount / 10.0,  # Normalize
            hour / 24.0,
            day_of_week,
            0.0,  # Placeholder: days since first transaction
            0.0,  # Placeholder: transaction velocity
            0.0,  # Placeholder: amount deviation
            0.0, 0.0, 0.0, 0.0  # Additional feature placeholders
        ], dtype=np.float32)

        return features

    def train_autoencoder(
        self,
        transactions: List[Transaction],
        num_epochs: int = 10,
        batch_size: int = 256
    ):
        """
        Train autoencoder on normal transactions

        Args:
            transactions: Training transactions (should be mostly normal)
            num_epochs: Training epochs
            batch_size: Batch size
        """
        print(f"\nTraining autoencoder on {len(transactions)} transactions...")

        # Build mappings
        self.build_entity_mappings(transactions)

        # Prepare data
        embeddings = []

        for transaction in transactions:
            # Get indices
            user_idx = self.user_id_to_idx.get(transaction.user_id, 0)
            merchant_idx = self.merchant_id_to_idx.get(transaction.merchant_id, 0)
            device_idx = self.device_id_to_idx.get(transaction.device_id, 0)

            # Extract features
            num_features = self.extract_features(transaction)

            # Encode
            user_ids = torch.tensor([user_idx], dtype=torch.long).to(self.device)
            merchant_ids = torch.tensor([merchant_idx], dtype=torch.long).to(self.device)
            device_ids = torch.tensor([device_idx], dtype=torch.long).to(self.device)
            num_features_tensor = torch.from_numpy(num_features).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.transaction_encoder(user_ids, merchant_ids, device_ids, num_features_tensor)
                embeddings.append(emb.cpu().numpy()[0])

        embeddings = np.array(embeddings)
        embeddings_tensor = torch.from_numpy(embeddings).to(self.device)

        # Train autoencoder
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.autoencoder.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            # Mini-batch training
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings_tensor[i:i+batch_size]

                # Forward pass
                _, reconstructed = self.autoencoder(batch)

                # Compute loss
                loss = criterion(reconstructed, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        print("✓ Training complete")

        # Calibrate threshold on training data
        self._calibrate_threshold(embeddings_tensor)

    def _calibrate_threshold(self, embeddings: torch.Tensor):
        """
        Calibrate anomaly score threshold

        Sets threshold at specified percentile of scores on normal data

        Args:
            embeddings: Normal transaction embeddings
        """
        self.autoencoder.eval()

        with torch.no_grad():
            scores = self.autoencoder.compute_anomaly_score(embeddings)
            scores_np = scores.cpu().numpy()

        # Set threshold at specified percentile
        self.score_threshold = np.percentile(scores_np, self.anomaly_threshold * 100)

        print(f"✓ Calibrated threshold: {self.score_threshold:.4f}")
        print(f"  {self.anomaly_threshold:.1%} of training data below threshold")

    def detect_fraud(
        self,
        transaction: Transaction,
        return_score: bool = False
    ) -> Tuple[bool, float]:
        """
        Detect if transaction is fraudulent

        Args:
            transaction: Transaction to check
            return_score: Whether to return anomaly score

        Returns:
            (is_fraud, anomaly_score)
        """
        if self.score_threshold is None:
            raise ValueError("Model not trained. Call train_autoencoder() first.")

        self.transaction_encoder.eval()
        self.autoencoder.eval()

        # Get indices
        user_idx = self.user_id_to_idx.get(transaction.user_id, 0)
        merchant_idx = self.merchant_id_to_idx.get(transaction.merchant_id, 0)
        device_idx = self.device_id_to_idx.get(transaction.device_id, 0)

        # Extract features
        num_features = self.extract_features(transaction)

        # Encode transaction
        user_ids = torch.tensor([user_idx], dtype=torch.long).to(self.device)
        merchant_ids = torch.tensor([merchant_idx], dtype=torch.long).to(self.device)
        device_ids = torch.tensor([device_idx], dtype=torch.long).to(self.device)
        num_features_tensor = torch.from_numpy(num_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.transaction_encoder(user_ids, merchant_ids, device_ids, num_features_tensor)

            # Compute anomaly score
            anomaly_score = self.autoencoder.compute_anomaly_score(emb).item()

        # Flag if score above threshold
        is_fraud = anomaly_score > self.score_threshold

        self.transaction_count += 1
        if is_fraud:
            self.fraud_count += 1

        return is_fraud, anomaly_score

# Example: Credit card fraud detection
def fraud_detection_example():
    """
    Credit card fraud detection

    Use case:
    - 10M transactions/day
    - 0.1% fraud rate (10K fraud transactions)
    - Detect fraud in real-time (<50ms)

    Challenge: Highly imbalanced (99.9% normal)

    Approach: Autoencoder trained on normal transactions
    """

    # Initialize system
    system = FraudDetectionSystem(embedding_dim=64, anomaly_threshold=0.95)

    # Generate synthetic normal transactions
    normal_transactions = []
    for i in range(1000):
        transaction = Transaction(
            transaction_id=f'txn_{i}',
            user_id=f'user_{i % 100}',
            merchant_id=f'merchant_{i % 50}',
            amount=20 + np.random.rand() * 100,  # $20-$120
            timestamp=time.time() - (1000 - i) * 3600,  # Last 1000 hours
            device_id=f'device_{i % 200}',
            is_fraud=False
        )
        normal_transactions.append(transaction)

    print("=== Training Fraud Detection System ===")

    # Train on normal transactions
    system.train_autoencoder(normal_transactions, num_epochs=5, batch_size=64)

    # Test on normal transaction
    print("\n=== Testing on Normal Transaction ===")
    test_normal = Transaction(
        transaction_id='test_normal',
        user_id='user_0',
        merchant_id='merchant_0',
        amount=50.0,
        timestamp=time.time(),
        device_id='device_0'
    )

    is_fraud, score = system.detect_fraud(test_normal)
    print(f"Transaction: ${test_normal.amount:.2f}")
    print(f"Anomaly score: {score:.4f}")
    print(f"Fraud detected: {is_fraud}")

    # Test on anomalous transaction
    print("\n=== Testing on Anomalous Transaction ===")
    test_fraud = Transaction(
        transaction_id='test_fraud',
        user_id='user_999',  # New user
        merchant_id='merchant_99',  # New merchant
        amount=5000.0,  # Large amount
        timestamp=time.time(),
        device_id='device_999'  # New device
    )

    is_fraud, score = system.detect_fraud(test_fraud)
    print(f"Transaction: ${test_fraud.amount:.2f}")
    print(f"Anomaly score: {score:.4f}")
    print(f"Fraud detected: {is_fraud}")

    # Statistics
    print("\n=== System Statistics ===")
    print(f"Total transactions processed: {system.transaction_count}")
    print(f"Fraud detected: {system.fraud_count}")
    print(f"Fraud rate: {system.fraud_count / system.transaction_count:.2%}")

# Uncomment to run:
# fraud_detection_example()
