# Code from Chapter 08
# Book: Embeddings at Scale
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FederatedEmbeddingServer:
    """
    Central server for federated embedding learning

    Coordinates training across multiple clients (hospitals, banks, companies)
    without accessing their raw data

    Protocol:
    1. Server sends current model to clients
    2. Clients train on local data
    3. Clients send updates (gradients or model weights) to server
    4. Server aggregates updates (FedAvg, FedProx, etc.)
    5. Repeat

    Applications:
    - Healthcare: Learn from patient data across hospitals
    - Finance: Fraud detection across banks
    - Mobile: Keyboard prediction across users' devices
    - Enterprise: Cross-company analytics (supply chain, standards bodies)

    Privacy guarantees:
    - Clients never share raw data
    - Optional: Differential privacy on updates
    - Optional: Secure aggregation (encrypted updates)
    """

    def __init__(self, embedding_model, num_clients, aggregation_method="fedavg"):
        """
        Args:
            embedding_model: Global embedding model architecture
            num_clients: Number of participating clients
            aggregation_method: 'fedavg', 'fedprox', or 'fedadam'
        """
        self.global_model = embedding_model
        self.num_clients = num_clients
        self.aggregation_method = aggregation_method

        # Track client participation
        self.client_data_sizes = {}

        # For FedProx: penalty term for client drift
        self.mu = 0.01 if aggregation_method == "fedprox" else 0

    def get_global_model(self):
        """Send current global model to clients"""
        return deepcopy(self.global_model)

    def aggregate_updates(self, client_models, client_data_sizes):
        """
        Aggregate client model updates into global model

        FedAvg: Weighted average by data size

        Args:
            client_models: List of updated models from clients
            client_data_sizes: Number of samples each client trained on

        Returns:
            Updated global model
        """
        if self.aggregation_method == "fedavg":
            return self._fedavg_aggregate(client_models, client_data_sizes)
        elif self.aggregation_method == "fedprox":
            return self._fedprox_aggregate(client_models, client_data_sizes)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation_method}")

    def _fedavg_aggregate(self, client_models, client_data_sizes):
        """
        FedAvg: Weighted average of client models

        Weight by number of samples (clients with more data have more influence)
        """
        total_data = sum(client_data_sizes)

        # Initialize aggregated weights
        global_state = self.global_model.state_dict()

        # Weighted average
        for key in global_state:
            # Start with zeros
            global_state[key] = torch.zeros_like(global_state[key])

            # Add weighted contribution from each client
            for client_model, client_size in zip(client_models, client_data_sizes):
                weight = client_size / total_data
                client_state = client_model.state_dict()
                global_state[key] += weight * client_state[key]

        # Update global model
        self.global_model.load_state_dict(global_state)

        return self.global_model

    def _fedprox_aggregate(self, client_models, client_data_sizes):
        """
        FedProx: FedAvg + regularization term

        Prevents clients from drifting too far from global model
        Better for heterogeneous data distributions
        """
        # Same as FedAvg (regularization happens on client side)
        return self._fedavg_aggregate(client_models, client_data_sizes)


class FederatedEmbeddingClient:
    """
    Client in federated embedding learning

    Each client (hospital, bank, company, device) trains on local data
    Shares only model updates, never raw data
    """

    def __init__(self, client_id, local_data, local_labels=None, learning_rate=0.01):
        """
        Args:
            client_id: Unique identifier for this client
            local_data: Private data (never leaves this client)
            local_labels: Optional labels for supervised learning
            learning_rate: Learning rate for local training
        """
        self.client_id = client_id
        self.local_data = local_data
        self.local_labels = local_labels
        self.learning_rate = learning_rate

        self.local_model = None
        self.optimizer = None

    def receive_global_model(self, global_model):
        """
        Receive current global model from server

        Initialize local model as copy of global
        """
        self.local_model = deepcopy(global_model)
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.learning_rate)

    def local_train(self, num_epochs=5, batch_size=32):
        """
        Train on local data for several epochs

        Args:
            num_epochs: Number of local epochs before sharing update
            batch_size: Local batch size

        Returns:
            Updated model, number of samples trained on
        """
        self.local_model.train()

        num_samples = len(self.local_data)

        for _epoch in range(num_epochs):
            # Shuffle local data
            indices = torch.randperm(num_samples)

            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_data = self.local_data[batch_indices]

                # Forward pass
                if self.local_labels is not None:
                    # Supervised learning
                    batch_labels = self.local_labels[batch_indices]
                    embeddings = self.local_model(batch_data)
                    loss = F.cross_entropy(embeddings, batch_labels)
                else:
                    # Self-supervised learning (e.g., contrastive)
                    loss = self.local_model(batch_data)  # Model computes own loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self.local_model, num_samples

    def add_differential_privacy(self, epsilon=1.0, delta=1e-5):
        """
        Add differential privacy to model updates

        Ensures that updates don't reveal information about
        individual data points

        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Probability of privacy violation
        """
        # Clip gradients (bound sensitivity)
        max_norm = 1.0
        for param in self.local_model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if grad_norm > max_norm:
                    param.grad.mul_(max_norm / grad_norm)

        # Add Gaussian noise
        noise_scale = max_norm * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

        for param in self.local_model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad.add_(noise)


def federated_learning_simulation():
    """
    Simulate federated learning across multiple hospitals

    Scenario:
    - 10 hospitals, each with patient data (cannot share due to HIPAA)
    - Want to learn embeddings for medical conditions
    - Each hospital has different patient population (non-IID data)

    Benefits:
    - Learn from 10x more data than any single hospital
    - Preserve patient privacy
    - Each hospital benefits from others' data
    """

    # Simple embedding model
    class MedicalEmbedding(nn.Module):
        def __init__(self, input_dim=100, embedding_dim=64):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, embedding_dim)
            )

        def forward(self, x):
            return self.encoder(x)

    # Initialize server
    global_model = MedicalEmbedding(input_dim=100, embedding_dim=64)
    server = FederatedEmbeddingServer(
        embedding_model=global_model, num_clients=10, aggregation_method="fedavg"
    )

    # Create clients (hospitals)
    clients = []
    for i in range(10):
        # Each hospital has different amount of data (realistic)
        num_patients = np.random.randint(500, 2000)
        local_data = torch.randn(num_patients, 100)  # Patient features
        local_labels = torch.randint(0, 10, (num_patients,))  # Conditions

        client = FederatedEmbeddingClient(
            client_id=i, local_data=local_data, local_labels=local_labels, learning_rate=0.01
        )
        clients.append(client)

    # Federated training loop
    num_rounds = 50

    print("Starting federated training...")

    for round_num in range(num_rounds):
        print(f"\nRound {round_num + 1}/{num_rounds}")

        # 1. Server sends global model to clients
        client_models = []
        client_data_sizes = []

        # 2. Each client trains locally
        for client in clients:
            client.receive_global_model(server.get_global_model())
            updated_model, num_samples = client.local_train(num_epochs=5, batch_size=32)

            # Optional: Add differential privacy
            # client.add_differential_privacy(epsilon=1.0)

            client_models.append(updated_model)
            client_data_sizes.append(num_samples)

        # 3. Server aggregates updates
        server.aggregate_updates(client_models, client_data_sizes)

        print(f"  Aggregated updates from {len(clients)} hospitals")

    print("\nFederated training complete!")
    print("Each hospital now has access to global model")
    print("trained on all hospitals' data without sharing patient information")

    return server.global_model


# Uncomment to run:
# global_model = federated_learning_simulation()
