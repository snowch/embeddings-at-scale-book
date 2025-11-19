# Code from Chapter 16
# Book: Embeddings at Scale

"""
Cybersecurity Threat Hunting with Embeddings

Architecture:
1. Event encoder: Network/system events → embeddings
2. User behavior model: Sequence of user actions
3. Device baseline: Normal device activity patterns
4. Anomaly detection: Deviation from user/device baselines

Event types:
- Login events (success, failure, location, device)
- File access events (read, write, delete)
- Network events (connections, data transfers)
- Process events (exec, terminate)

Threats detected:
- Account compromise (unusual login location, time, device)
- Lateral movement (unusual process execution, network connections)
- Data exfiltration (unusual outbound transfers)
- Insider threats (access to sensitive files outside normal behavior)
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SecurityEvent:
    """
    Security event from logs

    Attributes:
        event_id: Unique identifier
        event_type: Type (login, file_access, network, process)
        user_id: User associated with event
        device_id: Device associated with event
        timestamp: When event occurred
        features: Event-specific features
        is_malicious: Ground truth label (if available)
    """
    event_id: str
    event_type: str
    user_id: str
    device_id: str
    timestamp: float
    features: Dict[str, any]
    is_malicious: Optional[bool] = None

class UserBehaviorModel(nn.Module):
    """
    Model user behavior as sequence of events

    Architecture:
    - LSTM: Sequence of user events
    - Attention: Weight recent events more
    - Output: User behavior embedding

    Training:
    - Predict next event from history
    - Self-supervised on user logs

    Inference:
    - Encode user's recent events
    - Compare to learned baseline
    - Large deviation = anomaly
    """

    def __init__(
        self,
        event_dim: int = 64,
        hidden_dim: int = 128,
        num_event_types: int = 20
    ):
        super().__init__()

        # Event type embedding
        self.event_type_embedding = nn.Embedding(num_event_types, event_dim)

        # LSTM for sequential modeling
        self.lstm = nn.LSTM(
            input_size=event_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # Attention
        self.attention = nn.Linear(hidden_dim, 1)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, event_dim)

    def forward(
        self,
        event_sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode user behavior from event sequence

        Args:
            event_sequences: Event type IDs (batch, seq_len)

        Returns:
            User behavior embeddings (batch, event_dim)
        """
        # Embed events
        event_embs = self.event_type_embedding(event_sequences)  # (batch, seq_len, event_dim)

        # LSTM encoding
        lstm_out, _ = self.lstm(event_embs)  # (batch, seq_len, hidden_dim)

        # Attention mechanism
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        behavior_emb = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_dim)

        # Project
        behavior_emb = self.output_projection(behavior_emb)  # (batch, event_dim)

        # Normalize
        behavior_emb = F.normalize(behavior_emb, p=2, dim=1)

        return behavior_emb

class ThreatHuntingSystem:
    """
    Cybersecurity threat hunting system

    Components:
    1. User behavior baselines: Normal behavior for each user
    2. Device baselines: Normal activity for each device
    3. Anomaly detector: Deviation from baselines
    4. Alert prioritization: Rank anomalies by severity

    Features:
    - Real-time monitoring
    - User Entity Behavior Analytics (UEBA)
    - Threat intelligence integration
    - Investigation workflow
    """

    def __init__(
        self,
        event_dim: int = 64,
        anomaly_threshold: float = 0.90
    ):
        """
        Args:
            event_dim: Event embedding dimension
            anomaly_threshold: Percentile for anomaly cutoff
        """
        self.event_dim = event_dim
        self.anomaly_threshold = anomaly_threshold

        # User behavior model
        self.behavior_model = UserBehaviorModel(event_dim=event_dim)
        self.behavior_model.eval()

        # User baselines: user_id -> baseline embedding
        self.user_baselines: Dict[str, np.ndarray] = {}

        # Device baselines: device_id -> baseline embedding
        self.device_baselines: Dict[str, np.ndarray] = {}

        # Event history: user_id -> deque of recent events
        self.user_event_history: Dict[str, deque] = {}

        # Threshold for anomaly scores
        self.score_threshold: Optional[float] = None

        # Event type mapping
        self.event_type_to_idx = {
            'login_success': 0,
            'login_failure': 1,
            'file_read': 2,
            'file_write': 3,
            'file_delete': 4,
            'network_connection': 5,
            'process_start': 6,
            'process_terminate': 7
        }

        print("Initialized Threat Hunting System")
        print(f"  Event dimension: {event_dim}")
        print(f"  Anomaly threshold: {anomaly_threshold}")

    def build_user_baseline(
        self,
        user_id: str,
        events: List[SecurityEvent]
    ):
        """
        Build behavioral baseline for user

        Args:
            user_id: User ID
            events: Historical events for user (normal behavior)
        """
        # Extract event types
        event_types = [
            self.event_type_to_idx.get(event.event_type, 0)
            for event in events
        ]

        # Encode event sequence
        event_seq = torch.tensor([event_types], dtype=torch.long)

        with torch.no_grad():
            baseline_emb = self.behavior_model(event_seq)
            self.user_baselines[user_id] = baseline_emb.numpy()[0]

    def detect_threat(
        self,
        user_id: str,
        recent_events: List[SecurityEvent]
    ) -> Tuple[bool, float]:
        """
        Detect threats based on deviation from user baseline

        Args:
            user_id: User ID
            recent_events: Recent events for user

        Returns:
            (is_threat, anomaly_score)
        """
        if user_id not in self.user_baselines:
            # No baseline: Cannot detect anomaly
            return False, 0.0

        # Get baseline
        baseline_emb = self.user_baselines[user_id]

        # Encode recent events
        event_types = [
            self.event_type_to_idx.get(event.event_type, 0)
            for event in recent_events
        ]
        event_seq = torch.tensor([event_types], dtype=torch.long)

        with torch.no_grad():
            current_emb = self.behavior_model(event_seq).numpy()[0]

        # Compute distance from baseline
        distance = np.linalg.norm(current_emb - baseline_emb)

        # Flag if distance above threshold
        # In production: Learn threshold from training data
        is_threat = distance > 0.5  # Placeholder threshold

        return is_threat, float(distance)

# Example: Insider threat detection
def threat_hunting_example():
    """
    Insider threat detection

    Scenario:
    - Employee with access to sensitive files
    - Normal behavior: Log in 9-5, access project files
    - Threat: Log in at 2am, access HR/finance files, large download

    Detection: Deviation from learned baseline
    """

    # Initialize system
    system = ThreatHuntingSystem(event_dim=64)

    # Build baseline for normal user
    normal_events = [
        SecurityEvent(
            event_id=f'event_{i}',
            event_type='login_success',
            user_id='employee_123',
            device_id='laptop_456',
            timestamp=time.time() - (100 - i) * 3600,
            features={'location': 'office', 'time_of_day': 'business_hours'}
        )
        for i in range(0, 50, 5)
    ]

    # Add file access events
    for i in range(10):
        normal_events.append(
            SecurityEvent(
                event_id=f'file_event_{i}',
                event_type='file_read',
                user_id='employee_123',
                device_id='laptop_456',
                timestamp=time.time() - (100 - i * 5) * 3600,
                features={'file_path': '/projects/project_a/data.xlsx'}
            )
        )

    print("=== Building User Baseline ===")
    system.build_user_baseline('employee_123', normal_events)
    print(f"✓ Built baseline for employee_123 from {len(normal_events)} events")

    # Test: Normal behavior
    print("\n=== Testing Normal Behavior ===")
    test_normal = [
        SecurityEvent(
            event_id='test_1',
            event_type='login_success',
            user_id='employee_123',
            device_id='laptop_456',
            timestamp=time.time(),
            features={'location': 'office', 'time_of_day': 'business_hours'}
        ),
        SecurityEvent(
            event_id='test_2',
            event_type='file_read',
            user_id='employee_123',
            device_id='laptop_456',
            timestamp=time.time() + 100,
            features={'file_path': '/projects/project_a/report.pdf'}
        )
    ]

    is_threat, score = system.detect_threat('employee_123', test_normal)
    print(f"Recent events: {len(test_normal)}")
    print(f"Anomaly score: {score:.4f}")
    print(f"Threat detected: {is_threat}")

    # Test: Anomalous behavior (insider threat)
    print("\n=== Testing Anomalous Behavior ===")
    test_anomaly = [
        SecurityEvent(
            event_id='test_3',
            event_type='login_success',
            user_id='employee_123',
            device_id='laptop_456',
            timestamp=time.time(),
            features={'location': 'home', 'time_of_day': '2am'}  # Unusual time/location
        ),
        SecurityEvent(
            event_id='test_4',
            event_type='file_read',
            user_id='employee_123',
            device_id='laptop_456',
            timestamp=time.time() + 100,
            features={'file_path': '/hr/salaries.xlsx'}  # Sensitive file
        ),
        SecurityEvent(
            event_id='test_5',
            event_type='network_connection',
            user_id='employee_123',
            device_id='laptop_456',
            timestamp=time.time() + 200,
            features={'destination': 'personal_cloud_storage', 'bytes': 500000000}  # Large upload
        )
    ]

    is_threat, score = system.detect_threat('employee_123', test_anomaly)
    print(f"Recent events: {len(test_anomaly)}")
    print(f"Anomaly score: {score:.4f}")
    print(f"Threat detected: {is_threat}")

# Uncomment to run:
# threat_hunting_example()
