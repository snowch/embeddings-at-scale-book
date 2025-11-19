# Code from Chapter 17
# Book: Embeddings at Scale

"""
Predictive Maintenance with Embeddings

Architecture:
1. Equipment encoder: Sensor data, images, audio → embedding
2. Normal behavior: Baseline embeddings for healthy equipment
3. Anomaly detection: Deviation from normal → failure risk
4. Time-to-failure: Predict hours/days until failure

Data sources:
- Sensors: Temperature, vibration, pressure, current
- Images: Visual inspection photos
- Audio: Equipment sounds
- Maintenance history: Past failures, repairs

Techniques:
- LSTM: Sequential sensor data modeling
- Autoencoder: Reconstruct sensor values, high error = anomaly
- Survival analysis: Predict time until failure
- Multi-modal: Combine sensors + images + audio
"""

@dataclass
class EquipmentReading:
    """
    Equipment sensor reading
    
    Attributes:
        equipment_id: Equipment identifier
        timestamp: When reading was taken
        sensors: Sensor readings (temperature, vibration, etc.)
        image: Optional image data
        maintenance_history: Past maintenance events
        failure_time: Time until failure (if known, for training)
    """
    equipment_id: str
    timestamp: float
    sensors: Dict[str, float]
    image: Optional[np.ndarray] = None
    maintenance_history: Optional[List[Dict]] = None
    failure_time: Optional[float] = None
    
    def __post_init__(self):
        if self.sensors is None:
            self.sensors = {}
        if self.maintenance_history is None:
            self.maintenance_history = []

class EquipmentEncoder(nn.Module):
    """
    Encode equipment state from multi-modal data
    
    Architecture:
    - Sensor encoder: Time-series sensor data
    - Image encoder: Visual inspection images
    - Maintenance encoder: Past maintenance history
    - Fusion: Combine modalities
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        num_sensors: int = 10,
        sequence_length: int = 100
    ):
        super().__init__()
        
        # Sensor time-series encoder (LSTM)
        self.sensor_encoder = nn.LSTM(
            input_size=num_sensors,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )
    
    def forward(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        Encode equipment state
        
        Args:
            sensor_data: Sensor time series (batch_size, sequence_length, num_sensors)
        
        Returns:
            Equipment embeddings (batch_size, embedding_dim)
        """
        # Encode sensor time series
        _, (hidden, _) = self.sensor_encoder(sensor_data)
        sensor_emb = hidden[-1]  # Last hidden state
        
        # Fusion
        equipment_emb = self.fusion(sensor_emb)
        
        # Normalize
        equipment_emb = F.normalize(equipment_emb, p=2, dim=1)
        
        return equipment_emb

class FailurePredictionModel(nn.Module):
    """
    Predict equipment failure from embedding
    
    Outputs:
    - Failure probability: P(failure in next N hours)
    - Time to failure: Expected hours until failure
    - Failure mode: Type of failure (bearing, motor, etc.)
    """
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        
        # Failure probability head
        self.failure_prob_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Time to failure head
        self.time_to_failure_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.ReLU()  # Positive time
        )
    
    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict failure
        
        Args:
            embeddings: Equipment embeddings (batch_size, embedding_dim)
        
        Returns:
            (failure_probabilities, time_to_failure)
        """
        failure_prob = self.failure_prob_head(embeddings)
        time_to_failure = self.time_to_failure_head(embeddings)
        
        return failure_prob, time_to_failure

# Example: Predictive maintenance
def predictive_maintenance_example():
    """
    Predictive maintenance for industrial equipment
    
    Scenario:
    - Equipment: CNC machine
    - Sensors: Temperature, vibration, current, pressure
    - Goal: Predict bearing failure before it occurs
    
    Traditional:
    - Reactive: Fix after failure (24 hours downtime)
    - Preventive: Replace bearing every 1000 hours (often unnecessary)
    - Threshold: Alert if vibration > 10 mm/s (misses slow degradation)
    
    Embedding approach:
    - Learn normal operation embedding
    - Detect drift toward failure patterns
    - Predict time to failure
    - Schedule maintenance proactively
    """
    
    print("=== Predictive Maintenance ===")
    print("\nEquipment: CNC Machining Center")
    print("Component: Spindle bearing")
    print("Failure mode: Bearing wear → vibration → failure")
    print("Failure cost: $50K (downtime + repair)")
    print("Maintenance cost: $5K (planned bearing replacement)")
    
    print("\n--- Traditional Approaches ---")
    print("\n1. Reactive Maintenance:")
    print("   Wait for failure, then fix")
    print("   Downtime: 24 hours")
    print("   Total cost: $50K")
    print("   → Unacceptable: Disrupts production")
    
    print("\n2. Preventive Maintenance:")
    print("   Replace bearing every 1000 hours")
    print("   Some bearings last 1500 hours (wasted)")
    print("   Some bearings fail at 800 hours (still have failures)")
    print("   Average cost: $5K every 1000 hours")
    print("   → Suboptimal: Too early or too late")
    
    print("\n3. Threshold-Based:")
    print("   Alert if vibration > 10 mm/s")
    print("   Problem: Sudden failures still occur")
    print("   Problem: False alarms from normal variation")
    print("   → Better, but misses complex patterns")
    
    print("\n--- Embedding-Based Predictive Maintenance ---")
    print("\nApproach:")
    print("  1. Learn normal operation embedding from sensor data")
    print("  2. Continuously monitor equipment embedding")
    print("  3. Detect drift toward failure patterns")
    print("  4. Predict time to failure")
    print("  5. Schedule maintenance proactively")
    
    print("\n--- Equipment 1: Healthy ---")
    print("Hours of operation: 200")
    print("Temperature: 65°C (normal)")
    print("Vibration: 3.2 mm/s (normal)")
    print("Current: 15A (normal)")
    print("Embedding distance from normal: 0.05")
    print("Failure probability (next 100 hours): 2%")
    print("Time to failure: 800+ hours")
    print("Recommendation: Continue normal operation")
    
    print("\n--- Equipment 2: Early Degradation ---")
    print("Hours of operation: 650")
    print("Temperature: 72°C (slight increase)")
    print("Vibration: 5.1 mm/s (increasing trend)")
    print("Current: 16A (slight increase)")
    print("Embedding distance from normal: 0.28")
    print("Failure probability (next 100 hours): 15%")
    print("Time to failure: 150 hours")
    print("Recommendation: Schedule maintenance in 100 hours")
    print("  → Detected early degradation before vibration threshold")
    
    print("\n--- Equipment 3: Imminent Failure ---")
    print("Hours of operation: 820")
    print("Temperature: 85°C (high)")
    print("Vibration: 12.3 mm/s (high)")
    print("Current: 18A (high)")
    print("Embedding distance from normal: 0.67")
    print("Failure probability (next 100 hours): 85%")
    print("Time to failure: 20 hours")
    print("Recommendation: URGENT - Stop machine and replace bearing")
    print("  → Caught just before catastrophic failure")
    
    print("\n--- Results ---")
    print("Failures prevented: 95%")
    print("Unnecessary maintenance reduced: 60%")
    print("Average cost per machine: $3K/year")
    print("ROI: 10x (vs reactive) 3x (vs preventive)")
    print("\n→ Embedding-based approach optimizes maintenance timing")

# Uncomment to run:
# predictive_maintenance_example()
