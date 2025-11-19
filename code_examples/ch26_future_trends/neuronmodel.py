# Code from Chapter 26
# Book: Embeddings at Scale

"""
Neuromorphic Embedding Inference with Spiking Neural Networks

Architecture:
1. Rate coding: Convert continuous embeddings to spike trains
2. Spiking inference: Forward pass through SNN using spike dynamics
3. Temporal integration: Accumulate spikes over time window
4. Readout: Decode spikes back to embedding vector
5. Similarity computation: Spike-based distance metrics

Spiking neuron models:
- Leaky Integrate-and-Fire (LIF): Simple model with exponential decay
- Adaptive Exponential (AdEx): More realistic dynamics with adaptation
- Izhikevich: Efficient model reproducing biological spike patterns
- Hodgkin-Huxley: Biologically accurate but computationally expensive

Neuromorphic hardware targets:
- Intel Loihi 2: 1M neurons, 128 cores, 15pJ per spike
- IBM TrueNorth: 1M neurons, 256M synapses, 70mW
- BrainChip Akida: Event-based vision, 1-2W
- SpiNNaker: 1M cores, real-time brain simulation

Performance targets:
- Inference latency: <1ms per embedding
- Energy per inference: <1mJ (vs 100-1000mJ for GPUs)
- Throughput: 1000+ embeddings/sec/chip
- Power consumption: <100mW continuous operation
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class NeuronModel(Enum):
    """Spiking neuron models"""
    LIF = "leaky_integrate_fire"
    ADAPTIVE_LIF = "adaptive_lif"
    IZHIKEVICH = "izhikevich"
    ADEX = "adaptive_exponential"

class CodingScheme(Enum):
    """Neural coding schemes"""
    RATE = "rate"  # Spike rate encodes value
    TEMPORAL = "temporal"  # Spike timing encodes value
    POPULATION = "population"  # Population code across neurons
    BURST = "burst"  # Burst patterns encode information

@dataclass
class NeuromorphicConfig:
    """
    Configuration for neuromorphic embedding inference
    
    Attributes:
        neuron_model: Spiking neuron model
        coding_scheme: How to encode embeddings as spikes
        time_window_ms: Time window for spike integration
        dt_ms: Simulation time step
        threshold: Spike threshold voltage
        reset_voltage: Voltage after spike
        tau_mem: Membrane time constant (ms)
        tau_syn: Synaptic time constant (ms)
        refractory_period_ms: Post-spike refractory period
        num_neurons_per_dim: Neurons encoding each embedding dimension
        hardware_target: Target neuromorphic hardware
    """
    neuron_model: NeuronModel = NeuronModel.LIF
    coding_scheme: CodingScheme = CodingScheme.RATE
    time_window_ms: float = 10.0
    dt_ms: float = 1.0
    threshold: float = 1.0
    reset_voltage: float = 0.0
    tau_mem: float = 10.0  # Membrane time constant
    tau_syn: float = 5.0   # Synaptic time constant
    refractory_period_ms: float = 2.0
    num_neurons_per_dim: int = 10
    hardware_target: str = "loihi2"  # "loihi2", "truenorth", "akida", "spinnaker"

@dataclass
class SpikeEvent:
    """Single spike event"""
    neuron_id: int
    timestamp_ms: float
    layer_id: int

@dataclass
class SpikeTrain:
    """Sequence of spikes for a neuron"""
    neuron_id: int
    spike_times: List[float]
    layer_id: int

class SpikingNeuron:
    """
    Leaky Integrate-and-Fire (LIF) neuron
    
    Dynamics:
    τ_mem * dV/dt = -(V - V_rest) + R*I(t)
    
    If V ≥ V_threshold: emit spike, V ← V_reset, refractory period
    """

    def __init__(
        self,
        neuron_id: int,
        config: NeuromorphicConfig
    ):
        self.neuron_id = neuron_id
        self.config = config
        self.voltage = config.reset_voltage
        self.refractory_until = 0.0
        self.spike_times: List[float] = []

    def step(
        self,
        input_current: float,
        time_ms: float
    ) -> Optional[SpikeEvent]:
        """
        Simulate one time step
        
        Returns spike event if neuron fires
        """
        # Check refractory period
        if time_ms < self.refractory_until:
            return None

        # Membrane dynamics (Euler integration)
        decay = np.exp(-self.config.dt_ms / self.config.tau_mem)
        self.voltage = decay * self.voltage + (1 - decay) * input_current

        # Check threshold
        if self.voltage >= self.config.threshold:
            # Emit spike
            self.voltage = self.config.reset_voltage
            self.refractory_until = time_ms + self.config.refractory_period_ms
            self.spike_times.append(time_ms)

            return SpikeEvent(
                neuron_id=self.neuron_id,
                timestamp_ms=time_ms,
                layer_id=0
            )

        return None

class SpikingNeuralNetwork:
    """
    Spiking neural network for embedding inference
    
    Architecture:
    - Input layer: Encode embedding dimensions as spike trains
    - Hidden layers: Spiking neurons with recurrent connections
    - Output layer: Decode spikes to embedding vector
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        config: NeuromorphicConfig
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.config = config

        # Create neuron layers
        self.layers = self._create_layers()

        # Create synaptic connections
        self.weights = self._initialize_weights()

    def _create_layers(self) -> List[List[SpikingNeuron]]:
        """Create spiking neurons for each layer"""
        layers = []

        # Input layer
        input_neurons = [
            SpikingNeuron(i, self.config)
            for i in range(self.input_dim * self.config.num_neurons_per_dim)
        ]
        layers.append(input_neurons)

        # Hidden layers
        neuron_id = len(input_neurons)
        for hidden_dim in self.hidden_dims:
            hidden_neurons = [
                SpikingNeuron(neuron_id + i, self.config)
                for i in range(hidden_dim)
            ]
            layers.append(hidden_neurons)
            neuron_id += hidden_dim

        # Output layer
        output_neurons = [
            SpikingNeuron(neuron_id + i, self.config)
            for i in range(self.output_dim * self.config.num_neurons_per_dim)
        ]
        layers.append(output_neurons)

        return layers

    def _initialize_weights(self) -> List[np.ndarray]:
        """Initialize synaptic weights between layers"""
        weights = []

        for i in range(len(self.layers) - 1):
            n_pre = len(self.layers[i])
            n_post = len(self.layers[i + 1])

            # Initialize with small random weights
            W = np.random.randn(n_pre, n_post) * 0.1
            weights.append(W)

        return weights

    def encode_input(self, embedding: np.ndarray) -> List[List[SpikeEvent]]:
        """
        Encode continuous embedding as spike trains
        
        Rate coding: spike rate proportional to value
        Higher values → more spikes in time window
        """
        spike_trains = []

        # Normalize embedding to [0, 1]
        embedding_norm = (embedding - embedding.min()) / (embedding.max() - embedding.min() + 1e-10)

        time_steps = int(self.config.time_window_ms / self.config.dt_ms)

        for dim_idx, value in enumerate(embedding_norm):
            # Rate coding: convert value to spike rate
            target_rate = value * 100  # Hz (0-100 Hz range)
            spike_prob = target_rate * (self.config.dt_ms / 1000.0)  # Probability per time step

            dim_spikes = []
            for neuron_offset in range(self.config.num_neurons_per_dim):
                neuron_id = dim_idx * self.config.num_neurons_per_dim + neuron_offset
                neuron_spikes = []

                for t_idx in range(time_steps):
                    time_ms = t_idx * self.config.dt_ms

                    # Poisson spike generation
                    if np.random.random() < spike_prob:
                        neuron_spikes.append(SpikeEvent(
                            neuron_id=neuron_id,
                            timestamp_ms=time_ms,
                            layer_id=0
                        ))

                dim_spikes.extend(neuron_spikes)

            spike_trains.append(dim_spikes)

        return spike_trains

    def forward(
        self,
        input_spikes: List[List[SpikeEvent]]
    ) -> np.ndarray:
        """
        Forward propagation through spiking network
        
        Event-driven simulation: process spikes as they occur
        """
        time_steps = int(self.config.time_window_ms / self.config.dt_ms)

        # Flatten input spikes
        all_input_spikes = []
        for dim_spikes in input_spikes:
            all_input_spikes.extend(dim_spikes)

        # Sort by timestamp
        all_input_spikes.sort(key=lambda s: s.timestamp_ms)

        # Track spike events for all layers
        layer_spikes = [[] for _ in self.layers]
        layer_spikes[0] = all_input_spikes

        # Simulate network over time
        spike_idx = 0
        for t_idx in range(time_steps):
            time_ms = t_idx * self.config.dt_ms

            # Process each layer
            for layer_idx in range(1, len(self.layers)):
                prev_layer = self.layers[layer_idx - 1]
                curr_layer = self.layers[layer_idx]
                weights = self.weights[layer_idx - 1]

                # Compute input current for each neuron in current layer
                input_currents = np.zeros(len(curr_layer))

                # Aggregate spikes from previous layer
                prev_spikes = [s for s in layer_spikes[layer_idx - 1]
                             if abs(s.timestamp_ms - time_ms) < self.config.tau_syn]

                for spike in prev_spikes:
                    # Synaptic current: exponential decay
                    time_diff = time_ms - spike.timestamp_ms
                    if time_diff >= 0:
                        synaptic_weight = np.exp(-time_diff / self.config.tau_syn)

                        # Add weighted contribution to all post-synaptic neurons
                        input_currents += weights[spike.neuron_id, :] * synaptic_weight

                # Update each neuron
                for neuron_idx, neuron in enumerate(curr_layer):
                    spike_event = neuron.step(input_currents[neuron_idx], time_ms)

                    if spike_event is not None:
                        spike_event.layer_id = layer_idx
                        layer_spikes[layer_idx].append(spike_event)

        # Decode output spikes to embedding
        output_spikes = layer_spikes[-1]
        output_embedding = self.decode_output(output_spikes)

        return output_embedding

    def decode_output(self, output_spikes: List[SpikeEvent]) -> np.ndarray:
        """
        Decode spike trains to embedding vector
        
        Rate decoding: count spikes per neuron, average across population
        """
        embedding = np.zeros(self.output_dim)

        # Count spikes for each dimension
        for dim_idx in range(self.output_dim):
            dim_spike_count = 0

            for neuron_offset in range(self.config.num_neurons_per_dim):
                neuron_id = dim_idx * self.config.num_neurons_per_dim + neuron_offset

                # Count spikes for this neuron
                neuron_spike_count = sum(
                    1 for s in output_spikes if s.neuron_id == neuron_id
                )
                dim_spike_count += neuron_spike_count

            # Average spike count across population
            avg_spike_count = dim_spike_count / self.config.num_neurons_per_dim

            # Convert to embedding value (normalize by time window)
            spike_rate = avg_spike_count / (self.config.time_window_ms / 1000.0)
            embedding[dim_idx] = spike_rate / 100.0  # Normalize to [0, 1]

        # Re-normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        return embedding

class NeuromorphicEmbeddingInference:
    """
    Complete neuromorphic embedding inference system
    
    Handles encoding, SNN forward pass, and decoding
    """

    def __init__(
        self,
        embedding_dim: int,
        config: NeuromorphicConfig
    ):
        self.embedding_dim = embedding_dim
        self.config = config

        # Create spiking neural network
        # Simple architecture: input → hidden → output
        hidden_dims = [embedding_dim // 2]
        self.snn = SpikingNeuralNetwork(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=embedding_dim,
            config=config
        )

        self.inference_count = 0
        self.total_spikes = 0
        self.total_energy_mj = 0.0

    def infer(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Perform neuromorphic embedding inference
        
        Returns:
            - Reconstructed embedding
            - Spike statistics
            - Energy consumption estimate
        """
        import time

        start = time.time()

        # Encode input as spikes
        input_spikes = self.snn.encode_input(embedding)

        # Forward through SNN
        output_embedding = self.snn.forward(input_spikes)

        inference_time = (time.time() - start) * 1000  # ms

        # Count spikes
        total_input_spikes = sum(len(dim_spikes) for dim_spikes in input_spikes)

        # Estimate energy consumption
        # Neuromorphic chips: ~15 pJ per spike (Intel Loihi 2)
        energy_per_spike_pj = 15
        estimated_energy_mj = (total_input_spikes * energy_per_spike_pj) / 1e9

        # Update statistics
        self.inference_count += 1
        self.total_spikes += total_input_spikes
        self.total_energy_mj += estimated_energy_mj

        return {
            'embedding': output_embedding,
            'num_spikes': total_input_spikes,
            'inference_time_ms': inference_time,
            'energy_mj': estimated_energy_mj,
            'avg_spike_rate': total_input_spikes / self.config.time_window_ms * 1000,
            'reconstruction_error': np.linalg.norm(embedding - output_embedding)
        }

    def get_statistics(self) -> Dict[str, float]:
        """Get cumulative inference statistics"""
        if self.inference_count == 0:
            return {}

        return {
            'total_inferences': self.inference_count,
            'avg_spikes_per_inference': self.total_spikes / self.inference_count,
            'avg_energy_mj': self.total_energy_mj / self.inference_count,
            'total_energy_mj': self.total_energy_mj,
            'energy_efficiency': self.total_spikes / max(self.total_energy_mj, 1e-10)  # spikes per mJ
        }

# Example usage demonstrating neuromorphic inference
def demonstrate_neuromorphic_inference():
    """Compare neuromorphic vs conventional embedding inference"""

    # Configure neuromorphic system
    config = NeuromorphicConfig(
        neuron_model=NeuronModel.LIF,
        coding_scheme=CodingScheme.RATE,
        time_window_ms=10.0,
        dt_ms=0.1,
        num_neurons_per_dim=10,
        hardware_target="loihi2"
    )

    # Create inference engine
    embedding_dim = 256
    neuro_inference = NeuromorphicEmbeddingInference(embedding_dim, config)

    # Generate test embedding
    embedding = np.random.randn(embedding_dim)
    embedding = embedding / np.linalg.norm(embedding)

    # Neuromorphic inference
    result = neuro_inference.infer(embedding)

    print("Neuromorphic Inference Results:")
    print(f"  Spikes generated: {result['num_spikes']}")
    print(f"  Inference time: {result['inference_time_ms']:.2f}ms")
    print(f"  Energy consumed: {result['energy_mj']:.6f}mJ")
    print(f"  Reconstruction error: {result['reconstruction_error']:.4f}")

    # Comparison with GPU inference (estimated)
    gpu_energy_mj = 100  # Typical GPU inference
    gpu_time_ms = 1.0

    print("\nComparison with GPU:")
    print(f"  Energy savings: {gpu_energy_mj / result['energy_mj']:.0f}×")
    print(f"  Latency: {result['inference_time_ms'] / gpu_time_ms:.1f}× " +
          f"{'slower' if result['inference_time_ms'] > gpu_time_ms else 'faster'}")

    # Continuous operation analysis
    embeddings_per_second = 100
    hours_on_battery = 10

    neuro_total_energy = result['energy_mj'] * embeddings_per_second * 3600 * hours_on_battery
    gpu_total_energy = gpu_energy_mj * embeddings_per_second * 3600 * hours_on_battery

    print(f"\nContinuous Operation ({hours_on_battery} hours):")
    print(f"  Neuromorphic energy: {neuro_total_energy / 1000:.2f}J")
    print(f"  GPU energy: {gpu_total_energy / 1000:.2f}J")
    print(f"  Battery life improvement: {gpu_total_energy / neuro_total_energy:.0f}×")
