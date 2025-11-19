# Code from Chapter 26
# Book: Embeddings at Scale

"""
Edge Embedding Inference System

Architecture:
1. Model compression: Quantize and distill for edge deployment
2. On-device inference: Generate embeddings locally
3. Edge caching: Store frequent embeddings
4. Adaptive offloading: Offload to cloud when necessary
5. Federated learning: Improve model collaboratively

Compression techniques:
- Quantization: 32-bit → 8-bit or 4-bit (4-8× size reduction)
- Pruning: Remove <10% magnitude weights (2-10× speedup)
- Knowledge distillation: Train small model from large (5-20× size reduction)
- Low-rank decomposition: Factor weight matrices (2-4× compression)

Performance targets (smartphone):
- Model size: <10MB
- Inference time: <10ms per embedding
- Power: <100mW
- Accuracy: >95% of full model
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple

import numpy as np


class DeviceType(Enum):
    """Edge device types"""
    SMARTPHONE = "smartphone"
    IOTDEVICE = "iot_device"
    WEARABLE = "wearable"
    GATEWAY = "gateway"
    EDGE_SERVER = "edge_server"

class QuantizationMode(Enum):
    """Quantization precision"""
    INT8 = "int8"
    INT4 = "int4"
    BINARY = "binary"
    MIXED = "mixed_precision"

@dataclass
class EdgeDeviceConfig:
    """
    Configuration for edge device constraints

    Attributes:
        device_type: Type of edge device
        max_model_size_mb: Maximum model size
        max_memory_mb: Available RAM
        power_budget_mw: Power budget for inference
        target_latency_ms: Target inference latency
        quantization: Quantization mode
        use_accelerator: Use hardware accelerator (NPU, Neural Engine)
        cache_size: Number of embeddings to cache
        offload_threshold: Latency threshold for cloud offload
    """
    device_type: DeviceType = DeviceType.SMARTPHONE
    max_model_size_mb: float = 10.0
    max_memory_mb: float = 100.0
    power_budget_mw: float = 100.0
    target_latency_ms: float = 10.0
    quantization: QuantizationMode = QuantizationMode.INT8
    use_accelerator: bool = True
    cache_size: int = 1000
    offload_threshold_ms: float = 50.0

@dataclass
class EdgeEmbeddingModel:
    """Compressed embedding model for edge deployment"""
    weights: Dict[str, np.ndarray]
    quantization_params: Dict[str, Dict]
    input_dim: int
    output_dim: int
    model_size_mb: float
    compression_ratio: float

class ModelCompressor:
    """
    Compress embedding models for edge deployment

    Combines multiple compression techniques:
    1. Quantization: Reduce precision
    2. Pruning: Remove unimportant weights
    3. Distillation: Train smaller model
    4. Low-rank factorization: Decompose weight matrices
    """

    def __init__(self, quantization_mode: QuantizationMode = QuantizationMode.INT8):
        self.quantization_mode = quantization_mode

    def compress(
        self,
        model_weights: Dict[str, np.ndarray],
        target_size_mb: float,
        prune_threshold: float = 0.01
    ) -> EdgeEmbeddingModel:
        """
        Compress model to target size

        Steps:
        1. Prune small weights
        2. Quantize remaining weights
        3. Compute quantization parameters
        4. Measure final size
        """
        compressed_weights = {}
        quantization_params = {}
        original_size = 0
        compressed_size = 0

        for name, weights in model_weights.items():
            original_size += weights.nbytes

            # Prune small weights
            pruned = self._prune_weights(weights, prune_threshold)

            # Quantize
            quantized, params = self._quantize_weights(pruned)

            compressed_weights[name] = quantized
            quantization_params[name] = params
            compressed_size += quantized.nbytes

        # Extract dimensions (assuming first layer is input)
        first_layer = list(compressed_weights.values())[0]
        input_dim = first_layer.shape[0] if len(first_layer.shape) > 1 else first_layer.shape[0]
        last_layer = list(compressed_weights.values())[-1]
        output_dim = last_layer.shape[-1]

        return EdgeEmbeddingModel(
            weights=compressed_weights,
            quantization_params=quantization_params,
            input_dim=input_dim,
            output_dim=output_dim,
            model_size_mb=compressed_size / (1024 ** 2),
            compression_ratio=original_size / compressed_size
        )

    def _prune_weights(
        self,
        weights: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """Prune weights below threshold"""
        mask = np.abs(weights) > threshold
        return weights * mask

    def _quantize_weights(
        self,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Quantize weights to lower precision

        INT8 quantization:
        q = round((x - min) / (max - min) * 255)
        x_reconstructed = q / 255 * (max - min) + min
        """
        if self.quantization_mode == QuantizationMode.INT8:
            w_min, w_max = weights.min(), weights.max()

            # Scale to [0, 255]
            scale = (w_max - w_min) / 255.0
            zero_point = w_min

            quantized = np.round((weights - zero_point) / scale).astype(np.int8)

            params = {
                'scale': scale,
                'zero_point': zero_point,
                'dtype': 'int8'
            }

            return quantized, params

        elif self.quantization_mode == QuantizationMode.INT4:
            # 4-bit quantization
            w_min, w_max = weights.min(), weights.max()
            scale = (w_max - w_min) / 15.0
            zero_point = w_min

            quantized = np.round((weights - zero_point) / scale).astype(np.int8)
            quantized = np.clip(quantized, 0, 15)

            params = {
                'scale': scale,
                'zero_point': zero_point,
                'dtype': 'int4'
            }

            return quantized, params

        else:
            # No quantization
            return weights, {'dtype': 'float32'}

    def dequantize_weights(
        self,
        quantized: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Dequantize weights for inference"""
        if params['dtype'] == 'int8' or params['dtype'] == 'int4':
            scale = params['scale']
            zero_point = params['zero_point']
            return quantized.astype(np.float32) * scale + zero_point
        else:
            return quantized

class EdgeEmbeddingInference:
    """
    On-device embedding inference with caching and offloading

    Optimizations:
    - Quantized inference on device
    - LRU cache for frequent embeddings
    - Adaptive offloading to cloud for high latency queries
    - Batch processing when possible
    """

    def __init__(
        self,
        model: EdgeEmbeddingModel,
        config: EdgeDeviceConfig
    ):
        self.model = model
        self.config = config
        self.compressor = ModelCompressor(config.quantization)

        # Cache for frequent embeddings
        from collections import OrderedDict
        self.cache: OrderedDict = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

        # Statistics
        self.local_inferences = 0
        self.cloud_offloads = 0
        self.total_latency_ms = 0
        self.total_energy_mj = 0

    def infer(
        self,
        input_data: np.ndarray,
        allow_offload: bool = True
    ) -> Dict[str, Any]:
        """
        Generate embedding on edge device

        Decision flow:
        1. Check cache
        2. If cached: return cached embedding
        3. If not cached:
           a. Estimate latency
           b. If latency acceptable: local inference
           c. If latency too high and offload allowed: cloud offload
        """
        import hashlib

        # Create cache key
        cache_key = hashlib.sha256(input_data.tobytes()).hexdigest()

        # Check cache
        if cache_key in self.cache:
            self.cache_hits += 1
            embedding = self.cache[cache_key]

            # Move to end (LRU)
            self.cache.move_to_end(cache_key)

            return {
                'embedding': embedding,
                'source': 'cache',
                'latency_ms': 0.1,
                'energy_mj': 0.0
            }

        self.cache_misses += 1

        # Estimate local inference latency
        estimated_latency_ms = self._estimate_latency(input_data)

        # Decide: local or cloud
        if estimated_latency_ms <= self.config.offload_threshold_ms or not allow_offload:
            # Local inference
            result = self._local_inference(input_data)
            result['source'] = 'local'
        else:
            # Offload to cloud
            result = self._cloud_offload(input_data)
            result['source'] = 'cloud'

        # Cache result
        self._add_to_cache(cache_key, result['embedding'])

        # Update statistics
        if result['source'] == 'local':
            self.local_inferences += 1
        else:
            self.cloud_offloads += 1

        self.total_latency_ms += result['latency_ms']
        self.total_energy_mj += result['energy_mj']

        return result

    def _estimate_latency(self, input_data: np.ndarray) -> float:
        """Estimate inference latency based on input size and model complexity"""
        # Simple model: latency proportional to model size and input size
        model_factor = self.model.model_size_mb / 10.0  # Normalize to 10MB baseline
        input_factor = input_data.size / 1000.0  # Normalize to 1K elements

        base_latency = 5.0  # ms
        estimated = base_latency * model_factor * input_factor

        # Hardware accelerator speedup
        if self.config.use_accelerator:
            estimated /= 5.0

        return estimated

    def _local_inference(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Perform inference on edge device"""
        import time

        start = time.time()

        # Quantized inference
        # In practice, would use optimized kernels (NNAPI, Core ML, TensorFlow Lite)

        # Simple 2-layer network for demonstration
        hidden = input_data

        for layer_name in ['layer1', 'layer2']:
            if layer_name in self.model.weights:
                weights_quantized = self.model.weights[layer_name]
                params = self.model.quantization_params[layer_name]

                # Dequantize
                weights = self.compressor.dequantize_weights(weights_quantized, params)

                # Matrix multiply
                if len(weights.shape) == 2:
                    hidden = hidden @ weights

                # ReLU activation
                hidden = np.maximum(0, hidden)

        # Normalize
        embedding = hidden[:self.model.output_dim]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        latency_ms = (time.time() - start) * 1000

        # Energy estimate (simplified)
        # Modern mobile NPUs: ~0.1 mJ per inference for small models
        energy_mj = 0.1 if self.config.use_accelerator else 1.0

        return {
            'embedding': embedding,
            'latency_ms': latency_ms,
            'energy_mj': energy_mj
        }

    def _cloud_offload(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Offload inference to cloud"""
        # In practice, send request to cloud API
        # Simulate cloud latency and energy cost

        # Network latency
        network_latency_ms = 50 + np.random.exponential(20)  # 50-150ms typical

        # Cloud inference (fast but network-limited)
        cloud_inference_ms = 5.0
        total_latency = network_latency_ms + cloud_inference_ms

        # Energy: network transmission dominates
        # ~10mJ per request for cellular, ~1mJ for WiFi
        energy_mj = 10.0

        # Generate embedding (placeholder)
        embedding = np.random.randn(self.model.output_dim)
        embedding = embedding / np.linalg.norm(embedding)

        return {
            'embedding': embedding,
            'latency_ms': total_latency,
            'energy_mj': energy_mj
        }

    def _add_to_cache(self, key: str, embedding: np.ndarray):
        """Add embedding to LRU cache"""
        self.cache[key] = embedding

        # Enforce cache size limit
        if len(self.cache) > self.config.cache_size:
            # Remove oldest item
            self.cache.popitem(last=False)

    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics"""
        total_inferences = self.local_inferences + self.cloud_offloads

        if total_inferences == 0:
            return {}

        return {
            'total_inferences': total_inferences,
            'local_inferences': self.local_inferences,
            'cloud_offloads': self.cloud_offloads,
            'local_percentage': self.local_inferences / total_inferences * 100,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) * 100,
            'avg_latency_ms': self.total_latency_ms / total_inferences,
            'avg_energy_mj': self.total_energy_mj / total_inferences
        }


### Federated Learning for Collaborative Edge Embeddings

# Federated learning trains models across decentralized devices without centralizing data.
# This enables privacy-preserving collaborative improvement of embedding models while data remains on edge devices.
