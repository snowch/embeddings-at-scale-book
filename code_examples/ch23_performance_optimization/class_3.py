# Code from Chapter 23
# Book: Embeddings at Scale

"""
Compression Techniques for Vector Storage

Techniques:
1. Product Quantization (PQ): Split vectors into subvectors, quantize each
2. Scalar Quantization (SQ): Map floats to int8/int16
3. Binary Quantization: Map to binary (1-bit per dimension)
4. Dimensionality Reduction: PCA, random projection
5. Sparse Embeddings: Exploit natural sparsity
6. Learned Compression: Neural network autoencoders

Trade-offs:
- Compression ratio vs accuracy loss
- Encoding time vs decoding time
- Search compatibility (can search in compressed space?)
- Memory vs computation (decompression overhead)

Typical results:
- PQ: 16-32× compression, 3-8% accuracy loss
- SQ int8: 4× compression, <2% accuracy loss
- Binary: 32× compression, 10-20% accuracy loss
- PCA: 2-4× compression, 1-5% accuracy loss
- Sparse: 5-20× compression (data-dependent)
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass
class CompressionConfig:
    """
    Configuration for vector compression
    
    Attributes:
        method: Compression method (pq, sq, binary, pca, sparse, learned)
        compression_ratio: Target compression ratio (2-32×)
        accuracy_loss_tolerance: Maximum acceptable accuracy loss (0.0-0.2)
        encode_time_ms: Time budget for encoding (1-100ms)
        decode_time_ms: Time budget for decoding (0.1-10ms)
        searchable: Whether search can operate on compressed vectors
        training_required: Whether method requires training phase
    """
    method: str
    compression_ratio: float = 8.0
    accuracy_loss_tolerance: float = 0.05
    encode_time_ms: float = 10.0
    decode_time_ms: float = 1.0
    searchable: bool = False
    training_required: bool = True

@dataclass
class CompressionMetrics:
    """
    Metrics for compressed vector storage
    
    Attributes:
        original_size_bytes: Size before compression
        compressed_size_bytes: Size after compression
        compression_ratio: Actual compression achieved
        accuracy_loss: Measured accuracy loss (recall degradation)
        encode_time_ms: Actual encoding time
        decode_time_ms: Actual decoding time
        memory_savings_pct: Percentage memory saved
    """
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    accuracy_loss: float
    encode_time_ms: float
    decode_time_ms: float

    @property
    def memory_savings_pct(self) -> float:
        """Calculate percentage memory saved"""
        return (1.0 - 1.0 / self.compression_ratio) * 100

class ProductQuantizer:
    """
    Product Quantization for vector compression
    
    Splits D-dimensional vector into M subvectors of D/M dimensions,
    quantizes each subvector independently using k-means codebook
    
    Typical parameters:
    - M = 8-16 subvectors
    - k = 256 centroids per subvector (8-bit codes)
    - Compression: 4 bytes (float32) → 1 byte (uint8) per subvector
    - Ratio: 32× for 8 subvectors × 8 bits
    """

    def __init__(
        self,
        n_subvectors: int = 8,
        n_bits: int = 8,
        n_iterations: int = 20
    ):
        """
        Args:
            n_subvectors: Number of subvectors (M)
            n_bits: Bits per subvector code (typically 8)
            n_iterations: K-means iterations for training
        """
        self.n_subvectors = n_subvectors
        self.n_bits = n_bits
        self.n_centroids = 2 ** n_bits  # 256 for 8 bits
        self.n_iterations = n_iterations

        self.codebooks: List[np.ndarray] = []  # One per subvector
        self.subvector_dim: Optional[int] = None
        self.trained = False

    def train(
        self,
        vectors: np.ndarray,
        sample_size: Optional[int] = None
    ) -> None:
        """
        Train PQ codebooks using k-means on subvectors
        
        Args:
            vectors: Training vectors (n_samples, dim)
            sample_size: Use subset of vectors for training (faster)
        """
        n_samples, dim = vectors.shape

        # Check divisibility
        if dim % self.n_subvectors != 0:
            raise ValueError(
                f"Dimension {dim} not divisible by {self.n_subvectors}"
            )

        self.subvector_dim = dim // self.n_subvectors

        # Sample for training if needed
        if sample_size and sample_size < n_samples:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            train_vectors = vectors[indices]
        else:
            train_vectors = vectors

        # Train codebook for each subvector
        self.codebooks = []
        for i in range(self.n_subvectors):
            # Extract subvectors
            start = i * self.subvector_dim
            end = start + self.subvector_dim
            subvectors = train_vectors[:, start:end]

            # K-means clustering
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_centroids,
                max_iter=self.n_iterations,
                batch_size=min(10000, len(subvectors)),
                random_state=42
            )
            kmeans.fit(subvectors)

            # Store centroids (codebook)
            self.codebooks.append(kmeans.cluster_centers_)

        self.trained = True

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode vectors using PQ codes
        
        Args:
            vectors: Vectors to encode (n_samples, dim)
        
        Returns:
            codes: PQ codes (n_samples, n_subvectors) as uint8
        """
        if not self.trained:
            raise ValueError("Must train PQ before encoding")

        n_samples = len(vectors)
        codes = np.zeros((n_samples, self.n_subvectors), dtype=np.uint8)

        # Encode each subvector
        for i in range(self.n_subvectors):
            start = i * self.subvector_dim
            end = start + self.subvector_dim
            subvectors = vectors[:, start:end]

            # Find nearest centroid for each subvector
            codebook = self.codebooks[i]

            # Compute distances to all centroids
            # Shape: (n_samples, n_centroids)
            distances = np.linalg.norm(
                subvectors[:, np.newaxis, :] - codebook[np.newaxis, :, :],
                axis=2
            )

            # Find nearest centroid
            codes[:, i] = np.argmin(distances, axis=1)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode PQ codes back to approximate vectors
        
        Args:
            codes: PQ codes (n_samples, n_subvectors) as uint8
        
        Returns:
            vectors: Reconstructed vectors (n_samples, dim)
        """
        n_samples = len(codes)
        dim = self.n_subvectors * self.subvector_dim
        vectors = np.zeros((n_samples, dim), dtype=np.float32)

        # Decode each subvector
        for i in range(self.n_subvectors):
            start = i * self.subvector_dim
            end = start + self.subvector_dim

            # Look up centroids for codes
            subvector_codes = codes[:, i]
            vectors[:, start:end] = self.codebooks[i][subvector_codes]

        return vectors

    def compute_distance(
        self,
        query: np.ndarray,
        codes: np.ndarray
    ) -> np.ndarray:
        """
        Compute distances between query and PQ-encoded vectors
        
        Uses asymmetric distance computation (ADC):
        - Query is exact (not quantized)
        - Database vectors are quantized
        
        Much faster than decoding + distance computation
        
        Args:
            query: Query vector (dim,)
            codes: PQ codes (n_samples, n_subvectors)
        
        Returns:
            distances: Approximate L2 distances (n_samples,)
        """
        n_samples = len(codes)
        distances = np.zeros(n_samples, dtype=np.float32)

        # Precompute distance tables
        # For each subvector, compute distance from query to all centroids
        for i in range(self.n_subvectors):
            start = i * self.subvector_dim
            end = start + self.subvector_dim
            query_subvector = query[start:end]

            # Distance to each centroid
            # Shape: (n_centroids,)
            centroid_distances = np.linalg.norm(
                self.codebooks[i] - query_subvector,
                axis=1
            )

            # Look up distances for each code
            # Shape: (n_samples,)
            subvector_codes = codes[:, i]
            distances += centroid_distances[subvector_codes] ** 2

        return np.sqrt(distances)

    def get_metrics(
        self,
        original_vectors: np.ndarray
    ) -> CompressionMetrics:
        """
        Compute compression metrics
        
        Args:
            original_vectors: Original uncompressed vectors
        
        Returns:
            metrics: Compression performance metrics
        """
        # Encode
        codes = self.encode(original_vectors)

        # Decode
        reconstructed = self.decode(codes)

        # Compute accuracy loss (relative error)
        original_norms = np.linalg.norm(original_vectors, axis=1)
        reconstruction_error = np.linalg.norm(
            original_vectors - reconstructed,
            axis=1
        )
        relative_error = np.mean(reconstruction_error / original_norms)

        # Size comparison
        original_size = original_vectors.nbytes
        compressed_size = codes.nbytes + sum(
            cb.nbytes for cb in self.codebooks
        )

        return CompressionMetrics(
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            compression_ratio=original_size / compressed_size,
            accuracy_loss=relative_error,
            encode_time_ms=0.0,  # Would measure in production
            decode_time_ms=0.0
        )

class ScalarQuantizer:
    """
    Scalar Quantization: Map float32 → int8/int16
    
    For each dimension independently:
    - Find min/max values
    - Map linearly to integer range
    
    Simpler than PQ, less compression but faster and more accurate
    """

    def __init__(self, n_bits: int = 8):
        """
        Args:
            n_bits: Bits per dimension (8 or 16 typical)
        """
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits

        if n_bits == 8:
            self.dtype = np.uint8
        elif n_bits == 16:
            self.dtype = np.uint16
        else:
            raise ValueError("Only 8 or 16 bits supported")

        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None
        self.trained = False

    def train(self, vectors: np.ndarray) -> None:
        """
        Learn quantization parameters (min/max per dimension)
        
        Args:
            vectors: Training vectors (n_samples, dim)
        """
        # Compute min/max per dimension
        self.min_vals = vectors.min(axis=0)
        self.max_vals = vectors.max(axis=0)

        self.trained = True

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode vectors to quantized form
        
        Args:
            vectors: Vectors to encode (n_samples, dim)
        
        Returns:
            quantized: Quantized vectors (n_samples, dim) as uint8/uint16
        """
        if not self.trained:
            raise ValueError("Must train quantizer before encoding")

        # Normalize to [0, 1]
        normalized = (vectors - self.min_vals) / (
            self.max_vals - self.min_vals + 1e-8
        )

        # Clip to [0, 1]
        normalized = np.clip(normalized, 0.0, 1.0)

        # Map to integer range
        quantized = (normalized * (self.n_levels - 1)).astype(self.dtype)

        return quantized

    def decode(self, quantized: np.ndarray) -> np.ndarray:
        """
        Decode quantized vectors back to float32
        
        Args:
            quantized: Quantized vectors (n_samples, dim)
        
        Returns:
            vectors: Reconstructed float32 vectors
        """
        # Map integers back to [0, 1]
        normalized = quantized.astype(np.float32) / (self.n_levels - 1)

        # Denormalize
        vectors = (
            normalized * (self.max_vals - self.min_vals) + self.min_vals
        )

        return vectors

    def get_metrics(
        self,
        original_vectors: np.ndarray
    ) -> CompressionMetrics:
        """Compute compression metrics"""
        quantized = self.encode(original_vectors)
        reconstructed = self.decode(quantized)

        # Compute accuracy loss
        mse = np.mean((original_vectors - reconstructed) ** 2)
        original_variance = np.var(original_vectors)
        accuracy_loss = mse / original_variance

        # Size comparison
        original_size = original_vectors.nbytes
        compressed_size = (
            quantized.nbytes +
            self.min_vals.nbytes +
            self.max_vals.nbytes
        )

        return CompressionMetrics(
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            compression_ratio=original_size / compressed_size,
            accuracy_loss=accuracy_loss,
            encode_time_ms=0.0,
            decode_time_ms=0.0
        )

class BinaryQuantizer:
    """
    Binary Quantization: Map to {-1, +1} or {0, 1}
    
    Extreme compression (32× for float32)
    Works well for certain embeddings (e.g., locality-sensitive hashing)
    """

    def __init__(self, threshold: float = 0.0):
        """
        Args:
            threshold: Value above which maps to 1, below to 0
        """
        self.threshold = threshold

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode vectors to binary
        
        Args:
            vectors: Vectors to encode (n_samples, dim)
        
        Returns:
            binary: Binary vectors (n_samples, dim) as bool
        """
        return (vectors > self.threshold).astype(bool)

    def decode(self, binary: np.ndarray) -> np.ndarray:
        """
        Decode binary to float (maps to {-1, +1})
        
        Args:
            binary: Binary vectors (n_samples, dim) as bool
        
        Returns:
            vectors: Reconstructed vectors as float32
        """
        return binary.astype(np.float32) * 2 - 1  # Map {0, 1} to {-1, +1}

    def hamming_distance(
        self,
        query: np.ndarray,
        binary_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Compute Hamming distance between query and binary vectors
        
        Much faster than Euclidean distance
        
        Args:
            query: Binary query vector (dim,) as bool
            binary_vectors: Binary vectors (n_samples, dim) as bool
        
        Returns:
            distances: Hamming distances (n_samples,)
        """
        # XOR gives 1 where bits differ
        xor = np.logical_xor(query, binary_vectors)

        # Count 1s
        distances = np.sum(xor, axis=1)

        return distances

    def get_metrics(
        self,
        original_vectors: np.ndarray
    ) -> CompressionMetrics:
        """Compute compression metrics"""
        binary = self.encode(original_vectors)
        reconstructed = self.decode(binary)

        # Compute accuracy loss
        mse = np.mean((original_vectors - reconstructed) ** 2)
        original_variance = np.var(original_vectors)
        accuracy_loss = mse / original_variance

        # Size comparison (pack bits)
        original_size = original_vectors.nbytes
        # Each bool is 1 byte, but can pack 8 per byte
        compressed_size = binary.size // 8

        return CompressionMetrics(
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            compression_ratio=original_size / compressed_size,
            accuracy_loss=accuracy_loss,
            encode_time_ms=0.0,
            decode_time_ms=0.0
        )

class DimensionalityReducer:
    """
    Dimensionality reduction via PCA or random projection
    
    Reduces 768-dim → 256-dim (3× compression)
    Faster similarity search in lower dimensions
    """

    def __init__(
        self,
        target_dim: int = 256,
        method: str = "pca"  # "pca" or "random"
    ):
        """
        Args:
            target_dim: Target dimensionality
            method: Reduction method (pca or random projection)
        """
        self.target_dim = target_dim
        self.method = method
        self.projection_matrix: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.trained = False

    def train(
        self,
        vectors: np.ndarray,
        sample_size: Optional[int] = None
    ) -> None:
        """
        Learn projection matrix
        
        Args:
            vectors: Training vectors (n_samples, dim)
            sample_size: Use subset for training (faster)
        """
        n_samples, original_dim = vectors.shape

        # Sample if needed
        if sample_size and sample_size < n_samples:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            train_vectors = vectors[indices]
        else:
            train_vectors = vectors

        # Center data
        self.mean = train_vectors.mean(axis=0)
        centered = train_vectors - self.mean

        if self.method == "pca":
            # Compute PCA using SVD
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)

            # Take top target_dim components
            self.projection_matrix = Vt[:self.target_dim].T

        elif self.method == "random":
            # Random projection (Johnson-Lindenstrauss)
            self.projection_matrix = np.random.randn(
                original_dim,
                self.target_dim
            ) / np.sqrt(self.target_dim)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.trained = True

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Project vectors to lower dimension
        
        Args:
            vectors: Vectors to encode (n_samples, dim)
        
        Returns:
            projected: Projected vectors (n_samples, target_dim)
        """
        if not self.trained:
            raise ValueError("Must train before encoding")

        # Center and project
        centered = vectors - self.mean
        projected = centered @ self.projection_matrix

        return projected.astype(np.float32)

    def get_metrics(
        self,
        original_vectors: np.ndarray
    ) -> CompressionMetrics:
        """Compute compression metrics"""
        projected = self.encode(original_vectors)

        # Compute variance preserved (for PCA)
        if self.method == "pca":
            # Reconstruct
            reconstructed = (
                projected @ self.projection_matrix.T + self.mean
            )
            mse = np.mean((original_vectors - reconstructed) ** 2)
            original_variance = np.var(original_vectors)
            accuracy_loss = mse / original_variance
        else:
            # For random projection, estimate distance preservation
            accuracy_loss = 0.1  # Approximate

        # Size comparison
        original_size = original_vectors.nbytes
        compressed_size = (
            projected.nbytes +
            self.projection_matrix.nbytes +
            self.mean.nbytes
        )

        return CompressionMetrics(
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            compression_ratio=original_size / compressed_size,
            accuracy_loss=accuracy_loss,
            encode_time_ms=0.0,
            decode_time_ms=0.0
        )
