import torch.nn.functional as F

# Code from Chapter 16
# Book: Embeddings at Scale

"""
Manufacturing Quality Control with Embeddings

Architecture:
1. Product encoder: Sensor data + images → embedding
2. Normal cluster: Tight cluster of defect-free products
3. Defect detection: Distance from normal cluster
4. Defect classification: Identify defect type

Data sources:
- Sensor measurements (dimensions, weight, temperature, etc.)
- Images (surface quality, alignment, completeness)
- Process parameters (machine settings, environmental conditions)

Applications:
- Automotive: Detect paint defects, alignment issues
- Electronics: Detect solder defects, component misplacement
- Pharmaceuticals: Detect contamination, incorrect dosage
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


@dataclass
class Product:
    """
    Manufactured product with quality measurements

    Attributes:
        product_id: Unique identifier
        measurements: Sensor measurements
        image: Product image (optional)
        process_params: Manufacturing process parameters
        is_defective: Ground truth label
        defect_type: Type of defect (if defective)
    """
    product_id: str
    measurements: np.ndarray
    image: Optional[Image.Image] = None
    process_params: Optional[Dict[str, float]] = None
    is_defective: bool = False
    defect_type: Optional[str] = None

class ProductEncoder(nn.Module):
    """
    Encode product to embedding

    Architecture:
    - Measurement encoder: MLP for sensor data
    - Image encoder: CNN for product images
    - Fusion: Combine measurements + images
    """

    def __init__(
        self,
        measurement_dim: int = 100,
        image_embedding_dim: int = 256,
        output_dim: int = 128
    ):
        super().__init__()

        # Measurement encoder
        self.measurement_encoder = nn.Sequential(
            nn.Linear(measurement_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

        # Image encoder (simplified CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, image_embedding_dim)
        )

        # Fusion layer
        self.fusion = nn.Linear(128 + image_embedding_dim, output_dim)

    def forward(
        self,
        measurements: torch.Tensor,
        images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode product

        Args:
            measurements: Sensor measurements (batch, measurement_dim)
            images: Product images (batch, 3, H, W) (optional)

        Returns:
            Product embeddings (batch, output_dim)
        """
        # Encode measurements
        measurement_emb = self.measurement_encoder(measurements)

        # Encode images if available
        if images is not None:
            image_emb = self.image_encoder(images)
        else:
            # No image: Use zero embedding
            image_emb = torch.zeros(measurements.size(0), 256).to(measurements.device)

        # Fuse
        combined = torch.cat([measurement_emb, image_emb], dim=1)
        product_emb = self.fusion(combined)

        # Normalize
        product_emb = F.normalize(product_emb, p=2, dim=1)

        return product_emb

class QualityControlSystem:
    """
    Manufacturing quality control system

    Components:
    1. Product encoder: Product data → embedding
    2. Normal cluster: Centroid and radius of defect-free products
    3. Defect detector: Distance from normal cluster
    4. Defect classifier: Identify defect type (optional)

    Features:
    - Real-time quality assessment
    - Defect localization (which features are abnormal)
    - Process monitoring (drift detection)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        anomaly_threshold: float = 0.95
    ):
        """
        Args:
            embedding_dim: Embedding dimension
            anomaly_threshold: Percentile for anomaly cutoff
        """
        self.embedding_dim = embedding_dim
        self.anomaly_threshold = anomaly_threshold

        # Product encoder
        self.encoder = ProductEncoder(output_dim=embedding_dim)
        self.encoder.eval()

        # Normal cluster
        self.cluster_centroid: Optional[np.ndarray] = None
        self.cluster_radius: Optional[float] = None

        # Statistics
        self.products_inspected = 0
        self.defects_detected = 0

        print("Initialized Quality Control System")
        print(f"  Embedding dimension: {embedding_dim}")

    def build_normal_cluster(self, products: List[Product]):
        """
        Build cluster of normal (defect-free) products

        Args:
            products: Defect-free products for building baseline
        """
        print(f"Building normal cluster from {len(products)} products...")

        embeddings = []

        for product in products:
            # Encode product
            measurements = torch.from_numpy(product.measurements).unsqueeze(0).float()

            with torch.no_grad():
                emb = self.encoder(measurements)
                embeddings.append(emb.numpy()[0])

        embeddings = np.array(embeddings)

        # Compute cluster centroid
        self.cluster_centroid = np.mean(embeddings, axis=0)

        # Compute distances from centroid
        distances = np.linalg.norm(embeddings - self.cluster_centroid, axis=1)

        # Set radius at specified percentile
        self.cluster_radius = np.percentile(distances, self.anomaly_threshold * 100)

        print("✓ Built normal cluster")
        print(f"  Centroid: {self.cluster_centroid.shape}")
        print(f"  Radius (95th percentile): {self.cluster_radius:.4f}")

    def inspect_product(
        self,
        product: Product
    ) -> Tuple[bool, float]:
        """
        Inspect product for defects

        Args:
            product: Product to inspect

        Returns:
            (is_defective, distance_from_normal)
        """
        if self.cluster_centroid is None:
            raise ValueError("Normal cluster not built. Call build_normal_cluster() first.")

        # Encode product
        measurements = torch.from_numpy(product.measurements).unsqueeze(0).float()

        with torch.no_grad():
            emb = self.encoder(measurements).numpy()[0]

        # Compute distance from cluster centroid
        distance = np.linalg.norm(emb - self.cluster_centroid)

        # Flag if outside cluster radius
        is_defective = distance > self.cluster_radius

        self.products_inspected += 1
        if is_defective:
            self.defects_detected += 1

        return is_defective, float(distance)

# Example: Electronics manufacturing
def quality_control_example():
    """
    PCB (Printed Circuit Board) quality control

    Use case:
    - 1M PCBs/day
    - 0.05% defect rate (500 defects)
    - Defect types: solder defects, component misplacement, shorts

    Detection: Visual inspection + electrical measurements
    """

    # Initialize system
    system = QualityControlSystem(embedding_dim=64)

    # Generate synthetic normal products
    normal_products = []
    for i in range(500):
        # Normal measurements: Mean around 0, small variance
        measurements = np.random.randn(100).astype(np.float32) * 0.1

        product = Product(
            product_id=f'product_{i}',
            measurements=measurements,
            is_defective=False
        )
        normal_products.append(product)

    print("=== Building Normal Cluster ===")
    system.build_normal_cluster(normal_products)

    # Test: Normal product
    print("\n=== Inspecting Normal Product ===")
    test_normal = Product(
        product_id='test_normal',
        measurements=np.random.randn(100).astype(np.float32) * 0.1,
        is_defective=False
    )

    is_defective, distance = system.inspect_product(test_normal)
    print(f"Product ID: {test_normal.product_id}")
    print(f"Distance from normal: {distance:.4f}")
    print(f"Defective: {is_defective}")

    # Test: Defective product
    print("\n=== Inspecting Defective Product ===")
    # Defective: Large deviation in measurements
    defect_measurements = np.random.randn(100).astype(np.float32) * 0.1
    defect_measurements[0:10] += 2.0  # Anomaly in first 10 measurements

    test_defective = Product(
        product_id='test_defective',
        measurements=defect_measurements,
        is_defective=True,
        defect_type='solder_defect'
    )

    is_defective, distance = system.inspect_product(test_defective)
    print(f"Product ID: {test_defective.product_id}")
    print(f"Distance from normal: {distance:.4f}")
    print(f"Defective: {is_defective}")

    # Statistics
    print("\n=== System Statistics ===")
    print(f"Products inspected: {system.products_inspected}")
    print(f"Defects detected: {system.defects_detected}")
    print(f"Defect rate: {system.defects_detected / system.products_inspected:.2%}")

# Uncomment to run:
# quality_control_example()
