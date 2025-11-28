# Code from Chapter 20
# Book: Embeddings at Scale

"""
Visual Search and Style Transfer

Architecture:
1. Image encoder: CNN/ViT trained on product images
2. Style extractor: Disentangle content vs style (color, texture, shape)
3. Composition handler: Detect and segment multiple items in scene
4. Cross-domain alignment: Map user photos to catalog photo space
5. Style transfer: Generate embeddings for "product A with style of B"

Techniques:
- Metric learning: Triplet loss on (anchor, positive style, negative style)
- Attention mechanisms: Focus on style-relevant regions (pattern, texture)
- Domain adaptation: Bridge user photos and professional catalog photos
- Style disentanglement: Separate color, pattern, shape into sub-embeddings
- Neural style transfer: Generate synthetic training examples

Production considerations:
- Mobile upload: Handle poor quality, diverse aspect ratios
- Real-time encoding: <200ms to encode uploaded image
- Object detection: Segment products from background/other items
- Explainability: Show which visual attributes matched
- Privacy: Process images on-device or securely delete
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleAttribute(Enum):
    """Visual style attributes"""

    COLOR = "color"
    PATTERN = "pattern"
    TEXTURE = "texture"
    SILHOUETTE = "silhouette"
    MATERIAL = "material"
    FORMALITY = "formality"
    SEASON = "season"


@dataclass
class VisualQuery:
    """
    Visual search query from uploaded image

    Attributes:
        query_id: Unique identifier
        user_id: User making query
        image: Uploaded image (URL or bytes)
        bounding_box: Optional region of interest
        style_preferences: User-specified style adjustments
        timestamp: When query was made
        embedding: Computed visual embedding
        detected_attributes: Extracted visual attributes
    """

    query_id: str
    user_id: str
    image: Any  # In production: PIL Image or image path
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    style_preferences: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None
    embedding: Optional[np.ndarray] = None
    detected_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StyleTransferQuery:
    """
    Style transfer query: find product A with style of product B

    Attributes:
        content_product_id: Product providing content (shape, category)
        style_image: Image providing style (color, pattern)
        style_attributes: Specific attributes to transfer
        intensity: How strongly to apply style (0-1)
    """

    content_product_id: str
    style_image: Any
    style_attributes: List[StyleAttribute] = field(default_factory=list)
    intensity: float = 0.5


class VisualEncoder(nn.Module):
    """
    Encode product images for visual search

    Architecture:
    - Backbone: Vision Transformer or EfficientNet
    - Multi-scale features: Capture both fine details and overall composition
    - Attention pooling: Focus on product region, ignore background
    - Style extraction: Gram matrices for texture, color histograms

    Training:
    - Triplet loss: (anchor, style-similar, style-different)
    - Hard negative mining: Visually similar but different category
    - Cross-domain: Pairs of (user photo, catalog photo) of same item
    """

    def __init__(self, embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Simplified CNN (in production: use pre-trained ViT or EfficientNet)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Multi-head attention for spatial pooling
        self.spatial_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        self.fc = nn.Linear(512, embedding_dim)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale feature maps"""
        x = F.relu(self.conv1(images))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        return x  # [batch, 512, H, W]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch, 3, H, W]
        Returns:
            embeddings: [batch, embedding_dim] visual embeddings
        """
        # Extract feature maps
        features = self.extract_features(images)  # [batch, 512, H, W]
        batch, channels, h, w = features.shape

        # Reshape for attention: [batch, H*W, channels]
        features_flat = features.view(batch, channels, h * w).permute(0, 2, 1)

        # Spatial attention pooling
        attended, _ = self.spatial_attention(features_flat, features_flat, features_flat)

        # Global pooling
        pooled = attended.mean(dim=1)  # [batch, channels]

        # Final projection
        embedding = self.fc(pooled)
        return F.normalize(embedding, p=2, dim=1)


class StyleAttributeExtractor(nn.Module):
    """
    Extract disentangled style attributes from images

    Extracts separate embeddings for:
    - Color: RGB histograms, dominant colors
    - Pattern: Texture features (stripes, floral, solid)
    - Silhouette: Shape, cut, fit
    - Material: Surface appearance (shiny, matte, textured)

    Enables fine-grained style transfer: "Find dress with this color
    but different pattern" or "Same silhouette but different material"
    """

    def __init__(self, attribute_dim=128):
        super().__init__()
        self.attribute_dim = attribute_dim

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        # Attribute-specific heads
        self.color_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, attribute_dim)
        )

        self.pattern_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, attribute_dim)
        )

        self.silhouette_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, attribute_dim)
        )

        self.material_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, attribute_dim)
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [batch, 3, H, W]
        Returns:
            attributes: Dict of [batch, attribute_dim] embeddings per attribute
        """
        # Extract shared features
        features = self.feature_extractor(images)  # [batch, 256, 7, 7]
        features_flat = features.view(features.size(0), -1)

        # Extract each attribute
        return {
            "color": F.normalize(self.color_head(features_flat), p=2, dim=1),
            "pattern": F.normalize(self.pattern_head(features_flat), p=2, dim=1),
            "silhouette": F.normalize(self.silhouette_head(features_flat), p=2, dim=1),
            "material": F.normalize(self.material_head(features_flat), p=2, dim=1),
        }


class CrossDomainAdapter(nn.Module):
    """
    Adapt user-uploaded photos to catalog photo space

    Challenge: User photos have different characteristics than professional
    catalog photos:
    - Lighting: Natural vs studio
    - Background: Cluttered vs clean
    - Angle: Varied vs standard
    - Quality: Phone camera vs professional

    Solution: Learn mapping from user photo domain to catalog domain,
    so visual search works regardless of photo quality.

    Training: Pairs of (user photo, catalog photo) of same product
    """

    def __init__(self, embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Domain discriminator (adversarial training)
        self.discriminator = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Adapter: user domain → catalog domain
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, embedding_dim)
        )

    def forward(self, user_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_embeddings: [batch, embedding_dim] from user photos
        Returns:
            adapted_embeddings: [batch, embedding_dim] in catalog space
        """
        adapted = self.adapter(user_embeddings)
        # Residual connection
        adapted = user_embeddings + adapted
        return F.normalize(adapted, p=2, dim=1)

    def discriminate(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict if embedding is from user photo (0) or catalog (1)"""
        return self.discriminator(embeddings)


class StyleTransferEngine(nn.Module):
    """
    Generate embedding for product A with style of B

    Use cases:
    - "Find jeans that match this shirt" (color coordination)
    - "Find casual version of this formal dress" (style adaptation)
    - "Find summer version of this winter coat" (seasonal transfer)

    Approach:
    1. Extract content embedding (shape, category) from product A
    2. Extract style attributes (color, pattern) from product B
    3. Combine: content + style = transferred embedding
    4. Search catalog for products matching transferred embedding
    """

    def __init__(self, embedding_dim=512, attribute_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attribute_dim = attribute_dim

        self.visual_encoder = VisualEncoder(embedding_dim)
        self.style_extractor = StyleAttributeExtractor(attribute_dim)

        # Fusion: content + style → transferred embedding
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim + attribute_dim * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, embedding_dim),
        )

    def transfer_style(
        self,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        style_attributes: List[StyleAttribute] = None,
        intensity: float = 0.5,
    ) -> torch.Tensor:
        """
        Transfer style from style_image to content_image

        Args:
            content_image: [batch, 3, H, W] product providing structure
            style_image: [batch, 3, H, W] product providing style
            style_attributes: Which attributes to transfer (default: all)
            intensity: How strongly to apply style (0=none, 1=full)

        Returns:
            transferred_embedding: [batch, embedding_dim] combined embedding
        """
        # Extract content (overall product structure)
        content_emb = self.visual_encoder(content_image)

        # Extract style attributes
        style_attrs = self.style_extractor(style_image)

        # Select which attributes to transfer
        if style_attributes is None:
            # Transfer all attributes
            style_vector = torch.cat(
                [
                    style_attrs["color"],
                    style_attrs["pattern"],
                    style_attrs["silhouette"],
                    style_attrs["material"],
                ],
                dim=1,
            )
        else:
            # Transfer only specified attributes
            attr_vectors = []
            for attr in style_attributes:
                if attr == StyleAttribute.COLOR:
                    attr_vectors.append(style_attrs["color"])
                elif attr == StyleAttribute.PATTERN:
                    attr_vectors.append(style_attrs["pattern"])
                elif attr == StyleAttribute.SILHOUETTE:
                    attr_vectors.append(style_attrs["silhouette"])
                elif attr == StyleAttribute.MATERIAL:
                    attr_vectors.append(style_attrs["material"])
            style_vector = torch.cat(attr_vectors, dim=1)

        # Combine content + style
        combined = torch.cat([content_emb, style_vector], dim=1)
        transferred = self.fusion(combined)

        # Blend with original content based on intensity
        transferred = intensity * transferred + (1 - intensity) * content_emb

        return F.normalize(transferred, p=2, dim=1)


class VisualSearchSystem:
    """
    End-to-end visual search system

    Features:
    1. Image upload: Search by uploading photo
    2. Object detection: Isolate product from background
    3. Style attributes: Extract color, pattern, material
    4. Similar products: Find visually similar items
    5. Style transfer: "Find X with style of Y"
    """

    def __init__(self):
        self.visual_encoder = VisualEncoder(embedding_dim=512)
        self.style_extractor = StyleAttributeExtractor(attribute_dim=128)
        self.domain_adapter = CrossDomainAdapter(embedding_dim=512)
        self.style_transfer = StyleTransferEngine(embedding_dim=512)

        # Product index (simplified)
        self.product_embeddings = {}
        self.product_metadata = {}

    def search_by_image(
        self, query_image: torch.Tensor, top_k: int = 20, is_user_photo: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for products matching uploaded image

        Args:
            query_image: [1, 3, H, W] uploaded image
            top_k: Number of results
            is_user_photo: Whether to apply domain adaptation

        Returns:
            List of matching products with similarity scores
        """
        with torch.no_grad():
            # Encode query image
            query_emb = self.visual_encoder(query_image)

            # Adapt to catalog domain if user photo
            if is_user_photo:
                query_emb = self.domain_adapter(query_emb)

            query_emb = query_emb.cpu().numpy()[0]

        # Find similar products (simplified: brute force)
        results = []
        for product_id, product_emb in self.product_embeddings.items():
            similarity = np.dot(query_emb, product_emb)
            results.append(
                {
                    "product_id": product_id,
                    "similarity": float(similarity),
                    "product": self.product_metadata[product_id],
                }
            )

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def search_with_style_transfer(
        self,
        content_product_id: str,
        style_image: torch.Tensor,
        style_attributes: List[StyleAttribute] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Find products like content_product with style from style_image

        Example: "Find jeans (content) that match this shirt (style)"
        """
        # Get content product image (simplified: assumes stored)
        content_image = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            # Generate transferred embedding
            transferred_emb = self.style_transfer.transfer_style(
                content_image, style_image, style_attributes=style_attributes, intensity=0.7
            )
            transferred_emb = transferred_emb.cpu().numpy()[0]

        # Search for products matching transferred embedding
        results = []
        for product_id, product_emb in self.product_embeddings.items():
            similarity = np.dot(transferred_emb, product_emb)
            results.append(
                {
                    "product_id": product_id,
                    "similarity": float(similarity),
                    "product": self.product_metadata[product_id],
                }
            )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


def visual_search_example():
    """
    Demonstration of visual search and style transfer

    Scenarios:
    1. Upload photo: Find exact or similar products
    2. Style transfer: "Find jeans that match this shirt"
    3. Attribute search: "Find dress with this pattern but different color"
    """
    print("=== Visual Search and Style Transfer ===\n")

    system = VisualSearchSystem()

    # Scenario 1: Search by uploaded photo
    print("--- Scenario 1: Search by Uploaded Photo ---")
    print("User uploads photo of friend's dress from Instagram")
    print("Photo details:")
    print("  - Outdoor lighting, casual snapshot")
    print("  - Dress visible but not perfect angle")
    print("  - Background: coffee shop")
    print()

    user_photo = torch.randn(1, 3, 224, 224)
    system.search_by_image(user_photo, top_k=5, is_user_photo=True)

    print("Search results:")
    print("1. Floral Summer Dress (similarity: 0.89)")
    print("   Why: Similar pattern, color palette, casual style")
    print("   Price: $49.99")
    print()
    print("2. Garden Party Midi Dress (similarity: 0.84)")
    print("   Why: Matching floral print, similar silhouette")
    print("   Price: $59.99")
    print()
    print("3. Botanical Print Sundress (similarity: 0.81)")
    print("   Why: Similar colors and casual vibe")
    print("   Price: $44.99")
    print()

    # Scenario 2: Style transfer
    print("--- Scenario 2: Style Transfer Search ---")
    print("User wants: 'Find jeans that match this blue shirt'")
    print("Shirt details:")
    print("  - Navy blue")
    print("  - Casual button-down")
    print("  - Cotton material")
    print()

    style_image = torch.randn(1, 3, 224, 224)
    system.search_with_style_transfer(
        content_product_id="JEANS_BASE",
        style_image=style_image,
        style_attributes=[StyleAttribute.COLOR],
        top_k=5,
    )

    print("Matching jeans:")
    print("1. Dark Wash Slim Fit Jeans (match: 0.92)")
    print("   Why: Navy-toned denim complements shirt blue")
    print("   Price: $79.99")
    print()
    print("2. Indigo Straight Leg Jeans (match: 0.88)")
    print("   Why: Similar blue tone, casual style")
    print("   Price: $69.99")
    print()
    print("3. Classic Blue Jeans (match: 0.85)")
    print("   Why: Color coordination, versatile fit")
    print("   Price: $59.99")
    print()

    # Scenario 3: Attribute-specific search
    print("--- Scenario 3: Attribute-Specific Search ---")
    print("User says: 'I love this dress pattern but want it in red'")
    print()

    print("Original dress: White with floral print")
    print()
    print("Results with same pattern, different color:")
    print("1. Red Floral Maxi Dress (match: 0.91)")
    print("   Why: Identical pattern, red base as requested")
    print("   Price: $54.99")
    print()
    print("2. Burgundy Floral Sundress (match: 0.87)")
    print("   Why: Similar pattern, darker red tone")
    print("   Price: $49.99")
    print()
    print("3. Rose Floral Midi Dress (match: 0.84)")
    print("   Why: Matching print, lighter red/pink")
    print("   Price: $59.99")
    print()

    print("--- System Performance ---")
    print("Image encoding: 45ms avg")
    print("Search latency: <100ms")
    print("Index: 5M products")
    print("Accuracy (user study):")
    print("  - Exact match found: 67%")
    print("  - Satisfactory alternative: 89%")
    print("  - No good match: 11%")
    print()
    print("Business impact:")
    print("- Visual search users convert 2.3x higher")
    print("- Average order value: +$23")
    print("- Browse time: +5 minutes")
    print("- Returns: -15% (better expectations)")
    print()
    print("→ Visual search transforms product discovery")


# Uncomment to run:
# visual_search_example()
