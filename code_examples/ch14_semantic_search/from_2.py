# Code from Chapter 14
# Book: Embeddings at Scale

"""
Media and Content Discovery System

Architecture:
1. Visual encoders: CNN or ViT for images/video
2. Audio encoders: Mel-spectrogram + CNN for audio
3. Style embeddings: Capture color, composition, texture
4. Perceptual hashing: Find near-duplicates
5. Multi-modal fusion: Combine visual + audio for videos

Applications:
- Stock photo search (find similar images)
- Video recommendation (find similar videos)
- Music discovery (find similar songs)
- Style transfer (find images with similar style)
- Duplicate detection (find copyright violations)
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import numpy as np

@dataclass
class MediaAsset:
    """
    Media asset with metadata

    Attributes:
        asset_id: Unique identifier
        asset_type: Type ('image', 'video', 'audio')
        file_path: File path
        duration: Duration in seconds (for video/audio)
        resolution: Image/video resolution (width, height)
        metadata: Additional metadata (tags, description, etc.)
        visual_embedding: Visual style embedding
        content_embedding: Semantic content embedding
        perceptual_hash: Perceptual hash for duplicate detection
    """
    asset_id: str
    asset_type: str
    file_path: str
    duration: Optional[float] = None
    resolution: Optional[Tuple[int, int]] = None
    metadata: Dict = None
    visual_embedding: Optional[np.ndarray] = None
    content_embedding: Optional[np.ndarray] = None
    perceptual_hash: Optional[str] = None

class VisualStyleEncoder(nn.Module):
    """
    Encode visual style (color, composition, texture)

    Architecture:
    - Multi-scale CNN (extract features at multiple resolutions)
    - Global pooling (capture global statistics)
    - Style embedding (separate from content)

    Inspiration: Gram matrices from neural style transfer
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Multi-scale feature extractors
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Style embedding projection
        # Gram matrix captures correlations between features (style)
        self.style_projection = nn.Linear(256 * 256, embedding_dim)

    def gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix (feature correlations)

        Gram matrix captures style, not content

        Args:
            features: Features (batch, channels, height, width)

        Returns:
            Gram matrices (batch, channels, channels)
        """
        batch, channels, height, width = features.size()

        # Reshape: (batch, channels, height*width)
        features = features.view(batch, channels, height * width)

        # Compute Gram: G = F * F^T
        gram = torch.bmm(features, features.transpose(1, 2))

        # Normalize by spatial dimensions
        gram = gram / (height * width)

        return gram

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode visual style

        Args:
            images: Images (batch, 3, height, width)

        Returns:
            Style embeddings (batch, embedding_dim)
        """
        # Extract features
        x = F.relu(self.conv1(images))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))  # (batch, 256, H/4, W/4)

        # Compute Gram matrix (style)
        gram = self.gram_matrix(x)  # (batch, 256, 256)

        # Flatten Gram matrix
        gram_flat = gram.view(gram.size(0), -1)  # (batch, 256*256)

        # Project to style embedding
        style_embedding = self.style_projection(gram_flat)

        # Normalize
        style_embedding = F.normalize(style_embedding, p=2, dim=1)

        return style_embedding

class MediaSearchEngine:
    """
    Semantic search for media assets

    Capabilities:
    - Visual similarity: Find visually similar images/videos
    - Style matching: Find images with similar style (color, composition)
    - Audio similarity: Find similar audio tracks
    - Duplicate detection: Find near-duplicates (copyright, deduplication)
    - Multi-modal: Search videos by visual + audio

    Production optimizations:
    - Pre-compute embeddings offline
    - Store embeddings in vector database (Qdrant, Milvus)
    - Use perceptual hashing for fast duplicate detection
    - Cluster similar assets for browsing
    """

    def __init__(self, embedding_dim: int = 512, device: str = 'cuda'):
        """
        Args:
            embedding_dim: Embedding dimension
            device: Device for computation
        """
        self.embedding_dim = embedding_dim
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Initialize encoders
        self.content_encoder = ImageEncoder(embedding_dim=embedding_dim).to(self.device)
        self.style_encoder = VisualStyleEncoder(embedding_dim=embedding_dim).to(self.device)

        self.content_encoder.eval()
        self.style_encoder.eval()

        # Asset store
        self.assets: Dict[str, MediaAsset] = {}

        # Embedding indices
        self.asset_ids: List[str] = []
        self.content_embeddings: Optional[np.ndarray] = None
        self.style_embeddings: Optional[np.ndarray] = None

        print(f"Initialized Media Search Engine")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Device: {self.device}")

    def compute_perceptual_hash(self, image: Image.Image, hash_size: int = 8) -> str:
        """
        Compute perceptual hash for duplicate detection

        Perceptual hash (pHash):
        - Resize to small size (8x8 or 16x16)
        - Convert to grayscale
        - Compute DCT (discrete cosine transform)
        - Keep low-frequency components
        - Threshold to binary

        Args:
            image: PIL Image
            hash_size: Hash size (8 or 16)

        Returns:
            Hex string hash
        """
        # Resize to hash_size x hash_size
        img = image.resize((hash_size, hash_size), Image.Resampling.LANCZOS)

        # Convert to grayscale
        img = img.convert('L')

        # Get pixel values
        pixels = np.array(img).flatten()

        # Compute mean
        mean = np.mean(pixels)

        # Threshold to binary
        binary_hash = (pixels > mean).astype(int)

        # Convert to hex string
        hash_str = ''.join(str(b) for b in binary_hash)

        return hash_str

    def encode_image(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode image to content and style embeddings

        Args:
            image: PIL Image

        Returns:
            (content_embedding, style_embedding)
        """
        # Preprocess
        img = image.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0

        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Encode content
        with torch.no_grad():
            content_emb = self.content_encoder(img_tensor).cpu().numpy()[0]

        # Encode style
        with torch.no_grad():
            style_emb = self.style_encoder(img_tensor).cpu().numpy()[0]

        return content_emb, style_emb

    def index_asset(self, asset: MediaAsset, image: Optional[Image.Image] = None):
        """
        Index media asset

        Args:
            asset: Media asset
            image: PIL Image (for image/video assets)
        """
        if asset.asset_type in ['image', 'video'] and image:
            # Encode image
            content_emb, style_emb = self.encode_image(image)
            asset.content_embedding = content_emb
            asset.visual_embedding = style_emb

            # Compute perceptual hash
            asset.perceptual_hash = self.compute_perceptual_hash(image)

            # Add to indices
            self.asset_ids.append(asset.asset_id)

            if self.content_embeddings is None:
                self.content_embeddings = content_emb.reshape(1, -1)
                self.style_embeddings = style_emb.reshape(1, -1)
            else:
                self.content_embeddings = np.vstack([self.content_embeddings, content_emb])
                self.style_embeddings = np.vstack([self.style_embeddings, style_emb])

        # Store asset
        self.assets[asset.asset_id] = asset

    def search_by_content(
        self,
        query_image: Image.Image,
        top_k: int = 10
    ) -> List[Tuple[MediaAsset, float]]:
        """
        Search by visual content (semantic similarity)

        Args:
            query_image: Query image
            top_k: Number of results

        Returns:
            List of (asset, score) tuples
        """
        # Encode query
        content_emb, _ = self.encode_image(query_image)

        # Compute similarities
        scores = np.dot(self.content_embeddings, content_emb)

        # Rank results
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (self.assets[self.asset_ids[idx]], scores[idx])
            for idx in ranked_indices
        ]

        return results

    def search_by_style(
        self,
        query_image: Image.Image,
        top_k: int = 10
    ) -> List[Tuple[MediaAsset, float]]:
        """
        Search by visual style (color, composition, texture)

        Args:
            query_image: Query image
            top_k: Number of results

        Returns:
            List of (asset, score) tuples
        """
        # Encode query style
        _, style_emb = self.encode_image(query_image)

        # Compute similarities
        scores = np.dot(self.style_embeddings, style_emb)

        # Rank results
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (self.assets[self.asset_ids[idx]], scores[idx])
            for idx in ranked_indices
        ]

        return results

    def find_duplicates(
        self,
        asset_id: str,
        hamming_threshold: int = 5
    ) -> List[Tuple[MediaAsset, int]]:
        """
        Find near-duplicate assets using perceptual hashing

        Args:
            asset_id: Source asset ID
            hamming_threshold: Maximum Hamming distance for duplicates

        Returns:
            List of (asset, hamming_distance) tuples
        """
        if asset_id not in self.assets:
            return []

        source_asset = self.assets[asset_id]
        source_hash = source_asset.perceptual_hash

        if not source_hash:
            return []

        # Compute Hamming distance to all other assets
        duplicates = []

        for other_id, other_asset in self.assets.items():
            if other_id == asset_id or not other_asset.perceptual_hash:
                continue

            # Hamming distance (count differing bits)
            distance = sum(c1 != c2 for c1, c2 in zip(source_hash, other_asset.perceptual_hash))

            if distance <= hamming_threshold:
                duplicates.append((other_asset, distance))

        # Sort by distance
        duplicates.sort(key=lambda x: x[1])

        return duplicates

# Example: Media search for stock photos
def media_search_example():
    """
    Semantic search for media assets

    Use cases:
    - Stock photo search (find similar images)
    - Duplicate detection (find copyright violations)
    - Style matching (find images with similar aesthetic)

    Scale: 100M+ images (Shutterstock/Getty scale)
    """

    # Initialize search engine
    engine = MediaSearchEngine(embedding_dim=512)

    # Create sample media assets
    assets = [
        MediaAsset(
            asset_id='img_1',
            asset_type='image',
            file_path='/media/sunset_beach.jpg',
            resolution=(1920, 1080),
            metadata={'tags': ['sunset', 'beach', 'ocean']}
        ),
        MediaAsset(
            asset_id='img_2',
            asset_type='image',
            file_path='/media/mountain_sunrise.jpg',
            resolution=(1920, 1080),
            metadata={'tags': ['sunrise', 'mountain', 'landscape']}
        ),
        MediaAsset(
            asset_id='img_3',
            asset_type='image',
            file_path='/media/beach_vacation.jpg',
            resolution=(1920, 1080),
            metadata={'tags': ['beach', 'vacation', 'tropical']}
        )
    ]

    # Generate sample images (different colors for demo)
    sample_images = [
        Image.new('RGB', (224, 224), color='orange'),  # Sunset
        Image.new('RGB', (224, 224), color='purple'),  # Sunrise
        Image.new('RGB', (224, 224), color='blue')     # Beach
    ]

    # Index assets
    for asset, image in zip(assets, sample_images):
        engine.index_asset(asset, image)

    print(f"Indexed {len(assets)} media assets")

    # Search by content
    print("\n=== Content Search: Orange query image (sunset) ===")
    query_img = Image.new('RGB', (224, 224), color='orange')
    results = engine.search_by_content(query_img, top_k=3)

    for asset, score in results:
        print(f"{asset.asset_id}: {asset.file_path} (score: {score:.3f})")
        print(f"  Tags: {asset.metadata.get('tags', [])}")

    # Search by style
    print("\n=== Style Search: Blue query image (beach) ===")
    query_img_blue = Image.new('RGB', (224, 224), color='blue')
    results = engine.search_by_style(query_img_blue, top_k=3)

    for asset, score in results:
        print(f"{asset.asset_id}: {asset.file_path} (score: {score:.3f})")

    # Find duplicates
    print("\n=== Duplicate Detection for 'img_1' ===")
    duplicates = engine.find_duplicates('img_1', hamming_threshold=10)

    if duplicates:
        for dup_asset, distance in duplicates:
            print(f"{dup_asset.asset_id}: Hamming distance = {distance}")
    else:
        print("No duplicates found")

# Uncomment to run:
# media_search_example()
