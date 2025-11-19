# Code from Chapter 14
# Book: Embeddings at Scale

"""
Multi-Modal Semantic Search System

Architecture:
1. Modality encoders: Separate encoders for text, images, audio
2. Projection layers: Map to shared embedding space
3. Contrastive learning: Train encoders to align modalities
4. Cross-modal retrieval: Query in one modality, retrieve in another

Key techniques:
- CLIP (Contrastive Language-Image Pre-training)
- ALIGN (A Large-scale ImaGe and Noisy-text embedding)
- ImageBind (binds 6 modalities: images, text, audio, depth, thermal, IMU)

Production considerations:
- Modality-specific preprocessing (resize images, normalize audio)
- Batch encoding for efficiency
- Index separate modalities, unify at query time
- Handle missing modalities gracefully
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


@dataclass
class MultiModalQuery:
    """
    Multi-modal search query

    Attributes:
        text: Text query (optional)
        image: Image query as PIL Image (optional)
        audio: Audio query as waveform (optional)
        modality_weights: Weights for each modality when combining
    """
    text: Optional[str] = None
    image: Optional[Image.Image] = None
    audio: Optional[np.ndarray] = None
    modality_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.modality_weights is None:
            # Default: Equal weighting
            active_modalities = sum([
                self.text is not None,
                self.image is not None,
                self.audio is not None
            ])
            if active_modalities > 0:
                weight = 1.0 / active_modalities
                self.modality_weights = {}
                if self.text:
                    self.modality_weights['text'] = weight
                if self.image:
                    self.modality_weights['image'] = weight
                if self.audio:
                    self.modality_weights['audio'] = weight

@dataclass
class MultiModalDocument:
    """
    Multi-modal document with content in multiple modalities

    Attributes:
        doc_id: Unique identifier
        text: Text content (optional)
        image: Image content (optional)
        audio: Audio content (optional)
        metadata: Additional metadata
        embeddings: Cached embeddings per modality
    """
    doc_id: str
    text: Optional[str] = None
    image: Optional[Image.Image] = None
    audio: Optional[np.ndarray] = None
    metadata: Dict = None
    embeddings: Optional[Dict[str, np.ndarray]] = None

class TextEncoder(nn.Module):
    """
    Text encoder for multi-modal embeddings

    Architecture:
    - Transformer encoder (BERT-style)
    - Projects to shared embedding space (512-dim)

    In production: Use pre-trained CLIP text encoder
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 512,
        hidden_dim: int = 768,
        num_layers: int = 12
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection to shared space
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode text to embeddings

        Args:
            token_ids: Token IDs (batch_size, seq_len)

        Returns:
            Text embeddings (batch_size, embedding_dim)
        """
        # Embed tokens
        x = self.token_embedding(token_ids)  # (batch, seq_len, hidden_dim)

        # Encode with transformer
        x = self.transformer(x)  # (batch, seq_len, hidden_dim)

        # Pool: Take [CLS] token (first position)
        x = x[:, 0, :]  # (batch, hidden_dim)

        # Project to shared space
        x = self.projection(x)  # (batch, embedding_dim)

        # Normalize
        x = F.normalize(x, p=2, dim=1)

        return x

class ImageEncoder(nn.Module):
    """
    Image encoder for multi-modal embeddings

    Architecture:
    - Vision transformer (ViT) or ResNet
    - Projects to shared embedding space (512-dim)

    In production: Use pre-trained CLIP image encoder
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        image_size: int = 224
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.image_size = image_size

        # Simple CNN for demonstration
        # In production: Use ViT or ResNet50
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Projection to shared space
        self.projection = nn.Linear(256, embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings

        Args:
            images: Images (batch_size, 3, height, width)

        Returns:
            Image embeddings (batch_size, embedding_dim)
        """
        # Extract visual features
        x = self.conv_layers(images)  # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 256)

        # Project to shared space
        x = self.projection(x)  # (batch, embedding_dim)

        # Normalize
        x = F.normalize(x, p=2, dim=1)

        return x

class AudioEncoder(nn.Module):
    """
    Audio encoder for multi-modal embeddings

    Architecture:
    - Mel-spectrogram frontend
    - CNN or transformer encoder
    - Projects to shared embedding space (512-dim)
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        sample_rate: int = 16000,
        n_mels: int = 128
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # CNN for mel-spectrogram
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Projection to shared space
        self.projection = nn.Linear(128, embedding_dim)

    def forward(self, mel_spectrograms: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to embeddings

        Args:
            mel_spectrograms: Mel spectrograms (batch_size, 1, n_mels, time)

        Returns:
            Audio embeddings (batch_size, embedding_dim)
        """
        # Extract audio features
        x = self.conv_layers(mel_spectrograms)  # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 128)

        # Project to shared space
        x = self.projection(x)  # (batch, embedding_dim)

        # Normalize
        x = F.normalize(x, p=2, dim=1)

        return x

class MultiModalSearchEngine:
    """
    Multi-modal semantic search engine

    Architecture:
    1. Modality encoders: Text, image, audio
    2. Unified index: All modalities in shared vector space
    3. Cross-modal retrieval: Query in any modality, retrieve from any

    Capabilities:
    - Text → Image: "cat" finds cat photos
    - Image → Text: Upload cat photo, find articles about cats
    - Audio → Video: Hum melody, find music videos
    - Multi-modal queries: Text + Image together

    Production optimizations:
    - Pre-encode all documents (offline)
    - Store per-modality indices separately
    - Combine scores at query time (late fusion)
    - Cache popular queries
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        device: str = 'cuda'
    ):
        """
        Args:
            embedding_dim: Dimension of shared embedding space
            device: Device for computation ('cuda' or 'cpu')
        """
        self.embedding_dim = embedding_dim
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Initialize encoders
        self.text_encoder = TextEncoder(embedding_dim=embedding_dim).to(self.device)
        self.image_encoder = ImageEncoder(embedding_dim=embedding_dim).to(self.device)
        self.audio_encoder = AudioEncoder(embedding_dim=embedding_dim).to(self.device)

        # Set to eval mode
        self.text_encoder.eval()
        self.image_encoder.eval()
        self.audio_encoder.eval()

        # Document store: doc_id -> MultiModalDocument
        self.documents: Dict[str, MultiModalDocument] = {}

        # Per-modality indices: modality -> (doc_ids, embeddings)
        self.indices: Dict[str, Tuple[List[str], np.ndarray]] = {
            'text': ([], np.array([])),
            'image': ([], np.array([])),
            'audio': ([], np.array([]))
        }

        print("Initialized Multi-Modal Search Engine")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Device: {self.device}")

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text to embeddings

        Args:
            texts: List of text strings

        Returns:
            Text embeddings (len(texts), embedding_dim)
        """
        # Tokenize (simplified - use proper tokenizer in production)
        # For demo: Hash to token IDs
        max_len = 77  # CLIP uses 77
        token_ids = []
        for text in texts:
            ids = [hash(word) % 50000 for word in text.lower().split()[:max_len]]
            # Pad to max_len
            ids = ids + [0] * (max_len - len(ids))
            token_ids.append(ids)

        token_ids = torch.tensor(token_ids, dtype=torch.long).to(self.device)

        # Encode
        with torch.no_grad():
            embeddings = self.text_encoder(token_ids)

        return embeddings.cpu().numpy()

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        Encode images to embeddings

        Args:
            images: List of PIL Images

        Returns:
            Image embeddings (len(images), embedding_dim)
        """
        # Preprocess images
        image_tensors = []
        for img in images:
            # Resize to 224x224
            img = img.resize((224, 224))
            # Convert to tensor (normalize to [0, 1])
            img_array = np.array(img).astype(np.float32) / 255.0
            # Handle grayscale
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            # (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            image_tensors.append(img_tensor)

        images_batch = torch.stack(image_tensors).to(self.device)

        # Encode
        with torch.no_grad():
            embeddings = self.image_encoder(images_batch)

        return embeddings.cpu().numpy()

    def encode_audio(self, audio_samples: List[np.ndarray]) -> np.ndarray:
        """
        Encode audio to embeddings

        Args:
            audio_samples: List of audio waveforms

        Returns:
            Audio embeddings (len(audio_samples), embedding_dim)
        """
        # Convert to mel spectrograms (simplified)
        mel_specs = []
        for _audio in audio_samples:
            # In production: Use librosa.feature.melspectrogram
            # For demo: Create dummy mel spectrogram
            mel_spec = np.random.randn(1, 128, 100).astype(np.float32)
            mel_specs.append(torch.from_numpy(mel_spec))

        mel_batch = torch.stack(mel_specs).to(self.device)

        # Encode
        with torch.no_grad():
            embeddings = self.audio_encoder(mel_batch)

        return embeddings.cpu().numpy()

    def index_documents(self, documents: List[MultiModalDocument]):
        """
        Index multi-modal documents

        Process:
        1. Encode each modality present in documents
        2. Store embeddings in per-modality indices
        3. Store documents for retrieval

        Args:
            documents: Documents to index
        """
        print(f"Indexing {len(documents)} multi-modal documents...")

        # Separate by modality
        text_docs = [(i, doc) for i, doc in enumerate(documents) if doc.text]
        image_docs = [(i, doc) for i, doc in enumerate(documents) if doc.image]
        audio_docs = [(i, doc) for i, doc in enumerate(documents) if doc.audio]

        # Encode text
        if text_docs:
            texts = [doc.text for _, doc in text_docs]
            text_embeddings = self.encode_text(texts)

            # Update text index
            doc_ids = [doc.doc_id for _, doc in text_docs]
            existing_ids, existing_embs = self.indices['text']

            if len(existing_embs) > 0:
                self.indices['text'] = (
                    list(existing_ids) + doc_ids,
                    np.vstack([existing_embs, text_embeddings])
                )
            else:
                self.indices['text'] = (doc_ids, text_embeddings)

            print(f"  Indexed {len(text_docs)} text documents")

        # Encode images
        if image_docs:
            images = [doc.image for _, doc in image_docs]
            image_embeddings = self.encode_images(images)

            # Update image index
            doc_ids = [doc.doc_id for _, doc in image_docs]
            existing_ids, existing_embs = self.indices['image']

            if len(existing_embs) > 0:
                self.indices['image'] = (
                    list(existing_ids) + doc_ids,
                    np.vstack([existing_embs, image_embeddings])
                )
            else:
                self.indices['image'] = (doc_ids, image_embeddings)

            print(f"  Indexed {len(image_docs)} image documents")

        # Encode audio
        if audio_docs:
            audio_samples = [doc.audio for _, doc in audio_docs]
            audio_embeddings = self.encode_audio(audio_samples)

            # Update audio index
            doc_ids = [doc.doc_id for _, doc in audio_docs]
            existing_ids, existing_embs = self.indices['audio']

            if len(existing_embs) > 0:
                self.indices['audio'] = (
                    list(existing_ids) + doc_ids,
                    np.vstack([existing_embs, audio_embeddings])
                )
            else:
                self.indices['audio'] = (doc_ids, audio_embeddings)

            print(f"  Indexed {len(audio_docs)} audio documents")

        # Store documents
        for doc in documents:
            self.documents[doc.doc_id] = doc

        print("✓ Indexing complete")
        print(f"  Total documents: {len(self.documents)}")
        print(f"  Text index: {len(self.indices['text'][0])}")
        print(f"  Image index: {len(self.indices['image'][0])}")
        print(f"  Audio index: {len(self.indices['audio'][0])}")

    def search(
        self,
        query: MultiModalQuery,
        top_k: int = 10,
        modality_filter: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Search across modalities

        Cross-modal retrieval:
        - Query in one modality, retrieve from all modalities
        - Combine scores from multiple query modalities

        Args:
            query: Multi-modal query
            top_k: Number of results to return
            modality_filter: Only search this modality (optional)

        Returns:
            List of (doc_id, score) tuples
        """
        # Encode query modalities
        query_embeddings = {}

        if query.text:
            text_emb = self.encode_text([query.text])[0]
            query_embeddings['text'] = text_emb

        if query.image:
            image_emb = self.encode_images([query.image])[0]
            query_embeddings['image'] = image_emb

        if query.audio:
            audio_emb = self.encode_audio([query.audio])[0]
            query_embeddings['audio'] = audio_emb

        if not query_embeddings:
            return []

        # Search in each document modality
        all_scores = {}  # doc_id -> score

        # Determine which document modalities to search
        search_modalities = [modality_filter] if modality_filter else ['text', 'image', 'audio']

        for doc_modality in search_modalities:
            doc_ids, doc_embeddings = self.indices[doc_modality]

            if len(doc_ids) == 0:
                continue

            # Compute scores for each query modality
            for query_modality, query_emb in query_embeddings.items():
                # Cosine similarity
                scores = np.dot(doc_embeddings, query_emb)

                # Weight by modality
                weight = query.modality_weights.get(query_modality, 1.0)

                # Accumulate scores
                for doc_id, score in zip(doc_ids, scores):
                    if doc_id not in all_scores:
                        all_scores[doc_id] = 0.0
                    all_scores[doc_id] += weight * score

        # Sort by score
        ranked_results = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        return ranked_results[:top_k]

# Example: Multi-modal product search
def multimodal_search_example():
    """
    Multi-modal search for e-commerce products

    Use cases:
    - Text → Image: "red dress" finds red dress photos
    - Image → Product: Upload dress photo, find similar products
    - Text + Image: "red dress" + style image finds matching products

    Scale: 10M products with images and descriptions
    """

    # Initialize search engine
    engine = MultiModalSearchEngine(embedding_dim=512)

    # Create sample products
    products = [
        MultiModalDocument(
            doc_id='product_1',
            text='Red summer dress with floral pattern',
            image=Image.new('RGB', (224, 224), color='red'),
            metadata={'category': 'clothing', 'price': 49.99}
        ),
        MultiModalDocument(
            doc_id='product_2',
            text='Blue denim jeans with distressed look',
            image=Image.new('RGB', (224, 224), color='blue'),
            metadata={'category': 'clothing', 'price': 79.99}
        ),
        MultiModalDocument(
            doc_id='product_3',
            text='Wireless bluetooth headphones with noise cancellation',
            image=Image.new('RGB', (224, 224), color='black'),
            metadata={'category': 'electronics', 'price': 199.99}
        )
    ]

    # Index products
    engine.index_documents(products)

    # Search: Text query
    print("\n=== Text Query: 'red dress' ===")
    query1 = MultiModalQuery(text='red dress')
    results1 = engine.search(query1, top_k=3)

    for doc_id, score in results1:
        doc = engine.documents[doc_id]
        print(f"{doc_id}: {doc.text[:50]}... (score: {score:.3f})")

    # Search: Image query
    print("\n=== Image Query: Red image ===")
    query_image = Image.new('RGB', (224, 224), color='red')
    query2 = MultiModalQuery(image=query_image)
    results2 = engine.search(query2, top_k=3)

    for doc_id, score in results2:
        doc = engine.documents[doc_id]
        print(f"{doc_id}: {doc.text[:50]}... (score: {score:.3f})")

    # Search: Multi-modal query (text + image)
    print("\n=== Multi-Modal Query: 'dress' + blue image ===")
    query_image_blue = Image.new('RGB', (224, 224), color='blue')
    query3 = MultiModalQuery(
        text='dress',
        image=query_image_blue,
        modality_weights={'text': 0.6, 'image': 0.4}
    )
    results3 = engine.search(query3, top_k=3)

    for doc_id, score in results3:
        doc = engine.documents[doc_id]
        print(f"{doc_id}: {doc.text[:50]}... (score: {score:.3f})")

# Uncomment to run:
# multimodal_search_example()
