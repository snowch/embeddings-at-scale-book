import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Code from Chapter 07
# Book: Embeddings at Scale


class MultiModalSelfSupervised(nn.Module):
    """
    Multi-modal self-supervised learning

    Approach: Contrastive learning across modalities
    - Learn shared embedding space
    - Match corresponding pairs across modalities
    - No manual labels needed!

    Use cases:
    - Product images + descriptions
    - Medical images + clinical notes
    - Video + audio
    - IoT sensors + maintenance logs

    Reference: "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
    """

    def __init__(self, image_encoder, text_encoder, embedding_dim=512, temperature=0.07):
        """
        Args:
            image_encoder: Image encoder (ResNet, ViT)
            text_encoder: Text encoder (BERT, RoBERTa)
            embedding_dim: Shared embedding dimension
            temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature

        # Projection heads to shared space
        self.image_projection = nn.Linear(image_encoder.output_dim, embedding_dim)
        self.text_projection = nn.Linear(text_encoder.output_dim, embedding_dim)

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def encode_image(self, images):
        """
        Encode images to embeddings

        Args:
            images: (batch_size, channels, height, width)

        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        features = self.image_encoder(images)
        embeddings = self.image_projection(features)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    def encode_text(self, texts):
        """
        Encode texts to embeddings

        Args:
            texts: List of text strings or tokenized inputs

        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        features = self.text_encoder(texts)
        embeddings = self.text_projection(features)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    def forward(self, images, texts):
        """
        Forward pass: Compute contrastive loss

        Args:
            images: Batch of images
            texts: Corresponding text descriptions

        Returns:
            loss: Contrastive loss
            accuracy: Top-1 accuracy
        """
        # Encode both modalities
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(texts)

        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
        logits_per_text = logits_per_image.T

        # Labels: diagonal elements are positive pairs
        batch_size = image_embeddings.shape[0]
        labels = torch.arange(batch_size).to(images.device)

        # Contrastive loss (symmetric)
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_image + loss_text) / 2

        # Accuracy
        with torch.no_grad():
            pred_image = logits_per_image.argmax(dim=1)
            pred_text = logits_per_text.argmax(dim=1)
            accuracy = (
                (pred_image == labels).float().mean() + (pred_text == labels).float().mean()
            ) / 2

        return loss, accuracy.item()

    def find_similar_texts(self, image, text_candidates):
        """
        Find most similar texts for an image

        Args:
            image: Query image
            text_candidates: List of candidate texts

        Returns:
            ranked_texts: Texts ranked by similarity
            similarities: Similarity scores
        """
        image_embedding = self.encode_image(image.unsqueeze(0))
        text_embeddings = self.encode_text(text_candidates)

        # Compute similarities
        similarities = (image_embedding @ text_embeddings.T).squeeze(0)

        # Rank by similarity
        ranked_indices = similarities.argsort(descending=True)
        ranked_texts = [text_candidates[i] for i in ranked_indices]
        ranked_similarities = similarities[ranked_indices]

        return ranked_texts, ranked_similarities.tolist()

    def find_similar_images(self, text, image_candidates):
        """
        Find most similar images for a text query

        Args:
            text: Query text
            image_candidates: Batch of candidate images

        Returns:
            ranked_indices: Indices of images ranked by similarity
            similarities: Similarity scores
        """
        text_embedding = self.encode_text([text])
        image_embeddings = self.encode_image(image_candidates)

        # Compute similarities
        similarities = (text_embedding @ image_embeddings.T).squeeze(0)

        # Rank by similarity
        ranked_indices = similarities.argsort(descending=True)
        ranked_similarities = similarities[ranked_indices]

        return ranked_indices.tolist(), ranked_similarities.tolist()


# Example: Training multi-modal embeddings
def train_multimodal_ssl(image_text_pairs, num_epochs=100, batch_size=256):
    """
    Train multi-modal self-supervised model

    Args:
        image_text_pairs: Dataset of (image, text) pairs
        num_epochs: Number of training epochs
        batch_size: Batch size
    """
    # Initialize encoders
    from torchvision.models import resnet50
    from transformers import AutoModel

    # Image encoder
    image_encoder = resnet50(pretrained=True)
    image_encoder.output_dim = 2048

    # Text encoder
    text_encoder = AutoModel.from_pretrained("bert-base-uncased")
    text_encoder.output_dim = 768

    # Multi-modal model
    model = MultiModalSelfSupervised(image_encoder, text_encoder, embedding_dim=512).cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # DataLoader
    dataloader = torch.utils.data.DataLoader(
        image_text_pairs, batch_size=batch_size, shuffle=True, num_workers=8
    )

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy = 0

        for images, texts in dataloader:
            images = images.cuda()

            optimizer.zero_grad()
            loss, accuracy = model(images, texts)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy

        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")

    return model


# Enterprise application: Product matching
class ProductMatchingSystem:
    """
    Multi-modal product matching

    Use case: Match product images with descriptions
    across different catalogs (e-commerce, inventory systems)
    """

    def __init__(self, multimodal_model):
        self.model = multimodal_model
        self.product_embeddings = {}

    def index_products(self, products):
        """
        Index products (images + descriptions)

        Args:
            products: List of (product_id, image, description) tuples
        """
        for product_id, image, description in products:
            # Encode both modalities
            image_emb = self.model.encode_image(image.unsqueeze(0))
            text_emb = self.model.encode_text([description])

            # Average embeddings for robust matching
            combined_emb = (image_emb + text_emb) / 2

            self.product_embeddings[product_id] = combined_emb.cpu()

    def find_matches(self, query_image=None, query_text=None, top_k=10):
        """
        Find matching products

        Args:
            query_image: Query image (optional)
            query_text: Query text (optional)
            top_k: Number of results

        Returns:
            matches: List of (product_id, similarity) tuples
        """
        # Encode query
        if query_image is not None:
            query_emb = self.model.encode_image(query_image.unsqueeze(0))
        elif query_text is not None:
            query_emb = self.model.encode_text([query_text])
        else:
            raise ValueError("Must provide query_image or query_text")

        # Compute similarities
        similarities = {}
        for product_id, product_emb in self.product_embeddings.items():
            sim = F.cosine_similarity(query_emb, product_emb.to(query_emb.device), dim=1).item()
            similarities[product_id] = sim

        # Rank by similarity
        matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return matches
