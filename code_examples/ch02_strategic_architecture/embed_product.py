import torch

# Code from Chapter 02
# Book: Embeddings at Scale


# Import ModalityFusion from same directory
# In production: from .modalityfusion import ModalityFusion
class ModalityFusion:
    """Placeholder for ModalityFusion. See modalityfusion.py for full implementation."""

    @staticmethod
    def early_fusion(modality_embeddings, weights=None):
        if weights is None:
            weights = [1.0 / len(modality_embeddings)] * len(modality_embeddings)
        fused = sum(
            w * torch.tensor(emb) if not isinstance(emb, torch.Tensor) else w * emb
            for w, emb in zip(weights, modality_embeddings)
        )
        return fused / torch.norm(fused)


# Placeholder encoder with multiple encoding methods
class MultiModalEncoder:
    """Placeholder multi-modal encoder. Replace with actual model."""

    def encode_text(self, text):
        return torch.randn(768)

    def encode_image(self, image):
        return torch.randn(768)

    def encode_structured(self, data):
        return torch.randn(768)


encoder = MultiModalEncoder()


def embed_product(product):
    """Create comprehensive product embedding"""
    embeddings = []
    weights = []

    # Text: title + description + specifications
    text = f"{product.title} {product.description} {product.specifications}"
    text_emb = encoder.encode_text(text)
    embeddings.append(text_emb)
    weights.append(0.3)

    # Images: product images
    if product.images:
        image_embs = [encoder.encode_image(img) for img in product.images]
        product_image_emb = torch.stack(image_embs).mean(dim=0)
        embeddings.append(product_image_emb)
        weights.append(0.4)

    # Reviews: customer feedback
    if product.reviews:
        review_texts = [review.text for review in product.reviews[:50]]  # Top 50 reviews
        review_emb = encoder.encode_text(" ".join(review_texts))
        embeddings.append(review_emb)
        weights.append(0.15)

    # Structured: price, rating, category, brand
    structured_emb = encoder.encode_structured(
        {
            "price": product.price,
            "rating": product.avg_rating,
            "num_reviews": product.num_reviews,
            "category": product.category,
            "brand": product.brand,
        }
    )
    embeddings.append(structured_emb)
    weights.append(0.15)

    # Fused embedding
    product_emb = ModalityFusion.early_fusion(embeddings, weights)

    return product_emb
