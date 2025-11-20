# Code from Chapter 01
# Book: Embeddings at Scale

"""
Product Embedder - Multi-Modal Product Embeddings

Demonstrates how e-commerce platforms can embed products using multiple signals:
text (title + description), images, and categorical features.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_image_model():
    """Placeholder for image model loading"""

    # In practice, use a model like ResNet or CLIP
    class MockImageModel:
        def encode(self, image):
            # Returns a dummy embedding
            return np.random.rand(512)

    return MockImageModel()


class ProductEmbedder:
    """Embed products using multiple signals"""

    def __init__(self):
        self.text_model = SentenceTransformer("all-mpnet-base-v2")
        self.image_model = load_image_model()

    def embed_product(self, product):
        """Create product embedding from multiple modalities"""
        # Text embedding from title + description
        text = f"{product.title} {product.description}"
        text_embedding = self.text_model.encode(text)

        # Image embedding from product photo
        image_embedding = self.image_model.encode(product.image)

        # Combine embeddings (simple concatenation)
        combined = np.concatenate([text_embedding, image_embedding])

        # Optional: add categorical features
        category_features = self.encode_categories(product.category)

        return np.concatenate([combined, category_features])

    def encode_categories(self, category):
        """Encode categorical features"""
        # Placeholder implementation
        return np.zeros(64)

    def find_similar_products(self, product, all_products, top_k=10):
        """Find products similar to the query product"""
        query_embedding = self.embed_product(product)
        all_embeddings = [self.embed_product(p) for p in all_products]

        similarities = cosine_similarity([query_embedding], all_embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [all_products[i] for i in top_indices]
