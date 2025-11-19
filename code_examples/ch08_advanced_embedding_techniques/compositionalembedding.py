# Code from Chapter 08
# Book: Embeddings at Scale

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositionalEmbedding(nn.Module):
    """
    Learn to compose embeddings from multiple components

    Three composition strategies:
    1. Weighted sum: Learn component importance weights
    2. Gated combination: Components gate each other (like LSTM gates)
    3. Attention-based: Components attend to each other

    Applications:
    - Multi-field entity embeddings (user with demographics + behavior)
    - Hierarchical documents (title + paragraphs + metadata)
    - Product embeddings (category + brand + attributes + reviews)
    - Transaction embeddings (who + what + when + where + how much)
    """

    def __init__(
        self,
        component_dims,
        output_dim=256,
        composition_type='attention'
    ):
        """
        Args:
            component_dims: Dict mapping component name → embedding dimension
                          e.g., {'title': 128, 'body': 512, 'author': 64}
            output_dim: Final composed embedding dimension
            composition_type: 'weighted', 'gated', or 'attention'
        """
        super().__init__()
        self.component_dims = component_dims
        self.output_dim = output_dim
        self.composition_type = composition_type

        # Project each component to common dimension
        self.component_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in component_dims.items()
        })

        if composition_type == 'weighted':
            # Learned weights for each component
            self.component_weights = nn.Parameter(
                torch.ones(len(component_dims)) / len(component_dims)
            )

        elif composition_type == 'gated':
            # Gate networks for each component
            self.gates = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(output_dim, output_dim),
                    nn.Sigmoid()
                )
                for name in component_dims
            })

        elif composition_type == 'attention':
            # Multi-head attention for component composition
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=4,
                batch_first=True
            )

            # Query: what we're looking for
            self.query_projection = nn.Linear(output_dim, output_dim)

    def forward(self, component_embeddings, component_mask=None):
        """
        Compose component embeddings into unified representation

        Args:
            component_embeddings: Dict mapping component name → tensor
                                 e.g., {'title': (batch, 128), 'body': (batch, 512)}
            component_mask: Optional mask for missing components

        Returns:
            composed_embedding: (batch, output_dim)
        """
        # Project all components to common dimension
        projected = {}
        for name, emb in component_embeddings.items():
            projected[name] = self.component_projections[name](emb)

        if self.composition_type == 'weighted':
            return self._weighted_composition(projected)
        elif self.composition_type == 'gated':
            return self._gated_composition(projected)
        elif self.composition_type == 'attention':
            return self._attention_composition(projected, component_mask)

    def _weighted_composition(self, projected):
        """Simple weighted sum of components"""
        # Ensure weights are positive and sum to 1
        weights = F.softmax(self.component_weights, dim=0)

        # Stack component embeddings
        component_list = list(projected.values())
        stacked = torch.stack(component_list, dim=1)  # (batch, num_components, dim)

        # Weighted sum
        composed = torch.sum(
            stacked * weights.view(1, -1, 1),
            dim=1
        )

        return composed

    def _gated_composition(self, projected):
        """
        Gated composition: components gate each other

        Similar to LSTM gates but for composition:
        - Each component generates a gate
        - Gates control information flow from other components
        """
        batch_size = next(iter(projected.values())).shape[0]
        composed = torch.zeros(batch_size, self.output_dim)

        # Each component contributes based on its gate
        for name, emb in projected.items():
            gate = self.gates[name](emb)
            composed = composed + gate * emb

        # Normalize
        composed = composed / len(projected)

        return composed

    def _attention_composition(self, projected, component_mask=None):
        """
        Attention-based composition

        Components attend to each other to determine importance
        Captures interactions between components
        """
        # Stack components for attention
        component_list = list(projected.values())
        stacked = torch.stack(component_list, dim=1)  # (batch, num_comp, dim)

        # Use mean as query (could also be learned)
        query = stacked.mean(dim=1, keepdim=True)  # (batch, 1, dim)
        query = self.query_projection(query)

        # Multi-head attention
        attended, attention_weights = self.attention(
            query=query,
            key=stacked,
            value=stacked,
            key_padding_mask=component_mask
        )

        return attended.squeeze(1)  # (batch, dim)

class ProductEmbedding(nn.Module):
    """
    Compositional embeddings for products

    Combines:
    - Category hierarchy (from hierarchical embeddings)
    - Brand information
    - Product attributes (color, size, material, etc.)
    - Review sentiment and keywords
    - Visual features (from product images)
    - Temporal features (seasonality, trends)

    Applications:
    - Product search and discovery
    - Recommendation systems
    - Dynamic pricing
    - Inventory optimization
    """

    def __init__(
        self,
        num_categories=10000,
        num_brands=5000,
        num_attributes=1000,
        embedding_dim=256
    ):
        super().__init__()

        # Component embeddings
        self.category_emb = nn.Embedding(num_categories, embedding_dim)
        self.brand_emb = nn.Embedding(num_brands, embedding_dim)
        self.attribute_emb = nn.Embedding(num_attributes, embedding_dim)

        # Text encoder for reviews (pretrained BERT/RoBERTa)
        self.review_encoder = nn.Linear(768, embedding_dim)  # BERT → product dim

        # Image encoder (pretrained ResNet/ViT)
        self.image_encoder = nn.Linear(2048, embedding_dim)  # ResNet → product dim

        # Compositional model
        self.compositor = CompositionalEmbedding(
            component_dims={
                'category': embedding_dim,
                'brand': embedding_dim,
                'attributes': embedding_dim,
                'reviews': embedding_dim,
                'image': embedding_dim
            },
            output_dim=embedding_dim,
            composition_type='attention'
        )

    def forward(
        self,
        category_id,
        brand_id,
        attribute_ids,
        review_embedding,
        image_features
    ):
        """
        Create compositional product embedding

        Args:
            category_id: Category ID (batch,)
            brand_id: Brand ID (batch,)
            attribute_ids: Multiple attributes (batch, max_attributes)
            review_embedding: Pre-computed review embedding (batch, 768)
            image_features: Pre-extracted image features (batch, 2048)

        Returns:
            product_embedding: Composed representation (batch, embedding_dim)
        """
        # Encode each component
        category_vec = self.category_emb(category_id)
        brand_vec = self.brand_emb(brand_id)

        # Attributes: pool multiple attribute embeddings
        attr_vecs = self.attribute_emb(attribute_ids)  # (batch, max_attr, dim)
        attribute_vec = attr_vecs.mean(dim=1)  # (batch, dim)

        # Project text and image features
        review_vec = self.review_encoder(review_embedding)
        image_vec = self.image_encoder(image_features)

        # Compose all components
        components = {
            'category': category_vec,
            'brand': brand_vec,
            'attributes': attribute_vec,
            'reviews': review_vec,
            'image': image_vec
        }

        product_emb = self.compositor(components)

        return product_emb

class DocumentEmbedding(nn.Module):
    """
    Compositional document embeddings

    Combines:
    - Title (high importance, short)
    - Abstract/summary (medium importance, medium length)
    - Body (lower importance per token, long)
    - Metadata (author, date, venue)
    - Citations (who cites this, who is cited)
    - Figures/tables (visual content)

    Use cases:
    - Scientific paper search and recommendation
    - Legal document analysis
    - Patent search and prior art detection
    - News article clustering and tracking
    """

    def __init__(self, embedding_dim=512):
        super().__init__()

        # Different encoders for different text fields
        # Title: capture key concepts
        self.title_encoder = nn.LSTM(
            embedding_dim, embedding_dim,
            num_layers=1, batch_first=True
        )

        # Abstract: summary-level encoding
        self.abstract_encoder = nn.LSTM(
            embedding_dim, embedding_dim,
            num_layers=2, batch_first=True
        )

        # Body: hierarchical encoding (section → paragraph → sentence)
        self.body_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                batch_first=True
            ),
            num_layers=3
        )

        # Metadata encoder
        self.author_emb = nn.Embedding(100000, embedding_dim)  # 100K authors
        self.venue_emb = nn.Embedding(10000, embedding_dim)    # 10K venues

        # Compositional model
        self.compositor = CompositionalEmbedding(
            component_dims={
                'title': embedding_dim,
                'abstract': embedding_dim,
                'body': embedding_dim,
                'author': embedding_dim,
                'venue': embedding_dim
            },
            output_dim=embedding_dim,
            composition_type='attention'
        )

    def forward(
        self,
        title_tokens,
        abstract_tokens,
        body_tokens,
        author_id,
        venue_id
    ):
        """
        Create compositional document embedding

        Different components weighted based on task:
        - Citation recommendation: weight authors/venue heavily
        - Semantic search: weight title/abstract heavily
        - Duplicate detection: weight full body heavily
        """
        # Encode title (use final hidden state)
        _, (title_hidden, _) = self.title_encoder(title_tokens)
        title_vec = title_hidden[-1]

        # Encode abstract
        _, (abstract_hidden, _) = self.abstract_encoder(abstract_tokens)
        abstract_vec = abstract_hidden[-1]

        # Encode body (use mean pooling over transformer outputs)
        body_encoded = self.body_encoder(body_tokens)
        body_vec = body_encoded.mean(dim=1)

        # Encode metadata
        author_vec = self.author_emb(author_id)
        venue_vec = self.venue_emb(venue_id)

        # Compose
        components = {
            'title': title_vec,
            'abstract': abstract_vec,
            'body': body_vec,
            'author': author_vec,
            'venue': venue_vec
        }

        doc_emb = self.compositor(components)

        return doc_emb
