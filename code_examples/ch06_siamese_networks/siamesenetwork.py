# Code from Chapter 06
# Book: Embeddings at Scale

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    """
    Siamese Network for learning similarity metrics

    Architecture: Two identical networks (shared weights) process different
    inputs, producing embeddings that are compared using a distance metric.

    Use cases:
    - Face verification: "Is this the same person?"
    - Document similarity: "Are these papers related?"
    - Product matching: "Are these the same item?"
    - Anomaly detection: "Is this different from normal?"
    """

    def __init__(self, embedding_net, embedding_dim=512):
        """
        Args:
            embedding_net: The base network for creating embeddings
                          (e.g., ResNet, BERT, custom architecture)
            embedding_dim: Dimension of output embeddings
        """
        super().__init__()
        self.embedding_net = embedding_net
        self.embedding_dim = embedding_dim

    def forward(self, x1, x2):
        """
        Forward pass through Siamese network

        Args:
            x1: First input (batch_size, ...)
            x2: Second input (batch_size, ...)

        Returns:
            embedding1: Embeddings for x1 (batch_size, embedding_dim)
            embedding2: Embeddings for x2 (batch_size, embedding_dim)
        """
        # Both inputs go through the SAME network (shared weights)
        embedding1 = self.embedding_net(x1)
        embedding2 = self.embedding_net(x2)

        return embedding1, embedding2

    def get_embedding(self, x):
        """Get embedding for a single input"""
        return self.embedding_net(x)


class EmbeddingNet(nn.Module):
    """
    Example embedding network for structured/tabular data

    For images: Use ResNet, EfficientNet, Vision Transformer
    For text: Use BERT, RoBERTa, sentence transformers
    For multimodal: Use CLIP-style architectures
    """

    def __init__(self, input_dim, embedding_dim=512, hidden_dims=[1024, 512]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        # Final embedding layer
        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            embeddings: L2-normalized embeddings (batch_size, embedding_dim)
        """
        embeddings = self.network(x)
        # L2 normalization for cosine similarity
        return F.normalize(embeddings, p=2, dim=1)


# Example: Building a Siamese network for enterprise use
def create_enterprise_siamese_network(input_type='tabular', input_dim=None):
    """
    Factory function for creating Siamese networks

    Args:
        input_type: 'tabular', 'image', 'text', or 'multimodal'
        input_dim: Input dimension (for tabular data)

    Returns:
        SiameseNetwork instance configured for the input type
    """

    if input_type == 'tabular':
        if input_dim is None:
            raise ValueError("input_dim required for tabular data")
        embedding_net = EmbeddingNet(
            input_dim=input_dim,
            embedding_dim=512,
            hidden_dims=[1024, 768, 512]
        )

    elif input_type == 'image':
        # Use pre-trained ResNet
        import torchvision.models as models
        resnet = models.resnet50(pretrained=True)
        # Remove classification head
        embedding_net = nn.Sequential(*list(resnet.children())[:-1])

    elif input_type == 'text':
        # Use transformer-based encoder
        from transformers import AutoModel
        embedding_net = AutoModel.from_pretrained('bert-base-uncased')

    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    return SiameseNetwork(embedding_net, embedding_dim=512)
