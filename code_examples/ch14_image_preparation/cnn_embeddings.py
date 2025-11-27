"""CNN-based image embedding extraction."""

from typing import List, Optional

import numpy as np


def extract_resnet_embeddings(
    images: List,
    model_name: str = 'resnet50',
    layer: str = 'avg_pool'
) -> np.ndarray:
    """
    Extract embeddings using a pre-trained ResNet model.

    Args:
        images: List of PIL Images or numpy arrays
        model_name: ResNet variant ('resnet18', 'resnet50', 'resnet101')
        layer: Which layer to extract features from

    Returns:
        numpy array of shape (num_images, embedding_dim)
    """
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image

    # Select model
    model_fn = {
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
    }[model_name]

    model = model_fn(weights='IMAGENET1K_V1')
    model.eval()

    # Remove final classification layer to get embeddings
    # ResNet: features come from avgpool layer (before fc)
    modules = list(model.children())[:-1]  # Remove fc layer
    feature_extractor = torch.nn.Sequential(*modules)

    # Standard ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    embeddings = []

    with torch.no_grad():
        for img in images:
            # Convert to PIL if needed
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)

            # Preprocess and add batch dimension
            tensor = preprocess(img).unsqueeze(0)

            # Extract features
            features = feature_extractor(tensor)

            # Flatten from (1, 2048, 1, 1) to (2048,)
            embedding = features.squeeze().numpy()
            embeddings.append(embedding)

    return np.array(embeddings)


def extract_efficientnet_embeddings(
    images: List,
    model_name: str = 'efficientnet_b0'
) -> np.ndarray:
    """
    Extract embeddings using EfficientNet.

    Args:
        images: List of PIL Images
        model_name: EfficientNet variant

    Returns:
        numpy array of embeddings
    """
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image

    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.eval()

    # Remove classifier to get embeddings
    model.classifier = torch.nn.Identity()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    embeddings = []

    with torch.no_grad():
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)

            tensor = preprocess(img).unsqueeze(0)
            embedding = model(tensor).squeeze().numpy()
            embeddings.append(embedding)

    return np.array(embeddings)


class CNNEmbedder:
    """
    Wrapper class for CNN-based image embedding.

    Supports batched inference and GPU acceleration.
    """

    def __init__(
        self,
        model_name: str = 'resnet50',
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        # Load model
        if model_name == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1')
            self.embedding_dim = 2048
        elif model_name == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1')
            self.embedding_dim = 512
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Remove classifier
        modules = list(model.children())[:-1]
        self.model = torch.nn.Sequential(*modules)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def encode(self, images: List) -> np.ndarray:
        """
        Encode images to embeddings.

        Args:
            images: List of PIL Images

        Returns:
            numpy array of shape (num_images, embedding_dim)
        """
        import torch
        from PIL import Image

        all_embeddings = []

        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]

            # Preprocess batch
            tensors = []
            for img in batch_images:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                tensors.append(self.preprocess(img))

            batch_tensor = torch.stack(tensors).to(self.device)

            # Extract embeddings
            with torch.no_grad():
                features = self.model(batch_tensor)
                embeddings = features.squeeze(-1).squeeze(-1)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


# Example usage
if __name__ == "__main__":
    from PIL import Image

    # Create a sample image (gradient)
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        for j in range(224):
            img_array[i, j] = [i, j, (i + j) // 2]

    sample_image = Image.fromarray(img_array)

    # Extract embeddings using different methods
    print("Extracting ResNet50 embeddings...")
    embeddings = extract_resnet_embeddings([sample_image], model_name='resnet50')
    print(f"  Shape: {embeddings.shape}")
    print(f"  First 5 values: {embeddings[0][:5]}")

    print("\nUsing CNNEmbedder class...")
    embedder = CNNEmbedder(model_name='resnet50')
    embeddings = embedder.encode([sample_image, sample_image])
    print(f"  Shape: {embeddings.shape}")
    print(f"  Embedding dimension: {embedder.embedding_dim}")
