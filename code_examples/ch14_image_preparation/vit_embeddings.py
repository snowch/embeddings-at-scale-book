"""Vision Transformer (ViT) based image embeddings."""

from typing import List, Optional, Tuple

import numpy as np


def extract_vit_embeddings(
    images: List,
    model_name: str = 'vit_b_16'
) -> np.ndarray:
    """
    Extract embeddings using Vision Transformer.

    ViT processes images as sequences of patches:
    1. Split image into fixed-size patches (16x16 or 14x14)
    2. Linearly embed each patch
    3. Add position embeddings
    4. Process through transformer layers
    5. Use [CLS] token as image embedding

    Args:
        images: List of PIL Images
        model_name: ViT variant ('vit_b_16', 'vit_b_32', 'vit_l_16')

    Returns:
        numpy array of embeddings
    """
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image

    # Select model
    model_fn = {
        'vit_b_16': models.vit_b_16,
        'vit_b_32': models.vit_b_32,
        'vit_l_16': models.vit_l_16,
    }[model_name]

    model = model_fn(weights='IMAGENET1K_V1')
    model.eval()

    # Remove classification head
    model.heads = torch.nn.Identity()

    # ViT expects 224x224 images
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


def visualize_vit_patches(
    image,
    patch_size: int = 16
) -> Tuple[np.ndarray, int]:
    """
    Visualize how ViT splits an image into patches.

    Args:
        image: PIL Image or numpy array
        patch_size: Size of each patch (typically 16 or 14)

    Returns:
        Tuple of (image with patch grid, number of patches)
    """
    import numpy as np
    from PIL import Image, ImageDraw

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Resize to standard ViT input size
    image = image.resize((224, 224))
    draw = ImageDraw.Draw(image)

    # Draw patch grid
    num_patches_per_side = 224 // patch_size

    for i in range(1, num_patches_per_side):
        # Vertical lines
        x = i * patch_size
        draw.line([(x, 0), (x, 224)], fill='red', width=1)

        # Horizontal lines
        y = i * patch_size
        draw.line([(0, y), (224, y)], fill='red', width=1)

    total_patches = num_patches_per_side ** 2

    return np.array(image), total_patches


def extract_patch_embeddings(
    image,
    model_name: str = 'vit_b_16'
) -> np.ndarray:
    """
    Extract embeddings for each patch (before aggregation).

    Useful for:
    - Region-level retrieval
    - Attention visualization
    - Fine-grained image analysis

    Args:
        image: PIL Image
        model_name: ViT variant

    Returns:
        numpy array of shape (num_patches + 1, embedding_dim)
        The +1 is for the [CLS] token
    """
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image

    model = models.vit_b_16(weights='IMAGENET1K_V1')
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        # Get patch embeddings from encoder
        # ViT encoder outputs all token representations
        x = model._process_input(tensor)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Add position embeddings
        x = x + model.encoder.pos_embedding

        # Pass through transformer encoder
        for layer in model.encoder.layers:
            x = layer(x)

        x = model.encoder.ln(x)

        # Return all token embeddings (CLS + patches)
        return x.squeeze().numpy()


class ViTEmbedder:
    """
    Vision Transformer embedder with support for:
    - Different model sizes
    - Batch processing
    - GPU acceleration
    - Patch-level embeddings
    """

    def __init__(
        self,
        model_name: str = 'vit_b_16',
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.model_name = model_name

        # Model dimensions
        self.embedding_dims = {
            'vit_b_16': 768,
            'vit_b_32': 768,
            'vit_l_16': 1024,
        }
        self.embedding_dim = self.embedding_dims[model_name]

        # Load model
        model_fn = {
            'vit_b_16': models.vit_b_16,
            'vit_b_32': models.vit_b_32,
            'vit_l_16': models.vit_l_16,
        }[model_name]

        self.model = model_fn(weights='IMAGENET1K_V1')
        self.model.heads = torch.nn.Identity()
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

    def encode(
        self,
        images: List,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode images to embeddings.

        Args:
            images: List of PIL Images
            normalize: Whether to L2-normalize embeddings

        Returns:
            numpy array of shape (num_images, embedding_dim)
        """
        import torch
        from PIL import Image

        all_embeddings = []

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]

            tensors = []
            for img in batch_images:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                tensors.append(self.preprocess(img))

            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                embeddings = self.model(batch_tensor)

                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


# Example usage
if __name__ == "__main__":
    from PIL import Image

    # Create sample image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    sample_image = Image.fromarray(img_array)

    print("Vision Transformer Embeddings")
    print("=" * 50)

    # Visualize patches
    patched_img, num_patches = visualize_vit_patches(sample_image, patch_size=16)
    print(f"Number of patches (16x16): {num_patches}")

    # Extract CLS token embedding
    print("\nExtracting ViT embeddings...")
    embeddings = extract_vit_embeddings([sample_image])
    print(f"CLS token embedding shape: {embeddings.shape}")

    # Extract all patch embeddings
    print("\nExtracting patch-level embeddings...")
    patch_embeddings = extract_patch_embeddings(sample_image)
    print(f"All token embeddings shape: {patch_embeddings.shape}")
    print("  - CLS token: 1")
    print(f"  - Patch tokens: {patch_embeddings.shape[0] - 1}")

    # Using the embedder class
    print("\nUsing ViTEmbedder class...")
    embedder = ViTEmbedder(model_name='vit_b_16')
    embeddings = embedder.encode([sample_image])
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedder.embedding_dim}")
