# Code from Chapter 01
# Book: Embeddings at Scale

"""
Image Embeddings with ResNet

Demonstrates how to create image embeddings using a pre-trained ResNet model.
Images are converted to vectors that capture visual content, enabling
similarity search across images.
"""

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# Load pre-trained image model (suppress download progress bar)
weights = ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=None)
model.load_state_dict(weights.get_state_dict(progress=False))
model.eval()

# Remove final classification layer to get embeddings
embedding_model = torch.nn.Sequential(*list(model.children())[:-1])

# Transform image to tensor
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_image_embedding(image_path):
    """Convert image to embedding vector"""
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        embedding = embedding_model(img_tensor)

    return embedding.squeeze().numpy()  # 2048-dimensional vector


# Use case: find visually similar images
# image1_emb = get_image_embedding('cat1.jpg')
# image2_emb = get_image_embedding('cat2.jpg')
# similarity = cosine_similarity([image1_emb], [image2_emb])
