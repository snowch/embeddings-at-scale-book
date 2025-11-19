# Code from Chapter 05
# Book: Embeddings at Scale

class MultiModalContrastive(nn.Module):
    """
    Contrastive learning for multi-modal data (text + image, text + metadata, etc.)

    Use case: E-commerce (product text + images),
              Healthcare (clinical notes + scans),
              Media (articles + photos)
    """

    def __init__(self, text_encoder, image_encoder, projection_dim=256):
        super().__init__()

        self.text_encoder = text_encoder  # BERT, etc.
        self.image_encoder = image_encoder  # ResNet, ViT, etc.

        # Separate projections for each modality
        self.text_projection = nn.Linear(768, projection_dim)
        self.image_projection = nn.Linear(2048, projection_dim)

        self.temperature = 0.07

    def forward(self, text_inputs, images):
        """
        Compute cross-modal contrastive loss

        Args:
            text_inputs: Dictionary with input_ids, attention_mask
            images: (batch_size, 3, H, W)

        Returns:
            loss: bidirectional contrastive loss
        """
        # Encode text and images
        text_features = self.text_encoder(**text_inputs).last_hidden_state[:, 0]
        image_features = self.image_encoder(images)

        # Project to shared space
        text_emb = F.normalize(self.text_projection(text_features), dim=1)
        image_emb = F.normalize(self.image_projection(image_features), dim=1)

        # Compute similarities: (batch, batch)
        logits_text_to_image = torch.matmul(text_emb, image_emb.T) / self.temperature
        logits_image_to_text = logits_text_to_image.T

        # Labels: diagonal elements are positives
        batch_size = text_emb.shape[0]
        labels = torch.arange(batch_size, device=text_emb.device)

        # Bidirectional loss
        loss_text_to_image = F.cross_entropy(logits_text_to_image, labels)
        loss_image_to_text = F.cross_entropy(logits_image_to_text, labels)

        loss = (loss_text_to_image + loss_image_to_text) / 2

        return loss
