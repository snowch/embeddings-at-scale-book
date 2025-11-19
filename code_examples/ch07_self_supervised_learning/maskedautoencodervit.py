# Code from Chapter 07
# Book: Embeddings at Scale

import torch
import torch.nn as nn
from torchvision import transforms


class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder for Vision Transformers (MAE)

    Self-supervised learning for images:
    1. Mask random patches (75% of image)
    2. Encode visible patches with ViT
    3. Decode to reconstruct masked patches

    Use cases:
    - Manufacturing defect detection (learn from normal images)
    - Medical imaging (train on unlabeled scans)
    - Satellite imagery (learn from millions of images)
    - Security footage (understand visual patterns)

    Reference: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        mask_ratio=0.75
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Size of image patches
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Embedding dimension
            depth: Number of encoder layers
            num_heads: Number of attention heads
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder layers
            mask_ratio: Ratio of patches to mask (0.75 = 75%)
        """
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        # Encoder (ViT blocks)
        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim)
        )
        self.decoder = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, num_heads)
            for _ in range(decoder_depth)
        ])

        # Reconstruction head
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size ** 2 * in_channels
        )

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, imgs):
        """
        Convert images to patches

        Args:
            imgs: (batch_size, channels, height, width)

        Returns:
            patches: (batch_size, num_patches, patch_size^2 * channels)
        """
        batch_size = imgs.shape[0]

        # Extract patches
        patches = self.patch_embed(imgs)  # (B, embed_dim, H/P, W/P)
        patches = patches.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        return patches

    def random_masking(self, x, mask_ratio):
        """
        Random masking of patches

        Args:
            x: (batch_size, num_patches, embed_dim)
            mask_ratio: Ratio of patches to mask

        Returns:
            x_masked: Visible patches only
            mask: Binary mask (1 = masked, 0 = visible)
            ids_restore: Indices to restore original order
        """
        batch_size, num_patches, embed_dim = x.shape
        num_keep = int(num_patches * (1 - mask_ratio))

        # Random shuffle
        noise = torch.rand(batch_size, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first subset (visible patches)
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, embed_dim)
        )

        # Create mask (1 = masked, 0 = visible)
        mask = torch.ones([batch_size, num_patches], device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        """
        Forward pass through encoder

        Args:
            x: Input images (batch_size, channels, height, width)

        Returns:
            encoded: Encoded visible patches
            mask: Binary mask
            ids_restore: Indices for restoring order
        """
        # Patchify
        x = self.patchify(x)

        # Add positional embedding
        x = x + self.pos_embed

        # Random masking
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # Encoder
        for block in self.encoder:
            x = block(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Forward pass through decoder

        Args:
            x: Encoded patches
            ids_restore: Indices for restoring order

        Returns:
            reconstructed: Reconstructed patches
        """
        # Embed tokens
        x = self.decoder_embed(x)

        # Append mask tokens
        batch_size = x.shape[0]
        mask_tokens = self.mask_token.repeat(
            batch_size,
            ids_restore.shape[1] - x.shape[1],
            1
        )
        x_full = torch.cat([x, mask_tokens], dim=1)

        # Restore original order
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[2])
        )

        # Add positional embedding
        x_full = x_full + self.decoder_pos_embed

        # Decoder
        for block in self.decoder:
            x_full = block(x_full)

        # Predict patches
        reconstructed = self.decoder_pred(x_full)

        return reconstructed

    def forward(self, imgs):
        """
        Forward pass

        Args:
            imgs: Input images

        Returns:
            loss: Reconstruction loss
            reconstructed: Reconstructed images
            mask: Binary mask
        """
        # Encode
        encoded, mask, ids_restore = self.forward_encoder(imgs)

        # Decode
        reconstructed = self.forward_decoder(encoded, ids_restore)

        # Compute loss (only on masked patches)
        target = self.patchify(imgs)

        # Mean squared error on masked patches
        loss = (reconstructed - target) ** 2
        loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()

        return loss, reconstructed, mask

    def get_embeddings(self, imgs):
        """
        Extract embeddings from images

        Args:
            imgs: Input images

        Returns:
            embeddings: Image embeddings
        """
        # Encode without masking
        self.mask_ratio = 0  # Temporarily disable masking
        x = self.patchify(imgs)
        x = x + self.pos_embed

        for block in self.encoder:
            x = block(x)

        # Global average pooling
        embeddings = x.mean(dim=1)

        return embeddings


class TransformerBlock(nn.Module):
    """Transformer block for ViT"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


def train_mae_on_industrial_images(
    image_dir,
    output_dir='./mae_model',
    num_epochs=100,
    batch_size=256
):
    """
    Train MAE on industrial imagery

    Args:
        image_dir: Directory with unlabeled images
        output_dir: Where to save model
        num_epochs: Number of training epochs
        batch_size: Training batch size
    """
    # Initialize model
    model = MaskedAutoencoderViT(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12
    ).cuda()

    # Data augmentation (for self-supervised learning)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load dataset
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(image_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-4,
        betas=(0.9, 0.95),
        weight_decay=0.05
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for imgs, _ in dataloader:
            imgs = imgs.cuda()

            # Forward
            optimizer.zero_grad()
            loss, _, _ = model(imgs)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                f"{output_dir}/mae_epoch_{epoch}.pt"
            )

    print(f"Training complete. Model saved to {output_dir}")
