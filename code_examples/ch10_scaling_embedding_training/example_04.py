import torch.nn as nn

# Code from Chapter 10
# Book: Embeddings at Scale

# Placeholder model for demonstration
model = nn.Sequential(
    nn.Linear(768, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Linear(512, 256)
)

model = model.half()  # Convert to FP16
# Keep BatchNorm in FP32
for module in model.modules():
    if isinstance(module, nn.BatchNorm1d):
        module.float()
