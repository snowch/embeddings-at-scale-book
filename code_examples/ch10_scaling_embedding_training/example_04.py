# Code from Chapter 10
# Book: Embeddings at Scale

model = model.half()  # Convert to FP16
# Keep BatchNorm in FP32
for module in model.modules():
    if isinstance(module, nn.BatchNorm1d):
        module.float()
