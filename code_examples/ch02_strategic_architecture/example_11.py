# Code from Chapter 02
# Book: Embeddings at Scale

# Image query
image_emb = encoder.encode_image(uploaded_image)

# Initial results
initial_results = index.search_multimodal({'image': image_emb}, k=100)

# Text refinement
text_emb = encoder.encode_text("in blue")

# Combined query
refined_results = index.search_multimodal(
    {'image': image_emb, 'text': text_emb},
    modality_weights={'image': 0.7, 'text': 0.3},  # Image is primary
    k=20
)
