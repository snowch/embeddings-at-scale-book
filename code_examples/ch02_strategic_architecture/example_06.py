# Code from Chapter 02
# Book: Embeddings at Scale

# User query: "red summer dress" + uploads inspiration image
query_text_emb = text_encoder.encode("red summer dress")
query_image_emb = image_encoder.encode(inspiration_image)

# Unified multi-modal query
query_emb = combine_embeddings(query_text_emb, query_image_emb)

results = index.search(query_emb)
# Returns products matching both semantic text AND visual style
# Result quality dramatically higher
