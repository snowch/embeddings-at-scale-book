# Code from Chapter 09
# Book: Embeddings at Scale

# Allow clients to specify model version explicitly
query_embedding = embedding_service.get_embedding(
    query="...",
    model_version="v1.2.3"  # Pin to specific version
)
