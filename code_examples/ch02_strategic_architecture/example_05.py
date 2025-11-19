# Code from Chapter 02
# Book: Embeddings at Scale

# User query: "red summer dress"
query_embedding = text_encoder.encode("red summer dress")
results = index.search(query_embedding)
# Returns products with text matching "red summer dress"
# Misses: visually similar dresses described differently
