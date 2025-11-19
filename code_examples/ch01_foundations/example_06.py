# Code from Chapter 01
# Book: Embeddings at Scale

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

# Same word, different contexts, different embeddings
sentence1 = "The bank approved my loan application."
sentence2 = "I sat by the river bank watching the sunset."

embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

# "bank" has different representations based on context
# cosine_similarity(embedding1, embedding2) captures semantic similarity
