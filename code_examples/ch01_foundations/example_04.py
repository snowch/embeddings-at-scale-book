# Code from Chapter 01
# Book: Embeddings at Scale

from sklearn.decomposition import LatentDirichletAllocation

# Discover hidden topics
lda = LatentDirichletAllocation(n_components=50)
topic_distributions = lda.fit_transform(document_term_matrix)

# Documents with similar topic distributions are considered related
