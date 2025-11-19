# Code from Chapter 01
# Book: Embeddings at Scale

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

# Example data for demonstration
document_term_matrix = np.random.randint(0, 10, size=(100, 1000))  # 100 documents, 1000 terms

# Discover hidden topics
lda = LatentDirichletAllocation(n_components=50)
topic_distributions = lda.fit_transform(document_term_matrix)

# Documents with similar topic distributions are considered related
