# Code from Chapter 01
# Book: Embeddings at Scale

from gensim.models import Word2Vec

# Train embeddings that capture semantic relationships
model = Word2Vec(sentences, vector_size=300, window=5, min_count=5)

# Mathematical operations capture meaning:
# king - man + woman ≈ queen (with sufficient training data)
# Paris - France + Italy ≈ Rome

king = model.wv['king']
man = model.wv['man']
woman = model.wv['woman']
result = king - man + woman
# model.wv.most_similar([result]) often returns 'queen'
# Note: This famous example requires large corpora (billions of tokens)
