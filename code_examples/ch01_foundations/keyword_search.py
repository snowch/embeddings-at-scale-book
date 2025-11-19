# Code from Chapter 01
# Book: Embeddings at Scale

# The original sin of information retrieval
def keyword_search(query, documents):
    query_terms = query.lower().split()
    results = []
    for doc in documents:
        doc_terms = doc.lower().split()
        score = len(set(query_terms) & set(doc_terms))
        if score > 0:
            results.append((doc, score))
    return sorted(results, key=lambda x: x[1], reverse=True)

# Problems:
# - "laptop" doesn't match "notebook computer"
# - "running shoes" doesn't match "athletic footwear"
# - "cheap flights" doesn't match "affordable airfare"
