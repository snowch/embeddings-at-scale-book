# Code from Chapter 04
# Book: Embeddings at Scale

class AsymmetricSimilarity:
    """
    Handle asymmetric similarity (query → document differs from document → query)
    """

    def __init__(self, embedding_dim=512):
        self.query_encoder = QueryEncoder(embedding_dim)
        self.document_encoder = DocumentEncoder(embedding_dim)

    def encode_query(self, query_text):
        """
        Encode query with query-specific model
        Queries are typically short, focused, and incomplete
        """
        return self.query_encoder.encode(query_text)

    def encode_document(self, document_text):
        """
        Encode document with document-specific model
        Documents are longer, complete, and information-rich
        """
        return self.document_encoder.encode(document_text)

    def similarity(self, query_embedding, document_embedding):
        """
        Asymmetric similarity: query → document
        """
        # In asymmetric setup, similarity is directional
        # "running shoes" → "Nike Air Zoom Pegasus..." (HIGH similarity)
        # "Nike Air Zoom Pegasus..." → "running shoes" (LOWER similarity - too specific)

        return cosine_similarity(query_embedding, document_embedding)


# Use cases requiring asymmetric similarity:
asymmetric_use_cases = [
    {
        'domain': 'Question Answering',
        'query': 'Short question',
        'target': 'Long passage with answer',
        'asymmetry': 'Question seeks answer; answer does not seek question'
    },
    {
        'domain': 'Web Search',
        'query': '2-5 keywords',
        'target': 'Full web page content',
        'asymmetry': 'Query is intent; document is content'
    },
    {
        'domain': 'Image Search',
        'query': 'Text description',
        'target': 'Image',
        'asymmetry': 'Cross-modal: text → image different from image → text'
    },
    {
        'domain': 'Recommendation',
        'query': 'User behavior history',
        'target': 'Product catalog',
        'asymmetry': 'User history implies preferences; products have features'
    }
]
