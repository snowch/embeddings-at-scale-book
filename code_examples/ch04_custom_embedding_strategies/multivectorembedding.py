# Code from Chapter 04
# Book: Embeddings at Scale

class MultiVectorEmbedding:
    """
    Represent items with multiple specialized embeddings
    """

    def __init__(self):
        # Different encoders for different aspects
        self.semantic_encoder = SemanticEncoder(dim=512)     # Semantic meaning
        self.structural_encoder = StructuralEncoder(dim=256)  # Structured attributes
        self.behavioral_encoder = BehavioralEncoder(dim=256)  # User interaction patterns

    def encode(self, item, user_context=None):
        """
        Create multi-vector representation
        """
        vectors = {}

        # Semantic vector: text content
        vectors['semantic'] = self.semantic_encoder.encode(
            item['title'] + ' ' + item['description']
        )

        # Structural vector: categorical attributes
        vectors['structural'] = self.structural_encoder.encode({
            'category': item['category'],
            'brand': item['brand'],
            'price_tier': self.discretize_price(item['price']),
            'rating': item['avg_rating']
        })

        # Behavioral vector: how users interact with this item
        if 'user_interactions' in item:
            vectors['behavioral'] = self.behavioral_encoder.encode(
                item['user_interactions']
            )

        return vectors

    def search(self, query, user_context=None, objective='balanced'):
        """
        Search with different objectives
        """
        # Encode query with multiple vectors
        query_vectors = self.encode_query(query, user_context)

        # Different objectives use different vector combinations
        if objective == 'relevance':
            # Focus on semantic similarity
            weights = {'semantic': 1.0, 'structural': 0.2, 'behavioral': 0.1}
        elif objective == 'personalization':
            # Focus on behavioral patterns
            weights = {'semantic': 0.3, 'structural': 0.2, 'behavioral': 1.0}
        elif objective == 'balanced':
            # Balance all factors
            weights = {'semantic': 0.5, 'structural': 0.3, 'behavioral': 0.2}
        elif objective == 'exploration':
            # Emphasize diversity (structural differences)
            weights = {'semantic': 0.3, 'structural': 0.7, 'behavioral': 0.1}

        # Search each vector space
        results_by_vector = {}
        for vector_type, query_vec in query_vectors.items():
            results_by_vector[vector_type] = self.search_vector_space(
                query_vec,
                vector_space=vector_type
            )

        # Combine results with objective-specific weights
        final_results = self.weighted_fusion(results_by_vector, weights)

        return final_results
