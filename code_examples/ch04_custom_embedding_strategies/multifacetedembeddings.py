# Code from Chapter 04
# Book: Embeddings at Scale

class MultiFacetedEmbeddings:
    """
    Represent multiple facets of similarity in separate embedding spaces
    """

    def __init__(self):
        # E-commerce example: products similar in different ways
        self.visual_encoder = VisualEncoder()  # Visual appearance
        self.functional_encoder = FunctionalEncoder()  # Use case/function
        self.attribute_encoder = AttributeEncoder()  # Specific attributes (brand, price, etc.)

    def encode_product(self, product):
        """
        Encode product with multiple faceted embeddings
        """
        return {
            'visual': self.visual_encoder.encode(product.images),
            'functional': self.functional_encoder.encode(product.description),
            'attributes': self.attribute_encoder.encode({
                'brand': product.brand,
                'price_tier': self.discretize_price(product.price),
                'category': product.category
            })
        }

    def multi_faceted_search(self, query, facet_weights=None):
        """
        Search using multiple facets with different weights
        """
        if facet_weights is None:
            facet_weights = {'visual': 0.4, 'functional': 0.4, 'attributes': 0.2}

        # Encode query (may not have all facets)
        query_embs = self.encode_query(query)

        # Search each facet independently
        results_by_facet = {}
        for facet in query_embs:
            results_by_facet[facet] = self.search_facet(
                query_embs[facet],
                facet_index=getattr(self, f'{facet}_index')
            )

        # Combine results with weighted fusion
        final_results = self.fuse_facet_results(
            results_by_facet,
            weights=facet_weights
        )

        return final_results
