from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity

# Code from Chapter 02
# Book: Embeddings at Scale


class EmbeddingExplainability:
    """Explain embedding-based decisions"""

    def explain_similarity(self, query_embedding, result_embedding, metadata):
        """Explain why two items are similar"""
        # Decompose similarity by components
        similarity_components = self.decompose_similarity(query_embedding, result_embedding)

        # Identify which features contributed most
        top_features = self.identify_top_features(query_embedding, result_embedding, metadata)

        # Generate human-readable explanation
        explanation = {
            "overall_similarity": cosine_similarity(query_embedding, result_embedding),
            "similarity_breakdown": similarity_components,
            "key_matching_features": top_features,
            "explanation_text": self.generate_explanation_text(top_features),
        }

        return explanation

    def generate_explanation_text(self, top_features):
        """Generate human-readable explanation"""
        explanations = []

        for feature in top_features[:3]:  # Top 3 features
            explanations.append(
                f"{feature['name']}: {feature['contribution']:.1%} contribution "
                f"(query: {feature['query_value']}, match: {feature['match_value']})"
            )

        return " | ".join(explanations)

    def audit_decision(self, decision_id, embedding_query, results, chosen_result):
        """Create audit trail for embedding-based decision"""
        audit_record = {
            "decision_id": decision_id,
            "timestamp": datetime.now(),
            "query_embedding": embedding_query.tolist(),
            "all_results": [
                {"id": r["id"], "similarity": r["similarity"], "embedding": r["embedding"].tolist()}
                for r in results
            ],
            "chosen_result": chosen_result,
            "explanation": self.explain_similarity(
                embedding_query, chosen_result["embedding"], chosen_result["metadata"]
            ),
        }

        # Store audit record
        self.audit_log.append(audit_record)

        return audit_record
