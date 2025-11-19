# Code from Chapter 02
# Book: Embeddings at Scale

class EmbeddingBiasMonitor:
    """Monitor and mitigate bias in embeddings"""

    def audit_for_bias(self, embeddings, metadata, protected_attributes):
        """Audit embeddings for bias across protected attributes"""
        bias_report = {
            'timestamp': datetime.now(),
            'embeddings_audited': len(embeddings),
            'protected_attributes': protected_attributes,
            'bias_detected': False,
            'bias_details': []
        }

        for attribute in protected_attributes:
            # Test for disparate impact
            impact_ratio = self.measure_disparate_impact(
                embeddings,
                metadata,
                attribute
            )

            if impact_ratio < 0.8 or impact_ratio > 1.25:  # 80% rule
                bias_report['bias_detected'] = True
                bias_report['bias_details'].append({
                    'attribute': attribute,
                    'impact_ratio': impact_ratio,
                    'severity': 'high' if impact_ratio < 0.7 or impact_ratio > 1.43 else 'medium'
                })

            # Test for embedding space separation
            separation = self.measure_embedding_separation(
                embeddings,
                metadata,
                attribute
            )

            if separation > 0.5:  # Threshold
                bias_report['bias_detected'] = True
                bias_report['bias_details'].append({
                    'attribute': attribute,
                    'separation_score': separation,
                    'issue': 'Protected attribute forms distinct cluster in embedding space'
                })

        return bias_report

    def debias_embeddings(self, embeddings, metadata, protected_attribute):
        """Remove bias from embeddings"""
        # Identify bias direction in embedding space
        groups = self.split_by_attribute(metadata, protected_attribute)

        group_centroids = {
            group: embeddings[indices].mean(axis=0)
            for group, indices in groups.items()
        }

        # Bias direction: vector from one centroid to another
        bias_direction = group_centroids['group_1'] - group_centroids['group_0']
        bias_direction = bias_direction / np.linalg.norm(bias_direction)

        # Project out bias direction from all embeddings
        debiased_embeddings = embeddings - np.outer(
            embeddings @ bias_direction,
            bias_direction
        )

        # Renormalize
        debiased_embeddings = debiased_embeddings / np.linalg.norm(
            debiased_embeddings,
            axis=1,
            keepdims=True
        )

        return debiased_embeddings
