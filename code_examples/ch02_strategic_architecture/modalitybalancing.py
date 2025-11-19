# Code from Chapter 02
# Book: Embeddings at Scale


class ModalityBalancing:
    """Handle entities with missing modalities"""

    def handle_missing_modalities(self, available_embs, all_modalities):
        """
        Strategy 1: Zero-padding (simple but can bias results)
        Strategy 2: Modality-specific indices (requires modality-aware retrieval)
        Strategy 3: Learned imputation (predict missing modalities)
        """
        # Strategy 3: Learned imputation
        missing_modalities = set(all_modalities) - set(available_embs.keys())

        for modality in missing_modalities:
            # Use cross-modal predictor trained to predict missing modality
            # from available modalities
            available_emb = list(available_embs.values())[0]  # Use any available
            imputed_emb = self.cross_modal_predictors[modality].predict(available_emb)
            available_embs[modality] = imputed_emb

        return available_embs
