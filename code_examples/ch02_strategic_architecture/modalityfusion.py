# Code from Chapter 02
# Book: Embeddings at Scale

class ModalityFusion:
    """Strategies for combining multi-modal embeddings"""

    @staticmethod
    def early_fusion(modality_embeddings, weights=None):
        """
        Combine embeddings before indexing
        Best for: Static multi-modal entities (products with images + text)
        """
        if weights is None:
            weights = [1.0 / len(modality_embeddings)] * len(modality_embeddings)

        # Weighted average
        fused = sum(w * emb for w, emb in zip(weights, modality_embeddings))
        # L2 normalize for cosine similarity
        return fused / torch.norm(fused)

    @staticmethod
    def late_fusion(query_emb, candidate_embs_by_modality, weights=None):
        """
        Combine similarity scores after retrieval
        Best for: Queries with variable modalities
        """
        if weights is None:
            weights = {modality: 1.0 / len(candidate_embs_by_modality)
                      for modality in candidate_embs_by_modality}

        # Calculate similarity per modality
        similarities = {}
        for modality, candidate_emb in candidate_embs_by_modality.items():
            if modality in query_emb:
                sim = cosine_similarity(query_emb[modality], candidate_emb)
                similarities[modality] = sim

        # Weighted combination
        final_score = sum(weights[mod] * sim for mod, sim in similarities.items())
        return final_score

    @staticmethod
    def attention_fusion(modality_embeddings):
        """
        Learn attention weights across modalities
        Best for: Complex scenarios where modality importance varies
        """
        # Stack embeddings
        stacked = torch.stack(modality_embeddings)  # (num_modalities, embedding_dim)

        # Attention mechanism
        attention_weights = torch.softmax(
            torch.matmul(stacked, stacked.transpose(0, 1)),
            dim=-1
        )

        # Weighted combination
        attended = torch.matmul(attention_weights, stacked)
        fused = attended.mean(dim=0)

        return fused / torch.norm(fused)

    @staticmethod
    def cross_modal_attention(text_emb, image_emb):
        """
        Cross-attention between modalities (e.g., CLIP-style)
        Best for: Learning aligned multi-modal representations
        """
        # Text attends to image
        text_to_image = torch.matmul(text_emb, image_emb.transpose(-2, -1))
        text_attended = torch.matmul(
            torch.softmax(text_to_image, dim=-1),
            image_emb
        )

        # Image attends to text
        image_to_text = torch.matmul(image_emb, text_emb.transpose(-2, -1))
        image_attended = torch.matmul(
            torch.softmax(image_to_text, dim=-1),
            text_emb
        )

        # Combine
        fused = torch.cat([text_attended, image_attended], dim=-1)
        return fused
