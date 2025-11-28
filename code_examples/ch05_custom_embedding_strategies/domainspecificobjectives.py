import torch
import torch.nn.functional as F

# Code from Chapter 04
# Book: Embeddings at Scale


class DomainSpecificObjectives:
    """
    Domain-specific training objectives beyond standard contrastive learning
    """

    def ranking_loss(self, query_emb, doc_embs, relevance_labels):
        """
        Ranking loss: Learn to order documents by relevance

        Use case: Search, recommendation
        """
        # LambdaRank or similar ranking loss
        # Optimizes for ranking metrics (NDCG, MRR)

        scores = torch.matmul(query_emb, doc_embs.T)

        # Pairwise ranking loss
        loss = 0
        for i in range(len(doc_embs)):
            for j in range(i + 1, len(doc_embs)):
                if relevance_labels[i] > relevance_labels[j]:
                    # Doc i should rank higher than doc j
                    loss += torch.clamp(1.0 - (scores[i] - scores[j]), min=0.0)

        return loss / (len(doc_embs) * (len(doc_embs) - 1) / 2)

    def attribute_preservation_loss(self, embedding, attributes):
        """
        Ensure embeddings preserve important attributes

        Use case: E-commerce (preserve category, brand, price tier)
        """
        # Train auxiliary classifiers to predict attributes from embeddings
        # If embeddings contain attribute information, classifiers succeed

        losses = []
        for attr_name, attr_value in attributes.items():
            attr_classifier = self.attribute_classifiers[attr_name]
            pred = attr_classifier(embedding)
            loss = F.cross_entropy(pred, attr_value)
            losses.append(loss)

        return sum(losses)

    def diversity_loss(self, embeddings):
        """
        Encourage embedding diversity (avoid collapse)

        Use case: Recommendation (avoid filter bubbles)
        """
        # Maximize pairwise distances
        pairwise_sim = torch.matmul(embeddings, embeddings.T)

        # Penalize high similarity between different items
        mask = ~torch.eye(len(embeddings), dtype=torch.bool)
        diversity_loss = pairwise_sim[mask].mean()

        return diversity_loss

    def cross_domain_alignment(self, source_emb, target_emb):
        """
        Align embeddings across domains

        Use case: Cross-lingual search, multi-modal search
        """
        # Minimize distance between equivalent items across domains
        alignment_loss = F.mse_loss(source_emb, target_emb)

        return alignment_loss
