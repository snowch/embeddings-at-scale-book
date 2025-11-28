import torch
import torch.nn.functional as F

# Code from Chapter 06
# Book: Embeddings at Scale


class PrototypicalNetworkClassifier:
    """
    Prototypical Networks for few-shot learning

    Extension of one-shot learning:
    - Compute prototype (centroid) for each class from K examples
    - Classify by finding nearest prototype

    More robust than single example for noisy data.

    Reference: "Prototypical Networks for Few-shot Learning" (Snell et al., 2017)
    """

    def __init__(self, embedding_model):
        self.model = embedding_model
        self.prototypes = {}  # class_id -> prototype embedding

    def compute_prototypes(self, support_set):
        """
        Compute prototype for each class from support examples

        Args:
            support_set: Dict mapping class_id to list of examples
        """
        self.prototypes = {}

        with torch.no_grad():
            self.model.eval()

            for class_id, examples in support_set.items():
                # Stack examples
                if isinstance(examples, list):
                    examples = torch.stack(examples)

                # Compute embeddings
                embeddings = self.model.get_embedding(examples)

                # Prototype = mean of embeddings
                prototype = embeddings.mean(dim=0)
                self.prototypes[class_id] = prototype

    def predict(self, query):
        """Classify query by finding nearest prototype"""
        with torch.no_grad():
            self.model.eval()
            query_embedding = self.model.get_embedding(query)

            # Compute distances to all prototypes
            distances = {}
            for class_id, prototype in self.prototypes.items():
                dist = F.pairwise_distance(query_embedding, prototype.unsqueeze(0)).item()
                distances[class_id] = dist

            # Return class with minimum distance
            return min(distances.items(), key=lambda x: x[1])[0]
