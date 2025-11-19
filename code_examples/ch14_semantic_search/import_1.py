# Code from Chapter 14
# Book: Embeddings at Scale

"""
Enterprise Knowledge Graph with Embeddings

Architecture:
1. Entity encoder: Learn entity embeddings
2. Relation encoder: Learn relation embeddings
3. Link prediction: Predict (entity1, relation, entity2) triples
4. Entity resolution: Merge duplicate entities
5. Graph-aware search: Search considering relationships

Embedding models:
- TransE: Entities as points, relations as translations
- DistMult: Bilinear scoring function
- ComplEx: Complex-valued embeddings
- RotatE: Rotations in complex space

Applications:
- Customer 360 (link customers across systems)
- Product recommendations (similar products by graph)
- Fraud detection (unusual relationship patterns)
- Knowledge discovery (predict new relationships)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Entity:
    """
    Knowledge graph entity

    Attributes:
        entity_id: Unique identifier
        entity_type: Type (customer, product, document, etc.)
        attributes: Entity attributes
        embedding: Learned entity embedding
    """

    entity_id: str
    entity_type: str
    attributes: Dict = None
    embedding: Optional[np.ndarray] = None


@dataclass
class Relation:
    """
    Knowledge graph relation (edge)

    Attributes:
        subject: Subject entity ID
        predicate: Relation type
        object: Object entity ID
        confidence: Confidence score (0-1)
    """

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0


class KnowledgeGraphEmbedding:
    """
    Knowledge graph with learned embeddings

    Embedding model: TransE
    - Represents entities as vectors
    - Represents relations as translations
    - Score function: score(h, r, t) = ||h + r - t||
    - Training objective: Minimize score for true triples, maximize for false

    Applications:
    - Link prediction: Given (entity1, relation, ?), predict entity2
    - Relation prediction: Given (entity1, ?, entity2), predict relation
    - Entity search: Find entities similar to query entity
    - Graph completion: Predict missing edges
    """

    def __init__(self, embedding_dim: int = 128, margin: float = 1.0):
        """
        Args:
            embedding_dim: Dimension of entity/relation embeddings
            margin: Margin for ranking loss
        """
        self.embedding_dim = embedding_dim
        self.margin = margin

        # Entity and relation stores
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

        # Embeddings
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.relation_embeddings: Dict[str, np.ndarray] = {}

        # Graph structure: entity_id -> {relations from this entity}
        self.outgoing_edges: Dict[str, List[Relation]] = {}
        self.incoming_edges: Dict[str, List[Relation]] = {}

        print("Initialized Knowledge Graph Embeddings")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Margin: {margin}")

    def add_entity(self, entity: Entity):
        """
        Add entity to knowledge graph

        Args:
            entity: Entity to add
        """
        self.entities[entity.entity_id] = entity

        # Initialize random embedding
        if entity.entity_id not in self.entity_embeddings:
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            self.entity_embeddings[entity.entity_id] = embedding
            entity.embedding = embedding

    def add_relation(self, relation: Relation):
        """
        Add relation (edge) to knowledge graph

        Args:
            relation: Relation to add
        """
        self.relations.append(relation)

        # Initialize relation embedding if new
        if relation.predicate not in self.relation_embeddings:
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            self.relation_embeddings[relation.predicate] = embedding

        # Update graph structure
        if relation.subject not in self.outgoing_edges:
            self.outgoing_edges[relation.subject] = []
        self.outgoing_edges[relation.subject].append(relation)

        if relation.object not in self.incoming_edges:
            self.incoming_edges[relation.object] = []
        self.incoming_edges[relation.object].append(relation)

    def score_triple(self, subject_id: str, predicate: str, object_id: str) -> float:
        """
        Score a triple using TransE scoring function

        TransE: score(h, r, t) = -||h + r - t||

        Args:
            subject_id: Subject entity ID
            predicate: Relation type
            object_id: Object entity ID

        Returns:
            Score (higher = more likely)
        """
        if subject_id not in self.entity_embeddings:
            return -float("inf")
        if object_id not in self.entity_embeddings:
            return -float("inf")
        if predicate not in self.relation_embeddings:
            return -float("inf")

        h = self.entity_embeddings[subject_id]
        r = self.relation_embeddings[predicate]
        t = self.entity_embeddings[object_id]

        # TransE score: -||h + r - t||
        score = -np.linalg.norm(h + r - t)

        return score

    def predict_tail(
        self, subject_id: str, predicate: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Predict tail entity given head and relation

        Query: (subject, predicate, ?)

        Args:
            subject_id: Subject entity ID
            predicate: Relation type
            top_k: Number of predictions

        Returns:
            List of (entity_id, score) tuples
        """
        scores = []

        for entity_id in self.entity_embeddings:
            if entity_id == subject_id:
                continue

            score = self.score_triple(subject_id, predicate, entity_id)
            scores.append((entity_id, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def predict_relation(
        self, subject_id: str, object_id: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Predict relation given subject and object

        Query: (subject, ?, object)

        Args:
            subject_id: Subject entity ID
            object_id: Object entity ID
            top_k: Number of predictions

        Returns:
            List of (predicate, score) tuples
        """
        scores = []

        for predicate in self.relation_embeddings:
            score = self.score_triple(subject_id, predicate, object_id)
            scores.append((predicate, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def find_similar_entities(
        self, entity_id: str, top_k: int = 10, entity_type_filter: Optional[str] = None
    ) -> List[Tuple[Entity, float]]:
        """
        Find entities similar to given entity

        Uses entity embeddings (cosine similarity)

        Args:
            entity_id: Source entity ID
            top_k: Number of results
            entity_type_filter: Filter by entity type (optional)

        Returns:
            List of (entity, similarity) tuples
        """
        if entity_id not in self.entity_embeddings:
            return []

        source_emb = self.entity_embeddings[entity_id]

        similarities = []

        for other_id, other_emb in self.entity_embeddings.items():
            if other_id == entity_id:
                continue

            # Type filter
            if entity_type_filter:
                other_entity = self.entities.get(other_id)
                if not other_entity or other_entity.entity_type != entity_type_filter:
                    continue

            # Cosine similarity
            similarity = np.dot(source_emb, other_emb)
            similarities.append((other_id, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get entities
        results = [
            (self.entities[eid], sim) for eid, sim in similarities[:top_k] if eid in self.entities
        ]

        return results

    def get_neighbors(
        self, entity_id: str, relation_type: Optional[str] = None
    ) -> List[Tuple[str, str, str]]:
        """
        Get neighboring entities (1-hop neighborhood)

        Args:
            entity_id: Source entity ID
            relation_type: Filter by relation type (optional)

        Returns:
            List of (relation, direction, neighbor_id) tuples
        """
        neighbors = []

        # Outgoing edges
        if entity_id in self.outgoing_edges:
            for rel in self.outgoing_edges[entity_id]:
                if relation_type is None or rel.predicate == relation_type:
                    neighbors.append((rel.predicate, "outgoing", rel.object))

        # Incoming edges
        if entity_id in self.incoming_edges:
            for rel in self.incoming_edges[entity_id]:
                if relation_type is None or rel.predicate == relation_type:
                    neighbors.append((rel.predicate, "incoming", rel.subject))

        return neighbors


# Example: Enterprise knowledge graph
def knowledge_graph_example():
    """
    Enterprise knowledge graph with embeddings

    Use cases:
    - Customer 360 (link customers across systems)
    - Product recommendations (similar products)
    - Link prediction (discover relationships)

    Scale: 1B+ entities, 10B+ relations
    """

    # Initialize knowledge graph
    kg = KnowledgeGraphEmbedding(embedding_dim=128)

    # Add entities
    customers = [
        Entity("customer_1", "customer", {"name": "Alice"}),
        Entity("customer_2", "customer", {"name": "Bob"}),
        Entity("customer_3", "customer", {"name": "Charlie"}),
    ]

    products = [
        Entity("product_1", "product", {"name": "Laptop"}),
        Entity("product_2", "product", {"name": "Mouse"}),
        Entity("product_3", "product", {"name": "Keyboard"}),
    ]

    for entity in customers + products:
        kg.add_entity(entity)

    # Add relations
    relations = [
        Relation("customer_1", "purchased", "product_1"),
        Relation("customer_1", "purchased", "product_2"),
        Relation("customer_2", "purchased", "product_1"),
        Relation("customer_2", "purchased", "product_3"),
        Relation("customer_3", "purchased", "product_2"),
        Relation("product_2", "accessory_for", "product_1"),
        Relation("product_3", "accessory_for", "product_1"),
    ]

    for relation in relations:
        kg.add_relation(relation)

    print("\n=== Knowledge Graph Statistics ===")
    print(f"Entities: {len(kg.entities)}")
    print(f"Relations: {len(kg.relations)}")
    print(f"Relation types: {len(kg.relation_embeddings)}")

    # Link prediction: What might customer_3 purchase?
    print("\n=== Link Prediction: What might customer_3 purchase? ===")
    predictions = kg.predict_tail("customer_3", "purchased", top_k=3)

    for entity_id, score in predictions:
        entity = kg.entities.get(entity_id)
        if entity and entity.entity_type == "product":
            print(f"{entity.attributes.get('name')}: score = {score:.3f}")

    # Find similar customers
    print("\n=== Similar Customers to customer_1 ===")
    similar = kg.find_similar_entities("customer_1", top_k=2, entity_type_filter="customer")

    for entity, similarity in similar:
        print(f"{entity.attributes.get('name')}: similarity = {similarity:.3f}")

    # Get neighbors
    print("\n=== Neighbors of product_1 (Laptop) ===")
    neighbors = kg.get_neighbors("product_1")

    for relation, direction, neighbor_id in neighbors:
        neighbor = kg.entities.get(neighbor_id)
        if neighbor:
            print(f"{relation} ({direction}): {neighbor.attributes.get('name')}")


# Uncomment to run:
# knowledge_graph_example()
