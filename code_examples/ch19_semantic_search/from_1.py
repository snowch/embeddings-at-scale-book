# Code from Chapter 14
# Book: Embeddings at Scale

"""
Scientific Literature Search System

Architecture:
1. Document encoder: SPECTER, SciBERT (domain-specific BERT)
2. Citation graph: Incorporate citation network
3. Entity linking: Resolve entities (authors, chemicals, genes)
4. Multi-field search: Title, abstract, full text, citations
5. Temporal ranking: Prioritize recent + highly-cited

Training data:
- S2ORC: 81M+ papers with citations
- PubMed: 35M+ biomedical papers
- arXiv: 2M+ preprints

Applications:
- Literature review (find related work)
- Drug discovery (find chemical interactions)
- Researcher discovery (find experts)
- Patent search (prior art search)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class ScientificPaper:
    """
    Scientific paper with metadata

    Attributes:
        paper_id: Unique identifier
        title: Paper title
        abstract: Abstract text
        authors: List of author names
        year: Publication year
        venue: Publication venue (journal/conference)
        citations: List of cited paper IDs
        cited_by: List of citing paper IDs
        entities: Extracted entities (chemicals, genes, etc.)
        embedding: Cached embedding
    """

    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    year: int
    venue: Optional[str] = None
    citations: List[str] = None
    cited_by: List[str] = None
    entities: Dict[str, List[str]] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.citations is None:
            self.citations = []
        if self.cited_by is None:
            self.cited_by = []
        if self.entities is None:
            self.entities = {}


class ScientificSearchEngine:
    """
    Semantic search for scientific literature

    Features:
    - Semantic search: Find papers by concept, not keywords
    - Citation-aware ranking: Boost highly-cited papers
    - Co-citation analysis: Find papers cited together
    - Author search: Find papers by author
    - Entity search: Find papers mentioning specific entities

    Advanced features:
    - Citation graph embeddings: Node2Vec on citation graph
    - Multi-hop search: Find papers citing papers that cite X
    - Temporal ranking: Boost recent papers
    - Cross-lingual: Search in English, find in any language
    """

    def __init__(self, embedding_dim: int = 768):
        """
        Args:
            embedding_dim: Embedding dimension (768 for SPECTER/SciBERT)
        """
        self.embedding_dim = embedding_dim

        # Paper index: paper_id -> ScientificPaper
        self.papers: Dict[str, ScientificPaper] = {}

        # Embedding index
        self.paper_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

        # Citation graph: paper_id -> {cited_papers, citing_papers}
        self.citation_graph: Dict[str, Set[str]] = {}

        # Entity index: entity -> {paper_ids}
        self.entity_index: Dict[str, Set[str]] = {}

        # Author index: author -> {paper_ids}
        self.author_index: Dict[str, Set[str]] = {}

        print("Initialized Scientific Search Engine")
        print(f"  Embedding dimension: {embedding_dim}")

    def encode_paper(self, paper: ScientificPaper) -> np.ndarray:
        """
        Encode paper to embedding

        Uses SPECTER or SciBERT:
        - Encode title + abstract
        - Optionally incorporate citation context

        Args:
            paper: Scientific paper

        Returns:
            Paper embedding (embedding_dim,)
        """
        # Combine title and abstract

        # In production: Use SPECTER or SciBERT
        # For demo: Random embedding
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def index_papers(self, papers: List[ScientificPaper]):
        """
        Index scientific papers

        Builds:
        1. Embedding index (for semantic search)
        2. Citation graph (for citation-aware ranking)
        3. Entity index (for entity search)
        4. Author index (for author search)

        Args:
            papers: Papers to index
        """
        print(f"Indexing {len(papers)} papers...")

        for paper in papers:
            # Encode paper
            embedding = self.encode_paper(paper)
            paper.embedding = embedding

            # Add to paper index
            self.papers[paper.paper_id] = paper
            self.paper_ids.append(paper.paper_id)

            # Add to embedding index
            if self.embeddings is None:
                self.embeddings = embedding.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])

            # Build citation graph
            if paper.paper_id not in self.citation_graph:
                self.citation_graph[paper.paper_id] = set()

            for cited_id in paper.citations:
                self.citation_graph[paper.paper_id].add(cited_id)

                # Add reverse citation
                if cited_id not in self.citation_graph:
                    self.citation_graph[cited_id] = set()

            # Build entity index
            for entity_type, entities in paper.entities.items():
                for entity in entities:
                    entity_key = f"{entity_type}:{entity}"
                    if entity_key not in self.entity_index:
                        self.entity_index[entity_key] = set()
                    self.entity_index[entity_key].add(paper.paper_id)

            # Build author index
            for author in paper.authors:
                if author not in self.author_index:
                    self.author_index[author] = set()
                self.author_index[author].add(paper.paper_id)

        print(f"✓ Indexed {len(papers)} papers")
        print(f"  Papers: {len(self.papers)}")
        print(f"  Citations: {sum(len(v) for v in self.citation_graph.values())}")
        print(f"  Entities: {len(self.entity_index)}")
        print(f"  Authors: {len(self.author_index)}")

    def search(
        self,
        query: str,
        top_k: int = 20,
        min_year: Optional[int] = None,
        author_filter: Optional[str] = None,
        boost_citations: bool = True,
    ) -> List[Tuple[ScientificPaper, float]]:
        """
        Search for papers

        Args:
            query: Search query (natural language)
            top_k: Number of results
            min_year: Minimum publication year (optional)
            author_filter: Filter by author (optional)
            boost_citations: Boost highly-cited papers

        Returns:
            List of (paper, score) tuples
        """
        # Encode query (same as paper encoding)
        # In production: Use SPECTER query encoder
        query_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Compute semantic similarity
        scores = np.dot(self.embeddings, query_embedding)

        # Apply filters and boost citations
        final_scores = []

        for i, paper_id in enumerate(self.paper_ids):
            paper = self.papers[paper_id]
            score = scores[i]

            # Year filter
            if min_year and paper.year < min_year:
                continue

            # Author filter
            if author_filter and author_filter not in paper.authors:
                continue

            # Citation boost
            if boost_citations:
                citation_count = len(paper.cited_by)
                # Log-scale boost (highly-cited papers get moderate boost)
                citation_boost = np.log1p(citation_count) / 10.0
                score = score + citation_boost

            final_scores.append((paper, score))

        # Sort by score
        final_scores.sort(key=lambda x: x[1], reverse=True)

        return final_scores[:top_k]

    def find_related_by_citations(
        self, paper_id: str, top_k: int = 10
    ) -> List[Tuple[ScientificPaper, float]]:
        """
        Find related papers based on citation patterns

        Strategies:
        1. Co-citations: Papers that cite the same papers
        2. Bibliographic coupling: Papers cited by the same papers
        3. Direct citations: Papers that cite or are cited by this paper

        Args:
            paper_id: Source paper ID
            top_k: Number of results

        Returns:
            List of (paper, score) tuples
        """
        if paper_id not in self.papers:
            return []

        source_paper = self.papers[paper_id]

        # Get papers cited by source
        source_citations = set(source_paper.citations)

        # Compute co-citation scores
        cocitation_scores = {}

        for other_id in self.paper_ids:
            if other_id == paper_id:
                continue

            other_paper = self.papers[other_id]
            other_citations = set(other_paper.citations)

            # Co-citation: Papers that cite the same papers
            overlap = source_citations & other_citations

            if overlap:
                # Jaccard similarity
                union = source_citations | other_citations
                score = len(overlap) / len(union) if union else 0
                cocitation_scores[other_id] = score

        # Sort by score
        ranked = sorted(cocitation_scores.items(), key=lambda x: x[1], reverse=True)

        results = [(self.papers[pid], score) for pid, score in ranked[:top_k]]

        return results

    def search_by_entity(
        self, entity_type: str, entity_value: str, top_k: int = 20
    ) -> List[ScientificPaper]:
        """
        Search papers by entity (chemical, gene, etc.)

        Args:
            entity_type: Type of entity (e.g., 'chemical', 'gene')
            entity_value: Entity value
            top_k: Number of results

        Returns:
            List of papers mentioning this entity
        """
        entity_key = f"{entity_type}:{entity_value}"

        if entity_key not in self.entity_index:
            return []

        paper_ids = self.entity_index[entity_key]

        # Sort by year (most recent first)
        papers = [self.papers[pid] for pid in paper_ids if pid in self.papers]
        papers.sort(key=lambda p: p.year, reverse=True)

        return papers[:top_k]


# Example: Scientific literature search
def scientific_search_example():
    """
    Semantic search over scientific literature

    Use cases:
    - Literature review (find related work)
    - Drug discovery (find papers on specific chemicals)
    - Researcher discovery (find papers by author)
    - Citation analysis (find citation patterns)

    Scale: 35M+ papers (PubMed scale)
    """

    # Create sample papers
    papers = [
        ScientificPaper(
            paper_id="paper_1",
            title="Deep Learning for Protein Structure Prediction",
            abstract="We present a novel deep learning approach for predicting protein structures from amino acid sequences using transformers.",
            authors=["Alice Smith", "Bob Johnson"],
            year=2023,
            venue="Nature",
            citations=["paper_4"],
            entities={"protein": ["AlphaFold"], "method": ["transformer"]},
            cited_by=["paper_2", "paper_3"],
        ),
        ScientificPaper(
            paper_id="paper_2",
            title="Improved Protein Folding with Attention Mechanisms",
            abstract="Building on recent work, we improve protein folding predictions using multi-head attention and residual connections.",
            authors=["Charlie Brown"],
            year=2024,
            venue="Science",
            citations=["paper_1", "paper_4"],
            entities={"protein": ["AlphaFold"], "method": ["attention"]},
        ),
        ScientificPaper(
            paper_id="paper_3",
            title="Applications of AI in Drug Discovery",
            abstract="We survey recent applications of artificial intelligence in drug discovery, including protein structure prediction and molecular docking.",
            authors=["Diana Prince", "Alice Smith"],
            year=2024,
            venue="Nature Reviews Drug Discovery",
            citations=["paper_1"],
            entities={"application": ["drug discovery"]},
        ),
        ScientificPaper(
            paper_id="paper_4",
            title="Transformers for Sequence Modeling",
            abstract="A general framework for using transformers to model biological sequences including DNA, RNA, and proteins.",
            authors=["Eve Martinez"],
            year=2022,
            venue="NeurIPS",
            citations=[],
            entities={"method": ["transformer"]},
            cited_by=["paper_1", "paper_2"],
        ),
    ]

    # Initialize search engine
    engine = ScientificSearchEngine(embedding_dim=768)

    # Index papers
    engine.index_papers(papers)

    # Search: Find papers on protein structure prediction
    print("\n=== Query: 'protein structure prediction' ===")
    results = engine.search("protein structure prediction", top_k=3, boost_citations=True)

    for paper, score in results:
        print(f"\n{paper.title}")
        print(f"  Authors: {', '.join(paper.authors)}")
        print(f"  Year: {paper.year}, Venue: {paper.venue}")
        print(f"  Citations: {len(paper.cited_by)}")
        print(f"  Score: {score:.3f}")

    # Find related papers by co-citation
    print("\n\n=== Papers related to 'paper_1' (by co-citation) ===")
    related = engine.find_related_by_citations("paper_1", top_k=3)

    for paper, score in related:
        print(f"\n{paper.title}")
        print(f"  Co-citation score: {score:.3f}")

    # Search by entity
    print("\n\n=== Papers mentioning 'AlphaFold' ===")
    entity_results = engine.search_by_entity("protein", "AlphaFold", top_k=5)

    for paper in entity_results:
        print(f"\n{paper.title} ({paper.year})")


# Uncomment to run:
# scientific_search_example()
