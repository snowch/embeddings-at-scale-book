import time

# Code from Chapter 13
# Book: Embeddings at Scale

"""
Multi-Stage Retrieval Pipeline

Stages:
1. Coarse retrieval: Vector search (recall-focused)
2. Keyword filtering: Ensure key terms present
3. Reranking: Cross-encoder (precision-focused)
4. Diversity: Ensure variety in results
5. Final selection: Context optimization

Benefits:
- High recall (don't miss relevant docs)
- High precision (best docs ranked highest)
- Low latency (heavy computation on small set)
- Diversity (avoid redundant results)
"""

# Placeholder classes - see from.py for full implementation
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np


@dataclass
class Document:
    """Placeholder for Document."""

    doc_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    score: float = 0.0


@dataclass
class Query:
    """Placeholder for Query."""

    query_id: str
    text: str
    intent: Optional[str] = None
    entities: List[str] = None
    expanded_queries: List[str] = None
    filters: Dict[str, Any] = None


class MultiStageRetriever:
    """
    Multi-stage retrieval pipeline

    Stages:
    1. Vector retrieval (k=1000)
    2. Keyword filter (k=500)
    3. Reranking (k=20)
    4. Diversity filter (k=10)

    Performance:
    - Recall@1000: 98%+
    - Precision@10: 90%+
    - Latency: 200-400ms
    """

    def __init__(
        self,
        vector_index,
        embedding_model,
        reranker,
        stage1_k: int = 1000,
        stage2_k: int = 500,
        stage3_k: int = 20,
        stage4_k: int = 10,
    ):
        """
        Args:
            vector_index: Vector search index
            embedding_model: Embedding model
            reranker: Cross-encoder reranker
            stage1_k: Candidates after stage 1
            stage2_k: Candidates after stage 2
            stage3_k: Candidates after stage 3
            stage4_k: Final results
        """
        self.vector_index = vector_index
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k
        self.stage3_k = stage3_k
        self.stage4_k = stage4_k

        print("Initialized Multi-Stage Retriever")
        print(f"  Stage 1 (vector): {stage1_k}")
        print(f"  Stage 2 (keyword): {stage2_k}")
        print(f"  Stage 3 (rerank): {stage3_k}")
        print(f"  Stage 4 (diversity): {stage4_k}")

    def retrieve(self, query: Query) -> List[Document]:
        """
        Multi-stage retrieval

        Args:
            query: User query

        Returns:
            Final ranked documents
        """
        # Stage 1: Vector retrieval (recall)
        print(f"\nStage 1: Vector retrieval (k={self.stage1_k})")
        stage1_start = time.time()

        query_embedding = self._encode_query(query.text)
        stage1_docs = self.vector_index.search(query_embedding, k=self.stage1_k)

        stage1_latency = (time.time() - stage1_start) * 1000
        print(f"  Retrieved {len(stage1_docs)} documents in {stage1_latency:.1f}ms")

        # Stage 2: Keyword filtering (precision)
        print(f"\nStage 2: Keyword filtering (k={self.stage2_k})")
        stage2_start = time.time()

        stage2_docs = self._keyword_filter(query, stage1_docs, k=self.stage2_k)

        stage2_latency = (time.time() - stage2_start) * 1000
        print(f"  Filtered to {len(stage2_docs)} documents in {stage2_latency:.1f}ms")

        # Stage 3: Reranking (precision)
        print(f"\nStage 3: Reranking (k={self.stage3_k})")
        stage3_start = time.time()

        stage3_docs = self.reranker.rerank(query, stage2_docs, top_k=self.stage3_k)

        stage3_latency = (time.time() - stage3_start) * 1000
        print(f"  Reranked to {len(stage3_docs)} documents in {stage3_latency:.1f}ms")

        # Stage 4: Diversity filtering (quality)
        print(f"\nStage 4: Diversity filtering (k={self.stage4_k})")
        stage4_start = time.time()

        stage4_docs = self._diversity_filter(stage3_docs, k=self.stage4_k)

        stage4_latency = (time.time() - stage4_start) * 1000
        print(f"  Selected {len(stage4_docs)} diverse documents in {stage4_latency:.1f}ms")

        total_latency = (time.time() - stage1_start) * 1000
        print(f"\nTotal pipeline latency: {total_latency:.1f}ms")

        return stage4_docs

    def _encode_query(self, query_text: str) -> np.ndarray:
        """Encode query to embedding"""
        # Placeholder
        return np.random.randn(768).astype(np.float32)

    def _keyword_filter(self, query: Query, documents: List[Document], k: int) -> List[Document]:
        """
        Filter documents by keyword presence

        Strategy:
        - Extract key terms from query (entities, important words)
        - Score documents by presence of key terms
        - Keep top-k by combined score (vector + keyword)

        Args:
            query: User query
            documents: Retrieved documents
            k: Number to keep

        Returns:
            Filtered documents
        """
        # Extract query keywords
        query_terms = self._extract_keywords(query.text)

        # Score documents
        for doc in documents:
            keyword_score = self._keyword_overlap(query_terms, doc.content)
            # Combine with existing vector score
            doc.score = 0.7 * doc.score + 0.3 * keyword_score

        # Sort and return top-k
        documents.sort(key=lambda d: d.score, reverse=True)
        return documents[:k]

    def _extract_keywords(self, text: str) -> Set[str]:
        """
        Extract important keywords from text

        Production: Use TF-IDF, TextRank, or NER

        Args:
            text: Text to extract from

        Returns:
            Set of keywords
        """
        # Simple: Lowercase words, remove stopwords
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "how",
            "what",
            "where",
            "when",
            "why",
            "do",
            "does",
            "did",
        }

        words = text.lower().split()
        keywords = {w for w in words if w not in stopwords and len(w) > 3}

        return keywords

    def _keyword_overlap(self, query_terms: Set[str], document: str) -> float:
        """
        Compute keyword overlap score

        Args:
            query_terms: Query keywords
            document: Document text

        Returns:
            Overlap score (0-1)
        """
        doc_terms = set(document.lower().split())
        overlap = len(query_terms & doc_terms)
        return overlap / len(query_terms) if query_terms else 0.0

    def _diversity_filter(self, documents: List[Document], k: int) -> List[Document]:
        """
        Filter for diversity in results

        Problem: Top results often very similar (redundant)
        Solution: Maximal Marginal Relevance (MMR)
        - Balance relevance and diversity
        - Iteratively select: most relevant among those dissimilar to selected

        Args:
            documents: Reranked documents
            k: Number to select

        Returns:
            Diverse set of documents
        """
        if len(documents) <= k:
            return documents

        selected = [documents[0]]  # Start with most relevant
        remaining = documents[1:]

        while len(selected) < k and remaining:
            # Compute MMR for each remaining document
            best_mmr = -np.inf
            best_idx = 0

            for i, doc in enumerate(remaining):
                # Relevance
                relevance = doc.score

                # Diversity (dissimilarity to selected)
                max_similarity = max(
                    self._similarity(doc, selected_doc) for selected_doc in selected
                )
                diversity = 1 - max_similarity

                # MMR (lambda=0.5 balances relevance and diversity)
                mmr = 0.5 * relevance + 0.5 * diversity

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            # Select best MMR document
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)

        return selected

    def _similarity(self, doc1: Document, doc2: Document) -> float:
        """
        Compute similarity between documents

        Production: Use document embeddings

        Args:
            doc1: First document
            doc2: Second document

        Returns:
            Similarity score (0-1)
        """
        # Placeholder: Lexical overlap
        words1 = set(doc1.content.lower().split())
        words2 = set(doc2.content.lower().split())

        overlap = len(words1 & words2)
        union = len(words1 | words2)

        return overlap / union if union > 0 else 0.0


# Example: Multi-stage retrieval
def multi_stage_retrieval_example():
    """
    Demonstrate multi-stage retrieval

    Scenario: Technical Q&A over 10M documents
    """

    # Mock components
    class MockVectorIndex:
        def search(self, query_embedding, k=1000):
            docs = []
            for i in range(k):
                doc = Document(
                    doc_id=f"doc_{i}",
                    content=f"Document {i} about technical topic with relevant information and keywords.",
                    metadata={"title": f"Doc {i}"},
                    score=1.0 - (i * 0.001),
                )
                docs.append(doc)
            return docs

    class MockEmbeddingModel:
        pass

    class MockReranker:
        def rerank(self, query, documents, top_k=20):
            # Simulate reranking by slightly shuffling
            for doc in documents:
                doc.score += np.random.uniform(-0.1, 0.1)
            documents.sort(key=lambda d: d.score, reverse=True)
            return documents[:top_k]

    # Initialize multi-stage retriever
    retriever = MultiStageRetriever(
        vector_index=MockVectorIndex(),
        embedding_model=MockEmbeddingModel(),
        reranker=MockReranker(),
        stage1_k=1000,
        stage2_k=500,
        stage3_k=20,
        stage4_k=10,
    )

    # Query
    query = Query(query_id="q1", text="How do I configure authentication for the API using OAuth2?")

    # Retrieve
    results = retriever.retrieve(query)

    print(f"\n{'=' * 60}")
    print(f"Final Results ({len(results)} documents):")
    print(f"{'=' * 60}")
    for i, doc in enumerate(results):
        print(f"{i + 1}. {doc.doc_id} (score: {doc.score:.4f})")


# Uncomment to run:
# multi_stage_retrieval_example()
