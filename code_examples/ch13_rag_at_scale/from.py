# Code from Chapter 13
# Book: Embeddings at Scale

"""
Enterprise RAG System Architecture

Components:
1. Query Processor: Intent classification, entity extraction, query expansion
2. Retrieval Engine: Vector search across billion-document corpus
3. Reranker: Cross-encoder model for precise relevance scoring
4. Context Manager: Optimize context window utilization
5. Generator: LLM inference with structured prompting
6. Validator: Fact-checking and hallucination detection

Performance targets:
- Latency: p95 < 2 seconds end-to-end
- Accuracy: 95%+ on domain-specific questions
- Throughput: 10K queries/second
- Availability: 99.9%
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Document:
    """
    Document in knowledge base

    Attributes:
        doc_id: Unique identifier
        content: Document text
        metadata: Title, source, author, date, etc.
        embedding: Vector representation
        score: Relevance score (set during retrieval/reranking)
    """
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    score: float = 0.0

@dataclass
class Query:
    """
    User query with processing metadata

    Attributes:
        query_id: Unique identifier
        text: Original query text
        intent: Classified intent (factual, how-to, comparison, etc.)
        entities: Extracted named entities
        expanded_queries: Query variations for retrieval
        filters: Metadata filters (date range, source, etc.)
    """
    query_id: str
    text: str
    intent: Optional[str] = None
    entities: List[str] = None
    expanded_queries: List[str] = None
    filters: Dict[str, Any] = None

@dataclass
class RAGResponse:
    """
    Complete RAG system response

    Attributes:
        query_id: Links to original query
        answer: Generated answer text
        sources: Documents used as sources
        confidence: Model confidence score
        latency_ms: End-to-end latency
        metadata: Retrieval/generation metadata
    """
    query_id: str
    answer: str
    sources: List[Document]
    confidence: float
    latency_ms: float
    metadata: Dict[str, Any]

class QueryProcessor:
    """
    Query understanding and expansion

    Responsibilities:
    - Intent classification: Identify query type
    - Entity extraction: Extract key entities
    - Query expansion: Generate variations (synonyms, reformulations)
    - Filter extraction: Parse metadata filters

    Techniques:
    - NER models for entity extraction
    - Classification models for intent
    - Query expansion via embeddings (find similar queries)
    - Rule-based parsing for filters
    """

    def __init__(self):
        """Initialize query processor"""
        # In production: Load NER model, intent classifier, etc.
        self.supported_intents = [
            'factual',      # "What is X?"
            'how-to',       # "How do I do X?"
            'comparison',   # "What's the difference between X and Y?"
            'explanation',  # "Why does X happen?"
            'list'          # "What are the types of X?"
        ]

        print("Initialized Query Processor")

    def process(self, query_text: str) -> Query:
        """
        Process raw query into structured representation

        Args:
            query_text: Raw query string

        Returns:
            Query object with intent, entities, expansions
        """
        query_id = f"query_{int(time.time() * 1000)}"

        # Intent classification
        intent = self._classify_intent(query_text)

        # Entity extraction
        entities = self._extract_entities(query_text)

        # Query expansion
        expanded = self._expand_query(query_text, intent)

        # Filter extraction
        filters = self._extract_filters(query_text)

        return Query(
            query_id=query_id,
            text=query_text,
            intent=intent,
            entities=entities,
            expanded_queries=expanded,
            filters=filters
        )

    def _classify_intent(self, query_text: str) -> str:
        """
        Classify query intent

        Simple rule-based classification (production: use trained model)

        Args:
            query_text: Query text

        Returns:
            Intent label
        """
        text_lower = query_text.lower()

        if any(word in text_lower for word in ['what is', 'define', 'meaning of']):
            return 'factual'
        elif any(word in text_lower for word in ['how to', 'how do i', 'how can i']):
            return 'how-to'
        elif any(word in text_lower for word in ['difference between', 'compare', 'vs']):
            return 'comparison'
        elif any(word in text_lower for word in ['why', 'reason', 'cause']):
            return 'explanation'
        elif any(word in text_lower for word in ['list', 'what are', 'types of']):
            return 'list'
        else:
            return 'factual'

    def _extract_entities(self, query_text: str) -> List[str]:
        """
        Extract named entities

        Production: Use spaCy, Flair, or custom NER model

        Args:
            query_text: Query text

        Returns:
            List of entities
        """
        # Placeholder: Simple word extraction
        # In production: Use proper NER
        entities = []

        # Extract capitalized words as potential entities
        words = query_text.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 1:
                entities.append(word)

        return entities

    def _expand_query(self, query_text: str, intent: str) -> List[str]:
        """
        Expand query with variations

        Strategies:
        - Synonym replacement
        - Intent-specific reformulations
        - Related questions

        Args:
            query_text: Original query
            intent: Query intent

        Returns:
            List of expanded queries
        """
        expansions = [query_text]  # Include original

        # Intent-specific expansions
        if intent == 'how-to':
            # Add "steps" variation
            expansions.append(query_text.replace('how to', 'steps to'))
            expansions.append(query_text.replace('how do I', 'how can I'))
        elif intent == 'factual':
            # Add "explain" variation
            expansions.append(query_text.replace('what is', 'explain'))
            expansions.append(query_text.replace('what is', 'definition of'))

        return expansions

    def _extract_filters(self, query_text: str) -> Dict[str, Any]:
        """
        Extract metadata filters from query

        Examples:
        - "papers from 2023" → date_year: 2023
        - "articles by John Smith" → author: "John Smith"

        Args:
            query_text: Query text

        Returns:
            Dictionary of filters
        """
        filters = {}

        # Year extraction
        for year in range(2000, 2030):
            if str(year) in query_text:
                filters['date_year'] = year
                break

        # Author extraction (simple pattern)
        if 'by' in query_text.lower():
            parts = query_text.lower().split('by')
            if len(parts) > 1:
                author = parts[1].strip().split()[0:2]  # Get up to 2 words
                if author:
                    filters['author'] = ' '.join(author)

        return filters

class RetrievalEngine:
    """
    Vector-based retrieval from document corpus

    Architecture:
    - Embedding model for query encoding
    - Vector index (HNSW, IVF) for billion-doc search
    - Metadata filtering
    - Top-k retrieval (k=100-1000)

    Performance:
    - Latency: p95 < 100ms for 1B documents
    - Recall@1000: 95%+
    - Throughput: 10K queries/second
    """

    def __init__(
        self,
        vector_index,
        embedding_model,
        default_k: int = 100
    ):
        """
        Args:
            vector_index: Vector search index (HNSW, Faiss, etc.)
            embedding_model: Model for encoding queries
            default_k: Default number of results to retrieve
        """
        self.vector_index = vector_index
        self.embedding_model = embedding_model
        self.default_k = default_k

        print("Initialized Retrieval Engine")
        print(f"  Default k: {default_k}")

    def retrieve(
        self,
        query: Query,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve top-k relevant documents

        Strategy:
        1. Encode query (and expansions) to vectors
        2. Search vector index
        3. Apply metadata filters
        4. Merge results across query expansions
        5. Return top-k by score

        Args:
            query: Processed query
            k: Number of documents to retrieve

        Returns:
            List of documents with relevance scores
        """
        if k is None:
            k = self.default_k

        start_time = time.time()

        # Encode query
        query_embedding = self._encode_query(query.text)

        # Search vector index
        results = self.vector_index.search(query_embedding, k=k)

        # Apply filters
        if query.filters:
            results = self._apply_filters(results, query.filters)

        # If using query expansions, merge results
        if query.expanded_queries:
            expansion_results = []
            for expanded_query in query.expanded_queries[1:]:  # Skip first (original)
                expanded_embedding = self._encode_query(expanded_query)
                expanded_results = self.vector_index.search(expanded_embedding, k=k)
                expansion_results.extend(expanded_results)

            # Merge and deduplicate
            results = self._merge_results([results, expansion_results], k=k)

        latency_ms = (time.time() - start_time) * 1000

        print(f"Retrieved {len(results)} documents in {latency_ms:.1f}ms")

        return results

    def _encode_query(self, query_text: str) -> np.ndarray:
        """
        Encode query to embedding vector

        Args:
            query_text: Query string

        Returns:
            Embedding vector
        """
        # In production: Use actual embedding model
        # For now: Return random embedding
        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def _apply_filters(
        self,
        documents: List[Document],
        filters: Dict[str, Any]
    ) -> List[Document]:
        """
        Filter documents by metadata

        Args:
            documents: Documents to filter
            filters: Metadata filters

        Returns:
            Filtered documents
        """
        filtered = []

        for doc in documents:
            # Check all filters
            matches = True
            for key, value in filters.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    matches = False
                    break

            if matches:
                filtered.append(doc)

        return filtered

    def _merge_results(
        self,
        result_lists: List[List[Document]],
        k: int
    ) -> List[Document]:
        """
        Merge results from multiple searches

        Strategy: Reciprocal Rank Fusion (RRF)
        - Score = sum(1 / (rank + 60)) across all result lists
        - Robust to score scale differences

        Args:
            result_lists: Multiple result lists
            k: Number of results to return

        Returns:
            Merged and deduplicated results
        """
        # Compute RRF scores
        rrf_scores = defaultdict(float)
        doc_map = {}

        for results in result_lists:
            for rank, doc in enumerate(results):
                rrf_scores[doc.doc_id] += 1.0 / (rank + 60)
                doc_map[doc.doc_id] = doc

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Return top-k
        merged = []
        for doc_id in sorted_ids[:k]:
            doc = doc_map[doc_id]
            doc.score = rrf_scores[doc_id]
            merged.append(doc)

        return merged

class Reranker:
    """
    Precise reranking of retrieved documents

    Approach:
    - Cross-encoder model (BERT-based)
    - Processes query + document pairs
    - More accurate than vector similarity (but slower)
    - Reduces top-1000 to top-10-20 for context window

    Performance:
    - Latency: 50-200ms for 100 documents
    - Accuracy: 10-20% improvement over retrieval alone
    """

    def __init__(self, model_name: str = "cross-encoder"):
        """
        Args:
            model_name: Cross-encoder model identifier
        """
        self.model_name = model_name
        # In production: Load actual cross-encoder model
        print(f"Initialized Reranker with model: {model_name}")

    def rerank(
        self,
        query: Query,
        documents: List[Document],
        top_k: int = 10
    ) -> List[Document]:
        """
        Rerank documents by relevance

        Args:
            query: User query
            documents: Retrieved documents to rerank
            top_k: Number of top documents to return

        Returns:
            Reranked documents
        """
        start_time = time.time()

        # Compute relevance scores
        for doc in documents:
            doc.score = self._compute_relevance(query.text, doc.content)

        # Sort by score
        documents.sort(key=lambda d: d.score, reverse=True)

        # Return top-k
        reranked = documents[:top_k]

        latency_ms = (time.time() - start_time) * 1000
        print(f"Reranked {len(documents)} → {len(reranked)} documents in {latency_ms:.1f}ms")

        return reranked

    def _compute_relevance(self, query: str, document: str) -> float:
        """
        Compute query-document relevance score

        In production: Use cross-encoder model

        Args:
            query: Query text
            document: Document text

        Returns:
            Relevance score (0-1)
        """
        # Placeholder: Simple lexical overlap
        # In production: Use trained cross-encoder
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())

        overlap = len(query_words & doc_words)
        union = len(query_words | doc_words)

        return overlap / union if union > 0 else 0.0

class ContextManager:
    """
    Optimize context window utilization

    Challenges:
    - LLM context limits (4K-128K tokens)
    - Multiple documents may exceed limit
    - Need to maximize information density

    Strategies:
    - Truncation: Keep first N tokens per document
    - Extraction: Extract most relevant sentences
    - Summarization: Summarize long documents
    - Compression: Remove redundancy across documents
    """

    def __init__(
        self,
        max_context_tokens: int = 4096,
        max_tokens_per_doc: int = 500
    ):
        """
        Args:
            max_context_tokens: Maximum context window size
            max_tokens_per_doc: Maximum tokens per document
        """
        self.max_context_tokens = max_context_tokens
        self.max_tokens_per_doc = max_tokens_per_doc

        print("Initialized Context Manager")
        print(f"  Max context tokens: {max_context_tokens:,}")
        print(f"  Max tokens per doc: {max_tokens_per_doc}")

    def assemble_context(
        self,
        query: Query,
        documents: List[Document]
    ) -> str:
        """
        Assemble context from documents

        Strategy:
        1. Truncate each document to max_tokens_per_doc
        2. Concatenate documents in rank order
        3. Truncate total to max_context_tokens
        4. Add document citations

        Args:
            query: User query
            documents: Reranked documents

        Returns:
            Assembled context string
        """
        context_parts = []
        total_tokens = 0

        for i, doc in enumerate(documents):
            # Estimate tokens (rough: 1 token ≈ 4 characters)
            doc_tokens = len(doc.content) // 4

            # Truncate document if too long
            if doc_tokens > self.max_tokens_per_doc:
                # Keep first N characters
                char_limit = self.max_tokens_per_doc * 4
                content = doc.content[:char_limit] + "..."
                doc_tokens = self.max_tokens_per_doc
            else:
                content = doc.content

            # Check if adding this document would exceed limit
            if total_tokens + doc_tokens > self.max_context_tokens:
                # Stop adding documents
                break

            # Add document with citation
            doc_text = f"[Document {i+1}]\n{content}\n"
            context_parts.append(doc_text)
            total_tokens += doc_tokens

        context = "\n".join(context_parts)

        print(f"Assembled context: {total_tokens:,} tokens from {len(context_parts)} documents")

        return context

class RAGSystem:
    """
    Complete RAG system orchestration

    Architecture:
    - Query Processor → Retrieval → Reranking → Context Assembly → Generation

    Performance:
    - End-to-end latency: p95 < 2s
    - Accuracy: 95%+ on domain questions
    - Throughput: 10K queries/sec (with parallelization)
    """

    def __init__(
        self,
        vector_index,
        embedding_model,
        llm,
        retrieval_k: int = 100,
        rerank_k: int = 10
    ):
        """
        Args:
            vector_index: Vector search index
            embedding_model: Embedding model for queries
            llm: Large language model for generation
            retrieval_k: Number of documents to retrieve
            rerank_k: Number of documents to rerank to
        """
        self.query_processor = QueryProcessor()
        self.retrieval_engine = RetrievalEngine(vector_index, embedding_model, default_k=retrieval_k)
        self.reranker = Reranker()
        self.context_manager = ContextManager()
        self.llm = llm
        self.rerank_k = rerank_k

        print("Initialized RAG System")
        print(f"  Retrieval k: {retrieval_k}")
        print(f"  Rerank k: {rerank_k}")

    def answer(self, query_text: str) -> RAGResponse:
        """
        Answer user query using RAG

        Pipeline:
        1. Process query (intent, entities, expansion)
        2. Retrieve top-k documents (k=100)
        3. Rerank to top-n (n=10)
        4. Assemble context
        5. Generate answer
        6. Validate response

        Args:
            query_text: User query

        Returns:
            RAGResponse with answer and sources
        """
        start_time = time.time()

        # 1. Process query
        query = self.query_processor.process(query_text)
        print(f"\nProcessing query: {query_text}")
        print(f"  Intent: {query.intent}")
        print(f"  Entities: {query.entities}")

        # 2. Retrieve documents
        retrieved_docs = self.retrieval_engine.retrieve(query)

        # 3. Rerank documents
        reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=self.rerank_k)

        # 4. Assemble context
        context = self.context_manager.assemble_context(query, reranked_docs)

        # 5. Generate answer
        answer, confidence = self._generate_answer(query, context)

        # 6. Validate (placeholder)
        # In production: Check for hallucinations, verify citations

        latency_ms = (time.time() - start_time) * 1000

        return RAGResponse(
            query_id=query.query_id,
            answer=answer,
            sources=reranked_docs,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={
                'intent': query.intent,
                'num_retrieved': len(retrieved_docs),
                'num_reranked': len(reranked_docs),
                'context_tokens': len(context) // 4
            }
        )

    def _generate_answer(
        self,
        query: Query,
        context: str
    ) -> Tuple[str, float]:
        """
        Generate answer using LLM

        Args:
            query: User query
            context: Assembled context

        Returns:
            (answer, confidence)
        """
        # Construct prompt
        prompt = f"""Answer the following question based on the provided context.
Cite sources using [Document N] notation.

Question: {query.text}

Context:
{context}

Answer:"""

        # Generate (placeholder)
        # In production: Call actual LLM
        answer = f"Based on the provided documents, {query.text.lower().replace('?', '').replace('what is', 'refers to')}. [Document 1]"
        confidence = 0.85

        return answer, confidence

# Example: Enterprise RAG system
def rag_system_example():
    """
    Demonstrate enterprise RAG system

    Scenario: Technical documentation Q&A
    - 1M documents (product docs, API refs, tutorials)
    - User asks technical questions
    - System retrieves relevant docs and generates answers
    """

    # Mock components
    class MockVectorIndex:
        def search(self, query_embedding, k=100):
            # Return mock documents
            docs = []
            for i in range(k):
                doc = Document(
                    doc_id=f"doc_{i}",
                    content=f"This is document {i} containing relevant technical information about the topic.",
                    metadata={
                        'title': f'Document {i}',
                        'source': 'documentation',
                        'date_year': 2024
                    },
                    score=1.0 - (i * 0.01)  # Decreasing scores
                )
                docs.append(doc)
            return docs

    class MockEmbeddingModel:
        def encode(self, text):
            return np.random.randn(768).astype(np.float32)

    class MockLLM:
        def generate(self, prompt):
            return "Generated answer based on context..."

    # Initialize RAG system
    vector_index = MockVectorIndex()
    embedding_model = MockEmbeddingModel()
    llm = MockLLM()

    rag_system = RAGSystem(
        vector_index=vector_index,
        embedding_model=embedding_model,
        llm=llm,
        retrieval_k=100,
        rerank_k=10
    )

    # Query
    query_text = "How do I configure authentication for the API?"

    # Get answer
    response = rag_system.answer(query_text)

    # Display results
    print(f"\n{'='*60}")
    print(f"Query: {query_text}")
    print(f"{'='*60}")
    print(f"\nAnswer: {response.answer}")
    print(f"\nConfidence: {response.confidence:.2f}")
    print(f"Latency: {response.latency_ms:.1f}ms")
    print("\nSources:")
    for i, doc in enumerate(response.sources[:3]):
        print(f"  [{i+1}] {doc.metadata.get('title', doc.doc_id)} (score: {doc.score:.3f})")
    print("\nMetadata:")
    for key, value in response.metadata.items():
        print(f"  {key}: {value}")

# Uncomment to run:
# rag_system_example()
