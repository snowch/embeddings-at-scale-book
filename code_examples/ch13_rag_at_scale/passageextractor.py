# Code from Chapter 13
# Book: Embeddings at Scale

"""
Context Window Optimization Strategies

Techniques:
1. Passage extraction: Extract most relevant sentences
2. Deduplication: Remove redundant information
3. Summarization: Compress long documents
4. Hierarchical assembly: Progressively add detail
5. Token counting: Precise tracking to maximize usage
"""

import re
from typing import List, Tuple
import numpy as np

class PassageExtractor:
    """
    Extract most relevant passages from documents

    Approach:
    - Sentence-level relevance scoring
    - Extract top sentences per document
    - Maintain narrative flow (keep consecutive sentences)

    Benefits:
    - Preserves key information
    - Removes boilerplate/fluff
    - Reduces tokens 50-70%
    """

    def __init__(self, max_sentences_per_doc: int = 5):
        """
        Args:
            max_sentences_per_doc: Maximum sentences to extract per document
        """
        self.max_sentences_per_doc = max_sentences_per_doc
        print(f"Initialized Passage Extractor (max {max_sentences_per_doc} sentences/doc)")

    def extract(
        self,
        query: str,
        document: str
    ) -> str:
        """
        Extract most relevant passages

        Args:
            query: User query
            document: Full document text

        Returns:
            Extracted passages
        """
        # Split into sentences
        sentences = self._split_sentences(document)

        if len(sentences) <= self.max_sentences_per_doc:
            return document

        # Score sentences by relevance
        scores = []
        for sent in sentences:
            score = self._score_sentence(query, sent)
            scores.append(score)

        # Get top sentences
        top_indices = np.argsort(scores)[-self.max_sentences_per_doc:]
        top_indices = sorted(top_indices)  # Maintain order

        # Extract top sentences
        extracted = [sentences[i] for i in top_indices]

        return " ".join(extracted)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (production: use nltk or spacy)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _score_sentence(self, query: str, sentence: str) -> float:
        """
        Score sentence relevance to query

        Simple approach: Lexical overlap
        Production: Use sentence embeddings

        Args:
            query: Query text
            sentence: Sentence text

        Returns:
            Relevance score
        """
        query_words = set(query.lower().split())
        sent_words = set(sentence.lower().split())

        overlap = len(query_words & sent_words)
        return overlap

class ContextDeduplicator:
    """
    Remove redundant information across documents

    Problem: Multiple documents often contain overlapping information
    - Wastes tokens
    - Confuses LLM with repetition

    Solution: Detect and remove near-duplicate passages

    Approach:
    - Sentence-level similarity (embeddings or MinHash)
    - Remove sentences similar to previously included ones
    - Keep first occurrence
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Args:
            similarity_threshold: Threshold for considering sentences duplicate
        """
        self.similarity_threshold = similarity_threshold
        print(f"Initialized Context Deduplicator (threshold: {similarity_threshold})")

    def deduplicate(self, documents: List[str]) -> List[str]:
        """
        Remove redundant content across documents

        Args:
            documents: List of document texts

        Returns:
            Deduplicated documents
        """
        seen_sentences = set()
        deduplicated_docs = []

        for doc in documents:
            sentences = self._split_sentences(doc)
            unique_sentences = []

            for sent in sentences:
                # Normalize sentence for comparison
                normalized = sent.lower().strip()

                # Check if similar sentence already seen
                if not self._is_duplicate(normalized, seen_sentences):
                    unique_sentences.append(sent)
                    seen_sentences.add(normalized)

            if unique_sentences:
                deduplicated_docs.append(" ".join(unique_sentences))

        return deduplicated_docs

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _is_duplicate(self, sentence: str, seen: set) -> bool:
        """
        Check if sentence is duplicate of any seen sentence

        Simple approach: Exact match
        Production: Use embedding similarity or MinHash

        Args:
            sentence: Sentence to check
            seen: Set of seen sentences

        Returns:
            True if duplicate
        """
        return sentence in seen

class HierarchicalContextAssembler:
    """
    Hierarchical context assembly

    Strategy: Progressive detail
    1. Start with document titles/summaries (high-level)
    2. Add key passages (medium detail)
    3. Add supporting details (low detail)
    4. Stop when context window full

    Benefits:
    - Ensures high-level information included
    - Gracefully handles variable context sizes
    - LLM sees overview before details
    """

    def __init__(self, max_tokens: int = 4096):
        """
        Args:
            max_tokens: Maximum context tokens
        """
        self.max_tokens = max_tokens
        self.passage_extractor = PassageExtractor(max_sentences_per_doc=3)
        self.deduplicator = ContextDeduplicator()

        print(f"Initialized Hierarchical Context Assembler")
        print(f"  Max tokens: {max_tokens:,}")

    def assemble(
        self,
        query: str,
        documents: List[Document]
    ) -> str:
        """
        Assemble context hierarchically

        Levels:
        1. Titles and metadata
        2. Key passages
        3. Full content (if space available)

        Args:
            query: User query
            documents: Ranked documents

        Returns:
            Assembled context string
        """
        context_parts = []
        current_tokens = 0

        # Level 1: Titles and metadata
        for i, doc in enumerate(documents):
            title = doc.metadata.get('title', f'Document {i+1}')
            source = doc.metadata.get('source', 'unknown')
            header = f"[{i+1}] {title} (Source: {source})"

            header_tokens = len(header) // 4
            if current_tokens + header_tokens < self.max_tokens:
                context_parts.append(header)
                current_tokens += header_tokens

        # Level 2: Key passages
        for i, doc in enumerate(documents):
            # Extract relevant passages
            passages = self.passage_extractor.extract(query, doc.content)
            passages_tokens = len(passages) // 4

            if current_tokens + passages_tokens < self.max_tokens * 0.8:  # Leave 20% buffer
                context_parts.append(f"\nKey points from Document {i+1}:")
                context_parts.append(passages)
                current_tokens += passages_tokens
            else:
                break  # Context window nearly full

        # Level 3: Additional details (if space)
        # Skipped in this basic implementation

        # Deduplicate
        deduplicated = self.deduplicator.deduplicate(context_parts)
        context = "\n".join(deduplicated)

        print(f"Assembled hierarchical context: {current_tokens:,} tokens from {len(documents)} docs")

        return context

# Example: Context window optimization
def context_optimization_example():
    """
    Demonstrate context window optimization

    Scenario:
    - Retrieved 10 documents, each 1000 tokens
    - Total: 10K tokens
    - Context limit: 4K tokens
    - Need to reduce by 60% while preserving key info
    """

    # Mock documents
    documents = []
    for i in range(10):
        doc = Document(
            doc_id=f"doc_{i}",
            content=f"""This is document {i}. It contains important information about the topic.
            Key point 1: The system uses advanced algorithms.
            Key point 2: Performance is optimized for scale.
            Key point 3: Security is built-in at every layer.
            Additional detail: The architecture follows best practices.
            More detail: Integration is seamless with existing systems.
            Even more detail: The team has extensive experience.
            Background: Development started in 2023.
            Context: The project aims to solve real-world problems.
            Summary: This represents a significant advancement.""",
            metadata={
                'title': f'Technical Report {i}',
                'source': 'documentation'
            }
        )
        documents.append(doc)

    query = "What are the key features of the system?"

    # Standard assembly (truncation)
    print("=== Standard Assembly (Truncation) ===")
    context_manager = ContextManager(max_context_tokens=4096, max_tokens_per_doc=400)
    standard_context = context_manager.assemble_context(
        Query(query_id="q1", text=query),
        documents
    )
    print(f"Context length: {len(standard_context)} chars (~{len(standard_context)//4} tokens)")

    # Optimized assembly (hierarchical + extraction)
    print("\n=== Optimized Assembly (Hierarchical) ===")
    hierarchical_assembler = HierarchicalContextAssembler(max_tokens=4096)
    optimized_context = hierarchical_assembler.assemble(query, documents)
    print(f"Context length: {len(optimized_context)} chars (~{len(optimized_context)//4} tokens)")

    print(f"\nSample optimized context:")
    print(optimized_context[:500] + "...")

# Uncomment to run:
# context_optimization_example()
