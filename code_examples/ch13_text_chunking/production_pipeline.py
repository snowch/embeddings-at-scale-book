"""Production-ready text chunking pipeline."""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Supported document types."""
    PLAIN_TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    CODE = "code"


@dataclass
class ProcessedChunk:
    """A fully processed chunk ready for embedding."""
    chunk_id: str
    text: str
    document_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class PipelineConfig:
    """Configuration for the chunking pipeline."""
    # Chunk settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 50

    # Processing options
    preserve_structure: bool = True
    include_metadata: bool = True
    deduplicate: bool = True

    # Quality filters
    min_word_count: int = 10
    max_word_count: int = 2000
    remove_boilerplate: bool = True

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32


class ChunkingPipeline:
    """
    Production chunking pipeline with:
    - Document type detection
    - Appropriate chunker selection
    - Quality filtering
    - Metadata enrichment
    - Optional embedding generation
    - Deduplication
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._encoder = None

    def process_document(
        self,
        content: str,
        document_id: str,
        document_type: Optional[DocumentType] = None,
        source_metadata: Optional[Dict] = None
    ) -> List[ProcessedChunk]:
        """
        Process a single document through the full pipeline.

        Args:
            content: Document content
            document_id: Unique document identifier
            document_type: Type of document (auto-detected if not provided)
            source_metadata: Additional metadata about the source

        Returns:
            List of processed chunks ready for storage
        """
        logger.info(f"Processing document: {document_id}")

        # Step 1: Detect document type
        if document_type is None:
            document_type = self._detect_type(content)
        logger.debug(f"Document type: {document_type.value}")

        # Step 2: Select and run chunker
        raw_chunks = self._chunk_document(content, document_type)
        logger.info(f"Created {len(raw_chunks)} raw chunks")

        # Step 3: Filter chunks
        filtered_chunks = self._filter_chunks(raw_chunks)
        logger.info(f"After filtering: {len(filtered_chunks)} chunks")

        # Step 4: Deduplicate if enabled
        if self.config.deduplicate:
            filtered_chunks = self._deduplicate(filtered_chunks)
            logger.info(f"After deduplication: {len(filtered_chunks)} chunks")

        # Step 5: Enrich with metadata
        processed = self._enrich_chunks(
            filtered_chunks,
            document_id,
            document_type,
            source_metadata or {}
        )

        return processed

    def process_batch(
        self,
        documents: List[Dict[str, Any]],
        generate_embeddings: bool = True
    ) -> Iterator[ProcessedChunk]:
        """
        Process multiple documents, yielding chunks as they're ready.

        Args:
            documents: List of dicts with 'content', 'id', and optional metadata
            generate_embeddings: Whether to generate embeddings

        Yields:
            Processed chunks
        """
        all_chunks = []

        for doc in documents:
            chunks = self.process_document(
                content=doc['content'],
                document_id=doc['id'],
                document_type=doc.get('type'),
                source_metadata=doc.get('metadata')
            )
            all_chunks.extend(chunks)

        # Generate embeddings in batches
        if generate_embeddings:
            all_chunks = self._generate_embeddings_batch(all_chunks)

        yield from all_chunks

    def _detect_type(self, content: str) -> DocumentType:
        """Detect document type from content."""
        # Check for Markdown indicators
        if content.strip().startswith('#') or '```' in content:
            return DocumentType.MARKDOWN

        # Check for HTML
        if '<html' in content.lower() or '<body' in content.lower():
            return DocumentType.HTML

        # Check for code patterns
        code_patterns = ['def ', 'class ', 'function ', 'import ', 'const ', 'let ']
        code_matches = sum(1 for p in code_patterns if p in content)
        if code_matches >= 3:
            return DocumentType.CODE

        return DocumentType.PLAIN_TEXT

    def _chunk_document(
        self,
        content: str,
        doc_type: DocumentType
    ) -> List[str]:
        """Select appropriate chunker and process."""
        if doc_type == DocumentType.MARKDOWN:
            from markdown_chunking import MarkdownChunker
            chunker = MarkdownChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            md_chunks = chunker.chunk_markdown(content)
            return [c.text for c in md_chunks]

        elif doc_type == DocumentType.HTML:
            from html_chunking import HTMLChunker
            chunker = HTMLChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            html_chunks = chunker.chunk_html(content)
            return [c.text for c in html_chunks]

        elif doc_type == DocumentType.CODE:
            from code_chunking import CodeChunker
            chunker = CodeChunker(chunk_size=self.config.chunk_size)
            code_chunks = chunker.chunk_python(content)
            return [c.code for c in code_chunks]

        else:  # Plain text
            from recursive_chunking import RecursiveChunker
            chunker = RecursiveChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            return chunker.chunk(content)

    def _filter_chunks(self, chunks: List[str]) -> List[str]:
        """Apply quality filters to chunks."""
        filtered = []

        for chunk in chunks:
            # Skip empty or whitespace-only
            if not chunk.strip():
                continue

            word_count = len(chunk.split())

            # Check word count bounds
            if word_count < self.config.min_word_count:
                continue
            if word_count > self.config.max_word_count:
                continue

            # Remove boilerplate patterns
            if self.config.remove_boilerplate:
                if self._is_boilerplate(chunk):
                    continue

            filtered.append(chunk)

        return filtered

    def _is_boilerplate(self, text: str) -> bool:
        """Check if text is likely boilerplate."""
        boilerplate_patterns = [
            'all rights reserved',
            'terms of service',
            'privacy policy',
            'cookie policy',
            'copyright ©',
            'subscribe to our newsletter',
        ]
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in boilerplate_patterns)

    def _deduplicate(self, chunks: List[str]) -> List[str]:
        """Remove duplicate or near-duplicate chunks."""
        seen_hashes = set()
        unique = []

        for chunk in chunks:
            # Create hash of normalized text
            normalized = ' '.join(chunk.lower().split())
            chunk_hash = hashlib.md5(normalized.encode()).hexdigest()

            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                unique.append(chunk)

        return unique

    def _enrich_chunks(
        self,
        chunks: List[str],
        document_id: str,
        document_type: DocumentType,
        source_metadata: Dict
    ) -> List[ProcessedChunk]:
        """Add metadata and create ProcessedChunk objects."""
        processed = []

        for i, text in enumerate(chunks):
            chunk_id = hashlib.sha256(
                f"{document_id}:{i}:{text[:50]}".encode()
            ).hexdigest()[:16]

            metadata = {
                'document_type': document_type.value,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'word_count': len(text.split()),
                'char_count': len(text),
                'processed_at': datetime.now().isoformat(),
                **source_metadata
            }

            processed.append(ProcessedChunk(
                chunk_id=chunk_id,
                text=text,
                document_id=document_id,
                chunk_index=i,
                metadata=metadata
            ))

        return processed

    def _generate_embeddings_batch(
        self,
        chunks: List[ProcessedChunk]
    ) -> List[ProcessedChunk]:
        """Generate embeddings for chunks in batches."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.config.embedding_model)

        texts = [c.text for c in chunks]

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            embeddings = self._encoder.encode(batch)
            all_embeddings.extend(embeddings.tolist())

        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk.embedding = embedding

        return chunks


# Example usage
if __name__ == "__main__":
    sample_docs = [
        {
            'id': 'doc_001',
            'content': """
# Machine Learning Guide

Machine learning enables computers to learn from data.

## Types of Learning

- Supervised learning uses labeled data
- Unsupervised learning finds patterns
- Reinforcement learning learns through interaction

## Example Code

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
```
            """,
            'metadata': {'author': 'Jane Doe', 'category': 'tutorial'}
        },
        {
            'id': 'doc_002',
            'content': """
Neural networks are computing systems inspired by biological
neural networks. They consist of interconnected nodes called
neurons that process information. Deep learning extends neural
networks with multiple hidden layers.
            """,
            'metadata': {'author': 'John Smith', 'category': 'overview'}
        }
    ]

    # Create pipeline with custom config
    config = PipelineConfig(
        chunk_size=200,
        chunk_overlap=30,
        min_word_count=10,
        deduplicate=True
    )

    pipeline = ChunkingPipeline(config)

    print("Processing documents...\n")
    for chunk in pipeline.process_batch(sample_docs, generate_embeddings=False):
        print(f"Chunk {chunk.chunk_id}:")
        print(f"  Document: {chunk.document_id}")
        print(f"  Index: {chunk.chunk_index}")
        print(f"  Words: {chunk.metadata['word_count']}")
        print(f"  Text: {chunk.text[:100]}...")
        print()
