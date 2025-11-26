"""Chunk metadata preservation for filtering and context."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json


@dataclass
class ChunkMetadata:
    """Comprehensive metadata for a text chunk."""
    # Source identification
    source_id: str
    source_type: str  # pdf, html, markdown, etc.
    source_url: Optional[str] = None

    # Position in source
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    section_hierarchy: List[str] = field(default_factory=list)
    start_char: int = 0
    end_char: int = 0

    # Content characteristics
    language: str = "en"
    content_type: str = "text"  # text, code, table, list
    word_count: int = 0

    # Temporal information
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    indexed_at: datetime = field(default_factory=datetime.now)

    # Custom attributes
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'source_id': self.source_id,
            'source_type': self.source_type,
            'source_url': self.source_url,
            'page_number': self.page_number,
            'section_title': self.section_title,
            'section_hierarchy': self.section_hierarchy,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'language': self.language,
            'content_type': self.content_type,
            'word_count': self.word_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'modified_at': self.modified_at.isoformat() if self.modified_at else None,
            'indexed_at': self.indexed_at.isoformat(),
            'custom': self.custom
        }


@dataclass
class EnrichedChunk:
    """A chunk with its text, embedding, and metadata."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if not self.chunk_id:
            # Generate ID from content hash
            self.chunk_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique ID based on content and source."""
        content = f"{self.metadata.source_id}:{self.metadata.start_char}:{self.text[:100]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class MetadataPreservingChunker:
    """
    Chunker that preserves and enriches metadata throughout the process.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        text: str,
        source_metadata: Dict[str, Any]
    ) -> List[EnrichedChunk]:
        """
        Chunk a document while preserving metadata.

        Args:
            text: Document text
            source_metadata: Metadata about the source document

        Returns:
            List of EnrichedChunk with metadata
        """
        from recursive_chunking import RecursiveChunker

        chunker = RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        raw_chunks = chunker.chunk(text)
        enriched_chunks = []

        current_pos = 0
        for chunk_text in raw_chunks:
            # Find position in original text
            start_pos = text.find(chunk_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(chunk_text)

            # Create metadata
            metadata = ChunkMetadata(
                source_id=source_metadata.get('id', 'unknown'),
                source_type=source_metadata.get('type', 'unknown'),
                source_url=source_metadata.get('url'),
                page_number=self._estimate_page(start_pos, source_metadata),
                section_title=source_metadata.get('current_section'),
                section_hierarchy=source_metadata.get('section_path', []),
                start_char=start_pos,
                end_char=end_pos,
                language=source_metadata.get('language', 'en'),
                content_type=self._detect_content_type(chunk_text),
                word_count=len(chunk_text.split()),
                created_at=source_metadata.get('created_at'),
                modified_at=source_metadata.get('modified_at'),
                custom=source_metadata.get('custom', {})
            )

            enriched_chunks.append(EnrichedChunk(
                chunk_id='',  # Will be auto-generated
                text=chunk_text,
                metadata=metadata
            ))

            current_pos = end_pos

        return enriched_chunks

    def _estimate_page(
        self,
        char_position: int,
        source_metadata: Dict
    ) -> Optional[int]:
        """Estimate page number from character position."""
        chars_per_page = source_metadata.get('chars_per_page', 3000)
        if chars_per_page:
            return (char_position // chars_per_page) + 1
        return source_metadata.get('page_number')

    def _detect_content_type(self, text: str) -> str:
        """Detect the type of content in a chunk."""
        # Check for code patterns
        code_indicators = ['def ', 'class ', 'function ', 'import ', '{', '}', '();']
        if any(indicator in text for indicator in code_indicators):
            return 'code'

        # Check for list patterns
        lines = text.split('\n')
        list_lines = sum(1 for line in lines if line.strip().startswith(('-', '*', '•', '1.')))
        if list_lines > len(lines) * 0.5:
            return 'list'

        # Check for table-like content
        if '|' in text and text.count('|') > 5:
            return 'table'

        return 'text'


def prepare_for_vector_db(
    chunks: List[EnrichedChunk]
) -> List[Dict]:
    """
    Prepare chunks for insertion into a vector database.

    Returns format compatible with most vector DBs.
    """
    records = []

    for chunk in chunks:
        record = {
            'id': chunk.chunk_id,
            'text': chunk.text,
            'embedding': chunk.embedding,
            'metadata': chunk.metadata.to_dict()
        }
        records.append(record)

    return records


def build_filter_query(
    source_type: Optional[str] = None,
    language: Optional[str] = None,
    content_type: Optional[str] = None,
    date_range: Optional[tuple] = None,
    custom_filters: Optional[Dict] = None
) -> Dict:
    """
    Build a metadata filter query for vector database search.

    Returns a filter dict compatible with common vector DBs.
    """
    filters = {}

    if source_type:
        filters['source_type'] = {'$eq': source_type}

    if language:
        filters['language'] = {'$eq': language}

    if content_type:
        filters['content_type'] = {'$eq': content_type}

    if date_range:
        start, end = date_range
        filters['indexed_at'] = {
            '$gte': start.isoformat(),
            '$lte': end.isoformat()
        }

    if custom_filters:
        for key, value in custom_filters.items():
            filters[f'custom.{key}'] = {'$eq': value}

    return {'$and': [filters]} if filters else {}


# Example usage
if __name__ == "__main__":
    sample_text = """
    # Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that focuses
    on building systems that learn from data.

    ## Supervised Learning

    Supervised learning uses labeled training data to train models.

    - Classification predicts categorical labels
    - Regression predicts continuous values

    ## Code Example

    ```python
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    ```
    """

    source_metadata = {
        'id': 'doc_001',
        'type': 'markdown',
        'url': 'https://example.com/ml-guide.md',
        'language': 'en',
        'created_at': datetime(2024, 1, 15),
        'modified_at': datetime(2024, 6, 1),
        'custom': {
            'author': 'Jane Smith',
            'category': 'tutorial',
            'difficulty': 'beginner'
        }
    }

    chunker = MetadataPreservingChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_document(sample_text, source_metadata)

    print(f"Created {len(chunks)} enriched chunks:\n")
    for chunk in chunks:
        print(f"--- Chunk {chunk.chunk_id} ---")
        print(f"Type: {chunk.metadata.content_type}")
        print(f"Words: {chunk.metadata.word_count}")
        print(f"Position: chars {chunk.metadata.start_char}-{chunk.metadata.end_char}")
        print(f"Text: {chunk.text[:100]}...")
        print()

    # Prepare for vector DB
    records = prepare_for_vector_db(chunks)
    print(f"\nPrepared {len(records)} records for vector DB")
    print(f"Sample record metadata: {json.dumps(records[0]['metadata'], indent=2, default=str)}")
