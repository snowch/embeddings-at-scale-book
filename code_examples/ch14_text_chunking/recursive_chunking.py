"""Recursive/hierarchical text chunking implementation."""

from typing import List, Optional


class RecursiveChunker:
    """
    Recursively split text using a hierarchy of separators.

    Tries separators in order, falling back to the next when chunks
    are still too large.
    """

    def __init__(
        self, chunk_size: int = 500, chunk_overlap: int = 50, separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",  # Paragraph breaks (strongest)
            "\n",  # Line breaks
            ". ",  # Sentences
            ", ",  # Clauses
            " ",  # Words (last resort)
        ]

    def chunk(self, text: str) -> List[str]:
        """Split text recursively using separator hierarchy."""
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text, trying separators in order."""

        # Base case: text is small enough
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # No more separators to try
        if not separators:
            # Force split at chunk_size
            return self._force_split(text)

        current_sep = separators[0]
        remaining_seps = separators[1:]

        # Split on current separator
        splits = text.split(current_sep)

        # If separator not found, try next one
        if len(splits) == 1:
            return self._recursive_split(text, remaining_seps)

        # Merge splits into chunks
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split) + len(current_sep)

            if current_length + split_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = current_sep.join(current_chunk)
                chunks.append(chunk_text)

                # Handle overlap
                overlap_text = self._get_overlap(current_chunk, current_sep)
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text) if overlap_text else 0

            current_chunk.append(split)
            current_length += split_length

        # Process remaining text
        if current_chunk:
            remaining = current_sep.join(current_chunk)
            # Recursively split if still too large
            if len(remaining) > self.chunk_size:
                chunks.extend(self._recursive_split(remaining, remaining_seps))
            elif remaining.strip():
                chunks.append(remaining)

        return chunks

    def _get_overlap(self, parts: List[str], sep: str) -> str:
        """Get overlap text from the end of current chunk."""
        if not self.chunk_overlap or not parts:
            return ""

        overlap_parts = []
        overlap_length = 0

        for part in reversed(parts):
            if overlap_length + len(part) > self.chunk_overlap:
                break
            overlap_parts.insert(0, part)
            overlap_length += len(part) + len(sep)

        return sep.join(overlap_parts)

    def _force_split(self, text: str) -> List[str]:
        """Force split text at chunk_size boundaries."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - self.chunk_overlap

        return chunks


# Specialized chunkers for different content types
class MarkdownChunker(RecursiveChunker):
    """Chunker optimized for Markdown documents."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n## ",  # H2 headers
                "\n### ",  # H3 headers
                "\n#### ",  # H4 headers
                "\n\n",  # Paragraphs
                "\n",  # Lines
                ". ",  # Sentences
                " ",  # Words
            ],
        )


class CodeChunker(RecursiveChunker):
    """Chunker optimized for source code."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\nclass ",  # Class definitions
                "\ndef ",  # Function definitions
                "\n\n",  # Blank lines
                "\n",  # Lines
                " ",  # Words
            ],
        )


# Example usage
if __name__ == "__main__":
    sample_text = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence. It focuses on building
systems that learn from data.

## Supervised Learning

Supervised learning uses labeled data. The model learns to map inputs to outputs.

### Classification

Classification predicts categorical labels. Examples include spam detection and
image recognition.

### Regression

Regression predicts continuous values. Examples include price prediction and
temperature forecasting.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data. Clustering and
dimensionality reduction are common techniques.
    """

    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk(sample_text)

    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ({len(chunk)} chars) ---")
        print(chunk.strip()[:150] + "..." if len(chunk) > 150 else chunk.strip())
        print()

    # Markdown-aware chunking
    md_chunker = MarkdownChunker(chunk_size=150)
    md_chunks = md_chunker.chunk(sample_text)
    print(f"\nMarkdown chunker created {len(md_chunks)} chunks")
