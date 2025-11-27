"""Table-aware chunking that preserves tabular structure."""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TableChunk:
    """A chunk containing table data with context."""
    text: str
    table_markdown: str
    caption: Optional[str] = None
    headers: List[str] = None
    num_rows: int = 0
    num_cols: int = 0


def detect_tables(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect tables in text and return their positions.

    Returns list of (start_pos, end_pos, table_text) tuples.
    """
    tables = []

    # Detect Markdown tables
    md_table_pattern = r'(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n?)+)'
    for match in re.finditer(md_table_pattern, text):
        tables.append((match.start(), match.end(), match.group(0)))

    # Detect ASCII tables (simple grid format)
    ascii_pattern = r'(\+[-+]+\+\n(?:\|[^\n]+\|\n)+\+[-+]+\+)'
    for match in re.finditer(ascii_pattern, text):
        tables.append((match.start(), match.end(), match.group(0)))

    return sorted(tables, key=lambda x: x[0])


def parse_markdown_table(table_text: str) -> Tuple[List[str], List[List[str]]]:
    """Parse a Markdown table into headers and rows."""
    lines = [line.strip() for line in table_text.strip().split('\n')]

    if len(lines) < 2:
        return [], []

    # Parse header row
    headers = [cell.strip() for cell in lines[0].split('|')[1:-1]]

    # Skip separator line, parse data rows
    rows = []
    for line in lines[2:]:
        if line.startswith('|'):
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            rows.append(cells)

    return headers, rows


def table_to_text(
    headers: List[str],
    rows: List[List[str]],
    format: str = 'natural'
) -> str:
    """
    Convert table to natural language for embedding.

    Args:
        headers: Column headers
        rows: Table rows
        format: 'natural' for sentences, 'structured' for key-value pairs

    Returns:
        Text representation of the table
    """
    if format == 'natural':
        lines = []
        for row in rows:
            parts = []
            for header, value in zip(headers, row):
                if value and value != '-':
                    parts.append(f"{header} is {value}")
            if parts:
                lines.append(". ".join(parts) + ".")
        return "\n".join(lines)

    elif format == 'structured':
        lines = []
        for i, row in enumerate(rows):
            row_text = f"Row {i + 1}: "
            row_text += ", ".join(f"{h}={v}" for h, v in zip(headers, row) if v)
            lines.append(row_text)
        return "\n".join(lines)

    else:
        # Keep as markdown
        return f"| {' | '.join(headers)} |\n" + \
               f"|{'|'.join(['---'] * len(headers))}|\n" + \
               "\n".join(f"| {' | '.join(row)} |" for row in rows)


class TableAwareChunker:
    """
    Chunker that handles tables as atomic units.

    Tables are either:
    1. Kept whole if small enough
    2. Split row-by-row with header context if too large
    3. Converted to natural language for better embedding
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        table_format: str = 'natural',
        max_table_rows_per_chunk: int = 10
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_format = table_format
        self.max_table_rows = max_table_rows_per_chunk

    def chunk_with_tables(self, text: str) -> List[str]:
        """
        Chunk text while handling tables specially.

        Tables are converted to text format and chunked appropriately.
        """
        tables = detect_tables(text)

        if not tables:
            from recursive_chunking import RecursiveChunker
            chunker = RecursiveChunker(self.chunk_size, self.chunk_overlap)
            return chunker.chunk(text)

        chunks = []
        last_end = 0

        for start, end, table_text in tables:
            # Chunk text before table
            if start > last_end:
                pre_text = text[last_end:start]
                from recursive_chunking import RecursiveChunker
                chunker = RecursiveChunker(self.chunk_size, self.chunk_overlap)
                chunks.extend(chunker.chunk(pre_text))

            # Process table
            table_chunks = self._chunk_table(table_text)
            chunks.extend(table_chunks)

            last_end = end

        # Chunk remaining text
        if last_end < len(text):
            post_text = text[last_end:]
            from recursive_chunking import RecursiveChunker
            chunker = RecursiveChunker(self.chunk_size, self.chunk_overlap)
            chunks.extend(chunker.chunk(post_text))

        return chunks

    def _chunk_table(self, table_text: str) -> List[str]:
        """Convert and chunk a single table."""
        headers, rows = parse_markdown_table(table_text)

        if not headers or not rows:
            return [table_text]  # Return as-is if parsing fails

        # Small table: keep as single chunk
        if len(rows) <= self.max_table_rows:
            text = table_to_text(headers, rows, self.table_format)
            if len(text) <= self.chunk_size:
                return [f"[Table with {len(rows)} rows]\n{text}"]

        # Large table: split into row groups
        chunks = []
        for i in range(0, len(rows), self.max_table_rows):
            row_group = rows[i:i + self.max_table_rows]
            text = table_to_text(headers, row_group, self.table_format)

            header_context = f"[Table columns: {', '.join(headers)}]\n"
            chunk = header_context + text
            chunks.append(chunk)

        return chunks


# Example usage
if __name__ == "__main__":
    sample_text = """
# Product Comparison

Here is a comparison of different machine learning frameworks:

| Framework | Language | GPU Support | Ease of Use | Community |
|-----------|----------|-------------|-------------|-----------|
| TensorFlow | Python | Excellent | Medium | Very Large |
| PyTorch | Python | Excellent | High | Large |
| JAX | Python | Excellent | Low | Growing |
| scikit-learn | Python | Limited | Very High | Large |

TensorFlow and PyTorch are the most popular choices for deep learning.

## Performance Benchmarks

| Model | Framework | Training Time | Accuracy |
|-------|-----------|---------------|----------|
| ResNet-50 | TensorFlow | 2.5 hours | 76.1% |
| ResNet-50 | PyTorch | 2.3 hours | 76.3% |
| BERT-base | TensorFlow | 4 hours | 88.5% |
| BERT-base | PyTorch | 3.8 hours | 88.7% |

These benchmarks were run on identical hardware.
    """

    chunker = TableAwareChunker(
        chunk_size=300,
        chunk_overlap=30,
        table_format='natural',
        max_table_rows_per_chunk=5
    )

    chunks = chunker.chunk_with_tables(sample_text)

    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        print()
