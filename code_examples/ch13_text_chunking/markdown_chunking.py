"""Markdown-aware text chunking with header hierarchy preservation."""

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MarkdownChunk:
    """A chunk from a Markdown document with context."""

    text: str
    header_hierarchy: List[str]  # [h1, h2, h3, ...]
    header_level: int
    start_line: int
    end_line: int


class MarkdownChunker:
    """
    Chunk Markdown documents while preserving header hierarchy.

    Each chunk includes its full header context for better retrieval.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        include_header_context: bool = True,
        min_chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_header_context = include_header_context
        self.min_chunk_size = min_chunk_size

    def chunk_markdown(self, markdown: str) -> List[MarkdownChunk]:
        """
        Chunk a Markdown document.

        Args:
            markdown: Raw Markdown string

        Returns:
            List of MarkdownChunk objects
        """
        lines = markdown.split("\n")
        sections = self._parse_sections(lines)
        chunks = self._chunk_sections(sections)

        return chunks

    def _parse_sections(self, lines: List[str]) -> List[dict]:
        """Parse Markdown into sections based on headers."""
        sections = []
        current_section = {"headers": [], "content": [], "start_line": 0, "level": 0}
        header_stack = []  # Track header hierarchy

        for i, line in enumerate(lines):
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)

            if header_match:
                # Save current section if it has content
                if current_section["content"]:
                    current_section["end_line"] = i - 1
                    sections.append(current_section)

                # Update header stack
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Pop headers of same or lower level
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()

                header_stack.append((level, title))

                # Start new section
                current_section = {
                    "headers": [h[1] for h in header_stack],
                    "content": [line],
                    "start_line": i,
                    "level": level,
                }
            else:
                current_section["content"].append(line)

        # Don't forget last section
        if current_section["content"]:
            current_section["end_line"] = len(lines) - 1
            sections.append(current_section)

        return sections

    def _chunk_sections(self, sections: List[dict]) -> List[MarkdownChunk]:
        """Chunk each section into appropriately sized pieces."""
        chunks = []

        for section in sections:
            content = "\n".join(section["content"])

            # Add header context if requested
            if self.include_header_context and section["headers"]:
                header_context = self._build_header_context(section["headers"])
            else:
                header_context = ""

            # If section is small enough, keep as single chunk
            if len(content) <= self.chunk_size:
                if content.strip():
                    chunks.append(
                        MarkdownChunk(
                            text=header_context + content if header_context else content,
                            header_hierarchy=section["headers"],
                            header_level=section["level"],
                            start_line=section["start_line"],
                            end_line=section.get("end_line", section["start_line"]),
                        )
                    )
            else:
                # Split large sections
                sub_chunks = self._split_section(content, header_context)
                for sub_text in sub_chunks:
                    chunks.append(
                        MarkdownChunk(
                            text=sub_text,
                            header_hierarchy=section["headers"],
                            header_level=section["level"],
                            start_line=section["start_line"],
                            end_line=section.get("end_line", section["start_line"]),
                        )
                    )

        return chunks

    def _build_header_context(self, headers: List[str]) -> str:
        """Build a header context string."""
        if not headers:
            return ""

        # Format as breadcrumb
        return " > ".join(headers) + "\n\n"

    def _split_section(self, content: str, header_context: str) -> List[str]:
        """Split a large section into smaller chunks."""
        from recursive_chunking import RecursiveChunker

        # Account for header context in chunk size
        effective_chunk_size = self.chunk_size - len(header_context)

        chunker = RecursiveChunker(
            chunk_size=effective_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",  # Paragraphs
                "\n- ",  # List items
                "\n* ",  # List items
                "\n",  # Lines
                ". ",  # Sentences
                " ",  # Words
            ],
        )

        text_chunks = chunker.chunk(content)

        # Prepend header context to each chunk
        return [
            header_context + chunk if header_context else chunk
            for chunk in text_chunks
            if chunk.strip()
        ]


def extract_code_blocks(markdown: str) -> Tuple[str, List[dict]]:
    """
    Extract code blocks from Markdown for separate processing.

    Returns:
        Tuple of (markdown with placeholders, list of code blocks)
    """
    code_blocks = []
    placeholder_pattern = "<<<CODE_BLOCK_{}>>>"

    def replace_code(match):
        index = len(code_blocks)
        code_blocks.append(
            {"language": match.group(1) or "", "code": match.group(2), "full_match": match.group(0)}
        )
        return placeholder_pattern.format(index)

    # Match fenced code blocks
    pattern = r"```(\w*)\n(.*?)```"
    cleaned = re.sub(pattern, replace_code, markdown, flags=re.DOTALL)

    return cleaned, code_blocks


# Example usage
if __name__ == "__main__":
    sample_markdown = """
# Machine Learning Guide

An introduction to machine learning concepts and techniques.

## Supervised Learning

Supervised learning uses labeled data to train models.

### Classification

Classification predicts categorical outputs.

Common algorithms include:
- Decision Trees
- Random Forests
- Neural Networks

Classification is used for spam detection, image recognition,
and medical diagnosis.

### Regression

Regression predicts continuous values.

Linear regression is the simplest form:
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.

### Clustering

Clustering groups similar items together. K-means is
a popular clustering algorithm.

### Dimensionality Reduction

PCA and t-SNE reduce the number of features while
preserving important information.
    """

    chunker = MarkdownChunker(chunk_size=300, chunk_overlap=30, include_header_context=True)

    chunks = chunker.chunk_markdown(sample_markdown)

    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ---")
        print(f"Headers: {' > '.join(chunk.header_hierarchy)}")
        print(f"Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"Text preview: {chunk.text[:150]}...")
        print()
