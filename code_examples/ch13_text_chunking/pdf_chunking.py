"""PDF-aware text chunking with structure preservation."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PDFChunk:
    """A chunk extracted from a PDF with metadata."""

    text: str
    page_numbers: List[int]
    section_title: Optional[str] = None
    chunk_type: str = "text"  # text, table, header, footer
    metadata: Dict = field(default_factory=dict)


class PDFChunker:
    """
    Extract and chunk text from PDFs while preserving structure.

    Handles headers/footers, multi-column layouts, and section detection.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        remove_headers_footers: bool = True,
        detect_sections: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.remove_headers_footers = remove_headers_footers
        self.detect_sections = detect_sections

    def chunk_pdf(self, pdf_path: str) -> List[PDFChunk]:
        """
        Extract and chunk a PDF document.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of PDFChunk objects with text and metadata
        """
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        chunks = []
        current_section = None

        for page_num, page in enumerate(doc):
            # Extract text with layout preservation
            text = page.get_text("text")

            # Remove headers/footers if requested
            if self.remove_headers_footers:
                text = self._remove_headers_footers(text, page_num, len(doc))

            # Detect section titles
            if self.detect_sections:
                sections = self._detect_sections(text)
                for section_title, section_text in sections:
                    current_section = section_title or current_section
                    page_chunks = self._chunk_text(section_text, page_num + 1, current_section)
                    chunks.extend(page_chunks)
            else:
                page_chunks = self._chunk_text(text, page_num + 1, current_section)
                chunks.extend(page_chunks)

        doc.close()

        # Merge small chunks across pages
        return self._merge_small_chunks(chunks)

    def _remove_headers_footers(self, text: str, page_num: int, total_pages: int) -> str:
        """Remove common header/footer patterns."""
        lines = text.split("\n")

        # Remove page numbers
        page_pattern = re.compile(
            rf"^\s*({page_num + 1}|Page\s+{page_num + 1}|{page_num + 1}\s*/\s*{total_pages})\s*$",
            re.IGNORECASE,
        )

        # Filter lines
        filtered = []
        for i, line in enumerate(lines):
            # Skip likely headers (first few lines with short text)
            if i < 3 and len(line.strip()) < 50 and not any(c.islower() for c in line):
                continue

            # Skip page numbers
            if page_pattern.match(line):
                continue

            # Skip likely footers (last few lines with common patterns)
            if i >= len(lines) - 3:
                if re.match(r"^\s*©|confidential|draft", line, re.IGNORECASE):
                    continue

            filtered.append(line)

        return "\n".join(filtered)

    def _detect_sections(self, text: str) -> List[tuple]:
        """Detect section headers and split text accordingly."""
        # Common section header patterns
        header_patterns = [
            r"^(?:Chapter\s+)?(\d+\.?\s+[A-Z][^\n]+)$",  # Numbered sections
            r"^([A-Z][A-Z\s]+)$",  # ALL CAPS headers
            r"^(#{1,3}\s+.+)$",  # Markdown-style headers
        ]

        sections = []
        current_title = None
        current_text = []

        for line in text.split("\n"):
            is_header = False

            for pattern in header_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    # Save previous section
                    if current_text:
                        sections.append((current_title, "\n".join(current_text)))

                    current_title = match.group(1)
                    current_text = []
                    is_header = True
                    break

            if not is_header:
                current_text.append(line)

        # Don't forget last section
        if current_text:
            sections.append((current_title, "\n".join(current_text)))

        return sections if sections else [(None, text)]

    def _chunk_text(self, text: str, page_num: int, section_title: Optional[str]) -> List[PDFChunk]:
        """Chunk text into appropriately sized pieces."""
        from recursive_chunking import RecursiveChunker

        chunker = RecursiveChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        text_chunks = chunker.chunk(text)

        return [
            PDFChunk(
                text=chunk, page_numbers=[page_num], section_title=section_title, chunk_type="text"
            )
            for chunk in text_chunks
            if chunk.strip()
        ]

    def _merge_small_chunks(self, chunks: List[PDFChunk]) -> List[PDFChunk]:
        """Merge chunks that are too small."""
        if not chunks:
            return chunks

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            # Merge if same section and combined size is acceptable
            can_merge = (
                current.section_title == next_chunk.section_title
                and len(current.text) + len(next_chunk.text) < self.chunk_size
            )

            if can_merge:
                current = PDFChunk(
                    text=current.text + "\n\n" + next_chunk.text,
                    page_numbers=list(set(current.page_numbers + next_chunk.page_numbers)),
                    section_title=current.section_title,
                    chunk_type=current.chunk_type,
                )
            else:
                merged.append(current)
                current = next_chunk

        merged.append(current)
        return merged


# Example usage
if __name__ == "__main__":
    # Simulated PDF content for demonstration
    sample_pdf_text = """
    Page 1

    1. INTRODUCTION

    This document provides an overview of machine learning techniques.
    Machine learning enables computers to learn from data without
    explicit programming.

    2. SUPERVISED LEARNING

    Supervised learning uses labeled training data. The algorithm
    learns to map inputs to outputs based on example input-output pairs.

    Page 2

    2.1 Classification

    Classification predicts categorical labels. Common algorithms include
    decision trees, random forests, and neural networks.

    2.2 Regression

    Regression predicts continuous values. Linear regression and
    polynomial regression are foundational techniques.
    """

    # Demonstrate chunking logic
    chunker = PDFChunker(
        chunk_size=200, chunk_overlap=30, remove_headers_footers=True, detect_sections=True
    )

    # Simulate section detection
    sections = chunker._detect_sections(sample_pdf_text)
    print(f"Detected {len(sections)} sections:\n")
    for title, text in sections:
        print(f"Section: {title}")
        print(f"  Content preview: {text[:100].strip()}...")
        print()
