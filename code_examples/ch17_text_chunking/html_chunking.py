"""HTML-aware text chunking with structure preservation."""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HTMLChunk:
    """A chunk extracted from HTML with metadata."""

    text: str
    tag_path: str  # e.g., "html > body > article > section"
    heading: Optional[str] = None
    url_fragment: Optional[str] = None  # For linking back to source


class HTMLChunker:
    """
    Extract and chunk text from HTML while preserving semantic structure.

    Uses HTML tags to determine natural chunk boundaries.
    """

    # Tags that typically contain coherent content blocks
    BLOCK_TAGS = {
        "article",
        "section",
        "div",
        "p",
        "blockquote",
        "li",
        "td",
        "th",
        "figcaption",
        "aside",
    }

    # Tags that define sections with headings
    SECTION_TAGS = {"article", "section", "main", "aside"}

    # Heading tags in order of importance
    HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        preserve_structure: bool = True,
        include_headings: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_structure = preserve_structure
        self.include_headings = include_headings

    def chunk_html(self, html: str) -> List[HTMLChunk]:
        """
        Extract and chunk HTML content.

        Args:
            html: Raw HTML string

        Returns:
            List of HTMLChunk objects
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        if self.preserve_structure:
            return self._chunk_by_structure(soup)
        else:
            return self._chunk_flat(soup)

    def _chunk_by_structure(self, soup) -> List[HTMLChunk]:
        """Chunk based on HTML structure (sections, articles, etc.)."""
        chunks = []
        current_heading = None

        # Find main content area
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_=re.compile(r"content|main|article"))
            or soup.body
            or soup
        )

        # Process section by section
        for section in self._find_sections(main_content):
            section_heading = self._extract_heading(section)
            if section_heading:
                current_heading = section_heading

            section_text = self._extract_text(section)
            if not section_text.strip():
                continue

            # Chunk the section content
            section_chunks = self._split_text(
                section_text, current_heading, self._get_tag_path(section)
            )
            chunks.extend(section_chunks)

        return chunks

    def _chunk_flat(self, soup) -> List[HTMLChunk]:
        """Simple flat chunking of all text content."""
        text = soup.get_text(separator="\n", strip=True)
        return self._split_text(text, None, "body")

    def _find_sections(self, element):
        """Find content sections in the HTML."""
        # First try explicit section tags
        sections = element.find_all(self.SECTION_TAGS)

        if sections:
            yield from sections
        else:
            # Fall back to paragraphs and divs
            for child in element.children:
                if hasattr(child, "name"):
                    if child.name in self.BLOCK_TAGS:
                        yield child
                    elif child.name in self.HEADING_TAGS:
                        # Include heading with following content
                        yield child

    def _extract_heading(self, element) -> Optional[str]:
        """Extract the heading for a section."""
        for tag in self.HEADING_TAGS:
            heading = element.find(tag)
            if heading:
                return heading.get_text(strip=True)
        return None

    def _extract_text(self, element) -> str:
        """Extract clean text from an element."""
        # Get text with newlines between block elements
        text = element.get_text(separator="\n", strip=True)

        # Clean up multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def _get_tag_path(self, element) -> str:
        """Get the tag path to an element."""
        path = []
        current = element

        while current and hasattr(current, "name") and current.name:
            tag_info = current.name
            if current.get("id"):
                tag_info += f"#{current['id']}"
            elif current.get("class"):
                tag_info += f".{current['class'][0]}"
            path.insert(0, tag_info)
            current = current.parent

        return " > ".join(path[-4:])  # Last 4 levels

    def _split_text(self, text: str, heading: Optional[str], tag_path: str) -> List[HTMLChunk]:
        """Split text into chunks."""
        from recursive_chunking import RecursiveChunker

        # Prepend heading if requested
        if self.include_headings and heading:
            text = f"## {heading}\n\n{text}"

        chunker = RecursiveChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        text_chunks = chunker.chunk(text)

        return [
            HTMLChunk(text=chunk, tag_path=tag_path, heading=heading)
            for chunk in text_chunks
            if chunk.strip()
        ]


# Example usage
if __name__ == "__main__":
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head><title>ML Guide</title></head>
    <body>
        <nav>Navigation menu</nav>

        <main>
            <article>
                <h1>Introduction to Machine Learning</h1>
                <p>Machine learning is a subset of artificial intelligence
                that focuses on building systems that learn from data.</p>

                <section id="supervised">
                    <h2>Supervised Learning</h2>
                    <p>Supervised learning uses labeled training data.
                    The algorithm learns mappings from inputs to outputs.</p>

                    <h3>Classification</h3>
                    <p>Classification predicts categorical labels like
                    spam/not-spam or cat/dog.</p>

                    <h3>Regression</h3>
                    <p>Regression predicts continuous values like prices
                    or temperatures.</p>
                </section>

                <section id="unsupervised">
                    <h2>Unsupervised Learning</h2>
                    <p>Unsupervised learning finds patterns in unlabeled
                    data through clustering and dimensionality reduction.</p>
                </section>
            </article>
        </main>

        <footer>Copyright 2024</footer>
    </body>
    </html>
    """

    chunker = HTMLChunker(
        chunk_size=200, chunk_overlap=20, preserve_structure=True, include_headings=True
    )

    chunks = chunker.chunk_html(sample_html)

    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ---")
        print(f"Path: {chunk.tag_path}")
        print(f"Heading: {chunk.heading}")
        print(f"Text: {chunk.text[:150]}...")
        print()
