"""Handle code blocks embedded in documentation."""

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CodeBlock:
    """A code block extracted from documentation."""

    start_pos: int
    end_pos: int
    code: str
    language: str
    preceding_context: str = ""


def extract_code_blocks(text: str) -> Tuple[str, List[CodeBlock]]:
    """
    Extract fenced code blocks and replace with placeholders.

    Returns:
        Tuple of (text with placeholders, list of CodeBlock objects)
    """
    code_blocks = []
    placeholder_template = "<<<CODE_BLOCK_{}>>>"

    def replace_block(match):
        index = len(code_blocks)
        language = match.group(1) or "text"
        code = match.group(2)

        # Get preceding line for context
        start = match.start()
        preceding = text[max(0, start - 200) : start]
        last_line = preceding.split("\n")[-1].strip()

        code_blocks.append(
            CodeBlock(
                start_pos=match.start(),
                end_pos=match.end(),
                code=code,
                language=language,
                preceding_context=last_line,
            )
        )

        return placeholder_template.format(index)

    # Match fenced code blocks
    pattern = r"```(\w*)\n(.*?)```"
    text_with_placeholders = re.sub(pattern, replace_block, text, flags=re.DOTALL)

    return text_with_placeholders, code_blocks


def restore_code_blocks(
    chunks: List[str], code_blocks: List[CodeBlock], format: str = "inline"
) -> List[str]:
    """
    Restore code blocks to chunks.

    Args:
        chunks: Chunked text with placeholders
        code_blocks: Extracted code blocks
        format: 'inline' keeps code, 'reference' adds description only

    Returns:
        Chunks with code restored or referenced
    """
    placeholder_pattern = r"<<<CODE_BLOCK_(\d+)>>>"

    restored = []
    for chunk in chunks:
        matches = list(re.finditer(placeholder_pattern, chunk))

        if not matches:
            restored.append(chunk)
            continue

        result = chunk
        for match in reversed(matches):  # Reverse to maintain positions
            index = int(match.group(1))
            block = code_blocks[index]

            if format == "inline":
                replacement = f"```{block.language}\n{block.code}```"
            elif format == "reference":
                replacement = f"[Code block: {block.language}]"
            elif format == "summary":
                summary = summarize_code(block.code, block.language)
                replacement = f"[Code: {summary}]"
            else:
                replacement = block.code

            result = result[: match.start()] + replacement + result[match.end() :]

        restored.append(result)

    return restored


def summarize_code(code: str, language: str) -> str:
    """Generate a brief summary of code for embedding."""
    lines = code.strip().split("\n")

    if language == "python":
        # Extract function/class names
        defs = [line for line in lines if line.strip().startswith(("def ", "class "))]
        if defs:
            return defs[0].split("(")[0].replace("def ", "").replace("class ", "")

    elif language in ("javascript", "typescript"):
        # Extract function names
        funcs = [line for line in lines if "function" in line or "=>" in line]
        if funcs:
            return funcs[0][:50]

    # Default: first non-empty line
    for line in lines:
        if line.strip():
            return line.strip()[:50]

    return "code block"


class CodeAwareChunker:
    """
    Chunker that handles code blocks intelligently.

    Options:
    - Keep code inline (increases chunk size)
    - Extract and reference (smaller chunks, may lose context)
    - Keep code with explanation (pairs code with surrounding text)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        code_handling: str = "inline",  # 'inline', 'extract', 'pair'
        max_code_lines: int = 30,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.code_handling = code_handling
        self.max_code_lines = max_code_lines

    def chunk_with_code(self, text: str) -> List[str]:
        """
        Chunk documentation with embedded code.

        Returns chunks with code handled according to settings.
        """
        text_clean, code_blocks = extract_code_blocks(text)

        if self.code_handling == "pair":
            return self._chunk_paired(text, code_blocks)

        from recursive_chunking import RecursiveChunker

        chunker = RecursiveChunker(self.chunk_size, self.chunk_overlap)
        chunks = chunker.chunk(text_clean)

        if self.code_handling == "inline":
            return restore_code_blocks(chunks, code_blocks, "inline")
        else:  # 'extract'
            return restore_code_blocks(chunks, code_blocks, "reference")

    def _chunk_paired(self, text: str, code_blocks: List[CodeBlock]) -> List[str]:
        """
        Create chunks that pair code with its explanation.

        Each code block stays with the text that introduces it.
        """
        chunks = []
        last_end = 0

        for block in code_blocks:
            # Find the paragraph/section containing this code
            context_start = self._find_context_start(text, block.start_pos)
            context_end = block.end_pos

            # Extend to include following explanation if present
            next_para_end = self._find_paragraph_end(text, block.end_pos)
            if next_para_end - block.end_pos < 200:  # Include if short
                context_end = next_para_end

            # Chunk text before this context
            if context_start > last_end:
                pre_text = text[last_end:context_start]
                from recursive_chunking import RecursiveChunker

                chunker = RecursiveChunker(self.chunk_size, self.chunk_overlap)
                chunks.extend(chunker.chunk(pre_text))

            # Add paired chunk
            paired_chunk = text[context_start:context_end].strip()
            if paired_chunk:
                chunks.append(paired_chunk)

            last_end = context_end

        # Remaining text
        if last_end < len(text):
            from recursive_chunking import RecursiveChunker

            chunker = RecursiveChunker(self.chunk_size, self.chunk_overlap)
            chunks.extend(chunker.chunk(text[last_end:]))

        return chunks

    def _find_context_start(self, text: str, code_start: int) -> int:
        """Find the start of the context introducing a code block."""
        # Look back for paragraph break or heading
        search_text = text[max(0, code_start - 500) : code_start]

        # Find last paragraph break
        para_breaks = [m.end() for m in re.finditer(r"\n\n", search_text)]
        if para_breaks:
            return max(0, code_start - 500) + para_breaks[-1]

        # Find last heading
        headings = [m.start() for m in re.finditer(r"^#+\s", search_text, re.MULTILINE)]
        if headings:
            return max(0, code_start - 500) + headings[-1]

        return max(0, code_start - 200)

    def _find_paragraph_end(self, text: str, start: int) -> int:
        """Find the end of the paragraph after a position."""
        search_text = text[start : start + 500]
        match = re.search(r"\n\n", search_text)
        if match:
            return start + match.start()
        return min(start + 500, len(text))


# Example usage
if __name__ == "__main__":
    sample_doc = """
# Getting Started with Embeddings

Embeddings transform text into numerical vectors. Here's how to create them:

## Using Sentence Transformers

The sentence-transformers library makes it easy:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(['Hello world', 'How are you?'])
print(embeddings.shape)  # (2, 384)
```

This creates 384-dimensional embeddings for each input sentence.

## Comparing Similarity

You can compare embeddings using cosine similarity:

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
print(f"Similarity: {similarity[0][0]:.3f}")
```

Higher values indicate more similar meanings.
    """

    # Test different handling modes
    for mode in ["inline", "extract", "pair"]:
        print(f"\n{'=' * 50}")
        print(f"Mode: {mode}")
        print("=" * 50)

        chunker = CodeAwareChunker(chunk_size=300, chunk_overlap=30, code_handling=mode)

        chunks = chunker.chunk_with_code(sample_doc)

        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i + 1} ---")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
