"""Paragraph-based text chunking implementation."""

import re
from typing import List


def chunk_by_paragraphs(
    text: str, min_chunk_size: int = 100, max_chunk_size: int = 500, combine_short: bool = True
) -> List[str]:
    """
    Split text on paragraph boundaries, optionally combining short paragraphs.

    Args:
        text: Input text to chunk
        min_chunk_size: Minimum characters per chunk (combine if smaller)
        max_chunk_size: Maximum characters before splitting paragraph
        combine_short: Whether to combine paragraphs shorter than min_chunk_size

    Returns:
        List of text chunks
    """
    # Split on double newlines (paragraph boundaries)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        # Handle very long paragraphs
        if para_size > max_chunk_size:
            # Save current accumulation first
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            # Split long paragraph (fall back to sentence splitting)
            from sentence_chunking import chunk_by_sentences

            sub_chunks = chunk_by_sentences(para, target_size=max_chunk_size // 4)
            chunks.extend(sub_chunks)
            continue

        # Combine short paragraphs if enabled
        if (
            combine_short and current_size + para_size < min_chunk_size
        ) or current_size + para_size <= max_chunk_size:
            current_chunk.append(para)
            current_size += para_size
        else:
            # Save current and start new
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_size = para_size

    # Don't forget the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


# Example usage
if __name__ == "__main__":
    sample_text = """
Machine learning represents a paradigm shift in computing. Rather than
explicitly programming rules, we train models on data to discover patterns.

This approach excels when the underlying rules are complex or unknown.
Image recognition, natural language understanding, and game playing are
prime examples where ML outperforms traditional programming.

Neural networks are the workhorse of modern machine learning. Inspired by
biological neurons, these models consist of layers of interconnected nodes.

Each node applies a simple mathematical transformation. When combined
across thousands or millions of nodes, complex functions emerge.

Deep learning extends neural networks with many layers. This depth allows
hierarchical feature learning—early layers detect simple patterns like
edges, while later layers compose these into complex concepts like faces
or sentences.
    """

    chunks = chunk_by_paragraphs(
        sample_text, min_chunk_size=150, max_chunk_size=400, combine_short=True
    )

    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ({len(chunk)} chars) ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        print()
