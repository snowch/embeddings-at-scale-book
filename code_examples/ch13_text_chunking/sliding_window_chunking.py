"""Sliding window chunking with configurable overlap."""

from typing import List, Iterator, Tuple
from dataclasses import dataclass


@dataclass
class ChunkWithMetadata:
    """A chunk with position metadata for deduplication."""
    text: str
    start_char: int
    end_char: int
    chunk_index: int


def sliding_window_chunks(
    text: str,
    window_size: int = 500,
    stride: int = 400,
    respect_sentences: bool = True
) -> List[ChunkWithMetadata]:
    """
    Create overlapping chunks using a sliding window.

    Args:
        text: Input text to chunk
        window_size: Size of each chunk in characters
        stride: How far to move the window (overlap = window_size - stride)
        respect_sentences: Adjust boundaries to not split sentences

    Returns:
        List of chunks with position metadata
    """
    if stride > window_size:
        stride = window_size  # No overlap

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + window_size, len(text))

        # Adjust end to sentence boundary if requested
        if respect_sentences and end < len(text):
            # Look for sentence end within last 20% of window
            search_start = end - int(window_size * 0.2)
            sentence_end = find_sentence_end(text, search_start, end)
            if sentence_end > search_start:
                end = sentence_end

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(ChunkWithMetadata(
                text=chunk_text,
                start_char=start,
                end_char=end,
                chunk_index=chunk_index
            ))
            chunk_index += 1

        # Move window
        start += stride

        # Adjust start to sentence boundary
        if respect_sentences and start < len(text):
            sentence_start = find_sentence_start(text, start, min(start + 50, len(text)))
            if sentence_start > start:
                start = sentence_start

    return chunks


def find_sentence_end(text: str, start: int, end: int) -> int:
    """Find the last sentence ending in the range."""
    sentence_enders = '.!?'
    for i in range(end - 1, start - 1, -1):
        if text[i] in sentence_enders:
            # Check it's not an abbreviation
            if i + 1 < len(text) and text[i + 1] in ' \n':
                return i + 1
    return end


def find_sentence_start(text: str, start: int, end: int) -> int:
    """Find the first sentence start after position."""
    for i in range(start, end):
        if i > 0 and text[i - 1] in '.!?\n' and text[i] not in '.!?\n ':
            return i
    return start


def deduplicate_results(
    chunks: List[ChunkWithMetadata],
    query_matches: List[Tuple[int, float]]
) -> List[Tuple[int, float]]:
    """
    Remove duplicate matches from overlapping chunks.

    Args:
        chunks: List of chunks with position metadata
        query_matches: List of (chunk_index, similarity_score) pairs

    Returns:
        Deduplicated list of matches
    """
    # Sort by similarity score descending
    sorted_matches = sorted(query_matches, key=lambda x: x[1], reverse=True)

    seen_ranges = []
    deduplicated = []

    for chunk_idx, score in sorted_matches:
        chunk = chunks[chunk_idx]

        # Check if this range significantly overlaps with already seen ranges
        is_duplicate = False
        for seen_start, seen_end in seen_ranges:
            overlap = calculate_overlap(
                chunk.start_char, chunk.end_char,
                seen_start, seen_end
            )
            if overlap > 0.5:  # More than 50% overlap
                is_duplicate = True
                break

        if not is_duplicate:
            deduplicated.append((chunk_idx, score))
            seen_ranges.append((chunk.start_char, chunk.end_char))

    return deduplicated


def calculate_overlap(start1: int, end1: int, start2: int, end2: int) -> float:
    """Calculate overlap ratio between two ranges."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_start >= overlap_end:
        return 0.0

    overlap_length = overlap_end - overlap_start
    min_length = min(end1 - start1, end2 - start2)

    return overlap_length / min_length


# Example usage
if __name__ == "__main__":
    sample_text = """
    Machine learning has transformed how we process data. Neural networks
    form the foundation of modern AI systems. Deep learning enables
    hierarchical feature learning.

    Transformers revolutionized natural language processing. Attention
    mechanisms allow models to focus on relevant parts of the input.
    BERT and GPT are examples of transformer-based models.

    Embeddings represent data as dense vectors. Similar items have similar
    embeddings. This enables semantic search and recommendation systems.

    Vector databases store and retrieve embeddings efficiently. They use
    approximate nearest neighbor algorithms for fast search at scale.
    """ * 3

    # Create overlapping chunks
    chunks = sliding_window_chunks(
        sample_text,
        window_size=200,
        stride=150,  # 50 char overlap
        respect_sentences=True
    )

    print(f"Created {len(chunks)} overlapping chunks:\n")
    for chunk in chunks[:5]:  # Show first 5
        print(f"Chunk {chunk.chunk_index}: chars {chunk.start_char}-{chunk.end_char}")
        print(f"  {chunk.text[:80]}...")
        print()

    # Calculate overlap between consecutive chunks
    if len(chunks) > 1:
        overlap = calculate_overlap(
            chunks[0].start_char, chunks[0].end_char,
            chunks[1].start_char, chunks[1].end_char
        )
        print(f"Overlap between chunk 0 and 1: {overlap:.1%}")
