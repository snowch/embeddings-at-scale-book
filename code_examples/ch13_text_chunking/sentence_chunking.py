"""Sentence-based text chunking implementation."""

from typing import List
import re


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex patterns.

    Handles common abbreviations and edge cases.
    """
    # Common abbreviations that shouldn't end sentences
    abbreviations = r"(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<!\s[A-Z]\.)"

    # Split on sentence-ending punctuation followed by space and capital
    pattern = abbreviations + r'(?<=[.!?])\s+(?=[A-Z])'

    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_sentences(
    text: str,
    target_size: int = 256,
    max_size: int = 512,
    overlap_sentences: int = 1
) -> List[str]:
    """
    Group sentences into chunks of approximately target_size tokens.

    Args:
        text: Input text to chunk
        target_size: Target number of tokens per chunk
        max_size: Maximum tokens before forcing a split
        overlap_sentences: Number of sentences to overlap between chunks

    Returns:
        List of text chunks
    """
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for i, sentence in enumerate(sentences):
        sentence_tokens = len(encoding.encode(sentence))

        # If single sentence exceeds max, it becomes its own chunk
        if sentence_tokens > max_size:
            # Save current chunk if not empty
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            chunks.append(sentence)
            continue

        # Check if adding this sentence exceeds target
        if current_tokens + sentence_tokens > target_size and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))

            # Start new chunk with overlap
            if overlap_sentences > 0:
                current_chunk = current_chunk[-overlap_sentences:]
                current_tokens = sum(
                    len(encoding.encode(s)) for s in current_chunk
                )
            else:
                current_chunk = []
                current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Example usage
if __name__ == "__main__":
    sample_text = """
    Machine learning has fundamentally transformed data processing. It enables
    computers to learn from experience without explicit programming. Neural
    networks form the backbone of modern ML systems. They consist of
    interconnected nodes that process information in layers.

    Deep learning extends neural networks with multiple hidden layers. This
    allows the model to learn hierarchical representations. Each layer captures
    increasingly abstract features. The first layer might detect edges, while
    deeper layers recognize objects.

    Transformers introduced the attention mechanism to NLP. Self-attention
    allows the model to weigh the importance of different parts of the input.
    This architecture powers models like BERT and GPT. These models have
    achieved state-of-the-art results across many tasks.
    """

    chunks = chunk_by_sentences(
        sample_text,
        target_size=100,
        overlap_sentences=1
    )

    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk[:100]}...")
        print()
