"""Fixed-size text chunking implementation."""

from typing import List


def chunk_by_characters(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into fixed-size character chunks with optional overlap.

    Args:
        text: Input text to chunk
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Don't add empty chunks
        if chunk.strip():
            chunks.append(chunk)

        # Move start position, accounting for overlap
        start = end - overlap

        # Prevent infinite loop if overlap >= chunk_size
        if overlap >= chunk_size:
            start = end

    return chunks


def chunk_by_tokens(text: str, chunk_size: int = 256, overlap: int = 25) -> List[str]:
    """
    Split text into fixed-size token chunks using tiktoken.

    Args:
        text: Input text to chunk
        chunk_size: Number of tokens per chunk
        overlap: Number of overlapping tokens between chunks

    Returns:
        List of text chunks
    """
    import tiktoken

    # Use cl100k_base encoding (used by GPT-4, text-embedding-ada-002)
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)

        if chunk_text.strip():
            chunks.append(chunk_text)

        start = end - overlap
        if overlap >= chunk_size:
            start = end

    return chunks


# Example usage
if __name__ == "__main__":
    sample_text = (
        """
    Machine learning has transformed how we process and understand data.
    Neural networks, inspired by biological neurons, can learn complex patterns.
    Deep learning extends this with multiple layers of abstraction.
    Transformers revolutionized NLP with attention mechanisms.
    Embeddings provide dense vector representations of semantic meaning.
    """
        * 5
    )  # Repeat to create longer text

    # Character-based chunking
    char_chunks = chunk_by_characters(sample_text, chunk_size=200, overlap=20)
    print(f"Character chunking: {len(char_chunks)} chunks")
    print(f"First chunk ({len(char_chunks[0])} chars): {char_chunks[0][:100]}...")

    # Token-based chunking
    token_chunks = chunk_by_tokens(sample_text, chunk_size=50, overlap=5)
    print(f"\nToken chunking: {len(token_chunks)} chunks")
    print(f"First chunk: {token_chunks[0][:100]}...")
