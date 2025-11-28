"""Semantic chunking based on embedding similarity."""

from typing import List, Tuple

import numpy as np


def semantic_chunk(
    text: str,
    similarity_threshold: float = 0.5,
    min_chunk_sentences: int = 2,
    max_chunk_sentences: int = 10,
) -> List[str]:
    """
    Split text at semantic boundaries detected by embedding similarity drops.

    Args:
        text: Input text to chunk
        similarity_threshold: Split when similarity drops below this threshold
        min_chunk_sentences: Minimum sentences per chunk
        max_chunk_sentences: Maximum sentences before forcing a split

    Returns:
        List of semantically coherent text chunks
    """
    from sentence_chunking import split_into_sentences
    from sentence_transformers import SentenceTransformer

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Split into sentences
    sentences = split_into_sentences(text)
    if len(sentences) < 2:
        return [text]

    # Embed all sentences
    embeddings = model.encode(sentences)

    # Calculate similarity between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i + 1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
        )
        similarities.append(sim)

    # Find split points where similarity drops
    chunks = []
    current_chunk = [sentences[0]]

    for sentence, sim in zip(sentences[1:], similarities):
        should_split = sim < similarity_threshold and len(current_chunk) >= min_chunk_sentences
        force_split = len(current_chunk) >= max_chunk_sentences

        if should_split or force_split:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
        else:
            current_chunk.append(sentence)

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def semantic_chunk_with_breakpoints(
    text: str, percentile_threshold: int = 25
) -> Tuple[List[str], List[float]]:
    """
    Split at natural breakpoints using percentile-based threshold.

    Returns chunks and the similarity scores for analysis.
    """
    from sentence_chunking import split_into_sentences
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = split_into_sentences(text)

    if len(sentences) < 2:
        return [text], []

    embeddings = model.encode(sentences)

    # Calculate similarities
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i + 1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
        )
        similarities.append(sim)

    # Use percentile-based threshold (split at lowest N% similarity points)
    threshold = np.percentile(similarities, percentile_threshold)

    chunks = []
    current_chunk = [sentences[0]]

    for sentence, sim in zip(sentences[1:], similarities):
        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
        else:
            current_chunk.append(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks, similarities


# Example usage
if __name__ == "__main__":
    sample_text = """
    Machine learning enables computers to learn from data. It has revolutionized
    many industries including healthcare, finance, and technology.

    Neural networks are inspired by the human brain. They consist of layers of
    interconnected nodes that process information. Deep learning uses many
    layers to learn complex patterns.

    Natural language processing focuses on understanding human language. Modern
    NLP uses transformer models like BERT and GPT. These models can understand
    context and generate human-like text.

    Computer vision enables machines to interpret images. Convolutional neural
    networks are commonly used for image classification. Object detection and
    segmentation are key applications.
    """

    # Basic semantic chunking
    chunks = semantic_chunk(sample_text, similarity_threshold=0.6)

    print(f"Created {len(chunks)} semantic chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ---")
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)
        print()

    # With similarity analysis
    chunks, sims = semantic_chunk_with_breakpoints(sample_text, percentile_threshold=30)
    print(f"\nSimilarity scores between sentences: {[f'{s:.2f}' for s in sims]}")
