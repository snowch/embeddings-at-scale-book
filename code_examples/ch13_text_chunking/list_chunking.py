"""List-aware chunking that keeps related items together."""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class ListBlock:
    """A detected list block in text."""
    start_pos: int
    end_pos: int
    text: str
    list_type: str  # 'bullet', 'numbered', 'definition'
    items: List[str]
    indent_level: int = 0


def detect_lists(text: str) -> List[ListBlock]:
    """
    Detect list structures in text.

    Handles:
    - Bullet lists (-, *, •)
    - Numbered lists (1., 2., a., b.)
    - Definition lists (term: definition)
    """
    lists = []

    # Bullet list pattern
    bullet_pattern = r'((?:^[ \t]*[-*•][ \t]+.+$\n?)+)'

    # Numbered list pattern
    numbered_pattern = r'((?:^[ \t]*(?:\d+\.|[a-z]\.)[ \t]+.+$\n?)+)'

    for pattern, list_type in [(bullet_pattern, 'bullet'), (numbered_pattern, 'numbered')]:
        for match in re.finditer(pattern, text, re.MULTILINE):
            items = parse_list_items(match.group(0), list_type)
            lists.append(ListBlock(
                start_pos=match.start(),
                end_pos=match.end(),
                text=match.group(0),
                list_type=list_type,
                items=items
            ))

    return sorted(lists, key=lambda x: x.start_pos)


def parse_list_items(list_text: str, list_type: str) -> List[str]:
    """Parse individual items from a list block."""
    if list_type == 'bullet':
        pattern = r'^[ \t]*[-*•][ \t]+(.+)$'
    else:  # numbered
        pattern = r'^[ \t]*(?:\d+\.|[a-z]\.)[ \t]+(.+)$'

    items = []
    for match in re.finditer(pattern, list_text, re.MULTILINE):
        items.append(match.group(1).strip())

    return items


class ListAwareChunker:
    """
    Chunker that keeps list items together when possible.

    Lists are atomic units - better to have a slightly larger chunk
    than to split a list in the middle.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        max_list_items_per_chunk: int = 20,
        list_context_prefix: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_list_items = max_list_items_per_chunk
        self.list_context_prefix = list_context_prefix

    def chunk_with_lists(self, text: str) -> List[str]:
        """
        Chunk text while keeping lists together.

        If a list is too long, it's split with context preserved.
        """
        lists = detect_lists(text)

        if not lists:
            from recursive_chunking import RecursiveChunker
            chunker = RecursiveChunker(self.chunk_size, self.chunk_overlap)
            return chunker.chunk(text)

        chunks = []
        last_end = 0

        for list_block in lists:
            # Get context before the list
            context = self._get_list_context(text, list_block.start_pos)

            # Chunk text before list
            if list_block.start_pos > last_end:
                pre_text = text[last_end:list_block.start_pos]
                from recursive_chunking import RecursiveChunker
                chunker = RecursiveChunker(self.chunk_size, self.chunk_overlap)
                pre_chunks = chunker.chunk(pre_text)
                chunks.extend(pre_chunks)

            # Process list
            list_chunks = self._chunk_list(list_block, context)
            chunks.extend(list_chunks)

            last_end = list_block.end_pos

        # Chunk remaining text
        if last_end < len(text):
            post_text = text[last_end:]
            from recursive_chunking import RecursiveChunker
            chunker = RecursiveChunker(self.chunk_size, self.chunk_overlap)
            chunks.extend(chunker.chunk(post_text))

        return chunks

    def _get_list_context(self, text: str, list_start: int) -> str:
        """
        Get the context (heading/intro) before a list.

        Looks for the nearest heading or sentence before the list.
        """
        # Look back up to 200 chars for context
        search_start = max(0, list_start - 200)
        preceding_text = text[search_start:list_start]

        # Find last heading
        heading_match = re.search(r'^#+\s+.+$', preceding_text, re.MULTILINE)
        if heading_match:
            return heading_match.group(0).strip()

        # Find last sentence
        sentences = re.split(r'[.!?]\s+', preceding_text)
        if sentences and len(sentences[-1]) > 10:
            return sentences[-1].strip()

        return ""

    def _chunk_list(self, list_block: ListBlock, context: str) -> List[str]:
        """Convert list to chunks, splitting if necessary."""
        items = list_block.items

        # Small list: keep as single chunk
        if len(items) <= self.max_list_items:
            list_text = self._format_list(items, list_block.list_type)
            if len(list_text) <= self.chunk_size:
                if self.list_context_prefix and context:
                    return [f"{context}\n\n{list_text}"]
                return [list_text]

        # Large list: split into groups
        chunks = []
        for i in range(0, len(items), self.max_list_items):
            item_group = items[i:i + self.max_list_items]
            list_text = self._format_list(item_group, list_block.list_type)

            if self.list_context_prefix and context:
                chunk = f"{context} (continued)\n\n{list_text}"
            else:
                chunk = list_text

            chunks.append(chunk)

        return chunks

    def _format_list(self, items: List[str], list_type: str) -> str:
        """Format items back into a list."""
        if list_type == 'bullet':
            return '\n'.join(f"- {item}" for item in items)
        else:
            return '\n'.join(f"{i + 1}. {item}" for i, item in enumerate(items))


def merge_adjacent_lists(text: str) -> str:
    """
    Merge adjacent lists of the same type.

    Prevents artificial splits between related lists.
    """
    # Merge consecutive bullet lists
    text = re.sub(
        r'((?:^[-*•]\s+.+$\n)+)\n+((?:^[-*•]\s+.+$\n?)+)',
        r'\1\2',
        text,
        flags=re.MULTILINE
    )

    # Merge consecutive numbered lists (renumber)
    # This is more complex, skip for simplicity
    return text


# Example usage
if __name__ == "__main__":
    sample_text = """
# Machine Learning Algorithms

There are many types of machine learning algorithms. Here are the main categories:

## Supervised Learning Algorithms

The following algorithms use labeled data:

- Linear Regression: Used for predicting continuous values
- Logistic Regression: Used for binary classification
- Decision Trees: Tree-based models for classification and regression
- Random Forests: Ensemble of decision trees
- Support Vector Machines: Find optimal hyperplane for classification
- Neural Networks: Deep learning models with multiple layers
- K-Nearest Neighbors: Instance-based learning algorithm
- Naive Bayes: Probabilistic classifier based on Bayes theorem
- Gradient Boosting: Sequential ensemble method

## Unsupervised Learning Algorithms

These algorithms work with unlabeled data:

1. K-Means Clustering: Partition data into k clusters
2. Hierarchical Clustering: Build tree of clusters
3. DBSCAN: Density-based clustering
4. PCA: Dimensionality reduction technique
5. t-SNE: Visualization of high-dimensional data
6. Autoencoders: Neural network for representation learning

Each algorithm has its strengths and weaknesses.
    """

    chunker = ListAwareChunker(
        chunk_size=400,
        chunk_overlap=30,
        max_list_items_per_chunk=5,
        list_context_prefix=True
    )

    chunks = chunker.chunk_with_lists(sample_text)

    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ({len(chunk)} chars) ---")
        print(chunk[:250] + "..." if len(chunk) > 250 else chunk)
        print()
