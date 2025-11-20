#!/usr/bin/env python3
"""
Update Chapter 1 to use include directives for migrated code blocks.
"""

def main():
    chapter_file = 'chapters/ch01_embedding_revolution.qmd'

    # Read the file
    with open(chapter_file, 'r') as f:
        content = f.read()

    # Define replacements: (start_marker, end_marker, include_path)
    replacements = [
        # 1. Word embeddings similarity (lines 18-59)
        (
            "```python\nimport numpy as np\n\n# A simple 3-dimensional embedding space for illustration",
            "print(f\"king vs apple: {similarity('king', 'apple'):.3f}\")  # Low (~0.46) - unrelated concepts\n```",
            "/code_examples/ch01_foundations/word_embeddings_similarity.py"
        ),
        # 2. Semantic distance (lines 176-188)
        (
            "```python\nfrom scipy.spatial.distance import cosine\n\ndef semantic_distance(word1, word2, embeddings):",
            "print(f\"cat ↔ car: {semantic_distance('cat', 'car', embeddings):.3f}\")\n```",
            "/code_examples/ch01_foundations/semantic_distance.py"
        ),
        # 3. Vector analogy (lines 196-223)
        (
            "```python\n# The famous example: king - man + woman ≈ queen\n# (Note: This requires embeddings trained on large datasets)\n\ndef vector_analogy(a, b, c, embeddings):",
            "# vector_analogy('swimming', 'swimmer', 'running') → 'runner'\n```",
            "/code_examples/ch01_foundations/vector_analogy.py"
        ),
        # 4. SentenceTransformer example (lines 287-309)
        (
            "```python\nfrom sentence_transformers import SentenceTransformer\n\n# Load a pre-trained model\nmodel = SentenceTransformer('all-MiniLM-L6-v2')",
            "            print(f\"{word1} ↔ {word2}: {similarities[i][j]:.3f}\")\n```",
            "/code_examples/ch01_foundations/sentence_transformer_example.py"
        ),
        # 5. Image embeddings (lines 315-350)
        (
            "```python\nfrom torchvision import models, transforms\nfrom PIL import Image\nimport torch\n\n# Load pre-trained image model",
            "# similarity = cosine_similarity([image1_emb], [image2_emb])\n```",
            "/code_examples/ch01_foundations/image_embeddings_resnet.py"
        ),
        # 6. ProductEmbedder (lines 357-390)
        (
            "```python\nclass ProductEmbedder:\n    \"\"\"Embed products using multiple signals\"\"\"",
            "        return [all_products[i] for i in top_indices]\n```",
            "/code_examples/ch01_foundations/product_embedder.py"
        ),
        # 7. SimpleEmbeddingSearch (lines 396-464)
        (
            "```python\nfrom sentence_transformers import SentenceTransformer\nimport numpy as np\nfrom sklearn.metrics.pairwise import cosine_similarity\n\nclass SimpleEmbeddingSearch:",
            "# the word \"feline\" doesn't appear in any document!\n```",
            "/code_examples/ch01_foundations/simple_embedding_search.py"
        ),
        # 8. EmbeddingROICalculator (lines 863-930)
        (
            "```python\nclass EmbeddingROICalculator:\n    \"\"\"Complete ROI framework for embedding projects\"\"\"",
            "        }\n```",
            "/code_examples/ch01_foundations/embeddingroicalculator.py"
        ),
    ]

    # Apply replacements
    for i, (start, end, include_path) in enumerate(replacements, 1):
        # Find the full code block
        start_idx = content.find(start)
        if start_idx == -1:
            print(f"Warning: Could not find replacement {i} start marker")
            continue

        end_idx = content.find(end, start_idx)
        if end_idx == -1:
            print(f"Warning: Could not find replacement {i} end marker")
            continue

        # Include the end marker in the replacement
        end_idx += len(end)

        # Create the include directive
        include_directive = f"```python\n{{{{< include {include_path} >}}}}\n```"

        # Replace
        content = content[:start_idx] + include_directive + content[end_idx:]
        print(f"✓ Replaced block {i}: {include_path}")

    # Write back
    with open(chapter_file, 'w') as f:
        f.write(content)

    print(f"\n✓ Updated {chapter_file}")

if __name__ == '__main__':
    main()
