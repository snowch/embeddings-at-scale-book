#!/usr/bin/env python3
"""
Create documentation and requirements for the code examples.
"""

import json
from pathlib import Path

# Map imports to PyPI package names with versions
PACKAGE_MAPPING = {
    'torch': 'torch>=2.0.0',
    'transformers': 'transformers>=4.30.0',
    'sentence_transformers': 'sentence-transformers>=2.2.0',
    'faiss': 'faiss-cpu>=1.7.4',  # or faiss-gpu
    'numpy': 'numpy>=1.24.0',
    'pandas': 'pandas>=2.0.0',
    'sklearn': 'scikit-learn>=1.3.0',
    'matplotlib': 'matplotlib>=3.7.0',
    'scipy': 'scipy>=1.10.0',
    'torchvision': 'torchvision>=0.15.0',
    'openai': 'openai>=1.0.0',
    'nltk': 'nltk>=3.8.0',
    'gensim': 'gensim>=4.3.0',
    'datasets': 'datasets>=2.14.0',
    'tokenizers': 'tokenizers>=0.13.0',
    'PIL': 'Pillow>=10.0.0',
    'bitsandbytes': 'bitsandbytes>=0.41.0',
    'cryptography': 'cryptography>=41.0.0',
    'dwave': 'dwave-ocean-sdk>=6.0.0',  # Quantum computing
}

# Chapter descriptions
CHAPTER_DESCRIPTIONS = {
    'ch01': 'The Embedding Revolution - Fundamental examples demonstrating embeddings vs traditional approaches',
    'ch02': 'Strategic Architecture - Multi-modal embeddings, governance, cost optimization patterns',
    'ch03': 'Vector Database Fundamentals - HNSW, IVF-PQ, sharding, and trillion-scale indexing',
    'ch04': 'Custom Embedding Strategies - Fine-tuning, multi-objective design, dimensionality optimization',
    'ch05': 'Contrastive Learning - InfoNCE, SimCLR, MoCo, hard negative mining',
    'ch06': 'Siamese Networks - One-shot learning, triplet loss, threshold calibration',
    'ch07': 'Self-Supervised Learning - Masked language models, autoencoders, domain-specific pre-training',
    'ch08': 'Advanced Techniques - Hyperbolic embeddings, dynamic embeddings, federated learning',
    'ch09': 'Pipeline Engineering - Hybrid systems, deployment strategies, production patterns',
    'ch10': 'Scaling Training - Distributed training, gradient accumulation, memory optimization',
    'ch11': 'High Performance Vector Operations - GPU acceleration, memory-mapped stores',
    'ch12': 'Data Engineering - Data pipelines, quality monitoring, versioning',
    'ch13': 'RAG at Scale - Passage extraction, multi-stage retrieval, production RAG',
    'ch14': 'Semantic Search - Cross-modal search, query understanding, ranking',
    'ch15': 'Recommendation Systems - Embedding-based recommendations, cross-domain transfer',
    'ch16': 'Anomaly Detection & Security - Behavioral anomalies, security applications',
    'ch17': 'Automated Decision Systems - Embedding-driven automation',
    'ch18': 'Financial Services - Finance-specific embedding applications',
    'ch19': 'Healthcare & Life Sciences - Medical and healthcare embedding use cases',
    'ch20': 'Retail & E-commerce - Product embeddings, style matching, demand forecasting',
    'ch21': 'Manufacturing & Industry 4.0 - Industrial IoT, predictive maintenance',
    'ch22': 'Media & Entertainment - Content embeddings, recommendation',
    'ch23': 'Performance Optimization - Caching, quantization, hardware acceleration',
    'ch24': 'Security & Privacy - Differential privacy, secure aggregation, access control',
    'ch25': 'Monitoring & Observability - Metrics, drift detection, quality monitoring',
    'ch26': 'Future Trends - Quantum embeddings, neuromorphic computing, blockchain',
    'ch27': 'Organizational Transformation - Capability development, governance frameworks',
    'ch28': 'Implementation Roadmap - Deployment stages, technology selection',
    'ch30': 'Path Forward - Innovation stages, partnership models, transformation frameworks',
}

def create_requirements_txt(metadata, output_path):
    """Create requirements.txt from metadata."""
    all_imports = metadata['all_imports']

    # Filter to third-party packages and map to PyPI names
    requirements = []
    for imp in sorted(all_imports):
        if imp in PACKAGE_MAPPING:
            requirements.append(PACKAGE_MAPPING[imp])

    # Add some common dependencies that might not be explicitly imported
    additional = [
        'ipython>=8.12.0',  # For interactive use
        'jupyter>=1.0.0',  # For notebooks
        'pytest>=7.4.0',  # For testing
    ]

    all_requirements = sorted(set(requirements + additional))

    with open(output_path, 'w') as f:
        f.write("# Python dependencies for Embeddings at Scale code examples\n")
        f.write("# Install with: pip install -r requirements.txt\n\n")
        f.write("# Core ML & Deep Learning\n")
        for req in all_requirements:
            f.write(f"{req}\n")

    print(f"Created requirements.txt with {len(all_requirements)} packages")

def create_chapter_readme(chapter_prefix, chapter_info, output_dir):
    """Create README.md for a chapter directory."""
    name = chapter_info['name']
    files = chapter_info['files']
    imports = chapter_info['imports']

    readme_path = output_dir / 'README.md'

    # Get chapter description
    description = CHAPTER_DESCRIPTIONS.get(chapter_prefix, 'Code examples from this chapter')

    with open(readme_path, 'w') as f:
        # Title
        chapter_num = int(chapter_prefix[2:])
        f.write(f"# Chapter {chapter_num}: {name.replace('_', ' ').title()}\n\n")

        # Description
        f.write(f"{description}\n\n")

        # Code files
        f.write(f"## Code Examples ({len(files)} files)\n\n")

        # Group files by type
        classes = []
        functions = []
        enums = []
        examples = []

        for fname in sorted(files):
            if fname.startswith('example_') or fname.startswith('from') or fname.startswith('import'):
                examples.append(fname)
            elif any(x in fname.lower() for x in ['type', 'status', 'level', 'category', 'stage', 'model']):
                enums.append(fname)
            else:
                classes.append(fname)

        if classes:
            f.write("### Main Classes & Systems\n\n")
            for fname in classes:
                # Generate description from filename
                name_clean = fname.replace('.py', '').replace('_', ' ').title()
                f.write(f"- `{fname}` - {name_clean}\n")
            f.write("\n")

        if enums:
            f.write("### Data Models & Enums\n\n")
            for fname in enums:
                name_clean = fname.replace('.py', '').replace('_', ' ').title()
                f.write(f"- `{fname}` - {name_clean}\n")
            f.write("\n")

        if examples:
            f.write("### Examples & Utilities\n\n")
            for fname in examples:
                name_clean = fname.replace('.py', '').replace('_', ' ').title()
                f.write(f"- `{fname}` - {name_clean}\n")
            f.write("\n")

        # Dependencies
        if imports:
            f.write(f"## Dependencies\n\n")
            f.write("Key Python packages used in this chapter:\n\n")
            for imp in sorted(imports):
                if imp in PACKAGE_MAPPING:
                    f.write(f"- `{imp}` → {PACKAGE_MAPPING[imp]}\n")
                else:
                    f.write(f"- `{imp}` (standard library)\n")
            f.write("\n")

        # Usage note
        f.write("## Usage Notes\n\n")
        f.write("Most code examples in this chapter are **illustrative** and designed to demonstrate concepts. ")
        f.write("Some examples may require:\n\n")
        f.write("- Synthetic or sample data (not included)\n")
        f.write("- Pre-trained models from HuggingFace\n")
        f.write("- GPU for efficient execution\n")
        f.write("- Additional setup or configuration\n\n")

        f.write("Refer to the book chapter for full context and explanations.\n")

    print(f"Created README for {chapter_prefix}")

def create_master_readme(metadata, output_path):
    """Create master README.md."""
    with open(output_path, 'w') as f:
        f.write("# Embeddings at Scale - Code Examples\n\n")

        f.write("This repository contains all Python code examples from the book *Embeddings at Scale*. ")
        f.write(f"The examples are organized by chapter, with **{metadata['total_files']} code files** ")
        f.write(f"totaling **{metadata['total_lines']:,} lines of code**.\n\n")

        # Installation
        f.write("## Installation\n\n")
        f.write("### Prerequisites\n\n")
        f.write("- Python 3.9 or higher\n")
        f.write("- pip or conda package manager\n")
        f.write("- (Optional) GPU with CUDA for deep learning examples\n\n")

        f.write("### Install Dependencies\n\n")
        f.write("```bash\n")
        f.write("# Install all dependencies\n")
        f.write("pip install -r requirements.txt\n\n")
        f.write("# For GPU support (instead of faiss-cpu):\n")
        f.write("pip install faiss-gpu>=1.7.4\n")
        f.write("```\n\n")

        # Structure
        f.write("## Repository Structure\n\n")
        f.write("```\n")
        f.write("code_examples/\n")
        f.write("├── requirements.txt          # All Python dependencies\n")
        f.write("├── README.md                 # This file\n")

        chapters = sorted(metadata['chapters'].keys())
        for ch in chapters:
            info = metadata['chapters'][ch]
            f.write(f"├── {ch}_{info['name']}/\n")
            f.write(f"│   ├── README.md            # Chapter-specific documentation\n")
            f.write(f"│   └── *.py                 # {info['file_count']} code files\n")

        f.write("```\n\n")

        # Chapters
        f.write("## Chapters Overview\n\n")

        # Group chapters by section
        f.write("### Part I: Foundations (Chapters 1-4)\n\n")
        for ch in ['ch01', 'ch02', 'ch03', 'ch04']:
            if ch in metadata['chapters']:
                info = metadata['chapters'][ch]
                num = int(ch[2:])
                desc = CHAPTER_DESCRIPTIONS.get(ch, '')
                f.write(f"**Chapter {num}**: {desc}\n")
                f.write(f"- {info['file_count']} code files in `{ch}_{info['name']}/`\n\n")

        f.write("### Part II: Training Techniques (Chapters 5-10)\n\n")
        for ch in ['ch05', 'ch06', 'ch07', 'ch08', 'ch09', 'ch10']:
            if ch in metadata['chapters']:
                info = metadata['chapters'][ch]
                num = int(ch[2:])
                desc = CHAPTER_DESCRIPTIONS.get(ch, '')
                f.write(f"**Chapter {num}**: {desc}\n")
                f.write(f"- {info['file_count']} code files in `{ch}_{info['name']}/`\n\n")

        f.write("### Part III: Infrastructure & Operations (Chapters 11-12)\n\n")
        for ch in ['ch11', 'ch12']:
            if ch in metadata['chapters']:
                info = metadata['chapters'][ch]
                num = int(ch[2:])
                desc = CHAPTER_DESCRIPTIONS.get(ch, '')
                f.write(f"**Chapter {num}**: {desc}\n")
                f.write(f"- {info['file_count']} code files in `{ch}_{info['name']}/`\n\n")

        f.write("### Part IV: Applications (Chapters 13-22)\n\n")
        for ch in ['ch13', 'ch14', 'ch15', 'ch16', 'ch17', 'ch18', 'ch19', 'ch20', 'ch21', 'ch22']:
            if ch in metadata['chapters']:
                info = metadata['chapters'][ch]
                num = int(ch[2:])
                desc = CHAPTER_DESCRIPTIONS.get(ch, '')
                f.write(f"**Chapter {num}**: {desc}\n")
                f.write(f"- {info['file_count']} code files in `{ch}_{info['name']}/`\n\n")

        f.write("### Part V: Production & Future (Chapters 23-30)\n\n")
        for ch in ['ch23', 'ch24', 'ch25', 'ch26', 'ch27', 'ch28', 'ch30']:
            if ch in metadata['chapters']:
                info = metadata['chapters'][ch]
                num = int(ch[2:])
                desc = CHAPTER_DESCRIPTIONS.get(ch, '')
                f.write(f"**Chapter {num}**: {desc}\n")
                f.write(f"- {info['file_count']} code files in `{ch}_{info['name']}/`\n\n")

        # Usage guide
        f.write("## How to Use These Examples\n\n")

        f.write("### 1. Complete, Runnable Examples\n\n")
        f.write("Most examples in Chapters 4-10 are **complete and runnable** (with proper setup):\n\n")
        f.write("```bash\n")
        f.write("cd ch05_contrastive_learning\n")
        f.write("python infonceloss.py  # Requires synthetic data or modifications\n")
        f.write("```\n\n")

        f.write("### 2. Illustrative Code Snippets\n\n")
        f.write("Some examples are **illustrative patterns** showing best practices:\n\n")
        f.write("- Architecture patterns (Chapter 2, 3)\n")
        f.write("- Production deployment patterns (Chapter 6, 9, 23)\n")
        f.write("- Domain-specific applications (Chapters 13-22)\n\n")

        f.write("### 3. Framework Examples\n\n")
        f.write("Framework code provides **reusable components** for building systems:\n\n")
        f.write("- `ch02_strategic_architecture/embeddingdatagovernance.py`\n")
        f.write("- `ch04_custom_embedding_strategies/embeddingtco.py`\n")
        f.write("- `ch06_siamese_networks/thresholdcalibrator.py`\n\n")

        # Important notes
        f.write("## Important Notes\n\n")

        f.write("### Data Requirements\n\n")
        f.write("Most code examples assume you have:\n\n")
        f.write("- Training data (text, images, or structured data)\n")
        f.write("- Pre-trained models (from HuggingFace or similar)\n")
        f.write("- Vector databases (FAISS, Qdrant, Weaviate, etc.)\n\n")

        f.write("You will need to:\n\n")
        f.write("1. **Replace placeholder data** with your actual datasets\n")
        f.write("2. **Download pre-trained models** (many use HuggingFace models)\n")
        f.write("3. **Configure paths** and parameters for your environment\n\n")

        f.write("### GPU vs CPU\n\n")
        f.write("- **Deep learning examples** (Ch 5-8, 10) benefit from GPU\n")
        f.write("- **Vector operations** (Ch 3, 11) are much faster on GPU\n")
        f.write("- **Small examples** can run on CPU\n\n")

        f.write("For CPU-only:\n")
        f.write("```bash\n")
        f.write("# Most PyTorch code will fallback to CPU automatically\n")
        f.write("# Use faiss-cpu instead of faiss-gpu\n")
        f.write("```\n\n")

        # Key technologies
        f.write("## Key Technologies\n\n")

        tech_groups = {
            'Deep Learning Frameworks': ['torch', 'transformers', 'sentence_transformers', 'torchvision'],
            'Vector Databases & Search': ['faiss', 'numpy', 'sklearn'],
            'Data Processing': ['pandas', 'datasets', 'tokenizers'],
            'Visualization & Analysis': ['matplotlib', 'scipy'],
            'NLP': ['nltk', 'gensim', 'transformers'],
            'APIs & Services': ['openai'],
        }

        for category, packages in tech_groups.items():
            f.write(f"**{category}**:\n")
            for pkg in packages:
                if pkg in PACKAGE_MAPPING:
                    f.write(f"- {PACKAGE_MAPPING[pkg]}\n")
            f.write("\n")

        # Contributing
        f.write("## Code Completeness Assessment\n\n")
        f.write("| Category | Chapters | Completeness | Notes |\n")
        f.write("|----------|----------|--------------|-------|\n")
        f.write("| Foundations | 1-3 | Illustrative | Demonstrates concepts, needs data |\n")
        f.write("| Custom Embeddings | 4 | High | Most examples runnable with setup |\n")
        f.write("| Training | 5-8 | High | Complete training loops, needs data |\n")
        f.write("| Siamese Networks | 6 | Very High | Production-ready patterns |\n")
        f.write("| Scaling | 9-10 | Medium | Requires distributed setup |\n")
        f.write("| Infrastructure | 11-12 | Medium | Patterns and frameworks |\n")
        f.write("| Applications | 13-22 | Illustrative | Domain-specific examples |\n")
        f.write("| Production | 23-25 | Medium | Monitoring and optimization patterns |\n")
        f.write("| Future | 26-30 | Conceptual | Emerging technologies |\n\n")

        # License and attribution
        f.write("## License\n\n")
        f.write("These code examples are from the book *Embeddings at Scale*. ")
        f.write("Please refer to the book for full context and explanations.\n\n")

        f.write("## Getting Help\n\n")
        f.write("For questions about the code:\n\n")
        f.write("1. **Read the corresponding book chapter** for full context\n")
        f.write("2. **Check the chapter README** for specific setup instructions\n")
        f.write("3. **Review the code comments** for implementation details\n\n")

        f.write("## Quick Start Examples\n\n")
        f.write("### Example 1: Fine-tune Sentence Embeddings\n\n")
        f.write("```python\n")
        f.write("# From ch04_custom_embedding_strategies/embeddingfinetuner.py\n")
        f.write("from sentence_transformers import SentenceTransformer\n\n")
        f.write("# Load base model\n")
        f.write("model = SentenceTransformer('all-mpnet-base-v2')\n\n")
        f.write("# Fine-tune on your domain data\n")
        f.write("# (See embeddingfinetuner.py for complete example)\n")
        f.write("```\n\n")

        f.write("### Example 2: Train Siamese Network\n\n")
        f.write("```python\n")
        f.write("# From ch06_siamese_networks/siamesenetwork.py\n")
        f.write("import torch\n")
        f.write("import torch.nn as nn\n\n")
        f.write("# Create Siamese network\n")
        f.write("# (See siamesenetwork.py for complete architecture)\n")
        f.write("```\n\n")

        f.write("### Example 3: Contrastive Learning\n\n")
        f.write("```python\n")
        f.write("# From ch05_contrastive_learning/infonceloss.py\n")
        f.write("# Implement InfoNCE loss for self-supervised learning\n")
        f.write("# (See infonceloss.py for complete implementation)\n")
        f.write("```\n\n")

    print(f"Created master README.md")

def main():
    """Main documentation generation."""
    base_dir = Path('/home/user/embeddings-at-scale-book/code_examples')

    # Load metadata
    with open(base_dir / 'extraction_metadata.json', 'r') as f:
        metadata = json.load(f)

    # Create requirements.txt
    create_requirements_txt(metadata, base_dir / 'requirements.txt')

    # Create chapter READMEs
    for chapter_prefix, chapter_info in metadata['chapters'].items():
        chapter_dir = base_dir / f"{chapter_prefix}_{chapter_info['name']}"
        if chapter_dir.exists():
            create_chapter_readme(chapter_prefix, chapter_info, chapter_dir)

    # Create master README
    create_master_readme(metadata, base_dir / 'README.md')

    print("\n" + "="*60)
    print("Documentation generation complete!")
    print("="*60)
    print(f"Created:")
    print(f"  - requirements.txt")
    print(f"  - README.md (master)")
    print(f"  - {len(metadata['chapters'])} chapter READMEs")

if __name__ == '__main__':
    main()
