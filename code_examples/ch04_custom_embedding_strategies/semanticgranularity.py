# Code from Chapter 04
# Book: Embeddings at Scale

class SemanticGranularity:
    """
    Examples of semantic granularity requirements across domains
    """

    COARSE = {
        'name': 'Coarse-grained',
        'example': 'News article categorization',
        'requirement': 'Distinguish broad topics (sports vs. politics vs. technology)',
        'embedding_dim': '128-256 sufficient',
        'training_data': '10K-100K examples'
    }

    MEDIUM = {
        'name': 'Medium-grained',
        'example': 'E-commerce product search',
        'requirement': 'Distinguish product types and attributes (running shoes vs. hiking boots)',
        'embedding_dim': '256-512 recommended',
        'training_data': '100K-1M examples'
    }

    FINE = {
        'name': 'Fine-grained',
        'example': 'Legal document retrieval',
        'requirement': 'Distinguish subtle legal distinctions (contract types, precedent applicability)',
        'embedding_dim': '512-768 recommended',
        'training_data': '1M-10M examples'
    }

    ULTRA_FINE = {
        'name': 'Ultra-fine',
        'example': 'Molecular drug discovery',
        'requirement': 'Distinguish molecules with minor structural differences that dramatically affect properties',
        'embedding_dim': '768-1024+ required',
        'training_data': '10M+ examples or sophisticated augmentation'
    }
