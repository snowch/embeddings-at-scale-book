# Code from Chapter 14
# Book: Embeddings at Scale

"""
Semantic Code Search System

Architecture:
1. Code encoder: Learns code representations (GraphCodeBERT, CodeBERT)
2. Query encoder: Encodes natural language queries
3. Bi-encoder training: Align code and queries in shared space
4. Cross-encoder reranking: Rerank top results with detailed matching

Training data:
- Code-docstring pairs (GitHub, Stack Overflow)
- Code-comment pairs
- Query-code pairs (synthetic and real)

Applications:
- Code completion (GitHub Copilot)
- API discovery (find relevant functions)
- Bug detection (find similar known bugs)
- Code review (find similar code for patterns)
"""

import ast
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class CodeSnippet:
    """
    Code snippet with metadata

    Attributes:
        code_id: Unique identifier
        code: Source code
        language: Programming language
        docstring: Function/class docstring
        function_name: Function name (if applicable)
        file_path: Source file path
        embedding: Cached embedding
    """
    code_id: str
    code: str
    language: str
    docstring: Optional[str] = None
    function_name: Optional[str] = None
    file_path: Optional[str] = None
    embedding: Optional[np.ndarray] = None

class CodeEncoder(nn.Module):
    """
    Code encoder using transformer architecture

    Architecture:
    - Tokenize code (BPE or subword tokenization)
    - Transformer encoder (GraphCodeBERT or CodeBERT)
    - Pool to fixed-size embedding

    In production: Use pre-trained CodeBERT or GraphCodeBERT
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 512,
        hidden_dim: int = 768,
        num_layers: int = 6
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection to embedding space
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode code to embeddings

        Args:
            token_ids: Token IDs (batch_size, seq_len)

        Returns:
            Code embeddings (batch_size, embedding_dim)
        """
        # Embed tokens
        x = self.token_embedding(token_ids)

        # Encode with transformer
        x = self.transformer(x)

        # Pool: Mean pooling over sequence
        x = torch.mean(x, dim=1)

        # Project to embedding space
        x = self.projection(x)

        # Normalize
        x = F.normalize(x, p=2, dim=1)

        return x

class CodeSearchEngine:
    """
    Semantic code search engine

    Capabilities:
    - Natural language queries: "sort a list in descending order"
    - Code-to-code search: Find similar implementations
    - API discovery: Find relevant functions/classes
    - Cross-language search: Query in English, find in any language

    Architecture:
    - Bi-encoder: Separate encoders for queries and code
    - Vector index: HNSW index for fast retrieval
    - Reranker: Cross-encoder for top-k refinement

    Production scale:
    - 100M+ code snippets (all of GitHub)
    - Sub-second query latency
    - Incremental indexing (new code added continuously)
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        device: str = 'cuda'
    ):
        """
        Args:
            embedding_dim: Embedding dimension
            device: Device for computation
        """
        self.embedding_dim = embedding_dim
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Initialize encoders
        self.code_encoder = CodeEncoder(embedding_dim=embedding_dim).to(self.device)
        self.query_encoder = TextEncoder(embedding_dim=embedding_dim).to(self.device)

        # Set to eval mode
        self.code_encoder.eval()
        self.query_encoder.eval()

        # Code index: code_id -> CodeSnippet
        self.code_snippets: Dict[str, CodeSnippet] = {}

        # Embedding index: (code_ids, embeddings)
        self.code_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

        print(f"Initialized Code Search Engine")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Device: {self.device}")

    def tokenize_code(self, code: str) -> List[str]:
        """
        Tokenize code into tokens

        Strategies:
        - Split on whitespace and operators
        - Preserve identifiers and keywords
        - Handle language-specific syntax

        In production: Use language-specific tokenizers (tree-sitter)

        Args:
            code: Source code

        Returns:
            List of tokens
        """
        # Simple tokenization (split on whitespace and operators)
        # In production: Use tree-sitter or language-specific parsers

        # Replace operators with spaces
        operators = r'[+\-*/%=<>!&|^~(){}\[\];:,.]'
        code_spaced = re.sub(operators, ' ', code)

        # Split on whitespace
        tokens = code_spaced.split()

        # Filter empty tokens
        tokens = [t for t in tokens if t.strip()]

        return tokens

    def extract_functions(self, code: str, language: str = 'python') -> List[CodeSnippet]:
        """
        Extract functions from source code

        Uses AST parsing to extract function definitions with docstrings

        Args:
            code: Source code
            language: Programming language

        Returns:
            List of code snippets (one per function)
        """
        snippets = []

        if language == 'python':
            try:
                tree = ast.parse(code)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Extract function code
                        function_code = ast.unparse(node)

                        # Extract docstring
                        docstring = ast.get_docstring(node)

                        # Create snippet
                        snippet = CodeSnippet(
                            code_id=f"func_{node.name}_{node.lineno}",
                            code=function_code,
                            language='python',
                            docstring=docstring,
                            function_name=node.name
                        )
                        snippets.append(snippet)

            except SyntaxError:
                # If parsing fails, treat entire code as one snippet
                snippet = CodeSnippet(
                    code_id='code_0',
                    code=code,
                    language=language
                )
                snippets.append(snippet)
        else:
            # For other languages, treat as single snippet
            snippet = CodeSnippet(
                code_id='code_0',
                code=code,
                language=language
            )
            snippets.append(snippet)

        return snippets

    def encode_code(self, code_snippets: List[CodeSnippet]) -> np.ndarray:
        """
        Encode code snippets to embeddings

        Args:
            code_snippets: Code snippets to encode

        Returns:
            Embeddings (len(code_snippets), embedding_dim)
        """
        # Tokenize code
        tokenized = []
        for snippet in code_snippets:
            # Combine code and docstring
            text = snippet.code
            if snippet.docstring:
                text = f"{snippet.docstring}\n{text}"

            tokens = self.tokenize_code(text)

            # Convert to token IDs (hash-based for demo)
            max_len = 512
            token_ids = [hash(token) % 50000 for token in tokens[:max_len]]
            # Pad
            token_ids = token_ids + [0] * (max_len - len(token_ids))
            tokenized.append(token_ids)

        token_ids = torch.tensor(tokenized, dtype=torch.long).to(self.device)

        # Encode
        with torch.no_grad():
            embeddings = self.code_encoder(token_ids)

        return embeddings.cpu().numpy()

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode natural language query

        Args:
            query: Query string

        Returns:
            Query embedding (embedding_dim,)
        """
        # Tokenize query (same as TextEncoder)
        max_len = 77
        tokens = query.lower().split()[:max_len]
        token_ids = [hash(word) % 50000 for word in tokens]
        token_ids = token_ids + [0] * (max_len - len(token_ids))

        token_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)

        # Encode
        with torch.no_grad():
            embedding = self.query_encoder(token_ids)

        return embedding.cpu().numpy()[0]

    def index_code(self, code_snippets: List[CodeSnippet]):
        """
        Index code snippets for search

        Args:
            code_snippets: Code snippets to index
        """
        print(f"Indexing {len(code_snippets)} code snippets...")

        # Encode code
        embeddings = self.encode_code(code_snippets)

        # Update index
        for snippet, embedding in zip(code_snippets, embeddings):
            snippet.embedding = embedding
            self.code_snippets[snippet.code_id] = snippet
            self.code_ids.append(snippet.code_id)

        # Stack embeddings
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        print(f"✓ Indexed {len(code_snippets)} snippets")
        print(f"  Total index size: {len(self.code_ids)}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        language_filter: Optional[str] = None
    ) -> List[Tuple[CodeSnippet, float]]:
        """
        Search for code using natural language query

        Args:
            query: Natural language query
            top_k: Number of results
            language_filter: Filter by language (optional)

        Returns:
            List of (code_snippet, score) tuples
        """
        # Encode query
        query_embedding = self.encode_query(query)

        # Compute similarities
        scores = np.dot(self.embeddings, query_embedding)

        # Rank results
        ranked_indices = np.argsort(scores)[::-1]

        # Filter by language if specified
        results = []
        for idx in ranked_indices:
            code_id = self.code_ids[idx]
            snippet = self.code_snippets[code_id]
            score = scores[idx]

            # Language filter
            if language_filter and snippet.language != language_filter:
                continue

            results.append((snippet, score))

            if len(results) >= top_k:
                break

        return results

# Example: Code search for Python repository
def code_search_example():
    """
    Semantic code search over Python codebase

    Use cases:
    - Find sorting implementations
    - Find API usage examples
    - Find similar bug patterns
    - Discover relevant functions

    Scale: 1M+ Python functions
    """

    # Initialize search engine
    engine = CodeSearchEngine(embedding_dim=512)

    # Sample code repository
    sample_code = [
        '''
def bubble_sort(arr):
    """Sort a list using bubble sort algorithm"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
''',
        '''
def quick_sort(arr):
    """Sort a list using quick sort algorithm"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
''',
        '''
def binary_search(arr, target):
    """Search for target in sorted array using binary search"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
'''
    ]

    # Extract and index functions
    all_snippets = []
    for code in sample_code:
        snippets = engine.extract_functions(code, language='python')
        all_snippets.extend(snippets)

    engine.index_code(all_snippets)

    # Search: Find sorting implementations
    print("\n=== Query: 'sort a list' ===")
    results = engine.search('sort a list', top_k=3)

    for snippet, score in results:
        print(f"\nFunction: {snippet.function_name} (score: {score:.3f})")
        print(f"Docstring: {snippet.docstring}")
        print(f"Code preview: {snippet.code[:100]}...")

    # Search: Find search algorithms
    print("\n\n=== Query: 'find an element in array' ===")
    results = engine.search('find an element in array', top_k=3)

    for snippet, score in results:
        print(f"\nFunction: {snippet.function_name} (score: {score:.3f})")
        print(f"Docstring: {snippet.docstring}")
        print(f"Code preview: {snippet.code[:100]}...")

# Uncomment to run:
# code_search_example()
