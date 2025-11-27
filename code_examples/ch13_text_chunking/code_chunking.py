"""Code-aware chunking that preserves syntactic units."""

import ast
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CodeChunk:
    """A chunk of source code with metadata."""
    code: str
    language: str
    chunk_type: str  # function, class, module, block
    name: Optional[str] = None
    docstring: Optional[str] = None
    start_line: int = 0
    end_line: int = 0


class CodeChunker:
    """
    Chunk source code while preserving syntactic structure.

    Uses AST parsing when available, falls back to regex patterns.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        include_docstrings: bool = True,
        include_imports: bool = True
    ):
        self.chunk_size = chunk_size
        self.include_docstrings = include_docstrings
        self.include_imports = include_imports

    def chunk_python(self, code: str) -> List[CodeChunk]:
        """Chunk Python code using AST parsing."""
        import ast

        try:
            tree = ast.parse(code)
            return self._chunk_python_ast(code, tree)
        except SyntaxError:
            # Fall back to regex-based chunking
            return self._chunk_python_regex(code)

    def _chunk_python_ast(self, code: str, tree: ast.Module) -> List[CodeChunk]:
        """Extract chunks from Python AST."""
        chunks = []
        lines = code.split('\n')

        # Extract imports as a single chunk
        if self.include_imports:
            imports = self._extract_imports(tree, lines)
            if imports:
                chunks.append(CodeChunk(
                    code=imports,
                    language='python',
                    chunk_type='imports',
                    name='imports'
                ))

        # Extract classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                chunk = self._extract_class(node, lines)
                if chunk:
                    chunks.append(chunk)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip methods (they're included with classes)
                if not self._is_method(node, tree):
                    chunk = self._extract_function(node, lines)
                    if chunk:
                        chunks.append(chunk)

        return chunks

    def _extract_imports(self, tree: ast.Module, lines: List[str]) -> str:
        """Extract all import statements."""
        import_lines = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
                import_lines.extend(lines[start:end])

        return '\n'.join(import_lines)

    def _extract_class(self, node: ast.ClassDef, lines: List[str]) -> CodeChunk:
        """Extract a class definition."""
        start = node.lineno - 1
        end = node.end_lineno if hasattr(node, 'end_lineno') else self._find_end(lines, start)

        code = '\n'.join(lines[start:end])
        docstring = ast.get_docstring(node) if self.include_docstrings else None

        return CodeChunk(
            code=code,
            language='python',
            chunk_type='class',
            name=node.name,
            docstring=docstring,
            start_line=start + 1,
            end_line=end
        )

    def _extract_function(self, node, lines: List[str]) -> CodeChunk:
        """Extract a function definition."""
        start = node.lineno - 1
        end = node.end_lineno if hasattr(node, 'end_lineno') else self._find_end(lines, start)

        code = '\n'.join(lines[start:end])
        docstring = ast.get_docstring(node) if self.include_docstrings else None

        return CodeChunk(
            code=code,
            language='python',
            chunk_type='function',
            name=node.name,
            docstring=docstring,
            start_line=start + 1,
            end_line=end
        )

    def _is_method(self, node, tree: ast.Module) -> bool:
        """Check if a function is a method inside a class."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                for child in ast.iter_child_nodes(parent):
                    if child is node:
                        return True
        return False

    def _find_end(self, lines: List[str], start: int) -> int:
        """Find the end of a code block by indentation."""
        if start >= len(lines):
            return start

        # Get the indentation of the definition line
        base_indent = len(lines[start]) - len(lines[start].lstrip())

        for i in range(start + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                indent = len(line) - len(line.lstrip())
                if indent <= base_indent:
                    return i

        return len(lines)

    def _chunk_python_regex(self, code: str) -> List[CodeChunk]:
        """Fallback regex-based chunking for invalid Python."""
        chunks = []

        # Match class definitions
        class_pattern = r'^class\s+(\w+).*?(?=\nclass\s|\ndef\s|\Z)'
        for match in re.finditer(class_pattern, code, re.MULTILINE | re.DOTALL):
            chunks.append(CodeChunk(
                code=match.group(0).strip(),
                language='python',
                chunk_type='class',
                name=match.group(1)
            ))

        # Match function definitions
        func_pattern = r'^def\s+(\w+).*?(?=\ndef\s|\nclass\s|\Z)'
        for match in re.finditer(func_pattern, code, re.MULTILINE | re.DOTALL):
            chunks.append(CodeChunk(
                code=match.group(0).strip(),
                language='python',
                chunk_type='function',
                name=match.group(1)
            ))

        return chunks

    def chunk_javascript(self, code: str) -> List[CodeChunk]:
        """Chunk JavaScript/TypeScript code using regex patterns."""
        chunks = []

        # Match function declarations
        func_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{[^}]*\}'
        for match in re.finditer(func_pattern, code, re.DOTALL):
            chunks.append(CodeChunk(
                code=match.group(0),
                language='javascript',
                chunk_type='function',
                name=match.group(1)
            ))

        # Match arrow functions assigned to variables
        arrow_pattern = r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>\s*\{[^}]*\}'
        for match in re.finditer(arrow_pattern, code, re.DOTALL):
            chunks.append(CodeChunk(
                code=match.group(0),
                language='javascript',
                chunk_type='function',
                name=match.group(1)
            ))

        # Match class declarations
        class_pattern = r'(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{[^}]*\}'
        for match in re.finditer(class_pattern, code, re.DOTALL):
            chunks.append(CodeChunk(
                code=match.group(0),
                language='javascript',
                chunk_type='class',
                name=match.group(1)
            ))

        return chunks


def chunk_code_for_embedding(
    code: str,
    language: str,
    max_tokens: int = 512
) -> List[str]:
    """
    Prepare code chunks for embedding with context.

    Adds language tag and docstring summary for better retrieval.
    """
    chunker = CodeChunker(chunk_size=max_tokens * 4)  # Rough char estimate

    if language == 'python':
        chunks = chunker.chunk_python(code)
    elif language in ('javascript', 'typescript'):
        chunks = chunker.chunk_javascript(code)
    else:
        # Generic chunking for other languages
        return [code]  # Return as single chunk

    # Format for embedding
    formatted = []
    for chunk in chunks:
        text = f"[{language}] {chunk.chunk_type}"
        if chunk.name:
            text += f" {chunk.name}"
        if chunk.docstring:
            text += f"\n{chunk.docstring}"
        text += f"\n\n{chunk.code}"
        formatted.append(text)

    return formatted


# Example usage
if __name__ == "__main__":
    sample_python = '''
import numpy as np
from typing import List

class DataProcessor:
    """Process and transform data for analysis."""

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.config = config

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to zero mean and unit variance."""
        return (data - data.mean()) / data.std()


def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a ** 2 for a in vec1) ** 0.5
    norm2 = sum(b ** 2 for b in vec2) ** 0.5
    return dot / (norm1 * norm2)


async def fetch_embeddings(texts: List[str]) -> List[List[float]]:
    """Fetch embeddings from the API."""
    # Implementation here
    pass
    '''

    chunker = CodeChunker()
    chunks = chunker.chunk_python(sample_python)

    print(f"Found {len(chunks)} code chunks:\n")
    for chunk in chunks:
        print(f"--- {chunk.chunk_type}: {chunk.name} ---")
        print(f"Lines: {chunk.start_line}-{chunk.end_line}")
        if chunk.docstring:
            print(f"Docstring: {chunk.docstring[:50]}...")
        print(f"Code preview: {chunk.code[:100]}...")
        print()
