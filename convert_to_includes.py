#!/usr/bin/env python3
"""
Convert inline Python code blocks to include directives.

This script reads chapter files, finds inline Python code blocks,
matches them to extracted files in code_examples/, and replaces
the inline code with {{< include /code_examples/... >}} directives.
"""

import re
import os
from pathlib import Path
from difflib import SequenceMatcher

def normalize_code(code: str) -> str:
    """Normalize code for comparison by removing extra whitespace and blank lines."""
    lines = [line.rstrip() for line in code.strip().split('\n')]
    # Remove leading/trailing empty lines
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return '\n'.join(lines)

def similarity_ratio(str1: str, str2: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, str1, str2).ratio()

def find_matching_file(code_block: str, code_examples_dir: Path) -> str:
    """
    Find the Python file in code_examples/ that matches the given code block.
    Returns the relative path from project root.
    """
    normalized_block = normalize_code(code_block)

    best_match = None
    best_ratio = 0.0

    # Search all .py files in the code_examples directory
    for py_file in code_examples_dir.glob('*.py'):
        if py_file.name == '__init__.py' or py_file.name.startswith('.'):
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                file_content = normalize_code(f.read())

            # Calculate similarity
            ratio = similarity_ratio(normalized_block, file_content)

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = py_file

        except Exception as e:
            print(f"  Warning: Could not read {py_file}: {e}")
            continue

    if best_match and best_ratio > 0.85:  # At least 85% similarity
        # Return path relative to project root
        rel_path = best_match.relative_to(Path.cwd())
        return f"/{rel_path}"

    return None

def extract_code_blocks(content: str) -> list:
    """
    Extract all Python code blocks from the markdown content.
    Returns list of tuples: (start_pos, end_pos, code_content)
    """
    blocks = []
    pattern = r'```python\n(.*?)```'

    for match in re.finditer(pattern, content, re.DOTALL):
        blocks.append({
            'start': match.start(),
            'end': match.end(),
            'code': match.group(1),
            'full_match': match.group(0)
        })

    return blocks

def convert_chapter(chapter_file: Path, code_examples_dir: Path, dry_run: bool = False) -> int:
    """
    Convert a single chapter file to use include directives.
    Returns the number of conversions made.
    """
    print(f"\nProcessing: {chapter_file.name}")

    # Read chapter content
    with open(chapter_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract code blocks
    code_blocks = extract_code_blocks(content)

    if not code_blocks:
        print(f"  No Python code blocks found")
        return 0

    print(f"  Found {len(code_blocks)} Python code blocks")

    # Process blocks in reverse order to maintain positions
    conversions = 0
    new_content = content

    for i, block in enumerate(reversed(code_blocks), 1):
        block_num = len(code_blocks) - i + 1
        print(f"  Block {block_num}/{len(code_blocks)}: ", end='')

        # Find matching file
        matching_file = find_matching_file(block['code'], code_examples_dir)

        if matching_file:
            print(f"Matched to {matching_file}")

            # Create include directive
            include_directive = f"```python\n{{{{< include {matching_file} >}}}}\n```"

            # Replace in content
            new_content = (
                new_content[:block['start']] +
                include_directive +
                new_content[block['end']:]
            )
            conversions += 1
        else:
            print("No match found (< 85% similarity)")

    # Write back if not dry run and conversions were made
    if conversions > 0 and not dry_run:
        with open(chapter_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  ✓ Converted {conversions} blocks")
    elif conversions > 0:
        print(f"  [DRY RUN] Would convert {conversions} blocks")

    return conversions

def main():
    """Main conversion process."""
    import argparse

    parser = argparse.ArgumentParser(description='Convert inline code to include directives')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--chapter', type=str,
                       help='Convert specific chapter (e.g., ch01_embedding_revolution.qmd)')
    args = parser.parse_args()

    # Setup paths
    project_root = Path.cwd()
    chapters_dir = project_root / 'chapters'
    code_examples_base = project_root / 'code_examples'

    if not chapters_dir.exists():
        print(f"Error: chapters directory not found at {chapters_dir}")
        return

    if not code_examples_base.exists():
        print(f"Error: code_examples directory not found at {code_examples_base}")
        return

    # Get chapter files to process
    if args.chapter:
        chapter_files = [chapters_dir / args.chapter]
        if not chapter_files[0].exists():
            print(f"Error: Chapter file not found: {chapter_files[0]}")
            return
    else:
        chapter_files = sorted(chapters_dir.glob('ch*.qmd'))

    # Skip chapters that are already converted (ch23, ch28)
    skip_chapters = {'ch23_performance_optimization.qmd', 'ch28_implementation_roadmap.qmd'}

    print(f"{'='*60}")
    print(f"Converting inline code to include directives")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"{'='*60}")

    total_conversions = 0

    for chapter_file in chapter_files:
        if chapter_file.name in skip_chapters:
            print(f"\nSkipping {chapter_file.name} (already converted)")
            continue

        # Determine corresponding code_examples directory
        # Map chapter file name to code_examples directory
        chapter_num = chapter_file.stem.split('_')[0]  # e.g., 'ch01'

        # Find matching code_examples directory
        code_dirs = list(code_examples_base.glob(f'{chapter_num}_*'))

        if not code_dirs:
            print(f"\nWarning: No code_examples directory found for {chapter_file.name}")
            continue

        code_examples_dir = code_dirs[0]

        # Convert the chapter
        conversions = convert_chapter(chapter_file, code_examples_dir, dry_run=args.dry_run)
        total_conversions += conversions

    print(f"\n{'='*60}")
    print(f"Total conversions: {total_conversions}")
    if args.dry_run:
        print("This was a DRY RUN - no files were modified")
        print("Run without --dry-run to apply changes")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
