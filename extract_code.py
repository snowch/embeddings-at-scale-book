#!/usr/bin/env python3
"""
Extract Python code blocks from book chapters and organize into runnable examples.
"""

import re
import os
from pathlib import Path
from collections import defaultdict

# Chapter mapping
CHAPTER_INFO = {
    'ch01': 'foundations',
    'ch02': 'strategic_architecture',
    'ch03': 'vector_database_fundamentals',
    'ch04': 'custom_embedding_strategies',
    'ch05': 'contrastive_learning',
    'ch06': 'siamese_networks',
    'ch07': 'self_supervised_learning',
    'ch08': 'advanced_embedding_techniques',
    'ch09': 'embedding_pipeline_engineering',
    'ch10': 'scaling_embedding_training',
    'ch11': 'high_performance_vector_ops',
    'ch12': 'data_engineering',
    'ch13': 'rag_at_scale',
    'ch14': 'semantic_search',
    'ch15': 'recommendation_systems',
    'ch16': 'anomaly_detection_security',
    'ch17': 'automated_decision_systems',
    'ch18': 'financial_services',
    'ch19': 'healthcare_life_sciences',
    'ch20': 'retail_ecommerce',
    'ch21': 'manufacturing_industry40',
    'ch22': 'media_entertainment',
    'ch23': 'performance_optimization',
    'ch24': 'security_privacy',
    'ch25': 'monitoring_observability',
    'ch26': 'future_trends',
    'ch27': 'organizational_transformation',
    'ch28': 'implementation_roadmap',
    'ch29': 'case_studies',
    'ch30': 'path_forward'
}

def extract_code_blocks(file_path):
    """Extract all Python code blocks from a markdown/qmd file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all ```python ... ``` blocks
    pattern = r'```python\s*\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)

    code_blocks = []
    for match in matches:
        # Clean up the code
        code = match.strip()
        # Skip tiny snippets (< 5 lines)
        if len(code.split('\n')) >= 5:
            code_blocks.append(code)

    return code_blocks

def extract_imports(code):
    """Extract import statements from code."""
    imports = set()
    for line in code.split('\n'):
        line = line.strip()
        if line.startswith('import ') or line.startswith('from '):
            # Extract the base package name
            if line.startswith('import '):
                pkg = line.replace('import ', '').split()[0].split('.')[0]
            else:
                pkg = line.replace('from ', '').split()[0].split('.')[0]
            imports.add(pkg)
    return imports

def generate_filename(code, chapter_num, block_num):
    """Generate a descriptive filename for the code block."""
    # Try to extract class name
    class_match = re.search(r'class\s+(\w+)', code)
    if class_match:
        return f"{class_match.group(1).lower()}.py"

    # Try to extract function name
    func_match = re.search(r'def\s+(\w+)', code)
    if func_match:
        func_name = func_match.group(1)
        if func_name != '__init__' and not func_name.startswith('_'):
            return f"{func_name}.py"

    # Default to numbered file
    return f"example_{block_num:02d}.py"

def process_chapter(chapter_file, output_dir):
    """Process a single chapter file."""
    print(f"Processing {chapter_file.name}...")

    code_blocks = extract_code_blocks(chapter_file)

    if not code_blocks:
        print(f"  No substantial code blocks found")
        return [], set()

    print(f"  Found {len(code_blocks)} code blocks")

    # Extract chapter number
    chapter_match = re.match(r'ch(\d+)_', chapter_file.name)
    if not chapter_match:
        return [], set()

    chapter_num = chapter_match.group(1)

    files_created = []
    all_imports = set()

    for i, code in enumerate(code_blocks, 1):
        # Generate filename
        filename = generate_filename(code, chapter_num, i)

        # Avoid duplicates
        base_filename = filename
        counter = 1
        while filename in [f[1] for f in files_created]:
            name, ext = os.path.splitext(base_filename)
            filename = f"{name}_{counter}{ext}"
            counter += 1

        # Write code to file
        output_file = output_dir / filename
        with open(output_file, 'w') as f:
            f.write(f"# Code from Chapter {chapter_num}\n")
            f.write(f"# Book: Embeddings at Scale\n\n")
            f.write(code)
            f.write("\n")

        files_created.append((str(output_file), filename))

        # Extract imports
        imports = extract_imports(code)
        all_imports.update(imports)

        print(f"    Created: {filename}")

    return files_created, all_imports

def main():
    """Main extraction function."""
    base_dir = Path('/home/user/embeddings-at-scale-book')
    chapters_dir = base_dir / 'chapters'
    code_dir = base_dir / 'code_examples'

    all_imports = set()
    chapter_files = {}
    total_code_files = 0
    total_lines = 0

    # Process each chapter
    for ch_num in range(1, 31):
        ch_prefix = f"ch{ch_num:02d}"
        ch_name = CHAPTER_INFO.get(ch_prefix, f"chapter_{ch_num}")

        # Find chapter file
        chapter_files_list = list(chapters_dir.glob(f"{ch_prefix}_*.qmd"))
        if not chapter_files_list:
            print(f"Warning: No file found for {ch_prefix}")
            continue

        chapter_file = chapter_files_list[0]

        # Create chapter output directory
        ch_output_dir = code_dir / f"{ch_prefix}_{ch_name}"
        ch_output_dir.mkdir(parents=True, exist_ok=True)

        # Process chapter
        files, imports = process_chapter(chapter_file, ch_output_dir)

        if files:
            chapter_files[ch_prefix] = {
                'name': ch_name,
                'files': files,
                'imports': imports
            }

            all_imports.update(imports)
            total_code_files += len(files)

            # Count lines
            for file_path, _ in files:
                with open(file_path, 'r') as f:
                    total_lines += len(f.readlines())

    # Generate summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total chapters processed: {len(chapter_files)}")
    print(f"Total code files created: {total_code_files}")
    print(f"Total lines of code: {total_lines}")
    print(f"Unique imports found: {len(all_imports)}")
    print("\nImports by chapter:")
    for ch_prefix in sorted(chapter_files.keys()):
        info = chapter_files[ch_prefix]
        print(f"  {ch_prefix}: {len(info['files'])} files, {len(info['imports'])} unique imports")

    # Save metadata
    import json
    metadata = {
        'total_files': total_code_files,
        'total_lines': total_lines,
        'chapters': {
            ch: {
                'name': info['name'],
                'file_count': len(info['files']),
                'files': [f[1] for f in info['files']],
                'imports': sorted(list(info['imports']))
            }
            for ch, info in chapter_files.items()
        },
        'all_imports': sorted(list(all_imports))
    }

    with open(code_dir / 'extraction_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nMetadata saved to extraction_metadata.json")

    return metadata

if __name__ == '__main__':
    main()
