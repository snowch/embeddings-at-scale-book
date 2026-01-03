#!/usr/bin/env python3
"""Generate sitemap.xml for the Embeddings at Scale book."""

import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


def extract_chapters_from_config(config_path: Path) -> List[str]:
    """Extract all chapter URLs from _quarto.yml."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    urls = set()  # Use set to avoid duplicates

    # Extract chapters from book structure
    book_config = config.get('book', {})
    chapters = book_config.get('chapters', [])

    for item in chapters:
        if isinstance(item, str):
            # Direct chapter reference
            if item.endswith('.qmd'):
                urls.add(item.replace('.qmd', '.html'))
        elif isinstance(item, dict):
            # Part with chapters
            if 'chapters' in item:
                for chapter in item['chapters']:
                    if chapter.endswith('.qmd'):
                        urls.add(chapter.replace('.qmd', '.html'))

    # Add appendices
    appendices = book_config.get('appendices', [])
    for appendix in appendices:
        if appendix.endswith('.qmd'):
            urls.add(appendix.replace('.qmd', '.html'))

    # Convert to sorted list (index.html first, then chapters, then appendices)
    sorted_urls = sorted(urls, key=lambda x: (
        0 if x == 'index.html' else
        1 if x.startswith('chapters/') else
        2 if x.startswith('appendices/') else
        3
    ))

    return sorted_urls


def generate_sitemap(base_url: str, urls: List[str], output_path: Path):
    """Generate sitemap.xml file."""
    today = datetime.now().strftime('%Y-%m-%d')

    # Start XML
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]

    # Priority mapping based on page type
    def get_priority(url: str) -> str:
        if url == 'index.html':
            return '1.0'
        elif url.startswith('chapters/'):
            return '0.8'
        elif url.startswith('appendices/'):
            return '0.6'
        elif url == 'references.html':
            return '0.5'
        else:
            return '0.7'

    # Change frequency mapping
    def get_changefreq(url: str) -> str:
        if url == 'index.html':
            return 'weekly'
        elif url.startswith('chapters/'):
            return 'monthly'
        elif url.startswith('appendices/'):
            return 'monthly'
        else:
            return 'monthly'

    # Add each URL
    for url in urls:
        full_url = f"{base_url}/{url}"
        priority = get_priority(url)
        changefreq = get_changefreq(url)

        xml_lines.extend([
            '  <url>',
            f'    <loc>{full_url}</loc>',
            f'    <lastmod>{today}</lastmod>',
            f'    <changefreq>{changefreq}</changefreq>',
            f'    <priority>{priority}</priority>',
            '  </url>',
        ])

    # Add download files
    for download_file in ['Embeddings-at-Scale.pdf', 'Embeddings-at-Scale.epub']:
        xml_lines.extend([
            '  <url>',
            f'    <loc>{base_url}/downloads/{download_file}</loc>',
            f'    <lastmod>{today}</lastmod>',
            f'    <changefreq>monthly</changefreq>',
            f'    <priority>0.9</priority>',
            '  </url>',
        ])

    # Close XML
    xml_lines.append('</urlset>')

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(xml_lines))

    print(f"Generated sitemap with {len(urls) + 2} URLs at {output_path}")


def main():
    """Main function to generate sitemap."""
    # Paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / '_quarto.yml'
    output_path = project_root / '_book' / 'sitemap.xml'

    # Base URL for GitHub Pages
    base_url = 'https://snowch.github.io/embeddings-at-scale-book'

    # Extract chapters
    urls = extract_chapters_from_config(config_path)

    # Ensure _book directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate sitemap
    generate_sitemap(base_url, urls, output_path)

    print(f"Sitemap generated successfully!")
    print(f"Total pages: {len(urls)}")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
