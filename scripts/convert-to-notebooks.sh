#!/bin/bash
# Convert all chapter .qmd files to Jupyter notebooks and zip them
set -e

echo "Converting chapters to Jupyter notebooks..."

# Create output directory
NOTEBOOKS_DIR="_book/notebooks"
mkdir -p "$NOTEBOOKS_DIR"

# Find all chapter files and convert them
CHAPTER_COUNT=0
for qmd_file in chapters/ch*.qmd; do
    if [ -f "$qmd_file" ]; then
        filename=$(basename "$qmd_file" .qmd)
        echo "  Converting: $qmd_file -> $filename.ipynb"
        quarto convert "$qmd_file" --output "$NOTEBOOKS_DIR/$filename.ipynb"
        CHAPTER_COUNT=$((CHAPTER_COUNT + 1))
    fi
done

# Also convert appendices
for qmd_file in appendices/appendix_*.qmd; do
    if [ -f "$qmd_file" ]; then
        filename=$(basename "$qmd_file" .qmd)
        echo "  Converting: $qmd_file -> $filename.ipynb"
        quarto convert "$qmd_file" --output "$NOTEBOOKS_DIR/$filename.ipynb"
        CHAPTER_COUNT=$((CHAPTER_COUNT + 1))
    fi
done

echo "Converted $CHAPTER_COUNT files to Jupyter notebooks"

# Create zip archive
echo "Creating zip archive..."
mkdir -p _book/downloads
cd "$NOTEBOOKS_DIR"
zip -r ../downloads/Embeddings-at-Scale-Notebooks.zip *.ipynb
cd - > /dev/null

echo "Created: _book/downloads/Embeddings-at-Scale-Notebooks.zip"

# List the contents
echo ""
echo "Notebooks included in archive:"
unzip -l _book/downloads/Embeddings-at-Scale-Notebooks.zip | tail -n +4 | head -n -2
