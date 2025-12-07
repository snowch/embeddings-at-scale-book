#!/bin/bash
# Generate a JSON mapping of section IDs to chapter URLs
# This is used by the /go/ redirect page

set -e

OUTPUT_FILE="${1:-_book/section-map.json}"

echo "Generating section ID mapping..."

# Start JSON object
echo "{" > "$OUTPUT_FILE"

first=true

# Process all chapter and appendix .qmd files
for qmd_file in chapters/*.qmd appendices/*.qmd; do
    if [ ! -f "$qmd_file" ]; then
        continue
    fi

    # Get the corresponding HTML path (relative to _book)
    html_path="${qmd_file%.qmd}.html"

    # Extract section IDs from h1 headings: # Title {#sec-xxx}
    # Also extract h2+ section IDs for completeness
    while IFS= read -r line; do
        # Match {#sec-xxx} or {#xxx} patterns
        if [[ "$line" =~ \{#([a-zA-Z0-9_-]+)\} ]]; then
            sec_id="${BASH_REMATCH[1]}"

            if [ "$first" = true ]; then
                first=false
            else
                echo "," >> "$OUTPUT_FILE"
            fi

            # Write the mapping entry
            printf '  "%s": "%s"' "$sec_id" "$html_path" >> "$OUTPUT_FILE"
        fi
    done < "$qmd_file"
done

# Also add index.qmd sections
if [ -f "index.qmd" ]; then
    while IFS= read -r line; do
        if [[ "$line" =~ \{#([a-zA-Z0-9_-]+)\} ]]; then
            sec_id="${BASH_REMATCH[1]}"

            if [ "$first" = true ]; then
                first=false
            else
                echo "," >> "$OUTPUT_FILE"
            fi

            printf '  "%s": "%s"' "$sec_id" "index.html" >> "$OUTPUT_FILE"
        fi
    done < "index.qmd"
fi

# Close JSON object
echo "" >> "$OUTPUT_FILE"
echo "}" >> "$OUTPUT_FILE"

echo "Section map generated: $OUTPUT_FILE"
