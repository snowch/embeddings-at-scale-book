#!/bin/bash
# Generate individual EPUB files for each part of the book
set -e

echo "Generating per-part EPUB files..."

# Create output directory
EPUBS_DIR="_book/part-epubs"
mkdir -p "$EPUBS_DIR"

BOOK_TITLE="Embeddings at Scale"
AUTHOR="Chris Snow"

# Function to generate EPUB for a part
generate_part_epub() {
    local part_num="$1"
    local part_name="$2"
    shift 2
    local chapters=("$@")

    local output_file="$EPUBS_DIR/Part-${part_num}-${part_name// /-}.epub"

    echo "  Generating Part $part_num: $part_name"

    # Build pandoc command with all chapter files
    # Use 'quarto pandoc' to access pandoc bundled with Quarto
    quarto pandoc \
        --from=markdown \
        --to=epub \
        --toc \
        --toc-depth=3 \
        --number-sections \
        --metadata title="$BOOK_TITLE - Part $part_num: $part_name" \
        --metadata author="$AUTHOR" \
        --metadata lang="en" \
        --epub-cover-image=cover.jpg \
        -o "$output_file" \
        "${chapters[@]}"

    echo "    Created: $output_file"
}

# Part I: Foundations
generate_part_epub "I" "Foundations" \
    chapters/ch01_embedding_revolution.qmd \
    chapters/ch02_similarity_distance_metrics.qmd \
    chapters/ch03_vector_database_fundamentals.qmd

# Part II: Embedding Types
generate_part_epub "II" "Embedding Types" \
    chapters/ch04_text_embeddings.qmd \
    chapters/ch05_image_video_embeddings.qmd \
    chapters/ch06_multimodal_embeddings.qmd \
    chapters/ch07_graph_embeddings.qmd \
    chapters/ch08_timeseries_embeddings.qmd \
    chapters/ch09_code_embeddings.qmd \
    chapters/ch10_advanced_embedding_patterns.qmd

# Part III: Core Applications
generate_part_epub "III" "Core Applications" \
    chapters/ch11_rag_at_scale.qmd \
    chapters/ch12_semantic_search.qmd \
    chapters/ch13_recommendation_systems.qmd

# Part IV: Custom Embedding Development
generate_part_epub "IV" "Custom Embedding Development" \
    chapters/ch14_custom_embedding_strategies.qmd \
    chapters/ch15_contrastive_learning.qmd \
    chapters/ch16_siamese_networks.qmd \
    chapters/ch17_self_supervised_learning.qmd \
    chapters/ch18_advanced_embedding_techniques.qmd

# Part V: Production Engineering
generate_part_epub "V" "Production Engineering" \
    chapters/ch19_embedding_pipeline_engineering.qmd \
    chapters/ch20_scaling_embedding_training.qmd \
    chapters/ch21_embedding_quality_evaluation.qmd \
    chapters/ch22_high_performance_vector_ops.qmd \
    chapters/ch23_data_engineering.qmd \
    chapters/ch24_text_chunking.qmd \
    chapters/ch25_image_preparation.qmd

# Part VI: Cross-Industry Applications
generate_part_epub "VI" "Cross-Industry Applications" \
    chapters/ch26_cross_industry_patterns.qmd \
    chapters/ch27_video_surveillance_analytics.qmd \
    chapters/ch28_entity_resolution.qmd

# Part VII: Industry-Specific Applications
generate_part_epub "VII" "Industry-Specific Applications" \
    chapters/ch29_financial_services.qmd \
    chapters/ch30_healthcare_life_sciences.qmd \
    chapters/ch31_retail_ecommerce.qmd \
    chapters/ch32_manufacturing_industry40.qmd \
    chapters/ch33_media_entertainment.qmd \
    chapters/ch34_scientific_computing.qmd \
    chapters/ch35_defense_intelligence.qmd

# Part VIII: Future-Proofing & Optimization
generate_part_epub "VIII" "Future-Proofing and Optimization" \
    chapters/ch36_performance_optimization.qmd \
    chapters/ch37_security_privacy.qmd \
    chapters/ch38_monitoring_observability.qmd \
    chapters/ch39_future_trends.qmd

# Part IX: Implementation Roadmap
generate_part_epub "IX" "Implementation Roadmap" \
    chapters/ch40_organizational_transformation.qmd \
    chapters/ch41_implementation_roadmap.qmd \
    chapters/ch42_case_studies.qmd \
    chapters/ch43_embedding_governance.qmd \
    chapters/ch44_path_forward.qmd

# Create zip archive
echo ""
echo "Creating zip archive of all part EPUBs..."
mkdir -p _book/downloads
cd "$EPUBS_DIR"
zip -r ../downloads/Embeddings-at-Scale-Parts.zip *.epub
cd - > /dev/null

echo ""
echo "Created: _book/downloads/Embeddings-at-Scale-Parts.zip"
echo ""
echo "Part EPUBs included in archive:"
unzip -l _book/downloads/Embeddings-at-Scale-Parts.zip | tail -n +4 | head -n -2
