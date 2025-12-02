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

# Part I: Foundation & Strategy
generate_part_epub "I" "Foundation and Strategy" \
    chapters/ch01_embedding_revolution.qmd \
    chapters/ch02_foundational_embedding_types.qmd \
    chapters/ch03_advanced_embedding_patterns.qmd \
    chapters/ch04_strategic_architecture.qmd \
    chapters/ch05_similarity_distance_metrics.qmd \
    chapters/ch06_vector_database_fundamentals.qmd \
    chapters/ch07_embedding_model_fundamentals.qmd

# Part II: Custom Embedding Development
generate_part_epub "II" "Custom Embedding Development" \
    chapters/ch08_custom_embedding_strategies.qmd \
    chapters/ch09_contrastive_learning.qmd \
    chapters/ch10_siamese_networks.qmd \
    chapters/ch11_self_supervised_learning.qmd \
    chapters/ch12_advanced_embedding_techniques.qmd

# Part III: Production Engineering
generate_part_epub "III" "Production Engineering" \
    chapters/ch13_embedding_pipeline_engineering.qmd \
    chapters/ch14_scaling_embedding_training.qmd \
    chapters/ch15_high_performance_vector_ops.qmd \
    chapters/ch16_data_engineering.qmd \
    chapters/ch17_text_chunking.qmd \
    chapters/ch18_image_preparation.qmd

# Part IV: Advanced Applications
generate_part_epub "IV" "Advanced Applications" \
    chapters/ch19_rag_at_scale.qmd \
    chapters/ch20_semantic_search.qmd \
    chapters/ch21_recommendation_systems.qmd \
    chapters/ch22_anomaly_detection_security.qmd \
    chapters/ch23_automated_decision_systems.qmd

# Part V: Industry Applications
generate_part_epub "V" "Industry Applications" \
    chapters/ch24_financial_services.qmd \
    chapters/ch25_healthcare_life_sciences.qmd \
    chapters/ch26_retail_ecommerce.qmd \
    chapters/ch27_manufacturing_industry40.qmd \
    chapters/ch28_media_entertainment.qmd \
    chapters/ch29_scientific_computing.qmd \
    chapters/ch30_defense_intelligence.qmd \
    chapters/ch31_video_surveillance_analytics.qmd

# Part VI: Future-Proofing & Optimization
generate_part_epub "VI" "Future-Proofing and Optimization" \
    chapters/ch32_performance_optimization.qmd \
    chapters/ch33_security_privacy.qmd \
    chapters/ch34_monitoring_observability.qmd \
    chapters/ch35_future_trends.qmd

# Part VII: Implementation Roadmap
generate_part_epub "VII" "Implementation Roadmap" \
    chapters/ch36_organizational_transformation.qmd \
    chapters/ch37_implementation_roadmap.qmd \
    chapters/ch38_case_studies.qmd \
    chapters/ch39_path_forward.qmd

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
