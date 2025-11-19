# Code from Chapter 12
# Book: Embeddings at Scale

"""
Multi-Source Data Fusion for Embeddings

Fusion strategies:
1. Early fusion: Combine raw data before feature extraction
2. Late fusion: Generate embeddings separately, combine at query time
3. Hybrid fusion: Join on keys, enrich with context from other sources

Challenges:
- Schema alignment: Map fields across sources
- Entity resolution: Link same entities across sources
- Temporal alignment: Handle different update frequencies
- Quality weighting: Prioritize high-quality sources
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class DataSource:
    """
    Configuration for a data source

    Attributes:
        source_id: Unique identifier
        source_type: Type of source (database, API, file, stream)
        update_frequency: How often data updates (realtime, hourly, daily)
        schema_mapping: Map source fields to canonical fields
        quality_score: Data quality score (0-1)
        priority: Priority when conflicts occur (higher = preferred)
    """
    source_id: str
    source_type: str
    update_frequency: str
    schema_mapping: Dict[str, str]  # source_field -> canonical_field
    quality_score: float = 1.0
    priority: int = 0

class MultiSourceDataFusion:
    """
    Fuse data from multiple sources for embedding generation

    Architecture:
    1. Extract: Pull data from each source
    2. Align: Map to canonical schema
    3. Resolve: Handle conflicts (same entity, different values)
    4. Enrich: Combine features from multiple sources
    5. Transform: Generate unified feature vectors

    Strategies:
    - Entity resolution: Link same entity across sources (fuzzy matching)
    - Conflict resolution: Prioritize high-quality/recent sources
    - Temporal alignment: Snapshot at consistent timestamp
    - Schema mapping: Translate source schemas to canonical
    """

    def __init__(
        self,
        canonical_schema: Dict[str, type],
        primary_key: str = 'entity_id'
    ):
        """
        Args:
            canonical_schema: Target schema for fused data
            primary_key: Field used to join across sources
        """
        self.canonical_schema = canonical_schema
        self.primary_key = primary_key
        self.sources: Dict[str, DataSource] = {}

        print("Initialized Multi-Source Data Fusion")
        print(f"  Canonical schema: {len(canonical_schema)} fields")
        print(f"  Primary key: {primary_key}")

    def register_source(self, source: DataSource):
        """
        Register a data source

        Args:
            source: Data source configuration
        """
        self.sources[source.source_id] = source
        print(f"Registered source: {source.source_id}")
        print(f"  Type: {source.source_type}")
        print(f"  Update frequency: {source.update_frequency}")
        print(f"  Quality score: {source.quality_score}")

    def extract(
        self,
        source_id: str,
        query: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract data from source

        Args:
            source_id: Source to extract from
            query: Optional query/filter

        Returns:
            DataFrame with source data
        """
        if source_id not in self.sources:
            raise ValueError(f"Unknown source: {source_id}")

        source = self.sources[source_id]

        # In production: Actually extract from source
        # For now: Return mock data
        mock_data = pd.DataFrame({
            'id': [f'{source_id}_1', f'{source_id}_2'],
            'source_field_1': ['value_1', 'value_2'],
            'source_field_2': [1.0, 2.0]
        })

        print(f"Extracted {len(mock_data)} records from {source_id}")
        return mock_data

    def align_schema(
        self,
        source_id: str,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align source data to canonical schema

        Steps:
        1. Map source fields to canonical fields
        2. Add missing canonical fields (with defaults)
        3. Drop unmapped source fields
        4. Convert types to match canonical schema

        Args:
            source_id: Source identifier
            data: Source data

        Returns:
            DataFrame with canonical schema
        """
        source = self.sources[source_id]

        # Apply schema mapping
        aligned = pd.DataFrame()

        for source_field, canonical_field in source.schema_mapping.items():
            if source_field in data.columns:
                aligned[canonical_field] = data[source_field]

        # Add missing canonical fields with defaults
        for field, field_type in self.canonical_schema.items():
            if field not in aligned.columns:
                # Default values by type
                if field_type == str:
                    aligned[field] = ''
                elif field_type in (int, float):
                    aligned[field] = 0
                else:
                    aligned[field] = None

        # Convert types
        for field, field_type in self.canonical_schema.items():
            if field in aligned.columns:
                try:
                    if field_type == str:
                        aligned[field] = aligned[field].astype(str)
                    elif field_type == float:
                        aligned[field] = pd.to_numeric(aligned[field], errors='coerce')
                except Exception as e:
                    print(f"⚠️  Type conversion failed for {field}: {e}")

        print(f"Aligned {len(aligned)} records to canonical schema")
        return aligned

    def resolve_conflicts(
        self,
        datasets: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Resolve conflicts when same entity appears in multiple sources

        Conflict resolution strategies:
        1. Priority-based: Use value from highest-priority source
        2. Recency-based: Use most recently updated value
        3. Quality-based: Use value from highest-quality source
        4. Voting: Use most common value across sources

        Args:
            datasets: Map of source_id -> DataFrame

        Returns:
            Fused DataFrame with conflicts resolved
        """
        # Merge all datasets on primary key
        merged = None

        for source_id, df in datasets.items():
            source = self.sources[source_id]

            # Add source metadata
            df = df.copy()
            df['_source'] = source_id
            df['_priority'] = source.priority
            df['_quality'] = source.quality_score

            if merged is None:
                merged = df
            else:
                # Outer join to include all entities
                merged = pd.merge(
                    merged,
                    df,
                    on=self.primary_key,
                    how='outer',
                    suffixes=('', f'_{source_id}')
                )

        # Resolve conflicts for each field
        resolved = pd.DataFrame()
        resolved[self.primary_key] = merged[self.primary_key]

        for field in self.canonical_schema.keys():
            if field == self.primary_key:
                continue

            # Find all columns for this field across sources
            field_columns = [col for col in merged.columns if col.startswith(field)]

            if len(field_columns) == 1:
                # No conflict
                resolved[field] = merged[field_columns[0]]
            else:
                # Resolve conflict using priority
                resolved[field] = self._resolve_field_conflict(
                    merged,
                    field,
                    field_columns
                )

        print(f"Resolved conflicts for {len(resolved)} entities")
        return resolved

    def _resolve_field_conflict(
        self,
        merged: pd.DataFrame,
        field: str,
        field_columns: List[str]
    ) -> pd.Series:
        """
        Resolve conflict for single field

        Strategy: Use value from highest-priority source

        Args:
            merged: Merged DataFrame with all sources
            field: Field to resolve
            field_columns: Columns containing values for this field

        Returns:
            Series with resolved values
        """
        # Priority-based resolution
        # (In production: Also consider recency, quality)

        resolved_values = []

        for idx, row in merged.iterrows():
            # Get values and their priorities
            candidates = []
            for col in field_columns:
                if pd.notna(row[col]):
                    # Extract source from column name
                    source_id = col.split('_')[-1] if '_' in col else None
                    priority = row.get(f'_priority_{source_id}', 0) if source_id else 0
                    candidates.append((priority, row[col]))

            if candidates:
                # Choose highest priority
                candidates.sort(reverse=True)
                resolved_values.append(candidates[0][1])
            else:
                # No value available
                resolved_values.append(None)

        return pd.Series(resolved_values)

    def fuse(
        self,
        source_ids: List[str],
        timestamp: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fuse data from multiple sources

        Steps:
        1. Extract from each source
        2. Align to canonical schema
        3. Resolve conflicts
        4. Return unified dataset

        Args:
            source_ids: Sources to fuse
            timestamp: Snapshot timestamp (for temporal alignment)

        Returns:
            Fused DataFrame
        """
        print(f"\nFusing data from {len(source_ids)} sources...")

        # Extract and align from each source
        aligned_datasets = {}

        for source_id in source_ids:
            # Extract
            raw_data = self.extract(source_id)

            # Align schema
            aligned_data = self.align_schema(source_id, raw_data)

            aligned_datasets[source_id] = aligned_data

        # Resolve conflicts
        fused_data = self.resolve_conflicts(aligned_datasets)

        print(f"✓ Fusion complete: {len(fused_data)} entities")

        return fused_data

# Example: Fuse product data from multiple sources
def multi_source_fusion_example():
    """
    Fuse product data from catalog, inventory, and analytics

    Sources:
    - Product Catalog: Title, description, category (daily updates)
    - Inventory System: Price, stock_count (real-time updates)
    - Analytics Platform: View_count, rating (hourly updates)

    Goal: Unified product features for embedding generation
    """

    # Define canonical schema
    canonical_schema = {
        'product_id': str,
        'title': str,
        'description': str,
        'category': str,
        'price': float,
        'stock_count': int,
        'view_count': int,
        'rating': float
    }

    # Initialize fusion engine
    fusion = MultiSourceDataFusion(
        canonical_schema=canonical_schema,
        primary_key='product_id'
    )

    # Register sources
    catalog_source = DataSource(
        source_id='product_catalog',
        source_type='database',
        update_frequency='daily',
        schema_mapping={
            'id': 'product_id',
            'name': 'title',
            'desc': 'description',
            'cat': 'category'
        },
        quality_score=0.95,
        priority=2
    )
    fusion.register_source(catalog_source)

    inventory_source = DataSource(
        source_id='inventory_system',
        source_type='api',
        update_frequency='realtime',
        schema_mapping={
            'product_id': 'product_id',
            'current_price': 'price',
            'available_quantity': 'stock_count'
        },
        quality_score=0.98,
        priority=3  # Highest priority (most authoritative)
    )
    fusion.register_source(inventory_source)

    analytics_source = DataSource(
        source_id='analytics_platform',
        source_type='stream',
        update_frequency='hourly',
        schema_mapping={
            'product_id': 'product_id',
            'views_24h': 'view_count',
            'avg_rating': 'rating'
        },
        quality_score=0.90,
        priority=1
    )
    fusion.register_source(analytics_source)

    # Fuse data
    fused_data = fusion.fuse(
        source_ids=['product_catalog', 'inventory_system', 'analytics_platform'],
        timestamp=datetime.now()
    )

    print("\n=== Fused Data ===")
    print(fused_data.head())
    print(f"\nColumns: {list(fused_data.columns)}")
    print(f"Records: {len(fused_data)}")

# Uncomment to run:
# multi_source_fusion_example()
