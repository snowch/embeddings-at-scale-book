# Code from Chapter 12
# Book: Embeddings at Scale

"""
Schema Evolution for Embedding Systems

Strategies:
1. Additive changes: Add new fields (backwards compatible)
2. Deprecation: Mark fields as deprecated before removal
3. Versioning: Version schemas and coordinate migrations
4. Defaults: Provide defaults for missing fields
5. Transformers: Convert between schema versions
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ChangeType(Enum):
    """Type of schema change"""
    ADD_FIELD = "add_field"
    REMOVE_FIELD = "remove_field"
    RENAME_FIELD = "rename_field"
    CHANGE_TYPE = "change_type"
    ADD_CONSTRAINT = "add_constraint"

@dataclass
class SchemaChange:
    """
    Schema change event

    Attributes:
        change_type: Type of change
        field_name: Affected field
        old_value: Previous value/type (if applicable)
        new_value: New value/type (if applicable)
        version: Schema version after change
        timestamp: When change was made
        backwards_compatible: Whether change is backwards compatible
    """
    change_type: ChangeType
    field_name: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    version: str = ""
    timestamp: datetime = None
    backwards_compatible: bool = True

class SchemaVersion:
    """
    Schema version with validation and transformation

    Attributes:
        version: Version identifier (e.g., "1.0.0")
        fields: Field definitions {field_name -> type}
        required_fields: Fields that must be present
        deprecated_fields: Fields marked for future removal
        transformers: Functions to transform from previous versions
    """

    def __init__(
        self,
        version: str,
        fields: Dict[str, type],
        required_fields: List[str],
        deprecated_fields: Optional[List[str]] = None
    ):
        """
        Args:
            version: Version identifier
            fields: Field definitions
            required_fields: Required fields
            deprecated_fields: Deprecated fields (optional)
        """
        self.version = version
        self.fields = fields
        self.required_fields = required_fields
        self.deprecated_fields = deprecated_fields or []
        self.transformers: Dict[str, Callable] = {}  # from_version -> transformer

        print(f"Created schema version {version}")
        print(f"  Fields: {len(fields)}")
        print(f"  Required: {len(required_fields)}")
        print(f"  Deprecated: {len(self.deprecated_fields)}")

    def validate(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate data against this schema version

        Args:
            data: Data to validate

        Returns:
            (is_valid, errors): Validation result and error messages
        """
        errors = []

        # Check required fields
        for field in self.required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Check field types
        for field, value in data.items():
            if field in self.fields:
                expected_type = self.fields[field]
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Field {field}: expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )

        # Warn about deprecated fields
        for field in self.deprecated_fields:
            if field in data:
                print(f"⚠️  Warning: Using deprecated field '{field}'")

        is_valid = len(errors) == 0
        return is_valid, errors

    def add_transformer(
        self,
        from_version: str,
        transformer: Callable[[Dict], Dict]
    ):
        """
        Add transformer from previous version

        Args:
            from_version: Source version
            transformer: Function to transform data
        """
        self.transformers[from_version] = transformer
        print(f"Added transformer: {from_version} -> {self.version}")

class SchemaRegistry:
    """
    Registry of all schema versions with migration support

    Responsibilities:
    - Track all schema versions
    - Validate data against appropriate version
    - Migrate data between versions
    - Detect breaking changes

    Usage:
    1. Register all schema versions
    2. Validate data (auto-detect version or specify)
    3. Migrate data to target version
    """

    def __init__(self):
        """Initialize empty registry"""
        self.versions: Dict[str, SchemaVersion] = {}
        self.version_history: List[SchemaChange] = []
        self.current_version: Optional[str] = None

        print("Initialized Schema Registry")

    def register_version(
        self,
        schema_version: SchemaVersion,
        set_current: bool = True
    ):
        """
        Register new schema version

        Args:
            schema_version: Schema version to register
            set_current: Set as current version
        """
        self.versions[schema_version.version] = schema_version

        if set_current:
            self.current_version = schema_version.version

        print(f"Registered schema version {schema_version.version}")
        if set_current:
            print("  Set as current version")

    def detect_version(self, data: Dict) -> Optional[str]:
        """
        Auto-detect schema version from data

        Strategy:
        - Try validating against each version
        - Return first version that validates successfully
        - Prefer newer versions over older

        Args:
            data: Data to detect version for

        Returns:
            Detected version or None
        """
        # Try versions in reverse order (newest first)
        for version in reversed(list(self.versions.keys())):
            schema = self.versions[version]
            is_valid, _ = schema.validate(data)
            if is_valid:
                return version

        return None

    def migrate(
        self,
        data: Dict,
        from_version: str,
        to_version: str
    ) -> Dict:
        """
        Migrate data from one version to another

        Args:
            data: Data to migrate
            from_version: Source version
            to_version: Target version

        Returns:
            Migrated data
        """
        if from_version == to_version:
            return data

        # Get migration path
        path = self._get_migration_path(from_version, to_version)

        if not path:
            raise ValueError(
                f"No migration path from {from_version} to {to_version}"
            )

        # Apply transformers along path
        current_data = data
        for i in range(len(path) - 1):
            current_version = path[i]
            next_version = path[i + 1]

            # Get transformer
            next_schema = self.versions[next_version]
            if current_version in next_schema.transformers:
                transformer = next_schema.transformers[current_version]
                current_data = transformer(current_data)
                print(f"  Migrated: {current_version} -> {next_version}")
            else:
                raise ValueError(
                    f"No transformer from {current_version} to {next_version}"
                )

        return current_data

    def _get_migration_path(
        self,
        from_version: str,
        to_version: str
    ) -> Optional[List[str]]:
        """
        Find migration path between versions

        For now: Assume linear version history
        In production: Use graph search for complex version trees

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            List of versions along migration path or None
        """
        versions = list(self.versions.keys())

        try:
            from_idx = versions.index(from_version)
            to_idx = versions.index(to_version)

            if from_idx < to_idx:
                # Forward migration
                return versions[from_idx:to_idx + 1]
            else:
                # Backward migration (not typically supported)
                return None
        except ValueError:
            return None

    def record_change(self, change: SchemaChange):
        """
        Record schema change in history

        Args:
            change: Schema change to record
        """
        self.version_history.append(change)

        # Alert if breaking change
        if not change.backwards_compatible:
            print(f"⚠️  BREAKING CHANGE: {change.change_type.value} on {change.field_name}")

# Example: Schema evolution for product embeddings
def schema_evolution_example():
    """
    Demonstrate schema evolution for product embeddings

    Versions:
    - v1.0: Initial schema (title, description, price)
    - v2.0: Add category field
    - v3.0: Add embedding_vector field, deprecate raw features
    """

    registry = SchemaRegistry()

    # Version 1.0: Initial schema
    v1 = SchemaVersion(
        version="1.0",
        fields={
            'title': str,
            'description': str,
            'price': float
        },
        required_fields=['title', 'price']
    )
    registry.register_version(v1, set_current=False)

    # Version 2.0: Add category (backwards compatible)
    v2 = SchemaVersion(
        version="2.0",
        fields={
            'title': str,
            'description': str,
            'price': float,
            'category': str  # New field
        },
        required_fields=['title', 'price']
    )

    # Add transformer from v1 to v2
    def v1_to_v2(data: Dict) -> Dict:
        """Add default category for v1 data"""
        data_v2 = data.copy()
        if 'category' not in data_v2:
            data_v2['category'] = 'Unknown'  # Default value
        return data_v2

    v2.add_transformer("1.0", v1_to_v2)
    registry.register_version(v2, set_current=False)

    # Version 3.0: Add embedding_vector, deprecate raw features
    v3 = SchemaVersion(
        version="3.0",
        fields={
            'title': str,
            'description': str,
            'price': float,
            'category': str,
            'embedding_vector': list  # New field
        },
        required_fields=['title', 'price', 'embedding_vector'],
        deprecated_fields=['description']  # Deprecate
    )

    # Add transformer from v2 to v3
    def v2_to_v3(data: Dict) -> Dict:
        """Add embedding vector for v2 data"""
        data_v3 = data.copy()
        if 'embedding_vector' not in data_v3:
            # Generate embedding from title + description
            # In production: Use actual embedding model
            data_v3['embedding_vector'] = [0.1, 0.2, 0.3]
        return data_v3

    v3.add_transformer("2.0", v2_to_v3)
    registry.register_version(v3, set_current=True)

    # Test: Migrate v1 data to v3
    print("\n=== Migration Test ===")

    v1_data = {
        'title': 'Laptop',
        'description': 'High-performance laptop',
        'price': 999.99
    }

    print(f"Original data (v1.0): {v1_data}")

    # Migrate v1 -> v2
    v2_data = registry.migrate(v1_data, from_version="1.0", to_version="2.0")
    print(f"Migrated to v2.0: {v2_data}")

    # Migrate v2 -> v3
    v3_data = registry.migrate(v2_data, from_version="2.0", to_version="3.0")
    print(f"Migrated to v3.0: {v3_data}")

    # Validate against v3
    is_valid, errors = v3.validate(v3_data)
    print(f"\nValidation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if errors:
        for error in errors:
            print(f"  - {error}")

# Uncomment to run:
# schema_evolution_example()
