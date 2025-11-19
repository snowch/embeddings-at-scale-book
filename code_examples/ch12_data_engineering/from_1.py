# Code from Chapter 12
# Book: Embeddings at Scale

"""
Data Quality Framework for Embeddings

Quality dimensions:
1. Completeness: No missing required fields
2. Correctness: Values within expected ranges
3. Consistency: Relationships preserved across updates
4. Freshness: Data reflects current state
5. Uniqueness: No exact duplicates
6. Semantic validity: Features preserve meaning
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class QualityIssue:
    """
    Data quality issue detected during validation

    Attributes:
        issue_type: Type of issue (duplicate, outlier, missing, etc.)
        severity: Severity level (critical, warning, info)
        record_id: Affected record
        field: Affected field (if applicable)
        description: Human-readable description
        metadata: Additional context
    """

    issue_type: str
    severity: str  # 'critical', 'warning', 'info'
    record_id: str
    field: Optional[str] = None
    description: str = ""
    metadata: Dict = None


# Placeholder class - see from.py for full implementation
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class EmbeddingFeatures:
    """Placeholder for EmbeddingFeatures."""

    record_id: str
    text_content: str = ""
    structured_features: Dict[str, Any] = None
    metadata: Dict[str, Any] = None


class EmbeddingDataQualityValidator:
    """
    Comprehensive data quality validation for embedding training

    Validations:
    1. Schema validation: Required fields present and correct types
    2. Duplicate detection: Exact and near-duplicate records
    3. Outlier detection: Statistical anomalies in features
    4. Consistency validation: Relationship integrity
    5. Drift detection: Distribution shifts over time

    Usage:
    - Run before training (filter/fix bad data)
    - Run during serving (detect production drift)
    - Run after updates (validate quality maintained)
    """

    def __init__(
        self, required_fields: List[str], numeric_fields: List[str], categorical_fields: List[str]
    ):
        """
        Args:
            required_fields: Fields that must be present
            numeric_fields: Fields expected to be numeric
            categorical_fields: Fields expected to be categorical
        """
        self.required_fields = required_fields
        self.numeric_fields = numeric_fields
        self.categorical_fields = categorical_fields

        # Quality metrics
        self.issues: List[QualityIssue] = []
        self.records_validated = 0

        # Baseline statistics (for drift detection)
        self.baseline_stats: Optional[Dict] = None

        print("Initialized Data Quality Validator")
        print(f"  Required fields: {len(required_fields)}")
        print(f"  Numeric fields: {len(numeric_fields)}")
        print(f"  Categorical fields: {len(categorical_fields)}")

    def validate(
        self, records: List[EmbeddingFeatures]
    ) -> Tuple[List[EmbeddingFeatures], List[QualityIssue]]:
        """
        Validate data quality and return clean records

        Args:
            records: Records to validate

        Returns:
            (clean_records, issues): Valid records and detected issues
        """
        print(f"Validating {len(records):,} records...")

        self.issues = []
        clean_records = []

        for record in records:
            self.records_validated += 1

            # Run all validations
            if (
                self._validate_schema(record)
                and self._validate_values(record)
                and self._validate_semantic(record)
            ):
                clean_records.append(record)

        # Global validations (across all records)
        self._detect_duplicates(records)
        self._detect_drift(records)

        # Summary
        critical = sum(1 for i in self.issues if i.severity == "critical")
        warnings = sum(1 for i in self.issues if i.severity == "warning")

        print("✓ Validation complete")
        print(f"  Clean records: {len(clean_records):,} ({len(clean_records) / len(records):.1%})")
        print(f"  Issues found: {len(self.issues)}")
        print(f"    Critical: {critical}")
        print(f"    Warnings: {warnings}")

        return clean_records, self.issues

    def _validate_schema(self, record: EmbeddingFeatures) -> bool:
        """
        Validate schema: Required fields present and correct types

        Args:
            record: Record to validate

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not record.text_features and not record.structured_features:
            self.issues.append(
                QualityIssue(
                    issue_type="missing_features",
                    severity="critical",
                    record_id=record.record_id,
                    description="Record has no text or structured features",
                )
            )
            return False

        # Validate structured features types
        if record.structured_features:
            for field, value in record.structured_features.items():
                if field in self.numeric_fields:
                    if not isinstance(value, (int, float)):
                        self.issues.append(
                            QualityIssue(
                                issue_type="type_mismatch",
                                severity="critical",
                                record_id=record.record_id,
                                field=field,
                                description=f"Expected numeric, got {type(value).__name__}",
                            )
                        )
                        return False

        return True

    def _validate_values(self, record: EmbeddingFeatures) -> bool:
        """
        Validate values: Range checks, outlier detection

        Args:
            record: Record to validate

        Returns:
            True if valid, False otherwise
        """
        if not record.structured_features:
            return True

        for field, value in record.structured_features.items():
            # Check for invalid numbers
            if not np.isfinite(value):
                self.issues.append(
                    QualityIssue(
                        issue_type="invalid_value",
                        severity="critical",
                        record_id=record.record_id,
                        field=field,
                        description=f"Invalid numeric value: {value}",
                    )
                )
                return False

            # Check for extreme outliers (>6 sigma from mean)
            # In production: Use learned statistics from training set
            if abs(value) > 1e6:
                self.issues.append(
                    QualityIssue(
                        issue_type="outlier",
                        severity="warning",
                        record_id=record.record_id,
                        field=field,
                        description=f"Extreme value: {value}",
                        metadata={"value": value},
                    )
                )

        return True

    def _validate_semantic(self, record: EmbeddingFeatures) -> bool:
        """
        Validate semantic quality: Text quality, meaningful content

        Args:
            record: Record to validate

        Returns:
            True if valid, False otherwise
        """
        if record.text_features:
            text = record.text_features.strip()

            # Must have minimum length
            if len(text) < 10:
                self.issues.append(
                    QualityIssue(
                        issue_type="text_too_short",
                        severity="warning",
                        record_id=record.record_id,
                        field="text_features",
                        description=f"Text too short: {len(text)} chars",
                    )
                )
                return False

            # Check for gibberish (high ratio of non-alphanumeric)
            alphanumeric = sum(c.isalnum() for c in text)
            if alphanumeric / len(text) < 0.5:
                self.issues.append(
                    QualityIssue(
                        issue_type="gibberish",
                        severity="warning",
                        record_id=record.record_id,
                        field="text_features",
                        description=f"Low alphanumeric ratio: {alphanumeric / len(text):.2f}",
                    )
                )
                return False

        return True

    def _detect_duplicates(self, records: List[EmbeddingFeatures]):
        """
        Detect exact and near-duplicate records

        Duplicates poison contrastive learning:
        - Same example appears as positive and negative
        - Model learns spurious correlations

        Args:
            records: All records to check
        """
        # Hash-based exact duplicate detection
        seen_hashes: Dict[str, str] = {}  # hash -> record_id

        for record in records:
            if record.data_hash:
                if record.data_hash in seen_hashes:
                    self.issues.append(
                        QualityIssue(
                            issue_type="exact_duplicate",
                            severity="critical",
                            record_id=record.record_id,
                            description=f"Duplicate of {seen_hashes[record.data_hash]}",
                            metadata={"duplicate_of": seen_hashes[record.data_hash]},
                        )
                    )
                else:
                    seen_hashes[record.data_hash] = record.record_id

    def _detect_drift(self, records: List[EmbeddingFeatures]):
        """
        Detect distribution drift compared to baseline

        Drift detection:
        - Compare current batch statistics to baseline
        - Flag significant changes (>3 sigma)
        - Alert if mean, std, or quantiles shift

        Args:
            records: Current batch of records
        """
        # Compute current statistics
        current_stats = self._compute_statistics(records)

        # If no baseline, set current as baseline
        if self.baseline_stats is None:
            self.baseline_stats = current_stats
            print("  Set baseline statistics for drift detection")
            return

        # Compare to baseline
        for field in self.numeric_fields:
            if field not in current_stats or field not in self.baseline_stats:
                continue

            baseline_mean = self.baseline_stats[field]["mean"]
            baseline_std = self.baseline_stats[field]["std"]
            current_mean = current_stats[field]["mean"]

            # Detect significant mean shift
            if baseline_std > 0:
                z_score = abs(current_mean - baseline_mean) / baseline_std

                if z_score > 3:
                    self.issues.append(
                        QualityIssue(
                            issue_type="distribution_drift",
                            severity="warning",
                            record_id="<global>",
                            field=field,
                            description=f"Mean shifted by {z_score:.1f} sigma",
                            metadata={
                                "baseline_mean": baseline_mean,
                                "current_mean": current_mean,
                                "z_score": z_score,
                            },
                        )
                    )

    def _compute_statistics(self, records: List[EmbeddingFeatures]) -> Dict[str, Dict]:
        """
        Compute statistics for numeric fields

        Args:
            records: Records to analyze

        Returns:
            Dict mapping field -> {mean, std, min, max, quantiles}
        """
        stats = {}

        # Collect values per field
        field_values = defaultdict(list)
        for record in records:
            if record.structured_features:
                for field, value in record.structured_features.items():
                    if field in self.numeric_fields:
                        field_values[field].append(value)

        # Compute statistics
        for field, values in field_values.items():
            if values:
                stats[field] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "p25": np.percentile(values, 25),
                    "p50": np.percentile(values, 50),
                    "p75": np.percentile(values, 75),
                }

        return stats

    def get_quality_report(self) -> Dict:
        """
        Generate quality report

        Returns:
            Quality metrics and issue summary
        """
        issue_counts = defaultdict(int)
        for issue in self.issues:
            issue_counts[issue.issue_type] += 1

        return {
            "records_validated": self.records_validated,
            "total_issues": len(self.issues),
            "critical_issues": sum(1 for i in self.issues if i.severity == "critical"),
            "warnings": sum(1 for i in self.issues if i.severity == "warning"),
            "issue_breakdown": dict(issue_counts),
            "quality_score": 1 - (len(self.issues) / max(1, self.records_validated)),
        }


# Example: Validate product data quality
def data_quality_example():
    """
    Validate data quality for product embeddings

    Scenario: E-commerce product catalog
    - 100K products
    - Detect duplicates, outliers, drift
    """

    # Generate sample data with quality issues
    def generate_test_data(count=1000):
        records = []

        for i in range(count):
            # 10% duplicates
            if i > 0 and np.random.random() < 0.1:
                # Duplicate previous record
                dup_record = records[-1]
                records.append(
                    EmbeddingFeatures(
                        record_id=f"product_{i}",
                        text_features=dup_record.text_features,
                        structured_features=dup_record.structured_features,
                        data_hash=dup_record.data_hash,
                    )
                )
                continue

            # 5% missing features
            if np.random.random() < 0.05:
                records.append(
                    EmbeddingFeatures(
                        record_id=f"product_{i}", text_features=None, structured_features=None
                    )
                )
                continue

            # 2% outliers
            if np.random.random() < 0.02:
                price = 1e8  # Extreme outlier
            else:
                price = 10.0 + i % 100

            # Normal record
            text = f"Product {i}" if i % 20 != 0 else "X"  # 5% too short

            record = EmbeddingFeatures(
                record_id=f"product_{i}",
                text_features=text,
                structured_features={"price": price, "rating": 3.0 + (i % 5) * 0.5},
                data_hash=hashlib.md5(text.encode()).hexdigest(),
            )
            records.append(record)

        return records

    # Generate test data
    test_records = generate_test_data(1000)

    # Initialize validator
    validator = EmbeddingDataQualityValidator(
        required_fields=["text_features"], numeric_fields=["price", "rating"], categorical_fields=[]
    )

    # Validate
    clean_records, issues = validator.validate(test_records)

    # Print quality report
    report = validator.get_quality_report()
    print("\n=== Quality Report ===")
    print(f"Records validated: {report['records_validated']:,}")
    print(f"Quality score: {report['quality_score']:.2%}")
    print(f"Total issues: {report['total_issues']}")
    print(f"  Critical: {report['critical_issues']}")
    print(f"  Warnings: {report['warnings']}")
    print("\nIssue breakdown:")
    for issue_type, count in report["issue_breakdown"].items():
        print(f"  {issue_type}: {count}")

    # Print sample issues
    print("\nSample issues:")
    for issue in issues[:5]:
        print(f"  [{issue.severity.upper()}] {issue.issue_type}: {issue.description}")


# Uncomment to run:
# data_quality_example()
