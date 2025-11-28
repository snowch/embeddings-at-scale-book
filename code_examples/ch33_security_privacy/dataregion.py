# Code from Chapter 24
# Book: Embeddings at Scale

"""
GDPR and Data Sovereignty Compliance

Architecture:
1. Data residency: Geographic partitioning of embeddings
2. Consent management: Track lawful basis for processing
3. Deletion workflows: Remove user data from system
4. Export APIs: Provide data portability
5. Breach detection: Monitor for security incidents
6. Privacy documentation: Maintain compliance records

Components:
- Geographic partitions: EU, US, APAC data silos
- Consent database: Track user permissions
- Deletion queue: Asynchronous data removal
- Export service: Generate portable data packages
- Breach monitor: Detect anomalous access
- Compliance dashboard: Track regulatory status

Legal requirements:
- GDPR: EU data protection regulation
- CCPA: California Consumer Privacy Act
- LGPD: Brazilian data protection law
- PIPEDA: Canadian privacy law
- PDPA: Singapore/Thailand data protection

Performance targets:
- Data residency: 100% compliance with geo-fencing
- Right to access: <30 days response time (legal maximum)
- Right to deletion: <24 hours for removal request
- Breach notification: <72 hours detection and reporting
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class DataRegion(Enum):
    """Data residency regions"""

    EU = "eu"  # European Union
    US = "us"  # United States
    APAC = "apac"  # Asia-Pacific
    LATAM = "latam"  # Latin America
    MEA = "mea"  # Middle East & Africa


class LegalBasis(Enum):
    """GDPR lawful basis for processing"""

    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTEREST = "legitimate_interest"


class DataCategory(Enum):
    """Types of personal data"""

    BASIC_IDENTITY = "basic_identity"  # Name, email
    SENSITIVE = "sensitive"  # Health, race, religion
    FINANCIAL = "financial"  # Payment info, credit score
    BEHAVIORAL = "behavioral"  # Browsing, usage patterns
    BIOMETRIC = "biometric"  # Fingerprints, face scans
    LOCATION = "location"  # GPS coordinates, IP address


@dataclass
class ConsentRecord:
    """
    User consent for data processing

    Attributes:
        user_id: User identifier
        purpose: Purpose of processing
        legal_basis: Lawful basis for processing
        data_categories: Types of data covered
        granted_at: When consent was given
        expires_at: When consent expires (if applicable)
        withdrawn_at: When consent was withdrawn
        consent_text: Text shown to user
        consent_version: Version of privacy policy
    """

    user_id: str
    purpose: str
    legal_basis: LegalBasis
    data_categories: List[DataCategory]
    granted_at: datetime
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    consent_text: str = ""
    consent_version: str = "1.0"


@dataclass
class DeletionRequest:
    """
    User request for data deletion

    Attributes:
        request_id: Unique request identifier
        user_id: User requesting deletion
        requested_at: When request was made
        scope: What to delete (all, specific categories)
        status: Current status
        completed_at: When deletion completed
        verification: User identity verification details
        retention_exceptions: Data retained for legal reasons
    """

    request_id: str
    user_id: str
    requested_at: datetime
    scope: str = "all"  # "all", "embeddings", "training_data"
    status: str = "pending"  # "pending", "in_progress", "completed", "failed"
    completed_at: Optional[datetime] = None
    verification: Dict[str, Any] = field(default_factory=dict)
    retention_exceptions: List[str] = field(default_factory=list)


@dataclass
class ExportRequest:
    """
    User request for data export

    Attributes:
        request_id: Unique request identifier
        user_id: User requesting export
        requested_at: When request was made
        format: Export format (json, csv, xml)
        status: Current status
        download_url: URL to download export
        expires_at: When download link expires
    """

    request_id: str
    user_id: str
    requested_at: datetime
    format: str = "json"
    status: str = "pending"
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None


@dataclass
class BreachIncident:
    """
    Data breach incident record

    Attributes:
        incident_id: Unique incident identifier
        detected_at: When breach was detected
        breach_type: Type of breach
        affected_users: Number of affected users
        data_categories: Types of data exposed
        severity: Incident severity
        reported_at: When authorities were notified
        notification_sent: Whether users were notified
        mitigation: Steps taken to mitigate
    """

    incident_id: str
    detected_at: datetime
    breach_type: str
    affected_users: int
    data_categories: List[DataCategory]
    severity: str  # "low", "medium", "high", "critical"
    reported_at: Optional[datetime] = None
    notification_sent: bool = False
    mitigation: List[str] = field(default_factory=list)


class GDPRComplianceEngine:
    """
    GDPR and data sovereignty compliance engine

    Implements:
    - Data residency: Geographic partitioning
    - Consent management: Track legal basis
    - Right to access: Export user data
    - Right to deletion: Remove user data
    - Breach notification: Detect and report incidents
    """

    def __init__(self):
        # Data partitions by region
        self.data_partitions: Dict[DataRegion, Dict[str, Any]] = {
            region: {"embeddings": {}, "metadata": {}} for region in DataRegion
        }

        # Consent records
        self.consent_records: Dict[str, List[ConsentRecord]] = {}

        # Deletion requests
        self.deletion_requests: List[DeletionRequest] = []

        # Export requests
        self.export_requests: List[ExportRequest] = []

        # Breach incidents
        self.breach_incidents: List[BreachIncident] = []

        # Data processing activities (Article 30 records)
        self.processing_activities: List[Dict[str, Any]] = []

        print("GDPR Compliance Engine initialized")

    def determine_region(self, user_location: str) -> DataRegion:
        """
        Determine data residency region based on user location

        Args:
            user_location: User's country code (ISO 3166-1 alpha-2)

        Returns:
            Data residency region
        """
        # EU countries
        eu_countries = {
            "AT",
            "BE",
            "BG",
            "CY",
            "CZ",
            "DE",
            "DK",
            "EE",
            "ES",
            "FI",
            "FR",
            "GR",
            "HR",
            "HU",
            "IE",
            "IT",
            "LT",
            "LU",
            "LV",
            "MT",
            "NL",
            "PL",
            "PT",
            "RO",
            "SE",
            "SI",
            "SK",
        }

        # APAC countries
        apac_countries = {
            "AU",
            "CN",
            "HK",
            "ID",
            "IN",
            "JP",
            "KR",
            "MY",
            "NZ",
            "PH",
            "SG",
            "TH",
            "TW",
            "VN",
        }

        # LATAM countries
        latam_countries = {"AR", "BR", "CL", "CO", "MX", "PE"}

        if user_location in eu_countries:
            return DataRegion.EU
        elif user_location in apac_countries:
            return DataRegion.APAC
        elif user_location in latam_countries:
            return DataRegion.LATAM
        elif user_location == "US":
            return DataRegion.US
        else:
            return DataRegion.MEA

    def store_embedding(
        self,
        user_id: str,
        embedding: np.ndarray,
        user_location: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Store embedding in appropriate regional partition

        Args:
            user_id: User identifier
            embedding: Embedding vector
            user_location: User's location (country code)
            metadata: Additional metadata
        """
        # Determine region
        region = self.determine_region(user_location)

        # Check consent
        if not self.check_consent(user_id, "embedding_storage"):
            raise ValueError(f"User {user_id} has not consented to embedding storage")

        # Store in regional partition
        self.data_partitions[region]["embeddings"][user_id] = embedding
        self.data_partitions[region]["metadata"][user_id] = {
            "stored_at": datetime.now(),
            "location": user_location,
            "metadata": metadata or {},
        }

        print(f"Stored embedding for {user_id} in {region.value} region")

    def record_consent(
        self,
        user_id: str,
        purpose: str,
        legal_basis: LegalBasis,
        data_categories: List[DataCategory],
        consent_text: str,
        expires_in_days: Optional[int] = None,
    ) -> ConsentRecord:
        """
        Record user consent for data processing

        Args:
            user_id: User identifier
            purpose: Purpose of processing
            legal_basis: Legal basis for processing
            data_categories: Types of data
            consent_text: Text shown to user
            expires_in_days: Consent expiration (days)

        Returns:
            Consent record
        """
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        consent = ConsentRecord(
            user_id=user_id,
            purpose=purpose,
            legal_basis=legal_basis,
            data_categories=data_categories,
            granted_at=datetime.now(),
            expires_at=expires_at,
            consent_text=consent_text,
        )

        if user_id not in self.consent_records:
            self.consent_records[user_id] = []

        self.consent_records[user_id].append(consent)

        print(f"Recorded consent for {user_id}: {purpose}")
        return consent

    def check_consent(self, user_id: str, purpose: str) -> bool:
        """
        Check if user has valid consent for purpose

        Args:
            user_id: User identifier
            purpose: Purpose to check

        Returns:
            True if consent exists and is valid
        """
        consents = self.consent_records.get(user_id, [])

        for consent in consents:
            # Check if consent covers purpose
            if consent.purpose != purpose:
                continue

            # Check if withdrawn
            if consent.withdrawn_at:
                continue

            # Check if expired
            if consent.expires_at and datetime.now() > consent.expires_at:
                continue

            return True

        return False

    def withdraw_consent(self, user_id: str, purpose: str):
        """
        Withdraw user consent for purpose

        Args:
            user_id: User identifier
            purpose: Purpose to withdraw
        """
        consents = self.consent_records.get(user_id, [])

        for consent in consents:
            if consent.purpose == purpose and not consent.withdrawn_at:
                consent.withdrawn_at = datetime.now()
                print(f"Withdrew consent for {user_id}: {purpose}")

    def request_deletion(
        self, user_id: str, scope: str = "all", verification: Optional[Dict[str, Any]] = None
    ) -> DeletionRequest:
        """
        Submit deletion request (GDPR Article 17)

        Steps:
        1. Verify user identity
        2. Create deletion request
        3. Process asynchronously
        4. Confirm completion to user

        Args:
            user_id: User identifier
            scope: What to delete
            verification: Identity verification details

        Returns:
            Deletion request
        """
        request_id = hashlib.sha256(f"{user_id}:{datetime.now().isoformat()}".encode()).hexdigest()[
            :16
        ]

        request = DeletionRequest(
            request_id=request_id,
            user_id=user_id,
            requested_at=datetime.now(),
            scope=scope,
            verification=verification or {},
        )

        self.deletion_requests.append(request)

        # Process deletion (simplified - would be async in production)
        self._process_deletion(request)

        print(f"Deletion request submitted: {request_id}")
        return request

    def _process_deletion(self, request: DeletionRequest):
        """
        Process deletion request

        Steps:
        1. Identify all data for user
        2. Check retention requirements
        3. Remove from embeddings and indexes
        4. Remove from training data
        5. Remove from backups (gradual)
        6. Update request status

        Args:
            request: Deletion request
        """
        request.status = "in_progress"
        user_id = request.user_id

        # Find user data across regions
        for region in DataRegion:
            if user_id in self.data_partitions[region]["embeddings"]:
                del self.data_partitions[region]["embeddings"][user_id]
                print(f"  Removed embedding from {region.value}")

            if user_id in self.data_partitions[region]["metadata"]:
                del self.data_partitions[region]["metadata"][user_id]
                print(f"  Removed metadata from {region.value}")

        # Remove consent records
        if user_id in self.consent_records:
            del self.consent_records[user_id]
            print("  Removed consent records")

        # Note: In production, also:
        # - Remove from vector indexes
        # - Remove from training datasets
        # - Schedule backup deletion
        # - Update analytics (aggregate only)

        request.status = "completed"
        request.completed_at = datetime.now()

        print(f"Deletion completed for {user_id}")

    def request_export(self, user_id: str, format: str = "json") -> ExportRequest:
        """
        Submit data export request (GDPR Article 20)

        Generate machine-readable export of user's data

        Args:
            user_id: User identifier
            format: Export format

        Returns:
            Export request
        """
        request_id = hashlib.sha256(
            f"{user_id}:export:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        request = ExportRequest(
            request_id=request_id, user_id=user_id, requested_at=datetime.now(), format=format
        )

        self.export_requests.append(request)

        # Generate export (simplified - would be async in production)
        self._generate_export(request)

        print(f"Export request submitted: {request_id}")
        return request

    def _generate_export(self, request: ExportRequest):
        """
        Generate data export for user

        Include:
        - Personal information
        - Embeddings
        - Processing history
        - Consent records

        Args:
            request: Export request
        """
        request.status = "in_progress"
        user_id = request.user_id

        export_data = {
            "user_id": user_id,
            "export_date": datetime.now().isoformat(),
            "embeddings": {},
            "consents": [],
            "processing_history": [],
        }

        # Collect embeddings from all regions
        for region in DataRegion:
            if user_id in self.data_partitions[region]["embeddings"]:
                embedding = self.data_partitions[region]["embeddings"][user_id]
                metadata = self.data_partitions[region]["metadata"][user_id]

                export_data["embeddings"][region.value] = {
                    "vector": embedding.tolist(),
                    "stored_at": metadata["stored_at"].isoformat(),
                    "metadata": metadata.get("metadata", {}),
                }

        # Include consent records
        consents = self.consent_records.get(user_id, [])
        for consent in consents:
            export_data["consents"].append(
                {
                    "purpose": consent.purpose,
                    "legal_basis": consent.legal_basis.value,
                    "granted_at": consent.granted_at.isoformat(),
                    "withdrawn_at": consent.withdrawn_at.isoformat()
                    if consent.withdrawn_at
                    else None,
                }
            )

        # Generate download URL (simplified)
        request.download_url = f"https://exports.example.com/{request.request_id}"
        request.expires_at = datetime.now() + timedelta(days=7)
        request.status = "completed"

        print(f"Export generated for {user_id}: {request.download_url}")

    def report_breach(
        self,
        breach_type: str,
        affected_users: int,
        data_categories: List[DataCategory],
        severity: str,
        description: str,
    ) -> BreachIncident:
        """
        Report data breach incident

        GDPR Article 33: Report to supervisory authority within 72 hours
        GDPR Article 34: Notify affected individuals if high risk

        Args:
            breach_type: Type of breach
            affected_users: Number affected
            data_categories: Types of data exposed
            severity: Severity level
            description: Incident description

        Returns:
            Breach incident record
        """
        incident_id = hashlib.sha256(f"breach:{datetime.now().isoformat()}".encode()).hexdigest()[
            :16
        ]

        incident = BreachIncident(
            incident_id=incident_id,
            detected_at=datetime.now(),
            breach_type=breach_type,
            affected_users=affected_users,
            data_categories=data_categories,
            severity=severity,
        )

        self.breach_incidents.append(incident)

        print(f"Breach incident reported: {incident_id}")
        print(f"  Type: {breach_type}")
        print(f"  Affected users: {affected_users}")
        print(f"  Severity: {severity}")

        # Check if notification required
        if severity in ["high", "critical"]:
            print("  WARNING: High-risk breach, notify supervisory authority within 72 hours")
            print("  WARNING: Notify affected users without undue delay")

        return incident

    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate compliance status report

        Returns:
            Compliance metrics and status
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "data_residency": {},
            "consent_status": {},
            "deletion_requests": {},
            "export_requests": {},
            "breach_incidents": len(self.breach_incidents),
        }

        # Data residency breakdown
        for region in DataRegion:
            report["data_residency"][region.value] = {
                "embeddings": len(self.data_partitions[region]["embeddings"]),
                "metadata": len(self.data_partitions[region]["metadata"]),
            }

        # Consent statistics
        total_consents = sum(len(c) for c in self.consent_records.values())
        active_consents = sum(
            1
            for consents in self.consent_records.values()
            for c in consents
            if not c.withdrawn_at and (not c.expires_at or c.expires_at > datetime.now())
        )
        report["consent_status"] = {
            "total": total_consents,
            "active": active_consents,
            "expired_or_withdrawn": total_consents - active_consents,
        }

        # Deletion requests
        report["deletion_requests"] = {
            "total": len(self.deletion_requests),
            "completed": sum(1 for r in self.deletion_requests if r.status == "completed"),
            "pending": sum(1 for r in self.deletion_requests if r.status == "pending"),
        }

        # Export requests
        report["export_requests"] = {
            "total": len(self.export_requests),
            "completed": sum(1 for r in self.export_requests if r.status == "completed"),
            "pending": sum(1 for r in self.export_requests if r.status == "pending"),
        }

        return report


# Example usage
def gdpr_compliance_example():
    """
    Demonstrate GDPR compliance features
    """
    print("=== GDPR and Data Sovereignty Compliance ===")
    print()

    # Initialize compliance engine
    gdpr = GDPRComplianceEngine()
    print()

    # Record consent for EU user
    print("1. Recording consent:")
    gdpr.record_consent(
        user_id="user_eu_123",
        purpose="embedding_storage",
        legal_basis=LegalBasis.CONSENT,
        data_categories=[DataCategory.BEHAVIORAL, DataCategory.BASIC_IDENTITY],
        consent_text="I agree to the processing of my data for recommendation purposes",
        expires_in_days=365,
    )
    print()

    # Store embedding with data residency
    print("2. Storing embedding with data residency:")
    embedding = np.random.randn(768).astype(np.float32)
    gdpr.store_embedding(
        user_id="user_eu_123",
        embedding=embedding,
        user_location="DE",  # Germany
        metadata={"source": "web_app"},
    )
    print()

    # Request data export (Article 20)
    print("3. Data portability - user requests export:")
    gdpr.request_export(user_id="user_eu_123", format="json")
    print()

    # Withdraw consent
    print("4. User withdraws consent:")
    gdpr.withdraw_consent(user_id="user_eu_123", purpose="embedding_storage")
    print()

    # Request deletion (Article 17)
    print("5. Right to deletion - user requests removal:")
    gdpr.request_deletion(
        user_id="user_eu_123", scope="all", verification={"method": "email", "verified": True}
    )
    print()

    # Report data breach
    print("6. Breach notification:")
    gdpr.report_breach(
        breach_type="unauthorized_access",
        affected_users=1,
        data_categories=[DataCategory.BEHAVIORAL],
        severity="medium",
        description="Unauthorized API access detected",
    )
    print()

    # Generate compliance report
    print("7. Compliance status report:")
    report = gdpr.generate_compliance_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    gdpr_compliance_example()
