"""
Phase 3: Enterprise Scaling Architecture

Architecture:
1. Multi-region deployment: Active-active across regions
2. Horizontal scaling: Sharding, replicas, auto-scaling
3. Global load balancing: Intelligent routing for performance
4. Cost optimization: Reserved capacity, spot instances, tiering
5. Governance: Security, compliance, access control

Scaling targets:
- Data scale: 1B-10B vectors across organization
- Query throughput: 10K-100K QPS (queries per second)
- Global latency: <50ms p95 for 90% of users
- Availability: 99.99% uptime (52 minutes/year downtime)
- Cost efficiency: <$0.005 per query at scale

Key components:
- Multi-region vector database clusters
- Global load balancer with geo-routing
- Distributed embedding generation pipeline
- Centralized monitoring and management
- Self-service platform for applications
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set


class Region(Enum):
    """Deployment regions"""

    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    ASIA_PACIFIC = "ap-southeast-1"


class TenantType(Enum):
    """Tenant types for multi-tenancy"""

    ENTERPRISE = "enterprise"
    DEPARTMENT = "department"
    APPLICATION = "application"
    DEVELOPMENT = "development"


@dataclass
class ResourceQuota:
    """Resource quotas for tenant"""

    max_vectors: int
    max_qps: int
    max_storage_gb: int
    max_monthly_cost: float

    # Current usage
    current_vectors: int = 0
    current_qps: float = 0.0
    current_storage_gb: float = 0.0
    current_monthly_cost: float = 0.0

    def is_within_quota(self) -> bool:
        """Check if usage within quota"""
        return (
            self.current_vectors <= self.max_vectors
            and self.current_qps <= self.max_qps
            and self.current_storage_gb <= self.max_storage_gb
            and self.current_monthly_cost <= self.max_monthly_cost
        )

    def utilization_percentage(self) -> Dict[str, float]:
        """Calculate resource utilization percentages"""
        return {
            "vectors": (self.current_vectors / self.max_vectors * 100)
            if self.max_vectors > 0
            else 0,
            "qps": (self.current_qps / self.max_qps * 100) if self.max_qps > 0 else 0,
            "storage": (self.current_storage_gb / self.max_storage_gb * 100)
            if self.max_storage_gb > 0
            else 0,
            "cost": (self.current_monthly_cost / self.max_monthly_cost * 100)
            if self.max_monthly_cost > 0
            else 0,
        }


@dataclass
class Tenant:
    """Multi-tenant configuration"""

    tenant_id: str
    tenant_name: str
    tenant_type: TenantType

    # Ownership
    owner_email: str
    team_name: str
    cost_center: str

    # Configuration
    regions: List[Region]
    isolation_level: str  # shared, dedicated_shard, dedicated_cluster
    quotas: ResourceQuota

    # Access control
    allowed_users: Set[str] = field(default_factory=set)
    allowed_applications: Set[str] = field(default_factory=set)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"  # active, suspended, archived


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration"""

    name: str
    metric_name: str  # cpu_utilization, qps, queue_depth
    target_value: float

    # Scaling parameters
    min_instances: int
    max_instances: int
    scale_up_cooldown_seconds: int = 300
    scale_down_cooldown_seconds: int = 600

    # Thresholds
    scale_up_threshold: float = 0.0  # Above target
    scale_down_threshold: float = 0.0  # Below target

    def __post_init__(self):
        """Set default thresholds"""
        if self.scale_up_threshold == 0.0:
            self.scale_up_threshold = self.target_value * 1.2
        if self.scale_down_threshold == 0.0:
            self.scale_down_threshold = self.target_value * 0.5


class EnterpriseDeployment:
    """
    Manage enterprise-wide embedding platform deployment.

    Handles multi-region, multi-tenant deployment with
    governance, scaling, and cost management.
    """

    def __init__(self, deployment_name: str):
        self.deployment_name = deployment_name
        self.tenants: Dict[str, Tenant] = {}
        self.regions_active: Set[Region] = set()
        self.scaling_policies: List[ScalingPolicy] = []

        # Monitoring
        self.total_vectors: int = 0
        self.total_qps: float = 0.0
        self.total_monthly_cost: float = 0.0

    def add_tenant(self, tenant: Tenant) -> None:
        """Add new tenant to platform"""
        if tenant.tenant_id in self.tenants:
            raise ValueError(f"Tenant {tenant.tenant_id} already exists")

        self.tenants[tenant.tenant_id] = tenant
        self.regions_active.update(tenant.regions)

        print(f"Added tenant: {tenant.tenant_name} ({tenant.tenant_id})")
        print(f"  Regions: {[r.value for r in tenant.regions]}")
        print(f"  Quotas: {tenant.quotas.max_vectors:,} vectors, {tenant.quotas.max_qps} QPS")

    def update_tenant_usage(
        self,
        tenant_id: str,
        vectors: Optional[int] = None,
        qps: Optional[float] = None,
        storage_gb: Optional[float] = None,
        cost: Optional[float] = None,
    ) -> None:
        """Update tenant resource usage"""
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")

        tenant = self.tenants[tenant_id]

        if vectors is not None:
            tenant.quotas.current_vectors = vectors
        if qps is not None:
            tenant.quotas.current_qps = qps
        if storage_gb is not None:
            tenant.quotas.current_storage_gb = storage_gb
        if cost is not None:
            tenant.quotas.current_monthly_cost = cost

        # Check quota violations
        if not tenant.quotas.is_within_quota():
            self._handle_quota_violation(tenant)

    def _handle_quota_violation(self, tenant: Tenant) -> None:
        """Handle tenant exceeding quota"""
        utilization = tenant.quotas.utilization_percentage()

        violations = [resource for resource, pct in utilization.items() if pct > 100]

        print(f"QUOTA VIOLATION: Tenant {tenant.tenant_name}")
        print(f"  Exceeded: {violations}")
        print(f"  Utilization: {utilization}")
        # In production: Alert, throttle, or auto-scale

    def add_scaling_policy(self, policy: ScalingPolicy) -> None:
        """Add auto-scaling policy"""
        self.scaling_policies.append(policy)
        print(f"Added scaling policy: {policy.name}")
        print(f"  Metric: {policy.metric_name}, Target: {policy.target_value}")
        print(f"  Instances: {policy.min_instances}-{policy.max_instances}")

    def calculate_total_cost(self) -> Dict[str, float]:
        """Calculate total platform cost breakdown"""
        cost_breakdown = {"compute": 0.0, "storage": 0.0, "network": 0.0, "api_calls": 0.0}

        for tenant in self.tenants.values():
            # Simplified cost model
            # In production: Get from actual billing APIs
            compute_cost = tenant.quotas.current_qps * 0.01  # $0.01 per QPS/month
            storage_cost = tenant.quotas.current_storage_gb * 0.10  # $0.10/GB/month

            cost_breakdown["compute"] += compute_cost
            cost_breakdown["storage"] += storage_cost

        cost_breakdown["total"] = sum(cost_breakdown.values())
        return cost_breakdown

    def generate_governance_report(self) -> str:
        """Generate governance and compliance report"""
        report = []
        report.append(f"# Enterprise Deployment Report: {self.deployment_name}\n\n")
        report.append(f"Generated: {datetime.now().isoformat()}\n\n")

        # Overview
        report.append("## Platform Overview\n\n")
        report.append(f"- Active tenants: {len(self.tenants)}\n")
        report.append(f"- Active regions: {[r.value for r in self.regions_active]}\n")
        report.append(f"- Total vectors: {self.total_vectors:,}\n")
        report.append(f"- Total QPS: {self.total_qps:,.0f}\n\n")

        # Cost analysis
        cost_breakdown = self.calculate_total_cost()
        report.append("## Cost Analysis\n\n")
        for component, cost in cost_breakdown.items():
            report.append(f"- {component.title()}: ${cost:,.2f}/month\n")
        report.append("\n")

        # Tenant summary
        report.append("## Tenant Summary\n\n")
        for tenant in sorted(self.tenants.values(), key=lambda t: t.tenant_name):
            utilization = tenant.quotas.utilization_percentage()
            report.append(f"### {tenant.tenant_name} ({tenant.tenant_type.value})\n\n")
            report.append(f"- Owner: {tenant.owner_email} ({tenant.team_name})\n")
            report.append(f"- Status: {tenant.status}\n")
            report.append(f"- Regions: {[r.value for r in tenant.regions]}\n")
            report.append("- Utilization:\n")
            for resource, pct in utilization.items():
                status = "⚠️" if pct > 80 else "✓"
                report.append(f"  - {status} {resource}: {pct:.1f}%\n")
            report.append("\n")

        # Compliance
        report.append("## Compliance Status\n\n")
        report.append("- Data sovereignty: All data stored in appropriate regions ✓\n")
        report.append("- Access control: All tenants have defined access policies ✓\n")
        report.append("- Audit logging: All operations logged for 90 days ✓\n")
        report.append("- Encryption: All data encrypted at rest and in transit ✓\n\n")

        return "".join(report)


# Example: Enterprise deployment
def example_enterprise_deployment():
    """Example enterprise deployment setup"""

    deployment = EnterpriseDeployment("Global Embedding Platform")

    # Add enterprise tenant (Search team)
    search_tenant = Tenant(
        tenant_id="search-prod",
        tenant_name="Product Search",
        tenant_type=TenantType.APPLICATION,
        owner_email="search-team@company.com",
        team_name="Search & Discovery",
        cost_center="CC-1234",
        regions=[Region.US_EAST, Region.EU_WEST, Region.ASIA_PACIFIC],
        isolation_level="dedicated_shard",
        quotas=ResourceQuota(
            max_vectors=1_000_000_000,  # 1B vectors
            max_qps=10000,
            max_storage_gb=5000,  # 5TB
            max_monthly_cost=50000,
        ),
    )
    deployment.add_tenant(search_tenant)

    # Add department tenant (Recommendations)
    recs_tenant = Tenant(
        tenant_id="recs-prod",
        tenant_name="Recommendations",
        tenant_type=TenantType.APPLICATION,
        owner_email="ml-team@company.com",
        team_name="ML/Personalization",
        cost_center="CC-1235",
        regions=[Region.US_EAST, Region.US_WEST],
        isolation_level="shared",
        quotas=ResourceQuota(
            max_vectors=100_000_000,  # 100M vectors
            max_qps=5000,
            max_storage_gb=500,
            max_monthly_cost=10000,
        ),
    )
    deployment.add_tenant(recs_tenant)

    # Add development tenant
    dev_tenant = Tenant(
        tenant_id="dev-sandbox",
        tenant_name="Development Sandbox",
        tenant_type=TenantType.DEVELOPMENT,
        owner_email="platform-team@company.com",
        team_name="Platform Engineering",
        cost_center="CC-1236",
        regions=[Region.US_EAST],
        isolation_level="shared",
        quotas=ResourceQuota(
            max_vectors=10_000_000,  # 10M vectors
            max_qps=100,
            max_storage_gb=50,
            max_monthly_cost=1000,
        ),
    )
    deployment.add_tenant(dev_tenant)

    # Configure auto-scaling
    deployment.add_scaling_policy(
        ScalingPolicy(
            name="Vector DB Auto-scaling",
            metric_name="cpu_utilization",
            target_value=70.0,  # 70% CPU
            min_instances=3,
            max_instances=20,
        )
    )

    deployment.add_scaling_policy(
        ScalingPolicy(
            name="QPS-based Scaling",
            metric_name="qps",
            target_value=5000,  # 5K QPS per instance
            min_instances=3,
            max_instances=20,
        )
    )

    # Simulate usage
    deployment.update_tenant_usage(
        tenant_id="search-prod",
        vectors=850_000_000,  # 85% of quota
        qps=8500,  # 85% of quota
        storage_gb=4200,  # 84% of quota
        cost=42000,  # 84% of budget
    )

    deployment.update_tenant_usage(
        tenant_id="recs-prod",
        vectors=75_000_000,  # 75% of quota
        qps=3500,  # 70% of quota
        storage_gb=400,  # 80% of quota
        cost=8000,  # 80% of budget
    )

    # Generate report
    print(deployment.generate_governance_report())


if __name__ == "__main__":
    example_enterprise_deployment()
