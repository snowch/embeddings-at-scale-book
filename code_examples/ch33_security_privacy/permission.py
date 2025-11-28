# Code from Chapter 24
# Book: Embeddings at Scale

"""
Access Control and Audit Trails for Embeddings

Architecture:
1. Authentication: Verify user identity (OAuth, API keys, mTLS)
2. Authorization: Check permissions for requested operation
3. Query filtering: Apply row-level security based on attributes
4. Rate limiting: Enforce quotas per user/tenant
5. Audit logging: Record all access attempts and results
6. Monitoring: Real-time anomaly detection

Components:
- Identity provider: OAuth2, SAML, API keys
- Policy engine: Evaluate access policies
- Query rewriter: Inject security filters
- Audit log: Immutable access record
- Monitoring: Real-time alerting

Security properties:
- Least privilege: Users get minimum necessary access
- Defense in depth: Multiple layers of security
- Separation of duties: Different roles for different functions
- Audit trail: Complete record of all access
- Non-repudiation: Users cannot deny actions

Performance targets:
- Authorization: <5ms per query
- Audit logging: Async, no query latency impact
- Rate limiting: <1ms check
- Query filtering: <10% overhead
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Permission(Enum):
    """Permission types"""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    QUERY = "query"
    EXPORT = "export"


class ResourceType(Enum):
    """Resource types"""

    EMBEDDING = "embedding"
    INDEX = "index"
    COLLECTION = "collection"
    SYSTEM = "system"


@dataclass
class User:
    """
    User identity and attributes

    Attributes:
        user_id: Unique identifier
        username: Human-readable name
        email: Email address
        roles: Assigned roles
        attributes: User attributes for ABAC
        tenant_id: Tenant identifier for multi-tenancy
        created_at: Account creation time
        last_login: Last login time
    """

    user_id: str
    username: str
    email: str
    roles: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    tenant_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None


@dataclass
class Role:
    """
    Role-based access control role

    Attributes:
        role_id: Role identifier
        name: Role name
        permissions: List of granted permissions
        resource_patterns: Resource patterns this role can access
        constraints: Additional constraints (time, rate limits)
    """

    role_id: str
    name: str
    permissions: List[Permission] = field(default_factory=list)
    resource_patterns: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessPolicy:
    """
    Attribute-based access control policy

    Attributes:
        policy_id: Policy identifier
        name: Policy name
        effect: Allow or deny
        principals: Who this applies to (users, roles)
        actions: What actions are allowed
        resources: What resources can be accessed
        conditions: When policy applies
    """

    policy_id: str
    name: str
    effect: str  # "allow" or "deny"
    principals: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditLogEntry:
    """
    Audit log entry for access tracking

    Attributes:
        log_id: Unique log entry ID
        timestamp: When action occurred
        user_id: Who performed action
        action: What action was performed
        resource_type: Type of resource accessed
        resource_id: Which resource was accessed
        result: Success or failure
        metadata: Additional context
        query_details: For query actions, the query details
        ip_address: Source IP address
        user_agent: Client user agent
    """

    log_id: str
    timestamp: datetime
    user_id: str
    action: str
    resource_type: ResourceType
    resource_id: str
    result: str  # "success", "failure", "denied"
    metadata: Dict[str, Any] = field(default_factory=dict)
    query_details: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class QueryQuota:
    """
    Query quota for rate limiting

    Attributes:
        user_id: User identifier
        queries_per_hour: Hourly query limit
        queries_per_day: Daily query limit
        max_result_size: Maximum results per query
        max_concurrent: Maximum concurrent queries
        current_hour_count: Queries this hour
        current_day_count: Queries today
        reset_time: When quotas reset
    """

    user_id: str
    queries_per_hour: int = 1000
    queries_per_day: int = 10000
    max_result_size: int = 100
    max_concurrent: int = 10
    current_hour_count: int = 0
    current_day_count: int = 0
    reset_time: datetime = field(default_factory=datetime.now)


class AccessControlEngine:
    """
    Access control engine for embedding systems

    Implements:
    - Authentication: Verify user identity
    - Authorization: Check permissions
    - Row-level security: Filter queries by attributes
    - Rate limiting: Enforce quotas
    - Audit logging: Record all access
    """

    def __init__(self):
        # User database
        self.users: Dict[str, User] = {}

        # Role definitions
        self.roles: Dict[str, Role] = {}

        # Access policies
        self.policies: List[AccessPolicy] = []

        # Query quotas
        self.quotas: Dict[str, QueryQuota] = {}

        # Audit log (in production: use database or log aggregation)
        self.audit_log: List[AuditLogEntry] = []

        # Active sessions
        self.sessions: Dict[str, Dict[str, Any]] = {}

        # Initialize default roles
        self._initialize_default_roles()

        print("Access Control Engine initialized")

    def _initialize_default_roles(self):
        """Create default roles"""

        # Admin role: Full access
        self.roles["admin"] = Role(
            role_id="admin",
            name="Administrator",
            permissions=list(Permission),
            resource_patterns=["*"],
            constraints={},
        )

        # Analyst role: Read and query access
        self.roles["analyst"] = Role(
            role_id="analyst",
            name="Data Analyst",
            permissions=[Permission.READ, Permission.QUERY],
            resource_patterns=["embedding:*", "collection:*"],
            constraints={"max_result_size": 1000, "queries_per_hour": 100},
        )

        # Service role: Query-only access
        self.roles["service"] = Role(
            role_id="service",
            name="Application Service",
            permissions=[Permission.QUERY],
            resource_patterns=["embedding:*"],
            constraints={"max_result_size": 100, "queries_per_hour": 10000},
        )

        # Auditor role: Read audit logs only
        self.roles["auditor"] = Role(
            role_id="auditor",
            name="Security Auditor",
            permissions=[Permission.READ],
            resource_patterns=["audit:*"],
            constraints={},
        )

    def create_user(
        self,
        username: str,
        email: str,
        roles: List[str],
        tenant_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> User:
        """
        Create new user

        Args:
            username: Username
            email: Email address
            roles: List of role IDs
            tenant_id: Tenant identifier
            attributes: User attributes

        Returns:
            Created user
        """
        user_id = hashlib.sha256(f"{username}:{email}".encode()).hexdigest()[:16]

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles,
            attributes=attributes or {},
            tenant_id=tenant_id,
        )

        self.users[user_id] = user

        # Initialize quota
        self.quotas[user_id] = QueryQuota(user_id=user_id)

        # Audit log
        self._log_access(
            user_id=user_id,
            action="user_created",
            resource_type=ResourceType.SYSTEM,
            resource_id=user_id,
            result="success",
        )

        return user

    def authenticate(self, api_key: str, ip_address: Optional[str] = None) -> Optional[User]:
        """
        Authenticate user via API key

        In production:
        - Use OAuth2 tokens (JWT)
        - Verify token signature
        - Check expiration
        - Support multiple auth methods

        Args:
            api_key: API key or token
            ip_address: Client IP address

        Returns:
            User if authenticated, None otherwise
        """
        # Simplified: API key is just user_id
        # In production: verify JWT, check revocation, etc.

        user_id = api_key
        user = self.users.get(user_id)

        if user:
            user.last_login = datetime.now()

            self._log_access(
                user_id=user_id,
                action="authenticate",
                resource_type=ResourceType.SYSTEM,
                resource_id="auth",
                result="success",
                metadata={"ip_address": ip_address},
            )
        else:
            self._log_access(
                user_id=api_key,
                action="authenticate",
                resource_type=ResourceType.SYSTEM,
                resource_id="auth",
                result="failure",
                metadata={"ip_address": ip_address},
            )

        return user

    def authorize(
        self, user: User, action: Permission, resource_type: ResourceType, resource_id: str
    ) -> bool:
        """
        Check if user is authorized for action on resource

        Steps:
        1. Check user roles
        2. Check role permissions
        3. Check resource patterns
        4. Evaluate ABAC policies
        5. Apply deny policies

        Args:
            user: User requesting access
            action: Requested permission
            resource_type: Type of resource
            resource_id: Specific resource

        Returns:
            True if authorized, False otherwise
        """
        # Check each role
        for role_id in user.roles:
            role = self.roles.get(role_id)
            if not role:
                continue

            # Check if action is permitted
            if action not in role.permissions and Permission.ADMIN not in role.permissions:
                continue

            # Check resource patterns
            resource_str = f"{resource_type.value}:{resource_id}"
            if not self._match_resource_pattern(resource_str, role.resource_patterns):
                continue

            # Role grants access
            self._log_access(
                user_id=user.user_id,
                action=f"authorize_{action.value}",
                resource_type=resource_type,
                resource_id=resource_id,
                result="success",
            )
            return True

        # Check ABAC policies
        if self._evaluate_policies(user, action, resource_type, resource_id):
            return True

        # Access denied
        self._log_access(
            user_id=user.user_id,
            action=f"authorize_{action.value}",
            resource_type=resource_type,
            resource_id=resource_id,
            result="denied",
        )
        return False

    def _match_resource_pattern(self, resource: str, patterns: List[str]) -> bool:
        """
        Check if resource matches any pattern

        Supports wildcards:
        - "*" matches anything
        - "embedding:*" matches all embeddings
        - "collection:customer_*" matches customer collections

        Args:
            resource: Resource to check
            patterns: Allowed patterns

        Returns:
            True if matches
        """
        for pattern in patterns:
            if pattern == "*":
                return True

            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if resource.startswith(prefix):
                    return True

            if pattern == resource:
                return True

        return False

    def _evaluate_policies(
        self, user: User, action: Permission, resource_type: ResourceType, resource_id: str
    ) -> bool:
        """
        Evaluate ABAC policies

        Args:
            user: User
            action: Action
            resource_type: Resource type
            resource_id: Resource ID

        Returns:
            True if any allow policy matches
        """
        for policy in self.policies:
            # Check if policy applies to this user
            if user.user_id not in policy.principals and "*" not in policy.principals:
                continue

            # Check if action is covered
            if action.value not in policy.actions and "*" not in policy.actions:
                continue

            # Check if resource matches
            resource_str = f"{resource_type.value}:{resource_id}"
            if not self._match_resource_pattern(resource_str, policy.resources):
                continue

            # Evaluate conditions
            if policy.conditions:
                if not self._evaluate_conditions(user, policy.conditions):
                    continue

            # Policy applies
            if policy.effect == "allow":
                return True
            elif policy.effect == "deny":
                return False

        return False

    def _evaluate_conditions(self, user: User, conditions: Dict[str, Any]) -> bool:
        """
        Evaluate policy conditions

        Examples:
        - "time": {"after": "09:00", "before": "17:00"}
        - "attribute": {"region": "US"}
        - "tenant": {"equals": "tenant_123"}

        Args:
            user: User
            conditions: Condition dictionary

        Returns:
            True if all conditions met
        """
        for key, value in conditions.items():
            if key == "tenant":
                if user.tenant_id != value.get("equals"):
                    return False

            elif key == "attribute":
                for attr_key, attr_value in value.items():
                    if user.attributes.get(attr_key) != attr_value:
                        return False

            elif key == "time":
                now = datetime.now().time()
                if "after" in value:
                    after = datetime.strptime(value["after"], "%H:%M").time()
                    if now < after:
                        return False
                if "before" in value:
                    before = datetime.strptime(value["before"], "%H:%M").time()
                    if now > before:
                        return False

        return True

    def check_quota(self, user: User, result_size: int = 10) -> Tuple[bool, str]:
        """
        Check if user has remaining quota

        Args:
            user: User
            result_size: Number of results requested

        Returns:
            (allowed, reason) tuple
        """
        quota = self.quotas.get(user.user_id)
        if not quota:
            return False, "No quota found"

        # Reset counters if needed
        now = datetime.now()
        if now >= quota.reset_time:
            quota.current_hour_count = 0
            quota.reset_time = now + timedelta(hours=1)

        if now.date() != quota.reset_time.date():
            quota.current_day_count = 0

        # Check hourly limit
        if quota.current_hour_count >= quota.queries_per_hour:
            return False, "Hourly quota exceeded"

        # Check daily limit
        if quota.current_day_count >= quota.queries_per_day:
            return False, "Daily quota exceeded"

        # Check result size
        if result_size > quota.max_result_size:
            return False, f"Result size exceeds limit ({quota.max_result_size})"

        return True, "OK"

    def increment_quota(self, user: User):
        """Increment user's query count"""
        quota = self.quotas.get(user.user_id)
        if quota:
            quota.current_hour_count += 1
            quota.current_day_count += 1

    def apply_row_level_security(self, user: User, query_filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply row-level security filters based on user attributes

        Injects additional filters based on:
        - Tenant ID (multi-tenancy isolation)
        - User attributes (region, department, etc.)
        - Data classification level

        Args:
            user: User
            query_filter: Original query filter

        Returns:
            Modified filter with security constraints
        """
        secure_filter = query_filter.copy()

        # Enforce tenant isolation
        if user.tenant_id:
            secure_filter["tenant_id"] = user.tenant_id

        # Enforce attribute-based filters
        if "region" in user.attributes:
            secure_filter["region"] = user.attributes["region"]

        if "department" in user.attributes:
            secure_filter["department"] = user.attributes["department"]

        # Enforce classification level
        if "clearance_level" in user.attributes:
            clearance = user.attributes["clearance_level"]
            secure_filter["classification"] = {"$lte": clearance}

        return secure_filter

    def _log_access(
        self,
        user_id: str,
        action: str,
        resource_type: ResourceType,
        resource_id: str,
        result: str,
        metadata: Optional[Dict[str, Any]] = None,
        query_details: Optional[Dict[str, Any]] = None,
    ):
        """
        Log access attempt to audit trail

        Args:
            user_id: User ID
            action: Action performed
            resource_type: Resource type
            resource_id: Resource ID
            result: Result (success/failure/denied)
            metadata: Additional metadata
            query_details: Query details if applicable
        """
        log_entry = AuditLogEntry(
            log_id=hashlib.sha256(f"{user_id}:{action}:{time.time()}".encode()).hexdigest()[:16],
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            result=result,
            metadata=metadata or {},
            query_details=query_details,
        )

        self.audit_log.append(log_entry)

        # In production: write to immutable storage
        # - Append-only database
        # - Write to SIEM (Splunk, ELK)
        # - Store in S3 with versioning
        # - Use blockchain for tamper-proof audit

    def query_audit_log(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        result: Optional[str] = None,
    ) -> List[AuditLogEntry]:
        """
        Query audit log

        Args:
            user_id: Filter by user
            action: Filter by action
            start_time: Start time
            end_time: End time
            result: Filter by result

        Returns:
            Matching audit entries
        """
        results = self.audit_log

        if user_id:
            results = [e for e in results if e.user_id == user_id]

        if action:
            results = [e for e in results if e.action == action]

        if start_time:
            results = [e for e in results if e.timestamp >= start_time]

        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        if result:
            results = [e for e in results if e.result == result]

        return results


# Example usage
def access_control_example():
    """
    Demonstrate access control and audit trails
    """
    print("=== Access Control and Audit Trails ===")
    print()

    # Initialize access control engine
    ac = AccessControlEngine()
    print()

    # Create users with different roles
    admin_user = ac.create_user(
        username="alice_admin", email="alice@example.com", roles=["admin"], tenant_id="tenant_acme"
    )

    analyst_user = ac.create_user(
        username="bob_analyst",
        email="bob@example.com",
        roles=["analyst"],
        tenant_id="tenant_acme",
        attributes={"region": "US", "department": "marketing"},
    )

    service_user = ac.create_user(
        username="api_service",
        email="service@example.com",
        roles=["service"],
        tenant_id="tenant_acme",
    )

    print("Created users:")
    print(f"  Admin: {admin_user.username} (roles: {admin_user.roles})")
    print(f"  Analyst: {analyst_user.username} (roles: {analyst_user.roles})")
    print(f"  Service: {service_user.username} (roles: {service_user.roles})")
    print()

    # Test authorization
    print("Authorization tests:")

    # Admin can do everything
    can_delete = ac.authorize(admin_user, Permission.DELETE, ResourceType.EMBEDDING, "emb_123")
    print(f"  Admin delete embedding: {can_delete}")

    # Analyst can query but not delete
    can_query = ac.authorize(analyst_user, Permission.QUERY, ResourceType.EMBEDDING, "emb_123")
    can_delete = ac.authorize(analyst_user, Permission.DELETE, ResourceType.EMBEDDING, "emb_123")
    print(f"  Analyst query embedding: {can_query}")
    print(f"  Analyst delete embedding: {can_delete}")

    # Service can only query
    can_query = ac.authorize(service_user, Permission.QUERY, ResourceType.EMBEDDING, "emb_123")
    can_export = ac.authorize(service_user, Permission.EXPORT, ResourceType.EMBEDDING, "emb_123")
    print(f"  Service query embedding: {can_query}")
    print(f"  Service export embedding: {can_export}")
    print()

    # Test quota enforcement
    print("Quota checks:")
    allowed, reason = ac.check_quota(analyst_user, result_size=50)
    print(f"  Analyst query (k=50): {allowed} ({reason})")

    allowed, reason = ac.check_quota(analyst_user, result_size=5000)
    print(f"  Analyst query (k=5000): {allowed} ({reason})")
    print()

    # Test row-level security
    print("Row-level security:")
    query_filter = {"category": "products"}
    secure_filter = ac.apply_row_level_security(analyst_user, query_filter)
    print(f"  Original filter: {query_filter}")
    print(f"  Secure filter: {secure_filter}")
    print()

    # Query audit log
    print("Audit log (last 5 entries):")
    recent_logs = ac.query_audit_log()[-5:]
    for log in recent_logs:
        print(
            f"  [{log.timestamp.strftime('%H:%M:%S')}] "
            f"{log.user_id[:8]}... {log.action} "
            f"{log.resource_type.value}:{log.resource_id} -> {log.result}"
        )


if __name__ == "__main__":
    access_control_example()
