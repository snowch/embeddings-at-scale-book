from collections import defaultdict

# Code from Chapter 23
# Book: Embeddings at Scale

"""
Network Optimization for Distributed Vector Search

Architecture:
1. Global query router: Direct queries to optimal datacenter
2. Sharding strategy: Partition embeddings across regions
3. Replication policy: Replicate hot data globally
4. Query aggregator: Batch queries for efficiency
5. Compression: Reduce network transfer

Techniques:
- Geo-routing: Route based on user location and data availability
- Hot data replication: Popular embeddings in all datacenters
- Cold data sharding: Infrequent embeddings partitioned by region
- Query batching: Amortize network latency
- Result compression: Compress query results before transfer
- Prefetching: Predictive loading of likely-needed data

Performance targets:
- p99 latency: <100ms globally
- Bandwidth: <10GB/sec per datacenter
- Replication lag: <5 seconds
- Failover time: <30 seconds
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Datacenter:
    """
    Datacenter configuration
    
    Attributes:
        name: Datacenter identifier (us-west-1, eu-central-1, etc.)
        region: Geographic region
        capacity_vectors: Max vectors this datacenter can store
        capacity_qps: Max queries per second
        current_vectors: Current vector count
        current_qps: Current query load
        latency_map: Latency to other datacenters (ms)
        bandwidth_gbps: Network bandwidth available
        cost_per_gb: Egress cost per GB
    """
    name: str
    region: str
    capacity_vectors: int
    capacity_qps: float
    current_vectors: int = 0
    current_qps: float = 0.0
    latency_map: Dict[str, float] = field(default_factory=dict)
    bandwidth_gbps: float = 10.0
    cost_per_gb: float = 0.02

@dataclass
class ShardingStrategy:
    """
    Strategy for partitioning embeddings across datacenters
    
    Attributes:
        strategy_type: Type (hash, range, geo, semantic, hybrid)
        shard_count: Number of shards
        replication_factor: How many copies of each shard
        shard_assignments: Which shards in which datacenters
        hot_threshold: Access rate threshold for replication
    """
    strategy_type: str
    shard_count: int
    replication_factor: int
    shard_assignments: Dict[int, List[str]] = field(default_factory=dict)
    hot_threshold: float = 100.0  # Queries per hour

class QueryRouter:
    """
    Global query routing to optimal datacenter
    
    Routes queries based on:
    - User location (minimize latency)
    - Data location (where embeddings are)
    - Datacenter load (avoid overloaded nodes)
    - Cost (prefer cheaper regions when latency acceptable)
    """

    def __init__(
        self,
        datacenters: List[Datacenter],
        sharding: ShardingStrategy
    ):
        self.datacenters = {dc.name: dc for dc in datacenters}
        self.sharding = sharding

        # Track query patterns for optimization
        self.query_history: List[Dict[str, Any]] = []
        self.datacenter_loads: Dict[str, float] = defaultdict(float)

    def route_query(
        self,
        query_id: str,
        embedding_ids: List[str],
        user_region: str,
        latency_budget_ms: float = 100.0
    ) -> Tuple[str, List[str]]:
        """
        Route query to optimal datacenter(s)
        
        Args:
            query_id: Query identifier
            embedding_ids: Embeddings needed for query
            user_region: User's geographic region
            latency_budget_ms: Maximum acceptable latency
        
        Returns:
            primary_dc: Primary datacenter to handle query
            backup_dcs: Backup datacenters (for failover)
        """
        # Determine which datacenters have the needed embeddings
        candidate_dcs = self._find_datacenters_with_data(embedding_ids)

        if not candidate_dcs:
            raise ValueError("No datacenter contains required embeddings")

        # Score each candidate
        scores = []
        for dc_name in candidate_dcs:
            score = self._score_datacenter(
                dc_name,
                user_region,
                latency_budget_ms
            )
            scores.append((score, dc_name))

        # Sort by score (higher is better)
        scores.sort(reverse=True, key=lambda x: x[0])

        # Select primary and backups
        primary_dc = scores[0][1]
        backup_dcs = [dc for _, dc in scores[1:3]]  # Top 2 backups

        # Update load tracking
        self.datacenter_loads[primary_dc] += 1

        return primary_dc, backup_dcs

    def _find_datacenters_with_data(
        self,
        embedding_ids: List[str]
    ) -> Set[str]:
        """
        Find datacenters that have all required embeddings
        
        Args:
            embedding_ids: Required embedding IDs
        
        Returns:
            datacenters: Set of datacenter names with all data
        """
        # Determine which shards contain the embeddings
        needed_shards = set()
        for emb_id in embedding_ids:
            shard_id = self._get_shard_id(emb_id)
            needed_shards.add(shard_id)

        # Find datacenters with all needed shards
        candidate_dcs = None
        for shard_id in needed_shards:
            dcs_with_shard = set(self.sharding.shard_assignments.get(shard_id, []))

            if candidate_dcs is None:
                candidate_dcs = dcs_with_shard
            else:
                # Intersection - must have ALL shards
                candidate_dcs &= dcs_with_shard

        return candidate_dcs if candidate_dcs else set()

    def _get_shard_id(self, embedding_id: str) -> int:
        """
        Determine which shard contains this embedding
        
        Uses consistent hashing for even distribution
        """
        if self.sharding.strategy_type == "hash":
            # Hash-based sharding
            hash_val = int(
                hashlib.sha256(embedding_id.encode()).hexdigest()[:8],
                16
            )
            return hash_val % self.sharding.shard_count
        elif self.sharding.strategy_type == "range":
            # Range-based sharding (e.g., by ID prefix)
            # Simplified: use first character
            return ord(embedding_id[0]) % self.sharding.shard_count
        else:
            # Default to hash
            return 0

    def _score_datacenter(
        self,
        dc_name: str,
        user_region: str,
        latency_budget: float
    ) -> float:
        """
        Score datacenter for query routing
        
        Higher score = better choice
        
        Factors:
        - Latency to user (lower is better)
        - Current load (lower is better)
        - Cost (lower is better)
        """
        dc = self.datacenters[dc_name]

        score = 100.0  # Start at 100

        # Latency penalty
        latency = self._estimate_latency(dc_name, user_region)
        if latency > latency_budget:
            score -= 50  # Heavy penalty for exceeding budget
        else:
            # Linear penalty within budget
            score -= 20 * (latency / latency_budget)

        # Load penalty
        load_ratio = dc.current_qps / dc.capacity_qps
        score -= 30 * load_ratio

        # Cost consideration (minor)
        score -= dc.cost_per_gb * 0.1

        return max(score, 0)

    def _estimate_latency(
        self,
        dc_name: str,
        user_region: str
    ) -> float:
        """
        Estimate latency from user region to datacenter
        
        Returns latency in milliseconds
        """
        dc = self.datacenters[dc_name]

        # Check latency map
        if user_region in dc.latency_map:
            return dc.latency_map[user_region]

        # Default estimates by region
        if dc.region == user_region:
            return 10.0  # Same region
        elif self._same_continent(dc.region, user_region):
            return 50.0  # Same continent
        else:
            return 150.0  # Cross-continent

    def _same_continent(self, region1: str, region2: str) -> bool:
        """Check if regions on same continent"""
        # Simplified continent mapping
        continents = {
            'us-west': 'north-america',
            'us-east': 'north-america',
            'eu-west': 'europe',
            'eu-central': 'europe',
            'ap-southeast': 'asia',
            'ap-northeast': 'asia',
        }
        return continents.get(region1) == continents.get(region2)

class ReplicationManager:
    """
    Manage replication of hot embeddings across datacenters
    
    Hot data replicated globally for low latency
    Cold data sharded to save storage/bandwidth
    """

    def __init__(
        self,
        datacenters: List[Datacenter],
        hot_threshold_qps: float = 10.0
    ):
        self.datacenters = datacenters
        self.hot_threshold = hot_threshold_qps

        # Track access patterns
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.access_rates: Dict[str, float] = defaultdict(float)

        # Replication state
        self.replicated_embeddings: Dict[str, Set[str]] = defaultdict(set)

    def record_access(
        self,
        embedding_id: str,
        datacenter: str
    ) -> None:
        """
        Record embedding access for replication decisions
        
        Args:
            embedding_id: Accessed embedding
            datacenter: Datacenter that served it
        """
        self.access_counts[embedding_id] += 1

        # Update access rate (exponential moving average)
        alpha = 0.1
        current_rate = self.access_rates[embedding_id]
        self.access_rates[embedding_id] = (
            alpha * 1.0 + (1 - alpha) * current_rate
        )

    def should_replicate(self, embedding_id: str) -> bool:
        """
        Determine if embedding should be replicated globally
        
        Args:
            embedding_id: Embedding to check
        
        Returns:
            should_replicate: Whether to replicate
        """
        access_rate = self.access_rates[embedding_id]
        return access_rate >= self.hot_threshold

    def get_replication_plan(
        self,
        max_replications: int = 1000
    ) -> List[Tuple[str, List[str]]]:
        """
        Generate replication plan for hot embeddings
        
        Args:
            max_replications: Maximum replications to perform
        
        Returns:
            plan: List of (embedding_id, target_datacenters)
        """
        # Find hot embeddings not yet fully replicated
        plan = []

        # Sort by access rate
        sorted_embeddings = sorted(
            self.access_rates.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for embedding_id, access_rate in sorted_embeddings[:max_replications]:
            if access_rate < self.hot_threshold:
                break

            # Find datacenters that don't have this embedding
            current_dcs = self.replicated_embeddings[embedding_id]
            all_dcs = {dc.name for dc in self.datacenters}
            missing_dcs = all_dcs - current_dcs

            if missing_dcs:
                plan.append((embedding_id, list(missing_dcs)))

        return plan

class QueryBatcher:
    """
    Batch multiple queries for network efficiency
    
    Instead of N round-trips, make 1 round-trip with N queries
    Amortizes network latency across queries
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_ms: float = 10.0
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self.pending_queries: List[Dict[str, Any]] = []
        self.batch_start_time: Optional[datetime] = None

    def add_query(
        self,
        query_id: str,
        query_data: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Add query to batch
        
        Returns batch if ready to execute, None otherwise
        
        Args:
            query_id: Query identifier
            query_data: Query parameters
        
        Returns:
            batch: Batch of queries if ready, None otherwise
        """
        # Add to pending
        self.pending_queries.append({
            'query_id': query_id,
            'data': query_data,
            'arrival_time': datetime.now()
        })

        # Start timer if first query
        if self.batch_start_time is None:
            self.batch_start_time = datetime.now()

        # Check if batch ready
        if self._should_execute_batch():
            return self._get_batch()

        return None

    def _should_execute_batch(self) -> bool:
        """Determine if batch should execute"""
        # Execute if reached max size
        if len(self.pending_queries) >= self.max_batch_size:
            return True

        # Execute if waited too long
        if self.batch_start_time:
            elapsed = (
                datetime.now() - self.batch_start_time
            ).total_seconds() * 1000
            if elapsed >= self.max_wait_ms:
                return True

        return False

    def _get_batch(self) -> List[Dict[str, Any]]:
        """Get current batch and reset"""
        batch = self.pending_queries
        self.pending_queries = []
        self.batch_start_time = None
        return batch

class NetworkOptimizer:
    """
    Comprehensive network optimization
    
    Combines routing, replication, batching, and compression
    """

    def __init__(
        self,
        datacenters: List[Datacenter],
        sharding: ShardingStrategy
    ):
        self.router = QueryRouter(datacenters, sharding)
        self.replication = ReplicationManager(datacenters)
        self.batcher = QueryBatcher()

    def optimize_query(
        self,
        query_id: str,
        query_data: Dict[str, Any],
        user_region: str
    ) -> Dict[str, Any]:
        """
        Optimize query execution across network
        
        Args:
            query_id: Query identifier
            query_data: Query parameters
            user_region: User's region
        
        Returns:
            execution_plan: Optimized execution plan
        """
        # Route query
        primary_dc, backup_dcs = self.router.route_query(
            query_id=query_id,
            embedding_ids=query_data.get('embedding_ids', []),
            user_region=user_region
        )

        # Check if should batch
        batch = self.batcher.add_query(query_id, query_data)

        # Create execution plan
        plan = {
            'query_id': query_id,
            'primary_datacenter': primary_dc,
            'backup_datacenters': backup_dcs,
            'batched': batch is not None,
            'batch_size': len(batch) if batch else 1,
        }

        return plan
