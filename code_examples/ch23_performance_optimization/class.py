# Code from Chapter 23
# Book: Embeddings at Scale

"""
Query Optimization for Vector Similarity Search

Architecture:
1. Query analysis: Determine query characteristics, select strategy
2. Multi-stage retrieval: Coarse → medium → fine filtering
3. Index selection: Choose optimal index for query pattern
4. Parallel execution: Distribute work across cores/nodes
5. Result reranking: Refine top-k with exact distances

Techniques:
- Approximate nearest neighbor (ANN): HNSW, IVF, LSH
- Multi-stage filtering: Reduce candidates progressively
- Query-aware index selection: Adapt to query characteristics
- Parallel query execution: Leverage multiple cores/GPUs
- Result caching: Avoid redundant computation
- Early termination: Stop when quality threshold met
- Adaptive batching: Group similar queries for efficiency

Performance targets:
- Latency: p50 < 20ms, p99 < 50ms for 10M+ embeddings
- Throughput: 10,000+ queries/second per node
- Recall: >95% for top-10, >98% for top-100
- Cost: <$0.10 per million queries
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class QueryPlan:
    """
    Execution plan for vector similarity query
    
    Attributes:
        query_id: Unique query identifier
        query_vector: Query embedding
        k: Number of results to return
        filters: Metadata filters to apply
        min_score: Minimum similarity score threshold
        strategy: Execution strategy (exact, hnsw, ivf, hybrid)
        index_name: Index to use
        use_cache: Whether to check cache first
        parallel_degree: Number of parallel workers
        timeout_ms: Maximum execution time
        explain: Return execution statistics
    """
    query_id: str
    query_vector: np.ndarray
    k: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)
    min_score: float = 0.0
    strategy: str = "auto"  # "exact", "hnsw", "ivf", "hybrid", "auto"
    index_name: Optional[str] = None
    use_cache: bool = True
    parallel_degree: int = 4
    timeout_ms: int = 50
    explain: bool = False

@dataclass
class QueryResult:
    """
    Query execution result with statistics
    
    Attributes:
        query_id: Query identifier
        results: List of (id, score) tuples
        execution_time_ms: Total execution time
        candidates_scanned: Number of candidates examined
        exact_distances_computed: Number of exact distance calculations
        cache_hit: Whether result from cache
        strategy_used: Actual strategy used
        index_used: Index used for retrieval
        recall_estimate: Estimated recall (if ground truth available)
        explain_info: Detailed execution statistics
    """
    query_id: str
    results: List[Tuple[str, float]]
    execution_time_ms: float
    candidates_scanned: int
    exact_distances_computed: int
    cache_hit: bool = False
    strategy_used: str = ""
    index_used: str = ""
    recall_estimate: Optional[float] = None
    explain_info: Dict[str, Any] = field(default_factory=dict)

class QueryOptimizer:
    """
    Intelligent query optimization for vector similarity search
    
    Analyzes query characteristics and selects optimal execution strategy
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.query_history = []
        self.performance_stats = defaultdict(list)

    def analyze_query(self, plan: QueryPlan) -> Dict[str, Any]:
        """
        Analyze query characteristics to inform optimization
        
        Returns:
            analysis: Query characteristics and recommendations
        """
        analysis = {
            'query_id': plan.query_id,
            'vector_dim': len(plan.query_vector),
            'k_value': plan.k,
            'has_filters': bool(plan.filters),
            'filter_selectivity': self._estimate_filter_selectivity(plan.filters),
            'recommended_strategy': None,
            'recommended_index': None,
            'estimated_candidates': 0,
        }

        # Estimate filter selectivity
        if plan.filters:
            selectivity = analysis['filter_selectivity']
            # High selectivity (filters remove >90%) → scan filtered subset
            if selectivity < 0.1:
                analysis['recommended_strategy'] = 'filtered_scan'
                analysis['estimated_candidates'] = int(
                    self.config.get('total_vectors', 1e9) * selectivity
                )
            # Medium selectivity → HNSW with post-filtering
            elif selectivity < 0.5:
                analysis['recommended_strategy'] = 'hnsw_postfilter'
                analysis['estimated_candidates'] = plan.k * 10
            # Low selectivity → standard ANN
            else:
                analysis['recommended_strategy'] = 'hnsw'
                analysis['estimated_candidates'] = plan.k * 5
        else:
            # No filters - strategy based on k and dataset size
            if plan.k < 10:
                analysis['recommended_strategy'] = 'hnsw'
                analysis['estimated_candidates'] = plan.k * 5
            elif plan.k < 100:
                analysis['recommended_strategy'] = 'ivf_pq'
                analysis['estimated_candidates'] = plan.k * 10
            else:
                analysis['recommended_strategy'] = 'hybrid'
                analysis['estimated_candidates'] = plan.k * 20

        # Select optimal index
        analysis['recommended_index'] = self._select_index(
            analysis['recommended_strategy'],
            plan.filters
        )

        return analysis

    def _estimate_filter_selectivity(self, filters: Dict[str, Any]) -> float:
        """
        Estimate what fraction of vectors pass filters
        
        Uses statistics from previous queries and metadata distributions
        """
        if not filters:
            return 1.0

        # In production, query metadata statistics
        # For demonstration, use heuristics
        selectivity = 1.0
        for field, value in filters.items():
            if isinstance(value, list):
                # IN clause - estimate from list length and cardinality
                field_cardinality = self.config.get(
                    f'{field}_cardinality', 1000
                )
                selectivity *= min(len(value) / field_cardinality, 1.0)
            elif isinstance(value, tuple):
                # Range query - estimate from range width
                selectivity *= 0.1  # Conservative estimate
            else:
                # Equality - estimate from cardinality
                field_cardinality = self.config.get(
                    f'{field}_cardinality', 100
                )
                selectivity *= 1.0 / field_cardinality

        return max(selectivity, 0.0001)  # At least 0.01% selectivity

    def _select_index(
        self,
        strategy: str,
        filters: Dict[str, Any]
    ) -> str:
        """
        Select optimal index based on strategy and filters
        """
        if strategy == 'filtered_scan':
            return 'metadata_index'
        elif strategy == 'hnsw' or strategy == 'hnsw_postfilter':
            return 'hnsw_index'
        elif strategy == 'ivf_pq':
            return 'ivf_pq_index'
        elif strategy == 'hybrid':
            return 'hnsw_index'  # Start with HNSW, fall back if needed
        else:
            return 'hnsw_index'  # Default

    def optimize_k_value(self, k: int, strategy: str) -> int:
        """
        Adjust k value based on strategy to maintain recall
        
        ANN methods need to retrieve more candidates than k
        to achieve target recall after filtering
        """
        if strategy == 'exact':
            return k
        elif strategy == 'hnsw':
            # HNSW typically needs 1.5-2× k for 95% recall
            return int(k * 1.5)
        elif strategy == 'ivf_pq':
            # IVF-PQ needs 3-5× k for 95% recall
            return int(k * 3)
        elif strategy == 'filtered_scan':
            # Filtered scan is exact
            return k
        else:
            # Conservative default
            return int(k * 2)

    def estimate_query_cost(
        self,
        plan: QueryPlan,
        analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Estimate computational cost of query execution
        
        Returns:
            costs: Estimated CPU time, memory, I/O operations
        """
        strategy = analysis['recommended_strategy']
        candidates = analysis['estimated_candidates']

        costs = {
            'cpu_ms': 0.0,
            'memory_mb': 0.0,
            'io_operations': 0,
            'network_mb': 0.0,
        }

        # CPU cost (distance computations)
        vector_dim = len(plan.query_vector)
        if strategy == 'exact':
            # Full scan cost
            total_vectors = self.config.get('total_vectors', 1e9)
            costs['cpu_ms'] = total_vectors * vector_dim * 1e-6  # μs per distance
            costs['memory_mb'] = total_vectors * vector_dim * 4 / 1024**2  # float32
        elif strategy in ['hnsw', 'hnsw_postfilter']:
            # HNSW graph traversal cost
            costs['cpu_ms'] = candidates * vector_dim * 2e-6  # 2μs per comparison
            costs['memory_mb'] = candidates * vector_dim * 4 / 1024**2
            costs['io_operations'] = candidates // 100  # Graph structure reads
        elif strategy == 'ivf_pq':
            # IVF-PQ quantized distance cost
            costs['cpu_ms'] = candidates * 0.1e-6  # Quantized distances faster
            costs['memory_mb'] = candidates * 64 / 1024**2  # PQ codes smaller
            costs['io_operations'] = 100  # Coarse quantizer + cluster reads
        elif strategy == 'filtered_scan':
            # Scan filtered subset
            filtered_vectors = int(
                self.config.get('total_vectors', 1e9) *
                analysis['filter_selectivity']
            )
            costs['cpu_ms'] = filtered_vectors * vector_dim * 1e-6
            costs['memory_mb'] = filtered_vectors * vector_dim * 4 / 1024**2
            costs['io_operations'] = filtered_vectors // 10000

        # Network cost for distributed queries
        if self.config.get('distributed', False):
            costs['network_mb'] = (
                len(plan.query_vector) * 4 / 1024**2 +  # Query vector
                plan.k * (vector_dim * 4 + 16) / 1024**2  # Results
            )

        return costs

class MultiStageRetrieval:
    """
    Multi-stage retrieval pipeline: coarse → medium → fine filtering
    
    Progressively narrows candidate set while maintaining recall
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimizer = QueryOptimizer(config)

    def execute_query(
        self,
        plan: QueryPlan,
        stages: Optional[List[str]] = None
    ) -> QueryResult:
        """
        Execute multi-stage retrieval query
        
        Args:
            plan: Query execution plan
            stages: Override default stages (for testing)
        
        Returns:
            result: Query results with execution statistics
        """
        start_time = time.time()

        # Analyze query
        analysis = self.optimizer.analyze_query(plan)

        # Select execution stages
        if stages is None:
            stages = self._select_stages(plan, analysis)

        # Execute stages
        candidates = set()
        candidates_scanned = 0
        exact_distances = 0
        explain_info = {
            'stages': [],
            'analysis': analysis,
        }

        for stage_name in stages:
            stage_start = time.time()

            if stage_name == 'coarse_ivf':
                # Stage 1: IVF coarse filtering
                stage_candidates = self._coarse_ivf_stage(
                    plan.query_vector,
                    k=min(plan.k * 100, 10000)
                )
                candidates.update(stage_candidates)
                candidates_scanned += len(stage_candidates)

            elif stage_name == 'hnsw_graph':
                # Stage 2: HNSW graph search
                stage_candidates = self._hnsw_stage(
                    plan.query_vector,
                    k=plan.k * 10
                )
                if candidates:
                    # Intersect with previous stage
                    candidates &= set(stage_candidates)
                else:
                    candidates = set(stage_candidates)
                candidates_scanned += len(stage_candidates)

            elif stage_name == 'pq_filtering':
                # Stage 3: Product quantization refinement
                stage_candidates = self._pq_filtering_stage(
                    plan.query_vector,
                    candidates=list(candidates),
                    k=plan.k * 3
                )
                candidates = set(stage_candidates)
                candidates_scanned += len(stage_candidates)

            elif stage_name == 'exact_reranking':
                # Stage 4: Exact distance reranking
                results = self._exact_reranking_stage(
                    plan.query_vector,
                    candidates=list(candidates),
                    k=plan.k,
                    filters=plan.filters
                )
                exact_distances += len(candidates)

            stage_time = (time.time() - stage_start) * 1000
            explain_info['stages'].append({
                'name': stage_name,
                'time_ms': stage_time,
                'candidates_after': len(candidates) if stage_name != 'exact_reranking' else len(results),
            })

        # Finalize results
        execution_time = (time.time() - start_time) * 1000

        return QueryResult(
            query_id=plan.query_id,
            results=results,
            execution_time_ms=execution_time,
            candidates_scanned=candidates_scanned,
            exact_distances_computed=exact_distances,
            strategy_used=','.join(stages),
            index_used=analysis['recommended_index'],
            explain_info=explain_info if plan.explain else {}
        )

    def _select_stages(
        self,
        plan: QueryPlan,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Select optimal retrieval stages based on query characteristics
        """
        strategy = analysis['recommended_strategy']

        if strategy == 'exact':
            return ['exact_reranking']

        elif strategy == 'hnsw' or strategy == 'hnsw_postfilter':
            return ['hnsw_graph', 'exact_reranking']

        elif strategy == 'ivf_pq':
            return ['coarse_ivf', 'pq_filtering', 'exact_reranking']

        elif strategy == 'hybrid':
            return ['coarse_ivf', 'hnsw_graph', 'pq_filtering', 'exact_reranking']

        elif strategy == 'filtered_scan':
            # Filter first, then search filtered subset
            return ['exact_reranking']  # Filtering happens in reranking

        else:
            # Default: HNSW + reranking
            return ['hnsw_graph', 'exact_reranking']

    def _coarse_ivf_stage(
        self,
        query: np.ndarray,
        k: int
    ) -> List[str]:
        """
        IVF coarse filtering - identify relevant clusters
        
        Returns candidate IDs from top clusters
        """
        # In production, query actual IVF index
        # For demonstration, simulate cluster selection
        n_clusters = self.config.get('ivf_clusters', 4096)
        vectors_per_cluster = self.config.get(
            'total_vectors', 1e9
        ) // n_clusters

        # Simulate selecting top 10 clusters
        n_probe = min(10, n_clusters)
        candidates_per_cluster = k // n_probe

        candidates = []
        for i in range(n_probe):
            # Simulate cluster IDs
            cluster_id = f"cluster_{i}"
            for j in range(candidates_per_cluster):
                candidates.append(f"{cluster_id}_vec_{j}")

        return candidates

    def _hnsw_stage(
        self,
        query: np.ndarray,
        k: int
    ) -> List[str]:
        """
        HNSW graph search - navigate similarity graph
        
        Returns candidate IDs from graph traversal
        """
        # In production, query actual HNSW index
        # For demonstration, simulate graph search
        ef_search = max(k, 100)  # HNSW ef_search parameter

        candidates = []
        for i in range(ef_search):
            candidates.append(f"hnsw_vec_{i}")

        return candidates

    def _pq_filtering_stage(
        self,
        query: np.ndarray,
        candidates: List[str],
        k: int
    ) -> List[str]:
        """
        Product quantization filtering - refine with quantized distances
        
        Returns top-k candidates by PQ distance
        """
        # In production, compute actual PQ distances
        # For demonstration, simulate PQ filtering

        # Simulate PQ distance computation (much faster than exact)
        scored_candidates = []
        for cand_id in candidates[:k]:
            # Random score for demo
            score = np.random.random()
            scored_candidates.append((cand_id, score))

        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        return [cand_id for cand_id, _ in scored_candidates[:k]]

    def _exact_reranking_stage(
        self,
        query: np.ndarray,
        candidates: List[str],
        k: int,
        filters: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """
        Exact distance reranking - compute exact similarities for final results
        
        Returns top-k by exact distance, applying filters
        """
        # In production, fetch actual vectors and compute exact distances
        # For demonstration, simulate exact reranking

        scored_results = []
        for cand_id in candidates:
            # Simulate filter check
            if filters and not self._passes_filters(cand_id, filters):
                continue

            # Simulate exact distance (in production, actual cosine similarity)
            score = np.random.random()
            scored_results.append((cand_id, score))

        # Sort and return top-k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:k]

    def _passes_filters(
        self,
        candidate_id: str,
        filters: Dict[str, Any]
    ) -> bool:
        """
        Check if candidate passes metadata filters
        """
        # In production, query actual metadata
        # For demonstration, simulate filter check
        return np.random.random() > 0.3  # 70% pass rate

class ParallelQueryExecutor:
    """
    Parallel query execution for high throughput
    
    Distributes queries across CPU cores and GPU devices
    """

    def __init__(
        self,
        config: Dict[str, Any],
        n_workers: int = 4
    ):
        self.config = config
        self.n_workers = n_workers
        self.retrieval = MultiStageRetrieval(config)

    def execute_batch(
        self,
        plans: List[QueryPlan]
    ) -> List[QueryResult]:
        """
        Execute batch of queries in parallel
        
        Args:
            plans: List of query plans to execute
        
        Returns:
            results: Query results in same order as plans
        """
        # Group queries by strategy for batching efficiency
        strategy_groups = defaultdict(list)
        for i, plan in enumerate(plans):
            analysis = self.retrieval.optimizer.analyze_query(plan)
            strategy = analysis['recommended_strategy']
            strategy_groups[strategy].append((i, plan))

        # Execute each strategy group
        all_results = [None] * len(plans)

        for strategy, group in strategy_groups.items():
            if strategy == 'hnsw':
                # Batch HNSW queries - can share graph traversals
                results = self._batch_hnsw_queries(
                    [plan for _, plan in group]
                )
            elif strategy == 'ivf_pq':
                # Batch IVF-PQ queries - can share cluster lookups
                results = self._batch_ivf_queries(
                    [plan for _, plan in group]
                )
            else:
                # Execute individually
                results = [
                    self.retrieval.execute_query(plan)
                    for _, plan in group
                ]

            # Place results in correct positions
            for (i, _), result in zip(group, results):
                all_results[i] = result

        return all_results

    def _batch_hnsw_queries(
        self,
        plans: List[QueryPlan]
    ) -> List[QueryResult]:
        """
        Execute batch of HNSW queries efficiently
        
        Shares graph structure reads across queries
        """
        # In production, use batched HNSW search
        # For demonstration, execute individually
        return [
            self.retrieval.execute_query(plan)
            for plan in plans
        ]

    def _batch_ivf_queries(
        self,
        plans: List[QueryPlan]
    ) -> List[QueryResult]:
        """
        Execute batch of IVF queries efficiently
        
        Shares cluster lookups across queries
        """
        # In production, use batched IVF search
        # For demonstration, execute individually
        return [
            self.retrieval.execute_query(plan)
            for plan in plans
        ]
