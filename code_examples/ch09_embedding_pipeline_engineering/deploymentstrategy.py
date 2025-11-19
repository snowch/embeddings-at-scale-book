# Code from Chapter 09
# Book: Embeddings at Scale

from enum import Enum
from typing import Dict, Optional

import numpy as np
import torch


class DeploymentStrategy(Enum):
    """
    Strategies for deploying new embedding models

    1. BLUE_GREEN: Maintain two complete indices, switch traffic instantly
    2. INCREMENTAL: Gradually re-embed corpus, serve from multiple indices
    3. SHADOW: Run new model in shadow mode, compare before switching
    4. CANARY: Route small % of traffic to new model, monitor metrics
    """
    BLUE_GREEN = "blue_green"
    INCREMENTAL = "incremental"
    SHADOW = "shadow"
    CANARY = "canary"

class EmbeddingVersionCoordinator:
    """
    Coordinate embedding model versions across pipeline stages

    Responsibilities:
    - Track active model versions (training, staging, production)
    - Coordinate re-embedding campaigns for new versions
    - Manage multiple vector indices during transitions
    - Enable safe rollbacks with version pinning
    - Monitor version-specific metrics

    Architecture:
    - Version Registry: Maps model_version → index_version
    - Index Router: Routes queries to appropriate index based on version
    - Re-embedding Orchestrator: Manages gradual corpus re-embedding
    - Rollback Controller: Handles instant rollbacks to previous version
    """

    def __init__(self, model_registry):
        self.model_registry = model_registry

        # Track deployed versions
        self.active_versions: Dict[str, VersionDeployment] = {}

        # Index mapping: version → index_name
        self.version_to_index: Dict[str, str] = {}

        # Traffic routing: version → traffic_percentage
        self.traffic_routing: Dict[str, float] = {}

    def deploy_new_version(
        self,
        new_model_id: str,
        strategy: DeploymentStrategy,
        corpus_iterator=None
    ):
        """
        Deploy new embedding model version

        Args:
            new_model_id: New model to deploy
            strategy: Deployment strategy
            corpus_iterator: Iterator over corpus for re-embedding
        """
        new_model, metadata = self.model_registry.load_model(new_model_id)

        print(f"Deploying {new_model_id} using {strategy.value} strategy...")

        if strategy == DeploymentStrategy.BLUE_GREEN:
            self._deploy_blue_green(new_model_id, new_model, corpus_iterator)

        elif strategy == DeploymentStrategy.INCREMENTAL:
            self._deploy_incremental(new_model_id, new_model, corpus_iterator)

        elif strategy == DeploymentStrategy.SHADOW:
            self._deploy_shadow(new_model_id, new_model)

        elif strategy == DeploymentStrategy.CANARY:
            self._deploy_canary(new_model_id, new_model)

    def _deploy_blue_green(
        self,
        new_model_id: str,
        new_model,
        corpus_iterator
    ):
        """
        Blue-Green Deployment

        Steps:
        1. Build complete new index (GREEN) while serving from old (BLUE)
        2. Validate new index quality
        3. Switch traffic from BLUE → GREEN instantly
        4. Keep BLUE as rollback target

        Pros:
        - Instant switchover (no partial state)
        - Easy rollback (flip traffic back)
        - No version mixing

        Cons:
        - Expensive (2x storage during transition)
        - Long preparation time (full re-embedding)
        - All-or-nothing switch
        """
        print("Building GREEN index (new version)...")

        # Create new index
        green_index_name = f"embeddings_{new_model_id.replace('.', '_')}"
        self._create_new_index(green_index_name)

        # Re-embed entire corpus into GREEN
        self._reembed_corpus(
            new_model,
            corpus_iterator,
            target_index=green_index_name
        )

        # Validate GREEN index
        validation_passed = self._validate_index_quality(green_index_name)

        if not validation_passed:
            print("✗ Validation failed, aborting deployment")
            self._delete_index(green_index_name)
            return

        # Get current BLUE index for rollback
        current_prod = self.model_registry._get_current_production_model()
        blue_index_name = self.version_to_index.get(current_prod.model_id)

        # Switch traffic: BLUE → GREEN
        print("Switching traffic from BLUE → GREEN...")
        self._switch_traffic(
            from_index=blue_index_name,
            to_index=green_index_name
        )

        # Register new version
        self.version_to_index[new_model_id] = green_index_name
        self.traffic_routing[new_model_id] = 1.0  # 100% traffic

        # Keep BLUE as rollback target (don't delete yet)
        print("✓ Deployment complete. GREEN active, BLUE retained for rollback.")

    def _deploy_incremental(
        self,
        new_model_id: str,
        new_model,
        corpus_iterator
    ):
        """
        Incremental Deployment

        Steps:
        1. Create new index (initially empty)
        2. Re-embed corpus gradually (over hours/days)
        3. Route queries across OLD + NEW indices during transition
        4. Switch to NEW once re-embedding complete

        Pros:
        - Lower resource spike (gradual re-embedding)
        - Faster time-to-production (start using partially)
        - More control over transition

        Cons:
        - Complex routing logic (query both indices)
        - Duplicate storage during transition
        - Potential inconsistency (different items in different versions)
        """
        print("Starting incremental deployment...")

        # Create new index
        new_index_name = f"embeddings_{new_model_id.replace('.', '_')}"
        self._create_new_index(new_index_name)

        # Start re-embedding in background
        self._start_background_reembedding(
            new_model,
            corpus_iterator,
            target_index=new_index_name,
            rate_limit_items_per_sec=1000
        )

        # Route queries to BOTH old and new indices
        # (merge results, prefer new embeddings when available)
        current_prod = self.model_registry._get_current_production_model()
        old_index = self.version_to_index.get(current_prod.model_id)

        self._enable_dual_index_routing(old_index, new_index_name)

        print("✓ Incremental deployment started")
        print("  Re-embedding progress tracked in background")
        print("  Queries served from both OLD and NEW indices during transition")

    def _deploy_shadow(
        self,
        new_model_id: str,
        new_model
    ):
        """
        Shadow Deployment

        Steps:
        1. Deploy new model in "shadow mode"
        2. Serve all queries with OLD model (production)
        3. Simultaneously generate embeddings with NEW model (shadow)
        4. Compare OLD vs NEW results, collect metrics
        5. Switch to NEW if metrics better

        Pros:
        - Safe validation (no user impact)
        - Comprehensive metric collection
        - Detects issues before full deployment

        Cons:
        - 2x compute cost (running both models)
        - Longer validation period
        - Still need full re-embedding after shadow validation
        """
        print("Deploying in shadow mode...")

        # Register shadow model (doesn't serve production traffic)
        self.active_versions[new_model_id] = VersionDeployment(
            model_id=new_model_id,
            status=DeploymentStatus.SHADOW,
            traffic_percentage=0.0
        )

        # Shadow traffic logs comparisons but doesn't serve
        self._enable_shadow_mode(new_model_id)

        print("✓ Shadow deployment active")
        print("  NEW model: Running in shadow (no production traffic)")
        print("  OLD model: Serving 100% production traffic")
        print("  Comparison metrics being collected")

    def _deploy_canary(
        self,
        new_model_id: str,
        new_model
    ):
        """
        Canary Deployment

        Steps:
        1. Route 1% of traffic to NEW model
        2. Monitor metrics (latency, quality, errors)
        3. Gradually increase traffic: 1% → 5% → 10% → 50% → 100%
        4. Rollback if any stage shows degradation

        Pros:
        - Gradual validation with real traffic
        - Early detection of issues
        - Limits blast radius of problems

        Cons:
        - Requires mixing versions (compatibility issues)
        - Longer rollout timeline
        - Complex traffic routing
        """
        print("Starting canary deployment...")

        # Start with 1% traffic
        initial_canary_percentage = 0.01

        self.active_versions[new_model_id] = VersionDeployment(
            model_id=new_model_id,
            status=DeploymentStatus.CANARY,
            traffic_percentage=initial_canary_percentage
        )

        self.traffic_routing[new_model_id] = initial_canary_percentage

        print("✓ Canary deployment started")
        print(f"  NEW model: {initial_canary_percentage:.1%} traffic")
        print(f"  OLD model: {1 - initial_canary_percentage:.1%} traffic")
        print("  Monitor metrics, then gradually increase canary traffic")

    def rollback(self, target_model_id: Optional[str] = None):
        """
        Rollback to previous model version

        Critical for:
        - Quality degradation detected in production
        - Performance regression (latency spike)
        - Bugs in new model
        - Incompatibility issues

        Instant rollback: Switch traffic to previous index (Blue-Green)
        Gradual rollback: Reduce traffic to new version (Canary)
        """
        if target_model_id is None:
            # Rollback to previous production model
            current_prod = self.model_registry._get_current_production_model()
            if current_prod.rollback_model_id is None:
                raise ValueError("No rollback target defined")
            target_model_id = current_prod.rollback_model_id

        print(f"Rolling back to {target_model_id}...")

        # Get target index
        target_index = self.version_to_index.get(target_model_id)
        if target_index is None:
            raise ValueError(f"No index found for {target_model_id}")

        # Switch traffic
        self._switch_traffic(from_index=None, to_index=target_index)

        # Update traffic routing
        self.traffic_routing = {target_model_id: 1.0}

        print(f"✓ Rollback complete to {target_model_id}")

    def _reembed_corpus(self, model, corpus_iterator, target_index: str):
        """
        Re-embed entire corpus with new model

        For trillion-row scale:
        - Distribute across 100-1000 workers
        - Process 1M-10M items per worker
        - Checkpoint every 10M items for fault tolerance
        - Takes hours to days depending on scale
        """
        print(f"Re-embedding corpus into {target_index}...")

        processed = 0
        batch_size = 1024

        with torch.no_grad():
            for batch in corpus_iterator:
                # Generate embeddings
                embeddings = model(batch).cpu().numpy()

                # Write to new index
                self._write_to_index(target_index, embeddings)

                processed += len(batch)
                if processed % 1000000 == 0:
                    print(f"  Re-embedded {processed:,} items...")

        print(f"✓ Re-embedding complete: {processed:,} items")

    def _validate_index_quality(self, index_name: str) -> bool:
        """
        Validate new index meets quality thresholds

        Checks:
        - Retrieval quality (recall@k on test set)
        - Latency (p50, p95, p99)
        - Index size (compression ratio)
        - Coverage (% of corpus embedded)
        """
        print(f"Validating index {index_name}...")

        # Run test queries
        test_recall = 0.89  # Placeholder
        test_latency_p99_ms = 15  # Placeholder

        # Thresholds
        min_recall = 0.85
        max_latency_ms = 20

        passed = test_recall >= min_recall and test_latency_p99_ms <= max_latency_ms

        if passed:
            print(f"✓ Validation passed (recall={test_recall:.3f}, latency={test_latency_p99_ms}ms)")
        else:
            print(f"✗ Validation failed (recall={test_recall:.3f}, latency={test_latency_p99_ms}ms)")

        return passed

    # Placeholder helper methods
    def _create_new_index(self, index_name: str):
        pass

    def _delete_index(self, index_name: str):
        pass

    def _switch_traffic(self, from_index: Optional[str], to_index: str):
        pass

    def _write_to_index(self, index_name: str, embeddings: np.ndarray):
        pass

    def _start_background_reembedding(self, model, corpus_iterator, target_index: str, rate_limit_items_per_sec: int):
        pass

    def _enable_dual_index_routing(self, old_index: str, new_index: str):
        pass

    def _enable_shadow_mode(self, model_id: str):
        pass

@dataclass
class VersionDeployment:
    model_id: str
    status: 'DeploymentStatus'
    traffic_percentage: float

class DeploymentStatus(Enum):
    STAGING = "staging"
    SHADOW = "shadow"
    CANARY = "canary"
    PRODUCTION = "production"
    ROLLBACK_TARGET = "rollback_target"
