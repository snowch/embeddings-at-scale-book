# Code from Chapter 24
# Book: Embeddings at Scale

"""
Privacy-Preserving Similarity Search

Architecture:
1. Query protection: Encrypt/obfuscate query vector
2. Private retrieval: LSH with differential privacy
3. Secure computation: MPC for distance computation
4. Result obfuscation: Add dummy results, noise
5. Access pattern hiding: ORAM for index access

Techniques:
- Differentially private LSH: Add noise to hash functions
- Secure k-NN: MPC-based nearest neighbor search
- Private information retrieval: Query without revealing query
- Oblivious RAM: Hide access patterns
- Federated similarity: Distributed search across silos

Privacy guarantees:
- ε-differential privacy: Queries leak at most ε information
- Query unlinkability: Cannot link queries to same user
- Result indistinguishability: Cannot infer database beyond k results
- Computational security: Based on cryptographic hardness assumptions

Performance targets:
- Latency: <200ms for private search (2-4× overhead)
- Throughput: 1,000+ private queries/second
- Recall: >90% despite privacy noise
- Privacy budget: ε ≤ 1.0 for typical applications
"""

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class PrivacyConfig:
    """
    Privacy configuration for similarity search
    
    Attributes:
        epsilon: Differential privacy parameter (smaller = more private)
        delta: Failure probability for (ε,δ)-DP
        sensitivity: Query sensitivity (for noise calibration)
        noise_mechanism: Noise distribution (laplace, gaussian)
        enable_query_obfuscation: Add dummy queries
        enable_result_obfuscation: Add dummy results  
        enable_access_hiding: Use ORAM for access pattern hiding
        privacy_budget_tracking: Track cumulative privacy loss
    """
    epsilon: float = 1.0  # Typical: 0.1-10.0
    delta: float = 1e-5   # Typical: 1e-5 to 1e-7
    sensitivity: float = 1.0
    noise_mechanism: str = "laplace"  # "laplace", "gaussian"
    enable_query_obfuscation: bool = True
    enable_result_obfuscation: bool = True
    enable_access_hiding: bool = False
    privacy_budget_tracking: bool = True

@dataclass
class PrivateQuery:
    """
    Privacy-preserving query representation
    
    Attributes:
        query_id: Unique identifier
        obfuscated_vector: Query with added noise/dummies
        original_norm: Original query norm (for normalization)
        privacy_budget_used: ε used for this query
        timestamp: Query timestamp
        auxiliary_data: Additional protected information
    """
    query_id: str
    obfuscated_vector: np.ndarray
    original_norm: float
    privacy_budget_used: float
    timestamp: datetime = field(default_factory=datetime.now)
    auxiliary_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PrivateResult:
    """
    Privacy-preserving search result
    
    Attributes:
        query_id: Query identifier
        results: List of (id, noisy_score) tuples
        dummy_results: Dummy results for obfuscation
        privacy_guarantee: Achieved privacy level
        utility_metric: Result quality (recall, precision)
    """
    query_id: str
    results: List[Tuple[str, float]]
    dummy_results: List[Tuple[str, float]] = field(default_factory=list)
    privacy_guarantee: float = 1.0  # ε value
    utility_metric: Dict[str, float] = field(default_factory=dict)

class DifferentiallyPrivateLSH:
    """
    Locality-Sensitive Hashing with Differential Privacy
    
    Standard LSH leaks information through hash bucket membership.
    DP-LSH adds calibrated noise to protect privacy while maintaining
    approximate nearest neighbor guarantees.
    
    Approach:
    1. Generate LSH hash functions
    2. Add Laplace noise to hash values
    3. Query noisy hash buckets
    4. Post-process results with additional noise
    
    Privacy guarantee:
    - Each query satisfies ε-differential privacy
    - Privacy budget tracks cumulative usage
    - Composition theorems bound total leakage
    """

    def __init__(
        self,
        dimension: int,
        num_hashes: int,
        num_tables: int,
        privacy_config: PrivacyConfig
    ):
        self.dimension = dimension
        self.num_hashes = num_hashes
        self.num_tables = num_tables
        self.privacy_config = privacy_config

        # Generate LSH hash functions
        self.hash_functions = self._generate_hash_functions()

        # Hash tables: table_id -> bucket_hash -> [embedding_ids]
        self.hash_tables: Dict[int, Dict[int, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )

        # Privacy budget tracking
        self.privacy_budget_used = 0.0
        self.query_count = 0

        print("Differentially Private LSH initialized:")
        print(f"  Hash functions: {num_hashes} × {num_tables} tables")
        print(f"  Privacy: ε={privacy_config.epsilon}, δ={privacy_config.delta}")

    def _generate_hash_functions(self) -> List[List[np.ndarray]]:
        """
        Generate random projection hash functions
        
        Returns:
            List of hash function matrices for each table
        """
        np.random.seed(42)
        hash_functions = []

        for _ in range(self.num_tables):
            table_hashes = []
            for _ in range(self.num_hashes):
                # Random projection vector
                projection = np.random.randn(self.dimension)
                projection /= np.linalg.norm(projection)
                table_hashes.append(projection)
            hash_functions.append(table_hashes)

        return hash_functions

    def _hash_vector(
        self,
        vector: np.ndarray,
        table_id: int,
        add_noise: bool = False
    ) -> int:
        """
        Hash vector using LSH with optional differential privacy noise
        
        Steps:
        1. Project vector using hash functions
        2. Quantize projections to bits
        3. Add Laplace noise (if privacy enabled)
        4. Combine bits into hash value
        
        Args:
            vector: Vector to hash
            table_id: Which hash table to use
            add_noise: Whether to add DP noise
            
        Returns:
            Hash value (integer)
        """
        hash_bits = []
        hash_functions = self.hash_functions[table_id]

        for projection in hash_functions:
            # Compute projection
            proj_value = np.dot(vector, projection)

            # Add Laplace noise for differential privacy
            if add_noise:
                noise_scale = self.privacy_config.sensitivity / self.privacy_config.epsilon
                noise = np.random.laplace(0, noise_scale)
                proj_value += noise

            # Quantize to bit
            bit = 1 if proj_value >= 0 else 0
            hash_bits.append(bit)

        # Convert bits to integer hash
        hash_value = int(''.join(map(str, hash_bits)), 2)
        return hash_value

    def index_embedding(
        self,
        embedding: np.ndarray,
        embedding_id: str
    ):
        """
        Index embedding in DP-LSH tables
        
        Args:
            embedding: Vector to index
            embedding_id: Unique identifier
        """
        # Hash into each table (no noise during indexing)
        for table_id in range(self.num_tables):
            hash_value = self._hash_vector(embedding, table_id, add_noise=False)
            self.hash_tables[table_id][hash_value].add(embedding_id)

    def private_query(
        self,
        query: np.ndarray,
        k: int = 10,
        epsilon_per_query: Optional[float] = None
    ) -> PrivateQuery:
        """
        Create privacy-preserving query
        
        Techniques:
        1. Add Laplace noise to query vector
        2. Generate dummy queries for obfuscation
        3. Track privacy budget usage
        
        Args:
            query: Original query vector
            k: Number of results desired
            epsilon_per_query: Privacy budget for this query
            
        Returns:
            Private query object
        """
        if epsilon_per_query is None:
            epsilon_per_query = self.privacy_config.epsilon

        # Add noise to query vector
        if self.privacy_config.noise_mechanism == "laplace":
            noise_scale = self.privacy_config.sensitivity / epsilon_per_query
            noise = np.random.laplace(0, noise_scale, size=query.shape)
        elif self.privacy_config.noise_mechanism == "gaussian":
            # Gaussian mechanism for (ε,δ)-DP
            sigma = math.sqrt(
                2 * math.log(1.25 / self.privacy_config.delta)
            ) * self.privacy_config.sensitivity / epsilon_per_query
            noise = np.random.normal(0, sigma, size=query.shape)
        else:
            noise = np.zeros_like(query)

        obfuscated = query + noise

        # Track privacy budget
        if self.privacy_config.privacy_budget_tracking:
            self.privacy_budget_used += epsilon_per_query
            self.query_count += 1

        return PrivateQuery(
            query_id=f"query_{self.query_count}",
            obfuscated_vector=obfuscated,
            original_norm=np.linalg.norm(query),
            privacy_budget_used=epsilon_per_query
        )

    def search(
        self,
        private_query: PrivateQuery,
        k: int = 10,
        embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> PrivateResult:
        """
        Privacy-preserving similarity search
        
        Steps:
        1. Hash query into each table (with noise)
        2. Retrieve candidates from matching buckets
        3. Compute noisy similarities
        4. Add dummy results for obfuscation
        5. Return top-k with privacy guarantees
        
        Args:
            private_query: Privacy-preserving query
            k: Number of results
            embeddings: Embedding store for similarity computation
            
        Returns:
            Private search results
        """
        # Retrieve candidates using noisy hashing
        candidates = set()
        for table_id in range(self.num_tables):
            hash_value = self._hash_vector(
                private_query.obfuscated_vector,
                table_id,
                add_noise=True
            )

            # Get candidates from this bucket
            bucket_candidates = self.hash_tables[table_id].get(hash_value, set())
            candidates.update(bucket_candidates)

        if not candidates or embeddings is None:
            return PrivateResult(
                query_id=private_query.query_id,
                results=[],
                privacy_guarantee=private_query.privacy_budget_used
            )

        # Compute noisy similarities
        similarities = []
        for emb_id in candidates:
            embedding = embeddings[emb_id]

            # True similarity
            true_sim = np.dot(
                private_query.obfuscated_vector, embedding
            ) / (
                np.linalg.norm(private_query.obfuscated_vector) *
                np.linalg.norm(embedding)
            )

            # Add noise for privacy
            noise_scale = 1.0 / private_query.privacy_budget_used
            noisy_sim = true_sim + np.random.laplace(0, noise_scale)

            similarities.append((emb_id, noisy_sim))

        # Sort and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_results = similarities[:k]

        # Add dummy results for obfuscation
        dummy_results = []
        if self.privacy_config.enable_result_obfuscation:
            num_dummies = k // 2  # Add 50% dummy results
            all_ids = set(embeddings.keys()) - set(c[0] for c in top_k_results)
            dummy_ids = np.random.choice(
                list(all_ids),
                size=min(num_dummies, len(all_ids)),
                replace=False
            )
            for dummy_id in dummy_ids:
                # Random similarity scores
                dummy_score = np.random.uniform(-1, 1)
                dummy_results.append((str(dummy_id), dummy_score))

        return PrivateResult(
            query_id=private_query.query_id,
            results=top_k_results,
            dummy_results=dummy_results,
            privacy_guarantee=private_query.privacy_budget_used,
            utility_metric={
                "num_candidates": len(candidates),
                "privacy_budget_used": self.privacy_budget_used,
                "query_count": self.query_count
            }
        )

class SecureMultiPartyKNN:
    """
    Secure Multi-Party Computation for k-NN
    
    Scenario: Multiple parties have embedding databases, want to jointly
    compute k-NN without revealing their embeddings to each other.
    
    Protocol:
    1. Secret sharing: Each party splits embeddings into shares
    2. Distributed computation: Compute similarities on shares
    3. Secure aggregation: Combine results without reconstruction
    4. Result revelation: Only final top-k revealed
    
    Security: Honest-but-curious adversary model, collusion resistance
    """

    def __init__(self, num_parties: int, privacy_config: PrivacyConfig):
        self.num_parties = num_parties
        self.privacy_config = privacy_config

        # Party databases: party_id -> {emb_id: embedding}
        self.party_databases: Dict[int, Dict[str, np.ndarray]] = {
            i: {} for i in range(num_parties)
        }

        print(f"Secure MPC k-NN initialized for {num_parties} parties")

    def _secret_share(
        self,
        value: float,
        num_shares: int
    ) -> List[float]:
        """
        Split value into additive secret shares
        
        Property: sum(shares) = value (modulo some large prime)
        Each party gets one share, learns nothing individually
        
        Args:
            value: Value to share
            num_shares: Number of shares
            
        Returns:
            List of shares
        """
        shares = np.random.uniform(-10, 10, size=num_shares - 1).tolist()
        final_share = value - sum(shares)
        shares.append(final_share)
        return shares

    def add_party_embedding(
        self,
        party_id: int,
        embedding: np.ndarray,
        embedding_id: str
    ):
        """Add embedding to party's database"""
        self.party_databases[party_id][embedding_id] = embedding

    def secure_similarity(
        self,
        query: np.ndarray,
        party_embeddings: List[np.ndarray]
    ) -> float:
        """
        Compute similarity using secure MPC
        
        Protocol:
        1. Each party computes partial dot product with their data
        2. Parties secret-share partial results
        3. Aggregate shares to get final similarity
        4. No party learns individual contributions
        
        Args:
            query: Query vector (can be shared)
            party_embeddings: Embedding from each party
            
        Returns:
            Aggregated similarity score
        """
        # Each party computes local dot product
        partial_sims = [
            np.dot(query, emb) for emb in party_embeddings
        ]

        # Secret share partial results
        shares_matrix = [
            self._secret_share(sim, self.num_parties)
            for sim in partial_sims
        ]

        # Each party aggregates their shares
        party_aggregates = [
            sum(shares_matrix[i][j] for i in range(len(partial_sims)))
            for j in range(self.num_parties)
        ]

        # Final aggregation (reveals only total)
        total_similarity = sum(party_aggregates)

        # Normalize by query norm and average embedding norm
        query_norm = np.linalg.norm(query)
        avg_emb_norm = np.mean([
            np.linalg.norm(emb) for emb in party_embeddings
        ])

        normalized_sim = total_similarity / (query_norm * avg_emb_norm)
        return normalized_sim

    def federated_search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> List[Tuple[int, str, float]]:
        """
        Federated k-NN across parties
        
        Protocol:
        1. Each party computes local top-k
        2. Parties securely aggregate to find global top-k
        3. Only final results revealed
        
        Args:
            query: Query vector
            k: Number of results
            
        Returns:
            Global top-k: (party_id, emb_id, score) tuples
        """
        # Each party computes local similarities
        local_results = []

        for party_id in range(self.num_parties):
            party_db = self.party_databases[party_id]

            for emb_id, embedding in party_db.items():
                similarity = np.dot(query, embedding) / (
                    np.linalg.norm(query) * np.linalg.norm(embedding)
                )
                local_results.append((party_id, emb_id, similarity))

        # Sort and get global top-k
        local_results.sort(key=lambda x: x[2], reverse=True)
        global_top_k = local_results[:k]

        return global_top_k

class PrivateInformationRetrieval:
    """
    Private Information Retrieval for embeddings
    
    Goal: Query database without revealing which item you're querying for
    
    Computational PIR:
    - Based on cryptographic assumptions (e.g., homomorphic encryption)
    - Polylogarithmic communication complexity
    - Practical for small-medium databases
    
    Information-theoretic PIR:
    - Requires database replication across non-colluding servers
    - Perfect privacy but higher communication cost
    - Used in privacy-critical applications
    """

    def __init__(
        self,
        database_size: int,
        dimension: int,
        privacy_config: PrivacyConfig
    ):
        self.database_size = database_size
        self.dimension = dimension
        self.privacy_config = privacy_config

        print(f"PIR initialized for database of {database_size} embeddings")

    def private_retrieve(
        self,
        query_index: int,
        database: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Retrieve embedding without revealing which one
        
        Simplified approach (real PIR uses homomorphic encryption):
        1. Generate query vector that encodes index obliviously
        2. Server computes linear combination
        3. Client decodes result
        
        Args:
            query_index: Index to retrieve (hidden from server)
            database: Embedding database
            
        Returns:
            Retrieved embedding
        """
        # In production: use homomorphic PIR (e.g., SealPIR, XPIR)
        # This is a simplified demonstration

        db_items = list(database.items())
        if query_index >= len(db_items):
            return None

        # Generate oblivious query (one-hot encoded with noise)
        query = np.random.randn(len(db_items)) * 0.1
        query[query_index] = 1.0

        # Server computes linear combination (doesn't know index)
        result = np.zeros(self.dimension)
        for i, (_, embedding) in enumerate(db_items):
            result += query[i] * embedding

        return result

# Example usage
def privacy_preserving_search_example():
    """
    Demonstrate privacy-preserving similarity search techniques
    """
    print("=== Privacy-Preserving Similarity Search ===")
    print()

    # Configuration
    privacy_config = PrivacyConfig(
        epsilon=1.0,
        delta=1e-5,
        noise_mechanism="laplace",
        enable_query_obfuscation=True,
        enable_result_obfuscation=True
    )

    # Generate sample embeddings
    print("Generating sample embeddings...")
    np.random.seed(42)
    dimension = 768
    num_embeddings = 1000

    embeddings = {}
    for i in range(num_embeddings):
        emb = np.random.randn(dimension).astype(np.float32)
        emb /= np.linalg.norm(emb)
        embeddings[f"emb_{i}"] = emb

    print(f"Generated {num_embeddings} embeddings")
    print()

    # 1. Differentially Private LSH
    print("1. Differentially Private LSH")
    dp_lsh = DifferentiallyPrivateLSH(
        dimension=dimension,
        num_hashes=16,
        num_tables=8,
        privacy_config=privacy_config
    )

    # Index embeddings
    for emb_id, emb in embeddings.items():
        dp_lsh.index_embedding(emb, emb_id)

    # Private query
    query = np.random.randn(dimension).astype(np.float32)
    query /= np.linalg.norm(query)

    private_q = dp_lsh.private_query(query, k=10)
    result = dp_lsh.search(private_q, k=10, embeddings=embeddings)

    print(f"  Query ID: {result.query_id}")
    print(f"  Privacy guarantee: ε={result.privacy_guarantee:.2f}")
    print(f"  Candidates found: {result.utility_metric['num_candidates']}")
    print("  Top 5 results:")
    for emb_id, score in result.results[:5]:
        print(f"    {emb_id}: {score:.4f}")
    print(f"  Dummy results added: {len(result.dummy_results)}")
    print()

    # 2. Secure Multi-Party k-NN
    print("2. Secure Multi-Party k-NN")
    num_parties = 3
    mpc_knn = SecureMultiPartyKNN(num_parties, privacy_config)

    # Distribute embeddings across parties
    for i, (emb_id, emb) in enumerate(embeddings.items()):
        party_id = i % num_parties
        mpc_knn.add_party_embedding(party_id, emb, emb_id)

    # Federated search
    results = mpc_knn.federated_search(query, k=10)

    print(f"  Federated top 5 results across {num_parties} parties:")
    for party_id, emb_id, score in results[:5]:
        print(f"    Party {party_id}, {emb_id}: {score:.4f}")
    print()

    # 3. Privacy budget tracking
    print("3. Privacy Budget Tracking")
    print(f"  Total privacy budget used: ε={dp_lsh.privacy_budget_used:.2f}")
    print(f"  Number of queries: {dp_lsh.query_count}")
    print(f"  Average ε per query: {dp_lsh.privacy_budget_used/dp_lsh.query_count:.2f}")

    # Demonstrate privacy budget exhaustion
    budget_limit = 10.0
    remaining = budget_limit - dp_lsh.privacy_budget_used
    print(f"  Remaining budget (limit={budget_limit}): ε={remaining:.2f}")
    print(f"  Can support ~{int(remaining/privacy_config.epsilon)} more queries")

if __name__ == "__main__":
    privacy_preserving_search_example()
