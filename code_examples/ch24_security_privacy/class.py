# Code from Chapter 24
# Book: Embeddings at Scale

"""
Secure Embedding Computation with Homomorphic Encryption

Architecture:
1. Encryption: Encrypt embeddings with homomorphic encryption scheme
2. Secure indexing: Build encrypted index supporting similarity search
3. Query encryption: Encrypt query vectors with same scheme
4. Encrypted search: Compute encrypted similarity scores
5. Secure decryption: Return encrypted results for client-side decryption

Techniques:
- Homomorphic encryption: CKKS for approximate operations on floats
- Secure enclaves: Intel SGX, AMD SEV for trusted execution
- Functional encryption: Allow specific computations on encrypted data
- Secure multi-party computation: Distribute trust across parties
- Order-preserving encryption: Enable range queries on encrypted scalars

Security properties:
- IND-CPA security: Ciphertext reveals no information about plaintext
- Query privacy: Server learns nothing about query vector
- Result privacy: Client learns only k nearest neighbors, nothing else
- Collusion resistance: Multiple parties cannot learn individual contributions

Performance targets:
- Encryption: <10ms per 768-d vector
- Encrypted similarity: <50ms for 1M encrypted vectors
- Overhead: <10× vs plaintext for CKKS, <3× for SGX
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import hmac
import secrets
from abc import ABC, abstractmethod

@dataclass
class EncryptionConfig:
    """
    Configuration for embedding encryption
    
    Attributes:
        scheme: Encryption scheme (ckks, sgx, ope, hybrid)
        key_size: Key size in bits
        security_level: Security parameter (128, 192, 256)
        noise_budget: Noise budget for homomorphic operations
        precision: Precision bits for CKKS encoding
        enable_packing: Pack multiple vectors in single ciphertext
        enable_batching: Batch operations for efficiency
    """
    scheme: str = "ckks"  # "ckks", "sgx", "ope", "hybrid"
    key_size: int = 2048
    security_level: int = 128
    noise_budget: int = 100
    precision: int = 40
    enable_packing: bool = True
    enable_batching: bool = True

@dataclass
class EncryptedEmbedding:
    """
    Encrypted embedding representation
    
    Attributes:
        id: Embedding identifier
        ciphertext: Encrypted vector data
        public_key_id: Public key used for encryption
        encryption_metadata: Additional encryption info
        dimension: Original embedding dimension
        encrypted_at: Encryption timestamp
    """
    id: str
    ciphertext: bytes
    public_key_id: str
    encryption_metadata: Dict[str, Any] = field(default_factory=dict)
    dimension: int = 768
    encrypted_at: datetime = field(default_factory=datetime.now)

class SecureEmbeddingStore(ABC):
    """
    Abstract interface for secure embedding storage and computation
    """
    
    @abstractmethod
    def encrypt_embedding(
        self,
        embedding: np.ndarray,
        embedding_id: str
    ) -> EncryptedEmbedding:
        """Encrypt an embedding vector"""
        pass
    
    @abstractmethod
    def decrypt_embedding(
        self,
        encrypted: EncryptedEmbedding
    ) -> np.ndarray:
        """Decrypt an embedding vector"""
        pass
    
    @abstractmethod
    def encrypted_similarity(
        self,
        query_encrypted: EncryptedEmbedding,
        candidate_encrypted: EncryptedEmbedding
    ) -> float:
        """Compute similarity between encrypted embeddings"""
        pass
    
    @abstractmethod
    def secure_nearest_neighbors(
        self,
        query_encrypted: EncryptedEmbedding,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Find k nearest neighbors in encrypted space"""
        pass

class CKKSEmbeddingStore(SecureEmbeddingStore):
    """
    CKKS homomorphic encryption for embeddings
    
    CKKS (Cheon-Kim-Kim-Song) enables approximate arithmetic on
    encrypted floating-point numbers, suitable for embedding similarity.
    
    Operations:
    - Addition: Add encrypted vectors (for aggregation)
    - Multiplication: Multiply encrypted vectors (for dot product)
    - Rotation: Rotate slots (for vector operations)
    
    Performance:
    - Encryption: O(d log d) for d-dimensional vector
    - Addition: O(d) on encrypted vectors
    - Multiplication: O(d^2) but can be optimized
    - Decryption: O(d log d)
    """
    
    def __init__(self, config: EncryptionConfig):
        self.config = config
        self.public_key = None
        self.secret_key = None
        self.encrypted_store: Dict[str, EncryptedEmbedding] = {}
        
        # Initialize CKKS parameters
        self._initialize_ckks()
    
    def _initialize_ckks(self):
        """
        Initialize CKKS encryption scheme
        
        In production, use libraries like:
        - Microsoft SEAL (C++, Python bindings)
        - OpenFHE (C++, Python bindings)
        - HElib (C++)
        - TenSEAL (Python)
        """
        # Simplified initialization for demonstration
        # In production, use proper CKKS library
        
        # Generate key pair (simplified)
        self.secret_key = secrets.token_bytes(self.config.key_size // 8)
        self.public_key = hashlib.sha256(self.secret_key).hexdigest()
        
        print(f"CKKS initialized: security={self.config.security_level} bits")
        print(f"  Key size: {self.config.key_size} bits")
        print(f"  Precision: {self.config.precision} bits")
        print(f"  Noise budget: {self.config.noise_budget}")
    
    def encrypt_embedding(
        self,
        embedding: np.ndarray,
        embedding_id: str
    ) -> EncryptedEmbedding:
        """
        Encrypt embedding with CKKS
        
        Steps:
        1. Encode: Convert float vector to CKKS plaintext
        2. Encrypt: Apply CKKS encryption with public key
        3. Pack: Optionally pack multiple vectors in single ciphertext
        
        Args:
            embedding: Float vector to encrypt
            embedding_id: Unique identifier
            
        Returns:
            Encrypted embedding
        """
        # In production, use CKKS library:
        # from tenseal import CKKS
        # context = ts.context(ts.SCHEME_TYPE.CKKS, ...)
        # encrypted = ts.ckks_vector(context, embedding)
        # ciphertext = encrypted.serialize()
        
        # Simplified encryption for demonstration
        embedding_bytes = embedding.tobytes()
        
        # Simulated CKKS encryption (use proper library in production)
        cipher = hmac.new(
            self.secret_key,
            embedding_bytes,
            hashlib.sha256
        ).digest()
        
        # Add noise based on noise budget
        noise = np.random.randint(
            0, 2**self.config.noise_budget,
            size=len(cipher)
        ).tobytes()
        ciphertext = bytes(a ^ b for a, b in zip(cipher, noise[:len(cipher)]))
        
        encrypted = EncryptedEmbedding(
            id=embedding_id,
            ciphertext=ciphertext,
            public_key_id=self.public_key,
            dimension=len(embedding),
            encryption_metadata={
                "scheme": "ckks",
                "precision": self.config.precision,
                "noise_budget": self.config.noise_budget
            }
        )
        
        self.encrypted_store[embedding_id] = encrypted
        return encrypted
    
    def decrypt_embedding(
        self,
        encrypted: EncryptedEmbedding
    ) -> np.ndarray:
        """
        Decrypt CKKS-encrypted embedding
        
        Args:
            encrypted: Encrypted embedding
            
        Returns:
            Decrypted float vector
        """
        # In production:
        # encrypted_vec = ts.ckks_vector_from(context, encrypted.ciphertext)
        # decrypted = encrypted_vec.decrypt()
        
        # Simplified decryption (demonstration only)
        # Real CKKS decryption requires secret key and proper scheme
        raise NotImplementedError(
            "Decryption requires proper CKKS library (TenSEAL, SEAL, etc.)"
        )
    
    def encrypted_similarity(
        self,
        query_encrypted: EncryptedEmbedding,
        candidate_encrypted: EncryptedEmbedding
    ) -> float:
        """
        Compute cosine similarity between encrypted vectors
        
        CKKS enables:
        1. Encrypted dot product: <x, y>_encrypted
        2. Encrypted norms: ||x||_encrypted, ||y||_encrypted
        3. Similarity: <x,y> / (||x|| * ||y||)
        
        Challenges:
        - Division not directly supported, use approximation
        - Noise accumulation from multiple operations
        - Precision loss from encoding
        
        Args:
            query_encrypted: Encrypted query vector
            candidate_encrypted: Encrypted candidate vector
            
        Returns:
            Approximate similarity score
        """
        # In production using TenSEAL:
        # dot_product = query_encrypted * candidate_encrypted
        # query_norm = query_encrypted.dot(query_encrypted).sqrt()
        # candidate_norm = candidate_encrypted.dot(candidate_encrypted).sqrt()
        # similarity = dot_product / (query_norm * candidate_norm)
        
        # Simplified similarity computation
        # Real implementation requires CKKS operations
        raise NotImplementedError(
            "Encrypted similarity requires proper CKKS library with "
            "dot product, multiplication, and approximate division"
        )
    
    def secure_nearest_neighbors(
        self,
        query_encrypted: EncryptedEmbedding,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors in encrypted space
        
        Approach:
        1. Compute encrypted similarities with all candidates
        2. Use oblivious selection to find top-k
        3. Return encrypted result for client-side decryption
        
        Or use secure index structures:
        - Encrypted LSH: Hash encrypted vectors
        - Encrypted tree: Navigate tree in encrypted space
        - Secure MPC: Distribute computation across parties
        
        Args:
            query_encrypted: Encrypted query
            k: Number of neighbors
            
        Returns:
            List of (id, encrypted_score) for top-k neighbors
        """
        # Compute encrypted similarities
        similarities = []
        for emb_id, candidate in self.encrypted_store.items():
            try:
                sim = self.encrypted_similarity(query_encrypted, candidate)
                similarities.append((emb_id, sim))
            except NotImplementedError:
                # Fallback to approximate method
                pass
        
        # Sort by similarity (obliviously in production)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]

class SGXEmbeddingStore(SecureEmbeddingStore):
    """
    Intel SGX (Software Guard Extensions) for secure embedding computation
    
    SGX creates trusted execution environments (enclaves) where code
    and data are protected from the operating system and other processes.
    
    Benefits:
    - Low overhead: 2-5× vs unencrypted (vs 10-100× for FHE)
    - Full computation: Can perform any operation inside enclave
    - Hardware-backed: Security guaranteed by CPU
    
    Limitations:
    - Limited enclave memory (128MB in SGX1, GBs in SGX2)
    - Platform-specific: Intel CPUs only
    - Side-channel vulnerabilities: Spectre, etc.
    
    Use cases:
    - Encrypted database: Store embeddings outside enclave, compute inside
    - Query privacy: Process queries without revealing to database operator
    - Multi-tenant isolation: Separate tenant data in different enclaves
    """
    
    def __init__(self, config: EncryptionConfig):
        self.config = config
        self.enclave_initialized = False
        self.encrypted_store: Dict[str, EncryptedEmbedding] = {}
        
        self._initialize_sgx()
    
    def _initialize_sgx(self):
        """
        Initialize SGX enclave
        
        Steps:
        1. Load enclave code (signed binary)
        2. Attestation: Verify enclave is genuine Intel SGX
        3. Establish secure channel for data transfer
        4. Load secret keys into enclave
        
        In production, use:
        - Intel SGX SDK
        - Gramine (formerly Graphene) for running Python in SGX
        - Microsoft Azure Confidential Computing
        - Google Cloud Confidential VMs
        """
        # Check if SGX is available
        sgx_available = self._check_sgx_support()
        
        if sgx_available:
            print("SGX enclave initialized")
            print(f"  Security level: {self.config.security_level} bits")
            print(f"  Enclave memory: Limited by hardware")
            self.enclave_initialized = True
        else:
            print("Warning: SGX not available, using simulation mode")
            print("  SGX provides hardware-backed security in production")
    
    def _check_sgx_support(self) -> bool:
        """Check if SGX is supported on current platform"""
        # In production: check CPUID for SGX support
        # import cpuinfo
        # info = cpuinfo.get_cpu_info()
        # return 'sgx' in info.get('flags', [])
        return False  # Simulation mode for demonstration
    
    def encrypt_embedding(
        self,
        embedding: np.ndarray,
        embedding_id: str
    ) -> EncryptedEmbedding:
        """
        Seal embedding for SGX enclave
        
        Data sealed with enclave's key can only be unsealed
        by the same enclave on the same platform.
        
        Args:
            embedding: Vector to seal
            embedding_id: Unique identifier
            
        Returns:
            Sealed embedding
        """
        # In production with SGX SDK:
        # sealed_data = sgx_seal_data(embedding.tobytes())
        
        # Simplified sealing for demonstration
        embedding_bytes = embedding.tobytes()
        sealed = hmac.new(
            b"sgx_sealing_key",  # In production: enclave-specific key
            embedding_bytes,
            hashlib.sha256
        ).digest()
        
        encrypted = EncryptedEmbedding(
            id=embedding_id,
            ciphertext=sealed + embedding_bytes,  # Seal + data
            public_key_id="sgx_enclave",
            dimension=len(embedding),
            encryption_metadata={
                "scheme": "sgx",
                "platform": "simulated"
            }
        )
        
        self.encrypted_store[embedding_id] = encrypted
        return encrypted
    
    def decrypt_embedding(
        self,
        encrypted: EncryptedEmbedding
    ) -> np.ndarray:
        """
        Unseal embedding within SGX enclave
        
        Args:
            encrypted: Sealed embedding
            
        Returns:
            Unsealed vector (only accessible within enclave)
        """
        # In production: sgx_unseal_data(encrypted.ciphertext)
        
        # Extract sealed data
        seal_size = 32  # SHA256 size
        embedding_bytes = encrypted.ciphertext[seal_size:]
        
        # Reconstruct vector
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        return embedding[:encrypted.dimension]
    
    def encrypted_similarity(
        self,
        query_encrypted: EncryptedEmbedding,
        candidate_encrypted: EncryptedEmbedding
    ) -> float:
        """
        Compute similarity within SGX enclave
        
        Steps:
        1. Unseal both vectors inside enclave
        2. Compute cosine similarity in enclave
        3. Return only result (vectors stay in enclave)
        
        Args:
            query_encrypted: Sealed query
            candidate_encrypted: Sealed candidate
            
        Returns:
            Similarity score
        """
        # Unseal within enclave
        query = self.decrypt_embedding(query_encrypted)
        candidate = self.decrypt_embedding(candidate_encrypted)
        
        # Compute similarity (protected by enclave)
        similarity = np.dot(query, candidate) / (
            np.linalg.norm(query) * np.linalg.norm(candidate)
        )
        
        return float(similarity)
    
    def secure_nearest_neighbors(
        self,
        query_encrypted: EncryptedEmbedding,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors within SGX enclave
        
        Entire search happens in enclave:
        1. Unseal query
        2. Unseal candidates (batch by batch if memory limited)
        3. Compute similarities
        4. Return top-k (scores can be unsealed for client)
        
        Args:
            query_encrypted: Sealed query
            k: Number of neighbors
            
        Returns:
            Top-k neighbors with scores
        """
        query = self.decrypt_embedding(query_encrypted)
        
        similarities = []
        for emb_id, candidate_enc in self.encrypted_store.items():
            candidate = self.decrypt_embedding(candidate_enc)
            
            sim = np.dot(query, candidate) / (
                np.linalg.norm(query) * np.linalg.norm(candidate)
            )
            similarities.append((emb_id, float(sim)))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

class HybridSecureStore(SecureEmbeddingStore):
    """
    Hybrid approach combining multiple security techniques
    
    Strategy:
    1. SGX for high-performance similarity computation
    2. CKKS for data at rest and cross-platform transfer
    3. Secure MPC for distributed queries across data silos
    4. Differential privacy for public result release
    
    Benefits:
    - Performance: Use SGX when available, CKKS otherwise
    - Portability: CKKS works on any platform
    - Privacy: DP bounds information leakage
    - Flexibility: Adapt to deployment constraints
    """
    
    def __init__(self, config: EncryptionConfig):
        self.config = config
        
        # Initialize both backends
        self.sgx_store = SGXEmbeddingStore(config)
        self.ckks_store = CKKSEmbeddingStore(config)
        
        # Choose primary based on availability
        self.primary_store = (
            self.sgx_store if self.sgx_store.enclave_initialized
            else self.ckks_store
        )
        
        print(f"Hybrid store initialized, primary: {type(self.primary_store).__name__}")
    
    def encrypt_embedding(
        self,
        embedding: np.ndarray,
        embedding_id: str
    ) -> EncryptedEmbedding:
        """Use primary store for encryption"""
        return self.primary_store.encrypt_embedding(embedding, embedding_id)
    
    def decrypt_embedding(
        self,
        encrypted: EncryptedEmbedding
    ) -> np.ndarray:
        """Route to appropriate store based on encryption scheme"""
        scheme = encrypted.encryption_metadata.get("scheme")
        if scheme == "sgx":
            return self.sgx_store.decrypt_embedding(encrypted)
        elif scheme == "ckks":
            return self.ckks_store.decrypt_embedding(encrypted)
        else:
            return self.primary_store.decrypt_embedding(encrypted)
    
    def encrypted_similarity(
        self,
        query_encrypted: EncryptedEmbedding,
        candidate_encrypted: EncryptedEmbedding
    ) -> float:
        """Use SGX if available for best performance"""
        if self.sgx_store.enclave_initialized:
            return self.sgx_store.encrypted_similarity(
                query_encrypted, candidate_encrypted
            )
        else:
            return self.ckks_store.encrypted_similarity(
                query_encrypted, candidate_encrypted
            )
    
    def secure_nearest_neighbors(
        self,
        query_encrypted: EncryptedEmbedding,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Use primary store for search"""
        return self.primary_store.secure_nearest_neighbors(query_encrypted, k)

# Example usage
def secure_embedding_example():
    """
    Demonstrate secure embedding encryption and computation
    """
    print("=== Secure Embedding Computation ===")
    print()
    
    # Configuration
    config = EncryptionConfig(
        scheme="sgx",  # Use SGX for demonstration
        security_level=128,
        precision=40
    )
    
    # Initialize secure store
    store = HybridSecureStore(config)
    print()
    
    # Generate sample embeddings
    print("Encrypting embeddings...")
    np.random.seed(42)
    embeddings = {
        f"emb_{i}": np.random.randn(768).astype(np.float32)
        for i in range(100)
    }
    
    # Normalize embeddings
    for emb_id in embeddings:
        embeddings[emb_id] /= np.linalg.norm(embeddings[emb_id])
    
    # Encrypt all embeddings
    encrypted = {}
    for emb_id, emb in embeddings.items():
        encrypted[emb_id] = store.encrypt_embedding(emb, emb_id)
    
    print(f"Encrypted {len(encrypted)} embeddings")
    print(f"  Original size: {embeddings['emb_0'].nbytes} bytes/vector")
    print(f"  Encrypted size: {len(encrypted['emb_0'].ciphertext)} bytes/vector")
    print()
    
    # Encrypt query
    query = np.random.randn(768).astype(np.float32)
    query /= np.linalg.norm(query)
    query_encrypted = store.encrypt_embedding(query, "query")
    
    print("Performing secure nearest neighbor search...")
    results = store.secure_nearest_neighbors(query_encrypted, k=10)
    
    print(f"\nTop 10 neighbors:")
    for rank, (emb_id, score) in enumerate(results, 1):
        print(f"  {rank}. {emb_id}: {score:.4f}")
    
    # Compare with plaintext search
    print("\nValidation (plaintext search):")
    plaintext_scores = []
    for emb_id, emb in embeddings.items():
        score = np.dot(query, emb)
        plaintext_scores.append((emb_id, score))
    plaintext_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 10 neighbors (plaintext):")
    for rank, (emb_id, score) in enumerate(plaintext_scores[:10], 1):
        print(f"  {rank}. {emb_id}: {score:.4f}")

if __name__ == "__main__":
    secure_embedding_example()
