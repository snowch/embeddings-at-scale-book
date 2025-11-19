# Code from Chapter 26
# Book: Embeddings at Scale

"""
Decentralized Embedding Marketplace

Architecture:
1. Embedding providers: Upload embeddings to IPFS, register on blockchain
2. Smart contracts: Enforce access control, payments, quality guarantees
3. Discovery: Search decentralized embedding registry
4. Verification: Prove embedding quality without revealing data
5. Payment: Cryptocurrency payment for embedding access

Benefits:
- No central authority (censorship-resistant)
- Privacy-preserving (data never centralized)
- Fair compensation (providers monetize contributions)
- Transparent governance (rules encoded in smart contracts)
- Interoperability (open standards, cross-platform)

Challenges:
- Transaction costs (gas fees on blockchain)
- Latency (slower than centralized systems)
- Scalability (blockchain throughput limits)
- Complexity (cryptographic protocols)
- Regulatory uncertainty (legal status of tokens)
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class BlockchainNetwork(Enum):
    """Blockchain networks"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    SOLANA = "solana"
    ARBITRUM = "arbitrum"

@dataclass
class EmbeddingMetadata:
    """Metadata for blockchain-registered embedding"""
    embedding_id: str
    provider_address: str
    ipfs_hash: str  # Content address on IPFS
    dimension: int
    embedding_type: str  # "text", "image", "audio", etc.
    quality_score: float
    num_samples: int
    price_per_query: float  # In tokens
    created_at: datetime
    license: str

@dataclass
class EmbeddingContract:
    """Smart contract for embedding access"""
    contract_address: str
    provider: str
    embedding_metadata: EmbeddingMetadata
    access_rules: Dict[str, Any]
    payment_terms: Dict[str, float]

class DecentralizedEmbeddingRegistry:
    """
    Blockchain-based registry for embeddings
    
    Simulates decentralized registry with:
    - Registration (upload to IPFS, register on blockchain)
    - Discovery (search registry)
    - Access control (verify permissions)
    - Payment (token transfer)
    - Verification (quality proofs)
    """

    def __init__(self, blockchain: BlockchainNetwork = BlockchainNetwork.POLYGON):
        self.blockchain = blockchain
        self.registry: Dict[str, EmbeddingMetadata] = {}
        self.contracts: Dict[str, EmbeddingContract] = {}
        self.access_logs: List[Dict] = []

    def register_embedding(
        self,
        embeddings: np.ndarray,
        metadata: Dict[str, Any],
        provider_address: str
    ) -> str:
        """
        Register embeddings on decentralized network
        
        Steps:
        1. Upload embeddings to IPFS (content-addressed storage)
        2. Create smart contract for access control
        3. Register metadata on blockchain
        4. Return embedding ID for discovery
        """
        # Simulate IPFS upload (content addressing)
        ipfs_hash = self._upload_to_ipfs(embeddings)

        # Generate embedding ID
        embedding_id = hashlib.sha256(
            f"{provider_address}{ipfs_hash}{datetime.now()}".encode()
        ).hexdigest()[:16]

        # Create metadata
        embedding_metadata = EmbeddingMetadata(
            embedding_id=embedding_id,
            provider_address=provider_address,
            ipfs_hash=ipfs_hash,
            dimension=embeddings.shape[1],
            embedding_type=metadata.get('type', 'unknown'),
            quality_score=metadata.get('quality_score', 0.0),
            num_samples=len(embeddings),
            price_per_query=metadata.get('price', 0.0),
            created_at=datetime.now(),
            license=metadata.get('license', 'proprietary')
        )

        # Deploy smart contract
        contract = self._deploy_contract(embedding_metadata, provider_address)

        # Register on blockchain
        self.registry[embedding_id] = embedding_metadata
        self.contracts[embedding_id] = contract

        return embedding_id

    def _upload_to_ipfs(self, embeddings: np.ndarray) -> str:
        """
        Upload embeddings to IPFS
        
        Returns content address (CID)
        """
        # In practice, use IPFS client library (ipfshttpclient)
        # For simulation, create hash-based CID
        content_hash = hashlib.sha256(embeddings.tobytes()).hexdigest()
        cid = f"Qm{content_hash[:44]}"  # IPFS CID format
        return cid

    def _deploy_contract(
        self,
        metadata: EmbeddingMetadata,
        provider: str
    ) -> EmbeddingContract:
        """Deploy smart contract for embedding access"""
        contract_address = f"0x{hashlib.sha256(metadata.embedding_id.encode()).hexdigest()[:40]}"

        # Access rules
        access_rules = {
            'require_payment': metadata.price_per_query > 0,
            'max_queries_per_user': 1000,
            'require_verification': True
        }

        # Payment terms
        payment_terms = {
            'price_per_query': metadata.price_per_query,
            'provider_share': 0.95,  # Provider gets 95%
            'protocol_fee': 0.05  # Protocol gets 5%
        }

        return EmbeddingContract(
            contract_address=contract_address,
            provider=provider,
            embedding_metadata=metadata,
            access_rules=access_rules,
            payment_terms=payment_terms
        )

    def search_embeddings(
        self,
        query: Dict[str, Any]
    ) -> List[EmbeddingMetadata]:
        """
        Search decentralized registry
        
        Query filters:
        - embedding_type: Type of embeddings
        - min_quality: Minimum quality score
        - max_price: Maximum price per query
        - min_samples: Minimum number of samples
        """
        results = []

        for embedding_id, metadata in self.registry.items():
            # Filter by type
            if 'embedding_type' in query:
                if metadata.embedding_type != query['embedding_type']:
                    continue

            # Filter by quality
            if 'min_quality' in query:
                if metadata.quality_score < query['min_quality']:
                    continue

            # Filter by price
            if 'max_price' in query:
                if metadata.price_per_query > query['max_price']:
                    continue

            # Filter by samples
            if 'min_samples' in query:
                if metadata.num_samples < query['min_samples']:
                    continue

            results.append(metadata)

        # Sort by quality score (descending)
        results.sort(key=lambda m: m.quality_score, reverse=True)

        return results

    def request_access(
        self,
        embedding_id: str,
        user_address: str,
        num_queries: int = 1
    ) -> Dict[str, Any]:
        """
        Request access to embeddings
        
        Verifies payment and permissions via smart contract
        """
        if embedding_id not in self.contracts:
            return {'success': False, 'error': 'Embedding not found'}

        contract = self.contracts[embedding_id]
        metadata = contract.embedding_metadata

        # Calculate payment
        total_price = metadata.price_per_query * num_queries

        # Verify payment (simulated)
        payment_verified = self._verify_payment(user_address, total_price)

        if not payment_verified:
            return {'success': False, 'error': 'Payment verification failed'}

        # Grant access
        access_token = self._generate_access_token(embedding_id, user_address, num_queries)

        # Log access
        self.access_logs.append({
            'embedding_id': embedding_id,
            'user_address': user_address,
            'num_queries': num_queries,
            'price_paid': total_price,
            'timestamp': datetime.now()
        })

        return {
            'success': True,
            'access_token': access_token,
            'ipfs_hash': metadata.ipfs_hash,
            'queries_remaining': num_queries
        }

    def _verify_payment(self, user_address: str, amount: float) -> bool:
        """Verify cryptocurrency payment (simulated)"""
        # In practice, verify blockchain transaction
        return True  # Assume payment successful for demo

    def _generate_access_token(
        self,
        embedding_id: str,
        user_address: str,
        num_queries: int
    ) -> str:
        """Generate access token for embedding queries"""
        token_data = f"{embedding_id}{user_address}{num_queries}{datetime.now()}"
        token = hashlib.sha256(token_data.encode()).hexdigest()
        return token

    def download_embedding(
        self,
        ipfs_hash: str,
        access_token: str
    ) -> Optional[np.ndarray]:
        """
        Download embedding from IPFS
        
        Requires valid access token
        """
        # Verify access token
        if not self._verify_access_token(access_token):
            return None

        # Download from IPFS (simulated)
        # In practice, use IPFS client: ipfs.get(ipfs_hash)

        # Return placeholder embedding
        embedding = np.random.randn(1000, 768)
        return embedding

    def _verify_access_token(self, token: str) -> bool:
        """Verify access token validity"""
        # In practice, check blockchain state
        return True  # Assume valid for demo


### Zero-Knowledge Proofs for Private Embedding Verification

# Zero-knowledge proofs (ZKPs) are cryptographic protocols enabling verification without revealing underlying data.
# They allow embedding providers to prove quality, integrity, and properties without exposing embeddings.
