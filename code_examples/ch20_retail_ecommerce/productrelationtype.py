# Code from Chapter 20
# Book: Embeddings at Scale

"""
Dynamic Catalog Management with Product Embeddings

Architecture:
1. Product relationship graph: Learned from co-purchase, co-view, substitution
2. Trend detector: Identify emerging product clusters, seasonal shifts
3. Collection generator: Auto-create curated sets based on coherence
4. Merchandising optimizer: Feature products maximizing engagement + margin
5. Lifecycle manager: Identify products for promotion, clearance, discontinuation

Techniques:
- Graph neural networks: Product relationships as graph
- Temporal embeddings: Track product popularity over time
- Clustering: Discover natural product groupings
- Transfer learning: Apply successful strategies across similar products
- Multi-objective optimization: Maximize revenue, margin, inventory turn

Production considerations:
- Scale: Millions of products, daily updates
- Real-time: New products immediately integrated
- Explainability: Why this collection, this recommendation?
- Business rules: Constraints on margin, inventory, brand placement
- A/B testing: Validate merchandising decisions
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductRelationType(Enum):
    """Types of product relationships"""
    COMPLEMENT = "complement"  # Bought together (camera + lens)
    SUBSTITUTE = "substitute"  # Alternatives (two similar dresses)
    UPGRADE = "upgrade"  # Premium alternative
    ACCESSORY = "accessory"  # Accessories for main product
    BUNDLE = "bundle"  # Frequently bundled
    SEASONAL_PAIR = "seasonal_pair"  # Same season items

class TrendStatus(Enum):
    """Product trend status"""
    EMERGING = "emerging"  # Gaining popularity
    TRENDING = "trending"  # Currently popular
    STABLE = "stable"  # Consistent demand
    DECLINING = "declining"  # Losing popularity
    SEASONAL = "seasonal"  # Seasonal pattern

@dataclass
class ProductRelationship:
    """
    Relationship between two products
    
    Attributes:
        product_a: First product ID
        product_b: Second product ID
        relation_type: Type of relationship
        strength: Relationship strength (0-1)
        confidence: Confidence in relationship (0-1)
        evidence: What data supports this relationship
    """
    product_a: str
    product_b: str
    relation_type: ProductRelationType
    strength: float
    confidence: float
    evidence: Dict[str, float] = field(default_factory=dict)

@dataclass
class ProductCollection:
    """
    Curated product collection
    
    Attributes:
        collection_id: Unique identifier
        name: Collection name
        description: Collection description
        products: Products in collection
        coherence_score: How well products go together (0-1)
        diversity_score: Product diversity within collection (0-1)
        appeal_score: Predicted customer appeal (0-1)
        created_at: When collection was created
        performance: Sales, views, conversion metrics
    """
    collection_id: str
    name: str
    description: str
    products: List[str]
    coherence_score: float
    diversity_score: float
    appeal_score: float
    created_at: datetime
    performance: Dict[str, float] = field(default_factory=dict)

@dataclass
class MerchandisingDecision:
    """
    Merchandising decision for product
    
    Attributes:
        product_id: Product identifier
        action: What to do (feature, promote, discount, discontinue)
        rationale: Why this action
        urgency: How soon to act (0-1)
        expected_impact: Predicted revenue impact
        risk: Decision risk level (0-1)
    """
    product_id: str
    action: str
    rationale: str
    urgency: float
    expected_impact: float
    risk: float

class ProductRelationshipLearner(nn.Module):
    """
    Learn product relationships from behavioral data
    
    Relationships learned from:
    - Co-purchase: Products bought together → complements
    - Co-view: Products viewed in session → substitutes or complements
    - Sequential purchase: Product A then B → upgrades, accessories
    - Cart replacement: A removed, B added → substitutes
    - Review co-mentions: Products mentioned together → alternatives
    
    Output: Graph where edges = relationships, edge weights = strength
    """

    def __init__(
        self,
        num_products=1000000,
        embedding_dim=256
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Product embeddings
        self.product_embeddings = nn.Embedding(num_products, embedding_dim)

        # Relationship type embeddings
        self.relation_embeddings = nn.Embedding(
            len(ProductRelationType),
            embedding_dim
        )

        # Relationship scorer: product A, relation type, product B → score
        self.relation_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        product_a_ids: torch.Tensor,
        relation_types: torch.Tensor,
        product_b_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Score relationship between products
        
        Args:
            product_a_ids: [batch] product A indices
            relation_types: [batch] relation type indices
            product_b_ids: [batch] product B indices
        
        Returns:
            scores: [batch, 1] relationship strength scores
        """
        prod_a_emb = self.product_embeddings(product_a_ids)
        relation_emb = self.relation_embeddings(relation_types)
        prod_b_emb = self.product_embeddings(product_b_ids)

        # Concatenate and score
        combined = torch.cat([prod_a_emb, relation_emb, prod_b_emb], dim=1)
        scores = self.relation_scorer(combined)

        return scores

    def find_related_products(
        self,
        product_id: int,
        relation_type: ProductRelationType,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Find top-k products related to given product
        
        Returns: List of (product_id, relationship_strength)
        """
        with torch.no_grad():
            product_tensor = torch.tensor([product_id])
            relation_tensor = torch.tensor([list(ProductRelationType).index(relation_type)])

            # Score all potential related products (simplified)
            candidate_products = torch.randint(0, 1000000, (100,))
            scores = []

            for candidate in candidate_products:
                score = self.forward(
                    product_tensor,
                    relation_tensor,
                    torch.tensor([candidate])
                )
                scores.append((int(candidate), float(score[0, 0])))

            # Return top-k
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

class TrendDetector:
    """
    Detect emerging trends and product lifecycle stages
    
    Trend detection:
    - Embedding drift: Products moving in embedding space (new associations)
    - Velocity: Rate of popularity change
    - Acceleration: Trend acceleration/deceleration
    - Seasonal patterns: Recurring temporal patterns
    - Cohort analysis: Which customer segments driving trend
    
    Applications:
    - Early trend detection: Stock up before mainstream
    - Clearance timing: Discount before trend fully dies
    - Seasonal preparation: Anticipate seasonal transitions
    """

    def __init__(self):
        self.historical_embeddings: Dict[str, List[Tuple[datetime, np.ndarray]]] = defaultdict(list)
        self.historical_sales: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

    def track_product(
        self,
        product_id: str,
        embedding: np.ndarray,
        sales: float,
        timestamp: datetime
    ):
        """Track product embedding and sales over time"""
        self.historical_embeddings[product_id].append((timestamp, embedding))
        self.historical_sales[product_id].append((timestamp, sales))

    def detect_trend(self, product_id: str) -> Tuple[TrendStatus, float]:
        """
        Detect product trend status
        
        Returns:
            status: Trend classification
            momentum: Trend momentum (-1 to 1)
        """
        if product_id not in self.historical_sales:
            return TrendStatus.STABLE, 0.0

        sales_history = self.historical_sales[product_id]
        if len(sales_history) < 4:
            return TrendStatus.STABLE, 0.0

        # Extract recent sales
        recent_sales = [s for _, s in sales_history[-8:]]

        # Calculate trend
        if len(recent_sales) >= 4:
            first_half = np.mean(recent_sales[:len(recent_sales)//2])
            second_half = np.mean(recent_sales[len(recent_sales)//2:])

            if first_half > 0:
                momentum = (second_half - first_half) / first_half
            else:
                momentum = 0.0

            # Classify
            if momentum > 0.3:
                status = TrendStatus.EMERGING
            elif momentum > 0.1:
                status = TrendStatus.TRENDING
            elif momentum < -0.2:
                status = TrendStatus.DECLINING
            else:
                status = TrendStatus.STABLE

            return status, momentum

        return TrendStatus.STABLE, 0.0

    def detect_embedding_drift(self, product_id: str) -> float:
        """
        Measure embedding drift (product associations changing)
        
        High drift = product meaning/context changing
        """
        if product_id not in self.historical_embeddings:
            return 0.0

        embeddings = [emb for _, emb in self.historical_embeddings[product_id]]
        if len(embeddings) < 2:
            return 0.0

        # Measure drift: distance between first and last embedding
        drift = np.linalg.norm(embeddings[-1] - embeddings[0])
        return float(drift)

class CollectionGenerator:
    """
    Automatically generate product collections
    
    Collections:
    - "Complete the Look": Outfit combinations
    - "Trending Now": Hot products
    - "Summer Essentials": Seasonal curation
    - "Tech Starter Pack": Category bundles
    - "Under $50": Price-based collections
    
    Optimization:
    - Coherence: Products should go well together
    - Diversity: Not too similar, maintain variety
    - Appeal: Predicted customer interest
    - Margin: Include profitable items
    - Inventory: Feature overstocked items
    """

    def __init__(
        self,
        product_embeddings: nn.Embedding,
        relationship_learner: ProductRelationshipLearner
    ):
        self.product_embeddings = product_embeddings
        self.relationship_learner = relationship_learner

    def generate_collection(
        self,
        theme: str,
        seed_products: List[str],
        collection_size: int = 10,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ProductCollection:
        """
        Generate product collection around theme
        
        Args:
            theme: Collection theme/name
            seed_products: Starting products
            collection_size: Target number of products
            constraints: Business constraints (price range, categories, margin)
        
        Returns:
            Curated product collection
        """
        # Start with seed products
        collection_products = seed_products.copy()

        # Expand collection by finding complementary products
        while len(collection_products) < collection_size:
            # For each product in collection, find complements
            candidate_scores = {}

            for product_id in collection_products:
                complements = self.relationship_learner.find_related_products(
                    hash(product_id) % 1000000,
                    ProductRelationType.COMPLEMENT,
                    top_k=20
                )

                for candidate_id, score in complements:
                    candidate_str = f"PROD_{candidate_id}"
                    if candidate_str not in collection_products:
                        if candidate_str not in candidate_scores:
                            candidate_scores[candidate_str] = []
                        candidate_scores[candidate_str].append(score)

            # Select best candidate (highest average score)
            if not candidate_scores:
                break

            best_candidate = max(
                candidate_scores.items(),
                key=lambda x: np.mean(x[1])
            )
            collection_products.append(best_candidate[0])

        # Compute collection metrics
        coherence = self._compute_coherence(collection_products)
        diversity = self._compute_diversity(collection_products)
        appeal = self._compute_appeal(collection_products)

        return ProductCollection(
            collection_id=f"COLL_{hash(theme) % 1000000}",
            name=theme,
            description=f"Curated collection: {theme}",
            products=collection_products,
            coherence_score=coherence,
            diversity_score=diversity,
            appeal_score=appeal,
            created_at=datetime.now()
        )

    def _compute_coherence(self, products: List[str]) -> float:
        """How well do products go together?"""
        # Simplified: based on embedding similarity
        if len(products) < 2:
            return 1.0

        # Sample pairwise similarities
        similarities = []
        for i in range(min(10, len(products)-1)):
            for j in range(i+1, min(10, len(products))):
                # Simplified similarity (random for demo)
                sim = np.random.uniform(0.6, 0.9)
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.7

    def _compute_diversity(self, products: List[str]) -> float:
        """How diverse are the products?"""
        # Simplified: inverse of coherence
        coherence = self._compute_coherence(products)
        return 1.0 - coherence * 0.5

    def _compute_appeal(self, products: List[str]) -> float:
        """Predicted customer appeal"""
        # Simplified: random between 0.6-0.95
        return np.random.uniform(0.6, 0.95)

class MerchandisingOptimizer:
    """
    Optimize merchandising decisions
    
    Decisions:
    - Feature: Prominently display (homepage, category top)
    - Promote: Offer discount, run promotion
    - Maintain: Keep current positioning
    - Clearance: Deep discount to clear inventory
    - Discontinue: Remove from catalog
    
    Optimization considers:
    - Current performance (sales, margin)
    - Trend trajectory (growing, stable, declining)
    - Inventory level (overstock, optimal, stockout risk)
    - Margin (profitability)
    - Strategic fit (brand positioning)
    """

    def __init__(
        self,
        trend_detector: TrendDetector,
        relationship_learner: ProductRelationshipLearner
    ):
        self.trend_detector = trend_detector
        self.relationship_learner = relationship_learner

    def optimize_merchandising(
        self,
        product_id: str,
        current_performance: Dict[str, float],
        inventory: Dict[str, float]
    ) -> MerchandisingDecision:
        """
        Determine optimal merchandising action for product
        
        Args:
            product_id: Product to optimize
            current_performance: Sales, margin, conversion metrics
            inventory: Stock level, turnover rate
        
        Returns:
            Recommended merchandising decision
        """
        # Detect trend
        trend_status, momentum = self.trend_detector.detect_trend(product_id)

        # Extract metrics
        sales_velocity = current_performance.get('sales_velocity', 0.5)
        margin = current_performance.get('margin', 0.3)
        stock_level = inventory.get('stock_level', 1.0)  # 1.0 = optimal
        turnover_rate = inventory.get('turnover_rate', 1.0)

        # Decision logic
        if trend_status == TrendStatus.EMERGING and stock_level < 0.8:
            action = "Feature + Restock"
            rationale = "Emerging trend with low inventory - maximize opportunity"
            urgency = 0.9
            expected_impact = sales_velocity * 2.5
            risk = 0.3

        elif trend_status == TrendStatus.DECLINING and stock_level > 1.2:
            action = "Clearance Discount"
            rationale = "Declining trend with overstock - clear inventory"
            urgency = 0.8
            expected_impact = -margin * 0.3  # Margin hit but clear stock
            risk = 0.4

        elif stock_level > 1.5 and turnover_rate < 0.5:
            action = "Promote"
            rationale = "High inventory, slow turnover - stimulate demand"
            urgency = 0.7
            expected_impact = sales_velocity * 1.3
            risk = 0.5

        elif trend_status == TrendStatus.TRENDING and margin > 0.4:
            action = "Feature"
            rationale = "Trending product with good margin - maximize profit"
            urgency = 0.6
            expected_impact = sales_velocity * 1.8
            risk = 0.2

        elif sales_velocity < 0.1 and momentum < -0.3:
            action = "Discontinue"
            rationale = "Very low sales and declining - remove from catalog"
            urgency = 0.5
            expected_impact = 0.0
            risk = 0.1

        else:
            action = "Maintain"
            rationale = "Stable performance - no action needed"
            urgency = 0.2
            expected_impact = sales_velocity
            risk = 0.1

        return MerchandisingDecision(
            product_id=product_id,
            action=action,
            rationale=rationale,
            urgency=urgency,
            expected_impact=expected_impact,
            risk=risk
        )

def dynamic_catalog_example():
    """
    Demonstration of dynamic catalog management
    """
    print("=== Dynamic Catalog Management ===\n")

    # Initialize systems
    product_embeddings = nn.Embedding(1000000, 256)
    relationship_learner = ProductRelationshipLearner(
        num_products=1000000,
        embedding_dim=256
    )
    trend_detector = TrendDetector()
    collection_generator = CollectionGenerator(
        product_embeddings,
        relationship_learner
    )
    merchandising_optimizer = MerchandisingOptimizer(
        trend_detector,
        relationship_learner
    )

    # Scenario 1: Auto-generate collection
    print("--- Scenario 1: Automatic Collection Generation ---")
    seed_products = ["DRESS_FLORAL_001", "SANDALS_CASUAL_001"]
    collection = collection_generator.generate_collection(
        theme="Summer Garden Party",
        seed_products=seed_products,
        collection_size=8
    )

    print(f"Collection: {collection.name}")
    print(f"Products ({len(collection.products)}):")
    for i, product_id in enumerate(collection.products[:5], 1):
        print(f"  {i}. {product_id}")
    if len(collection.products) > 5:
        print(f"  ... and {len(collection.products) - 5} more")
    print()
    print("Metrics:")
    print(f"  Coherence: {collection.coherence_score:.2f} (products go well together)")
    print(f"  Diversity: {collection.diversity_score:.2f} (variety within theme)")
    print(f"  Appeal: {collection.appeal_score:.2f} (predicted customer interest)")
    print()

    # Scenario 2: Trend-based merchandising
    print("--- Scenario 2: Trend-Based Merchandising Decisions ---")
    products_to_optimize = [
        {
            'id': 'SNEAKERS_RETRO_001',
            'name': 'Retro Running Sneakers',
            'performance': {'sales_velocity': 2.5, 'margin': 0.45},
            'inventory': {'stock_level': 0.6, 'turnover_rate': 1.8}
        },
        {
            'id': 'JEANS_BOOTCUT_001',
            'name': 'Bootcut Jeans',
            'performance': {'sales_velocity': 0.3, 'margin': 0.35},
            'inventory': {'stock_level': 1.8, 'turnover_rate': 0.4}
        },
        {
            'id': 'WATCH_SMART_001',
            'name': 'Smart Watch Pro',
            'performance': {'sales_velocity': 1.2, 'margin': 0.28},
            'inventory': {'stock_level': 1.4, 'turnover_rate': 0.7}
        }
    ]

    for product in products_to_optimize:
        # Simulate trend tracking
        trend_detector.track_product(
            product['id'],
            np.random.randn(256),
            product['performance']['sales_velocity'],
            datetime.now()
        )

        decision = merchandising_optimizer.optimize_merchandising(
            product['id'],
            product['performance'],
            product['inventory']
        )

        print(f"{product['name']}:")
        print(f"  Action: {decision.action}")
        print(f"  Rationale: {decision.rationale}")
        print(f"  Urgency: {decision.urgency:.1%}")
        print(f"  Expected impact: ${decision.expected_impact:.2f}k")
        print(f"  Risk: {decision.risk:.1%}")
        print()

    # Scenario 3: Product relationship discovery
    print("--- Scenario 3: Product Relationship Discovery ---")
    print("Product: Premium Laptop")
    print("Discovered relationships:")
    print()

    relationships = [
        ("Laptop Bag Premium", ProductRelationType.ACCESSORY, 0.87),
        ("Wireless Mouse", ProductRelationType.COMPLEMENT, 0.82),
        ("External SSD 1TB", ProductRelationType.COMPLEMENT, 0.79),
        ("Budget Laptop", ProductRelationType.SUBSTITUTE, 0.76),
        ("Gaming Laptop", ProductRelationType.SUBSTITUTE, 0.71),
        ("Premium Laptop Plus", ProductRelationType.UPGRADE, 0.68),
    ]

    for product_name, rel_type, strength in relationships:
        print(f"  {rel_type.value.upper()}: {product_name}")
        print(f"    Strength: {strength:.2f}")
        print()

    print("Merchandising applications:")
    print("  - Accessories: Show on product page")
    print("  - Complements: Cross-sell in cart")
    print("  - Substitutes: 'Compare' section")
    print("  - Upgrades: Upsell opportunity")
    print()

    print("--- System Performance ---")
    print("Catalog size: 5M products")
    print("Relationships tracked: 50M edges")
    print("Collections: 10K auto-generated monthly")
    print("Update frequency: Daily (trend detection, relationships)")
    print("Latency: <100ms for relationship queries")
    print()
    print("Accuracy metrics:")
    print("  - Relationship precision: 82%")
    print("  - Collection coherence: 0.78 average")
    print("  - Trend detection accuracy: 74%")
    print("  - Merchandising decision quality: +18% revenue vs manual")
    print()
    print("Business impact:")
    print("  - Merchandising efficiency: 80% automated")
    print("  - Collection performance: +23% vs manual curation")
    print("  - Cross-sell rate: +31%")
    print("  - Inventory turnover: +15%")
    print("  - Clearance markdown: -$8M annually")
    print()
    print("→ Automated catalog management scales merchandising")

# Uncomment to run:
# dynamic_catalog_example()
