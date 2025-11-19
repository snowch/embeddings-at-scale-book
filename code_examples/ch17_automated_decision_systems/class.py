from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Code from Chapter 17
# Book: Embeddings at Scale

"""
Dynamic Pricing with Embeddings

Architecture:
1. Product encoder: Product features → embedding
2. Customer encoder: Customer features → embedding
3. Context encoder: Market conditions, time, inventory
4. Price model: (product, customer, context) → optimal price

Techniques:
- Demand modeling: Predict purchase probability at each price point
- Elasticity learning: Encode price sensitivity in customer embedding
- Competitive positioning: Products close in embedding space compete
- Inventory pressure: Adjust price based on stock levels

Production:
- Real-time: Recompute prices as inventory/demand changes
- A/B testing: Randomized price experiments
- Constraints: Minimum margins, price stability (avoid volatility)
"""

@dataclass
class Product:
    """
    Product available for purchase

    Attributes:
        product_id: Unique identifier
        category: Product category
        brand: Brand name
        features: Product features (size, color, etc.)
        cost: Unit cost
        inventory: Current inventory level
        base_price: Base price (MSRP)
        embedding: Learned product embedding
    """
    product_id: str
    category: str
    brand: str
    features: Dict[str, Any]
    cost: float
    inventory: int
    base_price: float
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.features is None:
            self.features = {}

@dataclass
class Customer:
    """
    Customer making purchase decision

    Attributes:
        customer_id: Unique identifier
        demographics: Age, location, income
        purchase_history: Past purchases
        browsing_history: Products viewed
        price_sensitivity: Estimated price sensitivity
        embedding: Learned customer embedding
    """
    customer_id: str
    demographics: Dict[str, Any]
    purchase_history: List[str]
    browsing_history: List[str]
    price_sensitivity: Optional[float] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.demographics is None:
            self.demographics = {}
        if self.purchase_history is None:
            self.purchase_history = []
        if self.browsing_history is None:
            self.browsing_history = []

class ProductEncoder(nn.Module):
    """
    Encode products to embeddings for pricing

    Architecture:
    - Product features: Category, brand, attributes
    - Price history: Historical prices and demand
    - Competitive context: Similar product prices
    - MLP: Combine features into product embedding
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_categories: int = 100,
        num_brands: int = 1000
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Categorical embeddings
        self.category_embedding = nn.Embedding(num_categories, 32)
        self.brand_embedding = nn.Embedding(num_brands, 32)

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(64 + 10, 128),  # +10 for numerical features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )

    def forward(
        self,
        category_ids: torch.Tensor,
        brand_ids: torch.Tensor,
        numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode products

        Args:
            category_ids: Category IDs (batch_size,)
            brand_ids: Brand IDs (batch_size,)
            numerical_features: Numerical features (batch_size, num_features)

        Returns:
            Product embeddings (batch_size, embedding_dim)
        """
        # Embed categorical features
        cat_emb = self.category_embedding(category_ids)
        brand_emb = self.brand_embedding(brand_ids)

        # Concatenate
        combined = torch.cat([cat_emb, brand_emb, numerical_features], dim=1)

        # Encode
        product_emb = self.feature_encoder(combined)

        # Normalize
        product_emb = F.normalize(product_emb, p=2, dim=1)

        return product_emb

class CustomerEncoder(nn.Module):
    """
    Encode customers to embeddings for pricing

    Architecture:
    - Demographics: Age, income, location
    - Purchase history: Past products, average spend
    - Price sensitivity: Inferred from purchase patterns
    - MLP: Combine features into customer embedding
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_customers: int = 1000000
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Customer ID embedding
        self.customer_id_embedding = nn.Embedding(num_customers, embedding_dim // 2)

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(embedding_dim // 2 + 20, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )

    def forward(
        self,
        customer_ids: torch.Tensor,
        numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode customers

        Args:
            customer_ids: Customer IDs (batch_size,)
            numerical_features: Numerical features (batch_size, num_features)

        Returns:
            Customer embeddings (batch_size, embedding_dim)
        """
        # Embed customer IDs
        cust_emb = self.customer_id_embedding(customer_ids)

        # Concatenate with features
        combined = torch.cat([cust_emb, numerical_features], dim=1)

        # Encode
        customer_emb = self.feature_encoder(combined)

        # Normalize
        customer_emb = F.normalize(customer_emb, p=2, dim=1)

        return customer_emb

class DemandModel(nn.Module):
    """
    Predict purchase probability as function of price

    Model: P(purchase | product, customer, price, context)

    Architecture:
    - Input: (product_emb, customer_emb, price, context)
    - MLP: Predict purchase probability
    - Training: Binary classification (purchased / not purchased)

    Usage:
    - For each price point, predict demand
    - Optimize: price * demand * (price - cost)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        context_dim: int = 10
    ):
        super().__init__()

        # Combined encoder
        input_dim = embedding_dim * 2 + 1 + context_dim  # product + customer + price + context

        self.demand_predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        product_emb: torch.Tensor,
        customer_emb: torch.Tensor,
        price: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict purchase probability

        Args:
            product_emb: Product embeddings (batch_size, embedding_dim)
            customer_emb: Customer embeddings (batch_size, embedding_dim)
            price: Prices (batch_size, 1)
            context: Context features (batch_size, context_dim)

        Returns:
            Purchase probabilities (batch_size, 1)
        """
        # Concatenate all inputs
        combined = torch.cat([product_emb, customer_emb, price, context], dim=1)

        # Predict purchase probability
        purchase_prob = self.demand_predictor(combined)

        return purchase_prob

class DynamicPricingEngine:
    """
    Dynamic pricing engine using embeddings

    Approach:
    1. Encode product and customer to embeddings
    2. For each candidate price:
       a. Predict purchase probability (demand model)
       b. Compute expected profit: price * prob * (price - cost)
    3. Select price maximizing expected profit
    4. Apply constraints (minimum margin, price stability)
    """

    def __init__(
        self,
        product_encoder: ProductEncoder,
        customer_encoder: CustomerEncoder,
        demand_model: DemandModel,
        min_margin: float = 0.2,
        max_price_change: float = 0.15
    ):
        self.product_encoder = product_encoder
        self.customer_encoder = customer_encoder
        self.demand_model = demand_model
        self.min_margin = min_margin
        self.max_price_change = max_price_change

    def optimize_price(
        self,
        product: Product,
        customer: Customer,
        context: Dict[str, Any],
        num_price_points: int = 20
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Optimize price for product-customer pair

        Args:
            product: Product to price
            customer: Customer considering purchase
            context: Market context (time, inventory, etc.)
            num_price_points: Number of price points to evaluate

        Returns:
            (optimal_price, analysis)
        """
        # Generate candidate prices
        min_price = product.cost * (1 + self.min_margin)
        max_price = product.base_price * (1 + self.max_price_change)
        candidate_prices = np.linspace(min_price, max_price, num_price_points)

        # Encode product and customer (simplified - would use actual features)
        product_emb = product.embedding
        customer_emb = customer.embedding

        # Evaluate each price
        best_price = None
        best_profit = -float('inf')
        price_analysis = []

        with torch.no_grad():
            for price in candidate_prices:
                # Convert to tensors
                product_emb_t = torch.tensor(product_emb).unsqueeze(0).float()
                customer_emb_t = torch.tensor(customer_emb).unsqueeze(0).float()
                price_t = torch.tensor([[price]]).float()

                # Simplified context
                context_t = torch.tensor([[
                    context.get('hour', 12) / 24.0,
                    context.get('day_of_week', 3) / 7.0,
                    product.inventory / 1000.0,
                    context.get('competitor_price', product.base_price) / product.base_price,
                    *[0.0] * 6  # Placeholder
                ]]).float()

                # Predict demand
                purchase_prob = self.demand_model(
                    product_emb_t,
                    customer_emb_t,
                    price_t,
                    context_t
                ).item()

                # Compute expected profit
                margin = price - product.cost
                expected_profit = purchase_prob * margin

                price_analysis.append({
                    'price': price,
                    'purchase_prob': purchase_prob,
                    'margin': margin,
                    'expected_profit': expected_profit
                })

                if expected_profit > best_profit:
                    best_profit = expected_profit
                    best_price = price

        return best_price, {
            'expected_profit': best_profit,
            'price_analysis': price_analysis,
            'base_price': product.base_price,
            'cost': product.cost
        }

# Example: E-commerce dynamic pricing
def dynamic_pricing_example():
    """
    Dynamic pricing for e-commerce

    Scenario:
    - Product: Wireless headphones
    - Cost: $50
    - Base price: $100
    - Customer segments: Price-sensitive, Premium, Impulse buyers

    Pricing optimization:
    - Price-sensitive: Lower price (higher demand elasticity)
    - Premium: Higher price (lower elasticity, values quality)
    - Impulse: Base price (time-sensitive)
    - Clearance: Lower price (high inventory pressure)
    """

    print("=== Dynamic Pricing System ===")
    print("\nProduct: Wireless Headphones")
    print("  Cost: $50")
    print("  Base price: $100")
    print("  Current inventory: 500 units")

    print("\n--- Customer Segment: Price-Sensitive ---")
    print("Characteristics:")
    print("  - Compares prices across sites")
    print("  - Waits for sales")
    print("  - High price elasticity")
    print("Traditional pricing: $100 (fixed)")
    print("  Purchase probability: 15%")
    print("  Expected profit: $100 * 0.15 * ($100 - $50) = $7.50")
    print("\nDynamic pricing: $79")
    print("  Purchase probability: 45%")
    print("  Expected profit: $79 * 0.45 * ($79 - $50) = $10.33")
    print("  → 38% profit increase by lowering price")

    print("\n--- Customer Segment: Premium ---")
    print("Characteristics:")
    print("  - Values quality and brand")
    print("  - Less price-sensitive")
    print("  - Low price elasticity")
    print("Traditional pricing: $100 (fixed)")
    print("  Purchase probability: 60%")
    print("  Expected profit: $100 * 0.60 * ($100 - $50) = $30.00")
    print("\nDynamic pricing: $115")
    print("  Purchase probability: 52%")
    print("  Expected profit: $115 * 0.52 * ($115 - $50) = $38.87")
    print("  → 30% profit increase by raising price")

    print("\n--- Customer Segment: Impulse Buyer ---")
    print("Characteristics:")
    print("  - Making quick decision")
    print("  - Moderate price sensitivity")
    print("  - Time pressure")
    print("Traditional pricing: $100 (fixed)")
    print("  Purchase probability: 40%")
    print("  Expected profit: $100 * 0.40 * ($100 - $50) = $20.00")
    print("\nDynamic pricing: $95 (slight discount for urgency)")
    print("  Purchase probability: 48%")
    print("  Expected profit: $95 * 0.48 * ($95 - $50) = $20.52")
    print("  → 3% profit increase")

    print("\n--- Clearance Scenario: High Inventory ---")
    print("Inventory: 2,000 units (excess stock)")
    print("Days until new model: 14 days")
    print("Traditional pricing: $100 → $70 (30% off)")
    print("  Purchase probability: 50%")
    print("  Expected profit: $70 * 0.50 * ($70 - $50) = $7.00")
    print("\nDynamic pricing: Personalized clearance")
    print("  Price-sensitive customers: $65 (65% purchase prob)")
    print("  Premium customers: $85 (40% purchase prob)")
    print("  Average expected profit: $10.20")
    print("  → 46% profit increase vs blanket discount")

# Uncomment to run:
# dynamic_pricing_example()
