# Code from Chapter 17
# Book: Embeddings at Scale

"""
Supply Chain Optimization with Embeddings

Architecture:
1. Facility encoder: Location, capacity, costs → embedding
2. Product encoder: Size, weight, fragility → embedding
3. Route encoder: Distance, time, reliability → embedding
4. Cost model: (facility, product, route) → predicted cost

Applications:
- Warehouse selection: Which warehouse to fulfill order from
- Supplier selection: Which supplier for each component
- Route optimization: Which route for each shipment
- Inventory allocation: How much inventory at each location

Techniques:
- Graph embeddings: Supply chain as graph, learn node embeddings
- Reinforcement learning: Learn routing policy from simulations
- Demand forecasting: Predict future demand using embeddings
"""

@dataclass
class Facility:
    """
    Warehouse or distribution center
    
    Attributes:
        facility_id: Unique identifier
        location: (latitude, longitude)
        capacity: Storage capacity
        inventory: Current inventory by product
        cost_structure: Fixed and variable costs
        embedding: Learned facility embedding
    """
    facility_id: str
    location: Tuple[float, float]
    capacity: float
    inventory: Dict[str, int]
    cost_structure: Dict[str, float]
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.inventory is None:
            self.inventory = {}
        if self.cost_structure is None:
            self.cost_structure = {}

@dataclass
class Order:
    """
    Customer order to fulfill
    
    Attributes:
        order_id: Unique identifier
        customer_location: (latitude, longitude)
        products: List of (product_id, quantity)
        delivery_deadline: Latest acceptable delivery time
        priority: Order priority (standard, expedited, overnight)
    """
    order_id: str
    customer_location: Tuple[float, float]
    products: List[Tuple[str, int]]
    delivery_deadline: float
    priority: str = 'standard'

class FacilityEncoder(nn.Module):
    """
    Encode facilities to embeddings
    
    Architecture:
    - Location: Geographic coordinates
    - Capacity: Storage capacity, throughput
    - Costs: Fixed costs, variable costs per unit
    - Performance: Historical on-time rate, damage rate
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(20, 128),  # Location, capacity, costs, performance
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, facility_features: torch.Tensor) -> torch.Tensor:
        """
        Encode facilities
        
        Args:
            facility_features: Facility features (batch_size, num_features)
        
        Returns:
            Facility embeddings (batch_size, embedding_dim)
        """
        facility_emb = self.encoder(facility_features)
        facility_emb = F.normalize(facility_emb, p=2, dim=1)
        return facility_emb

class RouteCostModel(nn.Module):
    """
    Predict cost of fulfilling order from facility
    
    Model: cost(facility, order) = f(facility_emb, order_emb, distance, ...)
    
    Factors:
    - Distance: Shipping distance
    - Urgency: Delivery deadline, priority
    - Inventory: Product availability at facility
    - Capacity: Facility utilization
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        input_dim = embedding_dim * 2 + 10  # facility + order + distance/urgency/etc

        self.cost_predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        facility_emb: torch.Tensor,
        order_emb: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict fulfillment cost
        
        Args:
            facility_emb: Facility embeddings (batch_size, embedding_dim)
            order_emb: Order embeddings (batch_size, embedding_dim)
            context: Context features (batch_size, num_features)
        
        Returns:
            Predicted costs (batch_size, 1)
        """
        combined = torch.cat([facility_emb, order_emb, context], dim=1)
        cost = self.cost_predictor(combined)
        return cost

class SupplyChainOptimizer:
    """
    Optimize supply chain decisions using embeddings
    
    Decisions:
    - Warehouse selection: Which warehouse fulfills each order
    - Inventory allocation: How much inventory at each warehouse
    - Supplier selection: Which supplier for each product
    - Route optimization: Which route for each shipment
    """

    def __init__(
        self,
        facility_encoder: FacilityEncoder,
        cost_model: RouteCostModel
    ):
        self.facility_encoder = facility_encoder
        self.cost_model = cost_model

    def select_fulfillment_facility(
        self,
        order: Order,
        facilities: List[Facility],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Select which facility should fulfill order
        
        Args:
            order: Order to fulfill
            facilities: Available facilities
            constraints: Service level, capacity constraints
        
        Returns:
            (facility_id, cost, analysis)
        """
        constraints = constraints or {}

        # Encode order (simplified)
        order_features = self._extract_order_features(order)
        order_emb = torch.tensor(order_features).unsqueeze(0).float()

        best_facility = None
        best_cost = float('inf')
        facility_analysis = []

        with torch.no_grad():
            for facility in facilities:
                # Check constraints
                if not self._check_facility_constraints(facility, order, constraints):
                    continue

                # Encode facility
                facility_features = self._extract_facility_features(facility, order)
                facility_emb = torch.tensor(facility_features).unsqueeze(0).float()

                # Context features
                context = self._extract_context_features(facility, order)
                context_t = torch.tensor(context).unsqueeze(0).float()

                # Predict cost
                cost = self.cost_model(facility_emb, order_emb, context_t).item()

                # Compute distance
                distance = self._compute_distance(facility.location, order.customer_location)

                facility_analysis.append({
                    'facility_id': facility.facility_id,
                    'cost': cost,
                    'distance': distance,
                    'has_inventory': all(
                        facility.inventory.get(pid, 0) >= qty
                        for pid, qty in order.products
                    )
                })

                if cost < best_cost:
                    best_cost = cost
                    best_facility = facility.facility_id

        return best_facility, best_cost, {
            'analysis': facility_analysis,
            'order_id': order.order_id
        }

    def _extract_order_features(self, order: Order) -> np.ndarray:
        """Extract order features for embedding"""
        # Simplified - would include product details, location, urgency, etc.
        return np.random.randn(128)

    def _extract_facility_features(self, facility: Facility, order: Order) -> np.ndarray:
        """Extract facility features for embedding"""
        # Simplified - would include location, capacity, costs, inventory
        return np.random.randn(128)

    def _extract_context_features(self, facility: Facility, order: Order) -> np.ndarray:
        """Extract context features (distance, urgency, etc.)"""
        distance = self._compute_distance(facility.location, order.customer_location)
        return np.array([distance / 1000.0] + [0.0] * 9)  # Simplified

    def _compute_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Compute distance between locations"""
        # Simplified Euclidean distance
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

    def _check_facility_constraints(
        self,
        facility: Facility,
        order: Order,
        constraints: Dict[str, Any]
    ) -> bool:
        """Check if facility satisfies constraints"""
        # Check inventory availability
        for product_id, quantity in order.products:
            if facility.inventory.get(product_id, 0) < quantity:
                return False

        # Check capacity
        if constraints.get('max_utilization'):
            # Simplified capacity check
            pass

        # Check service level (distance)
        if constraints.get('max_distance'):
            distance = self._compute_distance(facility.location, order.customer_location)
            if distance > constraints['max_distance']:
                return False

        return True

# Example: Warehouse selection
def supply_chain_example():
    """
    Warehouse selection for e-commerce order fulfillment
    
    Scenario:
    - 3 warehouses: East Coast, Central, West Coast
    - Order from California
    - Product in stock at East (2000 mi) and West (50 mi)
    - Not in stock at Central
    
    Traditional: Ship from nearest warehouse with inventory (West)
    
    Embedding optimization:
    - Consider all factors: shipping cost, inventory levels, capacity, etc.
    - Learn that sometimes farther warehouse is cheaper (lower labor costs)
    - Account for urgency (overnight vs standard shipping)
    """

    print("=== Supply Chain Optimization ===")
    print("\nOrder: Mountain bike")
    print("Customer location: San Francisco, CA")
    print("Delivery: Standard (5-7 days)")

    print("\n--- Traditional Approach ---")
    print("Rule: Ship from nearest warehouse with inventory")
    print("\nWarehouses:")
    print("  East Coast (Philadelphia): In stock, 2,800 miles")
    print("  Central (Kansas City): OUT OF STOCK")
    print("  West Coast (Reno): In stock, 200 miles")
    print("\nDecision: Ship from West Coast")
    print("  Shipping cost: $15")
    print("  Labor cost: $8")
    print("  Total cost: $23")

    print("\n--- Embedding-Based Optimization ---")
    print("Consider all factors via learned embeddings:")
    print("\nEast Coast:")
    print("  Shipping: $45 (far)")
    print("  Labor: $5 (automated facility)")
    print("  Inventory pressure: Low (overstocked)")
    print("  Predicted total cost: $50")
    print("\nWest Coast:")
    print("  Shipping: $15 (close)")
    print("  Labor: $8 (manual facility)")
    print("  Inventory pressure: High (low stock)")
    print("  Predicted total cost: $23")
    print("\nDecision: Ship from West Coast")
    print("  → In this case, same as traditional")

    print("\n--- Overnight Delivery Scenario ---")
    print("Order: Same bike, but overnight delivery")
    print("\nTraditional:")
    print("  Ship from West Coast (closest)")
    print("  Air shipping: $85")
    print("  Total cost: $93")
    print("\nEmbedding optimization:")
    print("  East Coast has direct overnight flight to SF")
    print("  East Coast overnight: $65 (bulk shipping discount)")
    print("  East Coast total: $70")
    print("\nDecision: Ship from East Coast")
    print("  → Save $23 by using farther warehouse with better logistics")
    print("  → Embedding learned that East has SF overnight contract")

# Uncomment to run:
# supply_chain_example()
