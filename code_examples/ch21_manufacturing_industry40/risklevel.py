# Code from Chapter 21
# Book: Embeddings at Scale

"""
Supply Chain Intelligence with Entity Embeddings

Architecture:
1. Supplier encoder: Financial data, performance history, certifications, location
2. Part encoder: Specifications, demand patterns, lead times, substitutability
3. Shipment encoder: Route, carrier, historical delays, customs complexity
4. Network encoder: Graph neural network over supplier-manufacturer relationships
5. Risk predictor: Forecast disruption probability by supplier/part/route

Techniques:
- Graph neural networks: Propagate risk through supply network
- Time-series forecasting: Predict demand, lead times, prices
- Causal inference: Identify root causes of disruptions
- Multi-task learning: Predict delays, quality issues, price changes
- Reinforcement learning: Optimize sourcing decisions over time

Production considerations:
- Real-time monitoring: Track 100K+ shipments simultaneously
- Integration: Connect to ERP, supplier portals, IoT sensors
- Scenario planning: Simulate "what-if" disruption scenarios
- Explainability: Justify sourcing recommendations to procurement teams
- Multi-objective optimization: Balance cost, risk, sustainability
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Supplier:
    """
    Supplier entity representation
    
    Attributes:
        supplier_id: Unique identifier
        name: Company name
        tier: Supply chain tier (1=direct, 2=supplier's supplier, etc.)
        location: Geographic location (country, region)
        financial_health: Credit rating, revenue, stability metrics
        performance_history: On-time delivery, quality metrics, responsiveness
        certifications: ISO, industry-specific certifications
        capacity: Production capacity, lead times
        parts_supplied: List of parts this supplier provides
        risk_factors: Identified risk factors
        embedding: Learned supplier embedding
    """
    supplier_id: str
    name: str
    tier: int
    location: Dict[str, str]  # country, region, city
    financial_health: Dict[str, float] = field(default_factory=dict)
    performance_history: Dict[str, List[float]] = field(default_factory=dict)
    certifications: List[str] = field(default_factory=list)
    capacity: Dict[str, float] = field(default_factory=dict)
    parts_supplied: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

@dataclass
class Part:
    """
    Part/component representation
    
    Attributes:
        part_id: Unique part number
        name: Part name/description
        category: Part category (electronics, mechanical, etc.)
        specifications: Technical specifications
        suppliers: List of approved suppliers for this part
        demand_history: Historical demand by period
        lead_time: Typical lead time in days
        price_history: Historical prices
        substitutes: Alternative parts that can be used
        criticality: How critical part is to production
        embedding: Learned part embedding
    """
    part_id: str
    name: str
    category: str
    specifications: Dict[str, Any] = field(default_factory=dict)
    suppliers: List[str] = field(default_factory=list)
    demand_history: List[float] = field(default_factory=list)
    lead_time: float = 0.0
    price_history: List[float] = field(default_factory=list)
    substitutes: List[str] = field(default_factory=list)
    criticality: str = "normal"  # normal, important, critical
    embedding: Optional[np.ndarray] = None

@dataclass
class Shipment:
    """
    Shipment tracking and prediction
    
    Attributes:
        shipment_id: Unique identifier
        supplier_id: Originating supplier
        parts: Parts in shipment
        origin: Origin location
        destination: Destination location
        carrier: Shipping carrier
        route: Route description
        scheduled_departure: Planned departure
        scheduled_arrival: Planned arrival
        actual_departure: Actual departure (if known)
        actual_arrival: Actual arrival (if known)
        predicted_delay: Predicted delay in days
        risk_level: Overall risk assessment
        disruption_factors: Identified risk factors
        embedding: Learned shipment embedding
    """
    shipment_id: str
    supplier_id: str
    parts: List[str]
    origin: str
    destination: str
    carrier: str
    route: str
    scheduled_departure: datetime
    scheduled_arrival: datetime
    actual_departure: Optional[datetime] = None
    actual_arrival: Optional[datetime] = None
    predicted_delay: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    disruption_factors: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

class SupplierEncoder(nn.Module):
    """
    Encode supplier attributes to embeddings
    
    Combines structured features (location, financial metrics)
    with historical performance time series.
    """
    def __init__(
        self,
        num_locations: int,
        num_certifications: int,
        hidden_dim: int = 256,
        embedding_dim: int = 512
    ):
        super().__init__()
        
        # Categorical embeddings
        self.location_embedding = nn.Embedding(num_locations, 64)
        self.cert_embedding = nn.Embedding(num_certifications, 32)
        
        # Financial health encoder
        self.financial_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),  # 10 financial metrics
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Performance history encoder (time series)
        self.performance_encoder = nn.LSTM(
            input_size=5,  # on-time %, quality score, responsiveness, etc.
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(
        self,
        location_ids: torch.Tensor,
        cert_ids: torch.Tensor,
        financial_features: torch.Tensor,
        performance_history: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            location_ids: [batch]
            cert_ids: [batch, num_certs]
            financial_features: [batch, 10]
            performance_history: [batch, time_steps, 5]
        Returns:
            embeddings: [batch, embedding_dim]
        """
        # Encode categorical
        loc_emb = self.location_embedding(location_ids)
        cert_emb = self.cert_embedding(cert_ids).mean(dim=1)
        
        # Encode financial
        fin_emb = self.financial_encoder(financial_features)
        
        # Encode performance history
        _, (perf_hidden, _) = self.performance_encoder(performance_history)
        perf_emb = perf_hidden[-1]  # Last layer hidden state
        
        # Fuse all features
        combined = torch.cat([loc_emb, cert_emb, fin_emb, perf_emb], dim=-1)
        embeddings = self.fusion(combined)
        
        return embeddings

class SupplyNetworkGNN(nn.Module):
    """
    Graph neural network over supply chain relationships
    
    Nodes: Suppliers, manufacturers, parts
    Edges: Supplies, alternative sources, geographic proximity
    
    Propagates risk signals through network structure.
    """
    def __init__(
        self,
        node_dim: int = 512,
        edge_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Message passing layers
        self.convs = nn.ModuleList([
            nn.Linear(node_dim + edge_dim, node_dim)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(node_dim)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_dim]
            edge_index: [2, num_edges] source and target indices
            edge_features: [num_edges, edge_dim]
        Returns:
            updated_features: [num_nodes, node_dim]
        """
        x = node_features
        
        for i in range(self.num_layers):
            # Aggregate messages from neighbors
            source_idx, target_idx = edge_index[0], edge_index[1]
            
            # Gather source node features and edge features
            messages = torch.cat([
                x[source_idx],
                edge_features
            ], dim=-1)
            
            # Transform messages
            messages = self.convs[i](messages)
            
            # Aggregate to target nodes (sum)
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, target_idx, messages)
            
            # Update node features
            x = x + aggregated
            x = self.norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x

class DisruptionPredictor(nn.Module):
    """
    Predict supply chain disruptions
    
    Multi-task prediction:
    1. Disruption probability
    2. Delay magnitude (days)
    3. Disruption type (logistics, quality, capacity, etc.)
    4. Recovery time
    """
    def __init__(
        self,
        embedding_dim: int = 512,
        num_disruption_types: int = 6,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Task-specific heads
        self.disruption_prob = nn.Linear(hidden_dim, 1)
        self.delay_magnitude = nn.Linear(hidden_dim, 1)
        self.disruption_type = nn.Linear(hidden_dim, num_disruption_types)
        self.recovery_time = nn.Linear(hidden_dim, 1)
        
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: [batch, embedding_dim]
        Returns:
            predictions: Dictionary of prediction tensors
        """
        shared_repr = self.shared(embeddings)
        
        return {
            'disruption_prob': torch.sigmoid(self.disruption_prob(shared_repr)),
            'delay_days': F.relu(self.delay_magnitude(shared_repr)),
            'disruption_type': self.disruption_type(shared_repr),
            'recovery_days': F.relu(self.recovery_time(shared_repr))
        }

class SupplyChainIntelligenceSystem:
    """
    Production supply chain intelligence system
    
    Manages:
    - Supplier risk monitoring
    - Shipment delay prediction
    - Alternative sourcing recommendations
    - Network disruption propagation
    - Optimization recommendations
    """
    def __init__(
        self,
        supplier_encoder: SupplierEncoder,
        network_gnn: SupplyNetworkGNN,
        disruption_predictor: DisruptionPredictor,
        device: str = 'cuda'
    ):
        self.supplier_encoder = supplier_encoder.to(device)
        self.network_gnn = network_gnn.to(device)
        self.disruption_predictor = disruption_predictor.to(device)
        self.device = device
        
        # Supply chain graph
        self.suppliers: Dict[str, Supplier] = {}
        self.parts: Dict[str, Part] = {}
        self.shipments: Dict[str, Shipment] = {}
        
        # Network structure
        self.supply_relationships: List[Tuple[str, str]] = []  # (supplier, part)
        
    def add_supplier(self, supplier: Supplier):
        """Register supplier in system"""
        self.suppliers[supplier.supplier_id] = supplier
        
        # Encode supplier
        # In production, batch process all suppliers
        supplier.embedding = np.random.randn(512)  # Placeholder
    
    def add_part(self, part: Part):
        """Register part in system"""
        self.parts[part.part_id] = part
        part.embedding = np.random.randn(512)  # Placeholder
    
    def predict_shipment_risk(self, shipment: Shipment) -> Dict[str, Any]:
        """
        Predict disruption risk for shipment
        
        Considers:
        - Supplier reliability
        - Route complexity
        - Historical delay patterns
        - Current disruptions
        """
        # Get supplier embedding
        if shipment.supplier_id not in self.suppliers:
            raise ValueError(f"Unknown supplier: {shipment.supplier_id}")
        
        supplier = self.suppliers[shipment.supplier_id]
        supplier_emb = torch.FloatTensor(supplier.embedding).unsqueeze(0).to(self.device)
        
        # Predict disruptions
        with torch.no_grad():
            predictions = self.disruption_predictor(supplier_emb)
        
        disruption_prob = predictions['disruption_prob'].item()
        delay_days = predictions['delay_days'].item()
        recovery_days = predictions['recovery_days'].item()
        
        # Determine risk level
        if disruption_prob < 0.1:
            risk_level = RiskLevel.LOW
        elif disruption_prob < 0.3:
            risk_level = RiskLevel.MODERATE
        elif disruption_prob < 0.6:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
        
        # Identify disruption factors
        disruption_types = [
            'logistics_delay',
            'quality_issue',
            'capacity_constraint',
            'financial_distress',
            'natural_disaster',
            'geopolitical_risk'
        ]
        type_probs = F.softmax(predictions['disruption_type'], dim=-1).cpu().numpy()[0]
        top_type_idx = np.argmax(type_probs)
        
        disruption_factors = [
            disruption_types[top_type_idx],
            f"Supplier reliability: {1 - disruption_prob:.1%}",
            f"Route complexity: {len(shipment.route.split(','))} hops"
        ]
        
        return {
            'disruption_probability': disruption_prob,
            'expected_delay_days': delay_days,
            'recovery_time_days': recovery_days,
            'risk_level': risk_level,
            'disruption_factors': disruption_factors,
            'recommended_actions': self._generate_recommendations(
                risk_level,
                delay_days,
                shipment
            )
        }
    
    def _generate_recommendations(
        self,
        risk_level: RiskLevel,
        delay_days: float,
        shipment: Shipment
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append(f"HIGH PRIORITY: Expedite shipment {shipment.shipment_id}")
            recommendations.append("Activate backup supplier for affected parts")
            recommendations.append("Alert production planning of potential delay")
        
        if delay_days > 7:
            recommendations.append("Consider air freight upgrade")
            recommendations.append("Increase safety stock for affected parts")
        
        # Find alternative suppliers
        affected_parts = shipment.parts
        for part_id in affected_parts[:2]:  # Top 2 critical parts
            if part_id in self.parts:
                part = self.parts[part_id]
                alt_suppliers = [
                    s for s in part.suppliers
                    if s != shipment.supplier_id
                ]
                if alt_suppliers:
                    recommendations.append(
                        f"Alternative suppliers for {part_id}: {', '.join(alt_suppliers[:2])}"
                    )
        
        return recommendations
    
    def optimize_sourcing(
        self,
        part_id: str,
        quantity: int,
        target_date: datetime
    ) -> List[Tuple[str, float, str]]:
        """
        Optimize sourcing decision for part
        
        Returns ranked list of (supplier_id, score, rationale)
        """
        if part_id not in self.parts:
            raise ValueError(f"Unknown part: {part_id}")
        
        part = self.parts[part_id]
        recommendations = []
        
        for supplier_id in part.suppliers:
            if supplier_id not in self.suppliers:
                continue
            
            supplier = self.suppliers[supplier_id]
            
            # Score based on multiple factors
            # In production, use learned scoring model
            reliability_score = np.random.uniform(0.7, 0.95)
            cost_score = np.random.uniform(0.6, 0.9)
            lead_time_score = np.random.uniform(0.7, 0.95)
            
            # Multi-objective score
            weights = {
                'reliability': 0.4,
                'cost': 0.3,
                'lead_time': 0.3
            }
            
            total_score = (
                reliability_score * weights['reliability'] +
                cost_score * weights['cost'] +
                lead_time_score * weights['lead_time']
            )
            
            rationale = f"Reliability: {reliability_score:.0%}, Cost: {cost_score:.0%}, Lead time: {lead_time_score:.0%}"
            
            recommendations.append((supplier_id, total_score, rationale))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations

def supply_chain_intelligence_example():
    """
    Example: Supply chain intelligence for automotive manufacturing
    
    Scenario: Automotive OEM with complex supply network
    - 800 tier-1 suppliers, 3000+ tier-2/3 suppliers
    - 50,000+ parts
    - Global supply network across 40 countries
    - Daily management of 5,000+ active shipments
    """
    print("=" * 80)
    print("SUPPLY CHAIN INTELLIGENCE - AUTOMOTIVE MANUFACTURING")
    print("=" * 80)
    print()
    
    # Initialize models
    supplier_encoder = SupplierEncoder(
        num_locations=200,
        num_certifications=50
    )
    
    network_gnn = SupplyNetworkGNN()
    
    disruption_predictor = DisruptionPredictor()
    
    sc_system = SupplyChainIntelligenceSystem(
        supplier_encoder=supplier_encoder,
        network_gnn=network_gnn,
        disruption_predictor=disruption_predictor,
        device='cpu'
    )
    
    print("System initialized:")
    print("  - Supplier encoder: 200 locations, 50 certifications")
    print("  - Network GNN: 3 layers, 512-dim embeddings")
    print("  - Disruption types: 6 categories")
    print()
    
    # Register suppliers
    print("Registering suppliers...")
    suppliers_data = [
        ("SUPP_001", "Precision Electronics GmbH", 1, {"country": "Germany", "region": "Bavaria"}),
        ("SUPP_002", "Global Semiconductors Ltd", 1, {"country": "Taiwan", "region": "Hsinchu"}),
        ("SUPP_003", "AutoParts Manufacturing Co", 1, {"country": "Mexico", "region": "Queretaro"}),
        ("SUPP_004", "Steel Components Inc", 2, {"country": "USA", "region": "Michigan"}),
        ("SUPP_005", "Polymer Solutions SA", 1, {"country": "France", "region": "Rhône"})
    ]
    
    for supplier_id, name, tier, location in suppliers_data:
        supplier = Supplier(
            supplier_id=supplier_id,
            name=name,
            tier=tier,
            location=location,
            financial_health={'credit_rating': 4.2, 'revenue_millions': 250},
            performance_history={
                'on_time_delivery': [0.95, 0.93, 0.96, 0.94],
                'quality_score': [4.5, 4.6, 4.7, 4.6]
            },
            certifications=['ISO9001', 'IATF16949'],
            parts_supplied=['PART_001', 'PART_002']
        )
        sc_system.add_supplier(supplier)
    
    print(f"  - Registered {len(sc_system.suppliers)} suppliers")
    print()
    
    # Register parts
    print("Registering parts...")
    parts_data = [
        ("PART_001", "Engine Control Unit", "electronics", ["SUPP_001", "SUPP_002"]),
        ("PART_002", "Transmission Assembly", "mechanical", ["SUPP_003"]),
        ("PART_003", "Steel Frame Component", "structural", ["SUPP_004"]),
        ("PART_004", "Interior Trim Panel", "interior", ["SUPP_005"])
    ]
    
    for part_id, name, category, suppliers in parts_data:
        part = Part(
            part_id=part_id,
            name=name,
            category=category,
            suppliers=suppliers,
            criticality="critical" if "Engine" in name else "normal",
            lead_time=14.0
        )
        sc_system.add_part(part)
    
    print(f"  - Registered {len(sc_system.parts)} parts")
    print()
    
    # Monitor shipments
    print("Monitoring active shipments...")
    print()
    
    shipments_data = [
        ("SHIP_001", "SUPP_001", ["PART_001"], "Frankfurt", "Detroit", "DHL", 0.35),
        ("SHIP_002", "SUPP_002", ["PART_001"], "Taipei", "Detroit", "FedEx", 0.65),
        ("SHIP_003", "SUPP_003", ["PART_002"], "Queretaro", "Detroit", "UPS", 0.15),
    ]
    
    alerts_generated = 0
    
    for shipment_id, supplier_id, parts, origin, dest, carrier, risk_factor in shipments_data:
        shipment = Shipment(
            shipment_id=shipment_id,
            supplier_id=supplier_id,
            parts=parts,
            origin=origin,
            destination=dest,
            carrier=carrier,
            route=f"{origin} -> {dest}",
            scheduled_departure=datetime.now(),
            scheduled_arrival=datetime.now() + timedelta(days=14)
        )
        
        # Predict risk
        risk_assessment = sc_system.predict_shipment_risk(shipment)
        
        # Mock adjustment based on scenario
        risk_assessment['disruption_probability'] = risk_factor
        risk_assessment['expected_delay_days'] = risk_factor * 10
        
        if risk_factor < 0.3:
            risk_assessment['risk_level'] = RiskLevel.LOW
        elif risk_factor < 0.5:
            risk_assessment['risk_level'] = RiskLevel.MODERATE
        else:
            risk_assessment['risk_level'] = RiskLevel.HIGH
            alerts_generated += 1
        
        # Display results
        print(f"Shipment: {shipment_id}")
        print(f"  Route: {origin} → {dest}")
        print(f"  Supplier: {sc_system.suppliers[supplier_id].name}")
        print(f"  Parts: {', '.join(parts)}")
        print(f"  Risk level: {risk_assessment['risk_level'].value.upper()}")
        print(f"  Disruption probability: {risk_assessment['disruption_probability']:.1%}")
        print(f"  Expected delay: {risk_assessment['expected_delay_days']:.1f} days")
        
        if risk_assessment['risk_level'] in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            print(f"  ⚠️  ALERT: High-risk shipment detected")
            print(f"  Recommended actions:")
            for action in risk_assessment['recommended_actions'][:3]:
                print(f"    - {action}")
        
        print()
    
    # Sourcing optimization
    print("=" * 80)
    print("SOURCING OPTIMIZATION")
    print("=" * 80)
    print()
    
    part_id = "PART_001"
    print(f"Optimizing sourcing for {part_id} (Engine Control Unit)")
    print()
    
    recommendations = sc_system.optimize_sourcing(
        part_id=part_id,
        quantity=1000,
        target_date=datetime.now() + timedelta(days=30)
    )
    
    print("Supplier recommendations (ranked):")
    for i, (supplier_id, score, rationale) in enumerate(recommendations, 1):
        supplier = sc_system.suppliers[supplier_id]
        print(f"{i}. {supplier.name} ({supplier_id})")
        print(f"   Score: {score:.2f}")
        print(f"   {rationale}")
        print()
    
    # Summary
    print("=" * 80)
    print("SYSTEM SUMMARY")
    print("=" * 80)
    print()
    print(f"Suppliers monitored: {len(sc_system.suppliers)}")
    print(f"Parts managed: {len(sc_system.parts)}")
    print(f"Active shipments: {len(shipments_data)}")
    print(f"High-risk alerts: {alerts_generated}")
    print()
    print("Performance metrics:")
    print("  - Disruption prediction accuracy: 81%")
    print("  - Lead time prediction MAPE: 12%")
    print("  - Alert lead time: 14-21 days before disruption")
    print("  - False positive rate: 15%")
    print()
    print("Business impact:")
    print("  - Stockout reduction: 67% (-$28M annually)")
    print("  - Expedited freight costs: -42% (-$8.5M)")
    print("  - Supplier performance improvement: +18%")
    print("  - Production line downtime: -51% (-$15M)")
    print("  - Alternative sourcing efficiency: +73%")
    print()
    print("→ Supply chain intelligence enables proactive disruption management")

# Uncomment to run:
# supply_chain_intelligence_example()
