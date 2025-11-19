# Code from Chapter 18
# Book: Embeddings at Scale

"""
Customer Behavior Analysis with Embeddings

Architecture:
1. Customer encoder: Transaction history → customer embedding
2. Lifecycle model: Map embeddings to lifecycle stages
3. Propensity models: Predict churn, cross-sell, upsell
4. Personalization engine: Tailor products/offers to embeddings
5. Clustering: Discover natural customer segments

Use cases:
- Churn prediction: Identify at-risk customers
- Cross-sell: Recommend relevant products
- Personalization: Customize offerings, pricing, messaging
- Lifetime value: Predict long-term customer value
- Acquisition: Find lookalike audiences

Production considerations:
- Privacy: Comply with data protection regulations
- Real-time: Update embeddings as new transactions arrive
- Explainability: Surface why recommendations made
- Fairness: Avoid discriminatory segments
"""

@dataclass
class Customer:
    """
    Customer profile
    
    Attributes:
        customer_id: Unique identifier
        demographics: Age, location, income, etc.
        products: Currently held products
        transaction_history: Past transactions
        interactions: Service calls, branch visits, etc.
        lifecycle_stage: Acquisition, growth, mature, at_risk, churned
        embedding: Learned customer embedding
    """
    customer_id: str
    demographics: Dict[str, Any]
    products: List[str]
    transaction_history: List[Dict[str, Any]]
    interactions: List[Dict[str, Any]]
    lifecycle_stage: Optional[str] = None
    embedding: Optional[np.ndarray] = None

class CustomerEncoder(nn.Module):
    """
    Encode customers from behavioral data
    
    Architecture:
    - Transaction encoder: LSTM over transaction history
    - Product encoder: Embeddings of held products
    - Interaction encoder: Service interaction patterns
    - Demographic encoder: Basic demographic features
    - Fusion: Combine all modalities
    
    Training:
    - Churn prediction: Embedding predicts churn probability
    - Product adoption: Predict next product customer adopts
    - Contrastive: High-LTV customers close together
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        num_products: int = 50
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Transaction encoder
        self.transaction_encoder = nn.LSTM(
            input_size=20,  # transaction features
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Product embeddings
        self.product_embedding = nn.Embedding(num_products, 32)
        
        # Interaction encoder
        self.interaction_encoder = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(160, 128),  # 64 + 32 + 64
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )
        
    def forward(
        self,
        transaction_history: torch.Tensor,
        product_ids: torch.Tensor,
        interaction_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode customers
        
        Args:
            transaction_history: Transaction history (batch_size, seq_len, 20)
            product_ids: Held product IDs (batch_size, max_products)
            interaction_features: Interaction features (batch_size, 30)
        
        Returns:
            Customer embeddings (batch_size, embedding_dim)
        """
        # Encode transaction history
        _, (transaction_hidden, _) = self.transaction_encoder(transaction_history)
        transaction_emb = transaction_hidden[-1]
        
        # Encode products (average of held products)
        product_embs = self.product_embedding(product_ids)
        product_emb = product_embs.mean(dim=1)
        
        # Encode interactions
        interaction_emb = self.interaction_encoder(interaction_features)
        
        # Fuse
        combined = torch.cat([transaction_emb, product_emb, interaction_emb], dim=1)
        customer_emb = self.fusion(combined)
        
        # Normalize
        customer_emb = F.normalize(customer_emb, p=2, dim=1)
        
        return customer_emb

# Example: Customer churn prevention
def churn_prevention_example():
    """
    Embedding-based churn prediction and prevention
    
    Demonstrates:
    1. Learning customer embeddings from behavior
    2. Identifying at-risk customers
    3. Personalized retention interventions
    """
    
    print("=== Customer Churn Prevention System ===")
    print("\nObjective: Identify and retain at-risk customers")
    print("Approach: Learn embeddings capturing lifecycle stage")
    print("         Detect drift toward churn cluster")
    
    print("\n--- Customer 1: Healthy ---")
    print("Customer ID: C001")
    print("Products: Checking, savings, credit card")
    print("Recent activity:")
    print("  - Regular direct deposits")
    print("  - Active credit card usage")
    print("  - Mobile app usage: 15x per month")
    print("  - Customer service: No recent calls")
    
    print("\nEmbedding analysis:")
    print("  Cluster: Engaged customers")
    print("  Distance from churn cluster: 0.89 (far)")
    print("  Lifecycle stage: Growth")
    
    print("\nAssessment:")
    print("  Churn probability (90 days): 3%")
    print("  Lifetime value: $8,500")
    print("  Action: No intervention needed")
    print("  Opportunity: Cross-sell mortgage")
    
    print("\n--- Customer 2: Early At-Risk Indicators ---")
    print("Customer ID: C002")
    print("Products: Checking, savings")
    print("Tenure: 3 years")
    print("Recent changes:")
    print("  - Balance declining (was $5K, now $1.2K)")
    print("  - Mobile app usage dropped (was 20x, now 5x per month)")
    print("  - No credit card usage last 30 days")
    print("  - Customer service: Called twice about fees")
    
    print("\nEmbedding analysis:")
    print("  Current cluster: Engaged customers")
    print("  Drift: Moving toward disengaged cluster")
    print("  Distance from churn cluster: 0.45 (closing)")
    print("  Similar to: Past churners 60 days before churn")
    
    print("\nAssessment:")
    print("  Churn probability (90 days): 35%")
    print("  Lifetime value at risk: $6,200")
    print("  Action: Proactive intervention")
    print("  ")
    print("  Recommended intervention:")
    print("    1. Fee waiver offer ($100 value)")
    print("    2. Personal outreach from relationship manager")
    print("    3. Survey about service issues")
    print("    4. Highlight unused benefits (free ATMs, overdraft protection)")
    print("\n→ Caught early, high retention probability")
    
    print("\n--- Customer 3: Imminent Churn ---")
    print("Customer ID: C003")
    print("Products: Checking only")
    print("Tenure: 8 months")
    print("Recent behavior:")
    print("  - Balance near zero")
    print("  - No transactions last 45 days")
    print("  - App uninstalled")
    print("  - Customer service: Canceled credit card last month")
    print("  - External signal: Opened account at competitor")
    
    print("\nEmbedding analysis:")
    print("  Current cluster: Churned/inactive customers")
    print("  Distance from churn cluster: 0.05 (inside)")
    print("  Embedding nearly identical to: Customers who churned")
    
    print("\nAssessment:")
    print("  Churn probability (90 days): 92%")
    print("  Likely already churned (inactive 45 days)")
    print("  Action: Win-back campaign")
    print("  ")
    print("  Recommended intervention:")
    print("    1. Account reactivation bonus ($200)")
    print("    2. Premium product upgrade offer")
    print("    3. Apology + service improvement message")
    print("  ")
    print("  ROI consideration: $200 bonus vs $2,500 acquisition cost")
    print("  → Win-back cheaper than new acquisition")
    
    print("\n--- System Performance ---")
    print("Customers monitored: 1.5M")
    print("At-risk identified: 45,000 (3%)")
    print("Interventions: 45,000")
    print("Retention rate: 68% (vs 40% without intervention)")
    print("Additional customers retained: 12,600")
    print("Lifetime value protected: $78M")
    print("Campaign cost: $4.5M")
    print("ROI: 17x")
    print("\n→ Proactive churn prevention highly profitable")

# Uncomment to run:
# churn_prevention_example()
