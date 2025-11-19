# Code from Chapter 18
# Book: Embeddings at Scale

"""
Trading Signal Generation with Embeddings

Architecture:
1. Security encoder: Maps securities to embeddings
2. Market encoder: Captures market regime (bull, bear, volatile)
3. News encoder: Extracts sentiment and topics from financial news
4. Alternative data encoders: Satellite imagery, web traffic, etc.
5. Signal generator: Predicts future returns from embeddings

Techniques:
- Time series embeddings: LSTM/Transformer over price history
- Cross-sectional learning: Securities with similar fundamentals behave similarly
- Graph embeddings: Capture supply chain, sector, geographic relationships
- Multi-modal fusion: Combine price, news, fundamentals, alternative data

Production considerations:
- Low latency: <10ms for real-time trading
- Interpretability: Explain signal sources for risk management
- Risk constraints: Sector limits, position sizing, stop losses
- Transaction costs: Model slippage, commissions, market impact
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Security:
    """
    Security (stock, bond, commodity, etc.)
    
    Attributes:
        ticker: Symbol (AAPL, TSLA, etc.)
        name: Company name
        sector: Industry sector
        market_cap: Market capitalization
        price_history: Historical prices (open, high, low, close, volume)
        fundamentals: Financial metrics (revenue, earnings, debt, etc.)
        news: Recent news articles
        alternative_data: Non-traditional data (web traffic, sentiment, etc.)
    """
    ticker: str
    name: str
    sector: str
    market_cap: float
    price_history: Optional[np.ndarray] = None
    fundamentals: Optional[Dict[str, float]] = None
    news: Optional[List[str]] = None
    alternative_data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.fundamentals is None:
            self.fundamentals = {}
        if self.news is None:
            self.news = []
        if self.alternative_data is None:
            self.alternative_data = {}

@dataclass
class TradingSignal:
    """
    Trading signal output
    
    Attributes:
        ticker: Security ticker
        timestamp: Signal generation time
        predicted_return: Expected return (next day, week, month)
        confidence: Signal confidence (0-1)
        factors: Contributing factors to signal
        risk_score: Risk assessment (0-1)
        position_size: Recommended position size
        explanation: Human-readable explanation
    """
    ticker: str
    timestamp: float
    predicted_return: float
    confidence: float
    factors: Dict[str, float]
    risk_score: float
    position_size: float
    explanation: str

class SecurityEncoder(nn.Module):
    """
    Encode securities to embeddings
    
    Architecture:
    - Price encoder: LSTM over historical prices
    - Fundamental encoder: MLP over financial metrics
    - News encoder: Transformer over recent news
    - Alternative data encoder: Custom encoders per data type
    - Fusion: Attention-based combination of all modalities
    
    Training:
    - Return prediction: Embedding predicts future returns
    - Contrastive: Securities in same sector closer
    - Triplet: High-correlation securities closer than low-correlation
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        price_lookback: int = 60,  # 60 days
        num_fundamental_features: int = 50
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.price_lookback = price_lookback

        # Price encoder (LSTM over OHLCV)
        self.price_encoder = nn.LSTM(
            input_size=5,  # open, high, low, close, volume
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Fundamental encoder
        self.fundamental_encoder = nn.Sequential(
            nn.Linear(num_fundamental_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )

    def forward(
        self,
        price_history: torch.Tensor,
        fundamentals: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode securities
        
        Args:
            price_history: Price history (batch_size, seq_len, 5)
            fundamentals: Fundamental features (batch_size, num_features)
        
        Returns:
            Security embeddings (batch_size, embedding_dim)
        """
        # Encode price history
        _, (price_hidden, _) = self.price_encoder(price_history)
        price_emb = price_hidden[-1]  # Last layer hidden state

        # Encode fundamentals
        fundamental_emb = self.fundamental_encoder(fundamentals)

        # Fuse
        combined = torch.cat([price_emb, fundamental_emb], dim=1)
        security_emb = self.fusion(combined)

        # Normalize
        security_emb = F.normalize(security_emb, p=2, dim=1)

        return security_emb

class MarketRegimeEncoder(nn.Module):
    """
    Encode market regime (bull, bear, volatile, calm)
    
    Captures macro conditions affecting all securities:
    - VIX level (volatility)
    - Interest rates
    - Credit spreads
    - Market breadth
    - Sector rotation
    
    Used to condition trading signals on market state.
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Market indicators encoder
        self.encoder = nn.Sequential(
            nn.Linear(20, 64),  # 20 market indicators
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, market_indicators: torch.Tensor) -> torch.Tensor:
        """
        Encode market regime
        
        Args:
            market_indicators: Market indicators (batch_size, 20)
        
        Returns:
            Market regime embeddings (batch_size, embedding_dim)
        """
        regime_emb = self.encoder(market_indicators)
        regime_emb = F.normalize(regime_emb, p=2, dim=1)
        return regime_emb

class TradingSignalGenerator(nn.Module):
    """
    Generate trading signals from security and market embeddings
    
    Predicts future returns conditioned on:
    - Security embedding (intrinsic characteristics)
    - Market regime embedding (macro environment)
    - Recent momentum (short-term price action)
    - Cross-sectional position (relative to sector/market)
    
    Outputs:
    - Expected return (alpha)
    - Confidence (signal strength)
    - Risk score (downside risk)
    """

    def __init__(
        self,
        security_dim: int = 256,
        regime_dim: int = 64,
        hidden_dim: int = 256
    ):
        super().__init__()

        # Signal generation network
        self.signal_network = nn.Sequential(
            nn.Linear(security_dim + regime_dim + 10, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 3)  # return, confidence, risk
        )

    def forward(
        self,
        security_emb: torch.Tensor,
        regime_emb: torch.Tensor,
        momentum_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate trading signals
        
        Args:
            security_emb: Security embeddings (batch_size, security_dim)
            regime_emb: Market regime embeddings (batch_size, regime_dim)
            momentum_features: Recent momentum (batch_size, 10)
        
        Returns:
            Tuple of (predicted_return, confidence, risk_score)
        """
        # Combine all inputs
        combined = torch.cat([security_emb, regime_emb, momentum_features], dim=1)

        # Generate signals
        outputs = self.signal_network(combined)

        # Split outputs
        predicted_return = outputs[:, 0]
        confidence = torch.sigmoid(outputs[:, 1])  # 0-1
        risk_score = torch.sigmoid(outputs[:, 2])  # 0-1

        return predicted_return, confidence, risk_score

# Example: End-to-end trading signal system
def trading_signal_example():
    """
    Complete trading signal generation pipeline
    
    Demonstrates:
    1. Encoding securities from price history and fundamentals
    2. Encoding market regime
    3. Generating trading signals
    4. Position sizing and risk management
    """

    print("=== Trading Signal Generation System ===")
    print("\nObjective: Generate alpha by identifying mispriced securities")
    print("Approach: Learn security embeddings from multi-modal data")
    print("         Predict future returns in embedding space")

    # Initialize components
    security_encoder = SecurityEncoder(embedding_dim=256)
    regime_encoder = MarketRegimeEncoder(embedding_dim=64)
    signal_generator = TradingSignalGenerator(security_dim=256, regime_dim=64)

    print("\n--- Example 1: Tech Growth Stock ---")
    print("Ticker: GROWTH_TECH")
    print("Sector: Technology")
    print("Recent Performance: +15% vs market +5%")
    print("Fundamentals: High revenue growth, negative earnings")
    print("News: New product launch, positive analyst coverage")
    print("Market Regime: Bull market, low volatility")

    # Simulate encoding
    print("\nSecurity Embedding Analysis:")
    print("  Similar to: Other high-growth tech stocks")
    print("  Cluster: Growth momentum cluster")
    print("  Distance from value stocks: 0.85 (far)")

    print("\nSignal:")
    print("  Predicted return (1 month): +8.5%")
    print("  Confidence: 0.72")
    print("  Risk score: 0.68 (high volatility)")
    print("  Position size: 2% of portfolio")
    print("  Explanation: Strong momentum, positive news sentiment")
    print("               But elevated risk due to negative earnings")

    print("\n--- Example 2: Value Stock Opportunity ---")
    print("Ticker: VALUE_IND")
    print("Sector: Industrials")
    print("Recent Performance: -10% vs market +5%")
    print("Fundamentals: Low P/E, high dividend yield, strong balance sheet")
    print("News: Temporary supply chain issues (resolved)")
    print("Market Regime: Bull market, low volatility")

    print("\nSecurity Embedding Analysis:")
    print("  Similar to: Other undervalued industrials")
    print("  Cluster: Value recovery cluster")
    print("  Recent shift: Moving toward growth cluster")

    print("\nSignal:")
    print("  Predicted return (1 month): +12.3%")
    print("  Confidence: 0.85")
    print("  Risk score: 0.35 (low volatility)")
    print("  Position size: 4% of portfolio")
    print("  Explanation: Temporary selloff created value opportunity")
    print("               Fundamentals strong, supply chain issues resolved")
    print("               Similar stocks historically recovered quickly")

    print("\n--- Example 3: Avoid Signal ---")
    print("Ticker: BUBBLE_STOCK")
    print("Sector: Technology")
    print("Recent Performance: +150% in 3 months")
    print("Fundamentals: No revenue, massive valuation")
    print("News: Heavy retail investor interest, social media hype")
    print("Market Regime: Bull market, increasing volatility")

    print("\nSecurity Embedding Analysis:")
    print("  Similar to: Past bubble stocks (2000, 2021)")
    print("  Cluster: Speculative bubble cluster")
    print("  Warning: High distance from fundamental value cluster")

    print("\nSignal:")
    print("  Predicted return (1 month): -15.2%")
    print("  Confidence: 0.68")
    print("  Risk score: 0.92 (extreme risk)")
    print("  Position size: 0% (AVOID)")
    print("  Explanation: Embedding similar to past bubble stocks")
    print("               Price disconnected from fundamentals")
    print("               High crash risk when sentiment reverses")

    print("\n--- Portfolio Construction ---")
    print("Strategy: Long-short equity")
    print("Long positions:")
    print("  VALUE_IND: 4% (high confidence, low risk)")
    print("  GROWTH_TECH: 2% (medium confidence, high risk)")
    print("  ... 10 other long positions")
    print("Short positions:")
    print("  BUBBLE_STOCK: -2% (betting on decline)")
    print("  ... 5 other short positions")
    print("Total exposure: 50% long, 20% short, 30% cash")
    print("Expected return: 12% annualized")
    print("Sharpe ratio: 1.8")

# Uncomment to run:
# trading_signal_example()
