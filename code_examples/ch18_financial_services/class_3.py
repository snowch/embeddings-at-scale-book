from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Code from Chapter 18
# Book: Embeddings at Scale

"""
Market Sentiment Analysis

Architecture:
1. Text encoder: Embed news articles, social media posts, earnings calls
2. Sentiment classifier: Map embeddings to sentiment (bullish, bearish, neutral)
3. Aspect extraction: Identify mentioned entities (stocks, sectors)
4. Signal generator: Convert sentiment to trading signals
5. Aggregator: Combine sentiment across sources

Use cases:
- News-driven trading: React to breaking news before prices move
- Earnings call analysis: Sentiment from management tone, not just numbers
- Social sentiment: Aggregate retail investor mood
- Analyst sentiment: Encode analyst reports for consensus shifts

Production considerations:
- Latency: <1s for news, <1min for social media aggregation
- Noise filtering: Remove spam, bots, duplicate content
- Entity disambiguation: "Apple" (company) vs apple (fruit)
- Temporal decay: Recent sentiment more important than old
"""

@dataclass
class SentimentSignal:
    """
    Sentiment-derived trading signal
    
    Attributes:
        ticker: Security ticker
        timestamp: When sentiment measured
        sentiment_score: Aggregated sentiment (-1 to +1)
        confidence: Signal confidence (0-1)
        source_breakdown: Sentiment by source (news, social, analyst)
        aspects: Aspect-specific sentiment (management, products, financials)
        volume: Number of mentions
        predicted_impact: Expected price impact
    """
    ticker: str
    timestamp: float
    sentiment_score: float
    confidence: float
    source_breakdown: Dict[str, float]
    aspects: Dict[str, float]
    volume: int
    predicted_impact: float

class FinancialTextEncoder(nn.Module):
    """
    Encode financial text to embeddings
    
    Fine-tuned on financial text + market outcomes:
    - News articles → future returns
    - Earnings call transcripts → post-earnings drift
    - Analyst reports → price target accuracy
    
    Learns: Positive sentiment words for finance (beat, exceed, strong)
            Negative sentiment words (miss, weak, concern)
            Hedging language (may, could, possible)
            Certainty language (will, definitely, commit)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        pretrained_model: str = "finbert"  # Financial BERT
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Use pre-trained financial BERT
        # In practice, load from transformers library
        self.bert_dim = 768

        # Projection to target dimension
        self.projection = nn.Sequential(
            nn.Linear(self.bert_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode financial text
        
        Args:
            text_embeddings: BERT embeddings (batch_size, 768)
        
        Returns:
            Financial text embeddings (batch_size, embedding_dim)
        """
        # Project BERT embeddings
        text_emb = self.projection(text_embeddings)

        # Normalize
        text_emb = F.normalize(text_emb, p=2, dim=1)

        return text_emb

class SentimentClassifier(nn.Module):
    """
    Classify sentiment from text embeddings
    
    Outputs:
    - Sentiment score (-1 to +1): Bearish to bullish
    - Confidence (0 to 1): How confident the model is
    - Aspects: Sentiment toward different aspects (management, products, etc.)
    """

    def __init__(self, embedding_dim: int = 256, num_aspects: int = 5):
        super().__init__()

        # Overall sentiment
        self.sentiment_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # sentiment, confidence
        )

        # Aspect-specific sentiment
        self.aspect_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_aspects)
        )

    def forward(
        self,
        text_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Classify sentiment
        
        Args:
            text_emb: Text embeddings (batch_size, embedding_dim)
        
        Returns:
            Tuple of (sentiment_score, confidence, aspect_sentiment)
        """
        # Overall sentiment
        overall = self.sentiment_head(text_emb)
        sentiment_score = torch.tanh(overall[:, 0])  # -1 to +1
        confidence = torch.sigmoid(overall[:, 1])  # 0 to 1

        # Aspect sentiment
        aspect_sentiment = torch.tanh(self.aspect_head(text_emb))  # -1 to +1

        return sentiment_score, confidence, aspect_sentiment

# Example: News-driven trading
def sentiment_trading_example():
    """
    News sentiment → trading signals
    
    Demonstrates:
    1. Processing breaking news
    2. Extracting sentiment
    3. Generating trading signals
    4. Timing and execution
    """

    print("=== Market Sentiment Analysis System ===")
    print("\nObjective: Extract trading signals from market sentiment")
    print("Sources: News, social media, earnings calls, analyst reports")

    print("\n--- Event 1: Positive Earnings Surprise ---")
    print("Stock: TECH_CO")
    print("Event: Q3 earnings report released")
    print("Time: After market close")
    print("\nNews headline: ")
    print("  'TECH_CO beats earnings expectations, raises guidance'")
    print("\nEarnings call excerpt:")
    print("  'We're extremely pleased with our performance this quarter.")
    print("   Revenue grew 25% year-over-year, driven by strong demand")
    print("   for our cloud products. We're raising full-year guidance")
    print("   and remain confident in our market position.'")

    print("\nSentiment analysis:")
    print("  Overall sentiment: +0.82 (very bullish)")
    print("  Confidence: 0.91")
    print("  Aspect breakdown:")
    print("    Revenue: +0.95 (very positive)")
    print("    Profitability: +0.75 (positive)")
    print("    Guidance: +0.88 (very positive)")
    print("    Management tone: +0.72 (positive, confident)")
    print("  Volume: 150 news articles, 5K social media mentions")

    print("\nTrading signal:")
    print("  Direction: LONG")
    print("  Predicted impact: +4.5% at open tomorrow")
    print("  Confidence: 0.88")
    print("  Timing: Buy at open, hold through day 1")
    print("  Position size: 3% of portfolio")
    print("  ")
    print("  Rationale:")
    print("    ✓ Strong earnings beat")
    print("    ✓ Guidance raise (forward-looking)")
    print("    ✓ Positive management tone")
    print("    ✓ Broad positive news coverage")
    print("\n→ High-confidence bullish signal")

    print("\n--- Event 2: Mixed News ---")
    print("Stock: PHARMA_CO")
    print("Event: Drug trial results announced")
    print("\nNews headline:")
    print("  'PHARMA_CO drug shows efficacy but safety concerns emerge'")

    print("\nSentiment analysis:")
    print("  Overall sentiment: +0.15 (slightly bullish)")
    print("  Confidence: 0.45 (uncertain)")
    print("  Aspect breakdown:")
    print("    Efficacy: +0.75 (positive)")
    print("    Safety: -0.55 (concerning)")
    print("    Regulatory: -0.30 (potential delays)")
    print("  Volume: 80 news articles")
    print("  Disagreement: High variance across sources")

    print("\nTrading signal:")
    print("  Direction: HOLD / WAIT")
    print("  Predicted impact: Unclear (-2% to +3%)")
    print("  Confidence: 0.42 (low)")
    print("  ")
    print("  Rationale:")
    print("    ✓ Positive efficacy data")
    print("    ✗ Safety concerns (regulatory risk)")
    print("    ! High disagreement among analysts")
    print("    ! Need more clarity before trading")
    print("\n→ No trade due to uncertainty")

    print("\n--- Event 3: Social Media Frenzy (Caution) ---")
    print("Stock: MEME_STOCK")
    print("Event: Viral social media attention")
    print("\nSocial media activity:")
    print("  Mentions: 50K tweets in 1 hour (sudden spike)")
    print("  Sentiment: +0.92 (extremely bullish)")
    print("  Common phrases: 'to the moon', 'diamond hands', 'shorts get squeezed'")

    print("\nSentiment analysis:")
    print("  Overall sentiment: +0.92 (very bullish)")
    print("  BUT: Multiple red flags")
    print("    ⚠ Bot activity detected: 35% of mentions")
    print("    ⚠ Coordinated timing: Suspicious synchronization")
    print("    ⚠ No fundamental news to justify sentiment")
    print("    ⚠ Historical pattern: Similar to past pump-and-dumps")

    print("\nTrading signal:")
    print("  Direction: AVOID / SHORT (cautiously)")
    print("  Rationale:")
    print("    ✗ Artificial sentiment (bots, coordination)")
    print("    ✗ No fundamental support")
    print("    ✗ High crash risk after initial spike")
    print("  ")
    print("  Risk management:")
    print("    - If shorting: Small position, tight stop-loss")
    print("    - Watch for short squeeze risk")
    print("\n→ Likely manipulation, avoid or counter-trade carefully")

    print("\n--- System Performance ---")
    print("News articles processed: 10K per day")
    print("Social media posts: 1M per day")
    print("Earnings calls: 50-100 per day (earnings season)")
    print("Signals generated: 150 per day")
    print("Traded signals: 30 per day (high confidence only)")
    print("Win rate: 64%")
    print("Average return per trade: 1.8%")
    print("Sharpe ratio: 2.1")
    print("\n→ Sentiment provides measurable alpha")

# Uncomment to run:
# sentiment_trading_example()
