# Code from Chapter 02
# Book: Embeddings at Scale


class BuildVsBuyDecisionFramework:
    """Framework for build vs. buy decisions"""

    def evaluate_decision(self, context):
        """
        Evaluate whether to build or buy

        context: {
            'scale': 1_000_000_000,  # num embeddings
            'qps': 10_000,
            'use_case_criticality': 'high',  # low, medium, high
            'competitive_differentiation': 'high',  # low, medium, high
            'team_ml_capability': 'medium',  # low, medium, high
            'budget': 5_000_000,  # annual
            'time_to_market_pressure': 'medium',  # low, medium, high
            'data_sensitivity': 'high'  # low, medium, high
        }
        """

        score_build = 0
        score_buy = 0

        # Scale considerations
        if context["scale"] > 10_000_000_000:  # 10B+
            score_build += 3  # Commercial solutions expensive at this scale
        elif context["scale"] > 100_000_000:  # 100M+
            score_build += 1
        else:
            score_buy += 2  # Commercial solutions cost-effective at smaller scale

        # Performance requirements
        if context["qps"] > 100_000:
            score_build += 2  # Need custom optimization
        elif context["qps"] > 10_000:
            score_build += 1

        # Competitive differentiation
        if context["competitive_differentiation"] == "high":
            score_build += 3  # Embeddings are moat, must build
        elif context["competitive_differentiation"] == "medium":
            score_build += 1

        # Team capability
        if context["team_ml_capability"] == "high":
            score_build += 2  # Can execute custom build
        elif context["team_ml_capability"] == "low":
            score_buy += 2  # Should leverage external expertise

        # Time to market
        if context["time_to_market_pressure"] == "high":
            score_buy += 3  # Buy for speed
        elif context["time_to_market_pressure"] == "medium":
            score_buy += 1

        # Data sensitivity
        if context["data_sensitivity"] == "high":
            score_build += 2  # Keep data in-house
        elif context["data_sensitivity"] == "low":
            score_buy += 1

        # Budget
        if context["budget"] > 10_000_000:
            score_build += 1  # Can afford custom build
        elif context["budget"] < 1_000_000:
            score_buy += 2  # Limited budget favors buy

        # Recommendation
        if score_build > score_buy + 3:
            return {
                "recommendation": "build",
                "confidence": "high",
                "rationale": "Strong case for building custom solution",
                "score_build": score_build,
                "score_buy": score_buy,
            }
        elif score_build > score_buy:
            return {
                "recommendation": "build",
                "confidence": "medium",
                "rationale": "Slight preference for building",
                "score_build": score_build,
                "score_buy": score_buy,
                "caveat": "Consider hybrid: buy infrastructure, build models",
            }
        elif score_buy > score_build + 3:
            return {
                "recommendation": "buy",
                "confidence": "high",
                "rationale": "Strong case for commercial solution",
                "score_build": score_build,
                "score_buy": score_buy,
            }
        else:
            return {
                "recommendation": "buy",
                "confidence": "medium",
                "rationale": "Slight preference for buying",
                "score_build": score_build,
                "score_buy": score_buy,
                "caveat": "Start with buy, consider build later",
            }
