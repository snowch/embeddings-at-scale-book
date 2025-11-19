# Code from Chapter 13
# Book: Embeddings at Scale

"""
Contradiction Detection and Resolution

Strategies:
1. Detect contradictions: NLI models, entity conflict detection
2. Resolve temporal conflicts: Prioritize recent information
3. Resolve source conflicts: Weigh by credibility
4. Present multiple perspectives: Show disagreement to user

Production approach:
- Detect: Use NLI model to find contradicting statements
- Resolve: Apply resolution strategy based on conflict type
- Present: Show confidence, multiple views, or ask user
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class Claim:
    """
    Claim extracted from document

    Attributes:
        text: Claim text
        doc_id: Source document
        entity: Entity claim is about
        attribute: Attribute being claimed
        value: Value of attribute
        date: When claim was made
        confidence: Model confidence in extraction
    """
    text: str
    doc_id: str
    entity: str
    attribute: str
    value: str
    date: Optional[datetime] = None
    confidence: float = 1.0

@dataclass
class Contradiction:
    """
    Detected contradiction between claims

    Attributes:
        claim1: First claim
        claim2: Contradicting claim
        contradiction_type: Type of contradiction
        confidence: Confidence in contradiction detection
    """
    claim1: Claim
    claim2: Claim
    contradiction_type: str  # 'temporal', 'source', 'perspective'
    confidence: float

class ContradictionDetector:
    """
    Detect contradictions in retrieved documents

    Approach:
    1. Extract claims from documents
    2. Group claims by entity + attribute
    3. Check if values conflict
    4. Classify contradiction type

    Models:
    - Claim extraction: IE model or LLM
    - Contradiction detection: NLI model
    """

    def __init__(self):
        """Initialize contradiction detector"""
        print("Initialized Contradiction Detector")

    def detect(
        self,
        documents: List[Document]
    ) -> List[Contradiction]:
        """
        Detect contradictions across documents

        Args:
            documents: Retrieved documents

        Returns:
            List of detected contradictions
        """
        # Extract claims from each document
        all_claims = []
        for doc in documents:
            claims = self._extract_claims(doc)
            all_claims.extend(claims)

        print(f"Extracted {len(all_claims)} claims from {len(documents)} documents")

        # Group claims by entity + attribute
        claim_groups = self._group_claims(all_claims)

        # Detect contradictions within each group
        contradictions = []
        for (entity, attribute), claims in claim_groups.items():
            if len(claims) > 1:
                conflicts = self._find_conflicts(claims)
                contradictions.extend(conflicts)

        print(f"Detected {len(contradictions)} contradictions")

        return contradictions

    def _extract_claims(self, document: Document) -> List[Claim]:
        """
        Extract factual claims from document

        Production: Use IE model or LLM with structured output

        Args:
            document: Source document

        Returns:
            List of claims
        """
        # Placeholder: Simple pattern-based extraction
        # In production: Use proper claim extraction model

        claims = []

        # Example: Extract "X is Y" patterns
        # Real implementation would use NER + relation extraction

        # Mock claim for demo
        if 'price' in document.content.lower():
            claims.append(Claim(
                text="Product price is $99",
                doc_id=document.doc_id,
                entity="Product",
                attribute="price",
                value="$99",
                date=document.metadata.get('date'),
                confidence=0.9
            ))

        return claims

    def _group_claims(
        self,
        claims: List[Claim]
    ) -> Dict[Tuple[str, str], List[Claim]]:
        """
        Group claims by entity and attribute

        Args:
            claims: All extracted claims

        Returns:
            Dictionary: (entity, attribute) → [claims]
        """
        groups = {}

        for claim in claims:
            key = (claim.entity, claim.attribute)
            if key not in groups:
                groups[key] = []
            groups[key].append(claim)

        return groups

    def _find_conflicts(
        self,
        claims: List[Claim]
    ) -> List[Contradiction]:
        """
        Find contradictions among claims

        Args:
            claims: Claims about same entity + attribute

        Returns:
            List of contradictions
        """
        contradictions = []

        # Compare all pairs
        for i, claim1 in enumerate(claims):
            for claim2 in claims[i+1:]:
                if self._are_contradictory(claim1, claim2):
                    # Determine contradiction type
                    contra_type = self._classify_contradiction(claim1, claim2)

                    contradictions.append(Contradiction(
                        claim1=claim1,
                        claim2=claim2,
                        contradiction_type=contra_type,
                        confidence=0.8
                    ))

        return contradictions

    def _are_contradictory(self, claim1: Claim, claim2: Claim) -> bool:
        """
        Check if two claims contradict

        Production: Use NLI model

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            True if contradictory
        """
        # Simple: Different values for same entity + attribute
        return claim1.value != claim2.value

    def _classify_contradiction(
        self,
        claim1: Claim,
        claim2: Claim
    ) -> str:
        """
        Classify type of contradiction

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Contradiction type
        """
        # Temporal: Different dates
        if claim1.date and claim2.date and claim1.date != claim2.date:
            return 'temporal'

        # Source: Different documents
        if claim1.doc_id != claim2.doc_id:
            return 'source'

        # Perspective (default)
        return 'perspective'

class ContradictionResolver:
    """
    Resolve contradictions using various strategies

    Strategies:
    1. Temporal: Use most recent claim
    2. Source authority: Use most credible source
    3. Confidence: Use highest confidence claim
    4. Present multiple: Show disagreement to user
    """

    def __init__(
        self,
        source_credibility: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            source_credibility: Map of source → credibility score
        """
        self.source_credibility = source_credibility or {}
        print("Initialized Contradiction Resolver")

    def resolve(
        self,
        contradiction: Contradiction
    ) -> Claim:
        """
        Resolve contradiction by selecting best claim

        Args:
            contradiction: Detected contradiction

        Returns:
            Resolved claim
        """
        if contradiction.contradiction_type == 'temporal':
            return self._resolve_temporal(contradiction)
        elif contradiction.contradiction_type == 'source':
            return self._resolve_by_authority(contradiction)
        else:
            return self._resolve_by_confidence(contradiction)

    def _resolve_temporal(self, contradiction: Contradiction) -> Claim:
        """
        Resolve temporal contradiction: use most recent

        Args:
            contradiction: Temporal contradiction

        Returns:
            Most recent claim
        """
        claim1 = contradiction.claim1
        claim2 = contradiction.claim2

        if claim1.date and claim2.date:
            if claim1.date > claim2.date:
                return claim1
            else:
                return claim2

        # If no dates, fall back to confidence
        return self._resolve_by_confidence(contradiction)

    def _resolve_by_authority(self, contradiction: Contradiction) -> Claim:
        """
        Resolve by source authority: use more credible source

        Args:
            contradiction: Source contradiction

        Returns:
            Claim from more credible source
        """
        claim1 = contradiction.claim1
        claim2 = contradiction.claim2

        credibility1 = self.source_credibility.get(claim1.doc_id, 0.5)
        credibility2 = self.source_credibility.get(claim2.doc_id, 0.5)

        if credibility1 > credibility2:
            return claim1
        else:
            return claim2

    def _resolve_by_confidence(self, contradiction: Contradiction) -> Claim:
        """
        Resolve by confidence: use higher confidence claim

        Args:
            contradiction: Contradiction

        Returns:
            Higher confidence claim
        """
        if contradiction.claim1.confidence > contradiction.claim2.confidence:
            return contradiction.claim1
        else:
            return contradiction.claim2

    def format_multiple_perspectives(
        self,
        contradictions: List[Contradiction]
    ) -> str:
        """
        Format contradictions for user presentation

        When unable to resolve automatically, present multiple views

        Args:
            contradictions: List of contradictions

        Returns:
            Formatted text presenting multiple perspectives
        """
        if not contradictions:
            return ""

        output = "Note: Sources provide different information:\n\n"

        for contra in contradictions:
            output += f"• According to [{contra.claim1.doc_id}]: {contra.claim1.text}\n"
            output += f"• According to [{contra.claim2.doc_id}]: {contra.claim2.text}\n"
            output += "\n"

        return output

# Example: Contradiction handling
def contradiction_handling_example():
    """
    Demonstrate contradiction detection and resolution

    Scenario: Product information from multiple sources
    """

    # Create documents with contradictory information
    doc1 = Document(
        doc_id="catalog_2023",
        content="The Premium Laptop is priced at $999 and includes 16GB RAM.",
        metadata={'date': datetime(2023, 6, 1), 'source': 'catalog'}
    )

    doc2 = Document(
        doc_id="website_2024",
        content="The Premium Laptop now costs $1299 with upgraded 32GB RAM.",
        metadata={'date': datetime(2024, 1, 15), 'source': 'website'}
    )

    doc3 = Document(
        doc_id="review_2024",
        content="The Premium Laptop at $1299 offers excellent performance.",
        metadata={'date': datetime(2024, 2, 1), 'source': 'review'}
    )

    documents = [doc1, doc2, doc3]

    # Detect contradictions
    detector = ContradictionDetector()
    contradictions = detector.detect(documents)

    # Resolve contradictions
    resolver = ContradictionResolver(
        source_credibility={
            'catalog_2023': 0.9,
            'website_2024': 0.95,  # Most authoritative
            'review_2024': 0.7
        }
    )

    print(f"\n{'='*60}")
    print("Contradiction Analysis")
    print(f"{'='*60}")

    for i, contra in enumerate(contradictions, 1):
        print(f"\nContradiction {i}: {contra.contradiction_type}")
        print(f"  Claim 1 [{contra.claim1.doc_id}]: {contra.claim1.text}")
        print(f"  Claim 2 [{contra.claim2.doc_id}]: {contra.claim2.text}")

        # Resolve
        resolved = resolver.resolve(contra)
        print(f"  → Resolved: {resolved.text} (from {resolved.doc_id})")
        print(f"  → Reasoning: {contra.contradiction_type} → {'most recent' if contra.contradiction_type == 'temporal' else 'most credible'}")

    # Alternative: Present multiple perspectives
    print(f"\n{'='*60}")
    print("Alternative: Multiple Perspectives")
    print(f"{'='*60}")
    print(resolver.format_multiple_perspectives(contradictions))

# Uncomment to run:
# contradiction_handling_example()
