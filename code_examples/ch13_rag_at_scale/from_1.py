# Code from Chapter 13
# Book: Embeddings at Scale

"""
RAG Evaluation Framework

Metrics:
1. Retrieval metrics: Recall@k, MRR, NDCG
2. Context utilization: Did LLM use retrieved context?
3. Answer accuracy: Is answer correct?
4. Factual consistency: Is answer consistent with context?
5. Attribution quality: Are citations accurate?
6. User satisfaction: Would user accept this answer?

Evaluation data:
- Query
- Ground truth answer
- Ground truth relevant documents
- System retrieved documents
- System generated answer
- System citations
"""

from dataclasses import dataclass
from typing import List, Set, Optional
import numpy as np

@dataclass
class EvaluationSample:
    """
    Single evaluation sample

    Attributes:
        query: User query
        ground_truth_answer: Gold standard answer
        ground_truth_doc_ids: Relevant document IDs
        retrieved_doc_ids: System retrieved document IDs
        generated_answer: System generated answer
        citations: Document IDs cited in answer
    """
    query: str
    ground_truth_answer: str
    ground_truth_doc_ids: Set[str]
    retrieved_doc_ids: List[str]
    generated_answer: str
    citations: List[str]

class RAGEvaluator:
    """
    Comprehensive RAG evaluation

    Metrics:
    1. Retrieval quality
       - Recall@k: % of relevant docs in top-k
       - Precision@k: % of top-k that are relevant
       - MRR: Mean reciprocal rank of first relevant doc

    2. Answer quality
       - Accuracy: % of answers judged correct
       - Factual consistency: Answer consistent with context
       - Faithfulness: Answer only uses context (no hallucination)

    3. Attribution quality
       - Citation recall: % of facts cited
       - Citation precision: % of citations accurate

    4. End-to-end
       - User satisfaction: Would user accept answer?
    """

    def __init__(self):
        """Initialize evaluator"""
        print("Initialized RAG Evaluator")

    def evaluate(
        self,
        samples: List[EvaluationSample]
    ) -> Dict[str, float]:
        """
        Evaluate RAG system on sample set

        Args:
            samples: Evaluation samples

        Returns:
            Dictionary of metrics
        """
        print(f"\nEvaluating {len(samples)} samples...")

        # Retrieval metrics
        retrieval_metrics = self._evaluate_retrieval(samples)

        # Answer quality metrics
        answer_metrics = self._evaluate_answers(samples)

        # Attribution metrics
        attribution_metrics = self._evaluate_attribution(samples)

        # Combine all metrics
        metrics = {
            **retrieval_metrics,
            **answer_metrics,
            **attribution_metrics
        }

        # Print results
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        for metric, value in metrics.items():
            print(f"{metric:30s}: {value:.3f}")

        return metrics

    def _evaluate_retrieval(
        self,
        samples: List[EvaluationSample]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality

        Metrics:
        - Recall@k: Did we retrieve relevant documents?
        - Precision@k: Are retrieved documents relevant?
        - MRR: Rank of first relevant document

        Args:
            samples: Evaluation samples

        Returns:
            Retrieval metrics
        """
        recalls = []
        precisions = []
        reciprocal_ranks = []

        for sample in samples:
            # Recall@k: % of relevant docs retrieved
            retrieved_set = set(sample.retrieved_doc_ids)
            relevant_retrieved = retrieved_set & sample.ground_truth_doc_ids

            recall = len(relevant_retrieved) / len(sample.ground_truth_doc_ids) if sample.ground_truth_doc_ids else 0
            recalls.append(recall)

            # Precision@k: % of retrieved docs that are relevant
            precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0
            precisions.append(precision)

            # MRR: Rank of first relevant document
            rr = 0.0
            for rank, doc_id in enumerate(sample.retrieved_doc_ids, 1):
                if doc_id in sample.ground_truth_doc_ids:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

        return {
            'recall@k': np.mean(recalls),
            'precision@k': np.mean(precisions),
            'mrr': np.mean(reciprocal_ranks)
        }

    def _evaluate_answers(
        self,
        samples: List[EvaluationSample]
    ) -> Dict[str, float]:
        """
        Evaluate answer quality

        Metrics:
        - Accuracy: Is answer correct?
        - Faithfulness: Does answer only use context (no hallucination)?
        - Completeness: Does answer cover all aspects?

        Args:
            samples: Evaluation samples

        Returns:
            Answer metrics
        """
        accuracies = []
        faithfulness_scores = []
        completeness_scores = []

        for sample in samples:
            # Accuracy: Compare to ground truth
            # Production: Use LLM-as-judge or human evaluation
            accuracy = self._semantic_similarity(
                sample.generated_answer,
                sample.ground_truth_answer
            )
            accuracies.append(accuracy)

            # Faithfulness: Answer grounded in context?
            # Check if answer makes claims not in context
            faithfulness = self._check_faithfulness(sample)
            faithfulness_scores.append(faithfulness)

            # Completeness: All aspects covered?
            completeness = self._check_completeness(sample)
            completeness_scores.append(completeness)

        return {
            'accuracy': np.mean(accuracies),
            'faithfulness': np.mean(faithfulness_scores),
            'completeness': np.mean(completeness_scores)
        }

    def _evaluate_attribution(
        self,
        samples: List[EvaluationSample]
    ) -> Dict[str, float]:
        """
        Evaluate citation quality

        Metrics:
        - Citation recall: % of facts with citations
        - Citation precision: % of citations that are accurate

        Args:
            samples: Evaluation samples

        Returns:
            Attribution metrics
        """
        citation_recalls = []
        citation_precisions = []

        for sample in samples:
            # Citation recall: Are facts cited?
            # Production: Extract claims, check if cited
            # Placeholder: Check if any citations present
            has_citations = len(sample.citations) > 0
            citation_recalls.append(1.0 if has_citations else 0.0)

            # Citation precision: Are citations accurate?
            # Check if cited docs actually support claims
            if sample.citations:
                accurate_citations = len(set(sample.citations) & set(sample.retrieved_doc_ids))
                precision = accurate_citations / len(sample.citations)
            else:
                precision = 0.0

            citation_precisions.append(precision)

        return {
            'citation_recall': np.mean(citation_recalls),
            'citation_precision': np.mean(citation_precisions)
        }

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between texts

        Production: Use sentence embeddings (SentenceTransformers)

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Placeholder: Lexical overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        overlap = len(words1 & words2)
        union = len(words1 | words2)

        return overlap / union if union > 0 else 0.0

    def _check_faithfulness(self, sample: EvaluationSample) -> float:
        """
        Check if answer is faithful to context

        Faithfulness = answer only makes claims supported by context

        Production: Use NLI model or LLM-as-judge

        Args:
            sample: Evaluation sample

        Returns:
            Faithfulness score (0-1)
        """
        # Placeholder: Check if answer mentions retrieved docs
        # In production: Use entailment checking
        mentioned_docs = sum(
            1 for doc_id in sample.retrieved_doc_ids
            if doc_id in sample.generated_answer
        )

        return min(mentioned_docs / 3, 1.0)  # Expect ~3 doc mentions

    def _check_completeness(self, sample: EvaluationSample) -> float:
        """
        Check if answer is complete

        Completeness = answer covers all aspects of ground truth

        Args:
            sample: Evaluation sample

        Returns:
            Completeness score (0-1)
        """
        # Placeholder: Length ratio
        # In production: Check coverage of ground truth aspects
        length_ratio = len(sample.generated_answer) / max(len(sample.ground_truth_answer), 1)

        # Penalize too short or too long
        if length_ratio < 0.5 or length_ratio > 2.0:
            return 0.5
        return 0.8

# Example: RAG evaluation
def rag_evaluation_example():
    """
    Demonstrate RAG evaluation

    Scenario: Technical Q&A system
    """

    # Create evaluation samples
    samples = [
        EvaluationSample(
            query="What is RAG?",
            ground_truth_answer="RAG (Retrieval-Augmented Generation) combines retrieval and generation.",
            ground_truth_doc_ids={'doc_1', 'doc_5'},
            retrieved_doc_ids=['doc_1', 'doc_2', 'doc_5', 'doc_8'],
            generated_answer="RAG combines retrieval and generation using retrieved context. [doc_1]",
            citations=['doc_1']
        ),
        EvaluationSample(
            query="How does vector search work?",
            ground_truth_answer="Vector search finds similar items using embedding similarity.",
            ground_truth_doc_ids={'doc_3', 'doc_7'},
            retrieved_doc_ids=['doc_3', 'doc_4', 'doc_7', 'doc_9'],
            generated_answer="Vector search uses embeddings to find similar items efficiently. [doc_3] [doc_7]",
            citations=['doc_3', 'doc_7']
        ),
        EvaluationSample(
            query="What are the benefits of embeddings?",
            ground_truth_answer="Embeddings capture semantic meaning and enable similarity search.",
            ground_truth_doc_ids={'doc_2', 'doc_6'},
            retrieved_doc_ids=['doc_2', 'doc_3', 'doc_6', 'doc_10'],
            generated_answer="Embeddings provide semantic representations for ML tasks. [doc_2]",
            citations=['doc_2']
        )
    ]

    # Evaluate
    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate(samples)

    # Analyze results
    print(f"\n{'='*60}")
    print("Analysis")
    print(f"{'='*60}")

    if metrics['recall@k'] < 0.8:
        print("⚠️  Low recall: Retrieval missing relevant documents")
    if metrics['faithfulness'] < 0.8:
        print("⚠️  Low faithfulness: Answers may contain hallucinations")
    if metrics['citation_recall'] < 0.7:
        print("⚠️  Low citation recall: Facts not properly attributed")

    if metrics['recall@k'] >= 0.8 and metrics['accuracy'] >= 0.8 and metrics['faithfulness'] >= 0.8:
        print("✓ System performing well across all metrics")

# Uncomment to run:
# rag_evaluation_example()
