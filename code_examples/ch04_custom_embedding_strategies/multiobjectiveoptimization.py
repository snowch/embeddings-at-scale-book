# Code from Chapter 04
# Book: Embeddings at Scale

class MultiObjectiveOptimization:
    """
    Navigate trade-offs between multiple objectives
    """

    def compute_pareto_frontier(self, models, test_data):
        """
        Compute Pareto frontier across objectives

        Args:
            models: List of embedding models with different objective weightings
            test_data: Evaluation data

        Returns:
            Pareto-optimal models
        """
        # Evaluate all models on all objectives
        evaluations = []

        for model in models:
            metrics = {
                'model': model,
                'relevance': self.evaluate_relevance(model, test_data),
                'diversity': self.evaluate_diversity(model, test_data),
                'personalization': self.evaluate_personalization(model, test_data),
                'business_metrics': self.evaluate_business(model, test_data)
            }
            evaluations.append(metrics)

        # Find Pareto-optimal models
        pareto_optimal = []

        for eval_i in evaluations:
            dominated = False

            for eval_j in evaluations:
                if eval_i == eval_j:
                    continue

                # Check if eval_j dominates eval_i
                # (better on all objectives)
                if self.dominates(eval_j, eval_i):
                    dominated = True
                    break

            if not dominated:
                pareto_optimal.append(eval_i)

        return pareto_optimal

    def dominates(self, eval_a, eval_b):
        """
        Check if eval_a dominates eval_b (better on all objectives)
        """
        objectives = ['relevance', 'diversity', 'personalization', 'business_metrics']

        # A dominates B if:
        # - A >= B on all objectives
        # - A > B on at least one objective

        better_on_at_least_one = False

        for obj in objectives:
            if eval_a[obj] < eval_b[obj]:
                return False  # A worse on this objective
            if eval_a[obj] > eval_b[obj]:
                better_on_at_least_one = True

        return better_on_at_least_one

    def select_operating_point(self, pareto_frontier, business_priorities):
        """
        Select model from Pareto frontier based on business priorities
        """
        # Weight objectives by business priority
        weights = business_priorities  # e.g., {'relevance': 0.4, 'business_metrics': 0.4, ...}

        best_model = None
        best_weighted_score = -float('inf')

        for eval_point in pareto_frontier:
            weighted_score = sum(
                weights.get(obj, 0) * eval_point[obj]
                for obj in ['relevance', 'diversity', 'personalization', 'business_metrics']
            )

            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                best_model = eval_point['model']

        return best_model
