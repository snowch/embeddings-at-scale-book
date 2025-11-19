# Code from Chapter 04
# Book: Embeddings at Scale

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

class EmbeddingFineTuner:
    """
    Production-ready fine-tuning for sentence embeddings
    """

    def __init__(self, base_model_name='all-mpnet-base-v2'):
        """
        Args:
            base_model_name: HuggingFace model identifier
        """
        self.model = SentenceTransformer(base_model_name)
        self.base_model_name = base_model_name

    def prepare_training_data(self, examples):
        """
        Prepare training data in correct format

        Args:
            examples: List of dicts with 'query', 'positive', 'negative' (optional)

        Returns:
            DataLoader for training
        """
        train_examples = []

        for ex in examples:
            if 'negative' in ex:
                # Triplet: query, positive, negative
                train_examples.append(
                    InputExample(texts=[ex['query'], ex['positive'], ex['negative']])
                )
            else:
                # Pair: query, positive (with label 1.0)
                train_examples.append(
                    InputExample(texts=[ex['query'], ex['positive']], label=1.0)
                )

        return DataLoader(train_examples, shuffle=True, batch_size=16)

    def fine_tune(self, train_dataloader, num_epochs=3,
                  loss_function='cosine', warmup_steps=100):
        """
        Fine-tune the model

        Args:
            train_dataloader: DataLoader with training examples
            num_epochs: Number of training epochs
            loss_function: 'cosine', 'triplet', or 'contrastive'
            warmup_steps: Learning rate warmup steps
        """

        # Select loss function
        if loss_function == 'cosine':
            # CosineSimilarityLoss: learns to maximize similarity of positive pairs
            train_loss = losses.CosineSimilarityLoss(self.model)
        elif loss_function == 'triplet':
            # TripletLoss: anchor, positive, negative triplets
            train_loss = losses.TripletLoss(
                model=self.model,
                distance_metric=losses.TripletDistanceMetric.COSINE,
                triplet_margin=0.5
            )
        elif loss_function == 'contrastive':
            # ContrastiveLoss: positive and negative pairs with labels
            train_loss = losses.ContrastiveLoss(self.model)
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

        # Training configuration
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * num_epochs

        print(f"Fine-tuning {self.base_model_name}")
        print(f"  Total steps: {total_steps}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Loss: {loss_function}")

        # Fine-tune
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': 2e-5},
            show_progress_bar=True
        )

        print("Fine-tuning complete!")

    def evaluate(self, test_examples):
        """
        Evaluate fine-tuned model

        Args:
            test_examples: List of {'query', 'positive', 'negatives': [...]}

        Returns:
            Evaluation metrics
        """
        from sentence_transformers.evaluation import InformationRetrievalEvaluator

        # Prepare evaluation data
        queries = {i: ex['query'] for i, ex in enumerate(test_examples)}
        corpus = {}
        relevant_docs = {}

        corpus_id = 0
        for i, ex in enumerate(test_examples):
            # Add positive
            corpus[corpus_id] = ex['positive']
            relevant_docs[i] = [corpus_id]
            corpus_id += 1

            # Add negatives
            for neg in ex.get('negatives', []):
                corpus[corpus_id] = neg
                corpus_id += 1

        # Create evaluator
        evaluator = InformationRetrievalEvaluator(
            queries, corpus, relevant_docs,
            name='test_evaluation'
        )

        # Run evaluation
        results = evaluator(self.model)

        return results

    def save_model(self, output_path):
        """Save fine-tuned model"""
        self.model.save(output_path)
        print(f"Model saved to {output_path}")


# Example: Fine-tune for product search
# Training data: 100K product queries with positive/negative product descriptions
training_data = [
    {
        'query': 'comfortable running shoes for marathon',
        'positive': 'Nike Air Zoom Pegasus - Premium cushioning for long-distance running',
        'negative': 'Nike Basketball Shoes - High-top design for court performance'
    },
    {
        'query': 'waterproof camping tent 4 person',
        'positive': 'Coleman 4-Person Tent - Waterproof rainfly, sleeps 4 comfortably',
        'negative': 'Coleman Sleeping Bag - Warm sleeping bag for camping'
    },
    # ... 100K more examples
]

# Initialize fine-tuner
finetuner = EmbeddingFineTuner(base_model_name='all-mpnet-base-v2')

# Prepare data
train_loader = finetuner.prepare_training_data(training_data)

# Fine-tune
finetuner.fine_tune(
    train_loader,
    num_epochs=3,
    loss_function='triplet',
    warmup_steps=500
)

# Save
finetuner.save_model('./models/product-search-embeddings-v1')
