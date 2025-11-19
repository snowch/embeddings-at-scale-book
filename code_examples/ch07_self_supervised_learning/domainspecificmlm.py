# Code from Chapter 07
# Book: Embeddings at Scale

import torch
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class DomainSpecificMLM:
    """
    Masked Language Modeling for domain-specific text

    Use cases:
    - Legal documents: Learn legal terminology and structure
    - Medical records: Capture clinical language patterns
    - Financial reports: Understand financial jargon
    - Scientific papers: Model academic writing style
    - Customer support: Learn product-specific terminology
    """

    def __init__(
        self, domain="general", vocab_size=30000, hidden_size=768, num_layers=12, num_heads=12
    ):
        """
        Args:
            domain: Domain name for logging
            vocab_size: Size of vocabulary (larger for specialized domains)
            hidden_size: Dimension of embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        self.domain = domain

        # Configure BERT architecture
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=512,
        )

        # Initialize model
        self.model = BertForMaskedLM(self.config)
        self.tokenizer = None

    def train_tokenizer(self, text_corpus, save_path="./tokenizer"):
        """
        Train domain-specific tokenizer

        Critical for specialized domains: generic tokenizers split
        domain terms into subwords, losing semantic coherence.

        Args:
            text_corpus: Iterator of text strings
            save_path: Path to save tokenizer
        """
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        # Initialize tokenizer
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        # Configure trainer
        trainer = BpeTrainer(
            vocab_size=self.config.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            show_progress=True,
        )

        # Train on corpus
        tokenizer.train_from_iterator(text_corpus, trainer=trainer)

        # Save
        tokenizer.save(f"{save_path}/tokenizer.json")

        # Create HuggingFace tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(save_path)

        print(f"Tokenizer trained and saved to {save_path}")

    def prepare_dataset(self, texts, tokenizer):
        """
        Prepare dataset for MLM training

        Args:
            texts: List of text strings
            tokenizer: HuggingFace tokenizer

        Returns:
            Dataset ready for MLM training
        """
        from datasets import Dataset

        # Create dataset
        dataset = Dataset.from_dict({"text": texts})

        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], truncation=True, max_length=512, padding="max_length"
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        return tokenized_dataset

    def train(
        self,
        train_dataset,
        output_dir="./mlm_model",
        num_epochs=3,
        batch_size=32,
        learning_rate=5e-5,
        mlm_probability=0.15,
    ):
        """
        Train MLM model

        Args:
            train_dataset: Tokenized dataset
            output_dir: Where to save model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            mlm_probability: Probability of masking each token
        """
        # Data collator handles masking
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=mlm_probability
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=500,
            logging_steps=100,
            save_steps=1000,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),  # Mixed precision training
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        # Train
        print(f"Starting MLM training for domain: {self.domain}")
        trainer.train()

        # Save final model
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")

    def get_embeddings(self, texts, layer=-1):
        """
        Extract embeddings from trained model

        Args:
            texts: List of text strings
            layer: Which layer to extract from (-1 = last layer)

        Returns:
            Embeddings: (num_texts, hidden_size)
        """
        self.model.eval()

        # Tokenize
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        # Forward pass
        with torch.no_grad():
            outputs = self.model.bert(**inputs, output_hidden_states=True)

        # Extract embeddings from specified layer
        hidden_states = outputs.hidden_states[layer]

        # Mean pooling over sequence length
        embeddings = hidden_states.mean(dim=1)

        return embeddings.numpy()


# Example: Training domain-specific MLM
def example_legal_mlm():
    """
    Example: Train MLM on legal documents

    Legal text has specialized terminology (tort, plaintiff, jurisdiction)
    and structure (citations, precedents) that generic models miss.
    """
    # Initialize
    legal_mlm = DomainSpecificMLM(domain="legal", vocab_size=32000, hidden_size=768, num_layers=12)

    # Sample legal texts (in production: millions of documents)
    legal_corpus = [
        "The plaintiff filed a motion for summary judgment...",
        "Under tort law, negligence requires duty, breach, causation...",
        "Precedent established in Smith v. Jones (2020) supports...",
        # ... millions more
    ]

    # Train domain-specific tokenizer
    print("Training legal tokenizer...")
    legal_mlm.train_tokenizer(legal_corpus, save_path="./legal_tokenizer")

    # Prepare dataset
    print("Preparing dataset...")
    train_dataset = legal_mlm.prepare_dataset(legal_corpus, legal_mlm.tokenizer)

    # Train MLM
    print("Training MLM...")
    legal_mlm.train(train_dataset, output_dir="./legal_mlm_model", num_epochs=3, batch_size=16)

    # Extract embeddings
    print("Extracting embeddings...")
    test_texts = [
        "The defendant's motion to dismiss was denied.",
        "Statutory interpretation follows the plain meaning rule.",
    ]
    embeddings = legal_mlm.get_embeddings(test_texts)

    print(f"Embeddings shape: {embeddings.shape}")

    return legal_mlm
