import numpy as np
import torch
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Code from Chapter 07
# Book: Embeddings at Scale


class AdvancedMLM:
    """
    Advanced MLM with production optimizations

    Enhancements:
    - Whole word masking: Mask entire words, not subwords
    - Dynamic masking: Different masks each epoch
    - Span masking: Mask contiguous spans
    - Entity masking: Preferentially mask named entities
    """

    def __init__(self, base_model, tokenizer):
        self.model = base_model
        self.tokenizer = tokenizer

    def whole_word_masking(self, input_ids, mlm_probability=0.15):
        """
        Mask entire words instead of subword tokens

        Better for learning word-level semantics
        """
        # Get word boundaries
        words = []
        current_word = []

        for idx, token_id in enumerate(input_ids):
            token = self.tokenizer.decode([token_id])

            if token.startswith("##"):
                # Continuation of previous word
                current_word.append(idx)
            else:
                # New word
                if current_word:
                    words.append(current_word)
                current_word = [idx]

        if current_word:
            words.append(current_word)

        # Sample words to mask
        num_words_to_mask = max(1, int(len(words) * mlm_probability))
        words_to_mask = np.random.choice(len(words), size=num_words_to_mask, replace=False)

        # Create mask
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for word_idx in words_to_mask:
            for token_idx in words[word_idx]:
                mask[token_idx] = True

        return mask

    def span_masking(self, input_ids, span_length=3, mlm_probability=0.15):
        """
        Mask contiguous spans of tokens

        Encourages learning longer-range dependencies
        Based on SpanBERT
        """
        seq_len = len(input_ids)
        num_masks = int(seq_len * mlm_probability / span_length)

        mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for _ in range(num_masks):
            # Sample span start
            start = np.random.randint(0, max(1, seq_len - span_length))

            # Mask span
            mask[start : start + span_length] = True

        return mask

    def entity_aware_masking(self, text, entities, mlm_probability=0.15):
        """
        Preferentially mask named entities

        Helps model learn domain-specific entities

        Args:
            text: Input text
            entities: List of (entity_text, entity_type) tuples
            mlm_probability: Masking probability

        Returns:
            Masked input_ids and labels
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Find entity positions
        entity_positions = []
        for entity_text, _entity_type in entities:
            entity_tokens = self.tokenizer.tokenize(entity_text)

            # Find entity in tokens
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if tokens[i : i + len(entity_tokens)] == entity_tokens:
                    entity_positions.extend(range(i, i + len(entity_tokens)))

        # Create mask
        mask = torch.zeros(len(input_ids), dtype=torch.bool)

        # Mask entities with higher probability
        entity_mask_prob = mlm_probability * 3  # 3x more likely to mask entities
        regular_mask_prob = mlm_probability

        for idx in range(len(input_ids)):
            prob = entity_mask_prob if idx in entity_positions else regular_mask_prob
            if np.random.rand() < prob:
                mask[idx] = True

        return input_ids, mask


class DomainAdaptiveMLM:
    """
    Domain adaptation for pre-trained MLM

    Strategy: Start with pre-trained BERT, continue training on domain data

    Benefits:
    - Faster than training from scratch
    - Retains general language understanding
    - Adapts to domain specifics
    """

    def __init__(self, pretrained_model_name="bert-base-uncased"):
        """
        Args:
            pretrained_model_name: HuggingFace model to adapt
        """
        self.model = BertForMaskedLM.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    def adapt_to_domain(
        self,
        domain_texts,
        output_dir="./adapted_model",
        num_epochs=3,
        learning_rate=2e-5,  # Lower LR for adaptation
    ):
        """
        Adapt pre-trained model to domain

        Args:
            domain_texts: Domain-specific texts
            output_dir: Where to save adapted model
            num_epochs: Adaptation epochs (fewer than training from scratch)
            learning_rate: Lower learning rate preserves pre-training
        """
        from datasets import Dataset

        # Prepare dataset
        dataset = Dataset.from_dict({"text": domain_texts})

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], truncation=True, max_length=512, padding="max_length"
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )

        # Training arguments (conservative for adaptation)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=100,
            save_steps=1000,
            fp16=torch.cuda.is_available(),
        )

        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
        )

        print("Adapting model to domain...")
        trainer.train()

        trainer.save_model(output_dir)
        print(f"Adapted model saved to {output_dir}")
