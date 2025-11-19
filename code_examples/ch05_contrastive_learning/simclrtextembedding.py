# Code from Chapter 05
# Book: Embeddings at Scale

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class SimCLRTextEmbedding(nn.Module):
    """
    SimCLR adapted for text embeddings

    Key adaptations from visual SimCLR:
    - Text augmentation instead of image augmentation
    - BERT/RoBERTa encoder instead of ResNet
    - Larger projection head for text (768 → 128 works well)
    """

    def __init__(self,
                 base_model='bert-base-uncased',
                 projection_dim=128,
                 hidden_dim=512,
                 temperature=0.07):
        super().__init__()

        # Base encoder (frozen or fine-tuned)
        self.encoder = AutoModel.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        encoder_dim = self.encoder.config.hidden_size  # 768 for BERT-base

        # Projection head: critical component
        # SimCLR paper shows projection head is essential
        # Without it, performance drops 10-20%
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

        self.temperature = temperature

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through encoder and projection head

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            embeddings: (batch_size, projection_dim)
            representations: (batch_size, encoder_dim) - before projection
        """
        # Encode text
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation or mean pooling
        # [CLS] token: outputs.last_hidden_state[:, 0]
        # Mean pooling: (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)

        # Mean pooling (often better for sentence embeddings)
        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size())
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        representations = sum_embeddings / sum_mask

        # Project to contrastive space
        embeddings = self.projection_head(representations)

        return embeddings, representations

    def compute_simclr_loss(self, embeddings):
        """
        Compute SimCLR NT-Xent loss

        Args:
            embeddings: (2*batch_size, projection_dim)
                       First half are view 1, second half are view 2

        Returns:
            loss: scalar
            metrics: dict with accuracy and similarities
        """
        batch_size = embeddings.shape[0] // 2

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Similarity matrix: (2*batch, 2*batch)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create labels: for each 2i, positive is 2i+1 and vice versa
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size),  # For first half
            torch.arange(0, batch_size)  # For second half
        ]).to(embeddings.device)

        # Mask out self-similarities
        mask = torch.eye(2*batch_size, dtype=torch.bool, device=embeddings.device)
        similarity_matrix.masked_fill_(mask, -9e15)

        # Cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)

        # Compute metrics
        with torch.no_grad():
            predictions = similarity_matrix.argmax(dim=1)
            accuracy = (predictions == labels).float().mean()

            # Positive similarities
            positive_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
            positive_mask[torch.arange(2*batch_size), labels] = True
            positive_sim = similarity_matrix[positive_mask].mean()

            # Negative similarities
            negative_sim = similarity_matrix[~positive_mask & ~mask].mean()

        metrics = {
            'accuracy': accuracy.item(),
            'positive_similarity': positive_sim.item(),
            'negative_similarity': negative_sim.item()
        }

        return loss, metrics


class TextAugmentation:
    """
    Text augmentation strategies for contrastive learning

    Unlike images (crop, flip, color jitter), text augmentation is trickier.
    Must preserve semantic meaning while changing surface form.
    """

    def __init__(self):
        self.augmentation_methods = [
            'synonym_replacement',
            'back_translation',
            'random_deletion',
            'random_swap',
            'paraphrase_generation'
        ]

    def augment_simple(self, text, method='random_deletion', p=0.1):
        """
        Simple augmentation: deletion, swapping, synonym replacement

        Args:
            text: Input text
            method: Augmentation method
            p: Probability of applying to each word

        Returns:
            augmented text
        """
        import random
        from nltk.corpus import wordnet

        words = text.split()

        if method == 'random_deletion':
            # Randomly delete words
            if len(words) == 1:
                return text

            new_words = [w for w in words if random.random() > p]

            # Ensure at least one word remains
            if len(new_words) == 0:
                return random.choice(words)

            return ' '.join(new_words)

        elif method == 'random_swap':
            # Randomly swap word positions
            new_words = words.copy()
            for i in range(len(new_words)):
                if random.random() < p:
                    swap_idx = random.randint(0, len(new_words) - 1)
                    new_words[i], new_words[swap_idx] = new_words[swap_idx], new_words[i]

            return ' '.join(new_words)

        elif method == 'synonym_replacement':
            # Replace words with synonyms
            new_words = []
            for word in words:
                if random.random() < p:
                    synonyms = self.get_synonyms(word)
                    if synonyms:
                        new_words.append(random.choice(synonyms))
                    else:
                        new_words.append(word)
                else:
                    new_words.append(word)

            return ' '.join(new_words)

        return text

    def augment_with_llm(self, text, model='gpt-3.5-turbo'):
        """
        Use LLM to generate paraphrases (higher quality, more expensive)

        For production at scale:
        - Pre-generate augmentations offline
        - Cache augmentations
        - Use smaller, faster models (T5, BART)
        """
        from openai import OpenAI

        client = OpenAI()

        prompt = f"""Paraphrase the following text while preserving its meaning:

Text: {text}

Paraphrase:"""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=len(text.split()) * 2,
            temperature=0.7
        )

        paraphrase = response.choices[0].message.content.strip()
        return paraphrase

    def get_synonyms(self, word):
        """Get synonyms using WordNet"""
        from nltk.corpus import wordnet

        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)

        return list(synonyms)


# Training loop
def train_simclr_epoch(model, dataloader, optimizer, device):
    """
    Training epoch for SimCLR
    """
    model.train()
    total_loss = 0
    total_accuracy = 0
    augmenter = TextAugmentation()

    for batch in dataloader:
        texts = batch['text']

        # Generate two augmented views for each text
        augmented_texts = []
        for text in texts:
            aug1 = augmenter.augment_simple(text, method='random_deletion')
            aug2 = augmenter.augment_simple(text, method='synonym_replacement')
            augmented_texts.extend([aug1, aug2])

        # Tokenize both views
        encoded = model.tokenizer(
            augmented_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)

        # Forward pass
        embeddings, _ = model(encoded['input_ids'], encoded['attention_mask'])

        # Compute loss
        loss, metrics = model.compute_simclr_loss(embeddings)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += metrics['accuracy']

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    return avg_loss, avg_accuracy


# Example usage
model = SimCLRTextEmbedding(
    base_model='bert-base-uncased',
    projection_dim=128,
    temperature=0.07
).to('cuda')

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Train
# for epoch in range(num_epochs):
#     loss, acc = train_simclr_epoch(model, train_loader, optimizer, 'cuda')
#     print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={acc:.2%}")
