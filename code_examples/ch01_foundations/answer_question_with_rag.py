import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Code from Chapter 01
# Book: Embeddings at Scale

# Placeholder encoder - in production, use actual SentenceTransformer or similar
class PlaceholderEncoder:
    """Placeholder encoder for demonstration. Replace with actual model."""
    def encode(self, text):
        if isinstance(text, str):
            # Return fixed-size embedding for single text
            return np.random.randn(768).astype(np.float32)
        else:
            # Return batch of embeddings
            return np.random.randn(len(text), 768).astype(np.float32)

encoder = PlaceholderEncoder()

# Placeholder LLM
class PlaceholderLLM:
    """Placeholder LLM for demonstration. Replace with actual LLM."""
    def generate(self, prompt):
        return "Generated response based on prompt..."

llm = PlaceholderLLM()

def answer_question_with_rag(question, knowledge_base_embeddings, knowledge_base_text):
    # 1. Embed the question
    question_embedding = encoder.encode(question)

    # 2. Find semantically relevant context via embeddings
    similarities = cosine_similarity([question_embedding], knowledge_base_embeddings)
    top_k_indices = similarities.argsort()[0][-5:][::-1]
    relevant_context = [knowledge_base_text[i] for i in top_k_indices]

    # 3. Generate answer using retrieved context
    prompt = f"""
    Context: {' '.join(relevant_context)}

    Question: {question}

    Answer based on the context above:
    """
    answer = llm.generate(prompt)

    return answer, relevant_context
