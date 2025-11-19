# Code from Chapter 03
# Book: Embeddings at Scale


class IVFPQTradeoffs:
    """Understanding when to use IVF-PQ vs HNSW"""

    def comparison(self):
        """Side-by-side comparison"""

        return {
            "memory": {
                "hnsw": "1.5-2x raw data size",
                "ivf_pq": "0.02-0.05x raw data (20-50x compression)",
                "winner": "IVF-PQ by far",
            },
            "recall": {
                "hnsw": "95-99% with proper tuning",
                "ivf_pq": "85-95% (quantization loses precision)",
                "winner": "HNSW",
            },
            "latency": {
                "hnsw_p99": "20-100ms",
                "ivf_pq_p99": "50-200ms (depends on n_probe)",
                "winner": "HNSW",
            },
            "build_time": {
                "hnsw": "Slower (must build graph)",
                "ivf_pq": "Faster (just k-means + assignment)",
                "winner": "IVF-PQ",
            },
            "updates": {
                "hnsw": "Easy incremental inserts",
                "ivf_pq": "Must reassign to centroids",
                "winner": "HNSW",
            },
            "scalability": {
                "hnsw": "Billions to low trillions",
                "ivf_pq": "Trillions+ (memory efficiency)",
                "winner": "IVF-PQ for massive scale",
            },
        }

    def when_to_use(self):
        """Decision matrix"""

        return {
            "use_hnsw_when": [
                "High recall required (>95%)",
                "Low latency critical (p99 <100ms)",
                "Frequent updates",
                "Memory budget allows (1.5-2x data size)",
                "Scale: up to 100B vectors per shard",
            ],
            "use_ivf_pq_when": [
                "Memory constrained (need 20x+ compression)",
                "Can tolerate lower recall (85-90%)",
                "Higher latency acceptable (100-200ms)",
                "Infrequent updates",
                "Scale: 100B+ to trillions of vectors",
            ],
            "hybrid_approach": {
                "strategy": "IVF for coarse search, HNSW within partitions",
                "benefit": "Memory efficiency of IVF + recall of HNSW",
                "when": "Best of both worlds for trillion+ scale",
            },
        }


# Recommendation engine
def recommend_index_strategy(num_vectors, memory_budget_gb, recall_requirement, latency_p99_ms):
    """Recommend index strategy based on requirements"""

    embedding_dim = 768

    # Calculate memory needs
    raw_data_gb = (num_vectors * embedding_dim * 4) / (1024**3)
    hnsw_memory_gb = raw_data_gb * 1.7
    ivf_pq_memory_gb = raw_data_gb * 0.03

    if memory_budget_gb < ivf_pq_memory_gb:
        return "Insufficient memory for any approach - need more machines or smaller dataset"

    if memory_budget_gb >= hnsw_memory_gb and recall_requirement >= 0.95 and latency_p99_ms <= 100:
        return "HNSW - you can afford the memory and need high recall + low latency"

    if memory_budget_gb < hnsw_memory_gb and recall_requirement < 0.90:
        return "IVF-PQ - memory constrained and can tolerate lower recall"

    if memory_budget_gb >= hnsw_memory_gb * 0.3 and recall_requirement >= 0.93:
        return "Hybrid IVF-HNSW - balance memory and recall"

    return "IVF-PQ with high n_probe - best fit for your constraints"


# Example
recommendation = recommend_index_strategy(
    num_vectors=100_000_000_000,
    memory_budget_gb=50_000,  # 50TB
    recall_requirement=0.95,
    latency_p99_ms=80,
)
print(f"Recommendation: {recommendation}")
