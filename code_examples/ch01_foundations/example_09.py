# Code from Chapter 01
# Book: Embeddings at Scale

# Query performance at 256T scale with proper indexing
index_type = "HNSW"  # Hierarchical Navigable Small World
num_embeddings = 256 * 10**12
embedding_dim = 768

# Theoretical complexity
# HNSW: O(log(N)) for insert and search
import math
avg_hops = math.log2(num_embeddings)  # ~48 hops

# Practical performance with distributed architecture:
num_shards = 1000  # Distribute across 1000 nodes
embeddings_per_shard = num_embeddings / num_shards  # 256B per shard
shards_to_search = 10  # Parallel search across 10 shards

# Query latency budget (p50, warm cache):
# - Shard selection: 5ms
# - Parallel shard search (10 shards): 50ms
# - Result aggregation: 5ms
# Total: ~60ms p50 latency
# Note: p99 latency typically 200-500ms due to network variability and cold cache
