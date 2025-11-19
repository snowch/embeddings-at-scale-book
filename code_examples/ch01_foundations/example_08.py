# Code from Chapter 01
# Book: Embeddings at Scale

# Cost calculation for 256 trillion embeddings
num_embeddings = 256 * 10**12  # 256 trillion
embedding_dim = 768  # Common dimension
bytes_per_float = 4  # float32

# Raw storage
total_bytes = num_embeddings * embedding_dim * bytes_per_float
total_petabytes = total_bytes / (10**15)  # 786 PB

# With compression (6:1 typical for embeddings)
compressed_petabytes = total_petabytes / 6  # 131 PB

# Storage cost at $0.02/GB/month (object storage)
monthly_storage_cost = compressed_petabytes * 1000 * 1000 * 0.02  # $2.6M/month

# Index cost (HNSW index adds ~50% storage overhead)
index_storage_cost = monthly_storage_cost * 1.5  # $3.9M/month

# Total infrastructure: ~$47M/year
