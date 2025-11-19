# Code from Chapter 01
# Book: Embeddings at Scale

# Wrong: Single-node architecture
embeddings = np.load('embeddings.npy')  # Doesn't scale
index = faiss.IndexFlatL2(dim)  # In-memory only
index.add(embeddings)

# Right: Distributed-first architecture
from distributed_index import ShardedIndex

index = ShardedIndex(
    num_shards=1000,
    shard_backend='s3',  # Cloud-native storage
    index_type='HNSW',
    replication_factor=3  # High availability
)
# Scales from millions to trillions with same API
