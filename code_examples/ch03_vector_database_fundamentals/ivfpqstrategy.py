import numpy as np
from sklearn.cluster import KMeans

# Code from Chapter 03
# Book: Embeddings at Scale

class IVFPQStrategy:
    """IVF with Product Quantization for memory efficiency"""

    def __init__(self, num_centroids=65536, num_subquantizers=8, bits_per_sq=8):
        """
        num_centroids: Number of Voronoi cells (typical: 4096-65536)
        num_subquantizers: Divide vector into subvectors (typical: 8-16)
        bits_per_sq: Bits per subquantizer (typical: 8 = 256 clusters)
        """
        self.num_centroids = num_centroids
        self.num_subquantizers = num_subquantizers
        self.bits_per_sq = bits_per_sq

        self.centroids = None  # Coarse centroids
        self.subquantizers = None  # PQ codebooks
        self.inverted_lists = {}  # {centroid_id: [vector_codes]}

    def train(self, training_vectors):
        """
        Train IVF-PQ index

        Steps:
        1. K-means clustering to create num_centroids Voronoi cells
        2. Assign each vector to nearest centroid
        3. Train product quantizers on residuals within each cell
        """

        # Step 1: Coarse quantization (K-means)
        print(f"Training {self.num_centroids} centroids...")
        kmeans = KMeans(n_clusters=self.num_centroids, max_iter=20)
        centroid_assignments = kmeans.fit_predict(training_vectors)
        self.centroids = kmeans.cluster_centers_

        # Step 2: Assign vectors to inverted lists
        for vector_id, centroid_id in enumerate(centroid_assignments):
            if centroid_id not in self.inverted_lists:
                self.inverted_lists[centroid_id] = []
            self.inverted_lists[centroid_id].append(vector_id)

        # Step 3: Train product quantizers
        print(f"Training {self.num_subquantizers} product quantizers...")
        self.train_product_quantization(training_vectors, centroid_assignments)

    def train_product_quantization(self, vectors, centroid_assignments):
        """Train PQ codebooks for compression"""

        embedding_dim = vectors.shape[1]
        subvector_dim = embedding_dim // self.num_subquantizers

        self.subquantizers = []

        for subq_idx in range(self.num_subquantizers):
            # Extract subvector
            start = subq_idx * subvector_dim
            end = start + subvector_dim
            subvectors = vectors[:, start:end]

            # Train KMeans on this subvector
            num_clusters = 2 ** self.bits_per_sq  # 256 for 8 bits
            kmeans = KMeans(n_clusters=num_clusters, max_iter=20)
            kmeans.fit(subvectors)

            self.subquantizers.append(kmeans)

    def add(self, vector, vector_id):
        """Add vector to index with compression"""

        # Find nearest centroid
        centroid_id = self.find_nearest_centroid(vector)

        # Compute residual
        residual = vector - self.centroids[centroid_id]

        # Encode residual with product quantization
        code = self.encode_pq(residual)

        # Add to inverted list
        self.inverted_lists[centroid_id].append({
            'vector_id': vector_id,
            'code': code
        })

    def encode_pq(self, vector):
        """Encode vector using product quantization"""

        embedding_dim = len(vector)
        subvector_dim = embedding_dim // self.num_subquantizers

        code = []
        for subq_idx in range(self.num_subquantizers):
            start = subq_idx * subvector_dim
            end = start + subvector_dim
            subvector = vector[start:end]

            # Find nearest cluster in this subquantizer
            cluster_id = self.subquantizers[subq_idx].predict([subvector])[0]
            code.append(cluster_id)

        return code

    def search(self, query, k=10, n_probe=10):
        """
        Search for k nearest neighbors

        n_probe: Number of centroids to search (higher = better recall, slower)
        """

        # Find n_probe nearest centroids
        centroid_distances = [
            (i, np.linalg.norm(query - centroid))
            for i, centroid in enumerate(self.centroids)
        ]
        centroid_distances.sort(key=lambda x: x[1])
        nearest_centroids = [c[0] for c in centroid_distances[:n_probe]]

        # Search within each centroid's inverted list
        candidates = []
        for centroid_id in nearest_centroids:
            # Compute residual
            residual_query = query - self.centroids[centroid_id]

            # Asymmetric distance computation (query vs compressed vectors)
            for item in self.inverted_lists.get(centroid_id, []):
                distance = self.compute_asymmetric_distance(
                    residual_query,
                    item['code']
                )
                candidates.append((item['vector_id'], distance))

        # Return top-k
        candidates.sort(key=lambda x: x[1])
        return [c[0] for c in candidates[:k]]

    def compute_asymmetric_distance(self, query_vector, pq_code):
        """Compute distance between full query and PQ-compressed vector"""

        embedding_dim = len(query_vector)
        subvector_dim = embedding_dim // self.num_subquantizers

        total_distance_sq = 0

        for subq_idx in range(self.num_subquantizers):
            start = subq_idx * subvector_dim
            end = start + subvector_dim
            query_subvector = query_vector[start:end]

            # Get quantized subvector from codebook
            cluster_id = pq_code[subq_idx]
            quantized_subvector = self.subquantizers[subq_idx].cluster_centers_[cluster_id]

            # Add squared distance
            diff = query_subvector - quantized_subvector
            total_distance_sq += np.sum(diff ** 2)

        return np.sqrt(total_distance_sq)

    def memory_analysis(self, num_vectors, embedding_dim):
        """Compare memory usage vs uncompressed"""

        # Uncompressed
        uncompressed_bytes = num_vectors * embedding_dim * 4  # float32

        # IVF-PQ compressed
        centroid_bytes = self.num_centroids * embedding_dim * 4
        code_bytes_per_vector = self.num_subquantizers * (self.bits_per_sq // 8)
        compressed_bytes = centroid_bytes + (num_vectors * code_bytes_per_vector)

        compression_ratio = uncompressed_bytes / compressed_bytes

        return {
            'uncompressed_gb': uncompressed_bytes / (1024**3),
            'compressed_gb': compressed_bytes / (1024**3),
            'compression_ratio': f'{compression_ratio:.1f}x',
            'savings_pct': f'{(1 - compressed_bytes/uncompressed_bytes) * 100:.1f}%'
        }

# Example: 100B vectors, 768-dim
ivf_pq = IVFPQStrategy(num_centroids=65536, num_subquantizers=96, bits_per_sq=8)
memory = ivf_pq.memory_analysis(num_vectors=100_000_000_000, embedding_dim=768)

print("IVF-PQ Memory Analysis:")
print(f"  Uncompressed: {memory['uncompressed_gb']:,.0f} GB")
print(f"  Compressed: {memory['compressed_gb']:,.0f} GB")
print(f"  Compression: {memory['compression_ratio']}")
# Output: 96x compression (768 dims → 96 subquantizers × 1 byte = 96 bytes)
