# Code from Chapter 26
# Book: Embeddings at Scale

@dataclass
class QuantumAnnealingConfig:
    """Configuration for quantum annealing"""
    annealer_type: str = "dwave"  # "dwave", "simulator"
    num_reads: int = 1000
    annealing_time: int = 20  # microseconds
    chain_strength: float = 1.0
    auto_scale: bool = True

class QuantumEmbeddingClustering:
    """
    Quantum annealing for embedding clustering
    
    Formulates k-means clustering as QUBO problem:
    minimize: Σᵢⱼ (distance(xᵢ, xⱼ) * same_cluster(i,j))
    subject to: each point assigned to exactly one cluster
    """
    
    def __init__(self, config: QuantumAnnealingConfig):
        self.config = config
        
    def cluster(
        self,
        embeddings: np.ndarray,
        k: int
    ) -> Dict[str, Any]:
        """
        Quantum annealing-based clustering
        
        Steps:
        1. Compute pairwise distances
        2. Formulate as QUBO problem
        3. Submit to quantum annealer
        4. Decode quantum solution to cluster assignments
        5. Refine with classical k-means
        """
        n = len(embeddings)
        
        # Compute distance matrix (sample for large N)
        if n > 1000:
            sample_idx = np.random.choice(n, 1000, replace=False)
            sample_embeddings = embeddings[sample_idx]
        else:
            sample_idx = np.arange(n)
            sample_embeddings = embeddings
        
        distances = self._compute_distances(sample_embeddings)
        
        # Formulate QUBO
        qubo = self._distances_to_qubo(distances, k)
        
        # Solve with quantum annealing (simulated)
        quantum_solution = self._solve_qubo(qubo)
        
        # Decode to cluster assignments
        cluster_assignments = self._decode_clustering(quantum_solution, k)
        
        # Refine with classical k-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, init='k-means++')
        final_assignments = kmeans.fit_predict(embeddings)
        
        return {
            'cluster_assignments': final_assignments,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'quantum_solution': quantum_solution
        }
    
    def _compute_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances"""
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(embeddings, metric='euclidean'))
        return distances
    
    def _distances_to_qubo(
        self,
        distances: np.ndarray,
        k: int
    ) -> Dict[Tuple[int, int], float]:
        """
        Convert clustering problem to QUBO formulation
        
        Variables: x_{i,c} = 1 if point i assigned to cluster c
        Objective: minimize Σᵢⱼ Σc distance(i,j) * x_{i,c} * x_{j,c}
        Constraints: Σc x_{i,c} = 1 (each point in exactly one cluster)
        """
        n = len(distances)
        qubo = {}
        
        # Objective: minimize intra-cluster distances
        for i in range(n):
            for j in range(i+1, n):
                for c in range(k):
                    var_i = (i, c)
                    var_j = (j, c)
                    qubo[(var_i, var_j)] = distances[i, j]
        
        # Constraint: each point in exactly one cluster
        # Penalty term: P * (Σc x_{i,c} - 1)²
        penalty = np.max(distances) * 2
        for i in range(n):
            for c1 in range(k):
                var1 = (i, c1)
                # Linear term: -2P * x_{i,c1}
                qubo[(var1, var1)] = qubo.get((var1, var1), 0) - 2 * penalty
                
                # Quadratic term: P * x_{i,c1} * x_{i,c2}
                for c2 in range(c1+1, k):
                    var2 = (i, c2)
                    qubo[(var1, var2)] = qubo.get((var1, var2), 0) + 2 * penalty
            
            # Constant term (omitted as doesn't affect optimization)
        
        return qubo
    
    def _solve_qubo(
        self,
        qubo: Dict[Tuple[int, int], float]
    ) -> Dict[int, int]:
        """
        Solve QUBO using quantum annealing (simulated)
        
        Real implementation would use D-Wave Ocean SDK:
        from dwave.system import DWaveSampler, EmbeddingComposite
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample_qubo(qubo, num_reads=self.config.num_reads)
        """
        # Simulated annealing as approximation
        from scipy.optimize import dual_annealing
        
        # Convert QUBO to array form
        variables = sorted(set([v for pair in qubo.keys() for v in pair]))
        var_to_idx = {v: i for i, v in enumerate(variables)}
        n_vars = len(variables)
        
        def objective(x):
            # Binary constraint
            x_binary = (x > 0.5).astype(int)
            energy = 0
            for (v1, v2), coeff in qubo.items():
                i1, i2 = var_to_idx[v1], var_to_idx[v2]
                energy += coeff * x_binary[i1] * x_binary[i2]
            return energy
        
        # Optimize
        bounds = [(0, 1)] * n_vars
        result = dual_annealing(objective, bounds, maxiter=1000)
        
        # Convert to binary solution
        solution = {}
        for var, idx in var_to_idx.items():
            solution[var] = int(result.x[idx] > 0.5)
        
        return solution
    
    def _decode_clustering(
        self,
        solution: Dict[int, int],
        k: int
    ) -> np.ndarray:
        """Decode binary variables to cluster assignments"""
        # Extract assignments from x_{i,c} variables
        point_to_cluster = {}
        
        for (i, c), value in solution.items():
            if value == 1:
                if i not in point_to_cluster:
                    point_to_cluster[i] = c
        
        # Convert to array
        n_points = max(point_to_cluster.keys()) + 1
        assignments = np.zeros(n_points, dtype=int)
        for i, c in point_to_cluster.items():
            assignments[i] = c
        
        return assignments
