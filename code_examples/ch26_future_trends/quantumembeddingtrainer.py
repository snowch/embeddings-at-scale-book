# Code from Chapter 26
# Book: Embeddings at Scale

class QuantumEmbeddingTrainer:
    """
    Variational quantum algorithm for embedding training
    
    Uses parameterized quantum circuits as feature extractors,
    trained with classical optimization of circuit parameters
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_qubits: int,
        num_layers: int
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.params = self._initialize_parameters()
    
    def _initialize_parameters(self) -> np.ndarray:
        """Initialize quantum circuit parameters"""
        # Parameters for rotation gates (RX, RY, RZ) in each layer
        num_params = self.num_qubits * 3 * self.num_layers
        return np.random.randn(num_params) * 0.1
    
    def quantum_circuit(
        self,
        x: np.ndarray,
        params: np.ndarray
    ) -> np.ndarray:
        """
        Parameterized quantum circuit
        
        Architecture:
        1. Data encoding: Encode input x in quantum state
        2. Variational layers: Parameterized rotations + entanglement
        3. Measurement: Extract embedding from quantum state
        
        Real implementation would use Qiskit/PennyLane:
        - Amplitude encoding for input
        - Ansatz: Hardware-efficient or problem-specific
        - Measurement: Expectation values of observables
        """
        # Encode input (amplitude encoding)
        # |ψ⟩ = Σᵢ xᵢ|i⟩
        state = x / (np.linalg.norm(x) + 1e-10)
        
        # Apply variational layers (simulated)
        param_idx = 0
        for layer in range(self.num_layers):
            # Rotation gates
            for qubit in range(self.num_qubits):
                if qubit < len(state):
                    # RX, RY, RZ rotations (simulated effect)
                    rx = params[param_idx]
                    ry = params[param_idx + 1]
                    rz = params[param_idx + 2]
                    
                    # Simple simulation of rotation effect
                    state[qubit] *= np.cos(rx/2) * np.cos(ry/2) * np.cos(rz/2)
                    param_idx += 3
            
            # Entanglement (CNOT gates) - simulated as correlation
            if len(state) > 1:
                for i in range(0, len(state)-1, 2):
                    # CNOT effect approximation
                    state[i] = 0.7 * state[i] + 0.3 * state[i+1]
                    state[i+1] = 0.3 * state[i] + 0.7 * state[i+1]
        
        # Measurement: extract embedding
        embedding = state[:self.output_dim]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        
        return embedding
    
    def train_step(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        learning_rate: float = 0.01
    ) -> float:
        """
        Training step using parameter-shift rule for gradients
        
        Quantum gradients computed via parameter-shift rule:
        ∂f/∂θᵢ = [f(θ + π/2 eᵢ) - f(θ - π/2 eᵢ)] / 2
        
        This requires 2 quantum circuit evaluations per parameter
        """
        batch_size = len(x_batch)
        
        # Forward pass
        embeddings = np.array([
            self.quantum_circuit(x, self.params)
            for x in x_batch
        ])
        
        # Compute loss (contrastive or supervised)
        loss = self._compute_loss(embeddings, y_batch)
        
        # Compute gradients via parameter-shift rule
        gradients = np.zeros_like(self.params)
        shift = np.pi / 2
        
        for i in range(len(self.params)):
            # Shift parameter up
            params_plus = self.params.copy()
            params_plus[i] += shift
            embeddings_plus = np.array([
                self.quantum_circuit(x, params_plus)
                for x in x_batch
            ])
            loss_plus = self._compute_loss(embeddings_plus, y_batch)
            
            # Shift parameter down
            params_minus = self.params.copy()
            params_minus[i] -= shift
            embeddings_minus = np.array([
                self.quantum_circuit(x, params_minus)
                for x in x_batch
            ])
            loss_minus = self._compute_loss(embeddings_minus, y_batch)
            
            # Gradient via parameter-shift rule
            gradients[i] = (loss_plus - loss_minus) / 2
        
        # Update parameters
        self.params -= learning_rate * gradients
        
        return loss
    
    def _compute_loss(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute contrastive loss"""
        # Simplified contrastive loss
        batch_size = len(embeddings)
        loss = 0
        
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                similarity = np.dot(embeddings[i], embeddings[j])
                
                if labels[i] == labels[j]:
                    # Similar: maximize similarity
                    loss += (1 - similarity) ** 2
                else:
                    # Dissimilar: minimize similarity
                    loss += max(0, similarity) ** 2
        
        return loss / (batch_size * (batch_size - 1) / 2)

# Example: Quantum kernel for similarity computation
class QuantumKernel:
    """
    Quantum kernel for embedding similarity
    
    Uses quantum feature maps to compute inner products
    in high-dimensional Hilbert space
    """
    
    def __init__(self, num_qubits: int, num_layers: int):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
    
    def feature_map(self, x: np.ndarray) -> np.ndarray:
        """Quantum feature map |φ(x)⟩"""
        # ZZ feature map: U(x) = Π exp(-i(π - xᵢ)(π - xⱼ)ZᵢZⱼ)
        # Creates entangled quantum state encoding input
        
        # Simplified simulation
        phi = x.copy()
        for layer in range(self.num_layers):
            # Nonlinear transformation
            phi = np.sin(phi * np.pi)
            # Entanglement effect
            phi = np.fft.fft(phi).real
        
        return phi / (np.linalg.norm(phi) + 1e-10)
    
    def kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Quantum kernel: K(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|²
        
        Computed by preparing states and measuring overlap
        """
        phi1 = self.feature_map(x1)
        phi2 = self.feature_map(x2)
        
        # Inner product in feature space
        overlap = np.abs(np.dot(phi1, phi2))
        
        return overlap ** 2
    
    def kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute kernel matrix for all pairs"""
        n = len(X)
        K = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                K[i, j] = self.kernel(X[i], X[j])
                K[j, i] = K[i, j]
        
        return K
