import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import random
import logging

class BiologicalSubstrate:
    def __init__(self, dim=512, sparsity=0.05, spectral_radius=0.9):
        """Initialize biological substrate with memory-efficient sparse matrices"""
        self.dim = dim
        self.state = np.zeros(dim)
        self.recovery_state = np.zeros(dim)
        self.instability_detected = False
        
        # Create sparse connectivity matrix
        self._create_sparse_connectivity(sparsity, spectral_radius)
        
        # Initialize sparse memory matrix
        self.memory_matrix = sp.lil_matrix((dim, dim))
        
        # Homeostatic drives
        self.drives = {
            "fatigue": 0.0,
            "social": 0.5,
            "curiosity": 0.7,
            "stability": 0.9
        }
        
        # Performance tracking
        self.state_history = []
        self.max_history_length = 100
        
        logging.info(f"Initialized BiologicalSubstrate with dim={dim}, sparsity={sparsity}")
    
    def _create_sparse_connectivity(self, sparsity, spectral_radius):
        """Create sparse connectivity matrix with controlled spectral radius"""
        try:
            # Generate sparse matrix with controlled sparsity
            n_connections = int(self.dim * self.dim * sparsity)
            rows = np.random.randint(0, self.dim, n_connections)
            cols = np.random.randint(0, self.dim, n_connections)
            values = np.random.randn(n_connections) * 0.1
            
            self.connectivity = sp.csr_matrix((values, (rows, cols)), shape=(self.dim, self.dim))
            
            # Normalize spectral radius for stability
            try:
                eigenvalue, _ = eigs(self.connectivity, k=1, which='LM')
                current_radius = np.abs(eigenvalue[0])
                if current_radius > 0:
                    self.connectivity *= (spectral_radius / current_radius)
            except:
                # Fallback: use simple scaling
                self.connectivity *= 0.1
                
        except Exception as e:
            logging.error(f"Error creating connectivity matrix: {e}")
            # Create identity matrix as fallback
            self.connectivity = sp.eye(self.dim) * 0.1
    
    def evolve(self, input_embedding):
        """Evolve substrate state with extensive safeguards"""
        self.recovery_state = self.state.copy()
        
        try:
            # Compute recurrent dynamics
            recurrent = self.connectivity.dot(self.state)
            
            # Compute memory influence
            memory_influence = self.memory_matrix.dot(self.state)
            if sp.issparse(memory_influence):
                memory_influence = memory_influence.toarray().flatten()
            
            # Update state with balanced dynamics
            self.state = (
                0.85 * self.state +
                0.1 * input_embedding +
                0.03 * recurrent +
                0.02 * np.tanh(memory_influence)
            )
            
            # Apply activation function
            self.state = np.tanh(self.state)
            
            # Add minimal noise for biological realism
            self.state += np.random.randn(self.dim) * 0.005
            
            # Stability checks
            if self._check_stability():
                self.instability_detected = True
                self.state = self.recovery_state
                self._normalize_state()
            
            # Update drives
            self._update_drives()
            
            # Track state history
            self._track_state()
            
            return self.state.copy()
            
        except Exception as e:
            logging.error(f"Evolution error: {e}")
            self.instability_detected = True
            self.state = self.recovery_state
            return self.state
    
    def _check_stability(self):
        """Check for numerical instability"""
        return (
            np.any(np.isnan(self.state)) or
            np.any(np.isinf(self.state)) or
            np.linalg.norm(self.state) > 10.0
        )
    
    def _normalize_state(self):
        """Normalize state if it becomes too large"""
        state_norm = np.linalg.norm(self.state)
        if state_norm > 5.0:
            self.state = self.state / (state_norm + 1e-10) * 2.0
        elif state_norm < 0.1:
            self.state = self.state / (state_norm + 1e-10) * 0.5
    
    def _update_drives(self):
        """Update homeostatic drives"""
        self.drives["fatigue"] = min(1.0, self.drives["fatigue"] + 0.0005)
        self.drives["curiosity"] = max(0.1, self.drives["curiosity"] - 0.0001)
        self.drives["stability"] = max(0.5, min(1.0, 1.0 - self.drives["fatigue"] * 0.5))
    
    def _track_state(self):
        """Track state history for analysis"""
        self.state_history.append({
            'timestamp': len(self.state_history),
            'state_norm': np.linalg.norm(self.state),
            'drives': self.drives.copy()
        })
        
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)
    
    def update_memory(self, experience_vector, strength=0.0005):
        """Update memory matrix with sparse updates"""
        try:
            # Only update significant connections
            significant_indices = np.where(np.abs(experience_vector) > 0.1)[0]
            
            if len(significant_indices) > 0:
                # Create sparse update
                rows, cols, data = [], [], []
                for i in significant_indices:
                    for j in significant_indices:
                        if i != j:  # Avoid self-connections
                            rows.append(i)
                            cols.append(j)
                            data.append(experience_vector[i] * experience_vector[j] * strength)
                
                # Apply update
                update_matrix = sp.csr_matrix((data, (rows, cols)), shape=(self.dim, self.dim))
                self.memory_matrix = self.memory_matrix + update_matrix
            
            # Apply memory decay
            self.memory_matrix = self.memory_matrix * 0.999
            
            # Clip extreme values
            self.memory_matrix.data = np.clip(self.memory_matrix.data, -0.5, 0.5)
            
            # Prune near-zero elements
            self.memory_matrix.data[np.abs(self.memory_matrix.data) < 0.001] = 0
            self.memory_matrix.eliminate_zeros()
            
        except Exception as e:
            logging.error(f"Memory update error: {e}")
    
    def get_state_summary(self):
        """Get summary of current state for monitoring"""
        return {
            'state_norm': np.linalg.norm(self.state),
            'sparsity': len(self.memory_matrix.data) / (self.dim * self.dim),
            'drives': self.drives.copy(),
            'instability_detected': self.instability_detected
        }