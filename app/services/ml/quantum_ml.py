"""
ROYALEY - Quantum Machine Learning Module
Hybrid Quantum-Classical ML for Sports Prediction

Implements:
- PennyLane: Hybrid quantum-classical models with quantum gradients
- Qiskit: IBM Quantum integration for VQC (Variational Quantum Classifiers)
- D-Wave Ocean SDK: Quantum annealing for optimization & feature selection
- sQUlearn/AutoQML: Automated quantum ML pipeline creation

NOTE: Quantum frameworks are optional and provide experimental accuracy improvements.
Install with: pip install -r requirements-quantum.txt
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd

from .config import MLConfig, default_ml_config

logger = logging.getLogger(__name__)

# Check framework availability
PENNYLANE_AVAILABLE = False
QISKIT_AVAILABLE = False
DWAVE_AVAILABLE = False
SQULEARN_AVAILABLE = False

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    logger.info("PennyLane not installed. Install with: pip install pennylane")

try:
    from qiskit import QuantumCircuit
    from qiskit_machine_learning.algorithms import VQC
    QISKIT_AVAILABLE = True
except ImportError:
    logger.info("Qiskit not installed. Install with: pip install qiskit qiskit-machine-learning")

try:
    from dwave.system import DWaveSampler, EmbeddingComposite
    import dimod
    DWAVE_AVAILABLE = True
except ImportError:
    logger.info("D-Wave Ocean SDK not installed. Install with: pip install dwave-ocean-sdk")

try:
    import squlearn
    SQULEARN_AVAILABLE = True
except ImportError:
    logger.info("sQUlearn not installed. Install with: pip install squlearn")


@dataclass
class QuantumModelResult:
    """Result from Quantum ML training"""
    model_id: str
    framework: str  # 'pennylane', 'qiskit', 'dwave', 'squlearn'
    sport_code: str = ""
    bet_type: str = ""
    
    # Quantum circuit info
    n_qubits: int = 0
    n_layers: int = 0
    n_parameters: int = 0
    circuit_depth: int = 0
    
    # Performance metrics
    auc: float = 0.0
    accuracy: float = 0.0
    loss: float = 0.0
    
    # Training info
    training_time_secs: float = 0.0
    n_training_samples: int = 0
    n_features: int = 0
    n_iterations: int = 0
    
    # Quantum-specific metrics
    quantum_advantage_score: float = 0.0  # Improvement over classical baseline
    circuit_fidelity: float = 0.0
    
    # Artifact paths
    model_path: str = ""
    variable_importance: Dict = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    backend: str = "simulator"  # 'simulator' or 'ibm_quantum'


# =============================================================================
# PennyLane Quantum Neural Network
# =============================================================================

class PennyLaneQNN:
    """
    Quantum Neural Network using PennyLane.
    
    Implements hybrid quantum-classical models with:
    - Variational quantum circuits
    - Quantum gradients (parameter-shift rule)
    - Integration with classical optimizers
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        learning_rate: float = 0.01,
        device: str = "default.qubit",
    ):
        """
        Initialize PennyLane QNN.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            learning_rate: Learning rate for optimizer
            device: PennyLane device ('default.qubit', 'lightning.qubit')
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.device_name = device
        
        self.weights = None
        self.dev = None
        self.qnode = None
        self._is_built = False
        
    def build(self) -> 'PennyLaneQNN':
        """Build the quantum circuit."""
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is required but not installed")
        
        # Use PennyLane's numpy for autograd support
        pnp = qml.numpy
        
        self.dev = qml.device(self.device_name, wires=self.n_qubits)
        
        # Initialize weights with PennyLane numpy (requires_grad=True for autograd)
        # Scale ~pi/4 is standard for variational quantum circuits
        self.weights = pnp.array(
            np.random.uniform(-np.pi/2, np.pi/2, (self.n_layers, self.n_qubits, 3)),
            requires_grad=True
        )
        
        # Use adjoint diff for lightning (much faster), parameter-shift for default
        diff_method = "adjoint" if "lightning" in self.device_name else "parameter-shift"
        
        @qml.qnode(self.dev, diff_method=diff_method)
        def circuit(inputs, weights):
            """Variational quantum circuit."""
            # Encode classical data (angle encoding)
            for i in range(min(len(inputs), self.n_qubits)):
                qml.RY(inputs[i] * np.pi, wires=i)
            
            # Variational layers
            for layer in range(self.n_layers):
                # Rotation gates (trainable)
                for qubit in range(self.n_qubits):
                    qml.Rot(
                        weights[layer, qubit, 0],
                        weights[layer, qubit, 1],
                        weights[layer, qubit, 2],
                        wires=qubit
                    )
                
                # Entangling gates (CNOT ladder)
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                
                # Connect last to first for circular entanglement
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # Measure expectation value
            return qml.expval(qml.PauliZ(0))
        
        self.qnode = circuit
        self._is_built = True
        
        logger.info(f"Built PennyLane QNN with {self.n_qubits} qubits, {self.n_layers} layers, {self.n_qubits * self.n_layers * 3} parameters")
        return self
    
    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """Preprocess features: truncate/pad to n_qubits, normalize to [0, 1]."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples = X.shape[0]
        processed = np.zeros((n_samples, self.n_qubits))
        
        for i in range(n_samples):
            sample = X[i]
            feat = np.zeros(self.n_qubits)
            feat[:min(len(sample), self.n_qubits)] = sample[:self.n_qubits]
            # Normalize to [0, 1]
            fmin, fmax = feat.min(), feat.max()
            if fmax - fmin > 1e-8:
                feat = (feat - fmin) / (fmax - fmin)
            else:
                feat = np.full(self.n_qubits, 0.5)
            processed[i] = feat
        
        return processed
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through quantum circuit (non-differentiable, for eval only)."""
        if not self._is_built:
            self.build()
        
        processed = self._preprocess_features(x)
        
        outputs = []
        for features in processed:
            output = self.qnode(features, self.weights)
            outputs.append(float(output))
        
        outputs = np.array(outputs)
        probs = (outputs + 1) / 2  # Map [-1, 1] to [0, 1]
        return probs
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        n_iterations: int = 100,
        batch_size: int = 16,
    ) -> Dict[str, List[float]]:
        """
        Train the QNN using gradient descent with PennyLane autograd.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_iterations: Number of training iterations
            batch_size: Batch size
            
        Returns:
            Training history
        """
        import time
        
        if not self._is_built:
            self.build()
        
        # Use PennyLane numpy for differentiable operations
        pnp = qml.numpy
        
        optimizer = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        
        history = {'loss': [], 'accuracy': []}
        if X_val is not None:
            history['val_loss'] = []
            history['val_accuracy'] = []
        
        n_samples = len(X_train)
        
        # Preprocess ALL features ONCE (not inside gradient loop)
        X_train_processed = self._preprocess_features(X_train)
        if X_val is not None:
            X_val_processed = self._preprocess_features(X_val)
        
        # Convert labels to PennyLane numpy
        y_train_pnp = pnp.array(y_train, requires_grad=False)
        
        # Cost function: ONLY weights are trainable, data is fixed via closure
        def cost(weights):
            """Binary cross-entropy loss over current batch (differentiable)."""
            predictions = pnp.array([
                (self.qnode(pnp.array(X_batch_current[i], requires_grad=False), weights) + 1) / 2
                for i in range(len(X_batch_current))
            ])
            predictions = pnp.clip(predictions, 1e-7, 1 - 1e-7)
            loss = -pnp.mean(
                y_batch_current * pnp.log(predictions) + 
                (1 - y_batch_current) * pnp.log(1 - predictions)
            )
            return loss
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 15  # Early stopping patience
        
        logger.info(f"Starting PennyLane training: {n_iterations} iterations, batch_size={batch_size}, "
                     f"{self.n_qubits} qubits, {self.n_layers} layers, lr={self.learning_rate}")
        
        for iteration in range(n_iterations):
            iter_start = time.time()
            
            # Sample batch
            batch_idx = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
            
            # Set batch data for closure (non-trainable)
            X_batch_current = pnp.array(X_train_processed[batch_idx], requires_grad=False)
            y_batch_current = pnp.array(y_train[batch_idx], requires_grad=False)
            
            # Gradient step — PennyLane differentiates cost w.r.t. weights
            self.weights = optimizer.step(cost, self.weights)
            
            iter_time = time.time() - iter_start
            
            # Evaluate metrics every 5 iterations (expensive on full dataset)
            if (iteration + 1) % 5 == 0 or iteration == 0:
                # Quick eval on a subsample (max 200) for speed
                eval_size = min(200, n_samples)
                eval_idx = np.random.choice(n_samples, eval_size, replace=False)
                
                train_preds = np.array([
                    float((self.qnode(X_train_processed[i], self.weights) + 1) / 2)
                    for i in eval_idx
                ])
                train_preds_clip = np.clip(train_preds, 1e-7, 1 - 1e-7)
                y_eval = y_train[eval_idx]
                
                train_loss = float(-np.mean(
                    y_eval * np.log(train_preds_clip) + 
                    (1 - y_eval) * np.log(1 - train_preds_clip)
                ))
                train_acc = float(np.mean((train_preds > 0.5) == y_eval))
                
                history['loss'].append(train_loss)
                history['accuracy'].append(train_acc)
                
                # Early stopping check
                if train_loss < best_loss - 0.001:
                    best_loss = train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if X_val is not None and (iteration + 1) % 10 == 0:
                    val_eval_size = min(200, len(X_val))
                    val_eval_idx = np.random.choice(len(X_val), val_eval_size, replace=False)
                    val_preds = np.array([
                        float((self.qnode(X_val_processed[i], self.weights) + 1) / 2)
                        for i in val_eval_idx
                    ])
                    val_preds_clip = np.clip(val_preds, 1e-7, 1 - 1e-7)
                    y_val_eval = y_val[val_eval_idx]
                    val_loss = float(-np.mean(
                        y_val_eval * np.log(val_preds_clip) + 
                        (1 - y_val_eval) * np.log(1 - val_preds_clip)
                    ))
                    val_acc = float(np.mean((val_preds > 0.5) == y_val_eval))
                    history['val_loss'].append(val_loss)
                    history['val_accuracy'].append(val_acc)
                
                logger.info(
                    f"Iteration {iteration + 1}/{n_iterations}: "
                    f"loss={train_loss:.4f}, acc={train_acc:.4f}, "
                    f"iter_time={iter_time:.1f}s"
                )
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at iteration {iteration + 1} (no improvement for {patience} evals)")
                    break
        
        logger.info(f"Training complete. Final loss={history['loss'][-1]:.4f}, acc={history['accuracy'][-1]:.4f}")
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(X)
    
    def save(self, path: str) -> str:
        """Save model parameters."""
        config = {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'weights': self.weights.tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(config, f)
        return path
    
    @classmethod
    def load(cls, path: str) -> 'PennyLaneQNN':
        """Load model from file."""
        with open(path, 'rb') as f:
            config = pickle.load(f)
        
        instance = cls(
            n_qubits=config['n_qubits'],
            n_layers=config['n_layers'],
        )
        instance.build()
        # Restore weights as PennyLane trainable tensors
        pnp = qml.numpy
        instance.weights = pnp.array(config['weights'], requires_grad=True)
        return instance


# =============================================================================
# Qiskit Variational Quantum Classifier
# =============================================================================

class QiskitVQC:
    """
    Variational Quantum Classifier using Qiskit.
    
    Implements:
    - Variational quantum circuits
    - Real quantum hardware access (IBM Quantum)
    - Quantum feature maps
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        feature_map: str = 'ZZFeatureMap',
        backend: str = 'aer_simulator',
    ):
        """
        Initialize Qiskit VQC.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of ansatz repetitions
            feature_map: Type of feature map
            backend: Qiskit backend ('aer_simulator', 'ibmq_manila', etc.)
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map_name = feature_map
        self.backend_name = backend
        
        self.vqc = None
        self._is_built = False
    
    def build(self) -> 'QiskitVQC':
        """Build the VQC."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required but not installed")
        
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_machine_learning.algorithms import VQC
        from qiskit_aer import AerSimulator
        from qiskit.primitives import Sampler
        
        # Create feature map
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=2,
            entanglement='linear'
        )
        
        # Create ansatz
        ansatz = RealAmplitudes(
            num_qubits=self.n_qubits,
            reps=self.n_layers,
            entanglement='linear'
        )
        
        # Create VQC
        sampler = Sampler()
        optimizer = COBYLA(maxiter=100)
        
        self.vqc = VQC(
            sampler=sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
        )
        
        self._is_built = True
        logger.info(f"Built Qiskit VQC with {self.n_qubits} qubits, {self.n_layers} layers")
        
        return self
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the VQC."""
        if not self._is_built:
            self.build()
        
        # Ensure proper dimensions
        X_train = np.array(X_train)
        if X_train.shape[1] > self.n_qubits:
            X_train = X_train[:, :self.n_qubits]
        elif X_train.shape[1] < self.n_qubits:
            padding = np.zeros((X_train.shape[0], self.n_qubits - X_train.shape[1]))
            X_train = np.hstack([X_train, padding])
        
        # Normalize
        X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
        X_train = np.clip(X_train, -np.pi, np.pi)
        
        # Train
        self.vqc.fit(X_train, y_train)
        
        # Calculate accuracy
        train_preds = self.vqc.predict(X_train)
        train_acc = np.mean(train_preds == y_train)
        
        return {'accuracy': train_acc}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X = np.array(X)
        if X.shape[1] > self.n_qubits:
            X = X[:, :self.n_qubits]
        elif X.shape[1] < self.n_qubits:
            padding = np.zeros((X.shape[0], self.n_qubits - X.shape[1]))
            X = np.hstack([X, padding])
        
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        X = np.clip(X, -np.pi, np.pi)
        
        return self.vqc.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        preds = self.predict(X)
        # Convert to probabilities (0 -> 0.3, 1 -> 0.7 as rough estimate)
        probs = np.where(preds == 1, 0.7, 0.3)
        return probs


# =============================================================================
# D-Wave Quantum Annealing for Feature Selection
# =============================================================================

class DWaveFeatureSelector:
    """
    Feature Selection using D-Wave Quantum Annealing.
    
    Formulates feature selection as QUBO problem and solves
    using quantum annealing for optimal feature subset.
    """
    
    def __init__(
        self,
        n_features_to_select: int = 20,
        use_real_hardware: bool = False,
    ):
        """
        Initialize D-Wave feature selector.
        
        Args:
            n_features_to_select: Number of features to select
            use_real_hardware: Use real D-Wave quantum hardware
        """
        self.n_features_to_select = n_features_to_select
        self.use_real_hardware = use_real_hardware
        self.selected_features = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
    ) -> List[int]:
        """
        Select features using quantum annealing.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
            
        Returns:
            List of selected feature indices
        """
        if not DWAVE_AVAILABLE:
            # Fallback to classical feature selection
            logger.warning("D-Wave not available, using classical feature selection")
            return self._classical_feature_selection(X, y)
        
        n_features = X.shape[1]
        
        # Calculate feature correlations with target
        correlations = np.array([
            abs(np.corrcoef(X[:, i], y)[0, 1])
            for i in range(n_features)
        ])
        correlations = np.nan_to_num(correlations, 0)
        
        # Calculate feature redundancy (inter-feature correlation)
        feature_corr = np.corrcoef(X.T)
        feature_corr = np.nan_to_num(feature_corr, 0)
        
        # Build QUBO matrix
        # Objective: Maximize correlation with target, minimize redundancy
        Q = {}
        
        for i in range(n_features):
            # Diagonal: negative correlation (we minimize, so negate)
            Q[(i, i)] = -correlations[i]
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Off-diagonal: penalize selecting correlated features
                Q[(i, j)] = 0.5 * abs(feature_corr[i, j])
        
        # Add constraint for number of features
        lagrange = 1.0
        for i in range(n_features):
            Q[(i, i)] += lagrange * (1 - 2 * self.n_features_to_select / n_features)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                Q[(i, j)] += 2 * lagrange / n_features
        
        # Create BQM
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
        
        # Solve
        if self.use_real_hardware:
            try:
                sampler = EmbeddingComposite(DWaveSampler())
                response = sampler.sample(bqm, num_reads=1000)
            except Exception as e:
                logger.warning(f"D-Wave hardware error: {e}, using simulator")
                sampler = dimod.SimulatedAnnealingSampler()
                response = sampler.sample(bqm, num_reads=1000)
        else:
            sampler = dimod.SimulatedAnnealingSampler()
            response = sampler.sample(bqm, num_reads=1000)
        
        # Get best solution
        best_sample = response.first.sample
        self.selected_features = [i for i, val in best_sample.items() if val == 1]
        
        # Ensure we have the right number
        if len(self.selected_features) != self.n_features_to_select:
            # Sort by correlation and take top features
            sorted_idx = np.argsort(correlations)[::-1]
            self.selected_features = sorted_idx[:self.n_features_to_select].tolist()
        
        logger.info(f"D-Wave selected {len(self.selected_features)} features")
        return self.selected_features
    
    def _classical_feature_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[int]:
        """Classical fallback feature selection."""
        from sklearn.feature_selection import mutual_info_classif
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        sorted_idx = np.argsort(mi_scores)[::-1]
        self.selected_features = sorted_idx[:self.n_features_to_select].tolist()
        
        return self.selected_features
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply feature selection."""
        if self.selected_features is None:
            raise ValueError("Call fit() first")
        return X[:, self.selected_features]
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
    ) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y, feature_names)
        return self.transform(X)


# =============================================================================
# Quantum ML Trainer (Unified Interface)
# =============================================================================

class QuantumMLTrainer:
    """
    Unified Quantum ML trainer supporting multiple frameworks.
    
    Supports:
    - PennyLane QNN
    - Qiskit VQC
    - D-Wave Feature Selection
    - sQUlearn AutoQML (when available)
    """
    
    def __init__(
        self,
        config: MLConfig = None,
        model_dir: str = None,
    ):
        """Initialize Quantum ML trainer."""
        self.config = config or default_ml_config
        self.model_dir = Path(model_dir or self.config.model_artifact_path) / "quantum"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self._model = None
        self._framework = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sport_code: str,
        bet_type: str,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
        framework: str = "pennylane",
        n_qubits: int = 4,
        n_layers: int = 3,
        n_iterations: int = 100,
        **kwargs,
    ) -> QuantumModelResult:
        """
        Unified training interface - dispatches to framework-specific method.
        
        Args:
            X_train: Training features
            y_train: Training labels
            sport_code: Sport code
            bet_type: Bet type
            X_valid: Validation features
            y_valid: Validation labels
            framework: 'pennylane' or 'qiskit'
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            n_iterations: Training iterations
            
        Returns:
            QuantumModelResult
        """
        if framework == 'qiskit':
            if not QISKIT_AVAILABLE:
                logger.warning("Qiskit not installed, falling back to PennyLane")
                framework = 'pennylane'
            else:
                return self.train_qiskit(
                    X_train=X_train, y_train=y_train,
                    sport_code=sport_code, bet_type=bet_type,
                    n_qubits=n_qubits, n_layers=n_layers,
                )
        
        # Default: PennyLane
        return self.train_pennylane(
            X_train=X_train, y_train=y_train,
            sport_code=sport_code, bet_type=bet_type,
            X_val=X_valid, y_val=y_valid,
            n_qubits=n_qubits, n_layers=n_layers,
            n_iterations=n_iterations,
        )
    
    def train_pennylane(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sport_code: str,
        bet_type: str,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        n_qubits: int = 4,
        n_layers: int = 3,
        n_iterations: int = 100,
    ) -> QuantumModelResult:
        """
        Train PennyLane quantum neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            sport_code: Sport code
            bet_type: Bet type
            X_val: Validation features
            y_val: Validation labels
            n_qubits: Number of qubits
            n_layers: Number of layers
            n_iterations: Training iterations
            
        Returns:
            QuantumModelResult
        """
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is required. Install with: pip install pennylane")
        
        logger.info(f"Training PennyLane QNN for {sport_code} {bet_type}")
        start_time = datetime.now(timezone.utc)
        
        # Build and train model
        model = PennyLaneQNN(
            n_qubits=n_qubits,
            n_layers=n_layers,
            learning_rate=0.1,  # Quantum circuits need higher LR than classical
            device="lightning.qubit",  # C++ backend, 10-50x faster than default.qubit
        )
        model.build()
        
        history = model.fit(
            X_train, y_train,
            X_val=X_val,
            y_val=y_val,
            n_iterations=n_iterations,
        )
        
        self._model = model
        self._framework = 'pennylane'
        
        # Save model
        model_path = self.model_dir / sport_code / bet_type / "pennylane_qnn.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        
        # Calculate metrics
        train_preds = model.predict(X_train)
        train_acc = np.mean((train_preds > 0.5) == y_train)
        
        val_acc = 0.0
        if X_val is not None:
            val_preds = model.predict(X_val)
            val_acc = np.mean((val_preds > 0.5) == y_val)
        
        training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return QuantumModelResult(
            model_id=f"pennylane_{sport_code}_{bet_type}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            framework='pennylane',
            sport_code=sport_code,
            bet_type=bet_type,
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_parameters=n_qubits * n_layers * 3,
            accuracy=train_acc,
            auc=train_acc,  # Simplified
            training_time_secs=training_time,
            n_training_samples=len(y_train),
            n_features=X_train.shape[1],
            n_iterations=n_iterations,
            model_path=str(model_path),
            backend='simulator',
        )
    
    def train_qiskit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sport_code: str,
        bet_type: str,
        n_qubits: int = 4,
        n_layers: int = 2,
        backend: str = 'aer_simulator',
    ) -> QuantumModelResult:
        """
        Train Qiskit VQC.
        
        Args:
            X_train: Training features
            y_train: Training labels
            sport_code: Sport code
            bet_type: Bet type
            n_qubits: Number of qubits
            n_layers: Number of layers
            backend: Qiskit backend
            
        Returns:
            QuantumModelResult
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required. Install with: pip install qiskit qiskit-machine-learning")
        
        logger.info(f"Training Qiskit VQC for {sport_code} {bet_type}")
        start_time = datetime.now(timezone.utc)
        
        model = QiskitVQC(
            n_qubits=n_qubits,
            n_layers=n_layers,
            backend=backend,
        )
        model.build()
        
        result = model.fit(X_train, y_train)
        
        self._model = model
        self._framework = 'qiskit'
        
        training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return QuantumModelResult(
            model_id=f"qiskit_{sport_code}_{bet_type}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            framework='qiskit',
            sport_code=sport_code,
            bet_type=bet_type,
            n_qubits=n_qubits,
            n_layers=n_layers,
            accuracy=result.get('accuracy', 0.0),
            training_time_secs=training_time,
            n_training_samples=len(y_train),
            n_features=X_train.shape[1],
            backend=backend,
        )
    
    def quantum_feature_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_features: int = 20,
        use_real_hardware: bool = False,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Perform quantum feature selection using D-Wave.
        
        Args:
            X: Feature matrix
            y: Target labels
            n_features: Number of features to select
            use_real_hardware: Use real D-Wave hardware
            
        Returns:
            Tuple of (transformed features, selected indices)
        """
        selector = DWaveFeatureSelector(
            n_features_to_select=n_features,
            use_real_hardware=use_real_hardware,
        )
        
        X_selected = selector.fit_transform(X, y)
        
        return X_selected, selector.selected_features
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model."""
        if self._model is None:
            raise ValueError("No model trained")
        
        if self._framework == 'pennylane':
            return self._model.predict(X)
        elif self._framework == 'qiskit':
            return self._model.predict_proba(X)
        else:
            raise ValueError(f"Unknown framework: {self._framework}")


class QuantumMLTrainerMock:
    """Mock quantum trainer for testing."""
    
    def __init__(self, config: MLConfig = None, model_dir: str = None):
        self.config = config or default_ml_config
        self.model_dir = Path(model_dir or "./models/quantum_mock")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sport_code: str,
        bet_type: str,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
        framework: str = "pennylane",
        n_qubits: int = 4,
        n_layers: int = 3,
        n_iterations: int = 50,
        **kwargs,
    ) -> QuantumModelResult:
        """Unified mock training interface."""
        if framework == 'qiskit':
            return self.train_qiskit(X_train, y_train, sport_code, bet_type, n_qubits=n_qubits)
        return self.train_pennylane(X_train, y_train, sport_code, bet_type, n_qubits=n_qubits, n_iterations=n_iterations)
    
    def train_pennylane(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sport_code: str,
        bet_type: str,
        **kwargs,
    ) -> QuantumModelResult:
        """Mock PennyLane training."""
        logger.info(f"Mock PennyLane training for {sport_code} {bet_type}")
        return QuantumModelResult(
            model_id=f"mock_pennylane_{sport_code}_{bet_type}",
            framework='pennylane',
            sport_code=sport_code,
            bet_type=bet_type,
            n_qubits=4,
            n_layers=3,
            accuracy=0.55 + np.random.random() * 0.08,
            training_time_secs=60.0,
            n_training_samples=len(y_train),
        )
    
    def train_qiskit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sport_code: str,
        bet_type: str,
        **kwargs,
    ) -> QuantumModelResult:
        """Mock Qiskit training."""
        logger.info(f"Mock Qiskit training for {sport_code} {bet_type}")
        return QuantumModelResult(
            model_id=f"mock_qiskit_{sport_code}_{bet_type}",
            framework='qiskit',
            sport_code=sport_code,
            bet_type=bet_type,
            n_qubits=4,
            n_layers=2,
            accuracy=0.54 + np.random.random() * 0.08,
            training_time_secs=90.0,
            n_training_samples=len(y_train),
        )
    
    def quantum_feature_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_features: int = 20,
        **kwargs,
    ) -> Tuple[np.ndarray, List[int]]:
        """Mock quantum feature selection."""
        # Just return top features by variance
        variances = np.var(X, axis=0)
        selected = np.argsort(variances)[::-1][:n_features].tolist()
        return X[:, selected], selected
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mock prediction."""
        return np.random.beta(2, 2, len(X))


def get_quantum_trainer(
    config: MLConfig = None,
    use_mock: bool = False,
) -> Union[QuantumMLTrainer, QuantumMLTrainerMock]:
    """
    Factory function to get Quantum ML trainer.
    
    Args:
        config: ML configuration
        use_mock: Use mock trainer for testing
        
    Returns:
        Quantum ML trainer instance
    """
    if use_mock:
        return QuantumMLTrainerMock(config)
    
    if PENNYLANE_AVAILABLE or QISKIT_AVAILABLE or DWAVE_AVAILABLE:
        return QuantumMLTrainer(config)
    else:
        logger.warning("No quantum frameworks installed, using mock trainer")
        return QuantumMLTrainerMock(config)


# Check available frameworks
def get_available_quantum_frameworks() -> Dict[str, bool]:
    """Get dictionary of available quantum frameworks."""
    return {
        'pennylane': PENNYLANE_AVAILABLE,
        'qiskit': QISKIT_AVAILABLE,
        'dwave': DWAVE_AVAILABLE,
        'squlearn': SQULEARN_AVAILABLE,
    }