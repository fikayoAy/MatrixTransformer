<!-- filepath: c:\Users\ayode\ConstantA\matrixTransfomer\README.md -->
# MatrixTransformer

A unified Python framework for structure-preserving matrix transformations in high-dimensional decision space.

> 📘 Based on the paper: **MatrixTransformer: A Unified Framework for Matrix Transformations**  
> 🔗 [Read the full paper on Zenodo](https://zenodo.org/records/15867279)  
> 📊 **NEW**: **Hyperdimensional Connection Method - A Lossless Framework Preserving Meaning, Structure, and Semantic Relationships across Modalities**  
> 🔗 [Read the hyperdimensional connections paper](https://doi.org/10.5281/zenodo.16051260)  
> 🧠 [Related project: QuantumAccel](https://github.com/fikayoAy/quantum_accel)

---

## 🧩 Overview

**MatrixTransformer** introduces a novel method for navigating between 16 matrix types (e.g., symmetric, Toeplitz, Hermitian, sparse) in a continuous, mathematically coherent space using a 16-dimensional decision hypercube.

🔹 Perform structure-preserving transformations  
🔹 Quantify information-structure trade-offs  
🔹 Interpolate between matrix types  
🔹 **NEW**: Find hyperdimensional connections between matrices  
🔹 Extendable with custom matrix definitions  
🔹 Applications in ML, signal processing, quantum simulation, and more

## 📦 Installation

### Requirements
⚠️ Ensure you are using Python 3.8+ and have NumPy, SciPy, and optionally PyTorch installed.

### Clone from github and Install from wheel file
```bash
git clone https://github.com/fikayoAy/MatrixTransformer.git
cd MatrixTransformer
pip install dist/matrixtransformer-0.1.0-py3-none-any.whl
```

### Install dependencies
```bash
pip install numpy scipy torch
```

### Verify installation
```python
import MatrixTransformer
print("MatrixTransformer installed successfully!")
```

---

## 🔧 Basic Usage

### Initialize the transformer

```python
import numpy as np
from MatrixTransformer import MatrixTransformer

# Create a transformer instance
transformer = MatrixTransformer()
```

### Transform a matrix to a specific type

```python
# Create a sample matrix
matrix = np.random.randn(4, 4)

# Transform to symmetric matrix
symmetric_matrix = transformer.process_rectangular_matrix(matrix, 'symmetric')

# Transform to positive definite
positive_def = transformer.process_rectangular_matrix(matrix, 'positive_definite')
```

### Find hyperdimensional connections between matrices

```python
from matrixtransformer import MatrixTransformer
import numpy as np

# Initialize the transformer
transformer = MatrixTransformer(dimensions=256)

# Add some sample matrices to the transformer's storage
sample_matrices = [
    np.random.randn(28, 28),  # Image-like matrix
    np.eye(10),               # Identity matrix
    np.random.randn(15, 15),  # Random square matrix
    np.random.randn(20, 30),  # Rectangular matrix
    np.diag(np.random.randn(12))  # Diagonal matrix
]

# Store matrices in the transformer
transformer.matrices = sample_matrices

# Optional: Add some metadata about the matrices
transformer.layer_info = [
    {'type': 'image', 'source': 'synthetic'},
    {'type': 'identity', 'source': 'standard'},
    {'type': 'random', 'source': 'synthetic'},
    {'type': 'rectangular', 'source': 'synthetic'},
    {'type': 'diagonal', 'source': 'synthetic'}
]

# Find hyperdimensional connections
print("Finding hyperdimensional connections...")
connections = transformer.find_hyperdimensional_connections(num_dims=8)

# Access stored matrices
print(f"\nAccessing stored matrices:")
print(f"Number of matrices stored: {len(transformer.matrices)}")
for i, matrix in enumerate(transformer.matrices):
    print(f"Matrix {i}: shape {matrix.shape}, type: {transformer._detect_matrix_type(matrix)}")

# Convert connections to matrix representation
print("\nConverting connections to matrix format...")
coords3d = []
for i, matrix in enumerate(transformer.matrices):
    coords = transformer._generate_matrix_coordinates(matrix, i)
    coords3d.append(coords)

coords3d = np.array(coords3d)
indices = list(range(len(transformer.matrices)))

# Create connection matrix with metadata
conn_matrix, metadata = transformer.connections_to_matrix(
    connections, coords3d, indices, matrix_type='general'
)

print(f"Connection matrix shape: {conn_matrix.shape}")
print(f"Matrix sparsity: {metadata.get('matrix_sparsity', 'N/A')}")
print(f"Total connections found: {metadata.get('connection_count', 'N/A')}")

# Reconstruct connections from matrix
print("\nReconstructing connections from matrix...")
reconstructed_connections = transformer.matrix_to_connections(conn_matrix, metadata)

# Compare original vs reconstructed
print(f"Original connections: {len(connections)} matrices")
print(f"Reconstructed connections: {len(reconstructed_connections)} matrices")

# Access specific matrix and its connections
matrix_idx = 0
if matrix_idx in connections:
    print(f"\nMatrix {matrix_idx} connections:")
    print(f"Original matrix shape: {transformer.matrices[matrix_idx].shape}")
    print(f"Number of connections: {len(connections[matrix_idx])}")
    
    # Show first few connections
    for i, conn in enumerate(connections[matrix_idx][:3]):
        target_idx = conn['target_idx']
        strength = conn.get('strength', 'N/A')
        print(f"  -> Connected to matrix {target_idx} (shape: {transformer.matrices[target_idx].shape}) with strength: {strength}")

# Example: Process a specific matrix through the transformer
print("\nProcessing a matrix through transformer:")
test_matrix = transformer.matrices[0]
matrix_type = transformer._detect_matrix_type(test_matrix)
print(f"Detected matrix type: {matrix_type}")

# Transform the matrix
transformed = transformer.process_rectangular_matrix(test_matrix, matrix_type)
print(f"Transformed matrix shape: {transformed.shape}")
```

### Convert between tensors and matrices

```python
# Convert a 3D tensor to a 2D matrix representation
tensor = np.random.randn(3, 4, 5)
matrix_2d, metadata = transformer.tensor_to_matrix(tensor)

# Convert back to the original tensor
reconstructed_tensor = transformer.matrix_to_tensor(matrix_2d, metadata)
```

### Combine matrices

```python
# Combine two matrices using different strategies
matrix1 = np.random.randn(3, 3)
matrix2 = np.random.randn(3, 3)

# Weighted combination
combined = transformer.combine_matrices(
    matrix1, matrix2, mode='weighted', weight1=0.6, weight2=0.4
)

# Other combination modes
max_combined = transformer.combine_matrices(matrix1, matrix2, mode='max')
multiply_combined = transformer.combine_matrices(matrix1, matrix2, mode='multiply')
```

### Add custom matrix types

```python
def custom_magic_matrix_rule(matrix):
    """Transform a matrix to have 'magic square' properties."""
    n = matrix.shape[0]
    result = matrix.copy()
    target_sum = n * (n**2 + 1) / 2
    
    # Simplified implementation for demonstration
    # (For a real implementation, you would need proper balancing logic)
    row_sums = result.sum(axis=1)
    for i in range(n):
        result[i, :] *= (target_sum / max(row_sums[i], 1e-10))
    
    return result

# Add the new transformation rule
transformer.add_transform(
    matrix_type="magic_square",
    transform_rule=custom_magic_matrix_rule,
    properties={"equal_row_col_sums": True},
    neighbors=["diagonal", "symmetric"]
)

# Now use your custom transformation
magic_matrix = transformer.process_rectangular_matrix(matrix, 'magic_square')
```

---

## 🏗️ Core Architecture

### Decision Hypercube

The distance between matrices is defined by the **decision hypercube**, which contains 16 different matrix types with properties:

```python
self.properties = [
    'symmetric', 'sparsity', 'constant_diagonal',
    'positive_eigenvalues', 'complex', 'zero_row_sum',
    'shift_invariant', 'binary',
    'diagonal_only', 'upper_triangular', 'lower_triangular',
    'nilpotent', 'idempotent', 'block_structure', 'band_limited',
    'anti_diagonal'
]
```

These various properties are represented as continuous values with a custom dynamic graph that links together all these properties and their continuous values along this hypercube space.

#### Hypercube Dimensions
- **Vertices**: If fully populated with discrete points, it could theoretically hold 2^16 = **65,536 vertices**
- **Edges**: In a fully connected 16D hypercube, each vertex connects to 16 neighbors, giving potentially **524,288+ edges**
- **Continuous Space**: Since it uses continuous values (0.0-1.0 for each dimension), the space is actually **infinite** in resolution

The entire hypercube is a 16D space that houses the 16 properties, and the decision hypercube dictates the distance between different matrix types.

### Matrix Coherence System

Coherence is computed through the `calculate_matrix_coherence(self, matrix, return_components=False)` method, which uses a coherence score (0.0 to 1.0) that measures how "well-structured" or "organized" a matrix is.

#### 1. State Coherence (for 1D vectors)
```python
if matrix_np.ndim <= 1:
    # Vector coherence based on variability
    components['state_coherence'] = 1.0 - np.std(matrix_np) / (np.mean(np.abs(matrix_np)) + 1e-10)
```

- Measures how uniform the values are in a vector
- **Formula**: `1 - (standard_deviation / mean_absolute_value)`
- Higher coherence = more uniform values
- Lower coherence = more variability

#### 2. Eigenvalue Coherence (for 2D matrices)
```python
# SVD decomposition
u, s, vh = np.linalg.svd(matrix_np, full_matrices=False)
total_variance = np.sum(s**2)

if total_variance > 0:
    # Normalized singular values
    s_normalized = s**2 / total_variance
    
    # Calculate entropy of eigenvalue distribution
    entropy = -np.sum(s_normalized * np.log2(s_normalized + 1e-10))
    max_entropy = np.log2(len(s))
    
    components['eigenvalue_coherence'] = 1.0 - entropy / (max_entropy + 1e-10)
```

- Uses **Shannon entropy** of the eigenvalue distribution
- **Lower entropy** = eigenvalues are concentrated (high coherence)
- **Higher entropy** = eigenvalues are spread out (low coherence)

#### 3. Structural Coherence (for square 2D matrices)
```python
if matrix_np.shape[0] == matrix_np.shape[1]:  # Square matrix
    # Measure how close the matrix is to being symmetric
    symmetry = np.linalg.norm(matrix_np - matrix_np.T) / (np.linalg.norm(matrix_np) + 1e-10)
    components['structural_coherence'] = 1.0 - symmetry
```

- Measures how close the matrix is to being **symmetric**
- **Formula**: `1 - ||A - A^T|| / ||A||`
- Perfect symmetry = coherence of 1.0

#### Final Coherence Calculation
The three components are combined using **weighted averaging**:

```python
overall_coherence = (
    0.4 * components['state_coherence'] +
    0.3 * components['structural_coherence'] +
    0.3 * components['eigenvalue_coherence']
)
```

**Interpretation:**
- **High coherence (0.8-1.0)**: Well-structured matrix with low variability, concentrated eigenvalues, high symmetry
- **Low coherence (0.0-0.2)**: Chaotic matrix with high variability, spread-out eigenvalues, asymmetric structure
- **Medium coherence (0.4-0.6)**: Balanced structure

### Quantum Field State

The quantum field state is a core component that maintains a dynamic quantum field state based on matrix transformations and attention mechanisms. Its purpose is to update the transformer's quantum field state to maintain coherence and stability across matrix transformations.

#### Key Components:

1. **Attention Score Analysis**: Identifies the top 3 matrix types with highest attention scores
2. **Coherence Calculation**: Adapts to matrix size for performance optimization
3. **Adaptive Time Calculation**: Creates non-linear time perception based on matrix properties
4. **Phase and Stability Updates**: Maintains temporal continuity across transformations

### Quantum Field Updates

The quantum field update mechanism provides temporal perception and adaptive feedback across the 16-dimensional state space. This transforms MatrixTransformer from a static optimization engine into an intelligent agent capable of contextual adaptation.

#### Components and Coverage

| Component | Coverage Area | Function |
|-----------|--------------|-----------|
| Dimensional Resonance | Full 16D hypercube | Matrix type projections |
| Phase Coherence | All graph edges | Transformation oscillations |
| Temporal Stability | Entire timeline | Speed/confidence regulation |

#### Implementation Details

The field update mechanism operates through three key processes:

1. **Coherence Analysis**
```python
coherence = 0.4 * C_state + 0.3 * C_structural + 0.3 * C_eigenvalue

# Where:
# C_state: Element-wise consistency (1D)
# C_structural: Geometric relationships (2D+)
# C_eigenvalue: Spectral properties (2D square)
```

2. **Temporal Perception**
```python
time_variation = (1/omega) * arctan((A * sin(omega*t + phi + theta))/r)
adapted_time = time_variation + tau

# Parameters:
theta = 1.0  # Phase angle
omega = 2.0  # Angular frequency
phi = pi/4   # Phase offset
A = 0.5      # Amplitude factor
r = 0.5      # Damping factor
tau = 1.0    # Base time constant
```

#### Key Capabilities

- **Intelligent Adaptation**: Adjusts processing speed based on matrix complexity
- **Contextual Memory**: Remembers successful strategies for similar transformations
- **Comprehensive Coverage**: Influences all 2^16 = 65,536 hypercube vertices
- **Feedback Integration**: Creates oscillation patterns for transformation quality awareness

Example usage:
```python
# Create transformer with quantum field updates enabled
transformer = MatrixTransformer(enable_quantum_field=True)

# Transform matrix with quantum field awareness
result = transformer.process_rectangular_matrix(
    matrix,
    target_type='positive_definite',
    quantum_coherence_threshold=0.8
)

# Get quantum field state
field_state = transformer.get_quantum_field_state()
print(f"Current coherence: {field_state['coherence']}")
print(f"Temporal adaptation: {field_state['adapted_time']}")
```

---

## 🎯 Advanced Features

### Hypercube decision space navigation

```python
# Find optimal transformation path between matrix types
source_type = transformer._detect_matrix_type(matrix1)
target_type = 'positive_definite'
path, attention_scores = transformer._traverse_graph(matrix1, source_type=source_type)

# Apply path-based transformation
result = matrix1.copy()
for matrix_type in path:
    transform_method = transformer._get_transform_method(matrix_type)
    if transform_method:
        result = transform_method(result)
```

### Hyperdimensional attention

```python
# Apply hyperdimensional attention for more robust transformations
query = np.random.randn(4, 4)
keys = [np.random.randn(4, 4) for _ in range(3)]
values = [np.random.randn(4, 4) for _ in range(3)]

result = transformer.hyperdimensional_attention(query, keys, values)
```

### AI Hypersphere Container

```python
# Create a hyperdimensional container for an AI entity
ai_entity = {"name": "Matrix Explorer", "capabilities": ["transform", "analyze"]}
container = transformer.create_ai_hypersphere_container(
    ai_entity, 
    dimension=8,
    base_radius=1.0
)

# Extract matrix from container
matrix = container['extract_matrix']()

# Update container state
container['update_state'](np.random.randn(8))

# Process temporal evolution of container
container['process_temporal_state']()
```

### Blended Matrix Construction

```python
# Create a blended matrix from multiple source matrices
matrix_indices = [0, 1, 2]  # Indices of matrices to blend
blend_weights = [0.5, 0.3, 0.2]  # Weights for blending

blended_matrix = transformer.blended_matrix_construction(
    source_matrices=matrix_indices,
    blend_weights=blend_weights,
    target_type='symmetric',
    preserve_properties=['energy'],
    evolution_strength=0.1
)
```

---

## 🔁 Related Projects

- [QuantumAccel](https://github.com/fikayoAy/quantum_accel): A quantum-inspired system built on MatrixTransformer's transformation logic, modeling coherence, flow dynamics, and structure-evolving computations.

---

- Visualization of results

---

## 🧠 Citation

If you use this library in your work, please cite the relevant papers:

### MatrixTransformer Framework
```bibtex
@misc{ayodele2025matrixtransformer,
  title={MatrixTransformer: A Unified Framework for Matrix Transformations},
  author={Ayodele, Fikayomi},
  year={2025},
  doi={10.5281/zenodo.15867279},
  url={https://zenodo.org/records/15867279}
}
```

### Hyperdimensional Connection Method
```bibtex
@misc{ayodele2025hyperdimensional,
  title={Hyperdimensional connection method - A Lossless Framework Preserving Meaning, Structure, and Semantic Relationships across Modalities. (A MatrixTransformer subsidiary)},
  author={Ayodele, Fikayomi},
  year={2025},
  doi={10.5281/zenodo.16051260},
  url={https://doi.org/10.5281/zenodo.16051260}
}
```

---

## 📩 Contact

Questions, suggestions, or collaboration ideas?
Open an issue or reach out via Ayodeleanjola4@gmail.com/ 2273640@swansea.ac.uk
