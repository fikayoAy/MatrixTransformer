<!-- filepath: c:\Users\ayode\ConstantA\matrixTransfomer\README.md -->
# MatrixTransformer

A unified Python framework for structure-preserving matrix transformations in high-dimensional decision space.

> 📘 Based on the paper: **MatrixTransformer: A Unified Framework for Matrix Transformations**  
> 🔗 [Read the full paper on Zenodo](https://zenodo.org/records/15867279)  
> 🧠 [Related project: QuantumAccel](https://github.com/fikayoAy/quantum_accel)

---

## 🧩 Overview

**MatrixTransformer** introduces a novel method for navigating between 16 matrix types (e.g., symmetric, Toeplitz, Hermitian, sparse) in a continuous, mathematically coherent space using a 16-dimensional decision hypercube.

🔹 Perform structure-preserving transformations  
🔹 Quantify information-structure trade-offs  
🔹 Interpolate between matrix types  
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

## 🧠 Citation

If you use this library in your work, please cite the paper:

```bibtex
@misc{ayodele2025matrixtransformer,
  title={MatrixTransformer: A Unified Framework for Matrix Transformations},
  author={Ayodele, Fikayomi},
  year={2025},
  doi={10.5281/zenodo.15867279},
  url={https://zenodo.org/records/15867279}
}
```

---

## 📩 Contact

Questions, suggestions, or collaboration ideas?
Open an issue or reach out via Ayodeleanjola4@gmail.com/ 2273640@swansea.ac.uk
