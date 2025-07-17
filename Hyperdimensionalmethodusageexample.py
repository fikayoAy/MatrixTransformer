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