import numpy as np
import time
from Matrixtransfomrer import MatrixTransformer
import torch
import matplotlib.pyplot as plt

def benchmark_matrix_detection():
    """
    A minimal benchmark to test the accuracy of MatrixTransformer's _detect_matrix_type method.
    Creates matrices of various types and tests if they're correctly identified.
    """
    print("Starting Matrix Type Detection Benchmark")
    print("----------------------------------------\n")
    
    # Initialize transformer
    transformer = MatrixTransformer()
    
    # Define matrix generators for different types
    matrix_generators = {
        'diagonal': lambda n: np.diag(np.random.rand(n)),
        'symmetric': lambda n: generate_symmetric_matrix(n),
        'upper_triangular': lambda n: np.triu(np.random.rand(n, n)),
        'lower_triangular': lambda n: np.tril(np.random.rand(n, n)),
        'hermitian': lambda n: generate_hermitian_matrix(n),
        'toeplitz': lambda n: generate_toeplitz_matrix(n),
        'circulant': lambda n: generate_circulant_matrix(n),
        'positive_definite': lambda n: generate_positive_definite_matrix(n),
        'sparse': lambda n: generate_sparse_matrix(n),
        'nilpotent': lambda n: generate_nilpotent_matrix(n),
        'idempotent': lambda n: generate_idempotent_matrix(n),
        'hankel': lambda n: generate_hankel_matrix(n),
        'banded': lambda n: generate_banded_matrix(n),
        'block': lambda n: generate_block_matrix(n),
        'laplacian': lambda n: generate_laplacian_matrix(n),
        'adjacency': lambda n: generate_adjacency_matrix(n),
        'general': lambda n: np.random.rand(n, n)
    }
    
    # Parameters
    matrix_sizes = [10, 20, 30]  # Different matrix sizes
    repetitions = 5  # Test each type multiple times
    
    # Store results
    results = {}
    timing = {}
    
    # Run benchmark
    print(f"Testing {len(matrix_generators)} matrix types with sizes {matrix_sizes}")
    print(f"Each test repeated {repetitions} times\n")
    
    for matrix_type, generator in matrix_generators.items():
        correct_detections = 0
        total_tests = 0
        detection_times = []
        incorrect_detections = {}
        
        print(f"Testing {matrix_type} matrices...")
        
        for size in matrix_sizes:
            for _ in range(repetitions):
                # Generate matrix of this type
                matrix = generator(size)
                
                # Time the detection
                start_time = time.time()
                detected_type = transformer._detect_matrix_type(matrix)
                end_time = time.time()
                
                detection_time = (end_time - start_time) * 1000  # Convert to ms
                detection_times.append(detection_time)
                
                # Check if detection was correct
                total_tests += 1
                if detected_type == matrix_type:
                    correct_detections += 1
                else:
                    # Track incorrect detections
                    if detected_type not in incorrect_detections:
                        incorrect_detections[detected_type] = 0
                    incorrect_detections[detected_type] += 1
        
        # Calculate accuracy and average detection time
        accuracy = correct_detections / total_tests * 100
        avg_detection_time = sum(detection_times) / len(detection_times)
        
        # Store results
        results[matrix_type] = {
            'accuracy': accuracy,
            'incorrect_detections': incorrect_detections,
            'total_tests': total_tests
        }
        
        timing[matrix_type] = avg_detection_time
        
        # Print results for this type
        print(f"  Accuracy: {accuracy:.1f}% ({correct_detections}/{total_tests})")
        print(f"  Avg detection time: {avg_detection_time:.2f} ms")
        if incorrect_detections:
            print("  Misclassified as:")
            for detected, count in incorrect_detections.items():
                print(f"    - {detected}: {count} times")
        print()
    
    # Calculate overall accuracy
    total_correct = sum(result['accuracy'] * result['total_tests'] / 100 for result in results.values())
    total_tests = sum(result['total_tests'] for result in results.values())
    overall_accuracy = total_correct / total_tests * 100
    
    print(f"Overall accuracy: {overall_accuracy:.1f}%")
    print(f"Average detection time: {sum(timing.values()) / len(timing):.2f} ms")
    
    # Plot results
    plot_results(results, timing)
    
    return results, timing

def plot_results(results, timing):
    """Plot benchmark results."""
    # Sort types by accuracy
    sorted_types = sorted(results.keys(), key=lambda t: results[t]['accuracy'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Accuracy plot
    accuracies = [results[t]['accuracy'] for t in sorted_types]
    bars = ax1.bar(sorted_types, accuracies, color='skyblue')
    ax1.set_title('Matrix Type Detection Accuracy')
    ax1.set_xlabel('Matrix Type')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim([0, 105])
    ax1.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='90% Threshold')
    
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', rotation=0)
    
    # Timing plot
    timing_values = [timing[t] for t in sorted_types]
    ax2.bar(sorted_types, timing_values, color='lightgreen')
    ax2.set_title('Matrix Type Detection Time')
    ax2.set_xlabel('Matrix Type')
    ax2.set_ylabel('Average Detection Time (ms)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('matrix_detection_benchmark.png')
    plt.close()

# Matrix generation functions
def generate_symmetric_matrix(n):
    """Generate a random symmetric matrix."""
    A = np.random.rand(n, n)
    return 0.5 * (A + A.T)

def generate_hermitian_matrix(n):
    """Generate a random Hermitian matrix."""
    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    return 0.5 * (A + A.conj().T)

def generate_toeplitz_matrix(n):
    """Generate a random Toeplitz matrix."""
    first_row = np.random.rand(n)
    first_col = np.random.rand(n)
    first_col[0] = first_row[0]  # Ensure consistency
    
    toeplitz = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i <= j:
                toeplitz[i, j] = first_row[j-i]
            else:
                toeplitz[i, j] = first_col[i-j]
    return toeplitz

def generate_circulant_matrix(n):
    """Generate a random circulant matrix."""
    first_row = np.random.rand(n)
    circulant = np.zeros((n, n))
    for i in range(n):
        circulant[i] = np.roll(first_row, i)
    return circulant

def generate_positive_definite_matrix(n):
    """Generate a random positive definite matrix."""
    A = np.random.rand(n, n)
    return A @ A.T + n * np.eye(n)  # Ensure it's positive definite

def generate_sparse_matrix(n):
    """Generate a random sparse matrix with ~90% zeros."""
    A = np.random.rand(n, n)
    mask = np.random.rand(n, n) < 0.9
    A[mask] = 0
    return A

def generate_nilpotent_matrix(n):
    """Generate a nilpotent matrix."""
    A = np.zeros((n, n))
    for i in range(n-1):
        A[i, i+1] = np.random.rand()
    return A

def generate_idempotent_matrix(n):
    """Generate an idempotent matrix."""
    rank = max(1, n//2)
    A = np.random.rand(n, rank)
    P = A @ np.linalg.pinv(A)  # Projection matrix is idempotent
    return P

def generate_hankel_matrix(n):
    """Generate a Hankel matrix."""
    values = np.random.rand(2*n - 1)
    hankel = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            hankel[i, j] = values[i + j]
    return hankel

def generate_banded_matrix(n):
    """Generate a banded matrix with bandwidth 2."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i-2), min(n, i+3)):
            A[i, j] = np.random.rand()
    return A

def generate_block_matrix(n):
    """Generate a block matrix."""
    if n < 4:
        return np.eye(n)
    
    block_size = n // 2
    block1 = np.random.rand(block_size, block_size)
    block2 = np.random.rand(block_size, block_size)
    
    result = np.zeros((n, n))
    result[:block_size, :block_size] = block1
    result[block_size:, block_size:] = block2
    return result

def generate_laplacian_matrix(n):
    """Generate a Laplacian matrix."""
    # Start with adjacency matrix
    A = generate_adjacency_matrix(n)
    # Create degree matrix
    D = np.diag(np.sum(A, axis=1))
    # Laplacian = D - A
    return D - A

def generate_adjacency_matrix(n):
    """Generate an adjacency matrix."""
    A = np.zeros((n, n))
    # Generate random edges
    for i in range(n):
        for j in range(i+1, n):
            if np.random.rand() < 0.5:  # 50% chance of edge
                A[i, j] = A[j, i] = 1
    return A

if __name__ == "__main__":
    benchmark_matrix_detection()