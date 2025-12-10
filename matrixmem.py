"""matrixmem.py

Cache system for GraphMatrixTransformer to improve temporal coherence and performance.
"""

import numpy as np


class MatrixMemoryCache:
    """Cache system for GraphMatrixTransformer to improve temporal coherence and performance."""
    
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.input_output_pairs = []  # Store recent transformations
        self.transformation_stats = {}  # Statistics on transformation effectiveness
        self.channel_memory = {}  # Store per-channel information for images
        self.temporal_sequence = []  # Store sequence of related transformations
    
    def store_transformation(self, input_matrix, output_matrix, matrix_type, time_pos, metrics=None):
        """Store a transformation result with metadata"""
        self.input_output_pairs.append({
            'input_hash': self._matrix_hash(input_matrix),
            'input_snippet': self._get_matrix_snippet(input_matrix),
            'output_snippet': self._get_matrix_snippet(output_matrix),
            'matrix_type': matrix_type,
            'time': time_pos,
            'metrics': metrics or {}
        })
        
        # Prune if needed
        if len(self.input_output_pairs) > self.max_size:
            self.input_output_pairs.pop(0)
            
        # Update transformation statistics
        if matrix_type not in self.transformation_stats:
            self.transformation_stats[matrix_type] = {
                'count': 0, 
                'coherence_sum': 0
            }
        
        self.transformation_stats[matrix_type]['count'] += 1
        if metrics and 'coherence' in metrics:
            self.transformation_stats[matrix_type]['coherence_sum'] += metrics['coherence']
    
    def store_channel_memory(self, channel_id, data):
        """Store channel-specific memory for image processing"""
        self.channel_memory[channel_id] = data
        
    def get_channel_memory(self, channel_id):
        """Retrieve channel-specific memory"""
        return self.channel_memory.get(channel_id)
    
    def find_similar_transformation(self, input_matrix, threshold=0.8):
        """Find previously seen similar input and its transformation"""
        input_hash = self._matrix_hash(input_matrix)
        input_snippet = self._get_matrix_snippet(input_matrix)
        
        for entry in reversed(self.input_output_pairs):
            if self._snippet_similarity(entry['input_snippet'], input_snippet) > threshold:
                return entry
        return None
        
    def get_best_transformation_type(self, matrix_type=None):
        """Get statistically best transformation type based on past results"""
        if not self.transformation_stats:
            return None
            
        if matrix_type and matrix_type in self.transformation_stats:
            return matrix_type
            
        # Find type with highest average coherence
        best_type = None
        best_avg_coherence = -1
        
        for t_type, stats in self.transformation_stats.items():
            if stats['count'] > 0:
                avg_coherence = stats['coherence_sum'] / stats['count']
                if avg_coherence > best_avg_coherence:
                    best_avg_coherence = avg_coherence
                    best_type = t_type
                    
        return best_type
    
    def add_to_temporal_sequence(self, matrix, time_pos):
        """Add matrix to temporal sequence for tracking changes over time"""
        snippet = self._get_matrix_snippet(matrix)
        self.temporal_sequence.append({
            'time': time_pos,
            'snippet': snippet
        })
        
        # Keep sequence bounded
        if len(self.temporal_sequence) > self.max_size:
            self.temporal_sequence.pop(0)
    
    def _matrix_hash(self, matrix):
        """Create a hash representation of matrix for quick comparison"""
        if isinstance(matrix, np.ndarray):
            # Simple hash based on sum, mean, and shape
            return hash((matrix.shape, np.sum(matrix), np.mean(matrix)))
        return hash(0)
    
    def _get_matrix_snippet(self, matrix):
        """Extract a representative snippet from the matrix"""
        if isinstance(matrix, np.ndarray):
            # Handle different dimensions
            if len(matrix.shape) == 1:
                # 1D array
                w = matrix.shape[0]
                return {
                    'shape': matrix.shape,
                    'corners': [matrix[0], 
                               matrix[min(w-1, 4)]]
                }
            elif len(matrix.shape) >= 2:
                # 2D or higher array
                h, w = matrix.shape[:2]
                return {
                    'shape': matrix.shape,
                    'corners': [matrix[0,0], 
                               matrix[0,min(w-1,4)], 
                               matrix[min(h-1,4),0], 
                               matrix[min(h-1,4),min(w-1,4)]],
                    'mean': np.mean(matrix),
                    'std': np.std(matrix),
                    'sparsity': np.sum(np.abs(matrix) < 1e-10) / matrix.size
                }
        return None
    
    def _snippet_similarity(self, snippet1, snippet2):
        """Calculate similarity between two matrix snippets"""
        if not snippet1 or not snippet2:
            return 0
            
        if snippet1['shape'] != snippet2['shape']:
            return 0.3  # Different shapes have lower base similarity
            
        # Compare statistics
        mean_diff = abs(snippet1['mean'] - snippet2['mean']) / (max(abs(snippet1['mean']), 1e-10))
        std_diff = abs(snippet1['std'] - snippet2['std']) / (max(abs(snippet1['std']), 1e-10))
        sparsity_diff = abs(snippet1['sparsity'] - snippet2['sparsity'])
        
        # Calculate corner similarities
        corner_sim = 0
        for i in range(min(len(snippet1['corners']), len(snippet2['corners']))):
            c1, c2 = snippet1['corners'][i], snippet2['corners'][i]
            if abs(c1) < 1e-10 and abs(c2) < 1e-10:
                corner_sim += 1
            else:
                corner_sim += max(0, 1 - abs(c1 - c2) / max(max(abs(c1), abs(c2)), 1e-10))
                
        corner_sim /= max(1, len(snippet1['corners']))
        
        # Combined similarity score (weighted)
        similarity = (
            0.3 * max(0, 1 - min(1, mean_diff)) + 
            0.2 * max(0, 1 - min(1, std_diff)) + 
            0.2 * max(0, 1 - min(1, sparsity_diff)) +
            0.3 * corner_sim
        )
        
        return similarity
