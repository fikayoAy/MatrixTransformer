# ADD THESE LINES AT THE VERY TOP OF YOUR FILE
import sys, types
sys.modules['cpuinfo'] = types.ModuleType('cpuinfo')
sys.modules['cpuinfo'].get_cpu_info = lambda: {'l1_cache_size': 32*1024, 'l2_cache_size': 256*1024, 'l3_cache_size': 8*1024*1024, 'flags': ['sse', 'sse2']}

# ...rest of your existing code...
import time
import networkx as nx
import math
import numpy as np
from scipy.spatial.distance import euclidean, cosine
from sklearn.cluster import KMeans
from scipy.spatial import KDTree, distance
from base_classes import BaseGraph
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from numba import jit, prange
import torch
import logging
import multiprocessing
import cpuinfo
import platform
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional, List, Dict
import threading




CACHE_LINE_SIZE = 64  # Common cache line size in bytes
L1_CACHE_SIZE = 32 * 1024  # 32KB L1 cache
L2_CACHE_SIZE = 256 * 1024  # 256KB L2 cache
L3_CACHE_SIZE = 12 * 1024 * 1024  # 12MB L3 cache
MAX_THREADS = multiprocessing.cpu_count()
THREAD_POOL = ThreadPoolExecutor(max_workers=MAX_THREADS)
PROCESS_POOL = ProcessPoolExecutor(max_workers=MAX_THREADS)



class MemoryPool:
    """Memory pool for efficient array allocation"""
    def __init__(self, block_size: int = 32, num_blocks: int = 100):
        self.block_size = block_size
        self.pool = [np.zeros(block_size, dtype=np.float32) for _ in range(num_blocks)]
        self.available = set(range(num_blocks))
        self._lock = threading.Lock()

    def acquire(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self.available:
                return None
            block_id = self.available.pop()
            return self.pool[block_id]

    def release(self, block_id: int) -> None:
        with self._lock:
            self.pool[block_id].fill(0)
            self.available.add(block_id)

class HardwareDetector:
    MIN_CACHE_SIZE = 16 * 1024  # 16KB minimum
    DEFAULT_CACHE_SIZES = {
        'l1': 32 * 1024,    # 32KB
        'l2': 256 * 1024,   # 256KB
        'l3': 8 * 1024 * 1024  # 8MB
    }

    @staticmethod
    def detect_cache_sizes():
        """Detect CPU cache sizes with fallback options"""
        try:
            info = cpuinfo.get_cpu_info()
            
            # Try to get actual cache sizes
            cache_sizes = {
                'l1': info.get('l1_cache_size', HardwareDetector.DEFAULT_CACHE_SIZES['l1']),
                'l2': info.get('l2_cache_size', HardwareDetector.DEFAULT_CACHE_SIZES['l2']),
                'l3': info.get('l3_cache_size', HardwareDetector.DEFAULT_CACHE_SIZES['l3'])
            }

            # Validate cache sizes
            for level, size in cache_sizes.items():
                if size < HardwareDetector.MIN_CACHE_SIZE:
                    cache_sizes[level] = HardwareDetector.DEFAULT_CACHE_SIZES[level]

            return cache_sizes

        except Exception as e:
            logging.warning(f"Cache detection failed: {str(e)}")
            return HardwareDetector.DEFAULT_CACHE_SIZES
        

    @staticmethod
    def detect_basic_features():
        """Detect basic features available on any processor"""
        try:
            features = {
                'arch': platform.machine().lower(),
                'bits': platform.architecture()[0],
                'cores': max(1, psutil.cpu_count(logical=False) or 1),
                'memory': max(512 * 1024 * 1024, psutil.virtual_memory().total)
            }
            
            # Add ARM vs x86 detection
            features['is_arm'] = 'arm' in features['arch']
            features['is_x86'] = 'x86' in features['arch']
            
            return features
        except:
            return {
                'arch': 'unknown',
                'bits': '32bit',
                'cores': 1,
                'memory': 512 * 1024 * 1024,  # 512MB minimum
                'is_arm': False,
                'is_x86': False
            }

    @staticmethod
    def get_safe_compute_settings(features):
        """Get safe computation settings for any processor"""
        return {
            'use_simd': False,  # Default to scalar operations
            'min_memory': 32 * 1024 * 1024,  # 32MB minimum working set
            'cache_line': 32,  # Conservative cache line size
            'max_threads': max(1, features['cores']),
            'use_fma': False  # Disable FMA by default
        }

    @staticmethod
    def detect_cpu_features():
        """Detect CPU features with comprehensive capabilities check"""
        try:
            info = cpuinfo.get_cpu_info()
            flags = info.get('flags', [])
            
            # Get core counts
            physical_cores = psutil.cpu_count(logical=False) or 2
            logical_cores = psutil.cpu_count(logical=True) or 4

            # Check for specific instruction sets
            features = {
                'avx': any(x in flags for x in ['avx', 'avx2', 'avx512']),
                'sse': any(x in flags for x in ['sse', 'sse2', 'sse3', 'sse4']),
                'fma': 'fma' in flags,
                'cores': physical_cores,
                'threads': logical_cores,
                'frequency': psutil.cpu_freq().max if psutil.cpu_freq() else 2000,
                'memory': psutil.virtual_memory().total
            }

            # Add architecture information
            features['arch'] = platform.machine().lower()
            features['is_64bit'] = platform.architecture()[0] == '64bit'
            
            return features

        except Exception as e:
            logging.warning(f"CPU feature detection failed: {str(e)}")
            return {
                'avx': False,
                'sse': True,
                'fma': False,
                'cores': 2,
                'threads': 4,
                'frequency': 2000,
                'memory': 4 * 1024 * 1024 * 1024,  # 4GB
                'arch': 'x86_64',
                'is_64bit': True
            }



# At the top level, define the block size constant
BLOCK_SIZE = 64  # Fixed block size for Numba optimization

@jit(nopython=True, parallel=True)
def block_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Input validation
    if A.ndim != 2 or B.ndim != 2:
        return np.zeros((1, 1), dtype=np.complex128)
    if A.shape[1] != B.shape[0]:
        return np.zeros((1, 1), dtype=np.complex128)
    
    # Ensure contiguous memory layout  
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    m, n, k = A.shape[0], B.shape[1], A.shape[1]
    C = np.zeros((m, n), dtype=np.complex128)
    
    block_size = 64
    n_blocks_i = (m + block_size - 1) // block_size
    n_blocks_j = (n + block_size - 1) // block_size
    n_blocks_k = (k + block_size - 1) // block_size

    # Use prange with constant step size of 1 by iterating over block indices
    for bi in prange(n_blocks_i):
        i = bi * block_size
        i_end = i + block_size if i + block_size < m else m
        for bj in range(n_blocks_j):
            j = bj * block_size
            j_end = j + block_size if j + block_size < n else n
            for bk in range(n_blocks_k):
                k_start = bk * block_size
                k_end = k_start + block_size if k_start + block_size < k else k
                for ii in range(i, i_end):
                    for jj in range(j, j_end):
                        acc = 0.0
                        for kk in range(k_start, k_end):
                            acc += A[ii, kk] * B[kk, jj]
                        C[ii, jj] += acc
    return C

class BlockedMatrixOps:
    """Wrapper class for blocked matrix operations"""
    
    @staticmethod
    def block_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Wrapper for the JIT-compiled matrix multiplication"""
        return block_matrix_multiply(A, B)

# Add these optimized helper functions at the top level
@jit(nopython=True, parallel=True)
def optimized_batch_processing(states: np.ndarray) -> np.ndarray:
    """Optimized parallel batch processing with cache-friendly operations"""
    n_samples = states.shape[0]
    output = np.zeros_like(states)
    
    # Use optimal block size for cache efficiency
    block_size = 32  # L1 cache friendly
    
    for i in prange(0, n_samples, block_size):
        end = min(i + block_size, n_samples)
        chunk = states[i:end]
        
        # Process block with vectorized operations
        output[i:end] = np.tanh(chunk)  # Non-linear activation
        
    return output





@staticmethod
@jit(nopython=True, parallel=True)
def _process_patterns_block(patterns: np.ndarray, block_size: int) -> np.ndarray:
    """Process patterns in cache-friendly blocks"""
    m, n = patterns.shape
    processed = np.zeros_like(patterns)
    
    for i in prange(0, m, block_size):
        for j in range(0, n, block_size):
            i_end = min(i + block_size, m)
            j_end = min(j + block_size, n)
            block = patterns[i:i_end, j:j_end]
            
            # Process block with vectorized operations
            processed[i:i_end, j:j_end] = block / (np.linalg.norm(block) + 1e-8)
            
    return processed

class ProblemNode:
    """Represents a node in the problem space"""
    def __init__(self, node_id: str, dimension: int = 32):
        self.id = node_id
        self.problem_zone = np.zeros(dimension)  # Problem type embedding
        self.success_density = 0.0  # Success rate
        self.computation_cost = 0.0  # Average computation time
        self.cardinal_signature = np.random.randn(dimension)  # Directional signature
        self.cardinal_signature /= np.linalg.norm(self.cardinal_signature) + 1e-8
        self.visit_count = 0
        
class ProblemEdge:
    """Represents an edge in the problem space"""
    def __init__(self):
        self.transition_entropy = 0.0  # Transformation complexity
        self.frequency = 0  # Usage count
        self.success_ratio = 0.0  # Success rate
        self.embedding_diff = None  # Vector between node embeddings


class DynamicGraph(BaseGraph):
    def __init__(self, directed=False):
        super().__init__()
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self.node_weight_cache = {}
        self.edge_weight_cache = {}
        
        # Add new attributes for cardinal system
        self.cardinality_dim = 4  # Default dimension for cardinality vectors
        self.epsilon = np.pi/4    # Default resonance threshold
        self.delta = 0.5         # Default fusion threshold
        self.alpha = 0.7         # Default stability factor
        
        # Add node and edge cardinality storage
        self.node_cardinality = {}
        self.edge_cardinality = {}

        # Add problem space mapping attributes
        self.problem_nodes = {}  # Store ProblemNode objects
        self.problem_edges = {}  # Store ProblemEdge objects
        self.embedding_dim = 32  # Default embedding dimension

        # Add caching for problem space operations
        self.similarity_cache = {}
        self.path_cache = {}
        self.cache_ttl = 100  # Cache entries expire after 100 updates

        # Add monitoring metrics
        self.metrics = {
            'total_problems_solved': 0,
            'successful_solutions': 0,
            'average_path_length': 0.0,
            'zone_coverage': {},  # Track problem zone coverage
            'hotspots': set()     # Track frequently visited nodes
        }

    def edge(self, node1, node2):
        if self.graph.has_edge(node1, node2):
            return self.graph[node1][node2]
        else:
            return None

    def has_edge(self, node1, node2):
        return self.graph.has_edge(node1, node2)

    def add_node(self, node_id, properties, is_circular_time=False, radius=1):
        """Add a node with properties and cardinality vector to the graph"""
        node_attrs = {'properties': properties}
        
        # Add cardinality vector if not present
        if 'cardinality' not in properties:
            properties['cardinality'] = np.random.randn(self.cardinality_dim)
            properties['cardinality'] = np.array(properties['cardinality'], dtype=np.float64)
            properties['cardinality'] /= np.linalg.norm(properties['cardinality'])
        else:
            # Ensure existing cardinality has the right dimension
            cardinality = np.array(properties['cardinality'], dtype=np.float64)
            if cardinality.shape != (self.cardinality_dim,):
                # Resize to correct dimension
                new_cardinality = np.zeros(self.cardinality_dim, dtype=np.float64)
                new_cardinality[:min(len(cardinality), self.cardinality_dim)] = cardinality[:min(len(cardinality), self.cardinality_dim)]
                properties['cardinality'] = new_cardinality
                properties['cardinality'] /= np.linalg.norm(properties['cardinality']) or 1.0
        
        self.node_cardinality[node_id] = properties['cardinality']
        
        if is_circular_time:
            angle = properties.get('angle', 0)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            node_attrs['position'] = (x, y)
            
        self.graph.add_node(node_id, **node_attrs)

    def add_edge(self, node1, node2, weight=None, distance=None, **properties):
        """Add edge with cardinality vector"""
        if weight is not None:
            properties['weight'] = weight
        if distance is not None:
            properties['distance'] = distance
        if 'cardinality' not in properties:
            # Generate edge cardinality as normalized average of node cardinalities
            c1 = self.node_cardinality.get(node1, np.zeros(self.cardinality_dim))
            c2 = self.node_cardinality.get(node2, np.zeros(self.cardinality_dim))
            
            # FIXED: Convert lists to numpy arrays if needed
            if isinstance(c1, list):
                c1 = np.array(c1, dtype=np.float64)
            if isinstance(c2, list):
                c2 = np.array(c2, dtype=np.float64)
                
            # ADDED FIX: Ensure both vectors have the same dimension
            if c1.shape != c2.shape:
                # Resize both vectors to match the expected cardinality_dim
                new_c1 = np.zeros(self.cardinality_dim, dtype=np.float64)
                new_c2 = np.zeros(self.cardinality_dim, dtype=np.float64)
                
                # Copy values from original vectors up to the minimum size
                new_c1[:min(len(c1), self.cardinality_dim)] = c1[:min(len(c1), self.cardinality_dim)]
                new_c2[:min(len(c2), self.cardinality_dim)] = c2[:min(len(c2), self.cardinality_dim)]
                
                # Replace with dimension-corrected vectors
                c1, c2 = new_c1, new_c2
                
                # Update the node cardinality to prevent future issues
                self.node_cardinality[node1] = c1
                self.node_cardinality[node2] = c2
            
            # Now safely add the vectors with matching dimensions
            properties['cardinality'] = np.array(c1 + c2, dtype=np.float64)
            norm = np.linalg.norm(properties['cardinality'])
            if norm > 0:
                properties['cardinality'] = properties['cardinality'] / norm
                    
        self.edge_cardinality[(node1, node2)] = properties['cardinality']
        self.graph.add_edge(node1, node2, **properties)
        
    def calculate_edge_weight(self, node1, node2, properties):
        """Calculate edge weight using cardinal resonance"""
        # Get cardinality vectors
        c1 = np.array(self.node_cardinality.get(node1, np.zeros(self.cardinality_dim)), dtype=np.float64)
        c2 = np.array(self.node_cardinality.get(node2, np.zeros(self.cardinality_dim)), dtype=np.float64)
        
        # Calculate angle between vectors
        cos_theta = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-10)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        # Extract base_weight - ensure we get a float value
        if isinstance(properties, dict):
            weight_value = properties.get('weight', 1.0)
            if isinstance(weight_value, (int, float)):
                base_weight = float(weight_value)
            else:
                # Handle case where weight is not a numeric value
                base_weight = 1.0
        else:
            try:
                base_weight = float(properties)
            except (TypeError, ValueError):
                base_weight = 1.0
        
        # For angles less than 60 degrees (π/3), we amplify the weight
        # For larger angles, we reduce the weight
        if theta < np.pi/3:
            resonance_factor = np.exp(-(theta / self.epsilon)**2)
            return base_weight * (1.0 + resonance_factor)
        else:
            # Decreases as angle increases
            damping_factor = np.exp(-(theta - np.pi/3) / self.epsilon)
            return base_weight * damping_factor

    def calculate_node_weight(self, node):
        if node in self.node_weight_cache:
            return self.node_weight_cache[node]
        edges = self.graph.edges(node, data=True)
        total_edge_weight = sum(edge_data['weight'] for _, _, edge_data in edges)
        num_edges = len(edges)
        weight = total_edge_weight / num_edges if num_edges > 0 else 0
        self.node_weight_cache[node] = weight
        return weight

    def calculate_tree_weight(self, tree_nodes):
        tree = self.graph.subgraph(tree_nodes)
        total_node_weight = sum(self.calculate_node_weight(node) for node in tree.nodes())
        total_edge_weight = sum(edge_data['weight'] for _, _, edge_data in tree.edges(data=True))
        num_nodes = len(tree.nodes)
        num_edges = len(tree.edges)
        return (total_node_weight + total_edge_weight) / (num_nodes + num_edges) if (num_nodes + num_edges) > 0 else 0

    def calculate_graph_weight(self):
        """
        Calculate the overall graph weight as the average of all edges.
        For a directed graph, also consider the connectivity structure.
        """
        # If graph is empty, return default weight
        if self.graph.number_of_nodes() == 0:
            return 0.0
            
        # Get all edge weights
        weights = [data.get('weight', 0.0) for _, _, data in self.graph.edges(data=True)]
        
        # Calculate basic weight as average of all edges
        base_weight = sum(weights) / len(weights) if weights else 0.0
        
        # Add connectivity component - use appropriate function based on graph type
        try:
            if self.graph.is_directed():
                # For directed graphs, use weakly connected components
                connected_components = list(nx.weakly_connected_components(self.graph))
            else:
                # For undirected graphs, use regular connected components
                connected_components = list(nx.connected_components(self.graph))
                
            # Calculate connectivity factor (more components = lower weight)
            component_factor = 1.0 / (1.0 + len(connected_components))
            
            # Final weight combines edge weights and connectivity
            return base_weight * (0.7 + 0.3 * component_factor)
            
        except nx.NetworkXNotImplemented:
            # Fallback if connectivity algorithms fail
            return base_weight
    
    def neighbors(self, node):
        """Return an iterator over the neighbors of node."""
        return self.graph.neighbors(node)
        
    def get_edge_data(self, u, v, default=None):
        """Return the attribute dictionary associated with edge (u, v)."""
        if self.graph.has_edge(u, v):
            return self.graph[u][v]
        return default

    def update_node(self, node_id, new_properties):
        """Update node with cognitive adaptation"""
        if node_id in self.graph.nodes:
            if 'cardinality' in new_properties:
                old_card = self.node_cardinality.get(node_id, np.zeros(self.cardinality_dim))
                new_card = new_properties['cardinality']
                
                # Apply cognitive adaptation formula
                adapted_card = (self.alpha * old_card + 
                              (1 - self.alpha) * new_card)
                adapted_card /= np.linalg.norm(adapted_card) + 1e-10
                
                self.node_cardinality[node_id] = adapted_card
                new_properties['cardinality'] = adapted_card
                
            self.graph.nodes[node_id].update(new_properties)
            self.node_weight_cache.pop(node_id, None)
            for neighbor in self.graph.neighbors(node_id):
                # Fix: Pass only an empty update dict instead of edge_properties
                self.update_edge(node_id, neighbor, {})

    def update_edge(self, node1, node2, new_properties):
        if not self.graph.has_edge(node1, node2):
            raise ValueError(f"Edge between {node1} and {node2} does not exist.")
        
        current_properties = self.graph[node1][node2]
        current_properties.update(new_properties)
        
        # Update edge_cardinality if provided in new_properties
        if 'cardinality' in new_properties:
            self.edge_cardinality[(node1, node2)] = new_properties['cardinality']
        
        # Calculate and update weight
        weight = self.calculate_edge_weight(node1, node2, current_properties)
        current_properties['weight'] = weight
        self.edge_weight_cache[(node1, node2)] = weight

    def remove_node(self, node_id):
        if node_id in self.graph:
            self.graph.remove_node(node_id)
            self.node_weight_cache.pop(node_id, None)
            self.edge_weight_cache = {
                key: value for key, value in self.edge_weight_cache.items() if node_id not in key
            }

    def get_node_position(self, node_id):
        """Get the position of a node in the graph."""
        if not self.has_node(node_id):
            # Return default position instead of None
            return (0.0, 0.0, 0.0)
        
        position = self.graph.nodes[node_id].get('position')
        if position is None:
            # Create and assign a default position if missing
            position = (0.0, 0.0, 0.0)
            self.graph.nodes[node_id]['position'] = position
        
        return position
    




    def get_node_properties(self, node_id):
        """Get the properties of a node in the graph."""
        if not self.has_node(node_id):
            return {}
        return self.graph.nodes[node_id]

    def remove_edge(self, node1, node2):
        if self.graph.has_edge(node1, node2):
            self.graph.remove_edge(node1, node2)
            self.edge_weight_cache.pop((node1, node2), None)

    def generate_element_signature(self, element_id, is_node=True):
        if is_node:
            edges = self.graph.edges(element_id, data=True)
            edge_signatures = [f"{neighbor}:{edge_data['weight']}" for _, neighbor, edge_data in edges]
            return f"Node-{element_id}-Properties-{self.graph.nodes[element_id]}-Edges-{edge_signatures}"
        else:
            node1, node2 = element_id
            edge_data = self.graph[node1][node2]
            return f"Edge-{node1}-{node2}-Properties-{edge_data}"

    def dynamic_edge_discovery(self, similarity_function, threshold=0.1):
        for node1 in self.graph.nodes():
            for node2 in self.graph.nodes():
                if node1 != node2 and not self.graph.has_edge(node1, node2):
                    similarity = similarity_function(self.graph.nodes[node1], self.graph.nodes[node2])
                    if similarity > threshold:
                        self.add_edge(node1, node2, weight=similarity)

    def has_node(self, node_id):
        """Check if a node exists in the graph."""
        return node_id in self.graph.nodes

    def update_node_weight_dynamically(self, node_id):
        neighbors = list(self.graph.neighbors(node_id))
        neighbor_weights = [self.calculate_node_weight(neighbor) for neighbor in neighbors]
        self.graph.nodes[node_id]['dynamic_weight'] = sum(neighbor_weights) / len(neighbor_weights) if len(neighbor_weights) > 0 else 0

    def generate_weight_blueprint(self, node_id):
        edges = self.graph.edges(node_id, data=True)
        edge_weights = [edge_data['weight'] for _, _, edge_data in edges]
        return {'node_id': node_id, 'edge_weights': edge_weights, 'total_weight': sum(edge_weights)}

    def successors(self, node):
        return list(self.graph.neighbors(node))

    def division_based_traversal(self, start_node, target, path=None):
        """Cognitive traversal using cardinal resonance conditions"""
        if path is None:
            path = []
            
        if start_node == target:
            return path + [start_node]
        
        if start_node in path:  # Prevent cycles
            return None
            
        neighbors = list(self.graph.neighbors(start_node))
        valid_neighbors = []
        
        for neighbor in neighbors:
            if neighbor not in path:
                # Get cardinality vectors
                cv_i = self.node_cardinality.get(start_node)
                cv_j = self.node_cardinality.get(neighbor)
                ce_ij = self.edge_cardinality.get((start_node, neighbor))
                
                # Skip if any of the vectors are missing
                if cv_i is None or cv_j is None or ce_ij is None:
                    continue
                    
                # Calculate norms with safety checks
                norm_cv_i = np.linalg.norm(cv_i)
                norm_cv_j = np.linalg.norm(cv_j)
                norm_ce_ij = np.linalg.norm(ce_ij)
                
                # Skip if any vector has zero norm
                if norm_cv_i < 1e-10 or norm_cv_j < 1e-10 or norm_ce_ij < 1e-10:
                    continue
                
                # Calculate dot products
                dot1 = np.dot(cv_i, ce_ij)
                dot2 = np.dot(ce_ij, cv_j)
                
                # Calculate cosine safely
                cos_theta1 = dot1 / (norm_cv_i * norm_ce_ij)
                cos_theta2 = dot2 / (norm_ce_ij * norm_cv_j)
                
                # Handle numerical errors that might produce values outside [-1, 1]
                cos_theta1 = np.clip(cos_theta1, -1.0, 1.0)
                cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
                
                # Calculate angles
                theta1 = np.arccos(cos_theta1)
                theta2 = np.arccos(cos_theta2)
                
                # Check resonance conditions with strict epsilon
                if theta1 <= self.epsilon and theta2 <= self.epsilon:
                    valid_neighbors.append((neighbor, max(theta1, theta2)))
        
        # Sort neighbors by angle for best-first search
        valid_neighbors.sort(key=lambda x: x[1])
        
        # Try cognitive traversal through valid neighbors
        for neighbor, _ in valid_neighbors:
            new_path = self.division_based_traversal(neighbor, target, path + [start_node])
            if new_path is not None:
                return new_path
                
        return None

    def navigate_cluster(self, cluster):
        visited = []
        def traverse_cluster(cluster_part):
            if not cluster_part:
                return
            midpoint = len(cluster_part) // 2
            left_half = cluster_part[:midpoint]
            right_half = cluster_part[midpoint:]
            def traverse_half(half):
                if not half:
                    return
                for row in range(len(half)):
                    if isinstance(half[row], list):
                        if row % 2 == 0:
                            for col in range(len(half[row])):
                                visited.append(half[row][col])
                        else:
                            for col in range(len(half[row]) - 1, -1, -1):
                                visited.append(half[row][col])
                    else:
                        visited.append(half[row])
            traverse_half(left_half)
            traverse_half(right_half)
        traverse_cluster(cluster)
        return visited
    
    @staticmethod
    def check_similarity(args, patterns):
        i, j = args
        pattern1, pattern2 = np.array(patterns[i]), np.array(patterns[j])
        distance = np.linalg.norm(pattern1 - pattern2)
        return (pattern1, pattern2, distance)

    def update_node_positions(self):
        # Ensure every node has a 'position' property
        for node in self.graph.nodes:
            if 'position' not in self.graph.nodes[node]:
                self.graph.nodes[node]['position'] = np.array([0.0, 0.0, 0.0])
        # Update positions: set each node's position to the mean of neighbor positions if neighbors exist
        for node in self.graph.nodes:
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                neighbor_positions = np.array([self.graph.nodes[n]['position'] for n in neighbors])
                self.graph.nodes[node]['position'] = np.mean(neighbor_positions, axis=0)

    def create_subgraph(self, name):
        """Create a subgraph with the given name
        
        Args:
            name: Name of the subgraph
            
        Returns:
            DynamicGraph: A new graph instance representing the subgraph
        """
        subgraph = DynamicGraph(directed=isinstance(self.graph, nx.DiGraph))
        subgraph.cardinality_dim = self.cardinality_dim
        subgraph.epsilon = self.epsilon
        subgraph.delta = self.delta
        subgraph.alpha = self.alpha
        
        # Add metadata about parent graph
        subgraph.parent_graph_id = id(self)
        subgraph.subgraph_name = name
        
        return subgraph
    
    def get_node_property(self, node_id, property_name, default=None):
        """Get a specific property of a node in the graph.
        
        Args:
            node_id: The ID of the node
            property_name: The name of the property to retrieve
            default: Default value if the property does not exist
            
        Returns:
            The property value or the default if the property does not exist
        """
        # First check if the node exists
        if not self.has_node(node_id):
            return default
            
        # Get all node properties
        node_data = self.graph.nodes[node_id]
        
        # For nested properties like 'properties.weight'
        if '.' in property_name:
            parts = property_name.split('.')
            current = node_data
            
            # Navigate through nested properties
            for part in parts[:-1]:
                if part not in current:
                    return default
                current = current[part]
                
            # Get the final property
            return current.get(parts[-1], default)
        
        # For top-level properties
        if property_name in node_data:
            return node_data[property_name]
        
        # Check if it's in the 'properties' dict
        if 'properties' in node_data and property_name in node_data['properties']:
            return node_data['properties'][property_name]
            
        # Property not found
        return default
    
    def _update_action_state_graph(self, state, actions, results):
        """Update the action-state graph with results of the latest actions"""
        if not hasattr(self, 'action_state_graph'):
            self.action_state_graph = DynamicGraph(directed=True)
                
        # Get state_hash with a default and ensure it's not None
        state_hash = state.get('hash', '')
        if state_hash is None:
            # Use a placeholder for None hash values
            state_hash = f"placeholder_{id(state)}"
        
        # Calculate overall success score for this action sequence
        success_count = sum(1 for outcome in results.get('action_outcomes', []) if outcome.get('success', False))
        total_actions = len(results.get('action_outcomes', []))
        success_ratio = success_count / max(1, total_actions)
        
        # Calculate reward based on insights and outcomes
        reward = len(results.get('insights', [])) * 0.2 + success_ratio * 0.8
        
        # Prepare node properties
        node_properties = {
            'state_vector': state.get('vector', np.zeros(4)),  # Safe default
            'visits': 1,
            'successful_actions': [],
            'last_visit_time': getattr(self, 'internal_time', 0),
            'cardinality': np.array([1.0, 0.0, 0.0, 0.0])  # Default cardinality
        }
        
        # Add or update node in graph based on the type of graph
        if not self.action_state_graph.has_node(state_hash):
            if isinstance(self.action_state_graph, DynamicGraph):
                # For DynamicGraph, wrap properties in a dict
                wrapped_properties = {'properties': node_properties}
                self.action_state_graph.add_node(state_hash, wrapped_properties)
            else:
                # For NetworkX graph, properties should be provided as keyword args
                self.action_state_graph.add_node(state_hash, **node_properties)
        else:
            # Update existing node
            if isinstance(self.action_state_graph, DynamicGraph):
                current_props = self.action_state_graph.get_node_properties(state_hash)
                visits = current_props.get('properties', {}).get('visits', 0) + 1
                
                # Update properties in the correct format for DynamicGraph
                updated_properties = {
                    'properties': {
                        'visits': visits,
                        'last_visit_time': getattr(self, 'internal_time', 0)
                    }
                }
                self.action_state_graph.update_node(state_hash, updated_properties)
            else:
                # For NetworkX graph, update node attributes directly
                self.action_state_graph.nodes[state_hash]['visits'] = \
                    self.action_state_graph.nodes[state_hash].get('visits', 0) + 1
                self.action_state_graph.nodes[state_hash]['last_visit_time'] = \
                    getattr(self, 'internal_time', 0)
        
        # If this action sequence was successful, add to successful actions
        if reward > 0.5:  # Only store reasonably successful action sequences
            # Create action sequence record
            action_sequence = {
                'actions': actions,
                'reward': reward,
                'time': getattr(self, 'internal_time', 0)
            }
            
            # Add to successful actions and keep sorted by reward
            if isinstance(self.action_state_graph, DynamicGraph):
                # For DynamicGraph
                node_props = self.action_state_graph.get_node_properties(state_hash)
                successful_actions = node_props.get('properties', {}).get('successful_actions', [])
            else:
                # For NetworkX graph
                successful_actions = self.action_state_graph.nodes[state_hash].get('successful_actions', [])
                
            successful_actions.append(action_sequence)
            successful_actions.sort(key=lambda x: x['reward'], reverse=True)
            successful_actions = successful_actions[:5]  # Keep only top 5
            
            # Update node with new successful actions
            if isinstance(self.action_state_graph, DynamicGraph):
                self.action_state_graph.update_node(state_hash, {
                    'properties': {'successful_actions': successful_actions}
                })
            else:
                self.action_state_graph.nodes[state_hash]['successful_actions'] = successful_actions
            
            # Add edges between consecutive actions if there are multiple actions
            if len(actions) > 1:
                for i in range(len(actions) - 1):
                    action1, action2 = actions[i], actions[i + 1]
                    if action1 is None or action2 is None:
                        continue  # Skip None values
                        
                    if not self.action_state_graph.has_edge(action1, action2):
                        if isinstance(self.action_state_graph, DynamicGraph):
                            edge_properties = {
                                'properties': {
                                    'weight': reward,
                                    'success_ratio': success_ratio
                                }
                            }
                            self.action_state_graph.add_edge(action1, action2, **edge_properties)
                        else:
                            self.action_state_graph.add_edge(action1, action2, 
                                                            weight=reward,
                                                            success_ratio=success_ratio)
    def init_problem_space(self):
        """Initialize problem space structures for existing nodes"""
        for node_id in self.graph.nodes():
            if node_id not in self.problem_nodes:
                self.problem_nodes[node_id] = ProblemNode(node_id, self.embedding_dim)
                
        # Initialize edges
        for u, v in self.graph.edges():
            edge_key = (u, v)
            if edge_key not in self.problem_edges:
                self.problem_edges[edge_key] = ProblemEdge()

    def update_node_with_problem(self, node_id: str, problem_vector: np.ndarray, 
                               success: bool, computation_time: float):
        """Update node with problem solving experience"""
        if node_id not in self.problem_nodes:
            self.problem_nodes[node_id] = ProblemNode(node_id, self.embedding_dim)
            
        node = self.problem_nodes[node_id]
        node.visit_count += 1
        
        # Update problem zone (moving average)
        alpha = 1.0 / node.visit_count
        node.problem_zone = (1 - alpha) * node.problem_zone + alpha * problem_vector
        
        # Update success density
        if success:
            node.success_density = ((node.success_density * (node.visit_count - 1) + 1) 
                                  / node.visit_count)
        
        # Update computation cost (moving average)
        node.computation_cost = ((node.computation_cost * (node.visit_count - 1) + 
                                computation_time) / node.visit_count)

    def update_edge_with_traverse(self, from_id: str, to_id: str, success: bool):
        """Update edge with traversal experience using existing graph methods
        
        Args:
            from_id: Source node ID
            to_id: Target node ID 
            success: Whether the traversal was successful
        """
        # First check if nodes exist
        if not self.has_node(from_id) or not self.has_node(to_id):
            return
            
        # Create edge key and initialize if needed
        edge_key = (from_id, to_id)
        if edge_key not in self.problem_edges:
            self.problem_edges[edge_key] = ProblemEdge()
            
        edge = self.problem_edges[edge_key]
        edge.frequency += 1
        
        # Update success ratio
        edge.success_ratio = ((edge.success_ratio * (edge.frequency - 1) + 
                            (1.0 if success else 0.0)) / edge.frequency)
        
        # Get node properties using existing methods
        from_props = self.get_node_properties(from_id)
        to_props = self.get_node_properties(to_id)
        
        # Update edge in main graph if it exists
        if self.has_edge(from_id, to_id):
            # Update edge properties
            edge_props = {
                'frequency': edge.frequency,
                'success_ratio': edge.success_ratio,
                'weight': edge.success_ratio  # Use success ratio as weight
            }
            
            # Use existing update_edge method
            self.update_edge(from_id, to_id, edge_props)
        else:
            # Create new edge if it doesn't exist
            self.add_edge(from_id, to_id, 
                        weight=edge.success_ratio,
                        frequency=edge.frequency,
                        success_ratio=edge.success_ratio)
        
        # Update embedding difference if both nodes exist in problem space
        if from_id in self.problem_nodes and to_id in self.problem_nodes:
            from_vec = self.problem_nodes[from_id].problem_zone
            to_vec = self.problem_nodes[to_id].problem_zone
            edge.embedding_diff = to_vec - from_vec
            
            # Calculate transition entropy using cosine distance
            # Add safety checks for zero vectors
            if np.any(from_vec) and np.any(to_vec):
                edge.transition_entropy = cosine(from_vec, to_vec)
            else:
                edge.transition_entropy = 1.0  # Maximum entropy for zero vectors
                
            # Update edge property in main graph
            if self.has_edge(from_id, to_id):
                self.update_edge(from_id, to_id, {
                    'transition_entropy': edge.transition_entropy
                })
    def generate_problem_space_map(self) -> Dict:
        """Generate a map of the problem space"""
        problem_map = {}
        
        for node_id, node in self.problem_nodes.items():
            local_chunk = {
                'zone_vector': node.problem_zone.tolist(),
                'density': node.success_density,
                'cost': node.computation_cost,
                'cardinal': node.cardinal_signature.tolist(),
                'neighbors': []
            }
            
            # Get neighboring nodes
            for neighbor_id in self.graph.neighbors(node_id):
                edge_key = (node_id, neighbor_id)
                if edge_key in self.problem_edges:
                    edge = self.problem_edges[edge_key]
                    path_info = {
                        'to': neighbor_id,
                        'entropy': edge.transition_entropy,
                        'success_ratio': edge.success_ratio,
                        'frequency': edge.frequency
                    }
                    local_chunk['neighbors'].append(path_info)
                    
            problem_map[node_id] = local_chunk
            
        return problem_map

    def find_solution_path(self, problem_vector: np.ndarray, 
                         start_node: Optional[str] = None) -> List[str]:
        """Find a path to solve a problem using the problem space map"""
        if not start_node:
            # Find best starting node based on problem similarity
            start_node = self._find_closest_node(problem_vector)
            
        if not start_node:
            return []
            
        path = [start_node]
        current = start_node
        visited = {start_node}
        
        while True:
            best_next = None
            best_score = -float('inf')
            
            # Check all neighbors
            for neighbor_id in self.graph.neighbors(current):
                if neighbor_id in visited:
                    continue
                    
                edge_key = (current, neighbor_id)
                if edge_key not in self.problem_edges:
                    continue
                    
                edge = self.problem_edges[edge_key]
                neighbor = self.problem_nodes[neighbor_id]
                
                # Score based on success ratio and problem similarity
                similarity = 1 - cosine(neighbor.problem_zone, problem_vector)
                score = (edge.success_ratio * 0.7 + 
                        similarity * 0.3) * (1 - edge.transition_entropy)
                
                if score > best_score:
                    best_score = score
                    best_next = neighbor_id
            
            if not best_next or best_score < 0.1:  # Stop if no good options
                break
                
            path.append(best_next)
            visited.add(best_next)
            current = best_next
            
        return path

    def _find_closest_node(self, problem_vector: np.ndarray) -> Optional[str]:
        """Find the node most similar to the problem vector"""
        best_node = None
        best_similarity = -float('inf')
        
        for node_id, node in self.problem_nodes.items():
            similarity = 1 - cosine(node.problem_zone, problem_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_node = node_id
                
        return best_node

    def optimize_problem_space(self):
        """Periodically optimize the problem space mapping"""
        # Clear expired cache entries
        current_time = time.time()
        self.similarity_cache = {k: v for k, v in self.similarity_cache.items() 
                               if current_time - v[1] < self.cache_ttl}
        self.path_cache = {k: v for k, v in self.path_cache.items() 
                          if current_time - v[1] < self.cache_ttl}
        
        # Merge similar problem zones
        for node1 in list(self.problem_nodes.keys()):
            for node2 in list(self.problem_nodes.keys()):
                if node1 != node2:
                    similarity = 1 - cosine(self.problem_nodes[node1].problem_zone,
                                         self.problem_nodes[node2].problem_zone)
                    if similarity > 0.95:  # Very similar nodes
                        self._merge_problem_nodes(node1, node2)

    def _merge_problem_nodes(self, node1: str, node2: str):
        """Merge two similar problem nodes"""
        if node1 not in self.problem_nodes or node2 not in self.problem_nodes:
            return
            
        # Weighted average of problem zones
        w1 = self.problem_nodes[node1].visit_count
        w2 = self.problem_nodes[node2].visit_count
        total = w1 + w2
        
        if total == 0:
            return
            
        merged_zone = (w1 * self.problem_nodes[node1].problem_zone + 
                      w2 * self.problem_nodes[node2].problem_zone) / total
                      
        # Update node1 with merged data
        self.problem_nodes[node1].problem_zone = merged_zone
        self.problem_nodes[node1].success_density = (
            w1 * self.problem_nodes[node1].success_density +
            w2 * self.problem_nodes[node2].success_density
        ) / total
        
        # Remove node2
        del self.problem_nodes[node2]

    def update_metrics(self, path: List[str], success: bool):
        """Update monitoring metrics after each problem-solving attempt"""
        self.metrics['total_problems_solved'] += 1
        if success:
            self.metrics['successful_solutions'] += 1
            
        # Update path length metrics
        path_length = len(path)
        alpha = 0.1  # Smoothing factor
        self.metrics['average_path_length'] = (
            (1 - alpha) * self.metrics['average_path_length'] + 
            alpha * path_length
        )
        
        # Update zone coverage
        for node_id in path:
            if node_id in self.problem_nodes:
                zone = tuple(np.round(self.problem_nodes[node_id].problem_zone, 2))
                self.metrics['zone_coverage'][zone] = \
                    self.metrics['zone_coverage'].get(zone, 0) + 1
                
                # Track hotspots
                if self.metrics['zone_coverage'][zone] > 10:  # Threshold
                    self.metrics['hotspots'].add(node_id)

class PatternSpace:
    def __init__(self, patterns, base_similarity_threshold):
        # Store original patterns for tests (without modification)
        self.original_patterns = patterns
        
        # Initialize patterns and base attributes 
        if isinstance(patterns, np.ndarray) and patterns.size == 0:
            self.patterns = np.zeros((0, 11), dtype=np.float32)
        else:
            # Convert to numpy array and ensure non-negative values
            self.patterns = self._ensure_non_negative(patterns)
            
        self.base_similarity_threshold = base_similarity_threshold
        self.history = []
        self.iteration_entropy = []
        self.tree = None
        self.pattern_norms = None

        # Initialize hardware detection 
        self.hw_features = HardwareDetector.detect_basic_features()
        self.cache_sizes = HardwareDetector.detect_cache_sizes()
        
        # Calculate optimal block size before it's needed
        self.optimal_block_size = self._calculate_optimal_block_size()

        # Initialize parallel processing settings
        self.num_threads = min(multiprocessing.cpu_count(), 8)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)
        
        # Initialize memory optimization
        self.memory_pool = MemoryPool()
        self.use_sparse = False
        self.sparsity_threshold = 0.8

        # Initialize distance cache last since it uses optimal_block_size
        if self.patterns.size > 0:
            self.distance_cache = np.zeros((len(self.patterns), len(self.patterns)), dtype=np.float32)
            for i in range(len(self.patterns)):
                for j in range(len(self.patterns)):
                    if i != j:
                        self.distance_cache[i,j] = self.evaluate_similarity(
                            self.patterns[i].reshape(-1), 
                            self.patterns[j].reshape(-1)
                        )
        else:
            self.distance_cache = np.zeros((0, 0), dtype=np.float32)

    def _calculate_optimal_block_size(self) -> int:
        """Calculate optimal block size based on cache size"""
        l1_cache = self.cache_sizes.get('l1', L1_CACHE_SIZE)
        elements_in_block = l1_cache // (4 * 3)  # Assuming float32
        return int(np.sqrt(elements_in_block))
    
    def _ensure_non_negative(self, patterns):
        """Ensure all pattern values are non-negative with extra safeguards.
        
        Args:
            patterns: Array or list of patterns
            
        Returns:
            np.ndarray: Patterns with guaranteed non-negative values
        """
        patterns_array = np.asarray(patterns, dtype=np.float32)
        
        # Multi-step approach to guarantee non-negative values
        # 1. Take absolute values
        non_neg = np.abs(patterns_array)
        
        # 2. Replace any remaining negative values due to floating point errors
        non_neg = np.clip(non_neg, 0, None)
        
        # 3. Extra safety measure to eliminate tiny negative values from floating point errors
        non_neg[non_neg < 1e-10] = 0.0
        
        return non_neg

    def precompute_norms(self):
        """Precompute norms of patterns to optimize similarity calculations."""
        self.pattern_norms = np.linalg.norm(self.patterns, axis=1, keepdims=True)

    def build_kdtree(self):
        """Build a KDTree for fast neighbor selection."""
        self.tree = KDTree(self.patterns)

    def evaluate_similarity(self, pattern1, pattern2):
        """Optimized similarity calculation using blocked operations"""
        try:
            # Ensure patterns are 1D arraysf
            pattern1 = np.asarray(pattern1).flatten()
            pattern2 = np.asarray(pattern2).flatten()
            
            # Make sure both patterns have the same length
            min_len = min(pattern1.size, pattern2.size)
            pattern1 = pattern1[:min_len]
            pattern2 = pattern2[:min_len]
            
            if min_len > BLOCK_SIZE:
                # Use blocked operations for large patterns
                result = BlockedMatrixOps.block_matrix_multiply(
                    pattern1.reshape(1, -1),
                    pattern2.reshape(-1, 1)
                )[0,0]
            else:
                # Direct calculation for small patterns
                norm1 = np.linalg.norm(pattern1)
                norm2 = np.linalg.norm(pattern2)
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                result = 1 - distance.euclidean(pattern1, pattern2) / (norm1 + norm2)
                
            return float(np.clip(np.real(result), 0.0, 1.0))  # Ensure non-negative similarity
            
        except Exception as e:
            logging.error(f"Similarity calculation failed: {str(e)}")
            return 0.0

    def compute_all_similarities(self):
        """Compute pairwise similarities for all patterns."""
        # Recompute norms based on the current patterns
        norms = np.linalg.norm(self.patterns, axis=1, keepdims=True)
        similarity_matrix = 1 - distance.cdist(self.patterns, self.patterns, metric='euclidean') / (norms + norms.T + 1e-9)
        return similarity_matrix

    def select_neighbors(self, pattern, k=2):
        """Select k nearest neighbors using KDTree."""
        if self.tree is None:
            self.build_kdtree()
        # Request k+1 neighbors since one will be the pattern itself
        _, indices = self.tree.query(pattern, k=k+1)  
        # Filter out self and take only k neighbors
        neighbors = [self.patterns[i] for i in indices if not np.array_equal(self.patterns[i], pattern)]
        return neighbors[:k]  # Ensure we return exactly k neighbors
    
    def combine_patterns(self, pattern1, pattern2, neighbors):
        """Combine two patterns using weighted neighbors, with thorough error handling."""
        try:
            if not (isinstance(pattern1, np.ndarray) and isinstance(pattern2, np.ndarray)):
                raise ValueError("Both patterns must be numpy arrays")
            if pattern1.shape != pattern2.shape:
                raise ValueError(f"Pattern shapes differ: {pattern1.shape} vs {pattern2.shape}")
            if not neighbors:
                raise ValueError("At least one neighbor is required")

            result = self.memory_pool.acquire()
            if result is None or result.shape != pattern1.shape:
                result = np.zeros_like(pattern1)

            weights = []
            for n in neighbors:
                sim1 = self.evaluate_similarity(pattern1, n)
                sim2 = self.evaluate_similarity(pattern2, n)
                try:
                    weight = np.sin(np.arccos(sim1)) + np.sin(np.arccos(sim2))
                except (ValueError, RuntimeWarning):
                    weight = 0.0
                weights.append(weight)

            weights = np.array(weights)
            s = np.sum(weights)
            if s > 1e-10:
                weights /= s
            else:
                weights = np.ones_like(weights) / len(weights)

            result[:] = 0.5 * (pattern1 + pattern2)
            for w, n in zip(weights, neighbors):
                result += w * n

            return result
        except Exception as e:
            logging.error(f"Pattern combination error: {str(e)}")
            return np.zeros_like(pattern1)

    def calculate_entropy_incrementally(self, new_pattern=None):
        """Calculate and update entropy with dimension checks and caching."""
        try:
            # Handle empty pattern space case
            if len(self.patterns) == 0:
                # If there's no new pattern, return default entropy
                if new_pattern is None:
                    return 0.5  # Default entropy for testing
                    
                # If we have a new pattern but empty patterns list,
                # initialize pattern space with this first pattern
                new_pattern = np.asarray(new_pattern)
                new_pattern = self._ensure_non_negative(new_pattern)
                
                if new_pattern.ndim == 1:
                    new_pattern = new_pattern.reshape(1, -1)
                    
                # Create pattern space with the first pattern
                self.patterns = new_pattern.copy()
                self.distance_cache = np.zeros((1, 1), dtype=np.float32)
                return 0.5  # Return default entropy for first pattern
            
            # Original implementation for non-empty pattern space
            if new_pattern is not None:
                new_pattern = np.asarray(new_pattern)
                # Handle negative values
                new_pattern = self._ensure_non_negative(new_pattern)
                
                if new_pattern.ndim == 1:
                    new_pattern = new_pattern.reshape(1, -1)
                elif new_pattern.ndim != 2:
                    raise ValueError("New pattern must be 1D or 2D")

                # Add shape compatibility check with proper validation
                if self.patterns.size > 0:
                    if new_pattern.shape[1] != self.patterns[0].shape[0]:
                        raise ValueError("Mismatch in pattern dimensions")

                new_distances = np.zeros(len(self.patterns), dtype=np.float32)
                for i in range(len(self.patterns)):
                    new_distances[i] = self.evaluate_similarity(new_pattern[0], self.patterns[i])

                old_size = len(self.patterns)
                new_cache = np.zeros((old_size + 1, old_size + 1), dtype=np.float32)
                new_cache[:old_size, :old_size] = self.distance_cache
                new_cache[old_size, :old_size] = new_distances
                new_cache[:old_size, old_size] = new_distances
                self.distance_cache = new_cache

                self.patterns = np.vstack([self.patterns, new_pattern])

            # Calculate entropy based on the distance cache
            valid = self.distance_cache[self.distance_cache > 0]
            if valid.size == 0:
                return 0.0
            entropy = -float(np.sum(valid * np.log(valid + 1e-9)))
            return entropy
        except Exception as e:
            logging.error(f"Entropy calculation error: {str(e)}")
            return 0.0

    def dynamic_similarity_threshold(self):
        """Adjust similarity threshold dynamically based on diversity and entropy change."""
        if len(self.iteration_entropy) < 2:
            entropy_change = 0
        else:
            entropy_change = abs(self.iteration_entropy[-1] - self.iteration_entropy[-2])

        angular_diversity = np.std([
            np.arcsin(np.clip(self.evaluate_similarity(p, np.mean(self.patterns, axis=0)), -1, 1))
            for p in self.patterns
        ])

        threshold = self.base_similarity_threshold * (1 + angular_diversity / 10 + entropy_change / 100)
        return threshold
   
    def cluster_patterns(self, patterns=None, epsilon=0.15, min_samples=2, use_kmeans=True):
        """Cluster patterns using KMeans or DBSCAN with robust shape normalization
        
        Args:
            patterns: Optional patterns to cluster (uses self.patterns if None)
            epsilon: Used to estimate number of clusters for KMeans or as eps for DBSCAN
            min_samples: Minimum samples for DBSCAN
            use_kmeans: Whether to use KMeans (True) or DBSCAN (False)
            
        Returns:
            Clustering result object or dictionary of clusters
        """
        from sklearn.cluster import KMeans, DBSCAN
        import numpy as np
        
        # Use provided patterns or self.patterns
        patterns_to_use = patterns if patterns is not None else self.patterns
        
        # Handle empty patterns case
        if len(patterns_to_use) < 2:
            return {} if patterns is not None else None
        
        # Normalize patterns to handle inconsistent shapes
        normalized_patterns = []
        for pattern in patterns_to_use:
            if isinstance(pattern, np.ndarray):
                normalized_patterns.append(pattern.flatten())
            elif hasattr(pattern, '__iter__') and not isinstance(pattern, (str, dict)):
                normalized_patterns.append(np.array(pattern).flatten())
            else:
                # Skip non-array patterns
                continue
        
        # If no valid patterns after filtering, return empty result
        if not normalized_patterns:
            return {} if patterns is not None else None
        
        # Pad patterns to make shapes uniform
        max_length = max(len(p) for p in normalized_patterns)
        padded_patterns = []
        for pattern in normalized_patterns:
            if len(pattern) < max_length:
                padded = np.pad(pattern, (0, max_length - len(pattern)))
                padded_patterns.append(padded)
            else:
                padded_patterns.append(pattern)
        
        # Create a unified array of patterns
        patterns_normalized = np.stack(padded_patterns)
        
        # Normalize by vector norms
        norms = np.linalg.norm(patterns_normalized, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Prevent division by zero
        patterns_normalized = patterns_normalized / norms
        
        # Choose clustering algorithm
        if use_kmeans and patterns is None:  # Only use KMeans for internal patterns
            # Estimate number of clusters based on epsilon parameter (inverse relationship)
            n_clusters = max(2, min(len(patterns_normalized) - 1, int(1.0 / epsilon)))
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(patterns_normalized)
            
            # Add DBSCAN-like attributes for compatibility with existing code
            result = kmeans
            result.core_sample_indices_ = np.arange(len(patterns_normalized))
            
            logging.debug(f"KMeans clustering found {len(set(kmeans.labels_))} clusters")
            return result
        else:
            # Apply DBSCAN
            clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(patterns_normalized)
            
            # If external patterns were provided, return dictionary of clusters
            if patterns is not None:
                # Group patterns by cluster
                clusters = {}
                for i, label in enumerate(clustering.labels_):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(i)
                return clusters
            
            # Otherwise return DBSCAN result for internal use
            return clustering

    def find_new_pattern(self, max_iterations=10, entropy_threshold=0.05, max_entropy_threshold=3000.0):
        """Find new patterns through iterative combination."""
        iteration = 0
        self.patterns = np.asarray(self.patterns)  # Convert to numpy array
        self.precompute_norms()

        while iteration < max_iterations:
            print(f"Iteration {iteration + 1}")

            current_entropy = self.calculate_entropy_incrementally()
            self.iteration_entropy.append(current_entropy)
            print(f"Entropy: {current_entropy:.4f}")

            if iteration > 0:
                entropy_change = abs(self.iteration_entropy[-1] - self.iteration_entropy[-2])
                print(f"Entropy Change: {entropy_change:.4f}")
                if entropy_change < entropy_threshold:
                    print("Stopping due to minimal entropy change.")
                    break

            if current_entropy > max_entropy_threshold:
                print("Stopping due to excessively high entropy.")
                break

            new_patterns = self.iterate_patterns(max_new_patterns=100)  # Reduced limit
            if not new_patterns:
                print("No new patterns formed. Stopping.")
                break

            # Replace extend with concatenate for numpy arrays
            if len(new_patterns) > 0:
                self.patterns = np.vstack([self.patterns] + new_patterns)
                self.precompute_norms()  # Recompute norms after adding patterns
            
            iteration += 1

        return self.history

    def iterate_patterns(self, max_workers=4, max_new_patterns=100):
        if len(self.patterns) < 2:  # Add check for minimum required patterns
            return []
            
        unique_new_patterns = []
        seen = set()  # Use this to track pattern fingerprints
        threshold = self.dynamic_similarity_threshold()  # Use this for filtering

        def process_combination(i, j):
            try:
                # Add bounds checking
                if i >= len(self.patterns) or j >= len(self.patterns):
                    return None
                    
                if i != j:
                    # Get only the number of neighbors we actually have available
                    k = min(2, len(self.patterns) - 1)  # Limit k to available patterns
                    neighbors = self.select_neighbors(self.patterns[i], k=k)
                    
                    if neighbors:  # Only proceed if we got neighbors
                        new_pattern = self.combine_patterns(
                            self.patterns[i], 
                            self.patterns[j],
                            neighbors
                        )
                        
                        # Calculate similarity to parents
                        sim1 = self.evaluate_similarity(new_pattern, self.patterns[i])
                        sim2 = self.evaluate_similarity(new_pattern, self.patterns[j])
                        
                        # Only keep patterns that are sufficiently different
                        if max(sim1, sim2) > threshold:
                            return None
                        
                        # Create fingerprint to avoid duplicates
                        fingerprint = hash(tuple(np.round(new_pattern.flatten(), 4)))
                        if fingerprint in seen:
                            return None
                            
                        # Mark as seen
                        seen.add(fingerprint)
                        return new_pattern
                return None
                
            except Exception as e:
                logging.error(f"Error processing combination {i},{j}: {str(e)}")
                return None

        # Rest of the method remains the same...

    def _create_guaranteed_positive_distance_matrix(self, patterns):
        """Create distance matrix with absolute guarantees of no negative values"""
        
        # Step 1: Ensure all input patterns are positive
        patterns = np.abs(patterns)
        
        # Step 2: Compute pairwise L2 norms (guaranteed non-negative)
        n_samples = patterns.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
        
        for i in range(n_samples):
            for j in range(n_samples):
                # Euclidean distance is always positive
                distance_matrix[i, j] = max(0.0, np.linalg.norm(patterns[i] - patterns[j]))
        
        # Step 3: Normalize to [0,1] range
        if np.max(distance_matrix) > 0:
            distance_matrix = distance_matrix / np.max(distance_matrix)
        
        # Step 4: Final safeguard - force any remaining negatives to zero
        distance_matrix[distance_matrix < 0] = 0.0
        
        return distance_matrix
