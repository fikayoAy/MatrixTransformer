"""tensortransform.py

Tensor to matrix conversion utilities with enhanced metadata preservation.
Provides bidirectional conversion between tensors of any dimension and 2D matrices.
"""

import numpy as np
import torch


def tensor_to_matrix(tensor):
    """
    Convert a tensor of any dimension to a 2D matrix representation with enhanced metadata.
    Preserves shape, energy, and structural information for accurate reconstruction.
    Including proper complex number handling.
    
    Args:
        tensor: Input tensor of any dimension
        
    Returns:
        tuple: (2D matrix representation, metadata dictionary)
    """
    # Handle None input
    if tensor is None:
        return np.array([[0.0]]), {
            'original_shape': (1, 1),
            'ndim': 2,
            'encoding_type': 'empty_tensor',
            'energy': 0.0
        }
            
    # Store original format information
    is_torch_tensor = isinstance(tensor, torch.Tensor)
    tensor_device = tensor.device if is_torch_tensor else None
    tensor_dtype = tensor.dtype
    
    # Convert tensor to numpy with proper complex handling
    if is_torch_tensor:
        # Use complex128 if tensor has complex values, otherwise float64
        if torch.is_complex(tensor):
            tensor_np = tensor.detach().cpu().numpy().astype(np.complex128)
        else:
            tensor_np = tensor.detach().cpu().numpy().astype(np.float64)
    else:
        # Use complex128 if input contains complex values
        if np.iscomplexobj(tensor):
            tensor_np = np.array(tensor, dtype=np.complex128)
        else:
            tensor_np = np.array(tensor, dtype=np.float64)
    
    # Store original shape and energy for reconstruction
    original_shape = tensor_np.shape
    original_energy = np.linalg.norm(tensor_np.reshape(-1))
    
    # Store comprehensive metadata
    metadata = {
        'original_shape': original_shape,
        'ndim': tensor_np.ndim,
        'is_torch': is_torch_tensor,
        'device': str(tensor_device) if tensor_device else None,
        'dtype': tensor_dtype,
        'energy': original_energy,
        'is_complex': np.iscomplexobj(tensor_np),
        'id': id(tensor)
    }

    # Handle empty tensor case
    if tensor_np.size == 0:
        metadata['encoding_type'] = 'empty_tensor'
        return np.array([[0.0]]), metadata

    # Handle different tensor dimensions with specialized representations
    if tensor_np.ndim == 1:
        # Convert 1D array to 2D by reshaping to column vector
        n = len(tensor_np)
        # Choose optimal 2D shape that's close to square
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Pad with zeros with proper dtype for complex values
        if np.iscomplexobj(tensor_np):
            padded = np.zeros(rows * cols, dtype=np.complex128)
        else:
            padded = np.zeros(rows * cols, dtype=np.float64)
        padded[:n] = tensor_np
        matrix_2d = padded.reshape(rows, cols)
        
        metadata['encoding_type'] = '1D_array'
        metadata['original_length'] = n
        metadata['matrix_rows'] = rows
        metadata['matrix_cols'] = cols
        
    elif tensor_np.ndim == 2:
        # Direct pass-through for 2D matrices
        matrix_2d = tensor_np.copy()
        metadata['encoding_type'] = '2D_direct'
        
    elif tensor_np.ndim == 3:
        # Enhanced 3D tensor to 2D using grid layout with detailed per-slice metadata
        depth, height, width = tensor_np.shape
        
        # Arrange slices in a grid
        grid_rows = int(np.ceil(np.sqrt(depth)))
        grid_cols = int(np.ceil(depth / grid_rows))
        
        # Create output matrix with proper dtype for complex values
        matrix_height = grid_rows * height
        matrix_width = grid_cols * width
        if np.iscomplexobj(tensor_np):
            matrix_2d = np.zeros((matrix_height, matrix_width), dtype=np.complex128)
        else:
            matrix_2d = np.zeros((matrix_height, matrix_width), dtype=np.float64)
        
        # Enhanced metadata with per-grid information
        grid_metadata = {}
        
        # Fill the grid with detailed metadata for each slice
        for i in range(depth):
            grid_row = i // grid_cols
            grid_col = i % grid_cols
            
            start_row = grid_row * height
            end_row = start_row + height
            start_col = grid_col * width
            end_col = start_col + width
            
            # Place slice in grid
            matrix_2d[start_row:end_row, start_col:end_col] = tensor_np[i]
            
            # Store detailed metadata for this slice
            slice_energy = np.linalg.norm(tensor_np[i])
            slice_mean = np.mean(np.abs(tensor_np[i]) if np.iscomplexobj(tensor_np[i]) else tensor_np[i])
            slice_std = np.std(np.abs(tensor_np[i]) if np.iscomplexobj(tensor_np[i]) else tensor_np[i])
            slice_sparsity = np.sum(np.abs(tensor_np[i]) < 1e-10) / tensor_np[i].size
            
            grid_metadata[f'slice_{i}'] = {
                'position': (grid_row, grid_col),
                'matrix_region': (start_row, end_row, start_col, end_col),
                'slice_energy': slice_energy,
                'slice_mean': slice_mean,
                'slice_std': slice_std,
                'slice_sparsity': slice_sparsity,
                'is_complex': np.iscomplexobj(tensor_np[i]),
                'processing_hints': {
                    'is_zero_slice': slice_energy < 1e-12,
                    'is_sparse': slice_sparsity > 0.8,
                    'is_uniform': slice_std < 1e-10
                }
            }
        
        # Enhanced metadata for 3D tensors
        metadata['encoding_type'] = '3D_grid_enhanced'
        metadata['depth'] = depth
        metadata['height'] = height
        metadata['width'] = width
        metadata['grid_rows'] = grid_rows
        metadata['grid_cols'] = grid_cols
        metadata['grid_metadata'] = grid_metadata
        metadata['total_slices'] = depth
        metadata['active_slices'] = sum(1 for gm in grid_metadata.values() if not gm['processing_hints']['is_zero_slice'])
        metadata['sparse_slices'] = sum(1 for gm in grid_metadata.values() if gm['processing_hints']['is_sparse'])
        metadata['uniform_slices'] = sum(1 for gm in grid_metadata.values() if gm['processing_hints']['is_uniform'])
        
        # Global statistics across all slices
        all_energies = [gm['slice_energy'] for gm in grid_metadata.values()]
        metadata['slice_energy_stats'] = {
            'min_energy': min(all_energies) if all_energies else 0.0,
            'max_energy': max(all_energies) if all_energies else 0.0,
            'mean_energy': np.mean(all_energies) if all_energies else 0.0,
            'std_energy': np.std(all_energies) if all_energies else 0.0
        }
        
    else:
        # ENHANCED: For 4D and higher, normalize before projection to preserve structure
        
        # Step 1: Normalize the tensor to preserve structural properties
        tensor_normalized = tensor_np.copy()
        tensor_norm = np.linalg.norm(tensor_normalized)
        
        if tensor_norm > 1e-10:
            # Normalize to unit energy, preserving relative magnitudes
            tensor_normalized = tensor_normalized / tensor_norm
        
        # Step 2: Store structural information before flattening
        # Capture important structural metrics
        structural_info = {
            'original_norm': float(tensor_norm),  # Ensure it's a Python float
            'shape_ratios': [float(tensor_np.shape[i] / tensor_np.shape[0]) for i in range(len(tensor_np.shape))],
            'axis_energies': [],
            'axis_means': [],
            'axis_stds': [],
            'is_complex': np.iscomplexobj(tensor_np)
        }
        
        # Calculate per-axis statistics to preserve structural information
        for axis in range(tensor_np.ndim):
            axis_data = np.mean(tensor_normalized, axis=tuple(i for i in range(tensor_np.ndim) if i != axis))
            structural_info['axis_energies'].append(float(np.linalg.norm(axis_data)))
            structural_info['axis_means'].append(float(np.mean(np.abs(axis_data) if np.iscomplexobj(axis_data) else axis_data)))
            structural_info['axis_stds'].append(float(np.std(np.abs(axis_data) if np.iscomplexobj(axis_data) else axis_data)))
        
        # Step 3: Flatten the normalized tensor and reshape to approximate square matrix
        flattened = tensor_normalized.reshape(-1)
        n = len(flattened)
        
        # Create approximately square matrix with proper dtype for complex values
        side = int(np.ceil(np.sqrt(n)))
        if np.iscomplexobj(tensor_normalized):
            padded = np.zeros(side * side, dtype=np.complex128)
        else:
            padded = np.zeros(side * side, dtype=np.float64)
        padded[:n] = flattened
        matrix_2d = padded.reshape(side, side)
        
        # Step 4: Store enhanced metadata with structural preservation info
        metadata['encoding_type'] = 'ND_projection_normalized'
        metadata['flattened_length'] = n
        metadata['matrix_side'] = side
        metadata['structural_info'] = structural_info
        metadata['normalization_applied'] = True
        
        # Additional structural preservation metadata
        metadata['dimension_products'] = [int(np.prod(tensor_np.shape[:i+1])) for i in range(len(tensor_np.shape))]
        metadata['cumulative_sizes'] = [int(x) for x in np.cumsum([np.prod(tensor_np.shape[i:]) for i in range(len(tensor_np.shape))])]
    
    return matrix_2d, metadata


def matrix_to_tensor(matrix, tensor_metadata=None, original_shape=None, original_dtype=None):
    """
    Convert a matrix back to its original tensor form using the enhanced metadata.
    Properly preserves complex number information.
    
    Args:
        matrix: The 2D matrix representation
        tensor_metadata: Metadata dictionary from tensor_to_matrix (can be nested dict with tensor IDs)
        original_shape: Optional shape override
        original_dtype: Optional dtype override
        
    Returns:
        Reconstructed tensor in its original format
    """
    if matrix is None:
        return np.array([])
    
    # Handle case where tensor_metadata is a nested dictionary (with tensor IDs as keys)
    metadata = None
    if tensor_metadata is not None:
        if isinstance(tensor_metadata, dict):
            if len(tensor_metadata) == 1 and 'encoding_type' not in tensor_metadata:
                # Nested dict case - extract the inner metadata
                metadata = list(tensor_metadata.values())[0]
            else:
                metadata = tensor_metadata
        else:
            metadata = None
    
    # Get target shape from parameters or metadata
    target_shape = None
    if isinstance(original_shape, (tuple, list)):
        target_shape = tuple(original_shape)
    elif metadata and 'original_shape' in metadata:
        target_shape = metadata['original_shape']
    
    # Get torch status and dtype from metadata if not provided directly
    is_torch = False
    device_str = None
    dtype = original_dtype
    encoding_type = None
    original_energy = None
    is_complex = np.iscomplexobj(matrix)
    
    # Extract metadata values
    if metadata:
        is_torch = metadata.get('is_torch', False)
        device_str = metadata.get('device', None)
        if dtype is None:
            dtype = metadata.get('dtype', None)
        encoding_type = metadata.get('encoding_type', None)
        original_energy = metadata.get('energy', None)
        # Check if original tensor was complex
        is_complex = is_complex or metadata.get('is_complex', False)
    
    # Ensure matrix has proper complex dtype if needed
    if is_complex and not np.iscomplexobj(matrix):
        matrix = matrix.astype(np.complex128)
    
    # Reconstruction approach based on encoding_type
    if encoding_type == 'empty_tensor':
        result = np.array([])
        if target_shape:
            result = np.zeros(target_shape, dtype=np.complex128 if is_complex else np.float64)
                
    elif encoding_type == '1D_array':
        # Reconstruct 1D array from 2D matrix
        if metadata:
            original_length = metadata.get('original_length', matrix.size)
            flattened = matrix.reshape(-1)
            result = flattened[:original_length]
        else:
            result = matrix.reshape(-1)
        
        # Reshape to target shape if provided
        if target_shape:
            try:
                result = result.reshape(target_shape)
            except ValueError:
                # If reshape fails, pad or truncate
                result_dtype = np.complex128 if is_complex else np.float64
                if len(result) < np.prod(target_shape):
                    padded = np.zeros(np.prod(target_shape), dtype=result_dtype)
                    padded[:len(result)] = result
                    result = padded.reshape(target_shape)
                else:
                    result = result[:np.prod(target_shape)].reshape(target_shape)
                    
    elif encoding_type == '2D_direct':
        # Direct copy for 2D matrices
        result = matrix.copy()
        if target_shape and result.shape != target_shape:
            try:
                result = result.reshape(target_shape)
            except ValueError:
                # Handle size mismatch by padding/truncating
                result_dtype = np.complex128 if is_complex else np.float64
                if result.size < np.prod(target_shape):
                    padded = np.zeros(target_shape, dtype=result_dtype)
                    min_shape = tuple(min(a, b) for a, b in zip(result.shape, target_shape))
                    padded[:min_shape[0], :min_shape[1]] = result[:min_shape[0], :min_shape[1]]
                    result = padded
                else:
                    result = result[:target_shape[0], :target_shape[1]]
                    
    elif encoding_type == '3D_grid_enhanced':
        # Reconstruct 3D tensor from grid layout using enhanced metadata
        if metadata:
            depth = metadata.get('depth', 1)
            height = metadata.get('height', 1) 
            width = metadata.get('width', 1)
            grid_rows = metadata.get('grid_rows', 1)
            grid_cols = metadata.get('grid_cols', 1)
            grid_metadata = metadata.get('grid_metadata', {})
            
            # Create result array with proper dtype
            result_dtype = np.complex128 if is_complex else np.float64
            result = np.zeros((depth, height, width), dtype=result_dtype)
            
            for i in range(depth):
                slice_is_complex = False
                if f'slice_{i}' in grid_metadata:
                    slice_is_complex = grid_metadata[f'slice_{i}'].get('is_complex', False)
                
                grid_row = i // grid_cols
                grid_col = i % grid_cols
                
                start_row = grid_row * height
                end_row = start_row + height
                start_col = grid_col * width
                end_col = start_col + width
                
                # Extract slice from matrix, ensuring complex values are preserved
                slice_data = matrix[start_row:end_row, start_col:end_col]
                if slice_is_complex and not np.iscomplexobj(slice_data):
                    slice_data = slice_data.astype(np.complex128)
                    
                result[i] = slice_data
        else:
            # Fallback reconstruction
            result = matrix.copy()
            
    elif encoding_type in ['ND_projection', 'ND_projection_normalized']:
        # ENHANCED: Reconstruct N-D tensor from normalized projection
        if metadata:
            flattened_length = metadata.get('flattened_length', matrix.size)
            structural_info = metadata.get('structural_info', {})
            original_norm = structural_info.get('original_norm', 1.0) if structural_info else 1.0
            normalization_applied = metadata.get('normalization_applied', False)
            original_is_complex = structural_info.get('is_complex', False) if structural_info else False
            
            # Extract flattened data with proper dtype
            flattened = matrix.reshape(-1)[:flattened_length]
            
            # Fix: Improved complex number handling
            if metadata.get('is_complex', False) or original_is_complex or np.iscomplexobj(matrix):
                flattened = flattened.astype(np.complex128)
            else:
                flattened = flattened.astype(np.float64)
            
            # Reshape to target shape
            if target_shape:
                try:
                    result = flattened.reshape(target_shape)
                    
                    # If normalization was applied, restore original energy with high precision
                    if normalization_applied and original_norm > 1e-10:
                        current_norm = np.linalg.norm(result)
                        if current_norm > 1e-10:
                            # Use high precision scaling that preserves complex phase relationships
                            scale_factor = original_norm / current_norm
                            result = result * scale_factor
                            
                except ValueError:
                    # Fallback if reshape fails
                    result_dtype = np.complex128 if (is_complex or original_is_complex) else np.float64
                    if len(flattened) < np.prod(target_shape):
                        padded = np.zeros(np.prod(target_shape), dtype=result_dtype)
                        padded[:len(flattened)] = flattened
                        result = padded.reshape(target_shape)
                    else:
                        result = flattened[:np.prod(target_shape)].reshape(target_shape)
                        
                    # Apply energy restoration even in fallback case
                    if normalization_applied and original_norm > 1e-10:
                        current_norm = np.linalg.norm(result)
                        if current_norm > 1e-10:
                            # This scaling preserves complex phase relationships
                            scale_factor = original_norm / current_norm
                            result = result * scale_factor
            else:
                result = flattened.copy()
        else:
            # Fallback reconstruction
            result = matrix.reshape(-1)
            if target_shape:
                try:
                    result = result.reshape(target_shape)
                except ValueError:
                    result = result[:np.prod(target_shape)].reshape(target_shape)
    else:
        # Fallback reconstruction
        if target_shape:
            try:
                result = matrix.reshape(target_shape)
            except ValueError:
                result = matrix.copy()
        else:
            result = matrix.copy()
    
    # FIX: Preserve data type with higher precision for PyTorch tensors
    if dtype is not None:
        try:
            # For PyTorch tensors, ensure we maintain precision and complex data
            if is_torch and hasattr(dtype, 'torch'):
                # Keep as complex if needed
                if np.iscomplexobj(result):
                    result = result.astype(np.complex128)
                else:
                    result = result.astype(np.float64)
            else:
                # Preserve complex type if present
                if np.iscomplexobj(result) or (dtype == np.complex64 or dtype == np.complex128):
                    result = result.astype(np.complex128 if dtype == np.complex128 else np.complex64)
                else:
                    result = result.astype(dtype)
        except (TypeError, ValueError, AttributeError):
            # If conversion fails, keep current dtype
            pass
    
    # FIX: Convert back to torch tensor with proper precision and complex handling
    if is_torch:
        try:
            # Ensure high precision conversion and complex support
            if not isinstance(result, torch.Tensor):
                # Convert with explicit dtype to maintain precision and complex values
                if np.iscomplexobj(result):
                    if dtype is not None and dtype in (torch.complex64, torch.complex128):
                        result = torch.tensor(result, dtype=dtype)
                    else:
                        result = torch.tensor(result, dtype=torch.complex128)
                elif dtype is not None:
                    result = torch.tensor(result, dtype=dtype)
                else:
                    result = torch.tensor(result, dtype=torch.float64)
            
            # Move to correct device
            if device_str and device_str != 'None':
                result = result.to(device_str)
        except Exception:
            # If conversion fails, return numpy array
            pass
    
    return result
