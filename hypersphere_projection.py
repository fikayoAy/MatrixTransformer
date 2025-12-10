"""hypersphere_projection.py

Hypersphere projection utilities for matrix and tensor processing.
Provides projection, logarithmic map, exponential map, parallel transport,
and distance computation on the hypersphere.

The geometry is now ADAPTIVE - radius can be:
- Set globally via set_sphere_radius() or DEFAULT_SPHERE_RADIUS
- Computed from data via compute_adaptive_radius()
- Passed per-call to any function

GPU-ACCELERATED VERSION using PyTorch for all operations.
"""

import logging
import math
import numpy as np
import torch

from tensortransform import tensor_to_matrix, matrix_to_tensor
from torch_utils import to_torch, to_numpy, DEVICE

# ============================================================================
# ADAPTIVE GEOMETRY CONFIGURATION
# ============================================================================

# Default sphere radius - can be modified globally or per-call
DEFAULT_SPHERE_RADIUS = 7.0

# Geometry mode: 'fixed', 'adaptive', 'learned'
_GEOMETRY_MODE = 'fixed'

# For adaptive mode: statistics from data
_ADAPTIVE_STATS = {
    'mean_norm': None,
    'std_norm': None,
    'min_norm': None,
    'max_norm': None,
    'computed_radius': None,
}


def set_sphere_radius(radius: float):
    """Set the global default sphere radius."""
    global DEFAULT_SPHERE_RADIUS
    DEFAULT_SPHERE_RADIUS = float(radius)
    logging.info(f"Sphere radius set to {DEFAULT_SPHERE_RADIUS}")


def get_sphere_radius() -> float:
    """Get the current global sphere radius."""
    return DEFAULT_SPHERE_RADIUS


def set_geometry_mode(mode: str):
    """
    Set the geometry mode.
    
    Args:
        mode: One of:
            - 'fixed': Use DEFAULT_SPHERE_RADIUS for all operations
            - 'adaptive': Compute radius from data statistics
            - 'learned': Allow radius to be updated during training
    """
    global _GEOMETRY_MODE
    if mode not in ('fixed', 'adaptive', 'learned'):
        raise ValueError(f"Unknown geometry mode: {mode}. Use 'fixed', 'adaptive', or 'learned'")
    _GEOMETRY_MODE = mode
    logging.info(f"Geometry mode set to '{mode}'")


def compute_adaptive_radius(data, method='mean_norm', scale_factor=1.0):
    """
    Compute an adaptive radius from data.
    
    This allows the hypersphere to adapt to the natural scale of your data
    rather than forcing everything to a fixed radius.
    
    Args:
        data: torch tensor, numpy array, or list of arrays to analyze
        method: How to compute radius:
            - 'mean_norm': Use mean of Frobenius norms
            - 'max_norm': Use maximum norm (ensures all data fits)
            - 'std_norm': Use mean + 2*std (covers 95% of data)
            - 'percentile_95': Use 95th percentile of norms
            - 'sqrt_dim': Use sqrt(dimensionality) * scale_factor
        scale_factor: Multiply computed radius by this factor
        
    Returns:
        Computed radius value
    """
    global DEFAULT_SPHERE_RADIUS, _ADAPTIVE_STATS
    
    # Collect norms (convert to torch for computation)
    if isinstance(data, (list, tuple)):
        norms = []
        dims = []
        for d in data:
            arr = to_torch(d)
            norms.append(torch.norm(arr).item())
            dims.append(arr.numel())
    else:
        arr = to_torch(data)
        if arr.ndim == 1:
            norms = [torch.norm(arr).item()]
            dims = [arr.numel()]
        elif arr.ndim == 2:
            norms = [torch.norm(arr).item()]
            dims = [arr.numel()]
        else:
            # Batch of matrices
            norms = [torch.norm(arr[i]).item() for i in range(arr.shape[0])]
            dims = [arr[0].numel()] * len(norms)
    
    norms = torch.tensor(norms, device=DEVICE)
    dims = torch.tensor(dims, device=DEVICE)
    
    # Store statistics
    _ADAPTIVE_STATS['mean_norm'] = float(torch.mean(norms).item())
    _ADAPTIVE_STATS['std_norm'] = float(torch.std(norms).item())
    _ADAPTIVE_STATS['min_norm'] = float(torch.min(norms).item())
    _ADAPTIVE_STATS['max_norm'] = float(torch.max(norms).item())
    
    # Compute radius based on method
    if method == 'mean_norm':
        radius = torch.mean(norms).item()
    elif method == 'max_norm':
        radius = torch.max(norms).item()
    elif method == 'std_norm':
        radius = (torch.mean(norms) + 2 * torch.std(norms)).item()
    elif method == 'percentile_95':
        radius = torch.quantile(norms, 0.95).item()
    elif method == 'sqrt_dim':
        radius = torch.sqrt(torch.mean(dims.float())).item()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    radius = float(radius * scale_factor)
    
    # Ensure minimum radius
    radius = max(radius, 1e-6)
    
    _ADAPTIVE_STATS['computed_radius'] = radius
    
    if _GEOMETRY_MODE == 'adaptive':
        DEFAULT_SPHERE_RADIUS = radius
        logging.info(f"Adaptive radius computed: {radius:.4f} (method={method}, scale={scale_factor})")
    
    return radius


def get_adaptive_stats():
    """Get statistics from the last adaptive radius computation."""
    return _ADAPTIVE_STATS.copy()


def _resolve_radius(radius=None):
    """Internal helper to resolve radius from argument or global default."""
    if radius is not None:
        return float(radius)
    return DEFAULT_SPHERE_RADIUS


def project_to_hypersphere(matrix, radius=1.0, preserve_type=True, batch_size=None, 
                           use_memmap=False, memmap_dir=None,
                           tensor_to_matrix_fn=None, matrix_to_tensor_fn=None,
                           detect_matrix_type_fn=None, get_transform_method_fn=None,
                           local_distance_sphere_fn=None, projection_distances=None,
                           projection_distances_global=None, project_2d_fn=None):
    """
    Project matrix to hypersphere with given radius, preserving structure.
    Works with tensors of any dimension, using the enhanced tensor_to_matrix system.
    
    Args:
        matrix: Input matrix or tensor of any dimension
        radius: Target radius (Frobenius norm)
        preserve_type: Whether to preserve matrix type properties
        batch_size: Batch size for processing multiple matrices
        use_memmap: Whether to use memory-mapped files for large batches
        memmap_dir: Directory for memmap files
        tensor_to_matrix_fn: Function to convert tensor to matrix (default: tensortransform.tensor_to_matrix)
        matrix_to_tensor_fn: Function to convert matrix to tensor (default: tensortransform.matrix_to_tensor)
        detect_matrix_type_fn: Function to detect matrix type (optional, for preserve_type)
        get_transform_method_fn: Function to get transform method (optional, for preserve_type)
        local_distance_sphere_fn: Function to compute local sphere distance (optional, for diagnostics)
        projection_distances: List to append projection distances (optional)
        projection_distances_global: List to append global projection distances (optional)
        project_2d_fn: Optional custom 2D projection function (ignored, for compatibility)
        
    Returns:
        Matrix/tensor projected to hypersphere with specified radius
    """
    # Use provided functions or defaults
    _tensor_to_matrix = tensor_to_matrix_fn or tensor_to_matrix
    _matrix_to_tensor = matrix_to_tensor_fn or matrix_to_tensor
    
    # Batch handling: if a list/tuple of matrices is provided, process in chunks
    is_list_input = isinstance(matrix, (list, tuple))
    # If a numpy array with ndim==3 and batch_size set, treat it as a batch of matrices
    if isinstance(matrix, np.ndarray) and matrix.ndim == 3 and batch_size is not None:
        is_list_input = True
    if is_list_input:
        matrices = list(matrix) if not isinstance(matrix, np.ndarray) else [matrix[i] for i in range(matrix.shape[0])]
        N = len(matrices)
        if batch_size is None:
            batch_size_calc = min(256, N)
        else:
            batch_size_calc = batch_size
        results = []
        # Setup memmap output if requested
        memmap_tmpdir = None
        results_mmap = None
        try:
            if use_memmap:
                import tempfile, os
                memmap_tmpdir = memmap_dir or tempfile.mkdtemp(prefix='tp_proj_')
                # Determine shape for memmap if possible: try to compute flattened lengths
                flat_lens = []
                for m in matrices:
                    try:
                        m_np = m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m)
                        flat_lens.append(m_np.size)
                    except Exception:
                        flat_lens.append(0)
                max_len = max(flat_lens) if flat_lens else 0
                if max_len > 0:
                    results_mmap = np.memmap(os.path.join(memmap_tmpdir, 'projected.dat'), dtype=np.float32, mode='w+', shape=(N, max_len))
            for i in range(0, N, batch_size_calc):
                j = min(i + batch_size_calc, N)
                batch = matrices[i:j]
                # If all shapes are identical, we can stack and vectorize
                homogeneous = True
                shapes = [ (np.asarray(m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else m).shape) for m in batch]
                for s in shapes[1:]:
                    if s != shapes[0]:
                        homogeneous = False
                        break
                if homogeneous and len(shapes[0]) <= 2:
                    # stack and use vectorized 2D helper
                    stacked = np.stack([np.asarray(m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else m, dtype=float) for m in batch], axis=0)
                    # If stacked.ndim == 3 and consistent, vectorize processing
                    proj_stack = project_2d_matrix_to_hypersphere(
                        stacked, radius=radius, preserve_type=preserve_type,
                        detect_matrix_type_fn=detect_matrix_type_fn,
                        get_transform_method_fn=get_transform_method_fn,
                        local_distance_sphere_fn=local_distance_sphere_fn,
                        projection_distances=projection_distances,
                        projection_distances_global=projection_distances_global
                    )
                    # Unpack results
                    for k in range(proj_stack.shape[0]):
                        res = proj_stack[k]
                        if results_mmap is not None:
                            flat = np.asarray(res).flatten()
                            results_mmap[i + k, :flat.size] = flat.astype(np.float32)
                        else:
                            results.append(res)
                else:
                    # heterogeneous batch; fallback to per-element processing
                    for k, m in enumerate(batch):
                        res = project_to_hypersphere(
                            m, radius=radius, preserve_type=preserve_type,
                            tensor_to_matrix_fn=_tensor_to_matrix,
                            matrix_to_tensor_fn=_matrix_to_tensor,
                            detect_matrix_type_fn=detect_matrix_type_fn,
                            get_transform_method_fn=get_transform_method_fn,
                            local_distance_sphere_fn=local_distance_sphere_fn,
                            projection_distances=projection_distances,
                            projection_distances_global=projection_distances_global
                        )
                        if results_mmap is not None:
                            flat = np.asarray(res).flatten()
                            results_mmap[i + k, :flat.size] = flat.astype(np.float32)
                        else:
                            results.append(res)
            return results_mmap if results_mmap is not None else results
        finally:
            # If memmap was used, keep files on disk; do not auto-delete to allow downstream reading
            pass

    # Handle scalar and None inputs
    if matrix is None:
        return None
        
    if isinstance(matrix, (int, float)):
        # For scalars, simply scale to radius
        return radius if matrix != 0 else radius  # Nonzero value with proper sign
    
    # Store original format information
    original_is_tensor = isinstance(matrix, torch.Tensor)
    original_device = matrix.device if original_is_tensor else None
    original_shape = matrix.shape
    original_ndim = len(original_shape)
    original_dtype = matrix.dtype
    
    # Convert to numpy for processing
    if original_is_tensor:
        matrix_np = matrix.detach().cpu().numpy()
    else:
        matrix_np = matrix
    
    # Handle empty arrays
    if matrix_np.size == 0:
        return matrix
    
    # For higher dimensional tensors (>2D), use tensor_to_matrix
    if original_ndim > 2:
        # Convert to 2D matrix representation
        matrix_2d, tensor_metadata = _tensor_to_matrix(matrix_np)
        
        # Project the 2D representation to the hypersphere
        projected_2d = project_2d_matrix_to_hypersphere(
            matrix_2d, radius, preserve_type,
            detect_matrix_type_fn=detect_matrix_type_fn,
            get_transform_method_fn=get_transform_method_fn,
            local_distance_sphere_fn=local_distance_sphere_fn,
            projection_distances=projection_distances,
            projection_distances_global=projection_distances_global
        )
        
        # Convert back to original tensor form
        result = _matrix_to_tensor(projected_2d, tensor_metadata, original_shape=original_shape)
    else:
        # For 1D and 2D matrices, use direct projection
        result = project_2d_matrix_to_hypersphere(
            matrix_np, radius, preserve_type,
            detect_matrix_type_fn=detect_matrix_type_fn,
            get_transform_method_fn=get_transform_method_fn,
            local_distance_sphere_fn=local_distance_sphere_fn,
            projection_distances=projection_distances,
            projection_distances_global=projection_distances_global
        )

    # Compute and store local spherical distance diagnostics (mirror of 2D helper)
    if local_distance_sphere_fn is not None:
        try:
            flat = np.array(result).flatten()
            fnorm = np.linalg.norm(flat)
            if fnorm > 1e-10:
                x = flat / fnorm
                x0 = np.ones_like(x)
                x0 = x0 / (np.linalg.norm(x0) + 1e-12)
                dist = float(local_distance_sphere_fn(x0, x))
                if projection_distances is not None:
                    try:
                        projection_distances.append(dist)
                    except Exception:
                        pass
                if projection_distances_global is not None:
                    try:
                        projection_distances_global.append(dist)
                    except Exception:
                        pass
        except Exception:
            pass

    # Convert back to original format
    if original_is_tensor:
        try:
            result = torch.tensor(result, device=original_device, dtype=original_dtype)
        except:
            logging.warning("Failed to convert result back to PyTorch tensor")

    return result


def project_2d_matrix_to_hypersphere(matrix, radius=7, preserve_type=True,
                                      detect_matrix_type_fn=None, get_transform_method_fn=None,
                                      local_distance_sphere_fn=None,
                                      projection_distances=None, projection_distances_global=None):
    """
    Project a 2D matrix to a hypersphere with given radius.
    Helper method for project_to_hypersphere.
    
    Args:
        matrix: 2D numpy array or 1D vector
        radius: Target radius (Frobenius norm)
        preserve_type: Whether to preserve matrix type properties
        detect_matrix_type_fn: Function to detect matrix type (optional)
        get_transform_method_fn: Function to get transform method (optional)
        local_distance_sphere_fn: Function to compute local sphere distance (optional)
        projection_distances: List to append projection distances (optional)
        projection_distances_global: List to append global projection distances (optional)
        
    Returns:
        2D numpy array or 1D vector projected to hypersphere
    """
    original_shape = matrix.shape
    original_dtype = matrix.dtype
    original_ndim = len(original_shape)

    # If a batch of matrices is passed in as a 3D array (B, H, W) or (B, L), handle vectorized processing
    if isinstance(matrix, np.ndarray) and matrix.ndim == 3:
        # Matrix stack: (B, H, W) or (B, L)
        B = matrix.shape[0]
        # Convert to torch for GPU computation
        mat_torch = to_torch(matrix)
        # Flatten per-matrix
        flat = mat_torch.reshape(B, -1)
        norms = torch.norm(flat, dim=1)
        result_stack = torch.empty_like(flat)
        small_mask = norms < 1e-10
        # For small norms, fill with scaled ones
        sizes = flat.shape[1]
        if torch.any(small_mask):
            result_stack[small_mask] = torch.ones((torch.sum(small_mask).item(), sizes), device=DEVICE, dtype=flat.dtype) * (radius / torch.sqrt(torch.tensor(sizes, device=DEVICE, dtype=flat.dtype)))
        # For valid norms, scale
        valid_mask = ~small_mask
        if torch.any(valid_mask):
            result_stack[valid_mask] = flat[valid_mask] * (radius / norms[valid_mask].unsqueeze(1))
        # Reshape back to original per-matrix shapes
        result_stack = result_stack.reshape(mat_torch.shape)
        # Convert back to numpy for type preservation operations
        result_stack_np = to_numpy(result_stack)
        # If preserve_type True, we need to apply per-matrix transform for square matrices
        if preserve_type and detect_matrix_type_fn is not None and get_transform_method_fn is not None:
            for i in range(B):
                mat_i = result_stack_np[i]
                if mat_i.shape[0] == mat_i.shape[1]:
                    mt = detect_matrix_type_fn(mat_i)
                    tr = get_transform_method_fn(mt)
                    if tr:
                        try:
                            result_stack_np[i] = tr(mat_i)
                        except Exception:
                            pass
            # Enforce final exact radius after transforming
            result_stack_torch = to_torch(result_stack_np)
            for i in range(B):
                final_norm = torch.norm(result_stack_torch[i])
                if final_norm > 1e-10:
                    result_stack_torch[i] = result_stack_torch[i] * (radius / final_norm)
            result_stack_np = to_numpy(result_stack_torch)
        return result_stack_np.astype(original_dtype)
    
    # Convert to torch for GPU computation
    mat_torch = to_torch(matrix)
    
    # Handle 1D vectors by reshaping to 2D for consistent processing
    if original_ndim == 1:
        mat_torch = mat_torch.reshape(-1, 1)
    
    # Calculate current Frobenius norm
    current_norm = torch.norm(mat_torch)
    
    # Handle near-zero matrices
    if current_norm < 1e-10:
        # Create a non-zero matrix with the desired norm
        result = torch.ones_like(mat_torch, dtype=torch.float32) * (radius / torch.sqrt(torch.tensor(mat_torch.numel(), device=DEVICE, dtype=torch.float32)))
    else:
        # Scale matrix to have desired norm
        result = mat_torch.float() * (radius / current_norm)
    
    # Convert back to numpy for type preservation operations
    result_np = to_numpy(result)
    
    # Apply type preservation if requested (only for square matrices)
    if preserve_type and result_np.shape[0] == result_np.shape[1]:
        if detect_matrix_type_fn is not None and get_transform_method_fn is not None:
            matrix_type = detect_matrix_type_fn(result_np)
            transform_method = get_transform_method_fn(matrix_type)
            if transform_method:
                result_np = transform_method(result_np)
                result = to_torch(result_np)
    
    # CRITICAL FIX: Always ensure the exact radius at the end
    # This must be the final operation before returning
    final_norm = torch.norm(result)
    if final_norm > 1e-10:
        # Force exact scaling to radius with no other operations after this
        result = result * (radius / final_norm)
    
    # Convert back to numpy for final operations
    result_np = to_numpy(result)
    
    # Compute local spherical distance from a canonical reference direction
    if local_distance_sphere_fn is not None:
        try:
            flat = result_np.flatten()
            fnorm = np.linalg.norm(flat)
            if fnorm > 1e-10:
                x = flat / fnorm
                x0 = np.ones_like(x)
                x0 = x0 / (np.linalg.norm(x0) + 1e-12)
                # local distance on sphere (in radians)
                dist = float(local_distance_sphere_fn(x0, x))
                # store per-instance and class-global diagnostics (safe, non-blocking)
                if projection_distances is not None:
                    try:
                        projection_distances.append(dist)
                    except Exception:
                        pass
                if projection_distances_global is not None:
                    try:
                        projection_distances_global.append(dist)
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Restore original shape if the input was 1D
    if original_ndim == 1:
        result_np = result_np.reshape(original_shape)
    
    # Don't cast back to integer types - keep as float
    if np.issubdtype(original_dtype, np.integer):
        return result_np.astype(np.float64)
    
    return result_np.astype(original_dtype)


def log_map_sphere(x0, x, eps=1e-7, batch_size=256, radius=None):
    """
    Compute the logarithmic map on the hypersphere.

    Supports both single vector inputs (1D arrays) and batched inputs (2D arrays)
    across the first dimension. If the inputs are batched and have more than
    `batch_size` vectors, the function will process them in chunks.

    Args:
        x0: numpy array of shape (D,) or (N, D) -- base vector(s) on sphere
        x: numpy array of shape (D,) or (N, D) -- target vector(s) on sphere
        eps: numeric epsilon threshold for numerical stability.
        batch_size: maximum per-chunk batch size when processing many vectors.
        radius: sphere radius (default: uses global DEFAULT_SPHERE_RADIUS)
                Set to None for adaptive, or pass explicit value.

    Returns:
        Tangent vector(s) at x0 with shape matching x (and x0 when batched).
        The norm of the tangent vector equals the geodesic distance.
    """
    # Resolve radius
    sphere_radius = _resolve_radius(radius)
    x0 = np.asarray(x0, dtype=float)
    x = np.asarray(x, dtype=float)
    logging.debug(f"log_map_sphere: x0 shape={x0.shape}, norm={np.linalg.norm(x0):.6f}, first 3 elements={x0.ravel()[:3]}")
    logging.debug(f"log_map_sphere: x shape={x.shape}, norm={np.linalg.norm(x) if x.ndim == 1 else np.linalg.norm(x, axis=1).mean():.6f}, first 3 elements={x.ravel()[:3]}")
    # Use theta-based tolerance: increased to 1e-6 to handle actual data with angles ~6.6e-7
    # This prevents treating valid small angles as identical points
    theta_identical_tol = max(1e-6, eps)
    antipodal_tol = max(1e-9, eps * 10)

    # Determine whether this is a single-vector case (both inputs 1D)
    single_case = (x.ndim == 1 and x0.ndim == 1)

    if single_case:
        # Convert to torch for GPU computation
        x0_torch = to_torch(x0)
        x_torch = to_torch(x)
        
        # Normalize to unit sphere for angular computation
        x0_norm = torch.norm(x0_torch)
        x_norm = torch.norm(x_torch)
        if x0_norm < eps or x_norm < eps:
            return to_numpy(torch.zeros_like(x0_torch))
        
        x0_unit = x0_torch / x0_norm
        x_unit = x_torch / x_norm
        
        # Compute dot product of unit vectors
        dot = torch.clamp(torch.dot(x0_unit, x_unit), -1.0, 1.0)
        
        # Use numerically stable formula for angle: theta = arccos(dot)
        # For small angles, use chord distance formula: theta = 2 * arcsin(||x - x0|| / 2)
        if dot > 0.9:
            # Use chord distance for numerical stability when points are close
            chord_dist = torch.norm(x_unit - x0_unit)
            theta = 2.0 * torch.asin(torch.clamp(chord_dist / 2.0, 0.0, 1.0))
        else:
            theta = torch.acos(dot)
        
        # Safely format dot and theta (could be arrays or scalars)
        dot_val = float(dot.item())
        theta_val = float(theta.item())
        logging.debug(f"log_map_sphere (single): dot={dot_val:.6f}, theta={theta_val:.6e}, theta_tol={theta_identical_tol:.6e}")
        # Use the actual scalar value for comparison
        theta_scalar = float(theta.item())
        if theta_scalar < theta_identical_tol:
            logging.debug(f"log_map_sphere (single): Points are identical (theta={theta_scalar:.6e} < {theta_identical_tol:.6e}), returning zero vector")
            return to_numpy(torch.zeros_like(x0_torch))
        if torch.abs(dot + 1.0) < antipodal_tol:
            # Antipodal: direction undefined, choose arbitrary tangent direction with norm=pi
            # Find an orthogonal direction
            D = len(x0_unit)
            if abs(x0_unit[0].item()) < 0.9:
                v_dir = torch.zeros_like(x0_unit)
                v_dir[0] = 1.0
            else:
                v_dir = torch.zeros_like(x0_unit)
                v_dir[1] = 1.0
            # Make it orthogonal to x0_unit
            v_dir = v_dir - torch.dot(v_dir, x0_unit) * x0_unit
            v_dir = v_dir / (torch.norm(v_dir) + eps)
            return to_numpy(np.pi * sphere_radius * v_dir)
        
        # Compute direction in tangent space (orthogonal component of x_unit from x0_unit)
        v_dir = x_unit - dot * x0_unit
        v_dir_norm = torch.norm(v_dir)
        logging.debug(f"log_map_sphere (single): v_dir_norm={v_dir_norm.item():.6e}")
        if v_dir_norm < theta_identical_tol:
            logging.debug(f"log_map_sphere (single): v_dir_norm too small ({v_dir_norm.item():.9e}), returning zero vector")
            return to_numpy(torch.zeros_like(x0_torch))
        v_dir = v_dir / v_dir_norm
        
        # Result: tangent vector with norm = geodesic distance = theta * radius
        result = theta * sphere_radius * v_dir
        logging.debug(f"log_map_sphere (single): result norm={torch.norm(result).item():.6f} (scaled by radius={sphere_radius})")
        return to_numpy(result)
    
    # Batch/mixed case: ensure both operands broadcast to (n, D)
    # Convert to torch for GPU computation
    x_torch = to_torch(x)
    x0_torch = to_torch(x0)
    
    if x_torch.ndim == 1:
        x_mat = x_torch.unsqueeze(0)
    else:
        x_mat = x_torch
    if x0_torch.ndim == 1:
        x0_mat = x0_torch.unsqueeze(0)
    else:
        x0_mat = x0_torch

    # Determine target batch size and broadcast as needed
    n = max(x_mat.shape[0], x0_mat.shape[0])
    D = x_mat.shape[1] if x_mat.ndim == 2 else x_mat.shape[-1]

    if x_mat.shape[0] != n:
        x_mat = x_mat.expand(n, D).clone()
    if x0_mat.shape[0] != n:
        x0_mat = x0_mat.expand(n, D).clone()

    # Normalize to unit sphere for angular computation
    x0_norms = torch.norm(x0_mat, dim=1, keepdim=True)
    x_norms = torch.norm(x_mat, dim=1, keepdim=True)
    
    # Handle zero vectors
    x0_norms = torch.maximum(x0_norms, torch.tensor(eps, device=DEVICE))
    x_norms = torch.maximum(x_norms, torch.tensor(eps, device=DEVICE))
    
    x0_unit = x0_mat / x0_norms
    x_unit = x_mat / x_norms

    # Compute dot product of unit vectors
    dot = torch.sum(x0_unit * x_unit, dim=1)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Compute angle using arccos (safe since dot is clipped)
    theta = torch.acos(dot)
    
    mask_identical = theta < theta_identical_tol
    mask_antipodal = torch.abs(dot + 1.0) < antipodal_tol
    
    # Compute tangent direction (orthogonal component of x_unit from x0_unit)
    v_dir = x_unit - dot.unsqueeze(1) * x0_unit
    v_dir_norm = torch.norm(v_dir, dim=1)
    v = torch.zeros_like(x_mat)
    mask = ~(mask_identical | mask_antipodal | (v_dir_norm < theta_identical_tol))
    logging.debug(f"log_map_sphere (batch): n={n}, identical={mask_identical.sum().item()}, antipodal={mask_antipodal.sum().item()}, valid={mask.sum().item()}")
    logging.debug(f"log_map_sphere (batch): dot range=[{dot.min().item():.6f}, {dot.max().item():.6f}], theta range=[{theta.min().item():.6e}, {theta.max().item():.6e}]")
    logging.debug(f"log_map_sphere (batch): v_dir_norm range=[{v_dir_norm.min().item():.9e}, {v_dir_norm.max().item():.6e}]")
    
    # Compute tangent vectors: theta * (v_dir / ||v_dir||) * radius
    v[mask] = (theta[mask] * sphere_radius).unsqueeze(1) * (v_dir[mask] / (v_dir_norm[mask].unsqueeze(1) + theta_identical_tol))
    
    # For antipodal points, choose arbitrary tangent direction with norm=pi*radius
    if torch.any(mask_antipodal):
        antipodal_indices = torch.where(mask_antipodal)[0]
        for i in antipodal_indices:
            if abs(x0_unit[i, 0].item()) < 0.9:
                v_dir_ant = torch.zeros(D, device=DEVICE)
                v_dir_ant[0] = 1.0
            else:
                v_dir_ant = torch.zeros(D, device=DEVICE)
                v_dir_ant[1] = 1.0
            v_dir_ant = v_dir_ant - torch.dot(v_dir_ant, x0_unit[i]) * x0_unit[i]
            v_dir_ant = v_dir_ant / (torch.norm(v_dir_ant) + eps)
            v[i] = np.pi * sphere_radius * v_dir_ant
    # For identical, keep zero vector
    return to_numpy(v)


def local_distance_sphere(x0, x, batch_size=256, radius=None):
    """
    Compute local spherical distance(s) between x0 and x.

    If x is a batched array, returns a 1D numpy array of distances; otherwise
    returns a scalar float for a single pair of vectors. Uses `log_map_sphere`
    under the hood and supports chunked processing.
    
    Args:
        x0: Base point(s) on sphere
        x: Target point(s) on sphere
        batch_size: Batch size for processing
        radius: Sphere radius (default: uses global DEFAULT_SPHERE_RADIUS)
    
    Returns:
        Geodesic distance(s) on the sphere
    """
    x0_np = np.asarray(x0, dtype=float)
    x_np = np.asarray(x, dtype=float)
    
    # Single vector case: compute distance directly using log_map
    if x_np.ndim == 1:
        v_log = log_map_sphere(x0_np, x_np, radius=radius)
        v_log_torch = to_torch(v_log)
        return torch.norm(v_log_torch).item()
    
    # Batch case: use log_map_sphere to get tangent vectors, then compute norms
    n, D = x_np.shape
    v_logs = log_map_sphere(x0_np, x_np, radius=radius)  # Returns (n, D) array of tangent vectors
    v_logs_torch = to_torch(v_logs)
    distances = torch.norm(v_logs_torch, dim=1)  # Compute norm of each tangent vector
    return to_numpy(distances)


def parallel_transport_sphere(x_from, x_to, v, eps=1e-7, batch_size=256, radius=None):
    """
    Parallel transport tangent vector(s) v in T_{x_from}S^{n-1} to T_{x_to}S^{n-1}.

    Supports single-vector and batched inputs. When batched, all inputs may be
    provided as (N, D) arrays, or x_from/x_to may be provided as (D,) and will
    be broadcast to the batch.
    
    Args:
        x_from: Starting point(s) on sphere
        x_to: Destination point(s) on sphere  
        v: Tangent vector(s) at x_from to transport
        eps: Numerical stability threshold
        batch_size: Batch size for processing
        radius: Sphere radius (default: uses global DEFAULT_SPHERE_RADIUS)
                Used for scaling if needed.
    
    Returns:
        Transported tangent vector(s) at x_to
    """
    # Note: parallel transport on sphere is radius-independent in direction,
    # but we keep the parameter for API consistency
    _ = _resolve_radius(radius)  # Validate radius if provided
    x_from = np.asarray(x_from, dtype=float)
    x_to = np.asarray(x_to, dtype=float)
    v = np.asarray(v, dtype=float)
    v_norm_str = f"{np.linalg.norm(v):.6f}" if v.ndim == 1 else f"{np.linalg.norm(v, axis=1).mean():.6f}"
    logging.debug(f"parallel_transport_sphere: x_from shape={x_from.shape}, x_to shape={x_to.shape}, v shape={v.shape}, v norm={v_norm_str}")
    if x_from.ndim == 1 and x_to.ndim == 1 and v.ndim == 1:
        # Convert to torch for GPU computation
        x_from_torch = to_torch(x_from)
        x_to_torch = to_torch(x_to)
        v_torch = to_torch(v)
        
        dot = torch.clamp(torch.dot(x_from_torch, x_to_torch), -1.0, 1.0)
        # Use theta-based tolerance consistent with log_map_sphere
        theta_identical_tol = max(1e-12, eps)
        antipodal_tol = max(1e-9, eps * 10)
        theta = torch.acos(dot)
        if theta < theta_identical_tol:
            # Identical points: return original vector unchanged
            return to_numpy(v_torch.clone())
        if torch.abs(dot + 1.0) < antipodal_tol:
            # Antipodal: project to tangent space
            v_tan = v_torch - torch.dot(v_torch, x_from_torch) * x_from_torch
            return to_numpy(v_tan)
        sin_theta = torch.sin(theta)
        k_from = (x_to_torch - dot * x_from_torch) / (sin_theta + eps)
        k_to = (x_from_torch - dot * x_to_torch) / (sin_theta + eps)
        a = torch.dot(v_torch, k_from)
        v_perp = v_torch - a * k_from
        transported = a * k_to + v_perp
        # Preserve norm exactly
        original_norm = torch.norm(v_torch)
        transported_norm = torch.norm(transported)
        if transported_norm > eps:
            transported = transported * (original_norm / transported_norm)
        return to_numpy(transported)
    # Convert to torch for GPU computation
    x_from_torch = to_torch(x_from)
    x_to_torch = to_torch(x_to)
    v_torch = to_torch(v)
    
    n, D = v_torch.shape if v_torch.ndim == 2 else (1, v_torch.shape[0])
    if x_from_torch.ndim == 1:
        x_from_mat = x_from_torch.unsqueeze(0).expand(n, D).clone()
    else:
        x_from_mat = x_from_torch.float()
    if x_to_torch.ndim == 1:
        x_to_mat = x_to_torch.unsqueeze(0).expand(n, D).clone()
    else:
        x_to_mat = x_to_torch.float()
    dot = torch.sum(x_from_mat * x_to_mat, dim=1)
    # Use theta-based tolerance consistent with log_map_sphere
    theta_identical_tol = max(1e-12, eps)
    antipodal_tol = max(1e-9, eps * 10)
    dot_clipped = torch.clamp(dot, -1.0, 1.0)
    theta = torch.acos(dot_clipped)
    mask_id = theta < theta_identical_tol
    mask_ant = torch.abs(dot + 1.0) < antipodal_tol
    transported = torch.zeros_like(v_torch)
    mask = ~(mask_id | mask_ant)
    # For identical points, return original vector
    transported[mask_id] = v_torch[mask_id].clone()
    # For antipodal, project to tangent space
    if torch.any(mask_ant):
        v_tan = v_torch[mask_ant] - torch.sum(v_torch[mask_ant] * x_from_mat[mask_ant], dim=1).unsqueeze(1) * x_from_mat[mask_ant]
        transported[mask_ant] = v_tan
    # For others, use rotation
    if torch.any(mask):
        theta_masked = torch.acos(dot[mask])
        sin_theta = torch.sin(theta_masked)
        xf = x_from_mat[mask]
        xt = x_to_mat[mask]
        vv = v_torch[mask]
        k_from = (xt - dot[mask].unsqueeze(1) * xf) / (sin_theta.unsqueeze(1) + eps)
        k_to = (xf - dot[mask].unsqueeze(1) * xt) / (sin_theta.unsqueeze(1) + eps)
        a = torch.sum(vv * k_from, dim=1)
        v_perp = vv - a.unsqueeze(1) * k_from
        transported[mask] = a.unsqueeze(1) * k_to + v_perp
        # Preserve norms exactly for normal cases
        original_norms = torch.norm(vv, dim=1)
        transported_norms = torch.norm(transported[mask], dim=1)
        valid = transported_norms > eps
        if torch.any(valid):
            scale_factors = original_norms[valid] / transported_norms[valid]
            mask_indices = torch.where(mask)[0]
            valid_indices = mask_indices[valid]
            transported[valid_indices] *= scale_factors.unsqueeze(1)
    return to_numpy(transported)


def exp_map_sphere(x0, v, eps=1e-9, radius=None):
    """
    Exponential map on the hypersphere: maps tangent vector v at x0 to a point on the sphere.
    
    This is the inverse of log_map_sphere. Given a base point x0 and a tangent
    vector v, returns the point reached by traveling along the geodesic in
    direction v for distance ||v||.
    
    Args:
        x0: Base point on the sphere
        v: Tangent vector at x0 (direction and distance to travel)
        eps: Numerical stability threshold
        radius: Sphere radius (default: uses global DEFAULT_SPHERE_RADIUS)
                The returned point will lie on the sphere of this radius.
    
    Returns:
        Point on sphere reached by following geodesic from x0 in direction v
    """
    sphere_radius = _resolve_radius(radius)
    
    # Convert to torch for GPU computation
    x0_torch = to_torch(x0)
    v_torch = to_torch(v)
    
    # Normalize x0 to unit sphere for computation
    x0_norm = torch.norm(x0_torch)
    if x0_norm < eps:
        # Degenerate case: return a point on sphere
        result = torch.zeros_like(x0_torch)
        result[0] = sphere_radius
        return to_numpy(result)
    
    x0_unit = x0_torch / x0_norm
    
    # The tangent vector norm gives the geodesic distance to travel
    # If v was computed with log_map using radius R, it was scaled by R
    norm_v = torch.norm(v_torch)
    if norm_v < eps:
        # No movement - return x0 scaled to target radius
        return to_numpy(x0_unit * sphere_radius)
    
    # Project v to tangent space at x0 (ensure orthogonality)
    # This handles cases where v might have a small component along x0
    v_tangent = v_torch - torch.dot(v_torch, x0_unit) * x0_unit
    v_tangent_norm = torch.norm(v_tangent)
    
    if v_tangent_norm < eps:
        # v is parallel to x0, no movement in tangent direction
        return to_numpy(x0_unit * sphere_radius)
    
    # Angle to travel on unit sphere (arc length / radius = angle)
    # The tangent vector was scaled by sphere_radius in log_map
    theta = norm_v / sphere_radius
    
    # Unit direction in tangent space
    v_tangent_unit = v_tangent / v_tangent_norm
    
    # Move along geodesic on unit sphere using Rodrigues' formula
    # exp_x0(v) = cos(theta) * x0 + sin(theta) * (v / ||v||)
    # where v is the tangent vector direction (orthogonal to x0)
    result_unit = torch.cos(theta) * x0_unit + torch.sin(theta) * v_tangent_unit
    
    # Ensure result is exactly on sphere (numerical safety)
    result_norm = torch.norm(result_unit)
    if result_norm > eps:
        result_unit = result_unit / result_norm
    
    return to_numpy(result_unit * sphere_radius)


def geodesic_interpolation(x0, x1, t, radius=None):
    """
    Interpolate along the geodesic between x0 and x1.
    
    This is a convenience function that combines log_map and exp_map
    to perform spherical linear interpolation (slerp).
    
    Args:
        x0: Starting point on sphere
        x1: Ending point on sphere
        t: Interpolation parameter in [0, 1]
           t=0 returns x0, t=1 returns x1
        radius: Sphere radius (default: uses global DEFAULT_SPHERE_RADIUS)
    
    Returns:
        Point on sphere at fraction t along geodesic from x0 to x1
    """
    sphere_radius = _resolve_radius(radius)
    
    # Get tangent vector from x0 to x1
    v = log_map_sphere(x0, x1, radius=sphere_radius)
    
    # Scale tangent vector by t and apply exp_map
    return exp_map_sphere(x0, t * v, radius=sphere_radius)


def curvature_at_point(x, neighbors, radius=None):
    """
    Estimate local curvature at a point based on its neighbors.
    
    Higher curvature indicates the point is in a region where the
    manifold bends more sharply. This can be used for adaptive
    learning rates or attention weighting.
    
    Args:
        x: Point on sphere
        neighbors: Array of neighboring points (N, D)
        radius: Sphere radius (default: uses global DEFAULT_SPHERE_RADIUS)
    
    Returns:
        Scalar curvature estimate
    """
    sphere_radius = _resolve_radius(radius)
    
    x_np = np.asarray(x, dtype=float)
    neighbors_np = np.asarray(neighbors, dtype=float)
    
    if neighbors_np.ndim == 1:
        neighbors_np = neighbors_np.reshape(1, -1)
    
    # Compute tangent vectors to all neighbors
    tangent_vectors = log_map_sphere(x_np, neighbors_np, radius=sphere_radius)
    
    # Convert to torch for GPU computation
    tangent_vectors_torch = to_torch(tangent_vectors)
    
    # Curvature can be estimated from the spread of tangent directions
    # On a sphere, intrinsic curvature = 1/R^2
    # Local variation adds to this baseline
    
    baseline_curvature = 1.0 / (sphere_radius ** 2)
    
    if len(tangent_vectors_torch) < 2:
        return baseline_curvature
    
    # Compute variance in tangent directions
    norms = torch.norm(tangent_vectors_torch, dim=1)
    valid = norms > 1e-10
    if torch.sum(valid).item() < 2:
        return baseline_curvature
    
    unit_tangents = tangent_vectors_torch[valid] / norms[valid].unsqueeze(1)
    
    # Angular variance - how spread out are the directions?
    mean_direction = torch.mean(unit_tangents, dim=0)
    mean_direction_norm = torch.norm(mean_direction)
    
    if mean_direction_norm < 1e-10:
        # Directions are very spread out - high local complexity
        angular_variance = 1.0
    else:
        # 1 - mean_direction_norm gives angular spread (0 = all same direction, 1 = uniform)
        angular_variance = 1.0 - mean_direction_norm.item()
    
    # Combine baseline curvature with local angular variance
    return baseline_curvature + angular_variance / (sphere_radius ** 2)
