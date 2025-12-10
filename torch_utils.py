"""
PyTorch GPU/CPU utilities for device management and tensor conversion.

Provides automatic device detection, tensor conversion between NumPy and PyTorch,
and utilities for multi-GPU support.
"""

import torch
import numpy as np
import warnings

# Suppress PyTorch warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Auto-detect best device (CUDA > CPU)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"✓ GPU Acceleration Enabled")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    if torch.cuda.device_count() > 1:
        print(f"  Multi-GPU: {torch.cuda.device_count()} devices detected")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    DEVICE = torch.device('cpu')
    print("⚠ No GPU detected, using CPU (slow)")

# Set default tensor type for efficiency
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def to_torch(x, device=None, dtype=torch.float32):
    """
    Convert numpy array or list to PyTorch tensor on specified device.
    
    Args:
        x: Input data (numpy array, list, or torch tensor)
        device: Target device (default: DEVICE global)
        dtype: Target dtype (default: float32)
    
    Returns:
        torch.Tensor on target device
    """
    if device is None:
        device = DEVICE
    
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    else:
        return torch.tensor(x, device=device, dtype=dtype)


def to_numpy(x):
    """
    Convert PyTorch tensor to numpy array.
    
    Args:
        x: Input tensor or numpy array
    
    Returns:
        numpy.ndarray on CPU
    """
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def get_device():
    """Get current default device."""
    return DEVICE


def is_gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()


def get_gpu_count():
    """Get number of available GPUs."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def get_gpu_memory_stats():
    """
    Get GPU memory usage statistics.
    
    Returns:
        dict: Memory statistics for each GPU
    """
    if not torch.cuda.is_available():
        return {}
    
    stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        
        stats[f'gpu_{i}'] = {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - allocated
        }
    
    return stats


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_tensor_info(x, name="tensor"):
    """
    Print debug information about a tensor.
    
    Args:
        x: Tensor to inspect
        name: Name for display
    """
    if isinstance(x, torch.Tensor):
        print(f"{name}: shape={x.shape}, dtype={x.dtype}, device={x.device}, "
              f"min={x.min().item():.4f}, max={x.max().item():.4f}, "
              f"mean={x.mean().item():.4f}")
    else:
        print(f"{name}: type={type(x)}, shape={np.array(x).shape}")


def batch_to_device(batch, device=None):
    """
    Move a batch (dict, list, or tensor) to specified device.
    
    Args:
        batch: Data batch
        device: Target device
    
    Returns:
        Batch on target device
    """
    if device is None:
        device = DEVICE
    
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [batch_to_device(x, device) for x in batch]
    elif isinstance(batch, tuple):
        return tuple(batch_to_device(x, device) for x in batch)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch


# Initialize on import
set_seed(42)  # For reproducibility
print(f"✓ torch_utils initialized (device: {DEVICE})")
