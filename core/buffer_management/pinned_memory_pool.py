"""
Pinned Memory Pool for faster CPU-GPU transfers.

This module provides a pool of pinned (page-locked) memory buffers that can be
reused for efficient CPU-GPU data transfers.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import threading
import queue

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    cuda = None


def get_nested_config(config: Dict, path: str, default=None):
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., 'buffer_management.pinned_memory.detection_frame_count')
        default: Default value if path not found
        
    Returns:
        Configuration value or default
    """
    if not config:
        return default
        
    keys = path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


class PinnedMemoryBuffer:
    """A reusable pinned memory buffer."""
    
    def __init__(self, size: int, dtype=np.float32):
        """
        Create a pinned memory buffer.
        
        Args:
            size: Size in elements (not bytes)
            dtype: NumPy data type
        """
        self.size = size
        self.dtype = dtype
        self.in_use = False
        
        if HAS_CUDA:
            self.host_buffer = cuda.pagelocked_empty(size, dtype)
            self.device_buffer = cuda.mem_alloc(self.host_buffer.nbytes)
        else:
            self.host_buffer = np.empty(size, dtype)
            self.device_buffer = None
    
    def acquire(self):
        """Mark buffer as in use."""
        self.in_use = True
        
    def release(self):
        """Mark buffer as available."""
        self.in_use = False
        
    def __del__(self):
        """Clean up device memory."""
        if HAS_CUDA and self.device_buffer is not None:
            try:
                self.device_buffer.free()
            except:
                pass


class PinnedMemoryPool:
    """
    Pool of pinned memory buffers for efficient CPU-GPU transfers.
    
    Features:
    - Pre-allocated pinned memory buffers
    - Thread-safe buffer allocation
    - Automatic buffer recycling
    - Multiple buffer sizes supported
    """
    
    def __init__(self, buffer_configs: List[Dict[str, int]] = None, config: Dict = None):
        """
        Initialize pinned memory pool.
        
        Args:
            buffer_configs: List of buffer configurations
                           [{'size': elements, 'count': num_buffers}, ...]
            config: Configuration dictionary for getting buffer counts
        """
        if buffer_configs is None:
            # Get buffer counts from config if available
            detection_count = 4
            roi_count = 16
            hd_count = 2
            
            if config:
                # Use centralized buffer_settings configuration
                detection_count = get_nested_config(config, 'buffer_settings.pinned_memory.detection_frames', 4)
                roi_count = get_nested_config(config, 'buffer_settings.pinned_memory.roi_buffers', 16)
                hd_count = get_nested_config(config, 'buffer_settings.pinned_memory.hd_frames', 2)
            
            # Default configuration for common use cases
            buffer_configs = [
                {'size': 640 * 640 * 3, 'count': detection_count},    # Detection frames
                {'size': 256 * 256 * 3, 'count': roi_count},          # ROIs
                {'size': 1920 * 1080 * 3, 'count': hd_count},        # Full HD frames
            ]
        
        self.pools: Dict[int, List[PinnedMemoryBuffer]] = {}
        self.locks: Dict[int, threading.Lock] = {}
        
        # Statistics
        self.stats = {
            'allocations': 0,
            'hits': 0,
            'misses': 0,
            'active_buffers': 0
        }
        
        self.enabled = HAS_CUDA
        
        if not self.enabled:
            print("[PinnedMemoryPool] CUDA not available, pool disabled")
            return
            
        # Create buffer pools
        for config in buffer_configs:
            size = config['size']
            count = config['count']
            
            if size not in self.pools:
                self.pools[size] = []
                self.locks[size] = threading.Lock()
            
            # Create buffers
            for _ in range(count):
                buffer = PinnedMemoryBuffer(size)
                self.pools[size].append(buffer)
                
        print(f"[PinnedMemoryPool] Initialized with {len(self.pools)} pool sizes")
        for size, buffers in self.pools.items():
            print(f"  - Size {size}: {len(buffers)} buffers")
    
    def allocate(self, size: int, dtype=np.float32) -> Optional[PinnedMemoryBuffer]:
        """
        Allocate a pinned memory buffer.
        
        Args:
            size: Required size in elements
            dtype: Data type (must match pool configuration)
            
        Returns:
            PinnedMemoryBuffer if available, None otherwise
        """
        if not self.enabled:
            return None
            
        self.stats['allocations'] += 1
        
        # Check if we have a pool for this size
        if size in self.pools:
            with self.locks[size]:
                # Find available buffer
                for buffer in self.pools[size]:
                    if not buffer.in_use:
                        buffer.acquire()
                        self.stats['hits'] += 1
                        self.stats['active_buffers'] += 1
                        return buffer
        
        # No available buffer
        self.stats['misses'] += 1
        
        # Could create a new buffer on-demand here
        # For now, return None to indicate pool exhaustion
        return None
    
    def release(self, buffer: PinnedMemoryBuffer):
        """
        Release a buffer back to the pool.
        
        Args:
            buffer: Buffer to release
        """
        if not self.enabled or buffer is None:
            return
            
        buffer.release()
        self.stats['active_buffers'] -= 1
    
    def get_buffer_or_create(self, shape: Tuple[int, ...], dtype=np.float32) -> Tuple[np.ndarray, Optional[PinnedMemoryBuffer]]:
        """
        Get a pinned buffer or create a regular numpy array.
        
        Args:
            shape: Shape of the required array
            dtype: Data type
            
        Returns:
            Tuple of (array, buffer) where buffer is None if not from pool
        """
        size = int(np.prod(shape))
        buffer = self.allocate(size, dtype)
        
        if buffer is not None:
            # Reshape the buffer to match requested shape
            array = buffer.host_buffer[:size].reshape(shape)
            return array, buffer
        else:
            # Fall back to regular numpy array
            array = np.empty(shape, dtype)
            return array, None
    
    def transfer_to_device_async(self, host_array: np.ndarray, 
                                device_ptr: cuda.DeviceAllocation,
                                stream: cuda.Stream = None):
        """
        Asynchronously transfer data to device using pinned memory.
        
        Args:
            host_array: Source array (should be pinned)
            device_ptr: Destination device pointer
            stream: CUDA stream for async transfer
        """
        if not self.enabled:
            return
            
        if stream:
            cuda.memcpy_htod_async(device_ptr, host_array, stream)
        else:
            cuda.memcpy_htod(device_ptr, host_array)
    
    def transfer_to_host_async(self, device_ptr: cuda.DeviceAllocation,
                              host_array: np.ndarray,
                              stream: cuda.Stream = None):
        """
        Asynchronously transfer data from device using pinned memory.
        
        Args:
            device_ptr: Source device pointer
            host_array: Destination array (should be pinned)
            stream: CUDA stream for async transfer
        """
        if not self.enabled:
            return
            
        if stream:
            cuda.memcpy_dtoh_async(host_array, device_ptr, stream)
        else:
            cuda.memcpy_dtoh(host_array, device_ptr)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        stats = self.stats.copy()
        stats['hit_rate'] = stats['hits'] / stats['allocations'] if stats['allocations'] > 0 else 0
        
        # Add per-size statistics
        stats['pools'] = {}
        for size, buffers in self.pools.items():
            in_use = sum(1 for b in buffers if b.in_use)
            stats['pools'][size] = {
                'total': len(buffers),
                'in_use': in_use,
                'available': len(buffers) - in_use
            }
        
        return stats
    
    def cleanup(self):
        """Clean up all buffers."""
        for buffers in self.pools.values():
            for buffer in buffers:
                del buffer
        self.pools.clear()
        self.locks.clear()


# Global pool instance
_global_pool = None


def get_pinned_memory_pool(config: Dict = None) -> PinnedMemoryPool:
    """Get or create the global pinned memory pool."""
    global _global_pool
    if _global_pool is None:
        _global_pool = PinnedMemoryPool(config=config)
    return _global_pool