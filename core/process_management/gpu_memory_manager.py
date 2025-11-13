"""
GPU Shared Memory Manager for Inter-Process Communication

This module provides GPU memory sharing between processes using CUDA IPC.
It allows zero-copy frame sharing between detection and ROI extraction processes.
"""

import multiprocessing as mp
from typing import Dict, Optional, Tuple, Any
import numpy as np
import logging
import weakref
import atexit
import time
from .confighandler import ConfigHandler

try:
    import cupy as cp
    import cupy.cuda as cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    cuda = None

logger = logging.getLogger(__name__)


class GPUMemoryHandle:
    """Wrapper for GPU memory handle that can be passed between processes."""
    
    def __init__(self, handle_bytes: bytes, shape: Tuple[int, ...], dtype: np.dtype, device_id: int = 0):
        self.handle_bytes = handle_bytes
        self.shape = shape
        self.dtype = dtype
        self.device_id = device_id
        self.size = np.prod(shape) * np.dtype(dtype).itemsize
        
    def __reduce__(self):
        # Make the handle pickleable for multiprocessing
        return (GPUMemoryHandle, (self.handle_bytes, self.shape, self.dtype, self.device_id))


class GPUMemoryPool:
    """Manages a pool of GPU memory allocations that can be shared between processes."""
    
    def __init__(self, pool_size: int = None, device_id: int = None, config=None):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA/CuPy not available for GPU memory sharing")
        
        # Load configuration
        self.config = config if config else ConfigHandler()
        
        # Get configuration values with defaults
        self.device_id = (
            device_id if device_id is not None 
            else self.config.get('process_management.gpu_memory.device_id', 0)
        )
        self.pool_size = (
            pool_size if pool_size is not None 
            else self.config.get('process_management.gpu_memory.pool_size', 100)
        )
        self.pool_limit_gb = self.config.get('process_management.gpu_memory.pool_limit_gb', 2.0)
        self.allocations = {}  # frame_id -> (gpu_array, handle)
        self.free_list = []
        self.lock = mp.Lock()
        
        # Set device
        with cuda.Device(self.device_id):
            # Pre-allocate memory pool
            self.memory_pool = cp.get_default_memory_pool()
            pool_limit_bytes = int(self.pool_limit_gb * 1024**3)  # Convert GB to bytes
            self.memory_pool.set_limit(size=pool_limit_bytes)
            
        # Register cleanup
        atexit.register(self.cleanup)
        
    def allocate(self, frame_id: str, shape: Tuple[int, ...], dtype: np.dtype) -> Optional[GPUMemoryHandle]:
        """Allocate GPU memory and return a shareable handle."""
        try:
            with self.lock:
                # Check if already allocated
                if frame_id in self.allocations:
                    gpu_array, handle = self.allocations[frame_id]
                    return handle
                    
                with cuda.Device(self.device_id):
                    # Allocate GPU memory
                    gpu_array = cp.empty(shape, dtype=dtype)
                    
                    # Get IPC handle
                    mem_ptr = gpu_array.data.ptr
                    ipc_handle = cp.cuda.runtime.ipcGetMemHandle(mem_ptr)
                    
                    # Create handle wrapper
                    handle = GPUMemoryHandle(
                        handle_bytes=bytes(ipc_handle),
                        shape=shape,
                        dtype=dtype,
                        device_id=self.device_id
                    )
                    
                    # Store allocation
                    self.allocations[frame_id] = (gpu_array, handle)
                    
                    return handle
                    
        except Exception as e:
            logger.error(f"Failed to allocate GPU memory: {e}")
            return None
            
    def write(self, frame_id: str, data: np.ndarray) -> Optional[GPUMemoryHandle]:
        """Write data to GPU memory and return handle."""
        try:
            # Allocate if needed
            handle = self.allocate(frame_id, data.shape, data.dtype)
            if not handle:
                return None
                
            with self.lock:
                gpu_array, _ = self.allocations[frame_id]
                
                with cuda.Device(self.device_id):
                    # Copy data to GPU
                    if isinstance(data, cp.ndarray):
                        cp.copyto(gpu_array, data)
                    else:
                        gpu_array[:] = cp.asarray(data)
                        
            return handle
            
        except Exception as e:
            logger.error(f"Failed to write GPU memory: {e}")
            return None
            
    def get_array(self, handle: GPUMemoryHandle) -> Optional[cp.ndarray]:
        """Get GPU array from handle (in same process)."""
        try:
            with cuda.Device(self.device_id):
                # Find existing allocation
                for gpu_array, stored_handle in self.allocations.values():
                    if stored_handle.handle_bytes == handle.handle_bytes:
                        return gpu_array
                        
            return None
        except Exception as e:
            logger.error(f"Failed to get GPU array: {e}")
            return None
            
    def free(self, frame_id: str):
        """Free GPU memory allocation."""
        with self.lock:
            if frame_id in self.allocations:
                del self.allocations[frame_id]
                
    def cleanup(self):
        """Clean up all allocations."""
        with self.lock:
            self.allocations.clear()
            if CUDA_AVAILABLE:
                self.memory_pool.free_all_blocks()


class GPUMemoryClient:
    """Client for accessing GPU memory in child processes."""
    
    def __init__(self, device_id: int = None, config=None):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA/CuPy not available for GPU memory sharing")
        
        # Load configuration
        self.config = config if config else ConfigHandler()
        
        # Get configuration values with defaults  
        self.device_id = (
            device_id if device_id is not None 
            else self.config.get('process_management.gpu_memory.device_id', 0)
        )
        self.handle_cache = {}  # handle_bytes -> gpu_array
        self.lock = mp.Lock()
        
        # Create and store context
        self.cuda_context = None
        try:
            with cuda.Device(device_id) as device:
                self.cuda_context = device.make_context()
                self.cuda_context.push()
                # Configure memory pool
                self.memory_pool = cp.get_default_memory_pool()
                self.cuda_context.pop()
                print(f"[GPUMemoryClient] Initialized with device {device_id}")
        except Exception as e:
            logger.error(f"Failed to initialize CUDA context: {e}")
            raise
    
    def get_array(self, handle: GPUMemoryHandle) -> Optional[cp.ndarray]:
        """Get GPU array from IPC handle."""
        if self.cuda_context is None:
            logger.error("CUDA context not initialized")
            return None
            
        try:
            # Push context for this thread
            self.cuda_context.push()
            
            with self.lock:
                # Check cache
                handle_key = handle.handle_bytes
                if handle_key in self.handle_cache:
                    return self.handle_cache[handle_key]
                    
                with cuda.Device(self.device_id):
                    # Open IPC memory handle
                    ipc_mem_handle = cp.cuda.runtime.ipcOpenMemHandle(
                        handle.handle_bytes,
                        cp.cuda.runtime.cudaIpcMemLazyEnablePeerAccess
                    )
                    
                    # Create CuPy array from IPC memory
                    gpu_array = cp.ndarray(
                        shape=handle.shape,
                        dtype=handle.dtype,
                        memptr=cp.cuda.MemoryPointer(
                            cp.cuda.UnownedMemory(ipc_mem_handle, handle.size, None),
                            0
                        )
                    )
                    
                    # Cache for reuse
                    self.handle_cache[handle_key] = gpu_array
                    
                    return gpu_array
                    
        except Exception as e:
            logger.error(f"Failed to open GPU IPC handle: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Always pop context
            if self.cuda_context:
                try:
                    self.cuda_context.pop()
                except:
                    pass
            
    def close_handle(self, handle: GPUMemoryHandle):
        """Close IPC handle and remove from cache."""
        with self.lock:
            handle_key = handle.handle_bytes
            if handle_key in self.handle_cache:
                del self.handle_cache[handle_key]
                # CuPy will handle cleanup when array is garbage collected
                
    def cleanup(self):
        """Clean up all cached handles."""
        with self.lock:
            self.handle_cache.clear()


class GPUFrameQueue:
    """Queue that passes GPU memory handles instead of data."""
    
    def __init__(self, maxsize: int = 0):
        self.queue = mp.Queue(maxsize=maxsize)
        
    def put(self, item: Tuple[str, GPUMemoryHandle, Any], block: bool = True, timeout: Optional[float] = None):
        """Put GPU memory handle and metadata into queue."""
        self.queue.put(item, block=block, timeout=timeout)
        
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Tuple[str, GPUMemoryHandle, Any]:
        """Get GPU memory handle and metadata from queue."""
        return self.queue.get(block=block, timeout=timeout)
        
    def empty(self) -> bool:
        return self.queue.empty()
        
    def full(self) -> bool:
        return self.queue.full()
        
    def qsize(self) -> int:
        try:
            return self.queue.qsize()
        except NotImplementedError:
            return -1


def create_gpu_memory_manager(pool_size: int = None, device_id: int = None, config=None) -> Optional[GPUMemoryPool]:
    """Create GPU memory manager if CUDA is available."""
    if not CUDA_AVAILABLE:
        logger.warning("CUDA not available, GPU memory sharing disabled")
        return None
        
    try:
        manager = GPUMemoryPool(pool_size=pool_size, device_id=device_id, config=config)
        actual_pool_size = manager.pool_size  # Get the actual configured value
        logger.info(f"GPU memory manager created with pool size {actual_pool_size}")
        return manager
    except Exception as e:
        logger.error(f"Failed to create GPU memory manager: {e}")
        return None


def create_gpu_memory_client(device_id: int = None, config=None) -> Optional[GPUMemoryClient]:
    """Create GPU memory client for child processes."""
    if not CUDA_AVAILABLE:
        return None
        
    try:
        client = GPUMemoryClient(device_id=device_id, config=config)
        logger.info("GPU memory client created")
        return client
    except Exception as e:
        logger.error(f"Failed to create GPU memory client: {e}")
        return None