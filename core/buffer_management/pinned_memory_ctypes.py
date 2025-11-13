"""
Lightweight pinned memory allocator using ctypes + libcudart.
NO external dependencies beyond built-in ctypes.

This provides CUDA pinned (page-locked) memory for faster PCIe transfers
without requiring cupy or pycuda installation.

NOTE: CUDA is NOT available in WSL2 (uses D3D12 translation layer instead).
This module automatically detects WSL2 and skips CUDA initialization to
prevent X11 crashes.
"""

import ctypes
import numpy as np
import logging
import os

logger = logging.getLogger('PinnedMemory')


def _is_wsl2():
    """
    Lightweight WSL2 detection without importing heavy platform_detection module.

    Returns:
        bool: True if running in WSL2, False if native Linux
    """
    # Method 1: Check for WSLInterop file (most reliable)
    if os.path.exists('/proc/sys/fs/binfmt_misc/WSLInterop'):
        return True

    # Method 2: Check /proc/version for Microsoft kernel
    try:
        with open('/proc/version', 'r') as f:
            version_str = f.read().lower()
            if 'microsoft' in version_str or 'wsl' in version_str:
                return True
    except:
        pass

    return False


class PinnedMemoryAllocator:
    """Allocate pinned (page-locked) memory using CUDA runtime via ctypes."""

    def __init__(self):
        self.cudart = None
        self.available = False

        # Check if running in WSL2 - CUDA is not available in WSL2
        # WSL2 uses D3D12 translation layer, attempting CUDA init causes X11 crashes
        if _is_wsl2():
            logger.info("ℹ️  WSL2 detected: Skipping CUDA initialization (using regular memory)")
            logger.info("   MediaPipe uses OpenGL ES (not CUDA), so this is expected behavior")
            self.available = False
            self.cudart = None
            return

        try:
            # Load CUDA runtime library (native Linux only)
            self.cudart = ctypes.CDLL('libcudart.so')

            # Define function signatures
            self.cudart.cudaHostAlloc.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),  # ptr
                ctypes.c_size_t,                   # size
                ctypes.c_uint                      # flags
            ]
            self.cudart.cudaHostAlloc.restype = ctypes.c_int

            self.cudart.cudaFreeHost.argtypes = [ctypes.c_void_p]
            self.cudart.cudaFreeHost.restype = ctypes.c_int

            # Test allocation to verify everything works
            self._test_allocation()
            self.available = True
            logger.info("✅ Pinned memory allocator initialized (ctypes)")

        except OSError as e:
            logger.info(f"ℹ️  CUDA runtime not available: {e}")
            logger.info("   Pinned memory optimization disabled (will use regular memory)")
            self.available = False
            self.cudart = None
        except Exception as e:
            logger.warning(f"⚠️  Pinned memory initialization failed: {e}")
            self.available = False
            self.cudart = None

    def _test_allocation(self):
        """Test that pinned memory allocation works."""
        ptr = ctypes.c_void_p()
        flags = 0x02  # cudaHostAllocMapped (accessible from GPU)
        result = self.cudart.cudaHostAlloc(ctypes.byref(ptr), 1024, flags)

        if result != 0:
            raise RuntimeError(f"cudaHostAlloc test failed: error {result}")

        # Test succeeded, free the test buffer
        self.cudart.cudaFreeHost(ptr)

    def allocate(self, size, dtype=np.uint8):
        """
        Allocate pinned memory buffer.

        Args:
            size: Number of elements
            dtype: NumPy dtype

        Returns:
            PinnedBuffer instance

        Raises:
            RuntimeError: If pinned memory not available
        """
        if not self.available:
            raise RuntimeError("Pinned memory not available")

        return PinnedBuffer(self.cudart, size, dtype)

    def allocate_mapped(self, size, dtype=np.uint8):
        """
        Allocate mapped pinned memory buffer accessible from both CPU and GPU.

        This creates memory that has both a CPU pointer (for numpy operations)
        and a GPU device pointer (for CUDA operations), pointing to the same
        physical memory. Enables GPU to read/write without explicit transfers.

        Args:
            size: Number of elements
            dtype: NumPy dtype

        Returns:
            MappedPinnedBuffer instance with both CPU and GPU pointers

        Raises:
            RuntimeError: If pinned memory not available
        """
        if not self.available:
            raise RuntimeError("Pinned memory not available")

        return MappedPinnedBuffer(self.cudart, size, dtype)


class PinnedBuffer:
    """A pinned memory buffer accessible as NumPy array."""

    def __init__(self, cudart, size, dtype):
        self.cudart = cudart
        self.size = size
        self.dtype = np.dtype(dtype)
        self.nbytes = size * self.dtype.itemsize
        self.ptr = None
        self.array = None

        # Allocate pinned memory
        self.ptr = ctypes.c_void_p()
        flags = 0x02  # cudaHostAllocMapped (accessible from both CPU and GPU)
        result = cudart.cudaHostAlloc(ctypes.byref(self.ptr), self.nbytes, flags)

        if result != 0:
            raise RuntimeError(f"cudaHostAlloc failed: error {result}")

        # Create NumPy array view of pinned memory
        # This allows zero-copy access from Python
        buffer = (ctypes.c_char * self.nbytes).from_address(self.ptr.value)
        self.array = np.frombuffer(buffer, dtype=self.dtype, count=size)

    def __del__(self):
        """Free pinned memory on cleanup."""
        if self.ptr and self.ptr.value and self.cudart:
            try:
                self.cudart.cudaFreeHost(self.ptr)
            except:
                pass  # Ignore errors during cleanup

    def as_array(self):
        """
        Get NumPy view of pinned memory.

        Returns:
            numpy.ndarray: View into pinned memory (zero-copy)
        """
        return self.array


class MappedPinnedBuffer(PinnedBuffer):
    """
    Pinned buffer accessible from BOTH CPU and GPU via mapped memory.

    Uses cudaHostAllocMapped to create memory that has both:
    - CPU pointer (accessible via numpy array)
    - GPU device pointer (accessible via CUDA operations)

    This enables GPU operations to read/write the same physical memory
    that CPU sees, without explicit transfers.
    """

    def __init__(self, cudart, size, dtype):
        self.cudart = cudart
        self.size = size
        self.dtype = np.dtype(dtype)
        self.nbytes = size * self.dtype.itemsize
        self.ptr = None
        self.array = None
        self.gpu_ptr = None  # NEW: GPU device pointer

        # Allocate mapped pinned memory (CPU + GPU accessible)
        self.ptr = ctypes.c_void_p()
        flags = 0x02  # cudaHostAllocMapped

        # Define cudaHostAlloc signature if not already done
        cudart.cudaHostAlloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint
        ]
        cudart.cudaHostAlloc.restype = ctypes.c_int

        result = cudart.cudaHostAlloc(ctypes.byref(self.ptr), self.nbytes, flags)

        if result != 0:
            raise RuntimeError(f"cudaHostAlloc failed: error {result}")

        # Get GPU device pointer for same physical memory
        self.gpu_ptr = ctypes.c_void_p()

        # Define cudaHostGetDevicePointer signature
        cudart.cudaHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint
        ]
        cudart.cudaHostGetDevicePointer.restype = ctypes.c_int

        result = cudart.cudaHostGetDevicePointer(
            ctypes.byref(self.gpu_ptr),
            self.ptr,
            0  # flags (must be 0)
        )

        if result != 0:
            raise RuntimeError(f"cudaHostGetDevicePointer failed: error {result}")

        # Create NumPy array view (CPU side)
        buffer = (ctypes.c_char * self.nbytes).from_address(self.ptr.value)
        self.array = np.frombuffer(buffer, dtype=self.dtype, count=size)

    def get_cpu_pointer(self):
        """
        Get CPU host pointer.

        Returns:
            int: CPU memory address
        """
        return self.ptr.value

    def get_gpu_pointer(self):
        """
        Get GPU device pointer.

        This pointer can be used in CUDA operations to access the
        same physical memory that the CPU sees via numpy array.

        Returns:
            int: GPU device memory address
        """
        return self.gpu_ptr.value


# Global allocator instance (singleton pattern)
_allocator = None


def get_allocator():
    """
    Get or create global pinned memory allocator.

    Returns:
        PinnedMemoryAllocator: Singleton allocator instance
    """
    global _allocator
    if _allocator is None:
        _allocator = PinnedMemoryAllocator()
    return _allocator
