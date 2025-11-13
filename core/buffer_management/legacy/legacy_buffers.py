"""
Legacy Buffer Classes - Deprecated

DEPRECATED: These classes are kept for backward compatibility only.
Use the modern buffer management classes instead:
- BufferCoordinator for centralized buffer management
- AtomicRingBuffer for lock-free ring buffers
- CommandBuffer for inter-process commands
"""

import logging
import time
from multiprocessing import Array, Value, shared_memory
import numpy as np


logger = logging.getLogger('BufferSystem')


class SharedScoreBuffer:
    """
    DEPRECATED: Lock-free shared memory buffer for score data.

    Use AtomicRingBuffer or BufferCoordinator instead.
    """
    def __init__(self, size=104):  # 52*2 for multiface, or 52 for single
        self.data = Array('f', size)  # float array
        self.timestamp = Value('d', 0.0)  # double timestamp
        self.new_data = Value('i', 0)  # integer flag
        self.size = size

    def write(self, scores):
        """Write scores without locking"""
        # Update timestamp first
        self.timestamp.value = time.time()
        # Copy data
        for i in range(min(len(scores), self.size)):
            self.data[i] = scores[i]
        # Set flag last to indicate new data
        self.new_data.value = 1

    def read_if_new(self):
        """Read only if new data available"""
        if self.new_data.value:
            # Clear flag
            self.new_data.value = 0
            # Return copy of data
            return list(self.data), self.timestamp.value
        return None, None

    def read_latest(self):
        """Always read latest data"""
        self.new_data.value = 0
        return list(self.data), self.timestamp.value


class NumpySharedBuffer:
    """
    DEPRECATED: Zero-copy shared memory using numpy arrays.

    Use AtomicRingBuffer or BufferCoordinator instead.
    """
    def __init__(self, size=104, name=None, shape=None):
        """
        Initialize shared buffer.

        Args:
            size: Total size of buffer (for backward compatibility)
            name: Name of shared memory (None to create new)
            shape: Optional shape tuple for multi-dimensional arrays
        """
        # Determine shape and size
        if shape is not None:
            self.shape = shape
            self.size = np.prod(shape)
        else:
            self.size = size
            self.shape = (size,)

        if name is None:
            # Create new shared memory
            self.shm = shared_memory.SharedMemory(create=True, size=self.size * 4 + 8)  # float32 + timestamp
            self.name = self.shm.name
            self._created = True
        else:
            # Connect to existing shared memory
            self.shm = shared_memory.SharedMemory(name=name)
            self.name = name
            self._created = False

        # Create views into the shared memory
        self.arr = np.ndarray(self.shape, dtype=np.float32, buffer=self.shm.buf[8:])
        self.timestamp = np.ndarray((1,), dtype=np.float64, buffer=self.shm.buf[:8])

    def write(self, scores):
        """Write scores with zero copy"""
        self.timestamp[0] = time.time()
        scores_len = min(len(scores), self.size)
        if len(self.shape) == 1:
            # 1D array
            self.arr[:scores_len] = scores[:scores_len]
            if scores_len < self.size:
                self.arr[scores_len:] = 0.0  # Clear remaining
        else:
            # Multi-dimensional - write to first row
            self.arr.flat[:scores_len] = scores[:scores_len]

    def update_column(self, col_idx, values):
        """Update a specific column (for 2D arrays)"""
        if len(self.shape) == 2 and 0 <= col_idx < self.shape[1]:
            self.timestamp[0] = time.time()
            values_len = min(len(values), self.shape[0])
            self.arr[:values_len, col_idx] = values[:values_len]

    def read_latest(self):
        """Read with zero copy"""
        return self.arr.copy(), float(self.timestamp[0])

    def cleanup(self):
        """Cleanup shared memory"""
        try:
            # Delete numpy array views first to release references
            if hasattr(self, 'arr'):
                del self.arr
            if hasattr(self, 'timestamp'):
                del self.timestamp

            # Now safe to close shared memory
            self.shm.close()
            if self._created:
                self.shm.unlink()
        except Exception as e:
            logger.error(f"Failed to cleanup NumpySharedBuffer: {e}")
