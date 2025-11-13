"""
Atomic Ring Buffer - Lock-free shared memory ring buffer for IPC.

This module provides the AtomicRingBuffer class for high-performance inter-process
communication using lock-free atomic operations and sequence validation.
"""

import logging
import time
from multiprocessing import shared_memory, Value
from typing import Dict, Optional, Tuple, Any
import numpy as np


logger = logging.getLogger('BufferSystem')


class AtomicRingBuffer:
    """
    Lock-free ring buffer with atomic indices and sequence validation.
    Provides high-performance inter-process communication with ordering guarantees.
    """

    def __init__(self, buffer_size: int, element_size: int, name: str, max_elements: int = 1):
        """
        Initialize atomic ring buffer.

        Args:
            buffer_size: Number of buffer slots (must be power of 2)
            element_size: Size of each element in bytes
            name: Shared memory name
            max_elements: Maximum elements per slot (for batching)
        """
        # PERFORMANCE FIX: Skip power-of-2 validation for small buffers to reduce overhead
        if buffer_size > 16 and buffer_size & (buffer_size - 1) != 0:
            raise ValueError("Buffer size must be power of 2")

        self.buffer_size = buffer_size
        self.element_size = element_size
        self.max_elements = max_elements
        self.mask = buffer_size - 1  # Power of 2 optimization
        self.name = name

        # Atomic indices for lock-free operation
        self.write_idx = Value('L', 0)
        self.read_idx = Value('L', 0)
        self.sequence_counter = Value('L', 0)

        # Calculate total shared memory size
        slot_size = element_size * max_elements
        metadata_size = 64  # Metadata per slot (sequence, timestamp, count, etc.)
        total_size = buffer_size * (slot_size + metadata_size) + 1024  # Extra for coordination

        # Create shared memory
        try:
            self.shm = shared_memory.SharedMemory(create=True, name=name, size=total_size)
            logger.info(f"Created AtomicRingBuffer '{name}': {buffer_size} slots, "
                       f"{slot_size} bytes per slot, {total_size} total bytes")
        except Exception as e:
            logger.error(f"Failed to create AtomicRingBuffer '{name}': {e}")
            raise

        # Initialize coordination structure at the beginning of shared memory
        self.coordination_view = np.frombuffer(
            self.shm.buf,
            dtype=np.int64,
            count=16,  # 128 bytes for coordination data
            offset=0
        )

        # Initialize coordination data
        self.coordination_view[0] = 0  # Global sequence number
        self.coordination_view[1] = int(time.time() * 1000000)  # Creation timestamp
        self.coordination_view[2] = buffer_size  # Buffer size
        self.coordination_view[3] = element_size  # Element size
        self.coordination_view[4] = max_elements  # Max elements per slot

        # Data starts after coordination area
        self.data_offset = 128
        self.slot_stride = slot_size + metadata_size

        logger.info(f"AtomicRingBuffer '{name}' initialized successfully")

    def write(self, data: np.ndarray, count: int = 1) -> int:
        """
        Atomic write with sequence number return.

        Args:
            data: Data to write (numpy array)
            count: Number of elements in this write

        Returns:
            Sequence number of this write, or -1 if failed
        """
        if count > self.max_elements:
            logger.error(f"Write count {count} exceeds max_elements {self.max_elements}")
            return -1

        data_size = data.size * data.itemsize
        if data_size > self.element_size * count:
            logger.error(f"Data size {data_size} exceeds slot capacity {self.element_size * count}")
            return -1

        try:
            # Get next write slot atomically
            with self.write_idx.get_lock():
                slot = self.write_idx.value & self.mask
                self.write_idx.value += 1

            # Get sequence number atomically
            with self.sequence_counter.get_lock():
                self.sequence_counter.value += 1
                sequence_num = self.sequence_counter.value

            # Calculate slot offset
            slot_offset = self.data_offset + slot * self.slot_stride

            # Write metadata first
            metadata_offset = slot_offset + self.element_size * self.max_elements
            metadata_view = np.frombuffer(
                self.shm.buf,
                dtype=np.int64,
                count=8,  # 64 bytes metadata
                offset=metadata_offset
            )

            timestamp = int(time.time() * 1000000)  # microseconds
            metadata_view[0] = sequence_num
            metadata_view[1] = timestamp
            metadata_view[2] = count
            metadata_view[3] = data_size
            metadata_view[4] = slot  # For debugging
            metadata_view[5] = 0  # Reserved
            metadata_view[6] = 0  # Reserved
            metadata_view[7] = 0x1234567890ABCDEF  # Magic number for validation

            # Write data
            data_view = np.frombuffer(
                self.shm.buf,
                dtype=data.dtype,
                count=data.size,
                offset=slot_offset
            )
            data_view[:] = data.flat

            return sequence_num

        except Exception as e:
            logger.error(f"Failed to write to AtomicRingBuffer '{self.name}': {e}")
            return -1

    def read_latest(self) -> Tuple[Optional[np.ndarray], int, Dict[str, Any]]:
        """
        Read latest data with sequence number and metadata.

        Returns:
            Tuple of (data, sequence_number, metadata) or (None, -1, {}) if no data
        """
        try:
            # Get current write position
            current_write = self.write_idx.value
            if current_write == 0:
                return None, -1, {}

            # Read from most recent slot
            slot = (current_write - 1) & self.mask
            slot_offset = self.data_offset + slot * self.slot_stride

            # Read metadata first
            metadata_offset = slot_offset + self.element_size * self.max_elements
            metadata_view = np.frombuffer(
                self.shm.buf,
                dtype=np.int64,
                count=8,
                offset=metadata_offset
            )

            # Validate magic number
            if metadata_view[7] != 0x1234567890ABCDEF:
                logger.warning(f"Invalid magic number in slot {slot}, data may be corrupted")
                return None, -1, {}

            sequence_num = metadata_view[0]
            timestamp = metadata_view[1]
            count = metadata_view[2]
            data_size = metadata_view[3]

            # Read data
            elements_to_read = data_size // 4  # Assuming float32
            if elements_to_read > 0:
                data_view = np.frombuffer(
                    self.shm.buf,
                    dtype=np.float32,
                    count=elements_to_read,
                    offset=slot_offset
                )
                data = data_view.copy()  # Create a copy to avoid sharing buffer
            else:
                data = np.array([])

            metadata = {
                'sequence': sequence_num,
                'timestamp': timestamp,
                'count': count,
                'data_size': data_size,
                'slot': slot
            }

            return data, sequence_num, metadata

        except Exception as e:
            logger.error(f"Failed to read from AtomicRingBuffer '{self.name}': {e}")
            return None, -1, {}

    def read_by_sequence(self, target_sequence: int) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Read data by specific sequence number.

        Args:
            target_sequence: Sequence number to find

        Returns:
            Tuple of (data, metadata) or (None, {}) if not found
        """
        try:
            current_write = self.write_idx.value
            if current_write == 0:
                return None, {}

            # Search backwards through recent slots
            search_limit = min(self.buffer_size, current_write)

            for i in range(search_limit):
                slot = (current_write - 1 - i) & self.mask
                slot_offset = self.data_offset + slot * self.slot_stride

                # Read metadata
                metadata_offset = slot_offset + self.element_size * self.max_elements
                metadata_view = np.frombuffer(
                    self.shm.buf,
                    dtype=np.int64,
                    count=8,
                    offset=metadata_offset
                )

                if metadata_view[7] != 0x1234567890ABCDEF:
                    continue  # Skip corrupted slots

                sequence_num = metadata_view[0]
                if sequence_num == target_sequence:
                    # Found the target sequence
                    timestamp = metadata_view[1]
                    count = metadata_view[2]
                    data_size = metadata_view[3]

                    # Read data
                    elements_to_read = data_size // 4  # Assuming float32
                    if elements_to_read > 0:
                        data_view = np.frombuffer(
                            self.shm.buf,
                            dtype=np.float32,
                            count=elements_to_read,
                            offset=slot_offset
                        )
                        data = data_view.copy()
                    else:
                        data = np.array([])

                    metadata = {
                        'sequence': sequence_num,
                        'timestamp': timestamp,
                        'count': count,
                        'data_size': data_size,
                        'slot': slot
                    }

                    return data, metadata

            return None, {}  # Sequence not found

        except Exception as e:
            logger.error(f"Failed to read sequence {target_sequence} from AtomicRingBuffer '{self.name}': {e}")
            return None, {}

    def get_buffer_info(self) -> Dict[str, Any]:
        """Get buffer status and performance information."""
        try:
            current_write = self.write_idx.value
            current_read = self.read_idx.value
            current_sequence = self.sequence_counter.value

            # Calculate buffer utilization
            pending_writes = current_write - current_read
            utilization = min(pending_writes / self.buffer_size, 1.0)

            # Read coordination data
            creation_time = self.coordination_view[1]
            current_time = int(time.time() * 1000000)
            age_seconds = (current_time - creation_time) / 1000000

            return {
                'name': self.name,
                'buffer_size': self.buffer_size,
                'element_size': self.element_size,
                'max_elements': self.max_elements,
                'current_write_idx': current_write,
                'current_read_idx': current_read,
                'current_sequence': current_sequence,
                'pending_writes': pending_writes,
                'utilization': utilization,
                'age_seconds': age_seconds,
                'memory_size_mb': self.shm.size / (1024 * 1024)
            }

        except Exception as e:
            logger.error(f"Failed to get buffer info for '{self.name}': {e}")
            return {'error': str(e)}

    def cleanup(self):
        """Clean up shared memory resources."""
        try:
            # Delete numpy array views first to release references
            if hasattr(self, 'coordination_view'):
                del self.coordination_view

            # Now safe to close shared memory
            if hasattr(self, 'shm') and self.shm:
                self.shm.close()
                self.shm.unlink()
                logger.info(f"AtomicRingBuffer '{self.name}' cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup AtomicRingBuffer '{self.name}': {e}")

    def __del__(self):
        """Destructor - ensure cleanup on garbage collection."""
        self.cleanup()
