"""
Shared Buffer - Unified buffer management system (Backward Compatibility Module).

This module re-exports classes from the modular buffer management system for
backward compatibility. New code should import directly from the specific modules:

- BufferCoordinator -> coordinator.py (import directly to avoid circular import)
- AtomicRingBuffer -> ring_buffer.py
- CommandBuffer -> command_buffer.py
- Legacy classes -> legacy/legacy_buffers.py

Active classes defined here:
- ParticipantQueryBuffer (used by participant management)
"""

import logging
import time
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Import from modular locations (excluding BufferCoordinator to avoid circular import)
from .ring_buffer import AtomicRingBuffer
from .command_buffer import CommandBuffer
from .legacy import SharedScoreBuffer, NumpySharedBuffer


logger = logging.getLogger('BufferSystem')


# Re-export for backward compatibility
__all__ = [
    'AtomicRingBuffer',
    'CommandBuffer',
    'ParticipantQueryBuffer',
    'SharedScoreBuffer',  # Legacy
    'NumpySharedBuffer',  # Legacy
]


class ParticipantQueryBuffer:
    """
    Specialized shared memory buffer for participant recovery queries.
    Enables bi-directional communication between detection process and participant manager.

    CRITICAL FIX: Uses shared memory for ALL coordination data to enable cross-process communication.
    """

    def __init__(self, name: str, max_queries: int = 16):
        """
        Initialize participant query buffer.

        Args:
            name: Shared memory name (e.g., 'yq_recovery_0')
            max_queries: Maximum concurrent queries
        """
        self.name = name
        self.max_queries = max_queries

        # Query structure size (in bytes)
        # camera_idx (4) + bbox (16) + shape (478*3*4) + embedding (512*4) + metadata (64)
        self.query_size = 4 + 16 + 478*3*4 + 512*4 + 64  # ~7.9KB per query
        self.response_size = 128  # participant_id + scores + metadata

        # Total buffer size
        total_size = (self.query_size + self.response_size) * max_queries + 1024  # coordination data

        # Create shared memory
        self.shm = shared_memory.SharedMemory(create=True, name=name, size=total_size)
        self._created = True

        # Create numpy views for structured access
        self._init_buffer_views()

        # Initialize atomic indices IN SHARED MEMORY (not separate Value objects)
        self.coordination[0] = 0  # query_write_idx
        self.coordination[1] = 0  # query_read_idx
        self.coordination[2] = 0  # response_write_idx
        self.coordination[3] = 0  # response_read_idx
        self.coordination[4] = max_queries  # max_queries for validation
        self.coordination[5] = int(time.time())  # creation timestamp

        logger.info(f"Created ParticipantQueryBuffer '{name}' with {max_queries} slots, indices in shared memory")

    def _init_buffer_views(self):
        """Initialize structured numpy views into shared memory."""
        # Coordination area (first 1024 bytes)
        self.coordination = np.frombuffer(self.shm.buf, dtype=np.int32, count=256, offset=0)

        # Query area
        query_offset = 1024
        self.query_camera_idx = np.frombuffer(
            self.shm.buf, dtype=np.int32,
            count=self.max_queries,
            offset=query_offset
        )

        query_offset += self.max_queries * 4
        self.query_bboxes = np.frombuffer(
            self.shm.buf, dtype=np.float32,
            count=self.max_queries * 4,
            offset=query_offset
        ).reshape(self.max_queries, 4)

        query_offset += self.max_queries * 16
        self.query_shapes = np.frombuffer(
            self.shm.buf, dtype=np.float32,
            count=self.max_queries * 478 * 3,
            offset=query_offset
        ).reshape(self.max_queries, 478, 3)

        query_offset += self.max_queries * 478 * 3 * 4
        self.query_embeddings = np.frombuffer(
            self.shm.buf, dtype=np.float32,
            count=self.max_queries * 512,
            offset=query_offset
        ).reshape(self.max_queries, 512)

        # Response area
        response_offset = 1024 + self.query_size * self.max_queries
        self.response_participant_ids = np.frombuffer(
            self.shm.buf, dtype=np.int32,
            count=self.max_queries,
            offset=response_offset
        )

        response_offset += self.max_queries * 4
        self.response_scores = np.frombuffer(
            self.shm.buf, dtype=np.float32,
            count=self.max_queries * 16,
            offset=response_offset
        ).reshape(self.max_queries, 16)

    def submit_query(self, camera_idx: int, bbox: np.ndarray,
                    shape: Optional[np.ndarray] = None,
                    embedding: Optional[np.ndarray] = None,
                    zombie_track_id: Optional[int] = None,
                    zombie_participant_id: Optional[int] = None) -> int:
        """
        Submit a participant recovery query.

        Returns:
            Query ID for retrieving response
        """
        # Atomic increment of write index in shared memory
        import threading

        # Simple atomic increment using numpy
        query_write_idx = int(self.coordination[0])
        slot = query_write_idx % self.max_queries
        query_id = query_write_idx
        self.coordination[0] = query_write_idx + 1  # Atomic write

        # Write query data
        self.query_camera_idx[slot] = camera_idx
        self.query_bboxes[slot] = bbox

        if shape is not None:
            self.query_shapes[slot] = shape
        else:
            self.query_shapes[slot] = 0  # Clear

        if embedding is not None:
            self.query_embeddings[slot] = embedding
        else:
            self.query_embeddings[slot] = 0  # Clear

        # Store zombie track info in coordination array
        if zombie_track_id is not None:
            self.coordination[24 + slot] = zombie_track_id
        else:
            self.coordination[24 + slot] = -1  # Use -1 for None

        if zombie_participant_id is not None:
            self.coordination[40 + slot] = zombie_participant_id
        else:
            self.coordination[40 + slot] = -1  # Use -1 for None

        # Write sequence/timestamp to signal data is ready
        timestamp_us = int(time.time() * 1000000) & 0x7FFFFFFF  # Limit to 31-bit signed int
        self.coordination[8 + slot] = timestamp_us  # Mark slot as ready with timestamp

        logger.debug(f"Submitted query {query_id} to slot {slot}, timestamp={self.coordination[8 + slot]}")
        return query_id

    def update_query_embedding(self, query_id: int, embedding: np.ndarray):
        """
        Update a pending query with an embedding.

        Args:
            query_id: Query ID returned from submit_query
            embedding: Face embedding vector
        """
        # Calculate slot from query ID
        slot = query_id % self.max_queries

        # Verify this slot is still valid for this query ID
        slot_timestamp = self.coordination[8 + slot]
        if slot_timestamp > 0:  # Query still pending
            self.query_embeddings[slot] = embedding
            logger.debug(f"Updated query {query_id} (slot {slot}) with embedding")
        else:
            logger.warning(f"Query {query_id} (slot {slot}) already processed, cannot update embedding")

    def get_query(self) -> Optional[Dict[str, Any]]:
        """
        Get next query from buffer (non-blocking with polling).

        Returns:
            Query data or None if no query available
        """
        current_read_idx = int(self.coordination[1])
        current_write_idx = int(self.coordination[0])

        # Check if there are unread queries
        if current_read_idx >= current_write_idx:
            return None  # No new queries

        # Get next slot to read
        slot = current_read_idx % self.max_queries

        # Check if this slot has data ready (timestamp > 0)
        slot_timestamp = self.coordination[8 + slot]
        if slot_timestamp == 0:
            return None  # Slot not ready

        # Read query data
        query_id = current_read_idx
        query = {
            'query_id': query_id,
            'camera_idx': int(self.query_camera_idx[slot]),
            'bbox': self.query_bboxes[slot].copy(),
            'shape': self.query_shapes[slot].copy() if np.any(self.query_shapes[slot]) else None,
            'embedding': self.query_embeddings[slot].copy() if np.any(self.query_embeddings[slot]) else None,
            'zombie_track_id': int(self.coordination[24 + slot]) if self.coordination[24 + slot] >= 0 else None,
            'zombie_participant_id': int(self.coordination[40 + slot]) if self.coordination[40 + slot] >= 0 else None
        }

        # Mark slot as consumed and advance read index
        self.coordination[8 + slot] = 0  # Clear timestamp
        self.coordination[1] = current_read_idx + 1  # Advance read index

        logger.debug(f"Retrieved query {query_id} from slot {slot}")
        return query

    def submit_response(self, query_id: int, participant_id: Optional[int],
                       scores: Dict[str, float]):
        """Submit response to a query."""
        slot = query_id % self.max_queries

        # Write response
        self.response_participant_ids[slot] = participant_id if participant_id else -1

        # Pack scores into response array
        score_array = np.zeros(16, dtype=np.float32)
        score_array[0] = scores.get('embedding', 0.0)
        score_array[1] = scores.get('shape', 0.0)
        score_array[2] = scores.get('position', 0.0)
        score_array[3] = scores.get('combined', 0.0)
        self.response_scores[slot] = score_array

        # Signal response available by setting timestamp in coordination
        response_slot_offset = 8 + self.max_queries + slot  # After query timestamps
        timestamp_us = int(time.time() * 1000000) & 0x7FFFFFFF  # Limit to 31-bit signed int
        self.coordination[response_slot_offset] = timestamp_us

        logger.debug(f"Submitted response for query {query_id} to slot {slot}, "
                    f"participant_id={participant_id}, timestamp={self.coordination[response_slot_offset]}")

    def get_response(self, query_id: int, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get response for a query with polling."""
        slot = query_id % self.max_queries
        response_slot_offset = 8 + self.max_queries + slot

        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if response is available (timestamp > 0)
            if self.coordination[response_slot_offset] > 0:
                # Read response data
                participant_id = int(self.response_participant_ids[slot])
                if participant_id == -1:
                    participant_id = None

                scores = {
                    'embedding': float(self.response_scores[slot, 0]),
                    'shape': float(self.response_scores[slot, 1]),
                    'position': float(self.response_scores[slot, 2]),
                    'combined': float(self.response_scores[slot, 3])
                }

                # Clear response timestamp to mark as consumed
                self.coordination[response_slot_offset] = 0

                logger.debug(f"Retrieved response for query {query_id} from slot {slot}, participant_id={participant_id}")
                return {
                    'participant_id': participant_id,
                    'scores': scores
                }

            # Brief sleep to avoid busy waiting
            time.sleep(0.001)

        return None  # Timeout

    @classmethod
    def connect(cls, name: str, max_queries: int = 16):
        """
        Connect to an existing ParticipantQueryBuffer.

        Args:
            name: Shared memory name to connect to
            max_queries: Expected number of query slots

        Returns:
            ParticipantQueryBuffer instance connected to existing memory
        """
        instance = object.__new__(cls)
        instance.name = name
        instance.max_queries = max_queries
        instance._created = False

        # Connect to existing shared memory
        instance.shm = shared_memory.SharedMemory(name=name)

        # Initialize buffer views
        instance.query_size = 4 + 16 + 478*3*4 + 512*4 + 64
        instance.response_size = 128
        instance._init_buffer_views()

        # Validate connection by checking coordination data
        stored_max_queries = int(instance.coordination[4])
        creation_time = int(instance.coordination[5])

        if stored_max_queries != max_queries:
            logger.warning(f"Max queries mismatch: expected {max_queries}, found {stored_max_queries}")

        logger.info(f"Connected to ParticipantQueryBuffer '{name}' (created at {creation_time}, max_queries={stored_max_queries})")
        return instance

    def cleanup(self):
        """Cleanup shared memory."""
        try:
            # First delete all numpy array views to release references
            if hasattr(self, 'coordination'):
                del self.coordination
            if hasattr(self, 'query_camera_idx'):
                del self.query_camera_idx
            if hasattr(self, 'query_bboxes'):
                del self.query_bboxes
            if hasattr(self, 'query_shapes'):
                del self.query_shapes
            if hasattr(self, 'query_embeddings'):
                del self.query_embeddings
            if hasattr(self, 'response_participant_ids'):
                del self.response_participant_ids
            if hasattr(self, 'response_scores'):
                del self.response_scores

            # Close shared memory
            self.shm.close()

            # Only unlink if we created the buffer
            if getattr(self, '_created', False):
                self.shm.unlink()
                logger.info(f"ParticipantQueryBuffer '{self.name}' cleaned up and unlinked")
            else:
                logger.info(f"ParticipantQueryBuffer '{self.name}' disconnected")

        except Exception as e:
            logger.error(f"Failed to cleanup ParticipantQueryBuffer '{self.name}': {e}")

    def __del__(self):
        """Destructor - ensure cleanup on garbage collection."""
        # Cleanup handled explicitly to avoid issues with numpy views
        pass
