"""
Buffer Management - Modular shared memory buffer system.

This package provides high-performance inter-process communication via shared memory.

Core Components:
- BufferCoordinator: Centralized buffer management
- AtomicRingBuffer: Lock-free ring buffer with atomic operations
- CommandBuffer: Inter-process command communication
- ParticipantQueryBuffer: Participant recovery queries

Legacy Components (deprecated):
- SharedScoreBuffer
- NumpySharedBuffer

For backward compatibility, all components can be imported from .sharedbuffer:
    from core.buffer_management.coordinator import BufferCoordinator

For new code, import directly from specific modules:
    from core.buffer_management.coordinator import BufferCoordinator
    from core.buffer_management.ring_buffer import AtomicRingBuffer
    from core.buffer_management.command_buffer import CommandBuffer
"""

# Re-export main classes for convenience
from .coordinator import BufferCoordinator
from .ring_buffer import AtomicRingBuffer
from .command_buffer import CommandBuffer
from .sharedbuffer import ParticipantQueryBuffer

# Legacy imports
from .legacy import SharedScoreBuffer, NumpySharedBuffer

__all__ = [
    'BufferCoordinator',
    'AtomicRingBuffer',
    'CommandBuffer',
    'ParticipantQueryBuffer',
    # Legacy
    'SharedScoreBuffer',
    'NumpySharedBuffer',
]
