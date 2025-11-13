"""
Legacy buffer classes - Deprecated, kept for backward compatibility.

These classes are no longer actively used and will be removed in a future version.
Use BufferCoordinator, AtomicRingBuffer, or CommandBuffer instead.
"""

from .legacy_buffers import SharedScoreBuffer, NumpySharedBuffer

__all__ = ['SharedScoreBuffer', 'NumpySharedBuffer']
