"""
Camera Sources Module

Provides abstraction layer for different camera input sources:
- V4L2 (local Linux camera devices)
- ZeroMQ (network camera bridge from Windows)

This allows transparent switching between camera backends without
changing the main camera worker logic.
"""

from .base import CameraSource
from .v4l2_source import V4L2CameraSource
from .zmq_source import ZMQCameraSource
from .factory import CameraSourceFactory

__all__ = [
    'CameraSource',
    'V4L2CameraSource',
    'ZMQCameraSource',
    'CameraSourceFactory'
]
