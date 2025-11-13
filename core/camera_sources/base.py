"""
Abstract Base Class for Camera Sources

Defines the interface that all camera sources must implement,
allowing transparent switching between V4L2, ZeroMQ, and future sources.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class CameraSource(ABC):
    """
    Abstract base class for camera input sources.

    All camera sources must implement this interface to be compatible
    with the EnhancedCameraWorker pipeline.
    """

    @abstractmethod
    def open(self) -> bool:
        """
        Open the camera source and initialize connection.

        Returns:
            bool: True if opened successfully, False otherwise
        """
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera source.

        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - success: True if frame read successfully
                - frame: BGR numpy array (H, W, 3) or None on failure
        """
        pass

    @abstractmethod
    def release(self):
        """
        Release the camera source and clean up resources.
        """
        pass

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """
        Get the actual resolution of the camera source.

        Returns:
            Tuple[int, int]: (width, height) in pixels
        """
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """
        Get the actual frame rate of the camera source.

        Returns:
            float: Frames per second
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """
        Check if the camera source is currently opened and ready.

        Returns:
            bool: True if opened and ready, False otherwise
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """
        Get the name of the backend (e.g., "V4L2", "ZMQ", "DSHOW").

        Returns:
            str: Backend name for logging and diagnostics
        """
        pass

    def warmup(self, num_frames: int = 10):
        """
        Read and discard frames to warm up the camera pipeline.

        This helps stabilize auto-exposure and removes stale buffered frames.

        Args:
            num_frames: Number of frames to read and discard
        """
        for _ in range(num_frames):
            self.read()
