"""
OpenCV Environment Setup for Linux

In the Windows version, this module sets up CUDA paths for OpenCV.
In the Linux version, OpenCV already has proper GPU support via system libraries.
This module is kept for compatibility but does nothing on Linux.
"""

import os
import sys

# No-op for Linux - OpenCV GPU support is handled by system libraries
# MediaPipe uses OpenGL ES 3.2, not CUDA

def setup_opencv_cuda():
    """No-op function for Linux compatibility"""
    pass

# Auto-run on import (for compatibility with Windows version)
setup_opencv_cuda()
