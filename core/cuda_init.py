"""
CUDA Environment Initialization Module for YouQuantiPy
Ensures proper CUDA environment setup before loading CUDA-dependent libraries.

This module MUST be imported before any CUDA-related imports (cv2, onnxruntime).

Author: YouQuantiPy Team
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)


def setup_cuda_environment(force=False):
    """
    Configure environment variables for CUDA in WSL2 and native Linux.

    This function ensures that CUDA libraries are discoverable by setting
    the LD_LIBRARY_PATH environment variable correctly.

    Args:
        force: Force re-setup even if already configured

    Returns:
        bool: True if environment was modified, False otherwise
    """
    # Check if already configured (unless forcing)
    if not force and os.environ.get("YOUQUANTIPY_CUDA_INIT") == "1":
        return False

    # Library paths for CUDA and cuDNN
    library_paths = [
        "/usr/lib/x86_64-linux-gnu",  # CUDA runtime and cuDNN location
        "/usr/lib/wsl/lib",            # WSL2 NVIDIA driver (libcuda.so)
    ]

    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    paths_added = []

    # Add each path if not already present
    for path in library_paths:
        if os.path.exists(path) and path not in current_ld_path.split(":"):
            paths_added.append(path)

    if paths_added:
        # Prepend new paths to LD_LIBRARY_PATH
        new_ld_path = ":".join(paths_added + [current_ld_path] if current_ld_path else paths_added)
        os.environ["LD_LIBRARY_PATH"] = new_ld_path

        logger.info(f"[CUDA Init] Added to LD_LIBRARY_PATH: {':'.join(paths_added)}")
        logger.info(f"[CUDA Init] Final LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
    else:
        logger.debug("[CUDA Init] LD_LIBRARY_PATH already configured")

    # Mark as initialized
    os.environ["YOUQUANTIPY_CUDA_INIT"] = "1"

    return bool(paths_added)


def verify_cuda_availability():
    """
    Verify CUDA availability for both OpenCV and ONNX Runtime.

    Returns:
        dict: Status of CUDA for each library
    """
    status = {
        "opencv_cuda": False,
        "opencv_devices": 0,
        "onnx_cuda": False,
        "onnx_providers": [],
        "environment_configured": os.environ.get("YOUQUANTIPY_CUDA_INIT") == "1"
    }

    # Check OpenCV CUDA
    try:
        import cv2
        if hasattr(cv2, 'cuda'):
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            status["opencv_cuda"] = device_count > 0
            status["opencv_devices"] = device_count
            logger.info(f"[CUDA Init] OpenCV CUDA: {device_count} device(s) available")
        else:
            logger.warning("[CUDA Init] OpenCV not compiled with CUDA support")
    except Exception as e:
        logger.error(f"[CUDA Init] OpenCV CUDA check failed: {e}")

    # Check ONNX Runtime CUDA
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        status["onnx_providers"] = providers
        status["onnx_cuda"] = 'CUDAExecutionProvider' in providers

        if status["onnx_cuda"]:
            logger.info("[CUDA Init] ONNX Runtime CUDA provider available")
        else:
            logger.warning("[CUDA Init] ONNX Runtime CUDA provider not available")
            logger.info(f"[CUDA Init] Available providers: {providers}")
    except Exception as e:
        logger.error(f"[CUDA Init] ONNX Runtime check failed: {e}")

    return status


def initialize_cuda_context():
    """
    Initialize CUDA context to ensure GPU is ready.

    This is especially important for multiprocessing scenarios where
    child processes may not inherit the CUDA context.
    """
    try:
        import cv2
        import numpy as np
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # Create a small GPU matrix to initialize context
            test_mat = cv2.cuda.GpuMat(1, 1, cv2.CV_8UC1)
            test_mat.upload(np.array([[0]], dtype=np.uint8))
            test_mat.download()
            logger.debug("[CUDA Init] CUDA context initialized")
            return True
    except Exception as e:
        logger.debug(f"[CUDA Init] Could not initialize CUDA context: {e}")

    return False


def initialize_opencv_cuda_spawn():
    """
    Special initialization for OpenCV CUDA in spawned processes.

    OpenCV CUDA has known issues with spawn multiprocessing context.
    This function attempts various workarounds to enable CUDA in spawned processes.

    Returns:
        tuple: (success: bool, device_count: int, error_msg: str or None)
    """
    import numpy as np

    # Ensure environment is set
    setup_cuda_environment(force=True)

    try:
        import cv2

        if not hasattr(cv2, 'cuda'):
            return False, 0, "OpenCV not built with CUDA support"

        # Initial device check
        initial_devices = cv2.cuda.getCudaEnabledDeviceCount()
        logger.info(f"[OpenCV CUDA Spawn] Initial device count: {initial_devices}")

        if initial_devices > 0:
            # Try to initialize CUDA context
            try:
                cv2.cuda.setDevice(0)
                # Force context creation with actual operation
                test_mat = cv2.cuda.GpuMat(1, 1, cv2.CV_8UC1)
                test_data = np.zeros((1, 1), dtype=np.uint8)
                test_mat.upload(test_data)
                result = test_mat.download()

                logger.info("[OpenCV CUDA Spawn] ✅ CUDA context initialized successfully")
                return True, initial_devices, None

            except Exception as init_error:
                logger.warning(f"[OpenCV CUDA Spawn] Context init failed: {init_error}")
                # Fall through to workarounds

        # Workaround 1: Try device reset
        logger.info("[OpenCV CUDA Spawn] Attempting device reset...")
        try:
            cv2.cuda.resetDevice()
            devices_after_reset = cv2.cuda.getCudaEnabledDeviceCount()
            logger.info(f"[OpenCV CUDA Spawn] After reset: {devices_after_reset} devices")

            if devices_after_reset > 0:
                cv2.cuda.setDevice(0)
                test_mat = cv2.cuda.GpuMat(1, 1, cv2.CV_8UC1)
                return True, devices_after_reset, None

        except Exception as reset_error:
            logger.warning(f"[OpenCV CUDA Spawn] Reset failed: {reset_error}")

        # Workaround 2: Re-import cv2 (sometimes helps)
        logger.info("[OpenCV CUDA Spawn] Attempting cv2 reimport...")
        try:
            import importlib
            importlib.reload(cv2)
            devices_after_reload = cv2.cuda.getCudaEnabledDeviceCount()
            logger.info(f"[OpenCV CUDA Spawn] After reload: {devices_after_reload} devices")

            if devices_after_reload > 0:
                return True, devices_after_reload, None

        except Exception as reload_error:
            logger.warning(f"[OpenCV CUDA Spawn] Reload failed: {reload_error}")

        # All workarounds failed
        error_msg = (
            "OpenCV CUDA not available in spawn context. "
            "This is a known limitation. ONNX CUDA should still work."
        )
        logger.warning(f"[OpenCV CUDA Spawn] {error_msg}")
        return False, 0, error_msg

    except Exception as e:
        error_msg = f"OpenCV CUDA initialization failed: {e}"
        logger.error(f"[OpenCV CUDA Spawn] {error_msg}")
        return False, 0, error_msg


# Auto-initialize when module is imported
_initialized = False

def auto_init():
    """Automatically initialize CUDA environment on import."""
    global _initialized
    if not _initialized:
        # Setup environment
        setup_cuda_environment()

        # Verify availability (optional, for logging)
        if logger.isEnabledFor(logging.DEBUG):
            verify_cuda_availability()

        _initialized = True


# Import guard to prevent numpy from being imported before environment setup
try:
    import numpy as np
except ImportError:
    # NumPy not yet imported, which is fine
    pass


# Run auto-initialization
auto_init()


# Convenience function for explicit initialization
def ensure_cuda_initialized():
    """
    Ensure CUDA environment is properly initialized.

    Call this function before using any CUDA functionality to guarantee
    proper environment setup.
    """
    if not os.environ.get("YOUQUANTIPY_CUDA_INIT") == "1":
        setup_cuda_environment(force=True)
        initialize_cuda_context()

    return verify_cuda_availability()


if __name__ == "__main__":
    # Test CUDA initialization
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("CUDA Initialization Test")
    print("="*60)

    # Force re-initialization for testing
    setup_cuda_environment(force=True)

    # Verify and display status
    status = verify_cuda_availability()

    print("\n📊 CUDA Status:")
    print(f"   Environment configured: {status['environment_configured']}")
    print(f"   OpenCV CUDA: {status['opencv_cuda']} ({status['opencv_devices']} devices)")
    print(f"   ONNX Runtime CUDA: {status['onnx_cuda']}")
    print(f"   ONNX Providers: {status['onnx_providers']}")

    # Initialize context
    if initialize_cuda_context():
        print("\n✅ CUDA context initialized successfully")
    else:
        print("\n⚠️  Could not initialize CUDA context")