"""
Centralized CUDA Environment Setup Module for YouQuantiPy
Ensures consistent CUDA configuration across all processes.

This module MUST be imported before any CUDA-dependent libraries.
"""

import os
import sys
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def setup_cuda_environment_complete(force: bool = False) -> Dict[str, any]:
    """
    Complete CUDA environment setup for GPU acceleration.
    This ensures CUDA libraries are discoverable by all GPU-dependent modules.

    Args:
        force: Force re-setup even if already configured

    Returns:
        dict: Status information about CUDA setup
    """
    status = {
        "configured": False,
        "ld_library_path": "",
        "cuda_visible_devices": "",
        "changes_made": [],
        "providers_available": []
    }

    # Check if already configured (unless forcing)
    if not force and os.environ.get("YOUQUANTIPY_CUDA_CONFIGURED") == "1":
        status["configured"] = True
        status["ld_library_path"] = os.environ.get("LD_LIBRARY_PATH", "")
        status["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        logger.debug("[CUDA ENV] Already configured, skipping setup")
        return status

    # Essential CUDA library paths
    cuda_paths = [
        "/usr/lib/x86_64-linux-gnu",  # CUDA runtime, cuDNN, cuBLAS
        "/usr/lib/wsl/lib",            # WSL2 NVIDIA driver (libcuda.so)
        "/usr/local/cuda/lib64",       # Standard CUDA installation (if exists)
        "/usr/local/cuda-12.0/lib64",  # Specific CUDA 12.0 (if exists)
        "/usr/local/cuda-12.8/lib64",  # Specific CUDA 12.8 (if exists)
    ]

    # Get current LD_LIBRARY_PATH
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    current_paths = current_ld_path.split(":") if current_ld_path else []

    # Add paths that exist and aren't already present
    new_paths = []
    for path in cuda_paths:
        if os.path.exists(path) and path not in current_paths:
            new_paths.append(path)
            status["changes_made"].append(f"Added {path} to LD_LIBRARY_PATH")

    # Prepend new paths to LD_LIBRARY_PATH (CUDA paths should come first)
    if new_paths:
        all_paths = new_paths + current_paths
        os.environ["LD_LIBRARY_PATH"] = ":".join(all_paths)
        logger.info(f"[CUDA ENV] Updated LD_LIBRARY_PATH with {len(new_paths)} new paths")
    else:
        logger.debug("[CUDA ENV] No new paths to add to LD_LIBRARY_PATH")

    status["ld_library_path"] = os.environ.get("LD_LIBRARY_PATH", "")

    # Set CUDA_VISIBLE_DEVICES if not set (use all devices)
    # In WSL2, we need to ensure this is set for CUDA to work
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        status["changes_made"].append("Set CUDA_VISIBLE_DEVICES=0")

    # WSL2-specific: Ensure libcuda.so can be found
    # This is critical for cudaGetDeviceCount to work
    if os.path.exists("/usr/lib/wsl/lib/libcuda.so"):
        # Force CUDA to look in WSL lib directory
        os.environ["CUDA_PATH"] = "/usr/lib/wsl/lib"
        status["changes_made"].append("Set CUDA_PATH for WSL2")

    status["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    # Additional CUDA optimizations
    # Disable TensorFlow's memory pre-allocation for better sharing
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Set CUDA cache directory (for kernel compilation)
    if not os.environ.get("CUDA_CACHE_PATH"):
        cache_dir = os.path.expanduser("~/.cache/cuda")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["CUDA_CACHE_PATH"] = cache_dir
        status["changes_made"].append(f"Set CUDA_CACHE_PATH={cache_dir}")

    # Mark as configured
    os.environ["YOUQUANTIPY_CUDA_CONFIGURED"] = "1"
    status["configured"] = True

    # Log final configuration
    logger.info(f"[CUDA ENV] Configuration complete:")
    logger.info(f"[CUDA ENV]   LD_LIBRARY_PATH: {status['ld_library_path'][:200]}...")
    logger.info(f"[CUDA ENV]   CUDA_VISIBLE_DEVICES: {status['cuda_visible_devices']}")
    logger.info(f"[CUDA ENV]   Changes made: {len(status['changes_made'])}")

    # Try to check available providers (without importing heavy libraries)
    try:
        import onnxruntime as ort
        status["providers_available"] = ort.get_available_providers()
        logger.info(f"[CUDA ENV] ONNX providers available: {status['providers_available'][:3]}")
    except ImportError:
        logger.debug("[CUDA ENV] ONNX Runtime not installed, skipping provider check")
    except Exception as e:
        logger.warning(f"[CUDA ENV] Could not check ONNX providers: {e}")

    return status


def get_cuda_environment_for_spawn() -> Dict[str, str]:
    """
    Get environment variables dict for spawning CUDA-enabled processes.
    This ensures child processes inherit proper CUDA configuration.

    Returns:
        dict: Environment variables to pass to spawned process
    """
    # Start with current environment
    env = os.environ.copy()

    # Ensure CUDA paths are set
    setup_cuda_environment_complete()

    # Critical CUDA environment variables
    cuda_vars = [
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_CACHE_PATH",
        "TF_FORCE_GPU_ALLOW_GROWTH",
        "YOUQUANTIPY_CUDA_CONFIGURED",
        # WSL2-specific
        "LIBGL_DRIVERS_PATH",
        "MESA_LOADER_DRIVER_OVERRIDE",
        "MESA_D3D12_DEFAULT_ADAPTER_NAME",
    ]

    # Ensure all critical variables are in the environment dict
    for var in cuda_vars:
        if var in os.environ:
            env[var] = os.environ[var]

    # Add any missing library paths
    if "LD_LIBRARY_PATH" not in env or not env["LD_LIBRARY_PATH"]:
        # Reconstruct minimal LD_LIBRARY_PATH if missing
        env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib"

    logger.debug(f"[CUDA ENV] Prepared environment for spawn with {len(env)} variables")
    logger.debug(f"[CUDA ENV]   LD_LIBRARY_PATH for spawn: {env.get('LD_LIBRARY_PATH', 'NOT SET')[:100]}...")

    return env


def verify_cuda_in_process() -> Dict[str, any]:
    """
    Verify CUDA is accessible and functional in the current process.

    WARNING: This function only checks if providers are available, NOT if they work!
    For threading mode, CUDA should work automatically (shared context with main process).
    For multiprocessing spawn mode, CUDA typically fails with error 100 (no device detected).

    Returns:
        dict: CUDA availability status (NOTE: availability != functionality!)
    """
    status = {
        "environment_set": False,
        "onnx_cuda_listed": False,  # Renamed: only checks if listed, not if it works!
        "opencv_cuda": False,
        "cuda_device_count": 0,
        "error": None
    }

    # Check environment
    status["environment_set"] = bool(os.environ.get("LD_LIBRARY_PATH"))

    # Check ONNX Runtime CUDA provider (only checks if listed)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        status["onnx_cuda_listed"] = "CUDAExecutionProvider" in providers

        # NOTE: We cannot verify if CUDA actually works without loading a model
        # Provider being "available" doesn't mean it will work (common in spawn mode)
        # Real test happens when model loads - check for silent CPU fallback

    except ImportError:
        status["error"] = "ONNX Runtime not installed"
    except Exception as e:
        status["error"] = f"ONNX check failed: {str(e)}"

    # Check OpenCV CUDA
    try:
        import cv2
        if hasattr(cv2, 'cuda'):
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            status["opencv_cuda"] = device_count > 0
            status["cuda_device_count"] = device_count
        else:
            status["error"] = "OpenCV not built with CUDA"
    except Exception as e:
        if not status["error"]:
            status["error"] = f"OpenCV check failed: {str(e)}"

    # Log status with accurate messaging
    logger.info(f"[CUDA ENV] Process CUDA status:")
    logger.info(f"[CUDA ENV]   Environment set: {status['environment_set']}")
    logger.info(f"[CUDA ENV]   ONNX CUDA listed: {status['onnx_cuda_listed']} (NOTE: may not work until model loads)")
    logger.info(f"[CUDA ENV]   OpenCV CUDA: {status['opencv_cuda']} ({status['cuda_device_count']} devices)")
    if status["error"]:
        logger.warning(f"[CUDA ENV]   Error: {status['error']}")

    return status


# Auto-initialize when module is imported
def _auto_init():
    """Automatically initialize CUDA environment on import."""
    # Only setup in main module import, not in spawned processes that will call explicitly
    if __name__ != "__mp_main__":
        setup_cuda_environment_complete()


# Initialize on import
_auto_init()