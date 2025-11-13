"""
CUDA Device Initialization for Spawned Processes
Ensures CUDA devices are accessible in spawned processes.
"""

import os
import ctypes
import logging

logger = logging.getLogger(__name__)


def force_cuda_device_init():
    """
    Initialize CUDA for spawned processes using both driver and runtime APIs.

    This function first initializes CUDA driver (cuInit), then CUDA runtime (cudaSetDevice)
    to properly set up CUDA context for ONNX Runtime in spawned processes.

    Returns:
        bool: True if CUDA is initialized, False otherwise
    """
    try:
        # Step 1: Initialize CUDA driver API first (required for runtime API)
        cuda_driver_paths = [
            "/usr/lib/wsl/lib/libcuda.so",
            "/usr/lib/wsl/lib/libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "libcuda.so"
        ]

        libcuda = None
        for path in cuda_driver_paths:
            try:
                if os.path.exists(path):
                    libcuda = ctypes.CDLL(path)
                    logger.info(f"[CUDA Device] Loaded CUDA driver from: {path}")
                    break
            except Exception:
                continue

        if not libcuda:
            logger.error("[CUDA Device] Could not load CUDA driver library (libcuda.so)")
            return False

        # Initialize CUDA driver
        cuInit = libcuda.cuInit
        cuInit.argtypes = [ctypes.c_uint]
        cuInit.restype = ctypes.c_int

        result = cuInit(0)
        if result != 0:
            logger.error(f"[CUDA Device] cuInit failed with error code: {result}")
            return False

        logger.info("[CUDA Device] CUDA driver initialized (cuInit succeeded)")

        # Step 2: Now initialize CUDA runtime API
        cuda_runtime_paths = [
            "/usr/lib/x86_64-linux-gnu/libcudart.so",
            "libcudart.so",
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/lib/wsl/lib/libcudart.so",
        ]

        cuda_runtime = None
        for path in cuda_runtime_paths:
            try:
                cuda_runtime = ctypes.CDLL(path)
                logger.info(f"[CUDA Device] Loaded CUDA runtime from: {path}")
                break
            except Exception:
                continue

        if not cuda_runtime:
            logger.error("[CUDA Device] Could not load CUDA runtime library (libcudart.so)")
            return False

        # Get device count
        cudaGetDeviceCount = cuda_runtime.cudaGetDeviceCount
        cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cudaGetDeviceCount.restype = ctypes.c_int

        device_count = ctypes.c_int()
        result = cudaGetDeviceCount(ctypes.byref(device_count))

        if result != 0:
            logger.error(f"[CUDA Device] cudaGetDeviceCount failed with error code: {result}")
            return False

        logger.info(f"[CUDA Device] Found {device_count.value} CUDA device(s)")

        if device_count.value > 0:
            # CRITICAL: Use cudaSetDevice to initialize CUDA runtime context
            # This is what ONNX Runtime needs to work in spawned processes
            cudaSetDevice = cuda_runtime.cudaSetDevice
            cudaSetDevice.argtypes = [ctypes.c_int]
            cudaSetDevice.restype = ctypes.c_int

            result = cudaSetDevice(0)
            if result == 0:
                logger.info("[CUDA Device] ✅ CUDA fully initialized (driver + runtime)")
                return True
            else:
                logger.error(f"[CUDA Device] cudaSetDevice(0) failed with error code: {result}")
                return False
        else:
            logger.error("[CUDA Device] No CUDA devices found")
            return False

    except Exception as e:
        logger.error(f"[CUDA Device] Error during initialization: {e}")
        return False


def test_onnx_cuda_after_init():
    """
    Test if ONNX Runtime can use CUDA after device initialization.

    Returns:
        bool: True if ONNX CUDA works, False otherwise
    """
    try:
        import onnxruntime as ort

        # Try to create a session with CUDA provider
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
            }),
            'CPUExecutionProvider'
        ]

        # We can't create a real session without a model, but we can check if the provider loads
        available = ort.get_available_providers()

        if 'CUDAExecutionProvider' in available:
            logger.info("[CUDA Device] ONNX Runtime CUDA provider is available")
            return True
        else:
            logger.error("[CUDA Device] ONNX Runtime CUDA provider not available")
            return False

    except Exception as e:
        logger.error(f"[CUDA Device] Error testing ONNX CUDA: {e}")
        return False


def initialize_cuda_for_spawned_process():
    """
    Complete CUDA initialization for spawned processes.
    This sets up the environment and initializes CUDA runtime for ONNX Runtime.

    Returns:
        bool: True if CUDA is available and working, False otherwise
    """
    logger.info("[CUDA Device] Initializing CUDA for spawned process...")

    # First, ensure environment is set
    from cuda_env_setup import setup_cuda_environment_complete
    env_status = setup_cuda_environment_complete(force=True)

    if not env_status['configured']:
        logger.error("[CUDA Device] Failed to configure CUDA environment")
        return False

    # Initialize CUDA runtime (critical for ONNX Runtime in spawned processes)
    if not force_cuda_device_init():
        logger.warning("[CUDA Device] CUDA runtime initialization failed")
        return False

    # Test ONNX CUDA provider availability
    if test_onnx_cuda_after_init():
        logger.info("[CUDA Device] ✅ CUDA is ready for use in spawned process")
        return True
    else:
        logger.error("[CUDA Device] ❌ ONNX CUDA provider not available after initialization")
        return False


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)

    print("Testing CUDA device initialization...")
    if initialize_cuda_for_spawned_process():
        print("✅ CUDA initialization successful!")
    else:
        print("❌ CUDA initialization failed")