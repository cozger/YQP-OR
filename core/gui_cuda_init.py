"""
GUI-specific CUDA initialization module.
This module MUST be imported at the very top of gui.py BEFORE any other imports
to ensure CUDA is properly initialized before cv2 or any other CUDA-using library.
"""

import os
import sys
import logging

# Import the centralized CUDA setup module
try:
    from .cuda_env_setup import setup_cuda_environment_complete, verify_cuda_in_process
except ImportError:
    # Fallback if running as standalone script
    sys.path.insert(0, os.path.dirname(__file__))
    from cuda_env_setup import setup_cuda_environment_complete, verify_cuda_in_process

def initialize_gui_cuda_environment():
    """
    Initialize CUDA environment for GUI process using centralized setup.
    This function MUST be called before importing cv2 or any CUDA-using libraries.

    Returns:
        bool: True if initialization successful
    """
    logger = logging.getLogger('GUI_CUDA_Init')

    # Use centralized CUDA setup
    logger.info("Initializing GUI CUDA environment using centralized setup...")

    cuda_status = setup_cuda_environment_complete(force=False)

    if cuda_status['configured']:
        logger.info(f"GUI CUDA environment configured successfully")
        logger.info(f"  LD_LIBRARY_PATH: {cuda_status['ld_library_path'][:100]}...")
        logger.info(f"  CUDA_VISIBLE_DEVICES: {cuda_status['cuda_visible_devices']}")

        if cuda_status['changes_made']:
            logger.info(f"  Changes made: {len(cuda_status['changes_made'])} modifications")
            for change in cuda_status['changes_made'][:3]:  # Show first 3 changes
                logger.debug(f"    - {change}")

        if cuda_status.get('providers_available'):
            logger.info(f"  ONNX providers: {cuda_status['providers_available'][:3]}")

        return True
    else:
        logger.error("Failed to configure GUI CUDA environment")
        return False


def verify_cuda_after_cv2_import():
    """
    Verify CUDA is available after cv2 import.
    Call this AFTER importing cv2 to confirm CUDA detection worked.

    Returns:
        dict: CUDA status information
    """
    import cv2

    status = {
        'opencv_cuda': False,
        'opencv_devices': 0,
        'cuda_initialized': False,
        'error': None
    }

    try:
        if hasattr(cv2, 'cuda'):
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            status['opencv_devices'] = device_count
            status['opencv_cuda'] = device_count > 0

            if device_count > 0:
                # Try to initialize CUDA context
                try:
                    cv2.cuda.setDevice(0)
                    # Small test to verify context
                    test_mat = cv2.cuda.GpuMat(1, 1, cv2.CV_8UC1)
                    status['cuda_initialized'] = True
                    print(f"[GUI_CUDA] ✅ OpenCV CUDA initialized: {device_count} device(s)")
                except Exception as e:
                    status['error'] = str(e)
                    print(f"[GUI_CUDA] ⚠️ OpenCV CUDA init failed: {e}")
            else:
                print("[GUI_CUDA] ⚠️ No CUDA devices detected by OpenCV")
        else:
            print("[GUI_CUDA] ⚠️ OpenCV built without CUDA support")

    except Exception as e:
        status['error'] = str(e)
        print(f"[GUI_CUDA] ❌ Error checking CUDA: {e}")

    return status


def verify_onnx_cuda():
    """
    Verify ONNX Runtime CUDA provider availability.

    Returns:
        dict: ONNX CUDA status
    """
    status = {
        'onnx_cuda': False,
        'providers': [],
        'error': None
    }

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        status['providers'] = providers
        status['onnx_cuda'] = 'CUDAExecutionProvider' in providers

        if status['onnx_cuda']:
            print("[GUI_CUDA] ✅ ONNX Runtime CUDA provider available")
        else:
            print("[GUI_CUDA] ⚠️ ONNX Runtime CUDA provider not available")
            print(f"[GUI_CUDA]    Available providers: {providers}")

    except ImportError:
        status['error'] = "onnxruntime not installed"
        print("[GUI_CUDA] ⚠️ onnxruntime not installed")
    except Exception as e:
        status['error'] = str(e)
        print(f"[GUI_CUDA] ❌ Error checking ONNX: {e}")

    return status


def verify_cuda_device_accessible():
    """
    Verify CUDA device is accessible via direct cudaGetDeviceCount call.

    This pre-flight check catches CUDA initialization failures before ONNX tries to use it.
    Uses ctypes to directly call CUDA runtime API.

    Returns:
        dict: {
            'accessible': bool,
            'device_count': int,
            'error': str or None
        }
    """
    import ctypes

    status = {
        'accessible': False,
        'device_count': 0,
        'error': None
    }

    try:
        # Try to load libcuda.so (CUDA driver API)
        cuda_paths = [
            '/usr/lib/wsl/lib/libcuda.so.1',  # WSL2
            '/usr/lib/wsl/lib/libcuda.so',
            '/usr/lib/x86_64-linux-gnu/libcuda.so.1',  # Native Linux
            '/usr/lib/x86_64-linux-gnu/libcuda.so',
            'libcuda.so.1',  # System default
            'libcuda.so'
        ]

        libcuda = None
        for path in cuda_paths:
            try:
                libcuda = ctypes.CDLL(path)
                break
            except OSError:
                continue

        if libcuda is None:
            status['error'] = "Could not load libcuda.so from any known path"
            return status

        # Call cuInit(0) to initialize CUDA
        result = libcuda.cuInit(0)
        if result != 0:
            status['error'] = f"cuInit failed with code {result}"
            return status

        # Call cudaGetDeviceCount via libcudart.so (CUDA runtime API)
        # Try both libcudart paths
        cudart_paths = [
            '/usr/lib/x86_64-linux-gnu/libcudart.so.12',  # CUDA 12.x
            '/usr/lib/x86_64-linux-gnu/libcudart.so',
            'libcudart.so.12',
            'libcudart.so'
        ]

        libcudart = None
        for path in cudart_paths:
            try:
                libcudart = ctypes.CDLL(path)
                break
            except OSError:
                continue

        if libcudart is None:
            status['error'] = "Could not load libcudart.so (CUDA runtime)"
            return status

        # Call cudaGetDeviceCount
        device_count = ctypes.c_int()
        result = libcudart.cudaGetDeviceCount(ctypes.byref(device_count))

        if result != 0:
            status['error'] = f"cudaGetDeviceCount failed with code {result}"
            return status

        status['accessible'] = device_count.value > 0
        status['device_count'] = device_count.value

        if device_count.value == 0:
            status['error'] = "No CUDA devices detected (device count = 0)"

    except Exception as e:
        status['error'] = f"Exception during CUDA verification: {str(e)}"

    return status


def initialize_onnx_cuda_context(model_path=None):
    """
    Initialize ONNX Runtime CUDA context in the main GUI thread.

    CRITICAL: This MUST be called in the main thread BEFORE starting face recognition
    worker threads. Threading mode allows worker threads to inherit the CUDA context
    created here, enabling GPU acceleration.

    Without this initialization, worker threads will fail with "CUDA error 100: no device detected"
    even though CUDA libraries are installed and environment is configured correctly.

    Args:
        model_path: Path to ONNX model (default: ArcFace model)

    Returns:
        dict: Initialization status with keys:
            - success (bool): True if CUDA initialized successfully
            - provider (str): Active provider name
            - latency_ms (float): Inference latency in milliseconds
            - error (str): Error message if failed
    """
    import time
    import numpy as np

    status = {
        'success': False,
        'provider': 'Unknown',
        'latency_ms': 0.0,
        'error': None
    }

    try:
        import onnxruntime as ort

        # Default to ArcFace model
        if model_path is None:
            model_path = "/home/canoz/Projects/youquantipy_mediapipe/models/arcface.onnx"

        if not os.path.exists(model_path):
            status['error'] = f"Model not found: {model_path}"
            print(f"[GUI_CUDA] ❌ Model not found: {model_path}")
            return status

        print(f"[GUI_CUDA] Initializing ONNX CUDA context...")
        print(f"[GUI_CUDA]   Model: {model_path}")
        print(f"[GUI_CUDA]   Available providers: {ort.get_available_providers()}")

        # Verify CUDA device accessibility before ONNX
        cuda_check = verify_cuda_device_accessible()
        if not cuda_check['accessible']:
            print(f"[GUI_CUDA] ⚠️ Pre-flight CUDA check failed!")
            print(f"[GUI_CUDA]   Error: {cuda_check['error']}")
            print(f"[GUI_CUDA]   Skipping CUDA provider, will use CPU")
            status['error'] = f"Pre-flight check failed: {cuda_check['error']}"
            # Don't try CUDA, just use CPU
            providers = ['CPUExecutionProvider']
        else:
            print(f"[GUI_CUDA] ✅ Pre-flight CUDA check passed (device count: {cuda_check['device_count']})")
            # Try CUDA provider
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                }),
                'CPUExecutionProvider'  # Fallback
            ]

        print(f"[GUI_CUDA] 🔄 Creating ONNX InferenceSession with providers: {providers}")

        # Load model with determined providers (either CUDA+CPU or CPU-only)
        session = ort.InferenceSession(model_path, providers=providers)

        # Check which provider is actually being used
        actual_provider = session.get_providers()[0]
        status['provider'] = actual_provider

        if actual_provider == 'CUDAExecutionProvider':
            print(f"[GUI_CUDA] ✅ CUDA provider active!")

            # Benchmark inference to verify it works
            dummy_input = np.random.randn(1, 3, 112, 112).astype(np.float32)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            # Warmup (initializes CUDA kernels)
            for _ in range(5):
                session.run([output_name], {input_name: dummy_input})

            # Measure latency
            start = time.time()
            iterations = 20
            for _ in range(iterations):
                session.run([output_name], {input_name: dummy_input})
            latency = (time.time() - start) / iterations * 1000

            status['success'] = True
            status['latency_ms'] = latency

            print(f"[GUI_CUDA] ⚡ CUDA context initialized successfully")
            print(f"[GUI_CUDA]   Inference latency: {latency:.1f}ms per face")
            print(f"[GUI_CUDA]   Worker threads will inherit this CUDA context")

        else:
            # Silent fallback to CPU
            status['error'] = f"CUDA provider failed, fell back to {actual_provider}"
            print(f"[GUI_CUDA] ⚠️ CUDA provider failed")
            print(f"[GUI_CUDA]   Fell back to: {actual_provider}")
            print(f"[GUI_CUDA]   Face recognition will use CPU (~200ms/face)")

    except Exception as e:
        status['error'] = str(e)
        print(f"[GUI_CUDA] ❌ ONNX CUDA initialization failed: {e}")
        import traceback
        traceback.print_exc()

    return status


# Auto-initialize when module is imported
# This ensures CUDA environment is set up immediately
_initialized = initialize_gui_cuda_environment()

if __name__ == "__main__":
    # Test the module directly
    print("\n" + "="*60)
    print("GUI CUDA Initialization Test")
    print("="*60)

    print("\n1. Environment initialized:", _initialized)
    print("2. LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "(not set)")[:200])

    print("\n3. Testing OpenCV CUDA...")
    cv2_status = verify_cuda_after_cv2_import()
    print("   Status:", cv2_status)

    print("\n4. Testing ONNX Runtime CUDA...")
    onnx_status = verify_onnx_cuda()
    print("   Status:", onnx_status)

    if cv2_status['cuda_initialized'] and onnx_status['onnx_cuda']:
        print("\n✅ All CUDA components working!")
    else:
        print("\n⚠️ Some CUDA components not available")