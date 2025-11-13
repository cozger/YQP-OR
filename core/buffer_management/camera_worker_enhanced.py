"""
Enhanced Camera Worker with Process Coordination
Manages detection and landmark sub-processes with optimized IPC

This implements the process separation architecture to resolve TensorRT engine
incompatibility between RetinaFace and MediaPipe models.
"""

import multiprocessing as mp
from multiprocessing import Process, Queue, Value, shared_memory
import queue
import threading
import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
import os
import platform
import atexit
import struct  # Required for pose frame buffer write index packing (line 1354)
from core.buffer_management.coordinator import BufferCoordinator
from core.buffer_management.sharedbuffer import CommandBuffer
from core.camera_sources.factory import CameraSourceFactory
from core.pose_processing.rtmpose3d_process import RTMPose3DProcess

# Pinned memory using ctypes (no cupy/pycuda needed!)
try:
    from core.buffer_management.pinned_memory_ctypes import get_allocator
    pinned_allocator = get_allocator()
    CUDA_AVAILABLE = pinned_allocator.available
    logger_init = logging.getLogger('EnhancedCameraWorker')
    if CUDA_AVAILABLE:
        logger_init.info("✅ Pinned memory available via ctypes (no cupy/pycuda needed)")
    else:
        logger_init.info("ℹ️  Pinned memory not available - using regular memory")
except Exception as e:
    logger_init = logging.getLogger('EnhancedCameraWorker')
    logger_init.info(f"ℹ️  Pinned memory not available: {e}")
    pinned_allocator = None
    CUDA_AVAILABLE = False

logger = logging.getLogger('EnhancedCameraWorker')


def get_nested_config(config: Dict, path: str, default=None):
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., 'buffer_management.timeouts.recovery_polling_interval_ms')
        default: Default value if path not found
        
    Returns:
        Configuration value or default
    """
    keys = path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


class EnhancedCameraWorker(Process):
    """
    Enhanced camera worker that spawns and manages detection/landmark sub-processes.
    Uses pinned memory and ring buffers for zero-copy IPC.
    
    Architecture:
    - Captures frames from camera
    - Manages shared memory buffers (frames, ROIs, results)
    - Spawns detection process (RetinaFace with isolated CUDA context)
    - Spawns landmark process (MediaPipe with isolated CUDA context)
    - Coordinates IPC between processes via ring buffers
    """
    
    def __init__(self, camera_index: int, gpu_device_id: int, config: Dict[str, Any],
                 control_queue: Queue, status_queue: Queue,
                 participant_update_queue: Queue = None,
                 frame_ready_semaphore = None,
                 pose_frame_ready_semaphore = None,
                 recovery_buffer_name: str = None, command_buffer_names: Dict[str, str] = None,
                 display_only: bool = False):
        super().__init__()
        self.camera_index = camera_index
        self.gpu_device_id = gpu_device_id
        self.config = config
        self.display_only = display_only

        # Control and status queues
        self.control_queue = control_queue
        self.status_queue = status_queue
        self.participant_update_queue = participant_update_queue

        # CRITICAL FIX: Don't store semaphores OR BufferCoordinator in __init__
        # They contain multiprocessing primitives that can't be pickled in spawn mode
        # Store configuration to create BufferCoordinator later in run() method
        self._coordinator_init_params = {
            'camera_index': camera_index,
            'frame_ready_semaphore': frame_ready_semaphore,
            'pose_frame_ready_semaphore': pose_frame_ready_semaphore,
            'max_cameras': config.get('system', {}).get('max_cameras', 10),
            'config': config
        }

        if frame_ready_semaphore is not None:
            logger.info(f"Camera {camera_index} will use frame_ready semaphore from parent process")
        else:
            logger.warning(f"Camera {camera_index} NO frame_ready semaphore provided - will create fallback")

        if pose_frame_ready_semaphore is not None:
            logger.info(f"Camera {camera_index} will use pose_frame_ready semaphore from parent process")
        else:
            logger.warning(f"Camera {camera_index} NO pose_frame_ready semaphore provided - will create fallback")

        # BufferCoordinator will be created in run() method to avoid pickling
        self.buffer_coordinator = None
        self._owns_coordinator = True

        self.command_buffer_names = command_buffer_names or {}
        
        # If command buffer names are provided, we should use existing buffers
        if self.command_buffer_names:
            logger.info(f"Camera {camera_index} using provided command buffer names: {self.command_buffer_names}")
            self._use_existing_buffers = True
        else:
            self._use_existing_buffers = False
        
        # Camera components
        self.cap = None  # Deprecated - use self.camera_source instead
        self.camera_source = None  # New abstraction layer (V4L2/ZMQ)
        self.actual_resolution = None
        
        # Sub-process management
        self.detection_process = None
        self.preprocessing_process = None  # NEW: MediaPipe preprocessing process
        self.landmark_process = None
        self.pose_process = None  # NEW: MediaPipe pose detection process
        
        # Command buffer system (replaces mp.Queue)
        self.command_buffers = None
        self.detection_command_buffer = None
        self.gui_status_buffer = None
        
        # Shared memory components
        self.frame_buffer_shm = None
        # MediaPipe native - no ROI buffer needed
        self.results_shm = None
        self.preprocessed_buffer_shm = None  # NEW: Preprocessed buffer
        self.preprocessed_buffer_name = None
        self.pose_buffer_shm = None  # NEW: Pose buffer
        self.pose_buffer_name = None
        self.pose_frame_buffer_shm = None  # NEW: Pose frame buffer (DISABLED - native resolution used)
        self.pose_frame_views = {}  # slot_index -> numpy view
        self.pose_frame_buffer_name = None

        # Calculate buffer sizes directly from config (BufferCoordinator created later in run())
        # This replicates BufferCoordinator's calculation logic to avoid accessing it before creation

        # Calculate max_faces (same logic as BufferCoordinator._calculate_max_faces)
        # Use centralized buffer_settings configuration
        participant_count = config.get('buffer_settings', {}).get('persons', {}).get('participant_count', 1)
        max_faces_config = config.get('buffer_settings', {}).get('faces', {}).get('max_faces_per_frame', 10)
        gpu_max_batch = config.get('advanced_detection', {}).get('gpu_settings', {}).get('max_batch_size', 8)
        self.max_faces = min(participant_count, max_faces_config, gpu_max_batch, 8)

        # Get ring_buffer_size (same logic as BufferCoordinator._get_ring_buffer_size)
        ring_size = config.get('buffer_settings', {}).get('ring_buffers', {}).get('frame_detection', 16)
        if ring_size & (ring_size - 1) != 0:  # Ensure power of 2
            ring_size = 1 << (ring_size - 1).bit_length()
        self.ring_buffer_size = ring_size

        # Get roi_buffer_size (same logic as BufferCoordinator._get_roi_buffer_size)
        roi_size = config.get('buffer_settings', {}).get('ring_buffers', {}).get('roi_processing', 8)
        if roi_size & (roi_size - 1) != 0:  # Ensure power of 2
            roi_size = 1 << (roi_size - 1).bit_length()
        self.roi_buffer_size = roi_size

        # Get gui_buffer_size
        self.gui_buffer_size = config.get('buffer_settings', {}).get('ring_buffers', {}).get('gui_display', 8)

        # CRITICAL FIX: Recalculate ring masks after getting correct buffer sizes
        self.preview_ring_size = self.gui_buffer_size
        self.ring_buffer_mask = self.ring_buffer_size - 1
        self.preview_ring_mask = self.preview_ring_size - 1

        logger.info(f"Camera {camera_index} calculated buffer sizes from config: "
                   f"ring={self.ring_buffer_size}, roi={self.roi_buffer_size}, "
                   f"max_faces={self.max_faces}, gui={self.gui_buffer_size}")
        logger.info(f"Camera {camera_index} ring masks: detection={self.ring_buffer_mask}, preview={self.preview_ring_mask}")
        
        # Atomic indices for lock-free operation
        self.frame_write_idx = Value('L', 0)
        self.frame_read_idx = Value('L', 0)
        self.roi_write_idx = Value('L', 0)
        self.roi_read_idx = Value('L', 0)
        
        # NEW: Preprocessing process indices (for 3-process pipeline)
        self.detection_read_idx = self.roi_read_idx  # Preprocessing reads from ROI buffer
        self.preprocessed_write_idx = Value('L', 0)  # Preprocessing writes to preprocessed buffer
        self.preprocessed_read_idx = Value('L', 0)   # Landmarks read from preprocessed buffer
        
        # Preview buffer write index (ring masks already set above)
        self.preview_write_idx = Value('L', 0)
        
        # Pinned memory for zero-copy transfers
        self.pinned_frame_buffers = []
        self.pinned_roi_buffers = []
        
        # Performance tracking
        # CRITICAL FIX: Start at 1 to avoid collision with uninitialized metadata (frame_id=0)
        # Combined with metadata initialization to -1, this ensures LIFO scan works correctly
        self.frame_counter = 1
        self.detection_interval = config.get('process_separation', {}).get('detection_interval', 3)  # Optimized for parallelism
        self.start_time = time.time()  # For relative timestamps
        self.frame_times = []
        self.write_times = []
        self.last_profile_report = time.time()
        self.enable_profiling = config.get('enable_profiling', False)

        # Frame arrival diagnostics (FPS debugging)
        self.last_frame_time = None
        self.inter_frame_intervals = []  # Track time between consecutive frames
        self.burst_count = 0  # Frames arriving <10ms apart
        self.last_diagnostics_log = time.time()
        self.diagnostics_frame_count = 0
        self.buffer_pressure_samples = []  # Track how many slots have unread frames

        # Frame synchronization (replaces semaphore IPC overhead)
        # Shared memory flag for zero-copy signaling (pure userspace, no kernel calls)
        self.frame_ready_flag = None  # Will be initialized to shared memory view after buffer creation
        
        # Pre-allocated array views for performance (initialized after shared memory)
        self.frame_views = []
        self.metadata_views = []
        self.preview_frame_view = None
        self.preview_metadata_view = None
        
        # Add recovery components
        self.recovery_buffer_name = recovery_buffer_name
        self.recovery_buffer = None
        self.recovery_processor_thread = None
        self.enable_recovery_processing = True
        
        # Detection response monitoring thread
        self.detection_response_thread = None
        
        # Command processor thread for incoming commands (from GUI/etc)
        self.command_processor_thread = None
        self.gui_command_buffer = None

        # Control flags
        self.running = Value('b', True)
        self.paused = Value('b', False)
        
        # Register cleanup
        atexit.register(self._cleanup)

    def _get_frame_ready_semaphore(self):
        """Get frame_ready_semaphore from BufferCoordinator (avoids pickling issues)."""
        if hasattr(self, 'buffer_coordinator') and self.buffer_coordinator:
            return self.buffer_coordinator.get_frame_ready_semaphore(self.camera_index)
        return None

    def _get_pose_frame_ready_semaphore(self):
        """Get pose_frame_ready_semaphore from BufferCoordinator (avoids pickling issues)."""
        if hasattr(self, 'buffer_coordinator') and self.buffer_coordinator:
            return self.buffer_coordinator.get_pose_frame_ready_semaphore(self.camera_index)
        return None

    def _cleanup_legacy_shared_memory(self, name: str, description: str):
        """Helper to cleanup legacy shared memory with proper error handling."""
        try:
            existing_shm = shared_memory.SharedMemory(name=name)
            existing_shm.close()
            existing_shm.unlink()
            logger.info(f"Cleaned up legacy {description}: {name}")
        except FileNotFoundError:
            pass  # No existing shared memory, which is fine
        except Exception as e:
            logger.debug(f"Could not cleanup legacy {description} {name}: {e}")
    

    def run(self):
        """Main process entry point - early command response design."""
        try:
            logger.info(f"Enhanced camera worker {self.camera_index} starting on GPU {self.gpu_device_id}")

            # CRITICAL: Create BufferCoordinator in run() method (not __init__) to avoid pickling
            # BufferCoordinator contains semaphores/Events that can't be pickled in spawn mode
            from core.buffer_management.coordinator import BufferCoordinator
            params = self._coordinator_init_params
            external_semaphores = {params['camera_index']: params['frame_ready_semaphore']} if params['frame_ready_semaphore'] else None
            external_pose_semaphores = {params['camera_index']: params['pose_frame_ready_semaphore']} if params['pose_frame_ready_semaphore'] else None

            self.buffer_coordinator = BufferCoordinator(
                camera_count=params['max_cameras'],
                config=params['config'],
                create_coordinator_info=False,  # Child process must not overwrite parent's info file
                external_semaphores=external_semaphores,
                external_pose_semaphores=external_pose_semaphores
            )
            logger.info(f"Camera {self.camera_index} created BufferCoordinator in run() method (spawn-safe)")

            # Create threading.Event for handshake synchronization (must be in child process, not pickled)
            self.handshake_confirmed = threading.Event()

            # PHASE 1: Initialize command buffers FIRST for immediate handshake response
            self._init_command_buffers()

            # Start command processor immediately for ping/pong handshake
            if self.gui_command_buffer:
                self.command_processor_thread = threading.Thread(
                    target=self._process_gui_commands,
                    name=f"CommandProcessor-{self.camera_index}",
                    daemon=True
                )
                self.command_processor_thread.start()
                logger.info(f"Command processor thread started for camera {self.camera_index} - ready for handshake")
            else:
                logger.warning(f"No GUI command buffer available, skipping command processor for camera {self.camera_index}")
            
            # Set booting state for ping responses
            self._initialization_state = "booting"
            
            # PHASE 2: Heavy initialization in background (camera, buffers, processes)
            logger.info(f"Starting heavy initialization for camera {self.camera_index}")
            
            # Initialize camera
            if not self._init_camera():
                self._send_status('error', {'message': 'Camera initialization failed'})
                return
            
            # Initialize shared memory and pinned buffers
            self._init_shared_memory()
            
            # Validate buffer integrity before proceeding
            validation_result = self.buffer_coordinator.validate_buffer_integrity(self.camera_index)
            if validation_result['status'] == 'error':
                logger.error(f"Buffer validation failed for camera {self.camera_index}: {validation_result['errors']}")
                raise RuntimeError(f"Buffer validation failed: {validation_result['errors']}")
            elif validation_result['status'] == 'warning':
                logger.warning(f"Buffer validation warnings for camera {self.camera_index}: {validation_result['warnings']}")
                
            if self.config.get('process_separation', {}).get('enable_pinned_memory', True):
                self._init_pinned_memory()

            # Check if this camera should run detection/pose processing
            if not self.display_only:
                # MMPose 3D PIPELINE: Person detection → 3D pose estimation
                logger.info(f"[CAMERA {self.camera_index}] ===== MMPose 3D MODE =====")
                logger.info(f"[CAMERA {self.camera_index}] Pipeline: Camera → MMPose Person Detection → MMPose 3D Pose Estimation")

                # Start RTMPose3D process for 3D pose estimation
                self._start_pose_process()

                # Send model ready status to GUI
                self._send_status('model_ready', {'message': 'MMPose pipeline initialized'})
            else:
                # DISPLAY-ONLY MODE: Skip all detection/pose processing
                logger.info(f"[CAMERA {self.camera_index}] ===== DISPLAY-ONLY MODE =====")
                logger.info(f"[CAMERA {self.camera_index}] Pipeline: Camera → GUI Display (no processing)")
                # Send realtime_ready status to indicate we're ready (no detection needed)
                self._send_status('realtime_ready', {
                    'message': 'Display-only mode (no detection)',
                    'display_only': True
                })

            # PHASE 3: Initialization complete - update state and send ready status
            self._initialization_state = "ready"
            logger.info(f"Heavy initialization complete for camera {self.camera_index}")

            # Send ready status with shared memory names
            logger.info(f"Sending ready status for camera {self.camera_index} with buffer names:")
            logger.info(f"  frame: {self.frame_buffer_name} (native resolution)")
            logger.info(f"  results: {self.results_buffer_name}")
            logger.info(f"  gui: {self.gui_buffer_name}")

            self._send_status('ready', {
                'camera_index': self.camera_index,
                'resolution': self.actual_resolution,  # Native resolution (e.g., 1920x1080)
                'shared_memory': {
                    'frame': self.frame_buffer_name,  # Frame buffer: native resolution frames for display & pose
                    'results': self.results_buffer_name,  # Results buffer: pose keypoints
                    'gui': self.gui_buffer_name,  # GUI buffer for dedicated display rendering
                    'pose': self.pose_buffer_name,  # Pose buffer for body pose detection
                }
            })

            # Optional: Wait for handshake acknowledgment from manager
            handshake_ack_timeout_s = 10.0  # Default for WSL2 compatibility
            if self.config:
                handshake_ack_timeout_s = float(get_nested_config(self.config, 'camera_timeouts.handshake_ack_timeout_s', handshake_ack_timeout_s))

            logger.info(f"[DEBUG] Worker {self.camera_index} waiting for handshake acknowledgment (timeout: {handshake_ack_timeout_s}s)...")

            # Wait for handshake acknowledgment via event (no race condition)
            handshake_received = self.handshake_confirmed.wait(timeout=handshake_ack_timeout_s)

            if handshake_received:
                logger.info(f"✅ Handshake confirmed for camera {self.camera_index}")
            else:
                logger.warning(f"⚠️ No handshake acknowledgment received for camera {self.camera_index} - continuing anyway")
            
            # Main capture loop
            logger.info(f"✅ Camera {self.camera_index} handshake complete, starting capture loop...")
            self._capture_loop()
            logger.info(f"Camera {self.camera_index} capture loop has ended")
            
        except Exception as e:
            logger.error(f"Camera worker {self.camera_index} fatal error: {e}")
            self._send_status('error', {'message': str(e)})
        finally:
            self._cleanup()
    
    def _cleanup_camera_processes(self, device_path: str) -> int:
        """
        Kill any processes currently accessing the camera device.
        Critical for WSL2 where zombie processes can block camera access.

        Args:
            device_path: Path to video device (e.g., "/dev/video0")

        Returns:
            Number of processes killed
        """
        killed_count = 0

        try:
            # Use fuser to find processes using the device
            result = subprocess.run(
                ['fuser', device_path],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split()
                logger.info(f"Cleaning up {len(pids)} process(es) blocking {device_path}: {pids}")

                for pid in pids:
                    try:
                        subprocess.run(['kill', '-9', pid], timeout=1, check=True)
                        logger.info(f"  Killed PID {pid}")
                        killed_count += 1
                    except subprocess.CalledProcessError:
                        logger.warning(f"  Could not kill PID {pid}")

                # Wait for cleanup to complete
                time.sleep(0.5)

        except FileNotFoundError:
            logger.debug("fuser not available, skipping cleanup")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

        return killed_count

    def _detect_camera_type(self, device_path: str) -> str:
        """
        Auto-detect if camera is IR or regular based on capabilities.

        Args:
            device_path: Path to video device

        Returns:
            "regular", "ir", or "unknown"
        """
        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', device_path, '--list-formats-ext'],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                output = result.stdout
                # IR cameras typically support GREY format
                if 'GREY' in output and '640x360' in output:
                    logger.info(f"{device_path} detected as IR camera")
                    return "ir"
                else:
                    logger.info(f"{device_path} detected as regular camera")
                    return "regular"
        except Exception as e:
            logger.warning(f"Camera type detection failed: {e}, assuming regular")

        return "regular"

    def _apply_manual_exposure(self, device_path: str, camera_settings: dict, target_fps: int):
        """
        Apply manual exposure control using v4l2-ctl to prevent auto-exposure from limiting FPS.

        Args:
            device_path: Path to video device (e.g., /dev/video0)
            camera_settings: Camera configuration dictionary
            target_fps: Target FPS for shutter speed calculation
        """
        import subprocess

        logger.info(f"Applying manual exposure control to {device_path}...")

        # Get exposure settings from config
        target_shutter_fps = camera_settings.get('target_shutter_fps', target_fps)
        gain = camera_settings.get('gain', 0)

        try:
            # Step 1: Set exposure mode to manual (1 = manual on most UVC cameras)
            result = subprocess.run(
                ['v4l2-ctl', '-d', device_path, '--set-ctrl=exposure_auto=1'],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                logger.info(f"  ✅ Set exposure_auto=1 (manual mode)")

                # Step 2: Calculate and set exposure_absolute
                # exposure_absolute is in 100µs (0.0001s) units
                # For 60 fps: 1/60s = 0.0166667s = 166.67 units → 167
                exposure_absolute = int(10000 / target_shutter_fps)
                logger.info(f"  Setting exposure_absolute={exposure_absolute} (1/{target_shutter_fps}s shutter)")

                result = subprocess.run(
                    ['v4l2-ctl', '-d', device_path, f'--set-ctrl=exposure_absolute={exposure_absolute}'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )

                if result.returncode == 0:
                    logger.info(f"  ✅ Exposure locked to 1/{target_shutter_fps}s (prevents auto-exposure FPS limiting)")
                else:
                    logger.warning(f"  ⚠️  Failed to set exposure_absolute: {result.stderr}")

                # Step 3: Set gain to prevent noise amplification
                result = subprocess.run(
                    ['v4l2-ctl', '-d', device_path, f'--set-ctrl=gain={gain}'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )

                if result.returncode == 0:
                    logger.info(f"  ✅ Set gain={gain}")
                else:
                    logger.debug(f"  Could not set gain (may not be supported): {result.stderr}")

                # Step 4: Verify settings
                result = subprocess.run(
                    ['v4l2-ctl', '-d', device_path, '--get-ctrl=exposure_auto'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )

                if result.returncode == 0:
                    exposure_mode = result.stdout.strip()
                    logger.info(f"  Verified: {exposure_mode}")

            else:
                logger.warning(f"  ⚠️  Camera does not support manual exposure control")
                logger.warning(f"      Auto-exposure may limit FPS in low light conditions")
                logger.warning(f"      Error: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.warning(f"  ⚠️  Timeout setting exposure controls (v4l2-ctl hung)")
        except FileNotFoundError:
            logger.warning(f"  ⚠️  v4l2-ctl not found - install v4l-utils for exposure control")
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to apply manual exposure: {e}")

    def _init_camera(self) -> bool:
        """
        Initialize camera using abstraction layer (V4L2 or ZMQ).
        Uses CameraSourceFactory to create appropriate source based on config.

        On success sets: self.camera_source, self.backend_name, self.actual_resolution
        """
        try:
            # Create camera source using factory
            logger.info(f"[DIAGNOSTIC] Worker {self.camera_index}: Creating camera source via factory")
            logger.info(f"Creating camera source for camera {self.camera_index}")
            self.camera_source = CameraSourceFactory.create(
                config=self.config,
                camera_index=self.camera_index
            )
            logger.info(f"[DIAGNOSTIC] Worker {self.camera_index}: Camera source created (type: {type(self.camera_source).__name__})")

            # Open camera source
            logger.info(f"[DIAGNOSTIC] Worker {self.camera_index}: Opening camera source...")
            if not self.camera_source.open():
                logger.error(f"[DIAGNOSTIC] Worker {self.camera_index}: ❌ Camera source open() returned False")
                logger.error(f"Failed to open camera source for camera {self.camera_index}")
                return False
            logger.info(f"[DIAGNOSTIC] Worker {self.camera_index}: ✓ Camera source opened successfully")

            # Get actual resolution and backend info
            width, height = self.camera_source.get_resolution()
            logger.info(f"[DIAGNOSTIC] Worker {self.camera_index}: Camera native resolution: {width}x{height}")

            # Use native camera resolution for buffer (no downsampling)
            # RTMDet benefits from higher resolution for better person detection
            # RTMPose3D will automatically resize person ROIs to 384x288 via TopdownAffine
            self.actual_resolution = [width, height]  # Buffer resolution (native)
            self.backend_name = self.camera_source.get_backend_name()
            logger.info(f"[DIAGNOSTIC] Worker {self.camera_index}: Buffer resolution: {width}x{height} (native), Backend: {self.backend_name}")

            # Warmup - discard initial frames
            logger.info(f"[DIAGNOSTIC] Worker {self.camera_index}: Starting warmup (10 frames)...")
            logger.info(f"Warming up camera {self.camera_index} (10 frames)...")
            self.camera_source.warmup(num_frames=10)
            logger.info(f"[DIAGNOSTIC] Worker {self.camera_index}: ✓ Warmup complete")

            logger.info(f"Camera {self.camera_index} initialized successfully using {self.backend_name} backend")
            logger.info(f"[DIAGNOSTIC] Worker {self.camera_index}: ✅ Initialization complete")
            return True

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False


    def _init_shared_memory(self):
        """Initialize shared memory buffers using centralized BufferCoordinator. MediaPipe native - no ROI buffer."""
        try:
            # Create all buffers through coordinator for consistency
            buffer_names = self.buffer_coordinator.create_camera_buffers(
                self.camera_index,
                self.actual_resolution
            )

            # Connect to pre-created buffers instead of creating new ones
            self.frame_buffer_shm = shared_memory.SharedMemory(name=buffer_names['frame'])
            self.frame_buffer_name = buffer_names['frame']

            self.results_shm = shared_memory.SharedMemory(name=buffer_names['results'])
            self.results_buffer_name = buffer_names['results']

            # GUI buffer for display data
            self.gui_buffer_shm = shared_memory.SharedMemory(name=buffer_names['gui'])
            self.gui_buffer_name = buffer_names['gui']

            # Pose buffer for body pose detection
            self.pose_buffer_shm = shared_memory.SharedMemory(name=buffer_names['pose'])
            self.pose_buffer_name = buffer_names['pose']

            # RESOLUTION SYNC FIX: Verify buffer header resolution matches actual camera resolution
            # Read resolution from buffer header (ground truth)
            header_resolution = np.frombuffer(self.frame_buffer_shm.buf, dtype=np.uint32, count=2, offset=8)
            header_w, header_h = int(header_resolution[0]), int(header_resolution[1])

            if (header_w, header_h) != tuple(self.actual_resolution):
                logger.warning(f"Camera {self.camera_index} resolution mismatch detected!")
                logger.warning(f"  Buffer header: {header_w}x{header_h}")
                logger.warning(f"  Actual camera: {self.actual_resolution[0]}x{self.actual_resolution[1]}")
                logger.warning(f"  Updating buffer header to match actual camera resolution")

                # Update buffer header with actual resolution
                np.frombuffer(self.frame_buffer_shm.buf, dtype=np.uint32, count=2, offset=8)[:] = self.actual_resolution

                # Also update BufferCoordinator's tracking
                self.buffer_coordinator.camera_resolutions[self.camera_index] = tuple(self.actual_resolution)
                logger.info(f"Camera {self.camera_index} resolution synchronized: {self.actual_resolution[0]}x{self.actual_resolution[1]}")
            else:
                logger.info(f"Camera {self.camera_index} resolution verified in buffer header: {header_w}x{header_h}")

            # Pre-allocate array views for performance optimization
            self._init_array_views()

            # Initialize frame_ready flag (shared memory flag for zero-copy signaling)
            # Location: offset 16 in frame buffer (after write_index and resolution)
            self.frame_ready_flag = np.frombuffer(
                self.frame_buffer_shm.buf,
                dtype=np.int32,
                count=1,
                offset=16
            )
            self.frame_ready_flag[0] = 0  # Initialize to 0 (no frame ready)
            logger.info(f"Camera {self.camera_index} initialized frame_ready flag at offset 16")

            # CLEAN ARCHITECTURE: Removed pose_frame_buffer setup
            # Camera now writes native resolution frames directly to main frame_buffer
            # Both Display and RTMPose3D read from same buffer (single path)
            # This eliminates dual-writer race conditions and simplifies architecture

            logger.info(f"Camera {self.camera_index} connected to coordinator-managed buffers:")
            logger.info(f"  Frame buffer: {buffer_names['frame']}")
            logger.info(f"  Results buffer: {buffer_names['results']}")
            logger.info(f"  GUI buffer: {buffer_names['gui']}")

        except Exception as e:
            logger.error(f"Shared memory initialization failed: {e}")
            raise
            
    def _init_array_views(self):
        """Pre-allocate array views using BufferCoordinator layout for consistency."""
        try:
            # Validate buffer state first
            if not self.frame_buffer_shm or not self.frame_buffer_shm.buf:
                raise ValueError("Frame buffer not initialized")
                
            # Get authoritative buffer layout from coordinator
            layout = self.buffer_coordinator.get_buffer_layout(self.camera_index, self.actual_resolution)
            
            # Validate buffer size matches layout expectations
            if len(self.frame_buffer_shm.buf) < layout['total_size']:
                raise ValueError(f"Buffer too small: {len(self.frame_buffer_shm.buf)} < {layout['total_size']}")
                
            # Initialize detection ring buffer views using coordinator offsets
            self.frame_views = []
            self.metadata_views = []
            
            for i in range(self.ring_buffer_size):
                frame_offset = layout['detection_frame_offsets'][i]
                metadata_offset = layout['detection_metadata_offsets'][i]
                
                # Validate offsets are within bounds
                if frame_offset + layout['frame_size'] > len(self.frame_buffer_shm.buf):
                    raise ValueError(f"Frame offset {frame_offset} out of bounds")
                if metadata_offset + 64 > len(self.frame_buffer_shm.buf):
                    raise ValueError(f"Metadata offset {metadata_offset} out of bounds")
                
                # Create frame view with coordinator-provided offset
                # Shape: (height, width, channels) - validated in _write_frame_optimized
                frame_view = np.frombuffer(
                    self.frame_buffer_shm.buf,
                    dtype=np.uint8,
                    count=layout['frame_size'],
                    offset=frame_offset
                ).reshape(self.actual_resolution[1], self.actual_resolution[0], 3)
                self.frame_views.append(frame_view)
                
                # Create metadata view with coordinator-provided offset
                # Use int64 for consistency across all components
                metadata_view = np.frombuffer(
                    self.frame_buffer_shm.buf,
                    dtype=np.int64,  # Use int64 consistently
                    count=8,  # 8 int64 values (64 bytes total)
                    offset=metadata_offset
                )
                self.metadata_views.append(metadata_view)

            # CRITICAL FIX: Initialize all metadata slots to -1 (invalid frame_id)
            # This ensures LIFO scan correctly identifies uninitialized slots
            # Without this, zero-initialized memory causes display to freeze on frame 0
            for metadata_view in self.metadata_views:
                metadata_view[0] = -1  # frame_id = -1 (uninitialized)
                metadata_view[3] = 0   # ready_flag = 0 (not ready)
            logger.info(f"  Initialized {len(self.metadata_views)} metadata slots to frame_id=-1")

            # Architecture B: GUI views removed - handled by separate display buffers
            # GUI buffer name kept for propagation only, no writes
            self.gui_frame_views = []  # Empty list for compatibility
            
            logger.info(f"Successfully pre-allocated {len(self.frame_views)} detection views using coordinator layout")
            
            # Log critical buffer validation info
            logger.info(f"Array view validation for camera {self.camera_index}:")
            logger.info(f"  Detection views: {len(self.frame_views)} (expected: {self.ring_buffer_size})")
            logger.info(f"  Metadata views: {len(self.metadata_views)} (expected: {self.ring_buffer_size})")
            logger.info(f"  GUI sections: Removed for Architecture B (separate display buffers)")
            
            # Test first view to ensure it's actually usable
            if self.frame_views and self.metadata_views:
                test_frame = np.zeros((self.actual_resolution[1], self.actual_resolution[0], 3), dtype=np.uint8)
                self.frame_views[0][:] = test_frame  # This should not raise an exception
                self.metadata_views[0][0] = 12345  # Test metadata write
                logger.info(f"  View write test: PASSED")
                # IMPORTANT: Clear test frame immediately to prevent GUI from displaying it
                self.metadata_views[0][0] = -1  # Reset frame_id to indicate no frame
            else:
                logger.error(f"  View write test: FAILED - no views available")
            
            # Create recovery buffer if participant manager is available
            if self.participant_update_queue and self.enable_recovery_processing:
                try:
                    self.recovery_buffer_name = self.buffer_coordinator.create_recovery_buffer(
                        self.camera_index
                    )
                    
                    # Access the recovery buffer object that was created by coordinator
                    if hasattr(self.buffer_coordinator, 'recovery_buffers') and \
                       self.recovery_buffer_name in self.buffer_coordinator.recovery_buffers:
                        self.recovery_buffer = self.buffer_coordinator.recovery_buffers[self.recovery_buffer_name]
                        logger.info(f"Connected to recovery buffer for camera {self.camera_index}: "
                                   f"{self.recovery_buffer_name}")
                    else:
                        logger.warning(f"Could not access recovery buffer {self.recovery_buffer_name} "
                                     f"from coordinator (has recovery_buffers: {hasattr(self.buffer_coordinator, 'recovery_buffers')})")
                        self.enable_recovery_processing = False
                    
                except Exception as e:
                    logger.warning(f"Failed to create recovery buffer: {e}")
                    self.enable_recovery_processing = False
            
        except Exception as e:
            logger.error(f"CRITICAL: Array view initialization failed: {e}")
            logger.error(f"Buffer size: {len(self.frame_buffer_shm.buf) if self.frame_buffer_shm else 'None'}")
            logger.error(f"Camera resolution: {self.actual_resolution}")
            
            # Try to get buffer layout for debugging
            try:
                layout = self.buffer_coordinator.get_buffer_layout(self.camera_index, self.actual_resolution)
                logger.error(f"Expected layout total size: {layout['total_size']}")
                logger.error(f"Buffer layout details: {layout}")
            except Exception as layout_error:
                logger.error(f"Could not get buffer layout: {layout_error}")
                
            # DON'T continue with empty lists - this causes NoneType errors later
            raise  # Force worker to fail cleanly rather than start in broken state
    
    def _init_pinned_memory(self):
        """Pre-allocate pinned memory for GPU transfers using ctypes."""
        if not CUDA_AVAILABLE or not pinned_allocator:
            logger.info(f"Camera {self.camera_index}: Pinned memory not available, using regular memory")
            self.pinned_frame_buffers = []
            return

        try:
            # Allocate 3 pinned buffers for triple buffering
            frame_size = self.actual_resolution[0] * self.actual_resolution[1] * 3

            for i in range(3):
                pinned_buf = pinned_allocator.allocate(frame_size, dtype=np.uint8)
                self.pinned_frame_buffers.append(pinned_buf)

            logger.info(f"Camera {self.camera_index} pinned memory allocated: "
                       f"{len(self.pinned_frame_buffers)} buffers × {frame_size/1024/1024:.1f} MB")
            logger.info(f"  Expected PCIe speedup: ~4× faster (2-5ms → 0.5-1ms per transfer)")

        except Exception as e:
            logger.warning(f"Pinned memory allocation failed: {e}")
            logger.warning("  Falling back to regular memory (slower PCIe transfers)")
            self.pinned_frame_buffers = []
    
    def _init_command_buffers(self):
        """Initialize command buffer system to replace mp.Queue."""
        try:
            # Use provided buffer names if available, otherwise create new ones
            if self._use_existing_buffers and self.command_buffer_names:
                # Use the buffer names provided by the parent process
                self.command_buffers = self.command_buffer_names
                logger.info(f"Camera {self.camera_index} using existing command buffers: {self.command_buffers}")
            else:
                # Create command buffers through BufferCoordinator
                self.command_buffers = self.buffer_coordinator.create_camera_command_buffers(self.camera_index)
                logger.info(f"Camera {self.camera_index} created new command buffers")
            
            # Connect to the camera-to-detection command buffer
            if 'camera_to_detection' in self.command_buffers:
                self.detection_command_buffer = CommandBuffer.connect(
                    self.command_buffers['camera_to_detection']
                )
            else:
                self.detection_command_buffer = None
                logger.warning(f"No camera_to_detection buffer for camera {self.camera_index}")
            
            # Connect to the GUI-to-camera command buffer for incoming commands
            if 'gui_to_camera' in self.command_buffers:
                self.gui_command_buffer = CommandBuffer.connect(
                    self.command_buffers['gui_to_camera']
                )
            else:
                self.gui_command_buffer = None
                logger.warning(f"No gui_to_camera buffer for camera {self.camera_index}")
            
            # Connect to the camera-to-GUI command buffer for outgoing status messages
            if 'camera_to_gui' in self.command_buffers:
                self.gui_status_buffer = CommandBuffer.connect(
                    self.command_buffers['camera_to_gui']
                )
            else:
                self.gui_status_buffer = None
                logger.warning(f"No camera_to_gui buffer for camera {self.camera_index}")
            
            logger.info(f"Camera {self.camera_index} command buffers initialized: {list(self.command_buffers.keys())}")
            
            # Critical validation - worker cannot function without gui_status_buffer
            if not self.gui_status_buffer:
                error_msg = f"CRITICAL: No camera_to_gui CommandBuffer connected for camera {self.camera_index}"
                logger.error(error_msg)
                logger.error(f"Received buffer names: {self.command_buffers}")
                logger.error(f"camera_to_gui name: {self.command_buffers.get('camera_to_gui', 'MISSING')}")
                raise RuntimeError(error_msg)
            
            # Log successful connection details
            logger.info(f"[SUCCESS] Camera {self.camera_index} connected to gui_status_buffer: {self.gui_status_buffer.name}")
            
            # Debug logging for all connected buffers
            logger.info(f"[DEBUG] Worker {self.camera_index} connected buffers:")
            logger.info(f"  gui_status_buffer: {self.gui_status_buffer} (name: {self.gui_status_buffer.name if self.gui_status_buffer else 'None'})")
            logger.info(f"  gui_command_buffer: {self.gui_command_buffer} (name: {self.gui_command_buffer.name if self.gui_command_buffer else 'None'})")
            logger.info(f"  detection_command_buffer: {self.detection_command_buffer} (name: {self.detection_command_buffer.name if self.detection_command_buffer else 'None'})")
            
        except Exception as e:
            logger.error(f"Command buffer initialization failed: {e}")
            # Fallback to None - methods will handle gracefully
            self.command_buffers = None
            self.detection_command_buffer = None
            self.gui_status_buffer = None
            raise
    
    def _start_detection_process(self):
        """Start the detection sub-process."""
        try:
            # Set PYTHONPATH for subprocess to find modules
            import os
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            
            # Add to current process path
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            # Set PYTHONPATH environment variable for subprocess
            current_pythonpath = os.environ.get('PYTHONPATH', '')
            if str(project_root) not in current_pythonpath:
                os.environ['PYTHONPATH'] = f"{project_root}{os.pathsep}{current_pythonpath}"

            # Import detection process (avoid circular imports)
            # REMOVED: Face detection/tracking module not implemented
            # from core.face_detection_tracking.detection_process_gpu import DetectionProcess
            
            # Use path as-is (no conversion needed in WSL2)
            retinaface_path = self.config['advanced_detection']['retinaface_trt_path']

            # CRITICAL FIX: Pass FULL config to ensure BufferCoordinator gets all required keys
            # Detection process needs process_separation.enable_preprocessing_process and max_faces calculation inputs
            detection_config = self.config.copy()
            detection_config.update({
                'gpu_device_id': self.gpu_device_id,
                'model_path': retinaface_path,
                'confidence_threshold': self.config['advanced_detection'].get('detection_confidence', 0.7),
                'detection_size': (608, 640),  # Standard RetinaFace input size
                'enable_fp16': self.config.get('advanced_detection', {}).get('gpu_settings', {}).get('enable_fp16', True),
                'detection_interval': self.detection_interval,
                'camera_index': self.camera_index
            })
            
            self.detection_process = DetectionProcess(
                camera_index=self.camera_index,
                frame_buffer_name=self.frame_buffer_shm.name,
                roi_buffer_name=self.roi_buffer_shm.name,
                frame_read_idx=self.frame_read_idx,
                roi_write_idx=self.roi_write_idx,
                config=detection_config,
                actual_resolution=self.actual_resolution,
                recovery_buffer_name=self.recovery_buffer_name,
                command_buffer_name=self.command_buffers['camera_to_detection'] if self.command_buffers else None
            )
            self.detection_process.start()
            logger.info(f"Detection process started for camera {self.camera_index} "
                       f"with recovery buffer: {self.recovery_buffer_name}")
            
        except Exception as e:
            logger.error(f"Failed to start detection process: {e}")
            raise

    
    def _start_preprocessing_process(self):
        """Start the preprocessing sub-process (MediaPipe preprocessing)."""
        try:
            logger.info(f"[CAMERA WORKER] _start_preprocessing_process() CALLED for camera {self.camera_index}")

            # Ensure PYTHONPATH is set (should already be set from detection process)
            import os
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent

            # Add to current process path
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            # Set PYTHONPATH environment variable for subprocess
            current_pythonpath = os.environ.get('PYTHONPATH', '')
            if str(project_root) not in current_pythonpath:
                os.environ['PYTHONPATH'] = f"{project_root}{os.pathsep}{current_pythonpath}"

            logger.info(f"[CAMERA WORKER] Importing PreprocessingProcess class...")
            # Import preprocessing process (avoid circular imports)
            # REMOVED: Face detection/tracking module not implemented
            # from core.face_detection_tracking.preprocessing_process_gpu import PreprocessingProcess
            logger.info(f"[CAMERA WORKER] PreprocessingProcess class import skipped (module not implemented)")

            # CRITICAL FIX: Pass FULL config to ensure BufferCoordinator gets all required keys
            preprocessing_config = self.config.copy()
            preprocessing_config.update({
                'gpu_device_id': self.gpu_device_id,
                'max_faces_per_frame': self.max_faces,
                'camera_index': self.camera_index
            })

            logger.info(f"[CAMERA WORKER] Creating PreprocessingProcess instance...")
            logger.info(f"[CAMERA WORKER]   detection_buffer={self.roi_buffer_shm.name}")
            logger.info(f"[CAMERA WORKER]   frame_buffer={self.frame_buffer_shm.name}")
            logger.info(f"[CAMERA WORKER]   preprocessed_buffer={self.preprocessed_buffer_name}")
            logger.info(f"[CAMERA WORKER]   resolution={self.actual_resolution}")

            self.preprocessing_process = PreprocessingProcess(
                camera_index=self.camera_index,
                detection_buffer_name=self.roi_buffer_shm.name,  # Reads ROI buffer (detection output)
                frame_buffer_name=self.frame_buffer_shm.name,    # Reads original frames
                preprocessed_buffer_name=self.preprocessed_buffer_name,  # Writes preprocessed tensors
                detection_read_idx=self.detection_read_idx,      # Read index for ROI buffer
                preprocessed_write_idx=self.preprocessed_write_idx,  # Write index for preprocessed buffer
                config=preprocessing_config,
                actual_resolution=self.actual_resolution
            )

            logger.info(f"[CAMERA WORKER] PreprocessingProcess instance created, calling start()...")
            self.preprocessing_process.start()
            logger.info(f"[CAMERA WORKER] Preprocessing process START CALLED for camera {self.camera_index}")
            logger.info(f"Preprocessing process started for camera {self.camera_index}")

        except Exception as e:
            logger.error(f"[CAMERA WORKER] FATAL: Failed to start preprocessing process: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _start_landmark_process(self):
        """Start the landmark sub-process."""
        try:
            # Ensure PYTHONPATH is set (should already be set from detection process)
            import os
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent

            # Add to current process path
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            # Set PYTHONPATH environment variable for subprocess
            current_pythonpath = os.environ.get('PYTHONPATH', '')
            if str(project_root) not in current_pythonpath:
                os.environ['PYTHONPATH'] = f"{project_root}{os.pathsep}{current_pythonpath}"

            # Import landmark process (avoid circular imports)
            # REMOVED: Face detection/tracking module not implemented
            # from core.face_detection_tracking.landmark_process_gpu import LandmarkProcess

            # Use path as-is (no conversion needed in WSL2)
            landmark_path = self.config['advanced_detection']['landmark_trt_path']

            # CRITICAL FIX: Pass FULL config to ensure BufferCoordinator gets all required keys
            landmark_config = self.config.copy()
            landmark_config.update({
                'gpu_device_id': self.gpu_device_id,
                'model_path': landmark_path,
                'enable_fp16': self.config.get('advanced_detection', {}).get('gpu_settings', {}).get('enable_fp16', True),
                'max_batch_size': self.config.get('advanced_detection', {}).get('gpu_settings', {}).get('max_batch_size', 8),
                'enable_temporal_smoothing': self.config.get('process_separation', {}).get('enable_temporal_smoothing', True),
                'smoothing_factor': self.config.get('process_separation', {}).get('smoothing_factor', 0.7),
                'camera_index': self.camera_index
            })

            # Check if preprocessing is enabled to determine buffer connections
            enable_preprocessing = self.config.get('process_separation', {}).get('enable_preprocessing_process', False)

            if enable_preprocessing:
                # 3-process pipeline - REQUIRED when preprocessing is enabled
                if not self.preprocessed_buffer_name:
                    raise RuntimeError("Cannot start landmark process: preprocessing enabled but no preprocessed buffer available")

                self.landmark_process = LandmarkProcess(
                    camera_index=self.camera_index,
                    preprocessed_buffer_name=self.preprocessed_buffer_name,  # Read from preprocessing output
                    results_buffer_name=self.results_shm.name,
                    preprocessed_read_idx=self.preprocessed_read_idx,  # Use preprocessed read index
                    config=landmark_config
                )
                logger.info(f"Landmark process configured for 3-process pipeline (preprocessed input required)")
            else:
                # 2-process pipeline - only when preprocessing is explicitly disabled
                self.landmark_process = LandmarkProcess(
                    camera_index=self.camera_index,
                    preprocessed_buffer_name=self.roi_buffer_shm.name,  # Use ROI buffer via preprocessed parameter
                    results_buffer_name=self.results_shm.name,
                    preprocessed_read_idx=self.roi_read_idx,  # Use ROI index via preprocessed parameter
                    config=landmark_config
                )
                logger.info(f"Landmark process configured for 2-process pipeline (preprocessing disabled)")

            self.landmark_process.start()
            logger.info(f"Landmark process started for camera {self.camera_index}")

        except Exception as e:
            logger.error(f"Failed to start landmark process: {e}")
            raise

    def _start_mediapipe_process(self):
        """Start the MediaPipe FaceLandmarker process (replaces 3-process TensorRT pipeline)."""
        try:
            # Ensure PYTHONPATH is set
            import os
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent

            # Add to current process path
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            # Set PYTHONPATH environment variable for subprocess
            current_pythonpath = os.environ.get('PYTHONPATH', '')
            if str(project_root) not in current_pythonpath:
                os.environ['PYTHONPATH'] = f"{project_root}{os.pathsep}{current_pythonpath}"

            # DEPRECATED: MediaPipe is no longer used (replaced by RTMPose3D)
            logger.warning(f"[CAMERA WORKER] MediaPipe process disabled - feature deprecated")
            return None

            # logger.info(f"[CAMERA WORKER] Importing MediaPipeProcess class...")
            # # Import MediaPipe process
            # from core.face_processing.mediapipe_process import MediaPipeProcess
            # logger.info(f"[CAMERA WORKER] MediaPipeProcess class imported successfully")

            # Get model path from config
            model_path = self.config.get('paths', {}).get(
                'model_path',
                '/mnt/d/Projects/youquantipy_mediapipe/models/face_landmarker.task'
            )

            # Use path as-is (no conversion needed in WSL2)

            # Create event for model ready signaling (simplified protocol)
            import multiprocessing
            model_ready_event = multiprocessing.Event()
            logger.info(f"[CAMERA WORKER] Created model_ready_event for MediaPipe initialization signaling")

            logger.info(f"[CAMERA WORKER] Creating MediaPipeProcess instance...")
            logger.info(f"[CAMERA WORKER]   frame_buffer={self.frame_buffer_shm.name}")
            logger.info(f"[CAMERA WORKER]   results_buffer={self.results_shm.name}")
            logger.info(f"[CAMERA WORKER]   model_path={model_path}")
            logger.info(f"[CAMERA WORKER]   camera_index={self.camera_index}")
            logger.info(f"[CAMERA WORKER]   passing frame_ready_semaphore to MediaPipeProcess")
            logger.info(f"[CAMERA WORKER]   passing model_ready_event to MediaPipeProcess")

            self.mediapipe_process = MediaPipeProcess(
                camera_index=self.camera_index,
                frame_buffer_name=self.frame_buffer_shm.name,
                results_buffer_name=self.results_shm.name,
                config=self.config,
                model_path=model_path,
                frame_ready_semaphore=self._get_frame_ready_semaphore(),  # Get from coordinator
                model_ready_event=model_ready_event,  # Pass event for initialization signaling
                gui_status_buffer_name=self.command_buffers.get('camera_to_gui') if self.command_buffers else None
            )

            logger.info(f"[CAMERA WORKER] MediaPipeProcess instance created, calling start()...")
            self.mediapipe_process.start()
            logger.info(f"MediaPipe process started for camera {self.camera_index} (replaces detection+preprocessing+landmark)")
            logger.info(f"[Camera {self.camera_index}] Face warmup running in background (will wait for completion later)")

            # Return event for parallel waiting (don't block here)
            return model_ready_event

        except Exception as e:
            logger.error(f"[CAMERA WORKER] FATAL: Failed to start MediaPipe process: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _start_pose_process(self):
        """
        Start RTMPose3D process for 3D whole-body pose estimation.

        Pipeline: Frame Buffer → RTMDet Detection → RTMW3D Pose → Pose Buffer
        """
        # Get MMPose configuration
        mmpose_config = self.config.get('mmpose_3d_pipeline', {})
        if not mmpose_config.get('enabled', False):
            logger.info(f"[CAMERA {self.camera_index}] MMPose pipeline disabled in config")
            return None

        logger.info(f"[CAMERA {self.camera_index}] Starting RTMPose3D process...")

        # Get frame buffer layout (source for pose process)
        frame_layout_obj = self.buffer_coordinator.get_layout('frame', camera_index=self.camera_index)
        frame_layout = frame_layout_obj.to_dict()

        # Get pose buffer layout (destination for pose results)
        pose_layout_obj = self.buffer_coordinator.get_layout('pose', camera_index=self.camera_index)
        pose_layout = pose_layout_obj.to_dict()

        # Create events for process coordination
        pose_stop_event = mp.Event()
        pose_ready_event = mp.Event()

        # Create RTMPose3D process
        self.pose_process = RTMPose3DProcess(
            camera_id=self.camera_index,
            frame_buffer_name=self.frame_buffer_name,
            frame_layout_dict=frame_layout,
            pose_buffer_name=self.pose_buffer_name,
            pose_layout_dict=pose_layout,
            detector_config=mmpose_config['person_detector']['config'],
            detector_checkpoint=mmpose_config['person_detector']['checkpoint'],
            pose_config=mmpose_config['pose_estimator']['config'],
            pose_checkpoint=mmpose_config['pose_estimator']['checkpoint'],
            device=mmpose_config.get('device', 'cuda:0'),
            det_conf_threshold=mmpose_config['person_detector'].get('confidence_threshold', 0.5),
            pose_conf_threshold=mmpose_config['pose_estimator'].get('confidence_threshold', 0.3),
            stop_event=pose_stop_event,
            ready_event=pose_ready_event,
            log_queue=None  # Could add logging queue if needed
        )

        # Store events for cleanup
        self.pose_stop_event = pose_stop_event
        self.pose_ready_event = pose_ready_event

        # Start the process
        self.pose_process.start()
        logger.info(f"[CAMERA {self.camera_index}] RTMPose3D process started (PID: {self.pose_process.pid})")

        # Wait for initialization (with timeout)
        if pose_ready_event.wait(timeout=30.0):
            logger.info(f"[CAMERA {self.camera_index}] ✅ RTMPose3D process ready")
        else:
            logger.error(f"[CAMERA {self.camera_index}] ⚠️ RTMPose3D process initialization timeout!")

        return self.pose_process

    def _spawn_scrfd_pipeline(self):
        """DEPRECATED: SCRFD + MediaPipe pipeline (replaced by RTMPose3D)."""
        # DEPRECATED: MediaPipe is no longer used (replaced by RTMPose3D)
        logger.warning(f"[CAMERA WORKER] SCRFD+MediaPipe pipeline disabled - feature deprecated")
        return None, None, None

    def _capture_loop(self):
        """Optimized main capture loop - SINGLE frame write, non-blocking operations."""
        buffer_idx = 0
        last_fps_report = time.time()
        fps_counter = 0
        last_health_check = time.time()
        
        # Pre-compute values to avoid repeated calculations
        detection_interval = self.detection_interval
        ring_mask = self.ring_buffer_mask
        
        # CRITICAL FIX: Create view for write index at offset 0 of the frame buffer
        # This allows GUI processing worker to read the current write position
        # FIX: Use uint64 to match visualizer and avoid dtype mismatch race conditions
        # ATOMIC WRITE FIX: Store as instance variable so write functions can update it atomically
        self.write_index_view = np.frombuffer(
            self.frame_buffer_shm.buf,
            dtype=np.uint64,
            count=1,
            offset=0
        )
        self.write_index_view[0] = 0  # Initialize to 0

        logger.info(f"🎥 Camera {self.camera_index} capture loop started - beginning frame capture from {'ZMQ' if self.config.get('zmq_camera_bridge', {}).get('enabled') else 'V4L2'} source")
        logger.info(f"Write index will be written to offset 0 of frame buffer")
        loop_iterations = 0

        # Track realtime_ready status (sent after N successful frames)
        realtime_ready_sent = False
        REALTIME_READY_THRESHOLD = 10  # Send after 10 successful frames

        while self.running.value:
            # Quick control command check (non-blocking)
            self._process_control_commands_fast()

            if self.paused.value:
                time.sleep(0.1)
                continue

            # Update health check timestamp
            current_time = time.time()
            if current_time - last_health_check >= 1.0:  # Check every second
                last_health_check = current_time
            
            # Capture frame using abstraction layer (V4L2 or ZMQ)
            ret, frame = self.camera_source.read()
            if not ret:
                logger.error(f"Camera {self.camera_index} failed to capture frame at iteration {loop_iterations}")
                self._handle_capture_failure()
                continue
            
            loop_iterations += 1
            # DIAGNOSTIC FIX: Log first 10 frames to verify capture continues
            if loop_iterations <= 10:
                logger.info(f"Camera {self.camera_index} captured frame #{loop_iterations}! Shape: {frame.shape}")

            # Send realtime_ready after successful frame processing startup
            # This clears the GUI overlay "Synchronizing real-time video..."
            if not realtime_ready_sent and loop_iterations >= REALTIME_READY_THRESHOLD:
                logger.warning(f"[REALTIME READY] ⚠️⚠️⚠️ Camera {self.camera_index}: "
                               f"Real-time video sync achieved after {loop_iterations} frames")
                self._send_status('realtime_ready', {
                    'message': 'Real-time video sync achieved',
                    'frames_processed': loop_iterations
                })
                realtime_ready_sent = True

            # DIAGNOSTICS: Track frame arrival timing
            current_frame_time = time.time()
            if self.last_frame_time is not None:
                inter_frame_ms = (current_frame_time - self.last_frame_time) * 1000
                self.inter_frame_intervals.append(inter_frame_ms)

                # Detect burst: frames arriving <10ms apart
                if inter_frame_ms < 10.0:
                    self.burst_count += 1

            # FRAME PACING DISABLED: Removed artificial throttle causing ArcFace starvation
            # Root cause: 30ms sleep was limiting pipeline to 33.3 FPS max, causing ArcFace
            # to run at only 15.4 FPS (11% efficiency) despite 140.9 FPS theoretical capacity.
            #
            # Processing components can handle 30+ FPS:
            #   - SCRFD: 8.37ms (119 FPS theoretical)
            #   - ArcFace: 7.10ms (140.9 FPS theoretical)
            #   - MediaPipe: parallel processing
            #
            # Throttle served no purpose and only created downstream bottleneck.
            #
            # Original code (now disabled):
            # if self.last_frame_time is not None:
            #     time_since_last_ms = (current_frame_time - self.last_frame_time) * 1000
            #     if time_since_last_ms < 30.0:
            #         sleep_time = (30.0 - time_since_last_ms) / 1000.0
            #         if sleep_time > 0:
            #             time.sleep(sleep_time)
            #             current_frame_time = time.time()

            self.last_frame_time = current_frame_time  # Keep for diagnostics/logging

            self.frame_counter += 1
            self.diagnostics_frame_count += 1
            fps_counter += 1
            is_detection_frame = (self.frame_counter % detection_interval == 0)

            # DIAGNOSTICS: Track ring buffer pressure (how many slots have unread frames)
            unread_slots = 0
            if self.metadata_views:
                for metadata_view in self.metadata_views:
                    if metadata_view[3] == 1:  # ready flag = 1 means unread
                        unread_slots += 1
                self.buffer_pressure_samples.append(unread_slots)

            # Pass native resolution frames to detector (no downsampling)
            # RTMDet benefits from higher resolution for better person detection accuracy
            # RTMPose3D will automatically resize person ROIs to 384x288 via TopdownAffine transform
            frame_native = frame  # Use native resolution directly

            # Write to detection ring buffer
            write_pos = self.frame_write_idx.value
            # DIAGNOSTIC FIX: Log first 10 frame writes to verify pipeline continues
            if loop_iterations <= 10:
                logger.info(f"Camera {self.camera_index} writing frame {self.frame_counter} (native res) to position {write_pos}, frame_id={self.frame_counter}")

            # ATOMIC WRITE FIX: Write frame data FIRST, then update write_index LAST
            # Calculate next position to pass to write function
            new_write_pos = (write_pos + 1) & ring_mask
            self._write_frame_optimized(frame_native, write_pos, is_detection_frame, new_write_pos)

            # Update multiprocessing Value for next iteration
            self.frame_write_idx.value = new_write_pos

            # Note: write_index in shared memory is updated INSIDE _write_frame_optimized
            # This ensures atomic write pattern: data FIRST, index LAST

            # CRITICAL FIX: Signal SCRFD that new frame is ready (event-driven wake up)
            # Without this, SCRFD times out every 100ms waiting for frames
            frame_ready_sem = self._get_frame_ready_semaphore()
            if frame_ready_sem is not None:
                frame_ready_sem.release()
                # DIAGNOSTIC FIX: Log semaphore posting for first 10 frames
                if loop_iterations <= 10:
                    logger.info(f"Camera {self.camera_index} posted frame_ready_semaphore for frame {self.frame_counter}")

            # CRITICAL FIX: Disabled Camera Worker write to pose_frame_buffer
            # ROOT CAUSE: Dual-writer race condition with GUI Processing Worker
            # - Camera Worker was writing plain frames at 30 FPS
            # - GUI Processing Worker writes skeleton-drawn frames at 5-10 FPS
            # - Both writing to same buffer caused color cycling and movement artifacts
            # SOLUTION: GUI Processing Worker is now sole writer (writes skeleton-overlaid frames)
            # NOTE: If pose estimation breaks, this indicates RTMPose3D still needs plain frames
            #       In that case, implement separate buffers (raw_pose_frame vs display_pose_frame)

            # DISABLED: Separate pose frame buffer no longer used
            # Pose process now reads native resolution frames from main frame_buffer
            if False:  # self.pose_frame_buffer_shm is not None:
                frame_480p = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

                # Write to pose frame ring buffer (same slot as main buffer)
                slot = write_pos & 15  # 16-slot ring buffer mask
                self.pose_frame_views[slot][:] = frame_480p

                # FRAME SYNC FIX: Write metadata so pose process can read actual frame_id
                # Without this, pose reads uninitialized memory (frame_id=0) causing massive lag
                if hasattr(self, 'pose_metadata_views') and slot in self.pose_metadata_views:
                    metadata = self.pose_metadata_views[slot]
                    metadata[0] = self.frame_counter  # CRITICAL: Same frame_id as main buffer!
                    metadata[1] = int((time.time() - self.start_time) * 1000)  # timestamp_ms
                    metadata[2] = 1 if is_detection_frame else 0  # is_detection_frame
                    metadata[3] = 1  # ready_flag

                # Update pose frame write index (same frame_id for sync)
                struct.pack_into('Q', self.pose_frame_buffer_shm.buf, 0, new_write_pos)

                # SIGNAL POSE PROCESS: Post to semaphore for event-driven wake up
                # Eliminates 1ms polling overhead in PoseProcess
                pose_frame_ready_sem = self._get_pose_frame_ready_semaphore()
                if pose_frame_ready_sem is not None:
                    pose_frame_ready_sem.release()

                # Track pose frame writes
                if not hasattr(self, '_pose_frame_write_count'):
                    self._pose_frame_write_count = 0
                self._pose_frame_write_count += 1

                if not hasattr(self, '_downsample_logged'):
                    logger.info(f"[Camera {self.camera_index}] ✅ Writing to pose buffer (DISABLED)")
                    self._downsample_logged = True

            # Architecture B: Preview writes removed - GUI Processing Worker handles display
            
            # Cycle pinned buffer index for triple buffering
            buffer_idx = (buffer_idx + 1) % 3
            
            # Non-blocking metadata send
            self._send_metadata_nonblocking(is_detection_frame)
            
            # Report FPS periodically (consolidated with diagnostics)
            if current_time - self.last_diagnostics_log >= 5.0:
                elapsed = current_time - self.last_diagnostics_log
                actual_fps = self.diagnostics_frame_count / elapsed if elapsed > 0 else 0

                # Single-line consolidated camera performance report
                logger.info(f"[Camera {self.camera_index}] {actual_fps:.1f} FPS")

                # Send heartbeat status to GUI
                self._send_status('heartbeat', {
                    'current_fps': round(actual_fps, 1),
                    'frames_processed': self.frame_counter,
                    'timestamp': current_time
                })

                # Reset diagnostic counters
                self.last_diagnostics_log = current_time
                self.diagnostics_frame_count = 0
                self.inter_frame_intervals.clear()
                self.burst_count = 0
                self.buffer_pressure_samples.clear()
    
    def _write_frame_optimized(self, frame, write_pos, is_detection_frame, new_write_pos):
        """Optimized single frame write with pinned memory support.

        ATOMIC WRITE PATTERN: Writes frame data FIRST, then updates write_index LAST.
        This prevents readers from seeing the new write_index before data is ready.
        """
        try:
            # CRITICAL FIX: Validate frame shape matches expected resolution
            # This prevents buffer offset artifacts caused by stride mismatches
            expected_shape = (self.actual_resolution[1], self.actual_resolution[0], 3)
            if frame.shape != expected_shape:
                logger.error(
                    f"Camera {self.camera_index}: Frame shape mismatch! "
                    f"Expected {expected_shape}, got {frame.shape}. "
                    f"This will cause buffer offset artifacts. Skipping frame {self.frame_counter}."
                )
                return

            # CRITICAL FIX: Ensure frame is C-contiguous to prevent stride issues
            # Non-contiguous frames can cause color artifacts and memory corruption
            if not frame.flags['C_CONTIGUOUS']:
                logger.warning(
                    f"Camera {self.camera_index}: Frame {self.frame_counter} is not contiguous. "
                    f"Converting to contiguous array (performance impact)."
                )
                frame = np.ascontiguousarray(frame)

            # CRITICAL FIX: Validate strides AFTER ensuring contiguity
            # Frames can be "contiguous" but have wrong row stride (e.g., padded rows)
            # This is the most likely root cause of buffer offset artifacts
            expected_strides = (self.actual_resolution[0] * 3, 3, 1)
            if frame.strides != expected_strides:
                logger.error(
                    f"Camera {self.camera_index}: STRIDE CORRUPTION DETECTED! "
                    f"Frame {self.frame_counter} has invalid strides. "
                    f"Expected: {expected_strides}, Got: {frame.strides}, "
                    f"Shape: {frame.shape}, Contiguous: {frame.flags['C_CONTIGUOUS']}. "
                    f"This indicates upstream buffer corruption or wrong memory layout. Skipping frame."
                )
                return  # Skip corrupted frame

            # Diagnostic logging every 100 frames to detect issues early
            if self.frame_counter % 100 == 0:
                logger.debug(
                    f"Camera {self.camera_index} Frame {self.frame_counter}: "
                    f"shape={frame.shape}, strides={frame.strides}, "
                    f"contiguous={frame.flags['C_CONTIGUOUS']}, "
                    f"expected_shape={expected_shape}"
                )
            # Use pre-allocated array views for maximum performance
            if self.frame_views and write_pos < len(self.frame_views):
                # Use pre-allocated view (fastest path)
                frame_view = self.frame_views[write_pos]
                metadata_view = self.metadata_views[write_pos]

                # Check if previous frame was consumed
                if metadata_view[3] == 1:  # Still ready
                    old_frame_id = metadata_view[0]
                    if self.frame_counter % 100 == 0:  # Log every 100 frames to avoid spam
                        logger.warning(f"Camera {self.camera_index}: Position {write_pos} still has unread frame {old_frame_id}, overwriting with {self.frame_counter}")

                # STEP 1: Write frame data to ring buffer slot
                # Use pinned memory for faster PCIe transfer if available
                if self.pinned_frame_buffers:
                    pinned_idx = self.frame_counter % 3  # Triple buffering
                    pinned_buf = self.pinned_frame_buffers[pinned_idx]

                    # CRITICAL FIX: Validate frame size before pinned memory copy
                    # This prevents flatten/reshape corruption from wrong strides
                    expected_size = self.actual_resolution[1] * self.actual_resolution[0] * 3
                    actual_size = frame.size  # Total number of elements

                    if actual_size != expected_size:
                        logger.error(
                            f"Camera {self.camera_index}: Frame size mismatch before pinned copy! "
                            f"Expected {expected_size}, got {actual_size}. "
                            f"Frame shape: {frame.shape}, strides: {frame.strides}. "
                            f"Falling back to direct copy."
                        )
                        # Fallback to direct copy without pinned memory
                        frame_view[:] = frame
                    else:
                        # Use expected shape for reshape (source of truth), NOT frame.shape
                        expected_shape = (self.actual_resolution[1], self.actual_resolution[0], 3)

                        # Copy to pinned memory with explicit size limit
                        # Use ravel() instead of flat to get 1D view, then limit to expected size
                        pinned_buf.as_array()[:expected_size] = frame.ravel()[:expected_size]

                        # Reshape using EXPECTED shape (not frame.shape which might be corrupted)
                        frame_view[:] = pinned_buf.as_array()[:expected_size].reshape(expected_shape)
                else:
                    # Fallback: direct copy
                    frame_view[:] = frame

                # STEP 2: Write metadata to ring buffer slot
                metadata_view[0] = self.frame_counter  # Frame ID
                metadata_view[1] = int((time.time() - self.start_time) * 1000)  # Timestamp
                metadata_view[2] = 1 if is_detection_frame else 0  # Detection flag
                metadata_view[3] = 1  # Ready flag

                # PERFORMANCE FIX: Track face detection timing for recovery optimization
                if is_detection_frame:
                    self._last_face_detection_time = time.time()

                # STEP 3: Update write_index LAST (atomic "data ready" signal)
                # This ensures readers never see new_write_pos before frame data is complete
                self.write_index_view[0] = new_write_pos

                # Architecture B: GUI writes removed - handled by GUI Processing Worker
                # Note: SCRFD parallel mode uses detection semaphores, not frame_ready_semaphore

            else:
                # Fallback to dynamic allocation (slower)
                logger.warning(f"Using fallback frame write for position {write_pos}")
                self._write_frame_fallback(frame, write_pos, is_detection_frame, new_write_pos)

        except Exception as e:
            logger.error(f"Optimized frame write failed: {e}")
            # Try fallback
            self._write_frame_fallback(frame, write_pos, is_detection_frame, new_write_pos)
    
    def _write_frame_fallback(self, frame, write_pos, is_detection_frame, new_write_pos):
        """Fallback frame write when pre-allocated views are not available.

        ATOMIC WRITE PATTERN: Writes frame data FIRST, then updates write_index LAST.
        This prevents readers from seeing the new write_index before data is ready.
        """
        try:
            # CRITICAL FIX: Validate frame shape matches expected resolution
            expected_shape = (self.actual_resolution[1], self.actual_resolution[0], 3)
            if frame.shape != expected_shape:
                logger.error(
                    f"Camera {self.camera_index}: Frame shape mismatch in fallback! "
                    f"Expected {expected_shape}, got {frame.shape}. Skipping frame {self.frame_counter}."
                )
                return

            # CRITICAL FIX: Ensure frame is C-contiguous
            if not frame.flags['C_CONTIGUOUS']:
                logger.warning(
                    f"Camera {self.camera_index}: Frame {self.frame_counter} not contiguous in fallback. Converting."
                )
                frame = np.ascontiguousarray(frame)

            # CRITICAL FIX: Validate strides in fallback path too
            expected_strides = (self.actual_resolution[0] * 3, 3, 1)
            if frame.strides != expected_strides:
                logger.error(
                    f"Camera {self.camera_index} FALLBACK: STRIDE CORRUPTION! "
                    f"Expected: {expected_strides}, Got: {frame.strides}. Skipping frame {self.frame_counter}."
                )
                return

            # Get the correct offsets from coordinator layout (includes 16-byte header)
            layout = self.buffer_coordinator.get_buffer_layout(self.camera_index, self.actual_resolution)

            frame_size = frame.shape[0] * frame.shape[1] * frame.shape[2]
            frame_offset = layout['detection_frame_offsets'][write_pos]

            # STEP 1: Write frame data to ring buffer slot
            frame_array = np.frombuffer(
                self.frame_buffer_shm.buf,
                dtype=np.uint8,
                count=frame_size,
                offset=frame_offset
            ).reshape(frame.shape)
            frame_array[:] = frame

            # STEP 2: Write metadata to ring buffer slot
            metadata_offset = layout['detection_metadata_offsets'][write_pos]
            metadata = np.frombuffer(
                self.frame_buffer_shm.buf,
                dtype=np.int64,  # Use int64 to match optimized path
                count=8,  # 8 int64 values (64 bytes total)
                offset=metadata_offset
            )
            metadata[0] = self.frame_counter
            metadata[1] = int((time.time() - self.start_time) * 1000)
            metadata[2] = 1 if is_detection_frame else 0
            metadata[3] = 1

            # Update preview (slot 0) - use correct offset from layout
            if write_pos != 0:
                preview_offset = layout['detection_frame_offsets'][0]
                preview_array = np.frombuffer(
                    self.frame_buffer_shm.buf,
                    dtype=np.uint8,
                    count=frame_size,
                    offset=preview_offset
                ).reshape(frame.shape)
                preview_array[:] = frame

            # STEP 3: Update write_index LAST (atomic "data ready" signal)
            # This ensures readers never see new_write_pos before frame data is complete
            self.write_index_view[0] = new_write_pos

            # Legacy preview metadata removed - now handled by ring buffer system in _write_preview_frame_latest

        except Exception as e:
            logger.error(f"Fallback frame write failed: {e}")
    
    def _write_preview_frame_latest(self, frame):
        """Architecture B: Preview writes removed - handled by GUI Processing Worker."""
        # This method is deprecated in Architecture B
        # GUI Processing Worker reads from detection buffers and writes to display buffers
        pass
    
    def _write_preview_frame_latest_deprecated(self, frame):
        """Deprecated method kept for reference only."""
        # Architecture B: This functionality moved to GUI Processing Worker
        pass
    
    def _process_control_commands_fast(self):
        """Process embedding-related commands from mp.Queue (required for large data > 2KB)."""
        try:
            cmd_data = self.control_queue.get_nowait()
            
            # Handle only dict commands with embedding data
            if isinstance(cmd_data, dict):
                command = cmd_data.get('command')
                
                if command == 'register_track_participant':
                    # Handle track registration with embedding data
                    track_id = cmd_data.get('track_id')
                    participant_id = cmd_data.get('participant_id')
                    embedding = cmd_data.get('embedding')
                    
                    if embedding is not None:
                        # Process embedding registration (this is the required functionality)
                        logger.info(f"Processing track registration with embedding: track={track_id}, participant={participant_id}")
                        # Forward to detection process or handle locally as needed
                        
                elif command == 'update_participant_embedding':
                    # Handle embedding updates
                    participant_id = cmd_data.get('participant_id')
                    embedding = cmd_data.get('embedding')
                    
                    if embedding is not None:
                        logger.info(f"Processing embedding update for participant {participant_id}")
                        # Forward to detection process or handle locally as needed
                        
                else:
                    # Simple control commands no longer use mp.Queue fallback
                    logger.warning(f"Non-embedding command '{command}' received via mp.Queue - use CommandBuffer instead")
            else:
                # String commands no longer supported via mp.Queue
                logger.warning(f"Simple command received via mp.Queue - use CommandBuffer instead: {cmd_data}")
                
        except:
            pass  # No commands available
            
            
    def _send_metadata_nonblocking(self, is_detection_frame):
        """Send metadata with overflow protection (non-blocking)."""
        # Metadata communication removed - using CommandBuffer status updates only
        pass
    
    def _send_detection_command(self, command_type: str, payload: Dict[str, Any], 
                              timeout: float = None, retry_count: int = None) -> bool:
        """
        Send command to detection process with robust error handling.
        
        Args:
            command_type: Type of command to send
            payload: Command payload data
            timeout: Timeout per attempt in seconds  
            retry_count: Number of retry attempts
            
        Returns:
            True if command sent successfully, False otherwise
        """
        if not self.detection_command_buffer:
            logger.error(f"No detection command buffer available for command: {command_type}")
            return False
        
        # Get config values or use defaults
        if timeout is None:
            timeout = get_nested_config(self.config, 'buffer_management.commands.timeout_seconds', 2.0)
        if retry_count is None:
            retry_count = get_nested_config(self.config, 'buffer_management.commands.retry_count', 3)
        
        try:
            # Implement retry logic manually since CommandBuffer.send_command doesn't support retry_count
            for attempt in range(retry_count):
                command_id = self.detection_command_buffer.send_command(
                    command_type=command_type,
                    payload=payload,
                    timeout=timeout
                )
                
                if command_id is not None:
                    logger.debug(f"Command sent successfully: {command_type} (id={command_id}) with payload keys: {list(payload.keys())}")
                    return True
                else:
                    logger.warning(f"Command send attempt {attempt + 1}/{retry_count} failed for {command_type}")
                    if attempt < retry_count - 1:
                        retry_delay = get_nested_config(self.config, 'buffer_management.timeouts.command_retry_delay_ms', 10) / 1000.0
                        time.sleep(retry_delay)  # Configurable delay before retry
            
            logger.warning(f"Failed to send command after {retry_count} retries: {command_type}")
            return False
            
        except Exception as e:
            logger.error(f"Exception sending detection command {command_type}: {e}")
            return False
    
    def _handle_capture_failure(self):
        """Handle camera capture failure with reconnection."""
        logger.warning(f"Camera {self.camera_index} capture failed, attempting reconnection")

        if self.camera_source:
            self.camera_source.release()

        time.sleep(1.0)  # Wait before retry

        if self._init_camera():
            logger.info(f"Camera {self.camera_index} reconnected successfully")
        else:
            logger.error(f"Camera {self.camera_index} reconnection failed")
            self.running.value = False
    
    def _process_recovery_queries(self):
        """
        Process participant recovery queries from detection process.
        Runs in a separate thread to avoid blocking main loop.
        """
        logger.info(f"Recovery query processor started for camera {self.camera_index}")
        
        while self.running.value:
            try:
                # PERFORMANCE FIX: Only process recovery queries when needed
                # Skip processing if no faces detected recently
                if not hasattr(self, '_last_face_detection_time') or \
                   time.time() - self._last_face_detection_time > 2.0:
                    no_faces_sleep = get_nested_config(self.config, 'buffer_management.timeouts.no_faces_sleep_ms', 10) / 1000.0
                    time.sleep(no_faces_sleep)  # Configurable sleep when no faces
                    continue
                
                # Get next query (blocks with timeout)
                query = self.recovery_buffer.get_query()
                
                if query is None:
                    # Adaptive sleep based on query frequency
                    polling_interval = get_nested_config(self.config, 'buffer_management.timeouts.recovery_polling_interval_ms', 5) / 1000.0
                    time.sleep(polling_interval)  # Configurable polling interval
                    continue
                
                # Forward to participant manager via queue
                recovery_request = {
                    'type': 'recovery_query',
                    'camera_index': self.camera_index,
                    'query_id': query['query_id'],
                    'bbox': query['bbox'],
                    'shape': query['shape'],
                    'embedding': query['embedding'],
                    'timestamp': time.time()
                }
                
                # Send to participant update queue
                self.participant_update_queue.put(recovery_request)
                
                # Wait for response (with timeout)
                start_time = time.time()
                response_received = False
                
                response_timeout = get_nested_config(self.config, 'buffer_management.timeouts.recovery_response_timeout_ms', 100) / 1000.0
                while time.time() - start_time < response_timeout:  # Configurable timeout
                    if not self.participant_update_queue.empty():
                        msg = self.participant_update_queue.get()
                        
                        if (msg.get('type') == 'recovery_response' and
                            msg.get('query_id') == query['query_id']):
                            
                            # Submit response back to detection process
                            self.recovery_buffer.submit_response(
                                query_id=query['query_id'],
                                participant_id=msg.get('participant_id'),
                                scores=msg.get('scores', {})
                            )
                            
                            response_received = True
                            break
                    
                    time.sleep(0.001)  # 1ms sleep
                
                if not response_received:
                    # Submit empty response on timeout
                    self.recovery_buffer.submit_response(
                        query_id=query['query_id'],
                        participant_id=None,
                        scores={}
                    )
                    
            except Exception as e:
                logger.error(f"Error processing recovery query: {e}")
                
        logger.info(f"Recovery query processor stopped for camera {self.camera_index}")
    
    def _monitor_detection_responses(self):
        """
        Monitor detection command buffer for responses from detection process.
        This includes participant lost notifications and command acknowledgments.
        """
        logger.info(f"Detection response monitor started for camera {self.camera_index}")
        
        # Connect to response command buffer (detection -> camera)
        response_buffer = None
        try:
            if self.command_buffers and 'detection_to_camera' in self.command_buffers:
                response_buffer = CommandBuffer.connect(self.command_buffers['detection_to_camera'])
                logger.info(f"Connected to detection response buffer for camera {self.camera_index}")
        except Exception as e:
            logger.warning(f"Could not connect to detection response buffer: {e}")
        
        while self.running.value:
            try:
                # Check process liveness first to prevent errors
                if not self.detection_process or not self.detection_process.is_alive():
                    logger.info(f"Detection process not alive for camera {self.camera_index}, stopping response monitor")
                    break
                
                # Check for responses from detection process using CommandBuffer
                if response_buffer:
                    response = response_buffer.get_command(timeout=0.1)
                    
                    if response:
                        command_type = response.get('type')
                        payload = response.get('payload', {})
                        
                        if command_type == 'participant_lost':
                            # Forward to participant update queue
                            self._notify_participant_lost(
                                track_id=payload.get('track_id'),
                                participant_id=payload.get('participant_id'),
                                last_data=payload.get('last_data', {})
                            )
                        elif command_type == 'command_ack':
                            # Handle command acknowledgments (for future use)
                            logger.debug(f"Received command acknowledgment: {payload}")
                        else:
                            logger.debug(f"Received unknown response type: {command_type}")
                else:
                    # No response buffer available, just sleep to avoid busy waiting
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in detection response monitor: {e}")
                break
                
        logger.info(f"Detection response monitor stopped for camera {self.camera_index}")
        
        # Cleanup response buffer
        if response_buffer:
            try:
                response_buffer.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup response buffer: {e}")

    def _process_gui_commands(self):
        """
        Process commands from GUI using CommandBuffer.
        Runs in separate thread to avoid blocking main capture loop.
        This replaces the old mp.Queue command processing.
        """
        logger.info(f"GUI command processor started for camera {self.camera_index}")
        
        while self.running.value:
            try:
                # Get command from GUI command buffer with timeout
                command = self.gui_command_buffer.get_command(timeout=0.1)
                
                if command:
                    command_type = command.get('type')
                    payload = command.get('payload', {})
                    command_id = command.get('id', -1)
                    
                    logger.debug(f"Processing GUI command: {command_type} (id={command_id})")
                    
                    # Process the command
                    success, error = self._handle_gui_command(command_type, payload)
                    
                    # Send acknowledgment back to GUI
                    self.gui_command_buffer.send_acknowledgment(command_id, success, error)
                    
                    if success:
                        logger.debug(f"GUI command {command_type} processed successfully")
                    else:
                        logger.error(f"GUI command {command_type} failed: {error}")
                else:
                    # No command available, brief sleep
                    time.sleep(0.001)  # 1ms
                    
            except Exception as e:
                logger.error(f"Error in GUI command processor: {e}")
                error_sleep = get_nested_config(self.config, 'buffer_management.timeouts.command_retry_delay_ms', 10) / 1000.0
                time.sleep(error_sleep)  # Configurable delay on error
                
        logger.info(f"GUI command processor stopped for camera {self.camera_index}")
    
    def _handle_gui_command(self, command_type: str, payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Handle individual GUI command.
        
        Args:
            command_type: Type of command to handle
            payload: Command payload data
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            if command_type == 'register_track_participant':
                # ARCHITECTURE NOTE: This command is sent by GUI when face recognition
                # identifies a participant. However, the MediaPipe process (detection/tracking)
                # already assigns participant IDs via shape-based tracking in the tracker.
                #
                # The face recognition system feeds participant IDs through the embedding
                # path in participantmanager_unified.py, which updates the global participant
                # manager directly (not through this command path).
                #
                # Therefore, this command can be safely acknowledged without action.
                # The tracker will receive updated participant IDs through the normal
                # embedding/recognition flow.

                track_id = payload.get('track_id')
                participant_id = payload.get('participant_id')

                if track_id is not None and participant_id is not None:
                    # Acknowledge command (no action needed - see note above)
                    logger.debug(f"[Camera {self.camera_index}] Acknowledged register_track_participant: "
                               f"track={track_id}, participant={participant_id} (no forwarding needed)")
                    return True, None
                else:
                    return False, f"Missing required fields: track_id={track_id}, participant_id={participant_id}"
                    
            elif command_type == 'update_participant_embedding':
                # ARCHITECTURE NOTE: Embedding updates are handled directly by the
                # GlobalParticipantManager (participantmanager_unified.py), not through
                # this command path. The participant manager has direct access to embeddings
                # from the face recognition system.
                #
                # This command path is unused in the current architecture.
                # Acknowledge and ignore (embedding updates happen through
                # participantmanager_unified.update_participant_embedding directly).

                participant_id = payload.get('participant_id')
                embedding = payload.get('embedding')

                if participant_id is not None and embedding is not None:
                    # Acknowledge command (no action needed - see note above)
                    logger.debug(f"[Camera {self.camera_index}] Acknowledged update_participant_embedding: "
                               f"participant={participant_id} (handled by GlobalParticipantManager)")
                    return True, None
                else:
                    return False, f"Missing required fields: participant_id={participant_id}, embedding={'present' if embedding else 'missing'}"
                    
            elif command_type == 'system_pause':
                self.paused.value = True
                return True, None
                
            elif command_type == 'system_resume':
                self.paused.value = False
                return True, None
                
            elif command_type == 'shutdown':
                self.running.value = False
                return True, None
                
            elif command_type == 'ping':
                # Handle ping/pong handshake with current initialization state
                current_state = getattr(self, '_initialization_state', 'booting')
                logger.info(f"[HANDSHAKE] Received ping from manager, sending pong (state: {current_state})")
                pong_data = {
                    'camera_index': self.camera_index,
                    'type': 'pong',
                    'state': current_state,
                    'timestamp': time.time(),
                    'ping_timestamp': payload.get('timestamp'),
                    'req_id': payload.get('req_id')  # Echo back correlation ID
                }
                # Send pong as status message with correct schema
                self._send_status('pong', pong_data)
                return True, None

            elif command_type == 'handshake_ack':
                # Handle handshake acknowledgment
                camera_idx = payload.get('camera_index', self.camera_index)
                req_id = payload.get('req_id')
                logger.info(f"✅ Handshake confirmed for camera {camera_idx} (req_id={req_id})")
                # Set the event to unblock the main thread
                self.handshake_confirmed.set()
                return True, None

            else:
                return False, f"Unknown command type: {command_type}"

        except Exception as e:
            return False, f"Command handling error: {e}"

    def _notify_participant_lost(self, track_id: int, participant_id: int, 
                                last_data: Dict[str, Any]):
        """Notify participant manager that a participant was lost."""
        if self.participant_update_queue:
            notification = {
                'type': 'participant_lost',
                'camera_index': self.camera_index,
                'track_id': track_id,
                'participant_id': participant_id,
                'last_bbox': last_data.get('bbox'),
                'last_shape': last_data.get('shape'),
                'last_embedding': last_data.get('embedding'),
                'timestamp': time.time()
            }
            
            self.participant_update_queue.put(notification)
            logger.info(f"Notified participant manager: participant {participant_id} "
                       f"lost on camera {self.camera_index}")

    def _send_status(self, status: str, data: Dict[str, Any]):
        """Send status update via CommandBuffer (replaces broken multiprocessing queue)."""
        try:
            # Try CommandBuffer first (preferred method)
            if self.gui_status_buffer:
                # Fix schema: wrap status details in payload envelope
                payload = {
                    'camera_index': self.camera_index,
                    'type': status,  # Status type goes inside payload
                    'timestamp': time.time(),
                    'data': data or {}
                }
                
                # Send via CommandBuffer with correct schema
                success = self.gui_status_buffer.send_command('status', payload)

                if success:
                    logger.info(f"[EnhancedCameraWorker] ✅ Sent status via CommandBuffer: type={status}, camera={self.camera_index}, command_id={success}")
                    return
                else:
                    logger.error(f"[EnhancedCameraWorker] ❌ CommandBuffer send returned False/None for status: {status}")
            
            # Status communication is CommandBuffer-only (no fallback)
            if not self.gui_status_buffer:
                logger.error(f"[EnhancedCameraWorker] No CommandBuffer available for status communication")
            else:
                logger.error(f"[EnhancedCameraWorker] CommandBuffer status send failed for: {status}")
                
        except Exception as e:
            logger.error(f"Failed to send status '{status}': {e}")
            
    def _report_profiling_data(self):
        """Report detailed profiling data periodically."""
        current_time = time.time()
        if current_time - self.last_profile_report >= 10.0:  # Every 10 seconds
            if self.frame_times and self.write_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                avg_write_time = sum(self.write_times) / len(self.write_times)
                max_frame_time = max(self.frame_times)
                max_write_time = max(self.write_times)
                
                logger.info(f"Camera {self.camera_index} profiling:")
                logger.info(f"  Avg frame time: {avg_frame_time:.2f}ms (target: 33.33ms for 30fps)")
                logger.info(f"  Max frame time: {max_frame_time:.2f}ms")
                logger.info(f"  Avg write time: {avg_write_time:.2f}ms")
                logger.info(f"  Max write time: {max_write_time:.2f}ms")
                
                # Clear buffers to avoid memory growth
                self.frame_times = self.frame_times[-100:]  # Keep last 100 samples
                self.write_times = self.write_times[-100:]
                
            self.last_profile_report = current_time
            
    def _log_performance_summary(self, fps):
        """Log performance summary with optimization status."""
        if self.frame_times:
            avg_frame_time = sum(self.frame_times[-30:]) / min(30, len(self.frame_times))
            logger.info(f"Camera {self.camera_index} performance: {fps:.1f} FPS, {avg_frame_time:.2f}ms/frame")
            
            if fps >= 25.0:
                logger.info(f"✓ Camera {self.camera_index} achieving target frame rate")
            elif fps >= 15.0:
                logger.warning(f"⚠ Camera {self.camera_index} moderate performance degradation")
            else:
                logger.error(f"✗ Camera {self.camera_index} significant performance issues")
    
    def _cleanup(self):
        """Clean up resources - FIXED VERSION"""
        logger.info(f"Cleaning up camera worker {self.camera_index}")
        
        # Signal threads to stop first
        self.running.value = False
        
        # Wait for monitoring threads to finish gracefully
        if self.detection_response_thread and self.detection_response_thread.is_alive():
            logger.info("Waiting for detection response monitor to stop")
            self.detection_response_thread.join(timeout=2)
            if self.detection_response_thread.is_alive():
                logger.warning("Detection response monitor did not stop gracefully")
        
        if self.recovery_processor_thread and self.recovery_processor_thread.is_alive():
            logger.info("Waiting for recovery processor to stop")
            self.recovery_processor_thread.join(timeout=2)
            if self.recovery_processor_thread.is_alive():
                logger.warning("Recovery processor did not stop gracefully")
        
        if self.command_processor_thread and self.command_processor_thread.is_alive():
            logger.info("Waiting for command processor to stop")
            self.command_processor_thread.join(timeout=2)
            if self.command_processor_thread.is_alive():
                logger.warning("Command processor did not stop gracefully")
        
        # Stop sub-processes
        # IMPROVED: Longer timeout (10s instead of 5s) to allow proper cleanup
        # This prevents orphaned processes that can block camera access

        # Pose Process
        if hasattr(self, 'pose_process') and self.pose_process and self.pose_process.is_alive():
            logger.info("Terminating Pose process (graceful shutdown with 10s timeout)")
            # Signal stop event for graceful shutdown
            if hasattr(self, 'pose_stop_event') and self.pose_stop_event:
                self.pose_stop_event.set()
            self.pose_process.terminate()
            self.pose_process.join(timeout=10)
            if self.pose_process.is_alive():
                logger.warning("Pose process did not terminate gracefully, force killing...")
                self.pose_process.kill()
                self.pose_process.join(timeout=2)
            else:
                logger.info("Pose process terminated successfully")

        if hasattr(self, 'detection_process') and self.detection_process and self.detection_process.is_alive():
            logger.info("Terminating detection process")
            self.detection_process.terminate()
            self.detection_process.join(timeout=5)
            if self.detection_process.is_alive():
                logger.warning("Detection process did not terminate gracefully, force killing...")
                self.detection_process.kill()
                self.detection_process.join(timeout=2)

        if hasattr(self, 'preprocessing_process') and self.preprocessing_process and self.preprocessing_process.is_alive():
            logger.info("Terminating preprocessing process")
            self.preprocessing_process.terminate()
            self.preprocessing_process.join(timeout=5)
            if self.preprocessing_process.is_alive():
                logger.warning("Preprocessing process did not terminate gracefully, force killing...")
                self.preprocessing_process.kill()
                self.preprocessing_process.join(timeout=2)

        if hasattr(self, 'landmark_process') and self.landmark_process and self.landmark_process.is_alive():
            logger.info("Terminating landmark process")
            self.landmark_process.terminate()
            self.landmark_process.join(timeout=5)
            if self.landmark_process.is_alive():
                logger.warning("Landmark process did not terminate gracefully, force killing...")
                self.landmark_process.kill()
                self.landmark_process.join(timeout=2)
        
        # CRITICAL: Delete numpy array views BEFORE closing shared memory
        # This prevents "cannot close exported pointers exist" errors
        cleanup_views = [
            'frame_views', 'metadata_views', 'preview_frame_view',
            'preview_metadata_view', 'gui_frame_views', 'pose_frame_views'
        ]
        
        for view_name in cleanup_views:
            if hasattr(self, view_name):
                try:
                    views = getattr(self, view_name)
                    if isinstance(views, list):
                        # Clear list of views
                        views.clear()
                    delattr(self, view_name)
                    logger.debug(f"Deleted numpy view: {view_name}")
                except Exception as e:
                    logger.warning(f"Error deleting view {view_name}: {e}")
        
        # Force garbage collection to ensure views are released
        import gc
        gc.collect()
        
        # Now safe to close shared memory
        try:
            if self.frame_buffer_shm:
                self.frame_buffer_shm.close()

            if self.results_shm:
                self.results_shm.close()

            if hasattr(self, 'pose_buffer_shm') and self.pose_buffer_shm:
                self.pose_buffer_shm.close()

            if hasattr(self, 'pose_frame_buffer_shm') and self.pose_frame_buffer_shm:
                self.pose_frame_buffer_shm.close()
                logger.info(f"[Camera {self.camera_index}] Pose frame buffer closed")

            if hasattr(self, 'gui_buffer_shm') and self.gui_buffer_shm:
                self.gui_buffer_shm.close()

            # Let coordinator handle unlinking if we own it
            if self._owns_coordinator:
                self.buffer_coordinator.cleanup_all_buffers()
                
        except Exception as e:
            logger.error(f"Error cleaning up shared memory: {e}")
        
        # Clean up command buffers
        try:
            if self.detection_command_buffer:
                logger.info("Cleaning up detection command buffer")
                self.detection_command_buffer.cleanup()
                self.detection_command_buffer = None
                
            if self.gui_command_buffer:
                logger.info("Cleaning up GUI command buffer")
                self.gui_command_buffer.cleanup()
                self.gui_command_buffer = None
                
        except Exception as e:
            logger.error(f"Error cleaning up command buffers: {e}")
        
        # Release camera
        if self.camera_source:
            self.camera_source.release()
            
        if hasattr(self, 'enable_profiling') and self.enable_profiling and hasattr(self, 'frame_times') and self.frame_times:
            # Final performance report
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            target_time = 33.33  # 30 FPS target
            performance_ratio = target_time / avg_frame_time
            logger.info(f"Camera {self.camera_index} final performance:")
            logger.info(f"  Average frame time: {avg_frame_time:.2f}ms")
            logger.info(f"  Performance ratio: {performance_ratio:.2f}x (1.0 = 30fps target)")
            
        logger.info(f"Camera worker {self.camera_index} cleanup complete")


if __name__ == "__main__":
    # Test the enhanced camera worker
    from confighandler import ConfigHandler
    
    config = ConfigHandler().config
    control_queue = Queue()
    status_queue = Queue()
    
    worker = EnhancedCameraWorker(
        camera_index=0,
        gpu_device_id=0,
        config=config,
        control_queue=control_queue,
        status_queue=status_queue
    )
    
    try:
        worker.start()
        
        # Monitor for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            try:
                status = status_queue.get(timeout=1)
                print(f"Status: {status}")
            except:
                continue
                
    finally:
        control_queue.put('stop')
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()