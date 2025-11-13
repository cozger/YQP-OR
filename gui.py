# CRITICAL FIX: Set LD_PRELOAD for PyTorch's nvJitLink BEFORE any subprocesses are spawned
# WSL2 has /usr/lib/wsl/lib in /etc/ld.so.conf which is searched before RUNPATH, causing
# the old system nvJitLink (CUDA 12.0, missing symbols) to be loaded instead of PyTorch's
# complete nvJitLink (CUDA 12.1+, has all symbols). LD_PRELOAD has highest priority.
import os
import sys
_pytorch_nvjitlink = os.path.join(
    sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}',
    'site-packages', 'nvidia', 'nvjitlink', 'lib', 'libnvJitLink.so.12'
)
if os.path.exists(_pytorch_nvjitlink):
    os.environ['LD_PRELOAD'] = _pytorch_nvjitlink
    print(f"[CUDA] ✅ Preloaded PyTorch's nvJitLink: {_pytorch_nvjitlink}")
else:
    print(f"[CUDA] ⚠️  PyTorch nvJitLink not found at: {_pytorch_nvjitlink}")

# CRITICAL: Initialize CUDA environment BEFORE any other imports
# This ensures CUDA is properly set up before cv2 or any CUDA-using libraries are imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
# CRITICAL: Import from core.gui_cuda_init to ensure consistent module loading
from core.gui_cuda_init import initialize_gui_cuda_environment, verify_cuda_after_cv2_import, verify_cuda_device_accessible
# Initialize immediately (this sets LD_LIBRARY_PATH and other env vars)
_cuda_initialized = initialize_gui_cuda_environment()

# CRITICAL FIX: Pre-load CUDA libraries BEFORE cv2 import
# This ensures CUDA libs are loaded with correct LD_LIBRARY_PATH before cv2 can corrupt it
# Once loaded into process memory, changing LD_LIBRARY_PATH won't affect them
_cuda_preload = verify_cuda_device_accessible()
if _cuda_preload['accessible']:
    print(f"[CUDA] ✅ Pre-loaded CUDA libraries: {_cuda_preload['device_count']} device(s) detected")
else:
    print(f"[CUDA] ⚠️  CUDA pre-load failed: {_cuda_preload['error']}")
    print(f"[CUDA]    GPU acceleration will not be available for face recognition")

import tkinter as tk
from tkinter import ttk
# Linux-compatible: pygrabber only available on Windows
try:
    from pygrabber.dshow_graph import FilterGraph
except ImportError:
    FilterGraph = None  # Not needed on Linux

from PIL import Image, ImageTk, ImageFont, ImageDraw

# CRITICAL: Set multiprocessing start method BEFORE any multiprocessing imports
# Must be called before importing Queue, Process, Array, etc. from multiprocessing
# Prevents semaphore serialization errors in spawn mode
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from multiprocessing import Queue as MPQueue
from multiprocessing import Process, shared_memory, Pipe, Array
import ctypes
from concurrent.futures import ThreadPoolExecutor
import queue

import threading
import time
import io
import pathlib
import logging
import atexit
import struct

# Setup OpenCV CUDA environment before importing cv2 (no-op on Linux)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
try:
    import setup_opencv_environment  # Windows only
except ImportError:
    pass  # Not needed on Linux

import numpy as np
import cv2

# CRITICAL FIX: Restore CUDA library path after cv2 import
# Issue: cv2 import prepends /usr/local/lib to LD_LIBRARY_PATH, which causes
# CUDA libraries to be loaded in wrong order (or not found at all).
# Solution: Ensure CUDA-related paths stay at the front of LD_LIBRARY_PATH
cuda_critical_paths = [
    '/usr/lib/x86_64-linux-gnu',  # CUDA runtime, cuDNN
    '/usr/lib/wsl/lib',            # WSL2 NVIDIA driver
]
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if current_ld_path:
    paths = current_ld_path.split(':')
    # Separate CUDA paths from others
    cuda_paths = []
    other_paths = []
    for p in paths:
        if any(critical in p for critical in ['/usr/lib/x86_64-linux-gnu', '/usr/lib/wsl/lib', '/usr/local/cuda']):
            if p not in cuda_paths:  # Avoid duplicates
                cuda_paths.append(p)
        else:
            if p not in other_paths:  # Avoid duplicates
                other_paths.append(p)

    # Reorder: CUDA paths first, then other paths
    new_ld_path = ':'.join(cuda_paths + other_paths)
    if new_ld_path != current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = new_ld_path
        print(f"[CUDA] Fixed LD_LIBRARY_PATH after cv2 import (CUDA libs at front)")
        print(f"[CUDA]   Before: {current_ld_path[:100]}...")
        print(f"[CUDA]   After:  {new_ld_path[:100]}...")

# CRITICAL FIX: Don't check OpenCV CUDA at module import time
# Issue: cv2 import prepends /usr/local/lib to LD_LIBRARY_PATH, and calling
# verify_cuda_after_cv2_import() here attempts CUDA initialization with corrupted
# library path, breaking CUDA for ONNX Runtime later.
# Solution: Skip OpenCV CUDA check - ONNX Runtime will initialize CUDA correctly in __main__ block.
#
# Original code (removed):
# _cuda_status = verify_cuda_after_cv2_import()
# if not _cuda_status.get('cuda_initialized'):
#     print("[GUI] WARNING: CUDA not available - GPU acceleration disabled")
#
# If OpenCV CUDA is needed in future, check at point of use, not at module import

# Initialize logger
logger = logging.getLogger('GUI')

from core.gui_interface.canvasdrawing import CanvasDrawingManager , draw_overlays_combined
from core.visualization.skeleton_3d_renderer import plot_3d_skeleton
from core.gui_interface.skeleton_3d_controls import Skeleton3DControlPanel
from core.data_streaming.correlator import ChannelCorrelator
from core.data_streaming.LSLHelper import lsl_helper_process
from gui.panel_animations import PanelAnimationController, create_collapse_button
from core.data_streaming.videorecorder import VideoRecorderProcess
from core.data_streaming.audiorecorder import AudioRecorder, VideoAudioRecorder, AudioDeviceManager
from core.gui_interface.guireliability import setup_reliability_monitoring, GUIReliabilityMonitor
# Unified participant management (replaces old multi-participant system)
from core.participant_management import SingleParticipantManager, GlobalParticipantManager
from tkinter import filedialog, messagebox
from core.process_management.confighandler import ConfigHandler
from core.platform_detection import PlatformContext
from core.process_management.cleanup_utils import defensive_cleanup_all
from datetime import datetime
from core.buffer_management.camera_worker_integration import CameraWorkerManager
# REMOVED: from core.buffer_management.gui_buffer_manager import GUIBufferManager
# Using only DisplayBufferManager for clean dual-buffer architecture
from core.buffer_management.coordinator import BufferCoordinator
# CRITICAL FIX: Import worker function at MODULE level for multiprocessing
# When using spawn mode, child process needs to import target function at module level
# If imported inside a method, child process cannot find it (import path not available)
from core.gui_interface.gui_processing_worker import gui_processing_worker_main
from core.gui_interface.display_buffer_manager import DisplayBufferManager
from core.utils.camera_cache import load_camera_cache, save_camera_cache, invalidate_camera_cache
from core.metrics import PoseMetrics  # For type hints in metrics panel updates
from pathlib import Path
from typing import Dict, Optional, Any, List

DEFAULT_DESIRED_FPS = 30
CAM_RESOLUTION = (1280, 720)
CAPTURE_FPS = 30  # Capture FPS for video input, can be different from desired FPS
THROTTLE_N = 1    # Only update GUI every 5th frame (~6-7Hz at 30Hz input)
GUI_sleep_time = 0.01 #Also for throttling GUI, time in seconds for refresh, lower is faster
GUI_scheduler_time = 16 #another throttling variable, in milliseconds, lower is faster

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:       # Pillow <10
    RESAMPLE = Image.LANCZOS

def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource for PyInstaller compatibility.

    When packaged with PyInstaller, resources are extracted to a temporary
    folder (_MEIPASS). This function returns the correct path whether running
    from source or as a compiled executable.

    Args:
        relative_path: Path relative to the script/executable

    Returns:
        Absolute path to the resource

    Example:
        >>> font_path = resource_path("resources/fonts/DejaVuSans.ttf")
        >>> theme_path = resource_path("themes/azure.tcl")
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # Running from source - use script directory
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

def cleanup_camera_processes(device_path: str) -> int:
    """
    Kill any processes blocking the camera device (WSL2 optimization).

    Args:
        device_path: Path to video device (e.g., "/dev/video0")

    Returns:
        Number of processes killed
    """
    import subprocess
    killed_count = 0

    try:
        result = subprocess.run(
            ['fuser', device_path],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split()
            logger.info(f"Cleaning up {len(pids)} process(es) blocking {device_path}")

            for pid in pids:
                try:
                    subprocess.run(['kill', '-9', pid], timeout=1, check=True)
                    killed_count += 1
                except:
                    pass

            time.sleep(0.5)  # Wait for cleanup

    except FileNotFoundError:
        logger.debug("fuser not available")
    except Exception as e:
        logger.debug(f"Cleanup failed: {e}")

    return killed_count

def detect_camera_type(device_path: str) -> str:
    """
    Auto-detect if camera is IR or regular.

    Returns:
        "regular" or "ir"
    """
    import subprocess

    try:
        result = subprocess.run(
            ['v4l2-ctl', '-d', device_path, '--list-formats-ext'],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0:
            output = result.stdout
            # IR cameras typically support GREY format at 640x360
            if 'GREY' in output and '640x360' in output:
                return "ir"
    except:
        pass

    return "regular"

def list_video_devices(config=None, max_devices=10, use_cache=True):
    """
    List available cameras using either ZMQ discovery or V4L2 enumeration.

    Supports caching to speed up subsequent launches (5 min TTL).

    Args:
        config: Configuration dict (checks for ZMQ bridge mode)
        max_devices: Maximum devices to enumerate
        use_cache: If True, check cache before discovery (default: True)

    Returns:
        List of tuples: (camera_index, device_name, width, height)
    """
    import glob
    import subprocess

    devices = []

    # Check if ZMQ bridge mode is enabled
    zmq_enabled = False
    if config:
        zmq_enabled = config.get('zmq_camera_bridge', {}).get('enabled', False)

    # CACHING: Try to load from cache first
    if use_cache:
        cached_data = load_camera_cache()
        if cached_data is not None:
            # FIX #2: Validate cache mode matches current config
            cache_mode = cached_data.get('metadata', {}).get('mode')
            current_mode = 'zmq' if zmq_enabled else 'v4l2'

            if cache_mode == current_mode:
                devices = cached_data.get('cameras', [])
                # FIX #8: Additional validation - ensure non-empty
                if devices:
                    cache_age = cached_data.get('cache_age', 0)
                    logger.info(f"[CACHE] Loaded {len(devices)} camera(s) from cache (age={cache_age:.1f}s, mode={current_mode})")
                    return devices
                else:
                    logger.warning("[CACHE] Cache returned empty camera list - invalidating")
                    invalidate_camera_cache()
                    # Fall through to fresh discovery
            else:
                logger.warning(f"[CACHE] Mode mismatch: cache={cache_mode}, current={current_mode} - invalidating")
                invalidate_camera_cache()
                # Fall through to fresh discovery

    if zmq_enabled:
        # ===== ZMQ MODE =====
        logger.info("ZMQ bridge mode: Discovering cameras via network...")

        try:
            from core.camera_sources.factory import CameraSourceFactory

            # Discover available ZMQ cameras (returns list of tuples: [(index, name, width, height), ...])
            available = CameraSourceFactory.discover_cameras(config, max_cameras=max_devices)

            if available:
                # INJECT DISCOVERED NAMES INTO CONFIG for subprocess access
                # Module-level cache doesn't persist across process boundaries,
                # so we pass camera names via config dict (which IS serialized)

                # Use ConfigHandler.set() method with dot notation
                # This auto-creates zmq_camera_bridge dict if missing and auto-saves
                # IMPORTANT: Use string keys to avoid JSON serialization type mismatch
                # Handle both old format (2-tuple) and new format (4-tuple with resolution)
                camera_names_dict = {}
                for item in available:
                    if len(item) >= 2:
                        idx, name = item[0], item[1]
                        camera_names_dict[str(idx)] = name

                config.set('zmq_camera_bridge.discovered_camera_names', camera_names_dict)

                logger.info(f"[GUI] Injected {len(available)} camera names into config for subprocess")
                logger.debug(f"[GUI] Camera name mapping: {camera_names_dict}")

            if available:
                # Build device list with camera names and resolution
                # Check if probe returned new format (4-tuple with resolution) or old format (2-tuple)
                if available and isinstance(available[0], tuple):
                    for item in available:
                        if len(item) == 4:
                            # New format: (index, name, width, height) - resolution from ZMQ metadata
                            cam_idx, cam_name, width, height = item
                            device_name = f"Cam {cam_idx}: {cam_name} (ZMQ)"
                            devices.append((cam_idx, device_name, width, height))
                            logger.info(f"  ✓ Windows Camera {cam_idx} ({cam_name}): {width}x{height} (ZMQ)")
                        elif len(item) == 2:
                            # Old format: (index, name) - use default resolution
                            cam_idx, cam_name = item
                            zmq_settings = config.get('zmq_camera_bridge', {})
                            default_res = zmq_settings.get('default_resolution', [1280, 720])
                            width, height = default_res[0], default_res[1]
                            device_name = f"Cam {cam_idx}: {cam_name} (ZMQ)"
                            devices.append((cam_idx, device_name, width, height))
                            logger.info(f"  ✓ Windows Camera {cam_idx} ({cam_name}): {width}x{height} (ZMQ, default res)")
                else:
                    # Old format (backward compatibility): [index, ...]
                    for cam_idx in available:
                        device_name = f"Camera {cam_idx} (ZMQ)"
                        devices.append((cam_idx, device_name, width, height))
                        logger.info(f"  ✓ Camera {cam_idx}: {width}x{height} (ZMQ)")

                logger.info(f"Successfully discovered {len(devices)} ZMQ camera(s)")
            else:
                logger.warning("No ZMQ cameras discovered! Check Windows senders are running.")

        except Exception as e:
            logger.error(f"Error discovering ZMQ cameras: {e}")
            import traceback
            traceback.print_exc()

        # CACHING: Save discovery result to cache
        if devices:
            metadata = {'mode': 'zmq', 'zmq_enabled': True}
            save_camera_cache(devices, metadata)
            logger.info(f"[CACHE] Saved {len(devices)} ZMQ camera(s) to cache")

            # ZMQ AUTO-DETECTION: Store default resolution for immediate use
            # Actual resolution will be detected from first camera and updated in camera status handler
            zmq_settings = config.get('zmq_camera_bridge', {})
            default_res = zmq_settings.get('detected_resolution') or zmq_settings.get('default_resolution', [1280, 720])

            # Ensure detected_resolution is set (either already detected or use default)
            if not zmq_settings.get('detected_resolution'):
                config.set('zmq_camera_bridge.detected_resolution', default_res)
                logger.info(f"[GUI] ZMQ initial resolution set to: {default_res[0]}x{default_res[1]} (will be updated from first camera)")

        return devices

    # ===== V4L2 MODE (original code) =====
    logger.info("V4L2 mode: Enumerating cameras via local devices...")

    # Enumerate /dev/video* devices directly
    video_devices = sorted(glob.glob("/dev/video*"))

    if not video_devices:
        logger.warning("No /dev/video* devices found")
        return devices

    logger.info(f"Found {len(video_devices)} video device(s): {video_devices}")

    for device_path in video_devices[:max_devices]:
        # Extract index from device path (e.g., /dev/video0 -> 0)
        try:
            camera_index = int(device_path.replace("/dev/video", ""))
        except ValueError:
            continue

        # Cleanup any blocking processes
        cleanup_camera_processes(device_path)

        # Detect camera type
        camera_type = detect_camera_type(device_path)
        use_mjpeg = (camera_type == "regular")

        cap = None
        try:
            # Open with V4L2 backend explicitly - with timeout to prevent hanging
            logger.info(f"  Probing {device_path} (type: {camera_type})...")

            # Use threading to add timeout to VideoCapture
            import threading
            cap_result = [None]
            cap_exception = [None]

            def open_camera():
                try:
                    cap_result[0] = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
                except Exception as e:
                    cap_exception[0] = e

            # Try to open camera with 2 second timeout
            open_thread = threading.Thread(target=open_camera)
            open_thread.daemon = True
            open_thread.start()
            open_thread.join(timeout=2.0)

            if open_thread.is_alive():
                logger.warning(f"  Timeout opening {device_path} - skipping")
                continue

            if cap_exception[0]:
                raise cap_exception[0]

            cap = cap_result[0]
            if not cap or not cap.isOpened():
                logger.warning(f"  Failed to open {device_path}")
                continue

            # CRITICAL: Set MJPEG format FIRST for regular cameras in WSL2
            if use_mjpeg:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)

            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RESOLUTION[1])

            # Set buffer size to 2 (optimal for WSL2)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            # Try to read a test frame to verify it works
            ret, _ = cap.read()
            if not ret:
                logger.warning(f"  Could not read test frame from {device_path}")
                cap.release()
                continue

            # Get actual resolution
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            device_name = f"Camera {camera_index} ({camera_type})"
            devices.append((camera_index, device_name, actual_w, actual_h))

            logger.info(f"  ✓ {device_path}: {actual_w}x{actual_h} ({camera_type})")

            cap.release()

            # Cleanup after enumeration
            cleanup_camera_processes(device_path)

        except Exception as e:
            logger.error(f"  Error probing {device_path}: {e}")
            if cap:
                cap.release()

    logger.info(f"Successfully enumerated {len(devices)} working camera(s)")

    # CACHING: Save discovery result to cache
    if devices:
        metadata = {'mode': 'v4l2', 'zmq_enabled': False}
        save_camera_cache(devices, metadata)
        logger.info(f"[CACHE] Saved {len(devices)} V4L2 camera(s) to cache")

    return devices

# PIL font cache for Unicode text rendering (performance optimization)
_pil_font_cache = {}

def _get_pil_font(font_size):
    """Load and cache PIL font for Unicode text rendering."""
    if font_size not in _pil_font_cache:
        try:
            _pil_font_cache[font_size] = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                font_size
            )
        except Exception as e:
            logger.warning(f"Failed to load DejaVu Sans font: {e}, using default")
            _pil_font_cache[font_size] = ImageFont.load_default()
    return _pil_font_cache[font_size]

def _draw_pil_text_with_background(frame, text, position, font_size=20, text_color=(0, 0, 0), bg_color=(0, 255, 0)):
    """
    Draw Unicode text with background rectangle using PIL.

    Args:
        frame: OpenCV BGR frame (numpy array)
        text: Text to render (supports Unicode)
        position: (x, y) tuple for top-left corner of text
        font_size: Font size in points (default 20 ≈ cv2 scale 0.6)
        text_color: Text color as RGB tuple (default black)
        bg_color: Background color as RGB tuple (default green)

    Returns:
        Modified frame with text rendered
    """
    # Convert BGR to RGB for PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)

    font = _get_pil_font(font_size)

    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x, y = position
    # Draw background rectangle with padding
    padding = 5
    bg_rect = [
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + padding)
    ]
    draw.rectangle(bg_rect, fill=bg_color)

    # Draw text on top of background
    draw.text((x, y - text_height), text, font=font, fill=text_color)

    # Convert back to BGR for OpenCV
    frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    frame[:] = frame_bgr
    return frame

class YouQuantiPyGUI(tk.Tk):
    BAR_HEIGHT = 150

    def __init__(self):
        super().__init__()
        print("[DEBUG] GUI __init__ started"); import sys; sys.stdout.flush()

        # Load Azure ttk theme (PyInstaller compatible)
        try:
            # Get the path to the azure.tcl file using resource_path for PyInstaller
            theme_file = resource_path(os.path.join('themes', 'azure.tcl'))

            if os.path.exists(theme_file):
                self.tk.call("source", theme_file)
                # Start with light theme by default
                self.tk.call("set_theme", "light")
                self.current_theme = "light"
                print(f"[GUI] Azure theme loaded successfully from {theme_file}")
            else:
                print(f"[GUI] Warning: Azure theme file not found at {theme_file}, using default theme")
                self.current_theme = None
        except Exception as e:
            print(f"[GUI] Error loading Azure theme: {e}")
            self.current_theme = None

        # NOTE: Not hiding window during initialization to avoid geometry conflicts
        # The brief flicker during widget creation is acceptable
        # self.withdraw()  # REMOVED - was interfering with maximize
        print("[DEBUG] Window visible during initialization (prevents maximize issues)"); sys.stdout.flush()

        # CRITICAL: Defensive cleanup of orphaned resources from previous crashes
        # This prevents GUI initialization failures due to stale processes/shared memory
        print("[STARTUP] Running defensive cleanup..."); sys.stdout.flush()
        try:
            processes_killed, shm_removed, camera_procs = defensive_cleanup_all()
            if processes_killed > 0 or shm_removed > 0 or camera_procs > 0:
                print(f"[STARTUP] Cleaned up: {processes_killed} orphaned processes, "
                      f"{shm_removed} shared memory objects, {camera_procs} camera locks")
        except Exception as e:
            print(f"[STARTUP] Warning: Defensive cleanup failed: {e}")
            # Continue anyway - cleanup is best-effort
        sys.stdout.flush()

        # Callback management system to prevent access violations during shutdown
        self._pending_callbacks = []
        self._shutdown_in_progress = False
        self._callback_lock = threading.Lock()
        self._emergency_cleanup_completed = False
        self._shutdown_started = False
        
        # Initialize critical flags for fast display integration
        self.shutdown_flag = False  # Required by schedule_preview
        self.actual_buffer_names = {}  # Store actual buffer names from camera workers
        self._optimized_display_active = False  # Control flag for optimized display loop
        self.camera_resolutions = {}  # Track detected camera resolutions
        self.camera_to_frame_map = {}  # Maps camera_idx to frame slot for stable display lookup
        self._cached_pose_data = {}  # Cache for latest pose data per camera (used by metrics and 3D display)

        # Camera enumeration caching (prevents repeated ZMQ probing on UI changes)
        self._cameras_enumerated = False  # Tracks if camera discovery has run
        self.cams = []  # Cached camera list from discovery

        # Loading UI components for async operations
        self._loading_overlay = None  # Loading overlay frame
        self._loading_label = None  # Loading text label
        self._loading_animation_running = False  # Animation control flag
        self._loading_animation_frame = 0  # Current animation frame
        self._loading_animation_after_id = None  # FIX #6: Track pending animation callback

        # Canvas loading spinners (for camera initialization)
        self._canvas_spinners = {}  # {canvas_idx: {'frame_count': int, 'after_id': str}}

        # PhotoImage references for transparent overlays (prevent garbage collection)
        self._overlay_images = {}  # {canvas_idx: PhotoImage}

        # FIX #3: Discovery thread guard (prevent concurrent discovery)
        self._discovery_in_progress = False  # Tracks if discovery thread is running

        # FIX #1: Prevent recursive build_frames calls
        self._rebuilding_frames = False  # Tracks if build_frames is currently executing

        # FIX #17: Track last values for revert on blocked changes
        self._last_participant_count = 1  # Track for revert on error
        self._last_camera_count = 1

        # Status polling rate limiting (adaptive)
        self._status_poll_interval_ms = 200    # 5 Hz by default
        self._next_status_poll_ts = 0.0
        self._ready_cameras = set()
        
        # Add global exception handler to catch what might be triggering shutdown
        import sys
        def global_exception_handler(exc_type, exc_value, exc_traceback):
            print(f"XXXXXX UNHANDLED EXCEPTION: {exc_type.__name__}: {exc_value}")
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            sys.stdout.flush()
        sys.excepthook = global_exception_handler
        
        #use handler to import configurations
        print("[DEBUG] Initializing ConfigHandler"); sys.stdout.flush()
        self.config = ConfigHandler()
        print("[DEBUG] ConfigHandler initialized"); sys.stdout.flush()
        self.DESIRED_FPS = self.config.get('camera_settings.target_fps', DEFAULT_DESIRED_FPS)
        print("[DEBUG] Config values loaded"); sys.stdout.flush()

        # Apply saved theme preference
        if self.current_theme is not None:
            saved_theme = self.config.get('gui_interface.theme.mode', 'light')
            if saved_theme in ['light', 'dark']:
                try:
                    self.tk.call("set_theme", saved_theme)
                    self.current_theme = saved_theme
                    print(f"[GUI] Applied saved theme preference: {saved_theme}")
                except Exception as e:
                    print(f"[GUI] Error applying saved theme preference: {e}")

        # Advanced detection will be automatically used if configured

        # For tracker
        self.tracker_id_to_slot = {}      # Maps tracker_id -> slot index (for "P#")
        self.slot_to_tracker_id = {}
        self.last_known_bboxes = {}       # track_id -> bbox for spatial matching
        self.participant_last_seen = {}   # participant_id -> timestamp
        
        # For advanced detection bboxes
        self.participant_bboxes = {}  # {(camera_idx, participant_id): bbox}  

        print("[DEBUG] GUI __init__")
        self.title("YouQuantiPy")

        # Disable Tk DPI scaling to avoid pointer/event offset under WSLg
        # This forces Tk logical pixels to match device pixels, fixing click coordinate offsets
        # when Windows display scaling is set to 125%/150%
        try:
            # Per-display where available
            self.tk.call('tk', 'scaling', '-displayof', '.', 1.0)
        except tk.TclError:
            # Fallback for older Tk builds
            self.tk.call('tk', 'scaling', 1.0)

        # Set initial size and position (but don't constrain maximize)
        # Position lower on screen to ensure title bar is visible
        self.geometry("1200x800+50+100")
        self.grid_columnconfigure(0, weight=0)   # fixed width  (left) - Control panel
        self.grid_columnconfigure(1, weight=1)   # expands      (middle) - Center container with dual canvas + timeseries
        self.grid_columnconfigure(2, weight=0, minsize=300)   # fixed width (right) - Metrics panel
        # give rows weight so everything grows vertically
        self.grid_rowconfigure(0, weight=1)  # Full height row (center_container spans both rows internally)

        # get drawing manager with GPU acceleration config
        print("[DEBUG] Initializing CanvasDrawingManager"); sys.stdout.flush()
        self.drawingmanager = CanvasDrawingManager(config=self.config.config)
        print("[DEBUG] CanvasDrawingManager initialized"); sys.stdout.flush()

        # Check if connections are available for mesh drawing
        if self.drawingmanager.face_connections is None:
            # Use points mode if connections not available
            self.drawingmanager.set_face_draw_mode('points')
            print("[GUI] Using points mode for face drawing (connections unavailable)")
        
        # DEBUG_RETINAFACE: Enable debug drawing of raw detections
        self.debug_retinaface = True  # Set to False to disable debug drawing
        # Initialize active camera procs early
        self.active_camera_procs = {}
        
        # Global participant management - create queue early for CameraWorkerManager
        self.participant_update_queue = MPQueue()

        # Enrollment state cache for event-based updates (eliminates lock contention)
        # Updated by 'enrollment_state_update' messages from EnrollmentWorkerThread
        self.enrollment_state_cache = {}  # {participant_id: state_name}

        # ARRAY-BASED IPC: Fixed-size arrays for participant state sharing
        # Replaces Manager().dict() to fix spawn mode semaphore issues
        # Arrays are indexed by participant_id (1-8), index [0] is unused
        MAX_PARTICIPANTS = 9  # Indices 1-8 for participants, [0] unused

        # Enrollment state array: integer codes for state names
        # 0=unknown, 1=idle, 2=enrolling, 3=collecting, 4=validating, 5=enrolled, 6=failed
        self.enrollment_state_array = Array(ctypes.c_int, MAX_PARTICIPANTS, lock=True)

        # Lock state array: 0=unlocked, 1=locked
        self.lock_state_array = Array(ctypes.c_int, MAX_PARTICIPANTS, lock=True)

        # Presence state array: 0=absent, 1=present
        self.participant_presence_array = Array(ctypes.c_int, MAX_PARTICIPANTS, lock=True)

        # Enrollment state name <-> code mappings
        self.ENROLLMENT_STATES = {
            'unknown': 0,
            'idle': 1,
            'enrolling': 2,
            'collecting': 3,
            'validating': 4,
            'enrolled': 5,
            'failed': 6
        }
        self.ENROLLMENT_STATES_REVERSE = {v: k for k, v in self.ENROLLMENT_STATES.items()}

        # Initialize all arrays to default state (0)
        for i in range(MAX_PARTICIPANTS):
            self.enrollment_state_array[i] = 0  # unknown
            self.lock_state_array[i] = 0  # unlocked
            self.participant_presence_array[i] = 0  # absent

        # Lock button visual styling configuration
        # Using distinct colors for each state to make button interactivity obvious
        self.lock_button_styles = {
            'lock_ready': {
                'bg': '#28a745',  # Green - ready to lock
                'fg': 'white',
                'activebackground': '#218838',  # Darker green on hover
                'state': 'normal',
                'relief': 'raised'
            },
            'locked': {
                'bg': '#007bff',  # Blue - currently locked
                'fg': 'white',
                'activebackground': '#0056b3',  # Darker blue on hover
                'state': 'normal',
                'relief': 'raised'
            },
            'enrolling': {
                'bg': '#e9ecef',  # Light grey - in progress
                'fg': '#6c757d',  # Dark grey text
                'activebackground': '#e9ecef',  # No hover effect when disabled
                'state': 'disabled',
                'relief': 'flat'
            },
            'disabled': {
                'bg': '#d6d6d6',  # Medium grey - not ready
                'fg': '#999999',  # Light grey text
                'activebackground': '#d6d6d6',  # No hover effect when disabled
                'state': 'disabled',
                'relief': 'flat'
            },
            'failed': {
                'bg': '#dc3545',  # Red - enrollment failed
                'fg': 'white',
                'activebackground': '#dc3545',  # No hover effect when disabled
                'state': 'disabled',
                'relief': 'flat'
            }
        }

        # Command delivery statistics for GUI monitoring
        self.command_stats = {'sent': 0, 'acknowledged': 0, 'failed': 0}
        
        # Initialize BufferCoordinator for recovery buffers ONLY
        #
        # IMPORTANT: This coordinator is used ONLY for recovery buffers created in the GUI process.
        # Camera buffers (frame, results, detection, embedding, pose) are created in child processes
        # using their own local BufferCoordinator instances to avoid multiprocessing pickling issues.
        #
        # To access camera buffers from GUI, use:
        #   self.camera_worker_manager.shared_memories[camera_index]['detection']['name']
        #   self.camera_worker_manager.shared_memories[camera_index]['frame']['shm']
        #   etc.
        #
        # DO NOT use:
        #   self.buffer_coordinator.detection_buffers[camera_index]  ← WRONG (empty)
        #   self.buffer_coordinator.buffer_registry[camera_index]    ← WRONG (empty)
        print("[DEBUG] Initializing BufferCoordinator"); sys.stdout.flush()
        max_cameras = self.config.get('system.max_cameras', 10)  # Single source of truth for max camera count
        self.buffer_coordinator = BufferCoordinator(camera_count=max_cameras, config=self.config.config)
        print("[DEBUG] BufferCoordinator initialized successfully"); sys.stdout.flush()

        # Create recovery buffers for all cameras
        print("[DEBUG] Creating recovery buffers"); sys.stdout.flush()
        self.recovery_buffer_names = {}
        for i in range(max_cameras):
            try:
                buffer_name = self.buffer_coordinator.create_recovery_buffer(i)
                self.recovery_buffer_names[i] = buffer_name  # Store actual buffer name with PID
                print(f"[DEBUG] Created recovery buffer: {buffer_name}"); sys.stdout.flush()
            except Exception as e:
                print(f"[ERROR] Failed to create recovery buffer for camera {i}: {e}"); sys.stdout.flush()
        print("[DEBUG] Recovery buffers creation completed"); sys.stdout.flush()
        
        # Initialize GUIBufferManager
        print("[DEBUG] Initializing GUIBufferManager"); sys.stdout.flush()
        # REMOVED: self.gui_buffer_manager = GUIBufferManager(self.buffer_coordinator)
        # Using only DisplayBufferManager for clean dual-buffer architecture
        print("[DEBUG] GUIBufferManager initialized successfully"); sys.stdout.flush()
        
        # Initialize Camera Worker Manager for GPU pipeline integration
        print("[DEBUG] Initializing CameraWorkerManager"); sys.stdout.flush()
        self.camera_worker_manager = CameraWorkerManager(
            config=self.config.config,
            participant_update_queue=self.participant_update_queue,
            buffer_coordinator=self.buffer_coordinator,  # Pass coordinator (for recovery buffers only; camera workers create their own)
            gui_buffer_manager=None  # REMOVED: GUIBufferManager, using DisplayBufferManager only
        )
        print("[DEBUG] CameraWorkerManager initialized successfully"); sys.stdout.flush()
        self.camera_health_status = {}  # Track camera health per index
        
        # Load config values BEFORE creating UI controls
        self.participant_count = tk.IntVar(value=self.config.get('startup_mode.participant_count', 2))
        self.camera_count = tk.IntVar(value=self.config.get('startup_mode.camera_count', 2))
        self.enable_mesh = tk.BooleanVar(value=self.config.get('startup_mode.enable_mesh', False))
        self.grid_debug_enabled = tk.BooleanVar(value=False)  # Grid debug visualization toggle
        self.desired_fps = tk.IntVar(value=self.config.get('camera_settings.target_fps', 30))

        # Panel collapse/expand state (load from config)
        gui_layout = self.config.get('gui_layout', {})
        self.left_panel_collapsed = gui_layout.get('left_panel_collapsed', False)
        # right_panel_collapsed will be added in Phase 2 with metrics panel
        self._panel_layout_info = {}  # Store original widget geometry for restoration
        default_res = self.config.get('camera_settings.resolution', '720p')
        self.res_choice = tk.StringVar(value=default_res)
        self.participant_mapping_pipes = {}

        # Single-person mode: No participant management needed
        self.face_recognition = None
        self.global_participant_manager = None
        self.grid_manager = None
        self.participant_update_thread = None
        self.participant_monitor_thread = None
        self.monitoring_active = False

        self.performance_stats = {
            'face_fps': {},
            'pose_fps': {},
            'fusion_fps': {}
        }

        self.data_thread = None
        self.streaming = False
        self.blend_labels = None
        self.worker_procs = []
        self.preview_queues = []
        self.score_queues = []
        self.recording_queues = {}  # Dict: {camera_index: Queue} - created when cameras start
        self.participant_names = {}
        self.score_reader = None
        self.gui_update_thread = None
        self.stop_evt = None

        # LSL helper process components (initialized lazily by _initialize_lsl_helper)
        self.lsl_helper_proc = None
        self.lsl_data_queue = None
        self.lsl_command_queue = None
        self.correlation_buffer = None
        self.corr_array = None

        # Left-side control panel
        self.control_panel = ttk.Frame(self)
        self.control_panel.grid(row=0, column=0, rowspan=2, sticky='ns', padx=10, pady=10)

        # Add collapse button for left panel
        self.left_collapse_btn = create_collapse_button(
            self.control_panel,
            direction='left',
            command=self._toggle_left_panel,
            theme=self.current_theme if self.current_theme else 'light'
        )
        self.left_collapse_btn.place(relx=1.0, rely=0.0, anchor='ne', x=-5, y=5)

        # Phase 2: Single participant mode (fixed to 1)
        self.participant_count.set(1)

        # Editable participant name
        participant_frame = ttk.Frame(self.control_panel)
        participant_frame.pack(anchor='w', pady=(0,20), fill='x')
        ttk.Label(participant_frame, text="Participant:", foreground='gray').pack(side='left')

        self.participant_name_var = tk.StringVar(value="P1")
        self.participant_name_entry = ttk.Entry(
            participant_frame,
            textvariable=self.participant_name_var,
            width=20
        )
        self.participant_name_entry.pack(side='left', padx=(5, 0))

        # Bind validation on focus out and Enter key
        self.participant_name_entry.bind('<FocusOut>', self._on_participant_name_change)
        self.participant_name_entry.bind('<Return>', self._on_participant_name_change)

        self.camera_count.set(1)  # Fixed to 1 active camera

        #FPS input
        ttk.Label(self.control_panel, text="FPS:").pack(anchor='w')
        self.fps_spin = tk.Spinbox(
            self.control_panel,
            from_=1, to=120,
            textvariable=self.desired_fps,
            width=5,
            command=lambda: None  # no‐op, we just read .get() later
        )
        self.fps_spin.pack(anchor='w', pady=(0,20))

        # ─── Resolution selector ───────────────────────────────────────
        ttk.Label(self.control_panel, text="Resolution:").pack(anchor='w')
        # map label→(w,h)
        self.res_map = {
            "4K":    (3840, 2160),
            "2K":    (2560, 1440),
            "1080p": (1920, 1080),
            "720p":  (1280, 720),
            "480p":  ( 640, 480),
            "240p":  ( 320, 240),
        }
        self.res_menu = ttk.Combobox(
            self.control_panel,
            textvariable=self.res_choice,
            values=list(self.res_map.keys()),
            state="readonly",
            width=7
        )
        self.res_menu.pack(anchor='w', pady=(0, 10))
        self.res_menu.bind("<<ComboboxSelected>>", lambda e: self.on_resolution_change())

        # Auto-detected resolution label (shown only in ZMQ mode)
        self.auto_res_label = ttk.Label(
            self.control_panel,
            text="",
            foreground="#4CAF50"  # Green color for auto-detected indicator
        )
        # Will be shown/hidden by _update_resolution_selector_state()

        # Initialize resolution selector state based on ZMQ mode
        self._update_resolution_selector_state()

        # Refresh cameras button (ZMQ re-enumeration)
        self.refresh_btn = ttk.Button(
            self.control_panel,
            text="⟳ Refresh Cameras",
            command=self.refresh_camera_list
        )
        self.refresh_btn.pack(anchor='w', pady=(0, 10))

        # Theme toggle button
        self.theme_toggle_btn = ttk.Button(
            self.control_panel,
            text="☾ Dark Mode",
            command=self.toggle_theme
        )
        self.theme_toggle_btn.pack(anchor='w', pady=(0, 20))

        # Update button text based on saved theme preference
        if self.current_theme == "dark":
            self.theme_toggle_btn.config(text="☀ Light Mode")

        # Phase 2: Center container with dual canvas + timeseries (internal row split)
        self.center_container = ttk.Frame(self)
        self.center_container.grid(row=0, column=1, rowspan=2, pady=10, padx=10, sticky='nsew')
        self.center_container.grid_columnconfigure(0, weight=1)  # Full width
        self.center_container.grid_rowconfigure(0, weight=8)     # 80% - Dual canvas area
        self.center_container.grid_rowconfigure(1, weight=2)     # 20% - Timeseries data

        # Phase 2: Metrics panel (Column 2)
        self.metrics_panel = ttk.LabelFrame(self, text="Metrics")
        self.metrics_panel.grid(row=0, column=2, rowspan=2, sticky='nsew', padx=(0, 10), pady=10)

        # Add collapse button for metrics panel
        self.right_collapse_btn = create_collapse_button(
            self.metrics_panel,
            direction='right',
            command=self._toggle_right_panel,
            theme=self.current_theme if self.current_theme else 'light'
        )
        self.right_collapse_btn.place(relx=0.0, rely=0.0, anchor='nw', x=5, y=5)

        # Head orientation section
        ttk.Label(self.metrics_panel, text="Head Orientation", font=('TkDefaultFont', 20, 'bold')).pack(anchor='w', padx=5, pady=(25, 2))
        self.head_pitch_label = ttk.Label(self.metrics_panel, text="Pitch: --", font=('TkDefaultFont', 20))
        self.head_pitch_label.pack(anchor='w', padx=10, pady=2)
        self.head_yaw_label = ttk.Label(self.metrics_panel, text="Yaw: --", font=('TkDefaultFont', 20))
        self.head_yaw_label.pack(anchor='w', padx=10, pady=2)
        self.head_roll_label = ttk.Label(self.metrics_panel, text="Roll: --", font=('TkDefaultFont', 20))
        self.head_roll_label.pack(anchor='w', padx=10, pady=(2, 10))

        # Neck biomechanics section
        ttk.Separator(self.metrics_panel, orient='horizontal').pack(fill='x', padx=5, pady=5)
        ttk.Label(self.metrics_panel, text="Neck & Posture", font=('TkDefaultFont', 20, 'bold')).pack(anchor='w', padx=5, pady=(5, 2))
        self.neck_angle_label = ttk.Label(self.metrics_panel, text="Neck Flexion: --", font=('TkDefaultFont', 20))
        self.neck_angle_label.pack(anchor='w', padx=10, pady=2)
        self.neck_sagittal_label = ttk.Label(self.metrics_panel, text="Sagittal Angle: --", font=('TkDefaultFont', 20))
        self.neck_sagittal_label.pack(anchor='w', padx=10, pady=2)
        self.forward_head_label = ttk.Label(self.metrics_panel, text="Forward Head: --", font=('TkDefaultFont', 20))
        self.forward_head_label.pack(anchor='w', padx=10, pady=2)

        # Shoulder metrics section (Enhanced v6.0 - Multi-Component System)
        ttk.Label(self.metrics_panel, text="Shoulder Elevation", font=('TkDefaultFont', 20, 'bold')).pack(anchor='w', padx=5, pady=(10, 2))
        self.shoulder_left_label = ttk.Label(self.metrics_panel, text="Left: --", font=('TkDefaultFont', 18))
        self.shoulder_left_label.pack(anchor='w', padx=10, pady=2)
        self.shoulder_right_label = ttk.Label(self.metrics_panel, text="Right: --", font=('TkDefaultFont', 18))
        self.shoulder_right_label.pack(anchor='w', padx=10, pady=(2, 10))

        # Performance stats section
        ttk.Separator(self.metrics_panel, orient='horizontal').pack(fill='x', padx=5, pady=5)
        ttk.Label(self.metrics_panel, text="Performance", font=('TkDefaultFont', 20, 'bold')).pack(anchor='w', padx=5, pady=(5, 2))
        self.fps_label = ttk.Label(self.metrics_panel, text="FPS: --", font=('TkDefaultFont', 20))
        self.fps_label.pack(anchor='w', padx=10, pady=2)
        self.latency_label = ttk.Label(self.metrics_panel, text="Latency: --", font=('TkDefaultFont', 20))
        self.latency_label.pack(anchor='w', padx=10, pady=(2, 10))

        # Keypoint counts section
        ttk.Separator(self.metrics_panel, orient='horizontal').pack(fill='x', padx=5, pady=5)
        ttk.Label(self.metrics_panel, text="Detection", font=('TkDefaultFont', 20, 'bold')).pack(anchor='w', padx=5, pady=(5, 2))
        self.face_keypoints_label = ttk.Label(self.metrics_panel, text="Face: --", font=('TkDefaultFont', 20))
        self.face_keypoints_label.pack(anchor='w', padx=10, pady=2)
        self.body_keypoints_label = ttk.Label(self.metrics_panel, text="Body: --", font=('TkDefaultFont', 20))
        self.body_keypoints_label.pack(anchor='w', padx=10, pady=2)

        # Phase 2: Triple canvas area (Row 0 of center_container - 80% height)
        # Updated to support laparoscopic video canvas
        self.dual_canvas_frame = ttk.Frame(self.center_container)
        self.dual_canvas_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 5))
        self.dual_canvas_frame.grid_columnconfigure(0, weight=1)  # Live feed canvas
        self.dual_canvas_frame.grid_columnconfigure(1, weight=1)  # 3D skeleton canvas
        self.dual_canvas_frame.grid_columnconfigure(2, weight=1)  # Laparoscopic video canvas
        self.dual_canvas_frame.grid_rowconfigure(0, weight=1)

        # Live feed canvas (left side)
        self.live_feed_frame = ttk.LabelFrame(self.dual_canvas_frame, text="Live Feed")
        self.live_feed_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        self.live_feed_frame.grid_columnconfigure(0, weight=1)
        self.live_feed_frame.grid_rowconfigure(0, weight=1)

        self.live_canvas = tk.Canvas(self.live_feed_frame, bg='black', highlightthickness=0, borderwidth=0)
        self.live_canvas.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        # Initialize canvas with drawing manager (frame_idx=0 for single camera)
        self.drawingmanager.initialize_canvas(self.live_canvas, 0)

        # Camera selector below canvas
        ttk.Label(self.live_feed_frame, text="Camera:").grid(row=1, column=0, sticky='w', padx=5, pady=(5, 2))
        self.camera_combo = ttk.Combobox(self.live_feed_frame, state='readonly', width=25)
        self.camera_combo.grid(row=2, column=0, sticky='ew', padx=5)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)

        # Camera status label below selector
        self.camera_status_label = ttk.Label(self.live_feed_frame, text="Camera not selected", foreground='gray')
        self.camera_status_label.grid(row=3, column=0, sticky='w', padx=5, pady=(2, 5))

        # 3D skeleton canvas (right side) - PLACEHOLDER for future implementation
        self.skeleton_3d_frame = ttk.LabelFrame(self.dual_canvas_frame, text="3D Skeleton")
        self.skeleton_3d_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        self.skeleton_3d_frame.grid_columnconfigure(0, weight=1)
        self.skeleton_3d_frame.grid_rowconfigure(0, weight=1)

        self.skeleton_canvas = tk.Canvas(self.skeleton_3d_frame, bg='#1e1e1e', highlightthickness=0, borderwidth=0)
        self.skeleton_canvas.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        # Placeholder text (will be replaced with matplotlib 3D plot later)
        self.skeleton_canvas.create_text(
            400, 300,  # Center position (will auto-adjust on resize)
            text="3D Skeleton Visualization\n(Phase 3)",
            fill='white',
            font=('Arial', 14),
            justify='center',
            tags='placeholder'
        )

        # 3D View Control Panel (modular component)
        self.skeleton_3d_controls = Skeleton3DControlPanel(
            parent=self.skeleton_3d_frame,
            config=self.config
        )
        # Position in bottom-right corner using place geometry
        self.skeleton_3d_controls.place(
            relx=1.0, rely=1.0,
            anchor='se',
            x=-10, y=-10  # 10px padding from corner
        )

        # Laparoscopic video canvas (right side) - NEW
        self.laparoscopic_frame = ttk.LabelFrame(self.dual_canvas_frame, text="Laparoscopic Video")
        self.laparoscopic_frame.grid(row=0, column=2, sticky='nsew', padx=(5, 0))
        self.laparoscopic_frame.grid_columnconfigure(0, weight=1)
        self.laparoscopic_frame.grid_rowconfigure(0, weight=1)

        self.laparoscopic_canvas = tk.Canvas(self.laparoscopic_frame, bg='#1e1e1e', highlightthickness=0, borderwidth=0)
        self.laparoscopic_canvas.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        # Initialize canvas with drawing manager (canvas_idx=1 for laparoscopic)
        self.drawingmanager.initialize_canvas(self.laparoscopic_canvas, 1)

        # Laparoscopic camera selector below canvas
        ttk.Label(self.laparoscopic_frame, text="Laparoscopic Video:").grid(row=1, column=0, sticky='w', padx=5, pady=(5, 2))
        self.laparoscopic_camera_combo = ttk.Combobox(self.laparoscopic_frame, state='readonly', width=25)
        self.laparoscopic_camera_combo.grid(row=2, column=0, sticky='ew', padx=5)
        self.laparoscopic_camera_combo.bind("<<ComboboxSelected>>", self.on_laparoscopic_camera_selected)

        # Laparoscopic status label below selector
        self.laparoscopic_status_label = ttk.Label(self.laparoscopic_frame, text="Not connected", foreground='gray')
        self.laparoscopic_status_label.grid(row=3, column=0, sticky='w', padx=5, pady=(2, 5))

        # Placeholder text (will be replaced when connected)
        self.laparoscopic_canvas.create_text(
            400, 300,  # Center position (will auto-adjust on resize)
            text="Laparoscopic Video\n(Not Connected)",
            fill='white',
            font=('Arial', 14),
            justify='center',
            tags='laparoscopic_placeholder'
        )

        # Phase 2, Step 9.1: Initialize self.frames for two canvases (main camera + laparoscopic)
        # MUST be after all canvases and labels are created
        self.frames = [
            {  # Frame 0: Main camera (live feed)
                'canvas': self.live_canvas,
                'meta_label': self.camera_status_label,
                'health_label': self.camera_status_label,
                'resolution_label': None,
                'combo': self.camera_combo,
                'proc': None,
                'use_auto_fps': tk.BooleanVar(value=False),
                'camera_index': 0,
            },
            {  # Frame 1: Laparoscopic video
                'canvas': self.laparoscopic_canvas,
                'meta_label': self.laparoscopic_status_label,
                'health_label': self.laparoscopic_status_label,
                'resolution_label': None,
                'combo': self.laparoscopic_camera_combo,
                'proc': None,
                'use_auto_fps': tk.BooleanVar(value=False),
                'camera_index': 1,
            }
        ]
        self._current_camera_index = None  # Track main camera device
        self._current_laparoscopic_index = None  # Track laparoscopic camera device

        print("[DEBUG] Triple canvas created - live feed, 3D skeleton, and laparoscopic video"); sys.stdout.flush()

        # Phase 2: Timeseries data frame (Row 1 of center_container - 20% height)
        self.timeseries_frame = ttk.LabelFrame(self.center_container, text="Timeseries Data")
        self.timeseries_frame.grid(row=1, column=0, sticky='nsew', pady=(5, 0))
        self.timeseries_frame.grid_columnconfigure(0, weight=1)
        self.timeseries_frame.grid_rowconfigure(0, weight=1)

        # Single participant ID for ECG/EEG (P1)
        self.single_participant_id = "P1"
        if not hasattr(self, 'participant_names'):
            self.participant_names = {}
        self.participant_names[0] = self.single_participant_id

        # Create single ECG strip for P1
        self._create_single_ecg_strip()

        print("[DEBUG] Timeseries frame and ECG strip created"); sys.stdout.flush()

        # ============================================================================
        # CRITICAL FIX: Initialize display processing worker BEFORE building frames
        # ============================================================================
        print("[DEBUG] Initializing display processing worker"); sys.stdout.flush()
        try:
            worker_init_success = self.initialize_display_processing()
            if worker_init_success:
                print("[DEBUG] ✅ Display processing worker started successfully"); sys.stdout.flush()

                # Wait for worker to be ready (with timeout)
                print("[DEBUG] Waiting for worker to become ready..."); sys.stdout.flush()
                # Note: _wait_for_worker_ready_async() is called automatically from initialize_display_processing()
            else:
                print("[DEBUG] ❌ WARNING: Display processing worker initialization failed!"); sys.stdout.flush()
                print("[DEBUG] Camera previews will NOT work without display worker!"); sys.stdout.flush()
        except Exception as e:
            print(f"[DEBUG] ❌ ERROR initializing display processing worker: {e}"); sys.stdout.flush()
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
        print("[DEBUG] Display processing worker initialization complete"); sys.stdout.flush()

        print("[DEBUG] About to call build_frames"); sys.stdout.flush()
        # MOVED: build_frames() now called after mainloop starts (see main section)
        # self.build_frames()
        print("[DEBUG] build_frames will be called after mainloop starts"); sys.stdout.flush()
        print("[DEBUG] Scheduling GUI updates"); sys.stdout.flush()
        # NOTE: schedule_preview now starts only when cameras are selected - moved to camera lifecycle
        self.after(100, self.update_gui_health)  # Independent GUI health monitoring
        self.after(GUI_scheduler_time, self.continuous_correlation_monitor)
        self.after(1000, self.update_participant_names)
        self.after(500, self._update_3d_skeleton_display)  # Start 3D skeleton visualization (10 FPS)
        self.after(500, self._update_metrics_panel)  # Start pose metrics panel updates (10 FPS)
        print("[DEBUG] GUI updates scheduled"); sys.stdout.flush()

        # Toggles
        ttk.Checkbutton(
            self.control_panel,
            text="Send Complete Keypoint Data to LSL",
            variable=self.enable_mesh,
            command=self.on_mesh_toggle
        ).pack(anchor='w')

        ttk.Checkbutton(
            self.control_panel,
            text="Show Grid Debug (MediaPipe)",
            variable=self.grid_debug_enabled,
            command=self.on_grid_debug_toggle
        ).pack(anchor='w', pady=(0,10))

        # ─── Video Recording Section ───────────────────────────────────
        record_frame = ttk.LabelFrame(self.control_panel, text="Video Recording")
        record_frame.pack(fill='x', pady=(20, 0))
        # Record toggle
        self.record_video = tk.BooleanVar(value=self.config.get('video_recording.enabled', False))
        ttk.Checkbutton(
            record_frame,
            text="Enable Video Recording",
            variable=self.record_video,
            command=self.on_record_toggle
        ).pack(anchor='w', padx=5, pady=(5, 0))
        # Save directory
        dir_frame = ttk.Frame(record_frame)
        dir_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(dir_frame, text="Save to:").pack(side='left')
        self.save_dir = tk.StringVar(value=self.config.get('video_recording.save_directory', './recordings'))
        self.dir_entry = ttk.Entry(dir_frame, textvariable=self.save_dir, width=25)
        self.dir_entry.pack(side='left', padx=(5, 0))
        ttk.Button(
            dir_frame,
            text="Browse",
            command=self.browse_save_directory,
            width=8
        ).pack(side='left', padx=(5, 0))
        # Filename template
        name_frame = ttk.Frame(record_frame)
        name_frame.pack(fill='x', padx=5, pady=(0, 5))
        ttk.Label(name_frame, text="Filename:").pack(side='left')
        self.filename_template = tk.StringVar(
            value=self.config.get('video_recording.filename_template', '{participant}_{timestamp}')
        )
        self.name_entry = ttk.Entry(name_frame, textvariable=self.filename_template, width=25)
        self.name_entry.pack(side='left', padx=(5, 0))
        # Help text
        help_text = ttk.Label(
            record_frame, 
            text="Available: {participant}, {camera}, {timestamp}",
            font=('Arial', 8),
            foreground='gray'
        )
        help_text.pack(anchor='w', padx=5)
        # Recording buttons
        button_frame = ttk.Frame(record_frame)
        button_frame.pack(fill='x', padx=5, pady=5)
        self.record_now_btn = ttk.Button(
            button_frame,
            text="Start Video Recording",
            command=self.toggle_immediate_recording,
            state='disabled'
        )
        self.record_now_btn.pack(side='left', padx=(0, 5))
        self.immediate_recording = False


        self.stop_record_btn = ttk.Button(
            button_frame,
            text="Stop Video Recording",
            command=self.stop_immediate_recording,
            state='disabled'
        )
        self.stop_record_btn.pack(side='left', padx=(0, 5))

        # LSL-synchronized recording toggle
        self.record_with_lsl_var = tk.BooleanVar(value=self.config.get('video_recording.auto_start_with_lsl', False))
        self.record_with_lsl_toggle = ttk.Checkbutton(
            record_frame,
            text="Start recording with LSL stream",
            variable=self.record_with_lsl_var,
            command=self.on_record_with_lsl_toggle
        )
        self.record_with_lsl_toggle.pack(anchor='w', padx=5, pady=(5, 0))

        # Recording status
        self.record_status = ttk.Label(record_frame, text="Not recording", foreground='gray')
        self.record_status.pack(anchor='w', padx=5, pady=(5, 5))
        # Initialize recording
        self.video_recorders = {}
        self.recording_active = False
        # Global frame counter for synchronized recording (shared across all cameras)
        self.global_frame_counter = 0
        self.global_frame_counter_lock = threading.Lock()
        # Initially hide/show based on toggle
        self.on_record_toggle()


        # ─── Audio Recording Section ───────────────────────────────────
        audio_frame = ttk.LabelFrame(self.control_panel, text="Audio Recording")
        audio_frame.pack(fill='x', pady=(10, 0))
        # Audio recording options
        self.audio_enabled = tk.BooleanVar(value=self.config.get('audio_recording.enabled', False))
        ttk.Checkbutton(
            audio_frame,
            text="Enable Audio Recording",
            variable=self.audio_enabled,
            command=self.on_audio_toggle
        ).pack(anchor='w', padx=5, pady=(5, 0))
        # Audio mode selection
        self.audio_mode_frame = ttk.Frame(audio_frame)
        self.audio_mode_frame.pack(fill='x', padx=20, pady=5)
        self.audio_mode = tk.StringVar(value="standalone")
        ttk.Radiobutton(
            self.audio_mode_frame,
            text="Standalone Audio",
            variable=self.audio_mode,
            value="standalone",
            command=self.on_audio_mode_change
        ).pack(anchor='w')
        ttk.Radiobutton(
            self.audio_mode_frame,
            text="Audio with Video",
            variable=self.audio_mode,
            value="with_video",
            command=self.on_audio_mode_change
        ).pack(anchor='w')

        # Audio control buttons
        audio_button_frame = ttk.Frame(audio_frame)
        audio_button_frame.pack(fill='x', padx=5, pady=5)
        self.start_audio_btn = ttk.Button(
            audio_button_frame,
            text="Start Audio Recording",
            command=self.start_audio_recording,
            state='disabled'
        )
        self.start_audio_btn.pack(side='left', padx=(0, 5))

        self.stop_audio_btn = ttk.Button(
            audio_button_frame,
            text="Stop Audio Recording",
            command=self.stop_audio_recording,
            state='disabled'
        )
        self.stop_audio_btn.pack(side='left')

        # Audio device assignment button
        self.audio_device_btn = ttk.Button(
            audio_frame,
            text="Configure Audio Devices",
            command=self.configure_audio_devices
        )
        self.audio_device_btn.pack(fill='x', padx=5, pady=5)
        # Audio status
        self.audio_status = ttk.Label(audio_frame, text="Audio: Not configured", foreground='gray')
        self.audio_status.pack(anchor='w', padx=5)
        # Initialize audio
        self.audio_recorders = {}
        self.audio_device_assignments = self.config.get('audio_devices', {})
        self.available_audio_devices = []
        self.audio_recording_active = False
        self.refresh_audio_devices()
        # Initially show/hide based on state
        self.on_audio_toggle()

        # ===== Bluetooth Devices Section =====
        bluetooth_frame = ttk.LabelFrame(self.control_panel, text="Bluetooth Devices", padding=5)
        bluetooth_frame.pack(fill='x', pady=5)

        # Bluetooth device assignment button
        self.bluetooth_device_btn = ttk.Button(
            bluetooth_frame,
            text="Configure Bluetooth Devices",
            command=self.configure_bluetooth_devices
        )
        self.bluetooth_device_btn.pack(fill='x', padx=5, pady=5)

        # Bluetooth status
        self.bluetooth_status = ttk.Label(bluetooth_frame, text="Bluetooth: Not configured", foreground='gray')
        self.bluetooth_status.pack(anchor='w', padx=5)

        # Initialize Bluetooth manager
        from core.data_streaming.bluetooth_manager import BluetoothDeviceManager
        self.bluetooth_manager = BluetoothDeviceManager(
            data_callback=self._bluetooth_data_callback,
            max_reconnect_attempts=self.config.get('bluetooth_settings', {}).get('reconnect_retries', 5),
            config=self.config  # Pass config for ZMQ backend
        )
        self.bluetooth_device_assignments = self.config.get('bluetooth_devices', {})

        # Auto-discover ZMQ Bluetooth devices if bridge is enabled
        if self.config.get('bluetooth_bridge', {}).get('enabled', False):
            print("[GUI] ZMQ Bluetooth bridge enabled, attempting auto-discovery...")
            try:
                discovered_devices = self.bluetooth_manager.discover_devices_sync(
                    backend_type='zmq_bluetooth',
                    timeout=5.0  # Short timeout for startup
                )
                if discovered_devices:
                    print(f"[GUI] Auto-discovered {len(discovered_devices)} ZMQ Bluetooth device(s)")
                    for dev in discovered_devices:
                        print(f"[GUI]   - {dev.name} ({dev.mac_address})")
                else:
                    print("[GUI] No ZMQ Bluetooth devices discovered (Windows bridge may not be running)")
            except Exception as e:
                print(f"[GUI] ZMQ Bluetooth auto-discovery failed: {e}")

        self._restore_bluetooth_assignments()
        self._update_bluetooth_status()

        # Action Buttons
        ttk.Button(self.control_panel, text="Start Data Stream", command=self.start_stream).pack(fill='x', pady=(2))
        ttk.Button(self.control_panel, text="Stop Data Stream", command=self.stop_stream).pack(fill='x', pady=(2))
        ttk.Button(self.control_panel, text="Reset", command=self.reset).pack(fill='x', pady=(2))
        ttk.Button(self.control_panel, text="Save Current Settings", command=self.save_current_settings).pack(fill='x', pady=(10,2))

        # Save window geometry on close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Bind F11 for fullscreen toggle
        self.bind('<F11>', self._toggle_fullscreen)

        # Periodic check disabled - was causing log spam
        # Event-based detection in _on_window_resize should be sufficient
        # if self._is_wsl():
        #     self.after(500, self._check_maximize_state)
        #     print("[GUI] Started periodic maximize check for WSLg")

        # === RELIABILITY MONITORING SETUP ===
        # Configure reliability monitoring (optional - uses defaults if not provided)
        reliability_config = {
            'memory_growth_threshold': 800,    # memory threshold in MB
            'max_queue_size': 8,               # Smaller queues for better responsiveness  
            'gui_freeze_threshold': 3.0,       # More sensitive freeze detection
            'stats_report_interval': 60,      # Report every N seconds
            'resource_check_interval': 10,      # More frequent memory checks (default: 10)
            'queue_check_interval': 5,         # More frequent queue checks (default: 5)
            'gui_check_interval': 1,         # More sensitive GUI freeze detection (default: 1)
        }
        
        # Setup monitoring - this adds convenience methods to self
        print("[DEBUG] Setting up reliability monitoring"); sys.stdout.flush()
        self.reliability_monitor, self.recording_protection = setup_reliability_monitoring(
            self, reliability_config
        )
        print("[DEBUG] Reliability monitoring setup completed"); sys.stdout.flush()
        
        # Start monitoring after GUI is fully initialized
        print("[DEBUG] Scheduling monitoring start"); sys.stdout.flush()
        self.after(1000, self._start_monitoring)  # Start 1 second after GUI loads
        self._last_cache_cleanup = time.time()

        # FIX: Start participant update thread AFTER GUI initialization to avoid fork() race condition
        print("[DEBUG] Scheduling participant update thread start"); sys.stdout.flush()
        self.after(100, self._start_participant_update_thread)  # Start early (100ms) since it's critical

        # Apply saved panel collapse state after GUI renders
        if self.left_panel_collapsed:
            print(f"[DEBUG] Scheduling panel state application: left={self.left_panel_collapsed}"); sys.stdout.flush()
            self.after(200, self._apply_initial_panel_state)

        # Mark initialization as complete to enable schedule_preview processing
        self._init_complete = True
        print("[DEBUG] GUI __init__ completed successfully"); sys.stdout.flush()

        # NOTE: Window visibility handling moved to AFTER __init__ completes
        # Removed premature update_idletasks() that was causing incomplete GUI display
        # Window will be shown properly before mainloop() starts

    def _should_log_for_camera(self, cam_idx: int, level: str = "debug") -> bool:
        """
        Check if logging should occur for this camera based on debug configuration.

        Args:
            cam_idx: Camera index to check
            level: Log level ("debug", "info", "warning", "error")

        Returns:
            True if logging should occur for this camera
        """
        # Always log errors/warnings if configured
        if level in ["error", "warning"]:
            debug_settings = self.config.get('debug_settings', {})
            if debug_settings.get(f'always_log_{level}s', True):
                return True

        # Check verbose camera filter
        debug_settings = self.config.get('debug_settings', {})
        verbose_cam = debug_settings.get('verbose_camera_index', 0)

        # None means log for all cameras
        if verbose_cam is None:
            return True

        # Otherwise only log for the specified camera
        return cam_idx == verbose_cam

    def after(self, ms, func=None, *args):
        """Override after to track callbacks for safe shutdown"""
        if self._shutdown_in_progress:
            return None
        
        callback_id = super().after(ms, func, *args)
        with self._callback_lock:
            self._pending_callbacks.append(callback_id)
        return callback_id
    
    def _cancel_all_pending_callbacks(self):
        """Cancel all pending Tkinter callbacks to prevent access violations"""
        print("[GUI] Cancelling all pending callbacks for safe shutdown")
        with self._callback_lock:
            for callback_id in self._pending_callbacks:
                try:
                    self.after_cancel(callback_id)
                except Exception as e:
                    # Callback might have already executed
                    pass
            self._pending_callbacks.clear()
        print(f"[GUI] Cancelled {len(self._pending_callbacks)} pending callbacks")
    
    def send_participant_command(self, command_type: str, payload: Dict):
        """
        Send participant-related commands to all active cameras via CommandBuffer system.
        
        Args:
            command_type: Type of command (e.g., 'register_track_participant')
            payload: Command payload data
        """
        try:
            if hasattr(self, 'camera_worker_manager'):
                active_cameras = []
                for camera_idx in range(4):  # Maximum 4 cameras
                    if self.camera_worker_manager.is_camera_active(camera_idx):
                        active_cameras.append(camera_idx)
                        command_id = self.camera_worker_manager.send_camera_command(
                            camera_idx, command_type, payload
                        )
                        if command_id:
                            self.command_stats['sent'] += 1
                            logger.debug(f"Sent {command_type} to camera {camera_idx} (id={command_id})")
                        else:
                            self.command_stats['failed'] += 1
                            logger.warning(f"Failed to send {command_type} to camera {camera_idx}")
                
                if active_cameras:
                    logger.debug(f"Sent {command_type} to {len(active_cameras)} active cameras: {active_cameras}")
                else:
                    logger.warning(f"No active cameras to send {command_type} command")
            else:
                logger.error("No camera_worker_manager available for sending commands")
        except Exception as e:
            self.command_stats['failed'] += 1
            logger.error(f"Error sending participant command {command_type}: {e}")

    # ===== Array-Based IPC Helper Methods =====

    def get_enrollment_state_name(self, participant_id: int) -> str:
        """
        Get enrollment state as string name.

        Args:
            participant_id: Participant ID (1-8)

        Returns:
            State name (e.g., 'enrolled', 'enrolling', 'unknown')
        """
        if participant_id < 1 or participant_id >= len(self.enrollment_state_array):
            return 'unknown'
        state_code = self.enrollment_state_array[participant_id]
        return self.ENROLLMENT_STATES_REVERSE.get(state_code, 'unknown')

    def set_enrollment_state(self, participant_id: int, state_name: str):
        """
        Set enrollment state by name.

        Args:
            participant_id: Participant ID (1-8)
            state_name: State name (e.g., 'enrolled', 'enrolling')
        """
        if participant_id < 1 or participant_id >= len(self.enrollment_state_array):
            logger.warning(f"Invalid participant_id {participant_id} for set_enrollment_state")
            return
        state_code = self.ENROLLMENT_STATES.get(state_name, 0)  # Default to 'unknown' (0)
        self.enrollment_state_array[participant_id] = state_code
        logger.debug(f"Set enrollment state for P{participant_id}: {state_name} (code={state_code})")

    def is_participant_locked(self, participant_id: int) -> bool:
        """
        Check if participant is locked.

        Args:
            participant_id: Participant ID (1-8)

        Returns:
            True if locked, False otherwise
        """
        if participant_id < 1 or participant_id >= len(self.lock_state_array):
            return False
        return bool(self.lock_state_array[participant_id])

    def is_participant_present(self, participant_id: int) -> bool:
        """
        Check if participant is present (detected in frame).

        Args:
            participant_id: Participant ID (1-8)

        Returns:
            True if present, False otherwise
        """
        if participant_id < 1 or participant_id >= len(self.participant_presence_array):
            return False
        return bool(self.participant_presence_array[participant_id])

    # ===== End Array-Based IPC Helper Methods =====

    def send_camera_specific_command(self, camera_idx: int, command_type: str, payload: Dict) -> bool:
        """
        Send command to specific camera with acknowledgment checking.

        Args:
            camera_idx: Camera index to send command to
            command_type: Type of command
            payload: Command payload data

        Returns:
            bool: True if command was sent and acknowledged successfully
        """
        # Skip command sending during shutdown to prevent error flood
        if getattr(self, '_shutdown_in_progress', False):
            return False

        try:
            if hasattr(self, 'camera_worker_manager'):
                command_id = self.camera_worker_manager.send_camera_command(
                    camera_idx, command_type, payload
                )
                
                if command_id:
                    self.command_stats['sent'] += 1
                    
                    # Check for acknowledgment with brief timeout
                    ack = self.camera_worker_manager.get_command_status(camera_idx, command_id)
                    if ack and ack.get('success'):
                        self.command_stats['acknowledged'] += 1
                        logger.debug(f"Command {command_type} acknowledged by camera {camera_idx}")
                        return True
                    else:
                        self.command_stats['failed'] += 1
                        return False
                else:
                    self.command_stats['failed'] += 1
                    logger.error(f"Failed to send {command_type} to camera {camera_idx}")
                    return False
            else:
                logger.error("No camera_worker_manager available")
                return False
        except Exception as e:
            self.command_stats['failed'] += 1
            logger.error(f"Error sending command to camera {camera_idx}: {e}")
            return False
    
    def _update_command_status_display(self):
        """Update GUI with command delivery statistics."""
        try:
            if hasattr(self, 'camera_worker_manager'):
                stats = self.camera_worker_manager.get_command_statistics()
                if stats and hasattr(self, 'status_label'):
                    # Calculate success rate
                    total_sent = self.command_stats['sent']
                    total_ack = self.command_stats['acknowledged']
                    success_rate = (total_ack / max(total_sent, 1)) * 100 if total_sent > 0 else 100
                    
                    # Update status display
                    command_status = f"Cmds: {total_sent} sent, {total_ack} ack ({success_rate:.1f}%)"
                    current_text = self.status_label.cget("text")
                    
                    # Append command status to existing status
                    if "Cmds:" not in current_text:
                        self.status_label.config(text=f"{current_text} | {command_status}")
                    else:
                        # Replace existing command status
                        parts = current_text.split(" | ")
                        non_cmd_parts = [p for p in parts if not p.startswith("Cmds:")]
                        new_text = " | ".join(non_cmd_parts + [command_status])
                        self.status_label.config(text=new_text)
        except Exception as e:
            logger.error(f"Error updating command status display: {e}")

    def _start_monitoring(self):
        """Start reliability monitoring after GUI is ready"""
        print("[DEBUG] _start_monitoring called"); import sys; sys.stdout.flush()
        print("[DEBUG] Starting reliability monitor"); sys.stdout.flush()
        self.reliability_monitor.start_monitoring()
        print("[DEBUG] Reliability monitor started"); sys.stdout.flush()
        
        # The recording protection is already checking recovery state on init
        # but we can add periodic state saves during recording
        print("[GUI] Reliability monitoring and recording protection active")
        print("[DEBUG] _start_monitoring completed"); sys.stdout.flush()

    def _start_participant_update_thread(self):
        """Single-person mode: No participant update thread needed"""
        pass

    def _process_participant_updates(self):
        """Single-person mode: No participant update processing needed"""
        pass

    def update_performance_stats(self):
        """Monitor and display performance statistics"""
        if not self.streaming:
            return


        # Check LSL helper health
        if hasattr(self, 'lsl_helper_proc') and self.lsl_helper_proc:
            if not self.lsl_helper_proc.is_alive():
                print("[GUI] WARNING: LSL helper process died - attempting to restart...")
                try:
                    # Re-initialize the LSL helper
                    self._initialize_lsl_helper()

                    # Notify that streaming has started again
                    if self.lsl_helper_proc.is_alive():
                        self.lsl_command_queue.put({
                            'type': 'streaming_started',
                            'max_participants': self.participant_count.get()
                        })
                        print("[GUI] LSL helper successfully restarted")
                    else:
                        print("[GUI] ERROR: Failed to restart LSL helper process")
                except Exception as e:
                    print(f"[GUI] ERROR: Exception while restarting LSL helper: {e}")

        # Collect stats from workers via control connection
        for idx, info in enumerate(self.frames):
            if info.get('control_conn'):
                info['control_conn'].send('get_stats')
        
        # Get reliability monitor stats
        if hasattr(self, 'reliability_monitor'):
            stats = self.reliability_monitor.get_stats()
            if stats.get('should_report'):
                print("\n[Reliability Monitor] Performance Report:")
                print(f"  Uptime: {stats['uptime']}")
                print(f"  Preview FPS: {stats['preview_fps']:.1f}")
                print(f"  Total frames: {stats['total_preview_updates']}")
                print(f"  Dropped frames: {stats['total_dropped_frames']} ({stats['drop_rate']:.1%})")
                print(f"  Queue overflows: {stats['total_queue_overflows']}")
                print(f"  Memory warnings: {stats['total_memory_warnings']}")
                print(f"  GUI freeze warnings: {stats['total_gui_freeze_warnings']}")
                print(f"  Emergency cleanups: {stats['total_emergency_cleanups']}")
                
        # Schedule next update
        self.after(5000, self.update_performance_stats) 
    
    def on_participant_count_change(self):
        """
        Fix 3: Handle participant count change.
        NOTE: In single-camera mode, participant count is fixed to 1.
        This method is kept for compatibility but returns early.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Cleanup 1: Participant count is fixed to 1 in single-camera mode - return early
        logger.info("[GUI] Participant count change ignored (fixed to 1 in single-camera mode)")
        return

        # Cleanup 1: Unreachable dead code removed (multi-camera logic no longer applicable)
        # Legacy code kept as reference only - contains invalid widget reference (self.cam_spin)

    def on_camera_count_change(self):
        """
        Fix 4: Handle camera count change.
        NOTE: In single-camera mode, camera count is fixed to 1.
        This method is kept for compatibility but returns early.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Cleanup 1: Camera count is fixed to 1 in single-camera mode - return early
        logger.info("[GUI] Camera count change ignored (fixed to 1 in single-camera mode)")
        return

        # Cleanup 1: Unreachable dead code removed (multi-camera logic no longer applicable)

    def refresh_camera_list(self):
        """
        Manually refresh camera enumeration.

        This forces camera re-discovery (ZMQ probing) and rebuilds the UI.
        Use this after starting/stopping Windows camera senders.
        """
        print("[GUI] Manual camera refresh requested by user")
        # Invalidate cache to force fresh discovery
        invalidate_camera_cache()
        print("[GUI] Camera cache invalidated")
        self._cameras_enumerated = False  # Force re-enumeration
        self.build_frames(force_refresh=True)
        print(f"[GUI] Camera refresh complete: {len(self.cams)} camera(s) available")

    def toggle_theme(self):
        """
        Toggle between light and dark Azure theme and save preference.
        """
        if self.current_theme is None:
            print("[GUI] Theme toggle ignored - Azure theme not loaded")
            return

        try:
            if self.current_theme == "light":
                self.tk.call("set_theme", "dark")
                self.current_theme = "dark"
                self.theme_toggle_btn.config(text="☀ Light Mode")
                print("[GUI] Switched to dark theme")
            else:
                self.tk.call("set_theme", "light")
                self.current_theme = "light"
                self.theme_toggle_btn.config(text="☾ Dark Mode")
                print("[GUI] Switched to light theme")

            # Bar canvas removed - metrics panel theme updates will be added in Phase 2
            theme_colors = self._get_theme_colors()

            # Recreate ECG strips with new theme colors
            if hasattr(self, 'ecg_strips') and self.ecg_strips:
                print(f"[GUI] Recreating ECG strips for {self.current_theme} theme")
                # Preserve buffer data and state
                old_buffers = self.ecg_buffers.copy() if hasattr(self, 'ecg_buffers') else {}
                old_heart_rates = self.ecg_heart_rates.copy() if hasattr(self, 'ecg_heart_rates') else {}

                # Destroy old strips
                if hasattr(self, 'ecg_strip_frame'):
                    self.ecg_strip_frame.destroy()

                # Recreate with new theme
                self._create_ecg_strips()

                # Restore buffer data
                self.ecg_buffers.update(old_buffers)
                self.ecg_heart_rates.update(old_heart_rates)

                # Set all strips as dirty to force immediate refresh
                for participant_id in self.ecg_buffers.keys():
                    self.ecg_dirty_flags[participant_id] = True

                print("[GUI] ECG strips recreated with new theme")

            # Save theme preference to config file
            try:
                self.config.set('gui_interface.theme.mode', self.current_theme)
                self.config.save()
                print(f"[GUI] Theme preference saved: {self.current_theme}")
            except Exception as save_error:
                print(f"[GUI] Warning: Could not save theme preference: {save_error}")
        except Exception as e:
            print(f"[GUI] Error toggling theme: {e}")

    def _toggle_left_panel(self):
        """Toggle left control panel collapse state with smooth animation."""
        if self.panel_animator.is_animating('left_panel'):
            return  # Debounce - ignore rapid clicks

        # Determine target width
        if not self.left_panel_collapsed:
            # Collapse to thin bar
            target_width = 30
            self._hide_panel_children(self.control_panel, except_widget=self.left_collapse_btn)
        else:
            # Expand to fixed width
            target_width = 200
            # Children will be shown after animation completes (prevents flicker)

        # Update state
        self.left_panel_collapsed = not self.left_panel_collapsed

        # Animate
        self.panel_animator.animate_width(
            self.control_panel,
            column_idx=0,
            target_width=target_width,
            duration_ms=250,
            on_complete=self._on_left_panel_toggle_complete,
            animation_id='left_panel'
        )

    def _toggle_right_panel(self):
        """Toggle right metrics panel collapse state with smooth animation."""
        if self.panel_animator.is_animating('right_panel'):
            return  # Debounce - ignore rapid clicks

        # Determine target width
        if not self.right_panel_collapsed:
            # Collapse to thin bar
            target_width = 30
            self._hide_panel_children(self.metrics_panel, except_widget=self.right_collapse_btn)
        else:
            # Expand to fixed width
            target_width = 300
            # Children will be shown after animation completes (prevents flicker)

        # Update state
        self.right_panel_collapsed = not self.right_panel_collapsed

        # Animate
        self.panel_animator.animate_width(
            self.metrics_panel,
            column_idx=2,
            target_width=target_width,
            duration_ms=250,
            on_complete=self._on_right_panel_toggle_complete,
            animation_id='right_panel'
        )

    def _hide_panel_children(self, panel, except_widget=None):
        """Hide all children of panel except specified widget and save layout info."""
        for child in panel.winfo_children():
            if child != except_widget:
                # Save original layout manager and parameters before hiding
                grid_info = child.grid_info()
                if grid_info:
                    # Child uses grid layout
                    self._panel_layout_info[child] = ('grid', grid_info)
                    child.grid_remove()
                else:
                    # Child uses pack layout
                    pack_info = child.pack_info()
                    if pack_info:
                        self._panel_layout_info[child] = ('pack', pack_info)
                        child.pack_forget()

    def _show_panel_children(self, panel):
        """Restore all children of panel using saved layout info."""
        for child in panel.winfo_children():
            if child in self._panel_layout_info:
                manager, info = self._panel_layout_info[child]
                try:
                    if manager == 'grid':
                        child.grid(**info)
                    else:  # pack
                        child.pack(**info)
                except tk.TclError as e:
                    print(f"[GUI] Warning: Could not restore widget layout: {e}")

    def _on_left_panel_toggle_complete(self):
        """Callback after left panel animation completes."""
        if not self.left_panel_collapsed:
            # Panel expanded - restore children AFTER animation (prevents flicker)
            self._show_panel_children(self.control_panel)

        # Update button icon
        self._update_collapse_button_icon(self.left_collapse_btn, 'left', self.left_panel_collapsed)

        # Save state to config
        self._save_panel_state()

    def _on_right_panel_toggle_complete(self):
        """Callback after right panel animation completes."""
        if not self.right_panel_collapsed:
            # Panel expanded - restore children AFTER animation (prevents flicker)
            self._show_panel_children(self.metrics_panel)

        # Update button icon
        self._update_collapse_button_icon(self.right_collapse_btn, 'right', self.right_panel_collapsed)

        # Save state to config
        self._save_panel_state()

    def _save_panel_state(self):
        """Save panel collapse state to config file."""
        try:
            # Update gui_layout section
            if 'gui_layout' not in self.config.config:
                self.config.config['gui_layout'] = {}

            self.config.config['gui_layout']['left_panel_collapsed'] = self.left_panel_collapsed
            # right_panel_collapsed will be added in Phase 2

            self.config.save()
            print(f"[GUI] Panel state saved: left={self.left_panel_collapsed}")
        except Exception as e:
            print(f"[GUI] Warning: Could not save panel state: {e}")

    def _update_collapse_button_icon(self, button, direction, is_collapsed):
        """Update button icon (◀/▶) based on collapse state."""
        if direction == 'left':
            button.config(text='▶' if is_collapsed else '◀')
        else:  # right
            button.config(text='◀' if is_collapsed else '▶')

    def _apply_initial_panel_state(self):
        """Apply saved panel collapse state after GUI initialization (no animation)."""
        try:
            if self.left_panel_collapsed:
                print("[DEBUG] Applying initial left panel collapsed state"); sys.stdout.flush()
                self._hide_panel_children(self.control_panel, except_widget=self.left_collapse_btn)
                self.grid_columnconfigure(0, minsize=30)
                self._update_collapse_button_icon(self.left_collapse_btn, 'left', True)

            # Right panel state will be added in Phase 2

            print("[DEBUG] Initial panel state applied successfully"); sys.stdout.flush()
        except Exception as e:
            print(f"[GUI] Warning: Could not apply initial panel state: {e}")

    def _validate_participant_name(self, name):
        """Validate participant name: alphanumeric + underscore only, non-empty.

        Args:
            name: The participant name to validate

        Returns:
            str: The validated name if valid, None otherwise
        """
        import re

        # Remove leading/trailing whitespace
        name = name.strip()

        # Check for empty name
        if not name:
            return None

        # Check for alphanumeric + underscore only
        if not re.match(r'^[A-Za-z0-9_]+$', name):
            return None

        return name

    def _is_system_idle(self):
        """Check if system is idle (not streaming and not recording).

        Returns:
            bool: True if system is idle, False otherwise
        """
        # Check streaming state
        if hasattr(self, 'streaming') and self.streaming:
            return False

        # Check recording state
        if hasattr(self, 'recording_active') and self.recording_active:
            return False

        return True

    def _on_participant_name_change(self, event=None):
        """Handle participant name change with validation and idle checking.

        Args:
            event: The tkinter event (FocusOut or Return key)
        """
        new_name = self.participant_name_var.get()

        # Validate the name
        validated_name = self._validate_participant_name(new_name)

        if validated_name is None:
            # Invalid name - show error and revert
            messagebox.showwarning(
                "Invalid Participant Name",
                "Participant name must contain only letters, numbers, and underscores, "
                "and cannot be empty.\n\nReverting to previous name."
            )
            # Revert to current value
            self.participant_name_var.set(self.single_participant_id)
            return

        # Check if name actually changed
        if validated_name == self.single_participant_id:
            return  # No change needed

        # Check if system is idle
        if not self._is_system_idle():
            # System is busy - show warning and revert
            messagebox.showwarning(
                "Cannot Change Name",
                "Participant name cannot be changed while streaming or recording is active.\n\n"
                "Please stop streaming/recording first, then try again."
            )
            # Revert to current value
            self.participant_name_var.set(self.single_participant_id)
            return

        # All checks passed - update the participant name
        old_name = self.single_participant_id
        self.single_participant_id = validated_name

        # Update participant_names dictionary
        if hasattr(self, 'participant_names'):
            self.participant_names[0] = validated_name

        print(f"[GUI] Participant name changed: {old_name} → {validated_name}")

        # Visual feedback - flash the entry to confirm change
        self.participant_name_entry.config(foreground='green')
        self.after(500, lambda: self.participant_name_entry.config(foreground='black'))

    def _show_loading_overlay(self, message="Discovering cameras..."):
        """
        Show animated loading overlay during async operations.

        Args:
            message: Status message to display
        """
        if self._loading_overlay is not None:
            # Already showing, just update message
            self._loading_label.config(text=message)
            return

        # Fix 2: Create semi-transparent overlay on main window (not self.container - that was removed)
        self._loading_overlay = tk.Frame(
            self,  # Use main window instead of deleted self.container
            bg='#f0f0f0',
            relief='raised',
            borderwidth=2
        )
        self._loading_overlay.place(relx=0.5, rely=0.5, anchor='center', width=400, height=150)

        # Spinner frame (using rotating text characters for animation)
        spinner_label = tk.Label(
            self._loading_overlay,
            text="◷",
            font=('DejaVu Sans', 36),
            bg='#f0f0f0'
        )
        spinner_label.pack(pady=(20, 10))

        # Status message
        self._loading_label = tk.Label(
            self._loading_overlay,
            text=message,
            font=('Arial', 12),
            bg='#f0f0f0',
            fg='#333333',
            wraplength=350
        )
        self._loading_label.pack(pady=(0, 10))

        # Subtitle (time estimate)
        subtitle = tk.Label(
            self._loading_overlay,
            text="This may take 10-30 seconds on first launch",
            font=('Arial', 9),
            bg='#f0f0f0',
            fg='#666666'
        )
        subtitle.pack(pady=(0, 20))

        # Start animation
        self._loading_animation_running = True
        self._loading_animation_frame = 0
        self._animate_loading_spinner(spinner_label)

        # Bring to front
        self._loading_overlay.lift()

    def _animate_loading_spinner(self, spinner_label):
        """
        Animate the loading spinner.

        Args:
            spinner_label: Label widget to animate
        """
        if not self._loading_animation_running:
            return

        # Rotating spinner characters (clock faces for cross-platform compatibility)
        spinner_chars = ['◷', '◶', '◵', '◴']  # Unicode clock symbols (U+25F7-U+25F4)
        char_index = self._loading_animation_frame % len(spinner_chars)
        spinner_label.config(text=spinner_chars[char_index])

        # Update loading message with animated dots
        if self._loading_label:
            base_text = self._loading_label.cget('text').split('\n')[0].rstrip('.')
            dots = '.' * ((self._loading_animation_frame % 4))
            self._loading_label.config(text=f"{base_text}{dots}")

        self._loading_animation_frame += 1

        # Schedule next frame (500ms for smooth animation)
        # FIX #6: Store after_id for proper cleanup
        if self._loading_animation_running:
            self._loading_animation_after_id = self.after(
                500, lambda: self._animate_loading_spinner(spinner_label)
            )

    def _hide_loading_overlay(self):
        """Hide and destroy the loading overlay."""
        # FIX #6: CRITICAL - Stop animation FIRST
        self._loading_animation_running = False

        # FIX #6: Cancel any pending animation callbacks
        if self._loading_animation_after_id:
            self.after_cancel(self._loading_animation_after_id)
            self._loading_animation_after_id = None

        # Now safe to destroy overlay
        if self._loading_overlay is not None:
            self._loading_overlay.destroy()
            self._loading_overlay = None
            self._loading_label = None
            self._loading_animation_frame = 0

    def _on_cameras_discovered(self, cameras, force_refresh):
        """
        Phase 2, Step 12.1: Callback when camera discovery completes successfully.
        Simplified for single-camera mode - populates dropdown instead of rebuilding frames.

        Args:
            cameras: List of discovered cameras
            force_refresh: Whether this was a forced refresh
        """
        logger.info(f"[GUI] Camera discovery completed: found {len(cameras)} camera(s)")

        self.cams = cameras  # Cache discovered cameras
        self._cameras_enumerated = True
        self._discovery_in_progress = False

        # Hide loading overlay
        self._hide_loading_overlay()

        # Populate camera dropdown (single-camera mode)
        self._populate_camera_dropdown()

        # Initialize LSL helper if needed
        if not hasattr(self, 'lsl_helper_proc') or self.lsl_helper_proc is None or not self.lsl_helper_proc.is_alive():
            self._initialize_lsl_helper()

    def _populate_camera_dropdown(self):
        """
        Phase 2, Step 12.2: Populate camera dropdowns with discovered cameras.
        Populates both main camera selector and laparoscopic camera selector.
        """
        if not self.cams:
            self.camera_combo['values'] = []
            self.camera_combo.set('')
            self.laparoscopic_camera_combo['values'] = []
            self.laparoscopic_camera_combo.set('')
            self.camera_status_label.config(
                text="No cameras found",
                foreground='gray'
            )
            logger.warning("No cameras discovered")
            return

        # Format: "0: Camera Name (1920x1080)"
        cam_vals = [f"{i}: {name}" for i, name, w, h in self.cams]

        # Populate both dropdowns with same camera list
        self.camera_combo['values'] = cam_vals
        self.laparoscopic_camera_combo['values'] = cam_vals

        # Auto-select first camera for main selector (visual pre-selection only)
        if cam_vals:
            self.camera_combo.set(cam_vals[0])
            # NOTE: Auto-start disabled - user must explicitly select camera to start feed
            # Uncomment line below to re-enable auto-start for testing:
            # self.after(100, self.on_camera_selected)

        logger.info(f"Populated both dropdowns with {len(cam_vals)} cameras")

    def _on_camera_discovery_failed(self, error_message):
        """
        Callback when camera discovery fails.

        Args:
            error_message: Error description
        """
        print(f"[GUI] Camera discovery FAILED: {error_message}")
        self._discovery_in_progress = False  # FIX #3: Clear flag

        # Hide loading overlay
        self._hide_loading_overlay()

        # Show error message to user
        import tkinter.messagebox as messagebox
        messagebox.showerror(
            "Camera Discovery Failed",
            f"Failed to discover cameras:\n\n{error_message}\n\n"
            "Please check:\n"
            "- Windows camera bridge is running (ZMQ mode)\n"
            "- Cameras are connected (V4L2 mode)\n"
            "- Network connectivity (ZMQ mode)"
        )

        # Set empty camera list to allow GUI to continue
        self.cams = []
        self._cameras_enumerated = True

        # Rebuild frames with empty camera list
        self.build_frames(force_refresh=False)

    def _show_camera_loading(self, frame_idx, camera_index):
        """
        Show loading indicator in camera preview slot.

        Args:
            frame_idx: Frame slot index
            camera_index: Camera device index
        """
        if frame_idx >= len(self.frames):
            return

        info = self.frames[frame_idx]
        canvas = info.get('canvas')

        if canvas:
            # Use the fancy overlay with spinner instead of simple text
            self._display_loading_overlay(
                canvas,
                f"Initializing Camera {camera_index}...",
                canvas_idx=frame_idx,  # Enable spinner animation
                transparent=True,      # Show live video underneath as soon as frames arrive
                opacity=0.75          # Semi-transparent overlay
            )

    def _on_camera_ready(self, frame_idx, camera_index, buffer_names, resolution):
        """
        Callback when camera initialization completes successfully.

        Args:
            frame_idx: Frame slot index
            camera_index: Camera device index
            buffer_names: Dictionary of shared memory buffer names
            resolution: (width, height) tuple from camera
        """
        print(f"[GUI] _on_camera_ready: frame_idx={frame_idx}, camera_index={camera_index}, resolution={resolution}")

        # FIX #4: Validate frame_idx is still valid (user may have rebuilt frames)
        if frame_idx >= len(self.frames):
            logger.warning(f"[GUI] Camera {camera_index} ready but frame_idx={frame_idx} invalid (only {len(self.frames)} frames exist)")
            logger.warning(f"[GUI] This likely means frames were rebuilt while camera was initializing")
            # Stop camera worker since it has nowhere to display
            self.camera_worker_manager.stop_camera(camera_index)
            return

        # Record which frame slot this camera occupies
        self.camera_to_frame_map[camera_index] = frame_idx

        # Store buffer names
        if not hasattr(self, 'actual_buffer_names'):
            self.actual_buffer_names = {}
        self.actual_buffer_names[camera_index] = buffer_names
        print(f"[GUI] SUCCESS: Captured buffer names for camera {camera_index}: {list(buffer_names.keys())}")

        # CRITICAL: Update coordinator registry so face recognition can discover buffers
        # Face recognition manager reads the PID file to find buffer names
        if hasattr(self, 'buffer_coordinator'):
            try:
                self.buffer_coordinator.update_coordinator_registry(camera_index, buffer_names)
                print(f"[GUI] Updated coordinator registry for camera {camera_index}")
            except Exception as e:
                logger.error(f"[GUI] Failed to update coordinator registry: {e}")
                import traceback
                traceback.print_exc()

        # NOTE: Status updates are handled by _process_camera_status_updates()
        # via CameraWorkerManager.process_status_updates() (see lines 5226-5390).
        # No direct CommandBuffer connection needed here - the manager handles
        # all status message routing through a single unified path.

        # Mark camera as active - store the actual process object for .is_alive() checks
        self.active_camera_procs[camera_index] = self.camera_worker_manager.workers[camera_index]

        # CRITICAL FIX: Update buffer coordinator resolution BEFORE notifying GUI worker
        # This ensures GUI worker uses correct resolution when calculating buffer offsets
        if resolution and hasattr(self, 'buffer_coordinator'):
            logger.info(f"[GUI] Setting buffer coordinator resolution for camera {camera_index}: {resolution}")
            self.buffer_coordinator.camera_resolutions[camera_index] = resolution
            # Also update local tracking
            self.camera_resolutions[camera_index] = resolution
        else:
            logger.warning(f"[GUI] No resolution provided for camera {camera_index}, using default")

        # Set initial resolution label state
        info = self.frames[frame_idx]
        resolution_label = info.get('resolution_label')

        if resolution_label and resolution:
            # Display detected resolution
            self._update_camera_resolution_display(camera_index, resolution)
            # Also update global resolution dropdown to match detected resolution
            self._update_resolution_dropdown_display(resolution)

        # Notify GUI processing worker with acknowledgment request (FIX #16)
        if hasattr(self, 'gui_processing_worker') and self.gui_processing_worker and self.gui_processing_worker.is_alive():
            try:
                self.processing_control_queue.put({
                    'type': 'buffer_names',
                    'actual_buffer_names': self.actual_buffer_names.copy(),
                    'camera_resolutions': self.buffer_coordinator.camera_resolutions.copy() if hasattr(self, 'buffer_coordinator') else {},
                    'request_ack': True,  # FIX #16: Request acknowledgment
                    'camera_index': camera_index
                })
                print(f"[GUI] Notified GUI worker of camera {camera_index} buffer names")

                # FIX #16: Start preview after brief delay to ensure worker processes command
                self.after(200, lambda: self._start_preview_for_camera(camera_index))
            except Exception as e:
                logger.error(f"[GUI] Failed to notify GUI worker: {e}")
                # Fallback: start preview anyway
                self._start_preview_for_camera(camera_index)
        else:
            # No worker, start preview immediately
            self._start_preview_for_camera(camera_index)

        # Start face recognition process if not already done
        self._start_face_recognition_if_needed()

        print(f"[GUI] Camera {camera_index} startup complete")

    def _on_camera_failed(self, frame_idx, camera_index, error_message):
        """
        Callback when camera initialization fails.

        Args:
            frame_idx: Frame slot index
            camera_index: Camera device index
            error_message: Error description
        """
        print(f"[GUI] _on_camera_failed: camera_index={camera_index}, error={error_message}")

        # Show error in canvas
        if frame_idx < len(self.frames):
            info = self.frames[frame_idx]
            canvas = info.get('canvas')

            if canvas:
                canvas.delete("all")
                canvas.create_text(
                    canvas.winfo_width() // 2 if canvas.winfo_width() > 1 else 320,
                    canvas.winfo_height() // 2 if canvas.winfo_height() > 1 else 180,
                    text=f"Camera {camera_index} Failed\n{error_message}",
                    fill="red",
                    font=("Arial", 12),
                    tags="error"
                )

        # Show error dialog
        import tkinter.messagebox as messagebox
        messagebox.showerror(
            "Camera Startup Failed",
            f"Failed to initialize camera {camera_index}:\n\n{error_message}\n\n"
            "Please check:\n"
            "- Camera is connected and accessible\n"
            "- No other application is using the camera\n"
            "- Windows camera sender is running (ZMQ mode)"
        )

    def on_resolution_change(self):
        """
        Called when the user picks a new resolution from the dropdown.
        Updates config for all cameras and restarts running cameras with new resolution.
        """
        # GUARD: Prevent resolution changes in ZMQ mode (should not happen since dropdown is disabled)
        zmq_enabled = self.config.get('zmq_camera_bridge', {}).get('enabled', False)
        if zmq_enabled:
            logger.warning("[GUI] Resolution change ignored - ZMQ mode uses auto-detected resolution")
            print("[GUI] ⚠️  Resolution change blocked - ZMQ mode uses auto-detected resolution from stream")
            return

        res_label = self.res_choice.get()
        new_resolution = self.res_map[res_label]
        print(f"[GUI] Resolution changed to {res_label} ({new_resolution})")

        # Update config for all cameras
        max_cameras = self.config.get('startup_mode', {}).get('camera_count', 4)
        for cam_idx in range(max_cameras):
            cam_key = f'camera_{cam_idx}'
            self.config.set(f'camera_settings.{cam_key}.width', new_resolution[0])
            self.config.set(f'camera_settings.{cam_key}.height', new_resolution[1])
            print(f"[GUI] Updated config for {cam_key}: {new_resolution[0]}x{new_resolution[1]}")

        # Also update global resolution setting (for backward compatibility)
        self.config.set('camera_settings.resolution', res_label)

        # Update UI labels
        for info in self.frames:
            if info.get('meta_label'):
                fps = self.desired_fps.get()
                info['meta_label'].config(text=f"Camera → {info['camera_index']} (Capture: {new_resolution[0]}x{new_resolution[1]}@{fps}fps)")

        # Restart running cameras with new resolution
        running_cameras = [cam_idx for cam_idx in range(max_cameras)
                          if self.camera_worker_manager.is_camera_active(cam_idx)]

        if running_cameras:
            print(f"[GUI] Restarting {len(running_cameras)} running cameras with new resolution: {running_cameras}")
            self._restart_cameras_with_new_resolution(running_cameras, new_resolution)
        else:
            print(f"[GUI] No running cameras - new resolution will apply at next camera start")

    def _update_resolution_selector_state(self):
        """
        Update resolution selector based on ZMQ bridge mode.
        - If ZMQ enabled: disable dropdown, show auto-detected label
        - If ZMQ disabled: enable dropdown, hide label
        """
        zmq_enabled = self.config.get('zmq_camera_bridge', {}).get('enabled', False)

        if zmq_enabled:
            # Disable resolution dropdown
            self.res_menu.config(state='disabled')

            # Get detected resolution from config or default
            zmq_settings = self.config.get('zmq_camera_bridge', {})
            detected_res = zmq_settings.get('detected_resolution')
            if detected_res is None:
                detected_res = zmq_settings.get('default_resolution', [1280, 720])

            # Update and show auto-detected label
            self.auto_res_label.config(text=f"Auto-detected: {detected_res[0]}x{detected_res[1]}")
            self.auto_res_label.pack(anchor='w', pady=(0, 10))

            # Update all camera configs with detected resolution
            self._apply_detected_resolution_to_all_cameras(detected_res)

            # Update resolution dropdown to show closest match (for visual consistency)
            self._update_resolution_dropdown_display(detected_res)

            logger.info(f"[GUI] ZMQ mode active - resolution selector disabled (auto-detected: {detected_res[0]}x{detected_res[1]})")
        else:
            # Enable resolution dropdown
            self.res_menu.config(state='readonly')

            # Hide auto-detected label
            self.auto_res_label.pack_forget()

            logger.info("[GUI] V4L2 mode active - resolution selector enabled")

    def _apply_detected_resolution_to_all_cameras(self, resolution):
        """
        Apply detected resolution to all camera configs.

        Args:
            resolution: List/tuple of [width, height]
        """
        max_cameras = self.config.get('startup_mode', {}).get('camera_count', 4)

        for cam_idx in range(max_cameras):
            cam_key = f'camera_{cam_idx}'
            self.config.set(f'camera_settings.{cam_key}.width', resolution[0])
            self.config.set(f'camera_settings.{cam_key}.height', resolution[1])

        logger.info(f"[GUI] Applied detected resolution {resolution[0]}x{resolution[1]} to all cameras")

    def _update_resolution_dropdown_display(self, resolution):
        """
        Update resolution dropdown to show closest matching preset.

        Args:
            resolution: List/tuple of [width, height]
        """
        # Find closest matching preset resolution
        width, height = resolution[0], resolution[1]

        for label, (preset_w, preset_h) in self.res_map.items():
            if preset_w == width and preset_h == height:
                self.res_choice.set(label)
                logger.info(f"[GUI] Updated resolution dropdown to show: {label}")
                return

        # No exact match - keep current selection but log warning
        logger.warning(f"[GUI] No preset resolution matches detected {width}x{height}")

    def _restart_cameras_with_new_resolution(self, camera_indices, new_resolution):
        """
        Restart cameras with new resolution by:
        1. Stopping camera worker
        2. Destroying old buffers
        3. Starting camera worker (creates new buffers with new size)
        4. Reconnecting display system

        Args:
            camera_indices: List of camera indices to restart
            new_resolution: Tuple of (width, height)
        """
        import time

        for cam_idx in camera_indices:
            try:
                print(f"\n[GUI] === Restarting Camera {cam_idx} with {new_resolution[0]}x{new_resolution[1]} ===")

                # Step 1: Stop camera worker
                print(f"[GUI] Step 1: Stopping camera worker {cam_idx}...")
                self.camera_worker_manager.stop_camera(cam_idx)

                # Step 2: Stop preview for this camera
                print(f"[GUI] Step 2: Stopping preview for camera {cam_idx}...")
                self._stop_preview_for_camera(cam_idx)

                # Clean up process tracking
                if hasattr(self, 'active_camera_procs') and cam_idx in self.active_camera_procs:
                    del self.active_camera_procs[cam_idx]

                # Step 3: Destroy old buffers
                print(f"[GUI] Step 3: Destroying old buffers for camera {cam_idx}...")
                self.buffer_coordinator.destroy_camera_buffers(cam_idx)

                # Small delay to ensure cleanup completes
                time.sleep(0.1)

                # Step 4: Update coordinator resolution tracking (already done in destroy, but ensure consistency)
                self.buffer_coordinator.camera_resolutions[cam_idx] = tuple(new_resolution)
                print(f"[GUI] Step 4: Updated buffer coordinator resolution tracking: {new_resolution}")

                # Step 5: Restart camera (will create new buffers with new size)
                print(f"[GUI] Step 5: Starting camera worker {cam_idx} with new resolution...")
                success, buffer_names = self.camera_worker_manager.start_camera(cam_idx)

                if success and buffer_names:
                    print(f"[GUI] ✅ Camera {cam_idx} restarted successfully!")
                    print(f"[GUI] New buffer names: {list(buffer_names.keys())}")

                    # Step 6: Update actual_buffer_names tracking
                    if not hasattr(self, 'actual_buffer_names'):
                        self.actual_buffer_names = {}
                    self.actual_buffer_names[cam_idx] = buffer_names

                    # Step 7: Reconnect display system
                    print(f"[GUI] Step 6: Reconnecting display system for camera {cam_idx}...")
                    self._reconnect_display_buffer(cam_idx, buffer_names['gui'])

                    # Step 8: Restart preview
                    print(f"[GUI] Step 7: Restarting preview for camera {cam_idx}...")
                    # Preview will start automatically when camera becomes ready

                    print(f"[GUI] === Camera {cam_idx} restart complete! ===\n")
                else:
                    print(f"[GUI] ❌ Failed to restart camera {cam_idx}")

            except Exception as e:
                print(f"[GUI] ❌ Error restarting camera {cam_idx}: {e}")
                import traceback
                traceback.print_exc()

    def _reconnect_display_buffer(self, cam_idx, new_buffer_name):
        """
        Reconnect display system to new buffer after resolution change.
        Sends updated buffer names and resolutions to GUI processing worker.

        Args:
            cam_idx: Camera index
            new_buffer_name: New GUI buffer name
        """
        try:
            # Check if GUI processing worker is running
            if not hasattr(self, 'gui_processing_worker') or not self.gui_processing_worker or not self.gui_processing_worker.is_alive():
                print(f"[GUI] GUI processing worker not running - will connect when started")
                return

            # Send updated buffer names and resolutions to worker
            if hasattr(self, 'processing_control_queue'):
                self.processing_control_queue.put({
                    'type': 'buffer_names',
                    'actual_buffer_names': self.actual_buffer_names.copy(),
                    'camera_resolutions': self.buffer_coordinator.camera_resolutions.copy()
                })
                print(f"[GUI] Sent updated buffer names and resolutions to GUI processing worker")
                print(f"[GUI] Camera {cam_idx} will reconnect to new buffer: {new_buffer_name}")
            else:
                print(f"[GUI] Warning: processing_control_queue not available")

        except Exception as e:
            print(f"[GUI] Error reconnecting display buffer for camera {cam_idx}: {e}")
            import traceback
            traceback.print_exc()

    def _is_wsl(self):
        """Detect if running in WSL2 environment."""
        import platform
        import os
        return "WSL_INTEROP" in os.environ or "microsoft" in platform.release().lower()

    def _toggle_fullscreen(self, *_):
        """Toggle fullscreen mode (F11 keybinding).

        Provides fallback mechanism if maximize button doesn't work well in WSLg.
        """
        fs = bool(self.attributes('-fullscreen'))
        self.attributes('-fullscreen', not fs)

    def on_closing(self):
        """Save configuration before closing"""
        print("XXXXXX DEBUG: on_closing() method called! XXXXXX")
        import sys
        sys.stdout.flush()
        import traceback
        traceback.print_stack()
        sys.stdout.flush()

        # First, shutdown all processes (camera workers, recordings, etc.)
        if hasattr(self, '_shutdown_all_processes'):
            print("[GUI] Calling comprehensive shutdown sequence...")
            self._shutdown_all_processes()

        # Stop reliability monitoring
        if hasattr(self, 'reliability_monitor'):
            self.reliability_monitor.stop_monitoring()
        
        # Stop participant update processor
        if hasattr(self, 'participant_update_queue'):
            try:
                self.participant_update_queue.put(None)
            except:
                pass
        
        # Stop face recognition integration
        if hasattr(self, 'face_recognition') and self.face_recognition is not None:
            try:
                self.face_recognition.stop()
                print("[CLEANUP] Face recognition integration stopped")
            except Exception as e:
                print(f"[CLEANUP] Error stopping face recognition: {e}")

        # Stop enrollment worker thread
        if hasattr(self, 'global_participant_manager') and self.global_participant_manager is not None:
            try:
                self.global_participant_manager.stop_enrollment_worker(timeout=5.0)
                print("[CLEANUP] Enrollment worker thread stopped")
            except Exception as e:
                print(f"[CLEANUP] Error stopping enrollment worker: {e}")

        # Clean up drawing manager
        if hasattr(self, 'drawingmanager'):
            for idx in range(len(self.frames)):
                self.drawingmanager.cleanup_canvas(idx)
        
        # Stop optimized display loop
        self._optimized_display_active = False
        
        # Shutdown fast display if active
        if hasattr(self, '_using_fast_display') and self._using_fast_display:
            try:
                self.shutdown_fast_display()
                print("[CLEANUP] Fast display mode shutdown")
            except Exception as e:
                print(f"[CLEANUP] Error shutting down fast display: {e}")
        
        # Save current settings
        self.config.set('camera_settings.target_fps', self.desired_fps.get())
        self.config.set('camera_settings.resolution', self.res_choice.get())
        self.config.set('startup_mode.participant_count', self.participant_count.get())
        self.config.set('startup_mode.camera_count', self.camera_count.get())
        self.config.set('startup_mode.enable_mesh', self.enable_mesh.get())

        # Clean up shared memory - Delete numpy view first
        if hasattr(self, 'corr_array'):
            try:
                del self.corr_array
            except:
                pass
                
        if hasattr(self, 'correlation_buffer'):
            try:
                self.correlation_buffer.close()
                self.correlation_buffer.unlink()
            except:
                pass
        
        # Stop LSL helper
        if hasattr(self, 'lsl_helper_proc') and self.lsl_helper_proc is not None and self.lsl_helper_proc.is_alive():
            self.lsl_command_queue.put({'type': 'stop'})
            self.lsl_helper_proc.terminate()
            self.lsl_helper_proc.join(timeout=1.0)
        
        # Clean up GUI buffer manager
        if hasattr(self, 'gui_buffer_manager'):
            try:
                # REMOVED: self.gui_buffer_manager.cleanup_all()
                # DisplayBufferManager handles its own cleanup
                print("[CLEANUP] GUI buffer manager cleaned up")
            except Exception as e:
                print(f"[CLEANUP] Error cleaning GUI buffer manager: {e}")
        
        # Clean up buffer coordinator (will clean up all shared memory)
        if hasattr(self, 'buffer_coordinator'):
            try:
                self.buffer_coordinator.cleanup_all_buffers()
                print("[CLEANUP] Buffer coordinator cleaned up")
            except Exception as e:
                print(f"[CLEANUP] Error cleaning buffer coordinator: {e}")
        
        self.destroy()

    def on_mesh_toggle(self):
        """Notify each worker to turn raw‐mesh pushing on/off."""
        val = self.enable_mesh.get()
        print(f"[GUI] Mesh toggle changed to: {val}")

        # Save to config file immediately (bidirectional: GUI ↔ Config)
        self.config.set('startup_mode.enable_mesh', val)
        print(f"[GUI] ✅ Saved enable_mesh={val} to config file")

        # Notify GUI processing worker
        if hasattr(self, 'processing_control_queue'):
            self.processing_control_queue.put({
                'type': 'set_mesh',
                'enabled': val
            })
            print(f"[GUI] Sent mesh toggle → {val} to GUI processing worker")

        # Cleanup 2: Notify LSL process about mesh state change (single-camera mode)
        # CRITICAL FIX: Send to command_queue (not data_queue) to prevent race condition
        # Command queue is processed before data queue, ensuring config_update happens before force_recreate_streams
        if hasattr(self, 'lsl_command_queue'):
            # Cleanup 2: Simplified for single-camera mode (camera_index always 0)
            cam_idx = 0
            # Check if camera process is alive (reliable check, not dependent on status)
            if cam_idx in self.active_camera_procs and self.active_camera_procs[cam_idx].is_alive():
                self.lsl_command_queue.put({
                    'type': 'config_update',
                    'camera_index': cam_idx,
                    'mesh_enabled': val
                })
                print(f"[GUI] Sent mesh config_update for single camera to LSL process (mesh={val})")

        # If streaming, force recreation of all streams
        if self.streaming and hasattr(self, 'lsl_command_queue'):
            # Send a special command to force recreation
            self.lsl_command_queue.put({
                'type': 'force_recreate_streams',
                'mesh_enabled': val
            })

    def on_grid_debug_toggle(self):
        """Handle grid debug visualization toggle."""
        enabled = self.grid_debug_enabled.get()
        logger.info(f"[GUI] 🔘 Grid debug toggle: {'ENABLED' if enabled else 'DISABLED'}")

        # Write flag to all active camera detection buffers
        if hasattr(self, 'camera_worker_manager'):
            import numpy as np
            from core.buffer_management.layouts import DetectionBufferLayout
            import multiprocessing.shared_memory as shm

            # Get layout for detection buffer
            layout = DetectionBufferLayout(max_faces=8)

            # Write to each active camera's detection buffer
            # Access detection buffer from CameraWorkerManager.shared_memories (correct location)
            for cam_idx in self.camera_worker_manager.workers.keys():
                try:
                    # Check if camera has shared memories connected
                    if cam_idx not in self.camera_worker_manager.shared_memories:
                        logger.warning(f"[GUI] Camera {cam_idx} not in shared_memories")
                        continue

                    shared_mem = self.camera_worker_manager.shared_memories[cam_idx]

                    # Check if detection buffer is connected
                    if 'detection' not in shared_mem:
                        logger.warning(f"[GUI] No detection buffer for camera {cam_idx}")
                        continue

                    det_buf_name = shared_mem['detection']['name']

                    # Get shared memory handle and write grid debug flag
                    det_shm = shm.SharedMemory(name=det_buf_name)

                    # Write flag at offset 32 (grid_debug_enabled_offset)
                    flag_view = np.ndarray((1,), dtype=np.int32,
                                          buffer=det_shm.buf[layout.grid_debug_enabled_offset:
                                                            layout.grid_debug_enabled_offset + 4])
                    flag_view[0] = 1 if enabled else 0

                    det_shm.close()  # Close handle (doesn't unlink the buffer)
                    logger.info(f"[GUI] ✅ Set grid_debug_enabled={enabled} for camera {cam_idx} (buffer: {det_buf_name})")

                except Exception as e:
                    logger.warning(f"[GUI] Failed to set grid debug for camera {cam_idx}: {e}")

    def on_record_toggle(self):
        """Handle recording toggle state change"""
        enabled = self.record_video.get()
        state = 'normal' if enabled else 'disabled'
        
        # Enable/disable recording controls
        self.dir_entry.config(state=state)
        self.name_entry.config(state=state)
        
        # Only enable record now button if we have active cameras and not streaming
        if enabled and not self.streaming:
            active_workers = [info for info in self.frames if info.get('proc') and info['proc'].is_alive()]
            if active_workers:
                self.record_now_btn.config(state='normal')
            else:
                self.record_now_btn.config(state='disabled')
        else:
            self.record_now_btn.config(state='disabled')
        
        # Save preference
        self.config.set('video_recording.enabled', enabled)
        
        # Update browse button state
        for widget in self.dir_entry.master.winfo_children():
            if isinstance(widget, ttk.Button) and widget.cget('text') == 'Browse':
                widget.config(state=state)

    def get_next_global_frame_id(self):
        """
        Get the next global frame ID for synchronized recording.
        Thread-safe counter shared across all cameras (pose + laparoscopic).

        Returns:
            int: Next sequential frame ID (1-indexed)
        """
        with self.global_frame_counter_lock:
            self.global_frame_counter += 1
            return self.global_frame_counter

    def on_record_with_lsl_toggle(self):
        """
        Handle toggle of 'Start recording with LSL stream' checkbox.
        Saves the setting to configuration.
        """
        enabled = self.record_with_lsl_var.get()
        self.config.set('video_recording.auto_start_with_lsl', enabled)
        logger.info(f"[GUI] LSL-synchronized recording {'enabled' if enabled else 'disabled'}")

    def _auto_start_lsl_synchronized_recording(self):
        """
        Auto-start video recording when LSL streaming begins.
        Called automatically when 'Start recording with LSL stream' toggle is enabled.
        """
        # Reset global frame counter for synchronized start
        with self.global_frame_counter_lock:
            self.global_frame_counter = 0
        logger.info("[LSL] Reset global frame counter for synchronized recording")

        # Check if we have active workers
        active_workers = [info for info in self.frames if info.get('proc') and info['proc'].is_alive()]
        if not active_workers:
            logger.warning("[LSL] Cannot auto-start recording: No active cameras")
            return

        # Check if canvases have content
        has_content = False
        for info in active_workers:
            canvas = info['canvas']
            if canvas.find_all():  # Check if canvas has any items
                has_content = True
                break

        if not has_content:
            logger.warning("[LSL] Cannot auto-start recording: No video content yet")
            # Retry after another delay
            self.after(500, self._auto_start_lsl_synchronized_recording)
            return

        # Enable recording buttons
        self.record_now_btn.config(state='disabled')
        self.stop_record_btn.config(state='normal')

        # Start recording
        self._start_video_recording(active_workers)
        logger.info("[LSL] Video recording auto-started (synchronized with LSL stream)")

        # Update status
        self.record_status.config(text="Recording (LSL-synchronized)", foreground='green')

    def toggle_immediate_recording(self):
        """Start recording immediately without LSL stream"""
        # Check if we have active workers
        active_workers = [info for info in self.frames if info.get('proc') and info['proc'].is_alive()]
        if not active_workers:
            messagebox.showwarning("No Active Cameras", "Please select cameras first")
            return
        
        # Check if canvases have content
        has_content = False
        for info in active_workers:
            canvas = info['canvas']
            if canvas.find_all():  # Check if canvas has any items
                has_content = True
                break
        
        if not has_content:
            messagebox.showwarning("No Video Content", "Please wait for video feed to start before recording")
            return
        
        self.record_now_btn.config(state='disabled')
        self.stop_record_btn.config(state='normal')
        self._start_video_recording(active_workers)

    def stop_immediate_recording(self):
        """Stop immediate recording"""
        self.record_now_btn.config(state='normal' if self.record_video.get() else 'disabled')
        self.stop_record_btn.config(state='disabled')
        self._stop_video_recording()
    
    def _update_participant_display(self):
        """Update GUI display after participant recovery"""
        # Force refresh of participant tracking display
        # This ensures recovered participants are shown correctly
        if hasattr(self, 'global_participant_manager'):
            # Trigger any display updates needed
            # The normal display loop will pick up the changes
            pass

    def update_participant_names(self):
        """Periodically update participant names from entry fields and command status"""
        self.after(1000, self.update_participant_names)

        # Update command status display
        self._update_command_status_display()

        # LOCK SYSTEM: Update lock button states based on enrollment and lock status
        if hasattr(self, 'participant_lock_buttons'):
            self.update_lock_button_states()

        # Update participant names from entries
        if hasattr(self, 'participant_entries'):
            names_changed = False
            for idx, entry in enumerate(self.participant_entries):
                name = entry.get().strip()
                # CRITICAL FIX: Strip any locked status text before storing
                name = name.replace(" (Locked, Absent)", "").replace(" (Locked)", "").strip()
                old_name = self.participant_names.get(idx, f"P{idx + 1}")
                if name:
                    if name != old_name:
                        names_changed = True
                    self.participant_names[idx] = name
                else:
                    if f"P{idx + 1}" != old_name:
                        names_changed = True
                    self.participant_names[idx] = f"P{idx + 1}"

            # Update the global participant manager with current names
            if hasattr(self, 'global_participant_manager') and self.global_participant_manager is not None:
                self.global_participant_manager.set_participant_names(self.participant_names)

            # If streaming and names changed, update all fusion processes
            if self.streaming and names_changed:
                # Get participant names from global manager if available, otherwise use local names
                participant_names = (
                    self.global_participant_manager.participant_names.copy()
                    if hasattr(self, 'global_participant_manager') and self.global_participant_manager is not None
                    else self.participant_names.copy()
                )
                for cam_idx, pipe in self.participant_mapping_pipes.items():
                    try:
                        pipe.send({
                            'type': 'participant_names',
                            'names': participant_names
                        })
                    except:
                        pass

    def _update_metrics_panel(self):
        """Update pose metrics panel with latest calculated metrics from GUI processing worker."""
        # Reschedule next update (10 FPS = 100ms)
        if not self._shutdown_in_progress:
            self.after(100, self._update_metrics_panel)

        # Get metrics from GUI processing worker
        if not hasattr(self, 'gui_processing_worker') or not self.gui_processing_worker:
            return

        try:
            # Get currently selected camera index
            cam_idx = self._current_camera_index
            if cam_idx is None:
                # No camera selected - show placeholders
                self.head_pitch_label.config(text="Pitch: --")
                self.head_yaw_label.config(text="Yaw: --")
                self.head_roll_label.config(text="Roll: --")
                self.neck_angle_label.config(text="Neck Flexion: --")
                self.shoulder_left_label.config(text="Left: --")
                self.shoulder_right_label.config(text="Right: --")
                self.fps_label.config(text="FPS: --")
                self.latency_label.config(text="Latency: --")
                return

            # Get metrics from cache (populated by pose_data_queue)
            cache_entry = self._cached_pose_data.get(cam_idx)
            pose_metrics_dict = cache_entry.get('metrics') if cache_entry else None

            # Convert dict back to PoseMetrics object if available
            if pose_metrics_dict:
                pose_metrics = PoseMetrics(**pose_metrics_dict)
            else:
                pose_metrics = None

            if pose_metrics is None:
                # No metrics available - show placeholders
                self.head_pitch_label.config(text="Pitch: --")
                self.head_yaw_label.config(text="Yaw: --")
                self.head_roll_label.config(text="Roll: --")
                self.neck_angle_label.config(text="Neck Flexion: --")
                self.shoulder_left_label.config(text="Left: --")
                self.shoulder_right_label.config(text="Right: --")
                self.fps_label.config(text="FPS: --")
                self.latency_label.config(text="Latency: --")
                return

            # Update head orientation labels
            if pose_metrics.head_pitch is not None:
                self.head_pitch_label.config(text=f"Pitch: {pose_metrics.head_pitch:+.1f}°")
            else:
                self.head_pitch_label.config(text="Pitch: --")

            if pose_metrics.head_yaw is not None:
                self.head_yaw_label.config(text=f"Yaw: {pose_metrics.head_yaw:+.1f}°")
            else:
                self.head_yaw_label.config(text="Yaw: --")

            if pose_metrics.head_roll is not None:
                self.head_roll_label.config(text=f"Roll: {pose_metrics.head_roll:+.1f}°")
            else:
                self.head_roll_label.config(text="Roll: --")

            # Update neck angle label
            if pose_metrics.neck_flexion_angle is not None:
                self.neck_angle_label.config(text=f"Neck Flexion: {pose_metrics.neck_flexion_angle:+.1f}°")
            else:
                self.neck_angle_label.config(text="Neck Flexion: --")

            # Update sagittal neck angle label
            if pose_metrics.neck_sagittal_angle is not None:
                self.neck_sagittal_label.config(text=f"Sagittal Angle: {pose_metrics.neck_sagittal_angle:+.1f}°")
            else:
                self.neck_sagittal_label.config(text="Sagittal Angle: --")

            # Update forward head translation label
            if pose_metrics.forward_head_translation is not None:
                # Display in cm for better readability
                fh_cm = pose_metrics.forward_head_translation * 100
                self.forward_head_label.config(text=f"Forward Head: {fh_cm:+.1f} cm")
            else:
                self.forward_head_label.config(text="Forward Head: --")

            # Update shoulder elevation labels (Enhanced v6.0 - Multi-Component System)
            # Format: "Left: E:1.48 T:1.62 C:+0.12"
            # E = Ear-shoulder ratio (lower = elevated)
            # T = Torso height ratio (higher = elevated)
            # C = Composite score (positive = elevated, 0 = neutral)
            if pose_metrics.shoulder_elevation_left is not None:
                ear_l = pose_metrics.shoulder_elevation_left
                torso_l = pose_metrics.torso_height_left if pose_metrics.torso_height_left is not None else 0.0
                comp_l = pose_metrics.shoulder_composite_left if pose_metrics.shoulder_composite_left is not None else 0.0
                self.shoulder_left_label.config(
                    text=f"Left:  E:{ear_l:.2f}  T:{torso_l:.2f}  C:{comp_l:+.2f}"
                )
            else:
                self.shoulder_left_label.config(text="Left: --")

            if pose_metrics.shoulder_elevation_right is not None:
                ear_r = pose_metrics.shoulder_elevation_right
                torso_r = pose_metrics.torso_height_right if pose_metrics.torso_height_right is not None else 0.0
                comp_r = pose_metrics.shoulder_composite_right if pose_metrics.shoulder_composite_right is not None else 0.0
                self.shoulder_right_label.config(
                    text=f"Right: E:{ear_r:.2f}  T:{torso_r:.2f}  C:{comp_r:+.2f}"
                )
            else:
                self.shoulder_right_label.config(text="Right: --")

            # Update performance metrics (FPS and latency)
            # Get from cache (populated by pose_data_queue)
            perf_metrics = cache_entry.get('performance') if cache_entry else None
            if perf_metrics:
                # Display detection and pose FPS
                det_fps = perf_metrics.get('detection_fps', 0.0)
                pose_fps = perf_metrics.get('pose_fps', 0.0)

                # Format FPS display: "Detection: X fps | Pose: Y fps"
                if pose_fps > 0:
                    fps_text = f"Detection: {det_fps:.1f} fps | Pose: {pose_fps:.1f} fps"
                else:
                    fps_text = f"Detection: {det_fps:.1f} fps | Pose: --"
                self.fps_label.config(text=fps_text)

                # Display latency
                latency_ms = perf_metrics.get('latency_ms', 0.0)
                if latency_ms > 0:
                    self.latency_label.config(text=f"Latency: {latency_ms:.1f} ms")
                else:
                    self.latency_label.config(text="Latency: --")
            else:
                # No performance metrics available
                self.fps_label.config(text="FPS: --")
                self.latency_label.config(text="Latency: --")

        except Exception as e:
            logger.error(f"[METRICS UPDATE] Error updating metrics panel: {e}")
            import traceback
            traceback.print_exc()

    def on_landmarker_result(self, idx, result):
        """
        Process face blendshape results from MediaPipe landmarker.
        """
        info = self.frames[idx]
        if result.face_blendshapes:
            scores = [cat.score for cat in result.face_blendshapes[0]]
        else:
            scores = []
        scores += [0.0] * (52 - len(scores))

        # Fix 1: Defensive check for participant key (single-camera mode doesn't use this)
        if 'participant' in info and hasattr(info['participant'], 'last_blend_scores'):
            self.last_scores = list(info['participant'].last_blend_scores)

        if self.blend_labels is None and result.face_blendshapes:
            self.blend_labels = [cat.category_name for cat in result.face_blendshapes[0]]
        info['detect_count'] = info.get('detect_count', 0) + (1 if any(scores) else 0)

    def build_frames(self, force_refresh=False):
        """
        Phase 2, Step 9.2: Simplified for single-camera mode.
        Discovers cameras and populates dropdown (no multi-camera slot UI).

        Args:
            force_refresh: If True, force camera re-enumeration (default: False)
                          If False, reuse cached camera list to avoid slow ZMQ probing
        """
        logger.info(f"[GUI] build_frames called (force_refresh={force_refresh})")

        # Stop current camera if running
        if self._current_camera_index is not None:
            logger.info(f"[GUI] Stopping current camera {self._current_camera_index}")
            self.camera_worker_manager.stop_camera(self._current_camera_index)
            self._stop_preview_for_camera(self._current_camera_index)
            self._current_camera_index = None
            self.frames[0]['proc'] = None

            # Clear live canvas
            self.live_canvas.delete("all")
            self.camera_status_label.config(text="Camera not selected", foreground='gray')

        # OPTIMIZATION: Only enumerate cameras on first call or explicit refresh
        # This prevents 12+ second ZMQ probing freeze
        if not self._cameras_enumerated or force_refresh:
            # Prevent concurrent discovery threads
            if self._discovery_in_progress:
                logger.warning("[GUI] Discovery already in progress, ignoring request")
                return

            # Run camera discovery in background thread to avoid blocking GUI
            logger.info("[GUI] Starting async camera enumeration...")
            self._discovery_in_progress = True
            self._show_loading_overlay("Discovering cameras...")

            def discover_cameras_async():
                """Background thread for camera discovery."""
                try:
                    logger.info("[DISCOVERY] Background thread started")
                    cameras = list_video_devices(config=self.config)
                    logger.info(f"[DISCOVERY] Found {len(cameras)} camera(s)")

                    # Update GUI on main thread via after()
                    try:
                        self.after(0, lambda: self._on_cameras_discovered(cameras, force_refresh))
                    except Exception as e:
                        logger.error(f"[DISCOVERY] Failed to schedule success callback: {e}")
                        try:
                            error_msg = f"Callback error: {e}"
                            self.after(0, lambda msg=error_msg: self._on_camera_discovery_failed(msg))
                        except:
                            logger.critical("[DISCOVERY] Complete callback failure - GUI may be destroyed")
                except Exception as e:
                    logger.error(f"[DISCOVERY] Discovery failed: {e}", exc_info=True)

                    # Report error to GUI
                    try:
                        self.after(0, lambda: self._on_camera_discovery_failed(str(e)))
                    except Exception as callback_error:
                        logger.critical(f"[DISCOVERY] Failed to report error to GUI: {callback_error}")

            # Start background thread
            threading.Thread(target=discover_cameras_async, daemon=True).start()
            return  # Exit early, will populate dropdown after discovery completes
        else:
            logger.info(f"[GUI] Reusing cached camera list ({len(self.cams)} camera(s))")

        # Initialize LSL helper if needed
        if not hasattr(self, 'lsl_helper_proc') or self.lsl_helper_proc is None or not self.lsl_helper_proc.is_alive():
            self._initialize_lsl_helper()

    def _broadcast_camera_selection(self, cam_idx: int):
        """
        Centralized handler for camera selection - single source of truth.

        When a camera is selected (via main dropdown), this method:
        1. Stops the previous camera worker
        2. Starts the selected camera worker
        3. Updates _current_camera_index state
        4. Broadcasts selection to GUI processing worker

        Args:
            cam_idx: Camera index to select (0-based)
        """
        import time

        # Skip if already selected and running
        if (self._current_camera_index == cam_idx and
            cam_idx in self.active_camera_procs):
            logger.debug(f"[CAMERA SELECTION] Camera {cam_idx} already active")
            return

        logger.info(f"[CAMERA SELECTION] Broadcasting selection: camera {cam_idx}")

        # Stop previous camera
        if (self._current_camera_index is not None and
            self._current_camera_index in self.active_camera_procs and
            self._current_camera_index != cam_idx):
            logger.info(f"[CAMERA SELECTION] Stopping previous camera {self._current_camera_index}")
            self.camera_worker_manager.stop_camera(self._current_camera_index)
            self._stop_preview_for_camera(self._current_camera_index)

            # Remove from active tracking
            if self._current_camera_index in self.active_camera_procs:
                del self.active_camera_procs[self._current_camera_index]
            if self._current_camera_index in self.camera_to_frame_map:
                del self.camera_to_frame_map[self._current_camera_index]

            # Clear canvas
            self.live_canvas.delete("all")

        # Store old selection for worker notification
        old_camera = self._current_camera_index

        # Update selection state
        self._current_camera_index = cam_idx

        # Start selected camera if not already running
        if cam_idx not in self.active_camera_procs:
            logger.info(f"[CAMERA SELECTION] Starting camera {cam_idx}")
            self._start_single_camera(cam_idx)

        # Broadcast to GUI processing worker
        if hasattr(self, 'processing_control_queue') and self.processing_control_queue:
            try:
                self.processing_control_queue.put({
                    'type': 'selected_camera_changed',
                    'camera_index': cam_idx,
                    'previous_camera_index': old_camera,
                    'timestamp': time.time()
                }, block=False)
                logger.info(f"[CAMERA SELECTION] ✅ Broadcast 'selected_camera_changed' to worker: {old_camera} → {cam_idx}")
            except Exception as e:
                logger.error(f"[CAMERA SELECTION] ❌ Failed to broadcast to worker: {e}")

    def on_camera_selected(self, event=None):
        """
        Handle camera selection from main dropdown.
        Uses centralized broadcast handler for coordination.
        """
        selected = self.camera_combo.get()
        if not selected:
            logger.warning("[CAMERA SELECTION] No camera selected in dropdown")
            return

        try:
            # Parse "0: Camera Name" format
            cam_idx = int(selected.split(":", 1)[0])
            logger.info(f"[CAMERA SELECTION] User selected camera {cam_idx} from dropdown")

            # Validate camera exists in discovered cameras
            if not hasattr(self, 'cams') or cam_idx < 0 or cam_idx >= len(self.cams):
                logger.error(f"[CAMERA SELECTION] Camera {cam_idx} not found in discovered cameras (total: {len(self.cams) if hasattr(self, 'cams') else 0})")
                from tkinter import messagebox
                messagebox.showerror("Invalid Camera", f"Camera {cam_idx} is not available.\n\nPlease refresh the camera list.")
                return

            # Disable dropdown during switch
            self.camera_combo.config(state='disabled')

            # Use centralized broadcast handler
            self._broadcast_camera_selection(cam_idx)

        except (ValueError, IndexError) as e:
            logger.error(f"[CAMERA SELECTION] Failed to parse camera index from '{selected}': {e}")

    def on_laparoscopic_camera_selected(self, event=None):
        """
        Handle laparoscopic camera selection from laparoscopic dropdown.
        Routes camera to laparoscopic canvas (frame_idx=1).
        """
        selected = self.laparoscopic_camera_combo.get()
        if not selected:
            logger.warning("[LAPAROSCOPIC SELECTION] No camera selected in dropdown")
            return

        try:
            # Parse "0: Camera Name" format
            cam_idx = int(selected.split(":", 1)[0])
            logger.info(f"[LAPAROSCOPIC SELECTION] User selected camera {cam_idx} for laparoscopic display")

            # Validate camera exists
            if not hasattr(self, 'cams') or cam_idx < 0 or cam_idx >= len(self.cams):
                logger.error(f"[LAPAROSCOPIC SELECTION] Camera {cam_idx} not found")
                from tkinter import messagebox
                messagebox.showerror("Invalid Camera", f"Camera {cam_idx} is not available.\n\nPlease refresh the camera list.")
                return

            # Disable dropdown during switch
            self.laparoscopic_camera_combo.config(state='disabled')

            # Start camera for laparoscopic canvas (frame_idx=1)
            self._start_camera_for_canvas(cam_idx, frame_idx=1)

        except (ValueError, IndexError) as e:
            logger.error(f"[LAPAROSCOPIC SELECTION] Failed to parse camera index from '{selected}': {e}")

    def _start_single_camera(self, cam_idx):
        """
        Phase 2, Step 10: Start single camera and display on live_canvas.
        """
        # Call generalized method with frame_idx=0 (main camera)
        self._start_camera_for_canvas(cam_idx, frame_idx=0)

    def _start_camera_for_canvas(self, cam_idx, frame_idx=0):
        """
        Generalized camera start method that works for any canvas.

        Args:
            cam_idx: Camera device index to start
            frame_idx: Which frame/canvas to display on (0=main, 1=laparoscopic)
        """
        logger.info(f"Starting camera {cam_idx} for canvas {frame_idx}")

        # Get frame info for this canvas
        frame_info = self.frames[frame_idx]
        canvas = frame_info['canvas']
        status_label = frame_info['meta_label']

        # Get settings from GUI controls
        fps = self.desired_fps.get()
        res_label = self.res_choice.get()
        resolution = self.res_map.get(res_label, (1280, 720))

        # Update config for this camera
        camera_key = f"camera_{cam_idx}"
        if 'camera_settings' not in self.config.config:
            self.config.config['camera_settings'] = {}

        self.config.config['camera_settings'][camera_key] = {
            'width': resolution[0],
            'height': resolution[1],
            'fps': fps,
            'backend': 'auto'
        }
        self.config.save_config()

        # Show loading state
        status_label.config(
            text=f"Camera {cam_idx} - Starting...",
            foreground='orange'
        )
        canvas.delete("all")
        canvas_width = canvas.winfo_width() or 800
        canvas_height = canvas.winfo_height() or 600
        canvas.create_text(
            canvas_width // 2, canvas_height // 2,
            text=f"Initializing Camera {cam_idx}...",
            fill='white',
            font=('Arial', 14),
            tags='loading'
        )

        # Determine if this camera should be display-only based on canvas assignment
        # frame_idx==1 (laparoscopic canvas) = display-only mode (no detection/pose processing)
        display_only = (frame_idx == 1)

        # Start camera worker with callbacks that know which frame_idx to use
        self.camera_worker_manager.start_camera_async(
            cam_idx,
            on_ready=lambda ci, bn, res: self._on_camera_ready_for_canvas(ci, bn, res, frame_idx),
            on_failed=lambda ci, err: self._on_camera_failed_for_canvas(ci, err, frame_idx),
            gui_after_func=self.after,
            display_only=display_only
        )

        # Track which camera device is active for which canvas
        if frame_idx == 0:
            self._current_camera_index = cam_idx
        elif frame_idx == 1:
            self._current_laparoscopic_index = cam_idx

    def _on_camera_ready_for_canvas(self, camera_index, buffer_names, resolution, frame_idx):
        """
        Generalized callback when camera initializes successfully for any canvas.

        Args:
            camera_index: Camera device index
            buffer_names: Buffer names for this camera
            resolution: Camera resolution tuple (width, height)
            frame_idx: Which canvas this camera is for (0=main, 1=laparoscopic)
        """
        logger.info(f"Camera {camera_index} ready for canvas {frame_idx}: {resolution}")

        # Get frame info for this canvas
        frame_info = self.frames[frame_idx]
        canvas = frame_info['canvas']
        status_label = frame_info['meta_label']
        combo = frame_info['combo']

        # Update frame info
        frame_info['proc'] = self.camera_worker_manager.workers.get(camera_index)
        frame_info['camera_index'] = camera_index  # Update with actual device index for recording

        # Update camera mappings
        self.camera_to_frame_map[camera_index] = frame_idx
        self.actual_buffer_names[camera_index] = buffer_names

        # Update BufferCoordinator registry
        self.buffer_coordinator.update_coordinator_registry(camera_index, buffer_names)
        self.buffer_coordinator.camera_resolutions[camera_index] = resolution
        self.camera_resolutions[camera_index] = resolution

        # Update global resolution dropdown to match detected resolution (only for main camera)
        if frame_idx == 0 and resolution:
            self._update_resolution_dropdown_display(resolution)

        # Mark camera as active
        self.active_camera_procs[camera_index] = self.camera_worker_manager.workers[camera_index]

        # Notify GUI processing worker
        self.processing_control_queue.put({
            'type': 'buffer_names',
            'actual_buffer_names': self.actual_buffer_names.copy(),
            'camera_resolutions': self.buffer_coordinator.camera_resolutions.copy(),
            'request_ack': True,
            'camera_index': camera_index
        })

        # DYNAMIC CONNECTION: Tell worker to connect to this camera's display buffer
        try:
            self.processing_control_queue.put({
                'type': 'connect_camera',
                'camera_index': camera_index,
                'buffer_names': buffer_names,
                'resolution': resolution
            })
            logger.info(f"[GUI] Sent connect_camera command to worker for camera {camera_index}")
        except Exception as e:
            logger.error(f"[GUI] Failed to send connect_camera command for camera {camera_index}: {e}")

        # Create recording queue for this camera (for video recording)
        if camera_index not in self.recording_queues:
            import multiprocessing as mp
            self.recording_queues[camera_index] = mp.Queue(maxsize=60)  # 2 seconds buffer at 30fps
            logger.info(f"[VIDEO RECORDING] Created recording queue for camera {camera_index}")

        # Update UI
        status_label.config(
            text=f"Camera {camera_index} - {resolution[0]}x{resolution[1]} @ {self.desired_fps.get()}fps",
            foreground='green'
        )

        # Clear loading message
        canvas.delete('loading')

        # Re-enable dropdown
        combo.config(state='readonly')

        # Start preview loop
        self.after(200, lambda: self._start_preview_for_camera(camera_index))

        # Start face recognition if enabled (only for main camera)
        if frame_idx == 0:
            self._start_face_recognition_if_needed()

    def _on_camera_failed_for_canvas(self, camera_index, error_message, frame_idx):
        """
        Generalized callback when camera fails to initialize for any canvas.

        Args:
            camera_index: Camera device index
            error_message: Error description
            frame_idx: Which canvas this camera was for (0=main, 1=laparoscopic)
        """
        logger.error(f"Camera {camera_index} failed for canvas {frame_idx}: {error_message}")

        # Get frame info for this canvas
        frame_info = self.frames[frame_idx]
        canvas = frame_info['canvas']
        status_label = frame_info['meta_label']
        combo = frame_info['combo']

        # Clear frame proc reference
        frame_info['proc'] = None

        # Clear tracking variables
        if frame_idx == 0:
            self._current_camera_index = None
        elif frame_idx == 1:
            self._current_laparoscopic_index = None

        # Show error on canvas
        canvas.delete("all")
        canvas_width = canvas.winfo_width() or 800
        canvas_height = canvas.winfo_height() or 600
        canvas.create_text(
            canvas_width // 2, canvas_height // 2,
            text=f"Camera {camera_index} Failed\n\n{error_message}\n\nTry selecting a different camera",
            fill="red",
            font=("Arial", 12),
            justify='center'
        )

        # Update status label
        status_label.config(
            text=f"Camera {camera_index} - FAILED",
            foreground='red'
        )

        # Re-enable dropdown
        combo.config(state='readonly')

        # Show error dialog
        from tkinter import messagebox
        messagebox.showerror(
            "Camera Initialization Failed",
            f"Camera {camera_index} failed to start:\n\n{error_message}\n\nPlease select a different camera."
        )

    def _on_single_camera_ready(self, camera_index, buffer_names, resolution):
        """
        Phase 2, Step 11: Callback when single camera initializes successfully.
        Always uses frame_idx=0 for single-camera mode.
        """
        logger.info(f"Camera {camera_index} ready: {resolution}, buffers: {list(buffer_names.keys())}")

        # Update frame info (always slot 0)
        self.frames[0]['proc'] = self.camera_worker_manager.workers.get(camera_index)

        # Update camera mappings (frame_idx always 0 for single-camera mode)
        self.camera_to_frame_map[camera_index] = 0
        self.actual_buffer_names[camera_index] = buffer_names

        # Update BufferCoordinator registry
        self.buffer_coordinator.update_coordinator_registry(camera_index, buffer_names)
        self.buffer_coordinator.camera_resolutions[camera_index] = resolution
        self.camera_resolutions[camera_index] = resolution

        # Update global resolution dropdown to match detected resolution
        if resolution:
            self._update_resolution_dropdown_display(resolution)

        # Mark camera as active
        self.active_camera_procs[camera_index] = self.camera_worker_manager.workers[camera_index]

        # Notify GUI processing worker
        self.processing_control_queue.put({
            'type': 'buffer_names',
            'actual_buffer_names': self.actual_buffer_names.copy(),
            'camera_resolutions': self.buffer_coordinator.camera_resolutions.copy(),
            'request_ack': True,
            'camera_index': camera_index
        })

        # Update UI
        self.camera_status_label.config(
            text=f"Camera {camera_index} - {resolution[0]}x{resolution[1]} @ {self.desired_fps.get()}fps",
            foreground='green'
        )

        # Clear loading message
        self.live_canvas.delete('loading')

        # Re-enable dropdown
        self.camera_combo.config(state='readonly')

        # Start preview loop (existing method)
        self.after(200, lambda: self._start_preview_for_camera(camera_index))

        # Start face recognition if enabled
        self._start_face_recognition_if_needed()

    def _on_single_camera_failed(self, camera_index, error_message):
        """
        Phase 2, Step 11: Callback when single camera fails to initialize.
        """
        logger.error(f"Camera {camera_index} failed: {error_message}")

        # Clear frame proc reference
        self.frames[0]['proc'] = None
        self._current_camera_index = None

        # Show error on canvas
        self.live_canvas.delete("all")
        canvas_width = self.live_canvas.winfo_width() or 800
        canvas_height = self.live_canvas.winfo_height() or 600

        self.live_canvas.create_text(
            canvas_width // 2,
            canvas_height // 2,
            text=f"Camera {camera_index} Failed\n\n{error_message}\n\nTry selecting a different camera",
            fill="red",
            font=("Arial", 12),
            justify='center'
        )

        # Update status label
        self.camera_status_label.config(
            text=f"Camera {camera_index} - FAILED",
            foreground='red'
        )

        # Re-enable dropdown
        self.camera_combo.config(state='readonly')

        # Show error dialog
        from tkinter import messagebox
        messagebox.showerror(
            "Camera Initialization Failed",
            f"Camera {camera_index} failed to start:\n\n{error_message}\n\nPlease select a different camera."
        )

    def _initialize_lsl_helper(self):
        """Initialize LSL helper process for streaming data to Lab Streaming Layer."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("[LSL] Initializing LSL helper process...")

        # Create queues for LSL process communication
        self.lsl_data_queue = MPQueue()
        self.lsl_command_queue = MPQueue()

        # Create correlation buffer (52 float32 values for comodulation)
        import multiprocessing.shared_memory as shared_memory
        self.correlation_buffer = shared_memory.SharedMemory(create=True, size=52 * 4)
        self.corr_array = np.ndarray((52,), dtype=np.float32, buffer=self.correlation_buffer.buf)

        # Start LSL helper process
        self.lsl_helper_proc = Process(
            target=lsl_helper_process,
            args=(
                self.lsl_command_queue,
                self.lsl_data_queue,
                self.correlation_buffer.name,  # Pass buffer name for child process access
                self.DESIRED_FPS,
                self.config.config  # Pass full config dict
            ),
            daemon=False  # Must complete cleanup on exit
        )
        self.lsl_helper_proc.start()

        logger.info(f"[LSL] Helper process started with PID: {self.lsl_helper_proc.pid}")

    def auto_detect_optimal_fps(self, cam_idx):
        """Test camera with different backends to find best resolution/FPS combination"""
        import platform

        print(f"\n[Camera Test] Checking if camera {cam_idx} exists...")

        # First check if camera exists at all
        try:
            test_cap = cv2.VideoCapture(cam_idx)
            if not test_cap.isOpened():
                print(f"[Camera Test] Camera {cam_idx} does not exist or cannot be opened")
                return None
            test_cap.release()
            print(f"[Camera Test] Camera {cam_idx} found, proceeding with backend tests...")
        except Exception as e:
            print(f"[Camera Test] Error checking camera {cam_idx}: {e}")
            return None

        # Define backends to test (Windows backends)
        backends_to_test = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Auto"),
        ]

        test_configs = [
            # (width, height, target_fps)
            # Test realistic configs first
            (640, 480, 30),
            (1280, 720, 30),
            (1920, 1080, 30),
            (640, 480, 60),
            (1280, 720, 60),
            (2560, 1440, 30),  # 2K
            (2560, 1440, 24),  # 2K lower FPS
            (3840, 2160, 30),  # 4K
            (3840, 2160, 24),  # 4K lower FPS
            # High FPS configs last (often unsupported)
            (320, 240, 60),
            (320, 240, 120),
        ]

        all_results = []
        best_overall = None
        best_fps = 0

        print(f"\n[Camera Test] Testing camera {cam_idx} with different backends...")

        for backend_id, backend_name in backends_to_test:
            print(f"\n[Camera Test] Testing backend: {backend_name}")
            backend_results = []

            # Skip FFmpeg for camera index capture (it only works with file paths)
            if backend_id == cv2.CAP_FFMPEG:
                print(f"  Skipping {backend_name} (doesn't support camera index capture)")
                continue

            for width, height, target_fps in test_configs:
                try:
                    # Set a timeout for this entire test
                    import signal

                    def timeout_handler(signum, frame):
                        raise TimeoutError("Backend test timed out")

                    # Skip signal-based timeout on Windows
                    if platform.system() != "Windows":
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(10)  # 10 second timeout per test

                    try:
                        # Try to open camera with specific backend
                        cap = cv2.VideoCapture(cam_idx, backend_id)
                        if not cap.isOpened():
                            continue

                        # Configure camera
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Optimal: buffer=1 limits to 15fps, buffer=2+ achieves 29fps
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        cap.set(cv2.CAP_PROP_FPS, target_fps)

                        # Verify settings took effect
                        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        # Skip if resolution didn't apply
                        if actual_width != width or actual_height != height:
                            cap.release()
                            continue

                        # Warm-up with timeout
                        start_time = time.time()
                        got_frame = False
                        while time.time() - start_time < 0.5:  # 0.5 sec warm-up
                            ret, _ = cap.read()
                            if ret:
                                got_frame = True
                                break
                            time.sleep(0.01)  # Small delay to avoid busy loop

                        if not got_frame:
                            cap.release()
                            continue

                        # Measure actual FPS
                        measure_time = 2.0
                        frames = 0
                        unique = 0
                        last_hash = None
                        measure_start = time.time()

                        while time.time() - measure_start < measure_time:
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                frames += 1
                                # Sample frame to detect unique frames
                                try:
                                    h = hash(frame[::10, ::10].tobytes())
                                    if h != last_hash:
                                        unique += 1
                                        last_hash = h
                                except:
                                    unique += 1  # Count frame if hashing fails

                        cap.release()

                    finally:
                        # Cancel alarm if set
                        if platform.system() != "Windows":
                            signal.alarm(0)

                    if unique > 0:
                        actual_fps = unique / measure_time
                        result = {
                            'backend': backend_name,
                            'backend_id': backend_id,
                            'resolution': f"{width}x{height}",
                            'target_fps': target_fps,
                            'actual_fps': actual_fps,
                            'efficiency': actual_fps / target_fps if target_fps > 0 else 0,
                            'width': width,
                            'height': height,
                            'pixels': width * height
                        }
                        backend_results.append(result)
                        all_results.append(result)

                        print(f"  {width}x{height}@{target_fps}: {actual_fps:.1f} FPS")

                        # Track best overall FPS
                        if actual_fps > best_fps:
                            best_fps = actual_fps
                            best_overall = result

                except TimeoutError:
                    print(f"  Timeout testing {width}x{height}@{target_fps} - skipping")
                    try:
                        cap.release()
                    except:
                        pass
                    continue  # Try next resolution instead of abandoning backend
                except Exception as e:
                    print(f"  Error testing {width}x{height}@{target_fps}: {e}")
                    try:
                        cap.release()
                    except:
                        pass
                    continue

            if not backend_results:
                print(f"  No successful configurations with {backend_name}")

        if not all_results:
            print(f"[Camera Test] No configurations worked for camera {cam_idx}")
            return None

        # Select best configuration with smart logic
        def select_smart_config(configs):
            """Prefers higher resolution unless it costs ≥10 fps vs next lowest."""
            # Group by backend
            by_backend = {}
            for cfg in configs:
                backend = cfg['backend']
                if backend not in by_backend:
                    by_backend[backend] = []
                by_backend[backend].append(cfg)

            # Find best config per backend
            best_per_backend = []
            for backend, backend_configs in by_backend.items():
                # Sort by pixel count (ascending)
                backend_configs = sorted(backend_configs, key=lambda x: x['pixels'])

                # Select best resolution for this backend
                best = backend_configs[0]
                for i in range(1, len(backend_configs)):
                    prev = backend_configs[i-1]
                    curr = backend_configs[i]
                    if prev['actual_fps'] - curr['actual_fps'] >= 10:
                        # Too much fps drop, stick with previous
                        break
                    else:
                        best = curr  # Move up to higher res

                best_per_backend.append(best)

            # Now select best across all backends (highest FPS at reasonable resolution)
            best_per_backend = sorted(best_per_backend,
                                     key=lambda x: (x['actual_fps'], x['pixels']),
                                     reverse=True)
            return best_per_backend[0] if best_per_backend else configs[0]

        # Find best configuration
        best = select_smart_config(all_results)

        print(f"\n[Camera Test] Best overall: {best['backend']} - {best['resolution']} @ {best['actual_fps']:.1f} FPS")
        print(f"[Camera Test] Selected backend: {best['backend']} (ID: {best['backend_id']})")

        return best
    
        
    def diagnose_pipeline(self):
        """Diagnostic method to test if the pipeline is working"""
        print("[DEBUG] diagnose_pipeline started"); import sys; sys.stdout.flush()
        print("\n=== PIPELINE DIAGNOSIS ===")
        
        # Check if workers are running
        print("[DEBUG] Checking workers"); sys.stdout.flush()
        active_workers = 0
        for idx, info in enumerate(self.frames):
            if info.get('proc') and info['proc'].is_alive():
                active_workers += 1
                print(f"Worker {idx}: RUNNING")
            else:
                print(f"Worker {idx}: NOT RUNNING")
        print("[DEBUG] Workers checked"); sys.stdout.flush()
        
        print(f"Active workers: {active_workers}")
        
        # Check if preview queues have data
        for idx, q in enumerate(self.preview_queues):
            if q:
                print(f"Preview queue {idx}: {q.qsize()} items")
            else:
                print(f"Preview queue {idx}: None")
        
        # Check LSL helper
        if not hasattr(self, 'lsl_helper_proc') or self.lsl_helper_proc is None or not self.lsl_helper_proc.is_alive():
            print("LSL Helper: RUNNING")
        else:
            print("LSL Helper: NOT RUNNING")
        
        # Check streaming state
        print(f"Streaming active: {self.streaming}")
        
        print("=== END DIAGNOSIS ===\n")
        print("[DEBUG] diagnose_pipeline completed"); import sys; sys.stdout.flush()


    def _start_face_recognition_if_needed(self):
        """Start face recognition process if cameras are ready and process not started."""
        if not self.face_recognition or self._face_recognition_started:
            return

        try:
            # BUGFIX: Only use cameras that actually started and have buffers
            # This prevents "Camera X not found in buffer registry" errors
            active_cameras = []
            for cam_idx in self.active_camera_procs.keys():
                if cam_idx in self.camera_worker_manager.shared_memories:
                    active_cameras.append(cam_idx)

            if active_cameras:
                print(f"[GUI] Starting face recognition worker (threading mode) for {len(active_cameras)} cameras: {active_cameras}")

                # CRITICAL FIX: Update face recognition camera indices to match actually running cameras
                # The face_recognition object was initialized with camera_count-based indices,
                # but we need to use only cameras that successfully started
                if hasattr(self.face_recognition, 'camera_indices'):
                    print(f"[GUI] Updating face recognition camera indices from {self.face_recognition.camera_indices} to {active_cameras}")
                    self.face_recognition.camera_indices = active_cameras
                    # Also update the manager if it exists
                    if hasattr(self.face_recognition, 'manager') and self.face_recognition.manager:
                        self.face_recognition.manager.camera_indices = active_cameras

                success = self.face_recognition.start()
                if success:
                    self._face_recognition_started = True
                    print("[GUI] Face recognition worker started successfully (GPU context shared)")
                else:
                    print("[GUI] Failed to start face recognition worker")
            else:
                print("[GUI] No active cameras with buffers found - skipping face recognition start")

        except Exception as e:
            print(f"[GUI] Failed to start face recognition worker: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_bbox_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        try:
            # bbox format: [x1, y1, x2, y2]
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)
            
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            # Calculate union
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            print(f"[IoU ERROR] Failed to calculate IoU: {e}")
            return 0.0

    def _update_track_to_participant_mapping(self, faces, assignments, idx, recognition_updates=None):
        """Maintain stable mapping between track IDs and participant slots."""
        if recognition_updates is None:
            recognition_updates = {}
            
        try:
            current_time = time.time()
            
            for i, face in enumerate(faces):
                track_id = face.get('track_id')
                participant_id = assignments[i] if i < len(assignments) else None
                
                if track_id is not None and participant_id is not None:
                    # Update mapping
                    self.tracker_id_to_slot[track_id] = participant_id
                    self.slot_to_tracker_id[participant_id] = track_id
                    self.participant_last_seen[participant_id] = current_time
                    
                    # Get embedding data if available from recognition
                    embedding = None
                    if track_id in recognition_updates and recognition_updates[track_id].get('embedding') is not None:
                        embedding = recognition_updates[track_id]['embedding']
                    
                    # Register track-participant mapping - use hybrid IPC based on data size
                    # GUI mapping debug - DISABLED FOR PERFORMANCE
                    # print(f"[GUI MAPPING] Registering track {track_id} -> participant {participant_id} for camera {idx}")
                    
                    if embedding is not None:
                        # Large data with embedding - use legacy mp.Queue (required for >2KB data)
                        if hasattr(self, 'camera_worker_manager'):
                            # GUI mapping debug - DISABLED FOR PERFORMANCE
                            # print(f"[GUI MAPPING] Using legacy queue for embedding data (track {track_id})")
                            self.camera_worker_manager.register_track_participant(
                                idx, track_id, participant_id, embedding.tolist()
                            )
                            success = True  # Assume success for legacy method
                        else:
                            success = False
                    else:
                        # Small data without embedding - use CommandBuffer
                        success = self.send_camera_specific_command(
                            idx,
                            'register_track_participant',
                            {
                                'track_id': track_id,
                                'participant_id': participant_id,
                                'camera_index': idx
                            }
                        )
                        if not success and hasattr(self, 'camera_worker_manager'):
                            # Fallback to legacy method
                            # GUI mapping debug - DISABLED FOR PERFORMANCE
                            # print(f"[GUI MAPPING] Fallback: Using legacy registration for camera {idx}")
                            self.camera_worker_manager.register_track_participant(
                                idx, track_id, participant_id
                            )
                    
                    # Clean old mappings for this participant slot
                    for old_track, old_participant in list(self.tracker_id_to_slot.items()):
                        if old_participant == participant_id and old_track != track_id:
                            del self.tracker_id_to_slot[old_track]
                            if idx == 0:  # Debug
                                print(f"[MAPPING] Cleaned old track {old_track} for participant {participant_id}")
            
            # Clean up old participants (not seen for 5 seconds)
            timeout = 5.0
            for participant_id in list(self.participant_last_seen.keys()):
                if current_time - self.participant_last_seen[participant_id] > timeout:
                    # Remove from all mappings
                    if participant_id in self.slot_to_tracker_id:
                        old_track = self.slot_to_tracker_id[participant_id]
                        if old_track in self.tracker_id_to_slot:
                            del self.tracker_id_to_slot[old_track]
                        del self.slot_to_tracker_id[participant_id]
                    del self.participant_last_seen[participant_id]
                    
                    if idx == 0:  # Debug
                        print(f"[MAPPING] Removed expired participant {participant_id}")
                        
        except Exception as e:
            print(f"[MAPPING ERROR] Failed to update mapping: {e}")

    def _handle_track_id_changes(self, detection_boxes, idx):
        """Handle when detection process assigns new track IDs to same face."""
        try:
            for det in detection_boxes:
                track_id = det['track_id']
                bbox = det['bbox']
                
                # Check if this bbox matches a known participant spatially
                best_match_participant = None
                best_iou = 0.0
                
                for participant_id, old_track_id in self.slot_to_tracker_id.items():
                    if old_track_id in self.last_known_bboxes:
                        old_bbox = self.last_known_bboxes[old_track_id]
                        iou = self._calculate_bbox_iou(bbox, old_bbox)
                        
                        if iou > best_iou and iou > 0.7:  # High spatial overlap threshold
                            best_iou = iou
                            best_match_participant = participant_id
                
                if best_match_participant is not None:
                    # Transfer participant assignment to new track ID
                    old_track_id = self.slot_to_tracker_id[best_match_participant]
                    self.tracker_id_to_slot[track_id] = best_match_participant
                    self.slot_to_tracker_id[best_match_participant] = track_id
                    
                    # Clean old mapping
                    if old_track_id in self.tracker_id_to_slot:
                        del self.tracker_id_to_slot[old_track_id]
                    if old_track_id in self.last_known_bboxes:
                        del self.last_known_bboxes[old_track_id]
                    
                    if idx == 0:  # Debug
                        print(f"[TRACK TRANSFER] Track {old_track_id} -> {track_id} for participant {best_match_participant} (IoU={best_iou:.3f})")
                
                # Store current bbox for future comparison
                self.last_known_bboxes[track_id] = bbox
                
        except Exception as e:
            print(f"[TRACK TRANSFER ERROR] Failed to handle track changes: {e}")
    
    def _build_unified_face_list(self, landmarks_data, detection_boxes, cam_idx):
        """Build unified face list tolerant of frame mismatches."""
        faces = []
        
        # Track frame IDs for debugging
        landmark_frame_id = landmarks_data.get('frame_id', -1) if landmarks_data else -1
        detection_frame_id = detection_boxes[0].get('frame_id', -1) if detection_boxes else -1
        
        # Allow up to 3 frame difference
        frame_diff = abs(landmark_frame_id - detection_frame_id)
        # Frame mismatch debug - DISABLED FOR PERFORMANCE
        # if frame_diff > 3 and landmark_frame_id > 0 and detection_frame_id > 0:
        #     print(f"[FRAME MISMATCH] Camera {cam_idx}: landmarks frame_id={landmark_frame_id}, "
        #           f"detection frame_id={detection_frame_id}, diff={frame_diff}")
        
        # Priority 1: Use landmark data if available (most complete)
        if landmarks_data and landmarks_data.get('n_faces', 0) > 0:
            landmarks = landmarks_data.get('landmarks', [])
            n_faces = landmarks_data.get('n_faces', 0)
            roi_metadata = landmarks_data.get('roi_metadata', None)
            
            for face_idx in range(n_faces):
                if face_idx < len(landmarks):
                    face_landmarks = landmarks[face_idx]
                    
                    # Get track ID and metadata
                    track_id = face_idx  # Default
                    is_detection = True
                    track_age = 0
                    confidence = 1.0
                    transform = None
                    
                    if roi_metadata is not None and face_idx < len(roi_metadata):
                        meta = roi_metadata[face_idx]
                        track_id = int(meta['track_id'])
                        is_detection = bool(meta['is_detection'])
                        track_age = int(meta['track_age'])
                        confidence = float(meta['confidence'])
                        transform = {
                            'x1': float(meta['x1']),
                            'y1': float(meta['y1']),
                            'scale_x': float(meta['scale_x']),
                            'scale_y': float(meta['scale_y'])
                        }
                    
                    # Create face dict
                    face = {
                        'id': track_id,
                        'track_id': track_id,
                        'landmarks': face_landmarks,
                        'track_age': track_age,
                        'confidence': confidence,
                        'transform': transform,
                        'centroid': (0.5, 0.5),
                        'source': 'landmarks'
                    }
                    
                    # Generate bbox from transform
                    if transform and all(k in transform for k in ['x1', 'y1', 'scale_x', 'scale_y']):
                        x1 = transform['x1']
                        y1 = transform['y1']
                        x2 = x1 + transform['scale_x'] * 256
                        y2 = y1 + transform['scale_y'] * 256
                        face['bbox'] = [x1, y1, x2, y2]
                        face['centroid'] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    
                    faces.append(face)
        
        # Priority 2: Add detection boxes not covered by landmarks
        if detection_boxes:
            landmark_track_ids = {f.get('track_id') for f in faces}
            
            for det_box in detection_boxes:
                track_id = det_box.get('track_id', -1)
                
                # Skip if already have landmarks for this track
                if track_id in landmark_track_ids:
                    continue
                
                # Create face from detection box
                face = {
                    'id': track_id,
                    'track_id': track_id,
                    'bbox': det_box.get('bbox', [0, 0, 100, 100]),
                    'confidence': det_box.get('confidence', 0.5),
                    'landmarks': None,  # No landmarks for detection-only
                    'track_age': 0,
                    'transform': None,
                    'source': 'detection'
                }
                
                # Calculate centroid
                bbox = face['bbox']
                face['centroid'] = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                
                faces.append(face)
        
        # Debug output
        if self._should_log_for_camera(cam_idx) and faces:
            track_ids = [f['track_id'] for f in faces]
            sources = [f['source'] for f in faces]
            print(f"[UNIFIED FACES] Camera {cam_idx}: {len(faces)} faces, "
                  f"track_ids={track_ids}, sources={sources}")
        
        return faces
    
    def _get_participant_labels(self, faces):
        """Get participant labels for face drawing."""
        labels = {}
        
        for face in faces:
            # Get participant ID from face data
            participant_id = face.get('participant_id')
            if not participant_id:
                # Try to get from track ID mapping
                track_id = face.get('track_id')
                if hasattr(self, 'participant_tracker') and self.participant_tracker:
                    participant_id = self.participant_tracker.get_participant_for_track(track_id)
            
            # Get label for participant
            if participant_id:
                if hasattr(self, 'global_participant_manager') and self.global_participant_manager:
                    label = self.global_participant_manager.get_participant_name(participant_id)
                else:
                    label = f"P{participant_id}"
                labels[participant_id] = label
        
        return labels

    def schedule_preview(self):
        """
        Camera frame preview loop - only runs when cameras are active.
        Moved from GUI startup to camera lifecycle management.
        """
        # Only reschedule if we have active cameras
        if hasattr(self, '_preview_active') and self._preview_active:
            self.after(16, self.schedule_preview)
        else:
            print("[GUI] Preview loop stopped - no active cameras")
            return

        # Early exits for shutdown
        if hasattr(self, '_shutdown_in_progress') and self._shutdown_in_progress:
            return
        if self.shutdown_flag:
            return
        if not hasattr(self, '_init_complete') or not self._init_complete:
            return

        # CRITICAL FIX: Update lock button states in display loop (60 FPS)
        # This ensures presence changes are reflected immediately (not just once per second)
        # Fixes delayed unlock/lock state updates
        if hasattr(self, 'participant_lock_buttons'):
            self.update_lock_button_states()
        if not hasattr(self, 'camera_worker_manager'):
            return
            
        # Check if we actually have active cameras
        if not hasattr(self, 'active_camera_procs') or not self.active_camera_procs:
            print("[GUI] No active cameras found - stopping preview loop")
            self._preview_active = False
            self._optimized_display_active = False  # Stop optimized display too
            return

        # Debug logging (reduced frequency)
        if not hasattr(self, '_preview_debug_count'):
            self._preview_debug_count = 0
        self._preview_debug_count += 1
        if self._preview_debug_count % 300 == 0:  # Log every 5 seconds at 60fps
            active_count = len(self.active_camera_procs)
            print(f"[GUI] Preview loop active: {self._preview_debug_count} cycles, {active_count} cameras active")
        
        # NOTE: GUI health monitoring moved to independent update_gui_health() method

        # CRITICAL: Process status updates with rate limiting
        now = time.time()
        if now >= self._next_status_poll_ts:
            self._process_camera_status_updates()
            self._check_processing_worker_status()  # Process worker notifications (display_buffer_added, etc.)
            self._next_status_poll_ts = now + (self._status_poll_interval_ms / 1000.0)

        # ARCHITECTURAL FIX: Display processing initialization moved to dedicated code paths
        # - Initialization: on_buffer_names_ready() callback (line ~2183)
        # - Display loop start: _wait_for_worker_ready_async() callback (line ~3127)
        # This ensures display_readers are always connected before loop starts
        # OLD CODE (REMOVED): Lines 3004-3032 duplicated initialization and started loop prematurely

    def _wait_for_buffer_names_async(self, cam_idx, callback, max_wait=5.0, poll_interval=100):
        """
        Asynchronously wait for camera buffer names using root.after().
        NEVER blocks Tkinter event loop - allows GUI to remain responsive.

        Args:
            cam_idx: Camera index
            callback: Function to call when ready (success=True/False, buffer_names=dict/None)
            max_wait: Maximum wait time in seconds
            poll_interval: Polling interval in milliseconds
        """
        start_time = time.time()

        def check_buffer_names():
            elapsed = time.time() - start_time

            # Process status updates (non-blocking)
            self._process_camera_status_updates()

            # Check if buffer names available
            if hasattr(self, 'actual_buffer_names') and cam_idx in self.actual_buffer_names:
                buffer_names = self.actual_buffer_names[cam_idx]
                if isinstance(buffer_names, dict) and 'frame' in buffer_names:
                    logger.info(f"[GUI] Got buffer names for camera {cam_idx}: {list(buffer_names.keys())}")
                    callback(success=True, buffer_names=buffer_names)
                    return  # Done

            # Check timeout
            if elapsed >= max_wait:
                logger.warning(f"[GUI] Timeout waiting for buffer names (camera {cam_idx}) after {max_wait}s")
                callback(success=False, buffer_names=None)
                return  # Timeout

            # Schedule next check (NON-BLOCKING - allows event processing)
            self.after(poll_interval, check_buffer_names)

        # Start async polling
        check_buffer_names()

    def _wait_for_worker_ready_async(self, timeout=60.0, poll_interval=100):
        """
        Asynchronously wait for GUI processing worker to be ready.
        NEVER blocks Tkinter event loop - allows GUI to remain responsive.

        TIMEOUT INCREASED TO 60s: Worker initialization can take 30-40s due to:
        - Buffer name wait: up to 10s
        - Recovery buffer connections: 20-30s (8 cameras × 2 connection attempts)
        Timeout should be 2× expected initialization time for reliability.

        Args:
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in milliseconds

        Sets:
            self._worker_ready_flag: True when ready, False on error/timeout, None while pending
        """
        start_time = time.time()

        def check_worker_status():
            elapsed = time.time() - start_time
            status = None

            try:
                # CRITICAL FIX: Drain ALL messages until we find 'ready' or queue is empty
                # This ensures we don't miss the 'ready' message if queue has performance messages
                messages_drained = 0
                while True:
                    msg = self.processing_status_queue.get_nowait()
                    messages_drained += 1
                    logger.info(f"[WORKER INIT] Received status: {msg.get('type', 'unknown')} (message {messages_drained})")

                    if msg.get('type') == 'ready':
                        status = msg
                        logger.info(f"[WORKER INIT] ✅ Found 'ready' message after draining {messages_drained} message(s)")
                        break
                    else:
                        # Discard performance/error messages during startup - we only need 'ready'
                        logger.info(f"[WORKER INIT] Discarding {msg.get('type')} message during startup scan")

            except Exception as queue_error:
                # No status available yet (Empty exception) or other error
                # status remains None, continue to polling logic below
                pass

            # Process status if received (AFTER try/except to avoid premature returns)
            if status is not None and status['type'] == 'ready':
                # FIX #3: Enhanced startup sequence logging
                logger.info(f"[WORKER INIT] ========== Worker Ready - Connecting Buffers ==========")
                logger.info(f"[WORKER INIT] Display buffers to connect: {list(status['display_buffers'].keys())}")

                # Connect to display buffers
                for cam_idx, buffer_name in status['display_buffers'].items():
                    try:
                        logger.info(f"[WORKER INIT] Connecting display buffer for camera {cam_idx}...")
                        self.display_buffer_manager.connect_display_buffer(cam_idx, buffer_name)
                        logger.info(f"[WORKER INIT] ✅ Connected display buffer for camera {cam_idx}: {buffer_name}")
                        logger.info(f"[WORKER INIT] Display readers now: {list(self.display_buffer_manager.display_readers.keys())}")

                        # Also connect to frame buffer
                        if hasattr(self, 'actual_buffer_names') and cam_idx in self.actual_buffer_names:
                            frame_buffer_name = self.actual_buffer_names[cam_idx].get('frame')
                            if frame_buffer_name:
                                logger.info(f"[WORKER INIT] Connecting frame buffer for camera {cam_idx}...")
                                self.display_buffer_manager.connect_frame_buffer(cam_idx, frame_buffer_name)
                                logger.info(f"[WORKER INIT] ✅ Connected frame buffer for camera {cam_idx}: {frame_buffer_name}")
                            else:
                                logger.warning(f"[WORKER INIT] ⚠️  No frame buffer name for camera {cam_idx}")
                        else:
                            logger.warning(f"[WORKER INIT] ⚠️  actual_buffer_names not available for camera {cam_idx}")
                    except Exception as connect_error:
                        logger.error(f"[WORKER INIT] ❌ Failed to connect buffers for camera {cam_idx}: {connect_error}")
                        import traceback
                        traceback.print_exc()
                        self._worker_ready_flag = False
                        return

                logger.info(f"[WORKER INIT] ========== All Buffers Connected Successfully ==========")
                logger.info(f"[WORKER INIT] Final display_readers: {list(self.display_buffer_manager.display_readers.keys())}")
                logger.info(f"[WORKER INIT] Final frame_buffers: {list(getattr(self.display_buffer_manager, 'frame_buffers', {}).keys())}")

                logger.info(f"[WORKER INIT] Worker ready after {elapsed:.1f}s")
                self._worker_ready_flag = True
                logger.info("GUI processing worker ready")

                # CRITICAL FIX: Start display loop immediately when worker is ready
                # This ensures the loop starts regardless of which code path called initialize_display_processing()
                if not getattr(self, '_optimized_display_active', False):
                    logger.info("[GUI] Starting optimized display loop (worker ready)")
                    self._optimized_display_active = True
                    self._schedule_optimized_display()
                else:
                    logger.info("[GUI] Optimized display loop already running")

                return  # Done - worker is ready

            elif status is not None and status['type'] == 'error':
                logger.error(f"[WORKER INIT] Worker reported error: {status.get('error')}")
                self._worker_ready_flag = False
                return  # Error

            # Check timeout (if not ready/error)
            if elapsed >= timeout:
                logger.error(f"[WORKER INIT] Worker failed to start within {timeout}s timeout")
                if hasattr(self, 'gui_processing_worker'):
                    logger.error(f"[WORKER INIT] Worker alive: {self.gui_processing_worker.is_alive()}")
                self._worker_ready_flag = False
                return  # Timeout

            # Log progress every 5 seconds
            if int(elapsed * 10) % 50 == 0 and elapsed > 0:  # Every 5 seconds
                logger.info(f"[WORKER INIT] Still waiting... {elapsed:.1f}s elapsed")

            # CRITICAL: Always reschedule unless we returned above (ready/error/timeout)
            # This ensures the polling loop continues until worker is ready
            self.after(poll_interval, check_worker_status)

        # Start async polling
        self._worker_ready_flag = None  # None = pending, True = ready, False = error
        check_worker_status()

    def initialize_display_processing(self):
        """Initialize display processing components with separate worker process."""
        try:
            # REMOVED: Imports moved to module level to fix multiprocessing spawn mode issue
            # (Worker function must be importable at module level for child process to find it)
            import multiprocessing as mp

            logger.info("Initializing display processing...")
            
            # Check if we have actual buffer names for all cameras
            active_cameras = list(self.active_camera_procs.keys())
            if not hasattr(self, 'actual_buffer_names'):
                logger.warning("No actual buffer names captured yet - waiting for camera ready status")
                return False
                
            # Ensure we have buffer names for all active cameras
            missing_cameras = [cam for cam in active_cameras if cam not in self.actual_buffer_names]
            if missing_cameras:
                logger.warning(f"Still waiting for buffer names from cameras: {missing_cameras}")
                return False
            
            logger.info(f"Have actual buffer names for all {len(active_cameras)} cameras - proceeding with display processing")
            
            # CRITICAL FIX: Clean up any existing display buffers first
            try:
                if hasattr(self, 'buffer_coordinator'):
                    for cam_idx in active_cameras:
                        self.buffer_coordinator.cleanup_display_buffer(cam_idx)
                        logger.info(f"Cleaned up existing display buffer for camera {cam_idx}")
            except Exception as cleanup_error:
                logger.warning(f"Display buffer cleanup error (non-fatal): {cleanup_error}")
            
            # Create display buffer manager (only once - reuse on worker restart)
            if not hasattr(self, 'display_buffer_manager') or self.display_buffer_manager is None:
                self.display_buffer_manager = DisplayBufferManager(self.buffer_coordinator)
                logger.info("Created new DisplayBufferManager (universal instance)")
            else:
                logger.info("Reusing existing DisplayBufferManager (worker restart)")
            
            # Create control queues for processing worker
            # CRITICAL FIX: Use bounded queues to prevent queue overflow crashes
            # Unbounded queues can grow indefinitely, causing memory exhaustion and worker crashes
            # Setting maxsize=500 provides backpressure if consumer can't keep up with producer
            # INCREASED from 100 to 500 to handle high-frequency performance messages + critical ready status
            self.processing_control_queue = MPQueue(maxsize=100)
            self.processing_status_queue = MPQueue(maxsize=500)  # Increased for performance messages
            self.pose_data_queue = MPQueue(maxsize=100)  # For pose data from worker to GUI
            
            # FIXED: Don't create display buffers here - the worker creates them
            # This was causing duplication (GUI created yq_display_0_55100, worker created yq_display_0_48580)
            # The worker will create the display buffers and send them back in the ready status
            logger.info("Skipping GUI display buffer creation - worker will create them")
            
            # CRITICAL: Always ensure correlation buffer exists for correlator output
            # Correlator runs whenever 2+ faces detected, independent of LSL streaming
            if not hasattr(self, 'correlation_buffer') or self.correlation_buffer is None:
                if not hasattr(self, 'lsl_data_queue') or self.lsl_data_queue is None:
                    logger.info("LSL helper not initialized yet - initializing now (includes correlation buffer)")
                    self._initialize_lsl_helper()
                else:
                    # LSL helper exists but correlation buffer doesn't - recreate just the buffer
                    logger.info("Recreating correlation buffer (LSL helper already active)")
                    import multiprocessing.shared_memory as shared_memory
                    self.correlation_buffer = shared_memory.SharedMemory(create=True, size=52 * 4)
                    self.corr_array = np.ndarray((52,), dtype=np.float32, buffer=self.correlation_buffer.buf)
            elif not hasattr(self, 'lsl_data_queue') or self.lsl_data_queue is None:
                # Correlation buffer exists but LSL helper doesn't - initialize helper
                logger.info("LSL helper not initialized yet - initializing now (correlation buffer already exists)")
                self._initialize_lsl_helper()

            # Start GUI processing worker
            try:
                self.gui_processing_worker = Process(
                    target=gui_processing_worker_main,
                    args=(
                        self._get_buffer_coordinator_info(),
                        self.config.config,
                        active_cameras,
                        self.recovery_buffer_names,
                        self.processing_control_queue,
                        self.processing_status_queue,
                        self.lsl_data_queue,  # CRITICAL: Pass queue at creation, not via control message
                        self.pose_data_queue,  # Pose data from worker to GUI
                        self.participant_update_queue,  # CRITICAL FIX: Enable enrollment state updates in worker
                        self.enrollment_state_array,  # ARRAY-BASED IPC: Shared enrollment state for overlay colors
                        self.lock_state_array,  # ARRAY-BASED IPC: Shared lock state for slot reservation
                        self.participant_presence_array,  # ARRAY-BASED IPC: Shared presence state for Absent/Present display
                        self.ENROLLMENT_STATES  # Pass state name->code mapping to worker
                    )
                )

                # VERBOSE: Log worker start details
                logger.info(f"[GUI] About to start worker process")
                logger.info(f"[GUI]   Target function: {gui_processing_worker_main.__name__}")
                logger.info(f"[GUI]   Active cameras: {active_cameras}")

                self.gui_processing_worker.start()

                logger.info("GUI processing worker started")
                logger.info(f"[GUI]   Worker PID: {self.gui_processing_worker.pid}")
                logger.info(f"[GUI]   Worker alive: {self.gui_processing_worker.is_alive()}")

                # CRITICAL FIX: Check for early worker crash (import errors, etc.)
                # Worker may crash during module import before logging is configured
                def check_worker_alive():
                    if not self.gui_processing_worker.is_alive():
                        logger.error("[WORKER FATAL] ❌ GUI worker process died immediately after start!")
                        logger.error(f"[WORKER FATAL] Exit code: {self.gui_processing_worker.exitcode}")
                        logger.error("[WORKER FATAL] Check logs above for import errors or initialization failures")
                        logger.error(f"[WORKER FATAL] Possible causes:")
                        logger.error(f"[WORKER FATAL]   1. Module import error in child process")
                        logger.error(f"[WORKER FATAL]   2. Argument pickling failure")
                        logger.error(f"[WORKER FATAL]   3. CUDA initialization error in child process")
                        logger.error(f"[WORKER FATAL]")
                        logger.error(f"[WORKER FATAL] Target function: {gui_processing_worker_main}")
                        logger.error(f"[WORKER FATAL] Function module: {gui_processing_worker_main.__module__}")
                        logger.error(f"[WORKER FATAL] Function qualname: {gui_processing_worker_main.__qualname__}")
                        self._worker_ready_flag = False
                        # Try to show error dialog to user
                        try:
                            import tkinter.messagebox as mb
                            mb.showerror("Worker Crashed",
                                        "GUI processing worker crashed during startup.\n\n"
                                        "This is usually caused by import errors or\n"
                                        "CUDA initialization in child process.\n\n"
                                        "Check the console logs for details.")
                        except:
                            pass  # GUI might not be fully initialized yet
                    else:
                        # Worker still alive - check again after 2 more seconds for delayed crash
                        logger.info(f"[WORKER ALIVE CHECK] Worker still alive after 1s (PID {self.gui_processing_worker.pid})")

                        def check_worker_alive_2s():
                            if not self.gui_processing_worker.is_alive():
                                logger.error("[WORKER FATAL] ❌ Worker crashed 2-3 seconds after start (likely during initialization)")
                                logger.error(f"[WORKER FATAL] Exit code: {self.gui_processing_worker.exitcode}")
                            else:
                                logger.info(f"[WORKER ALIVE CHECK] Worker still alive after 3s - likely healthy")

                        self.after(2000, check_worker_alive_2s)  # Check again after 2 more seconds

                self.after(1000, check_worker_alive)  # Check after 1 second

                # CRITICAL FIX: Start worker health monitoring
                # Monitors worker every 5 seconds and auto-restarts if dead
                logger.info("[WORKER MONITOR] Starting health monitoring (check every 5s)")
                self.after(5000, self._monitor_worker_health)

            except Exception as worker_error:
                logger.error(f"Failed to start GUI processing worker: {worker_error}")
                return False
            
            # Send actual buffer names to worker immediately after starting
            if hasattr(self, 'actual_buffer_names') and self.actual_buffer_names:
                logger.info(f"[GUI] Sending actual buffer names to worker: {list(self.actual_buffer_names.keys())}")

                # CRITICAL FIX: Get camera resolutions from BufferCoordinator which has the actual values
                # This ensures we send the correct resolutions that match the created buffers
                camera_resolutions_to_send = {}
                if hasattr(self, 'buffer_coordinator') and hasattr(self.buffer_coordinator, 'camera_resolutions'):
                    camera_resolutions_to_send = self.buffer_coordinator.camera_resolutions.copy()
                    logger.info(f"[GUI] Using BufferCoordinator resolutions: {camera_resolutions_to_send}")
                else:
                    # Fallback to GUI's camera_resolutions if available
                    camera_resolutions_to_send = getattr(self, 'camera_resolutions', {})
                    if camera_resolutions_to_send:
                        logger.info(f"[GUI] Using GUI resolutions (fallback): {camera_resolutions_to_send}")
                    else:
                        logger.warning("[GUI] No camera resolutions available - worker will use config defaults")

                self.processing_control_queue.put({
                    'type': 'buffer_names',
                    'actual_buffer_names': self.actual_buffer_names,
                    'camera_resolutions': camera_resolutions_to_send
                })
            else:
                logger.warning("[GUI] No actual_buffer_names available yet - worker will wait for them")

            # Send correlation buffer connection command to worker
            if hasattr(self, 'correlation_buffer') and self.correlation_buffer:
                self.processing_control_queue.put({
                    'type': 'connect_correlation_buffer',
                    'buffer_name': self.correlation_buffer.name
                })
                logger.info(f"[GUI] Sent correlation buffer name to worker: {self.correlation_buffer.name}")
            
            # ASYNC NON-BLOCKING: Wait for worker to be ready
            # This DOES NOT block the Tkinter event loop - GUI remains responsive
            logger.info("[WORKER INIT] Starting async wait for GUI processing worker initialization (max 60s)")

            # Start async wait (returns immediately, callback fires when ready)
            # Note: Worker initialization happens asynchronously in the background
            # Timeout increased to 60s to accommodate slow recovery buffer connections (20-30s)
            self._wait_for_worker_ready_async(timeout=60.0, poll_interval=100)

            # Fast display mode is now always active (no boolean needed)
            logger.info("Fast display processing activated successfully")

            # CRITICAL: Returns True immediately after starting worker process
            # Display loop will start LATER in _wait_for_worker_ready_async() callback (line ~3103)
            # when worker sends 'ready' status and display_readers dict is populated
            # DO NOT start display loop here - it will run with empty display_readers!
            return True
            
        except Exception as e:
            logger.error(f"Critical error in initialize_display_processing: {e}")
            import traceback
            logger.error(f"Initialize display processing traceback: {traceback.format_exc()}")
            # Ensure cleanup on any failure
            try:
                self.shutdown_display_processing()
            except:
                pass
            return False

    def _get_buffer_coordinator_info(self):
        """Get serializable info about BufferCoordinator for worker process."""
        # Include complete buffer registry so worker can connect to buffers
        info = {
            'camera_count': self.buffer_coordinator.camera_count,
            'buffer_sizes': self.buffer_coordinator.get_buffer_sizes(),
            'buffer_registry': self.buffer_coordinator.buffer_registry.copy(),
            'camera_resolutions': self.buffer_coordinator.camera_resolutions.copy(),  # CRITICAL: Pass actual resolutions
            'max_faces': self.buffer_coordinator.max_faces,
            'ring_buffer_size': self.buffer_coordinator.ring_buffer_size,
            'roi_buffer_size': self.buffer_coordinator.roi_buffer_size,
            'gui_buffer_size': self.buffer_coordinator.gui_buffer_size,
            'results_buffer_layout': self.buffer_coordinator.get_results_buffer_layout(),
            'roi_buffer_layout': self.buffer_coordinator.get_roi_buffer_layout(0)  # Use camera 0 as reference (layout is same for all cameras)
        }
        
        # Include actual buffer names captured from camera status updates
        if hasattr(self, 'actual_buffer_names'):
            info['actual_buffer_names'] = self.actual_buffer_names.copy()
            logger.info(f"Including actual buffer names for {len(self.actual_buffer_names)} cameras")
        else:
            info['actual_buffer_names'] = {}
            logger.warning("No actual buffer names captured yet - worker may fail to connect")
        
        # Log what we're passing
        logger.info(f"Passing buffer info to worker: {len(info['buffer_registry'])} cameras registered")
        for cam_idx, buffers in info['buffer_registry'].items():
            logger.info(f"  Camera {cam_idx}: {list(buffers.keys())} buffers")
            
        return info

    def _get_camera_display_name(self, cam_idx: int) -> str:
        """
        Get user-friendly camera display name.

        Tries to get the camera name from the running camera source.
        Falls back to generic name if camera is not running or source unavailable.

        Args:
            cam_idx: Camera device index

        Returns:
            str: Display name like "HD Webcam" or "Camera 0"
        """
        try:
            # Check if camera worker is running
            if not hasattr(self, 'camera_worker_manager'):
                return f"Camera {cam_idx}"

            # Try to get camera worker
            if cam_idx not in self.camera_worker_manager.workers:
                return f"Camera {cam_idx}"

            worker = self.camera_worker_manager.workers[cam_idx]

            # Try to access camera source through worker
            # Note: camera_source is an attribute of EnhancedCameraWorker
            if hasattr(worker, 'camera_source') and worker.camera_source:
                if hasattr(worker.camera_source, 'get_camera_name'):
                    camera_name = worker.camera_source.get_camera_name()
                    if camera_name and camera_name != f"Camera {cam_idx}":
                        # Return name with index for clarity
                        return f"Camera {cam_idx}: {camera_name}"

        except Exception as e:
            logger.debug(f"Could not get camera name for camera {cam_idx}: {e}")

        # Fallback
        return f"Camera {cam_idx}"

    def _get_latest_poses(self, cam_idx: int) -> List[Dict]:
        """
        Get latest pose data for a camera by draining the pose_data_queue.
        Returns the most recent pose data for the specified camera.

        FIX: Added timestamp-based staleness detection to prevent frozen overlay.
        Cache entries older than 1 second are discarded.
        """
        if not hasattr(self, 'pose_data_queue'):
            return []

        # Drain the queue and keep only the latest data per camera
        # DEBUG: Track how many messages are drained
        messages_drained = 0
        latest_frame_id = -1
        try:
            while True:
                pose_msg = self.pose_data_queue.get(block=False)
                cam_id = pose_msg.get('camera_idx')
                frame_id = pose_msg.get('frame_id', -1)
                # FIX: Store with timestamp for staleness detection
                self._cached_pose_data[cam_id] = {
                    'poses': pose_msg.get('poses', []),
                    'timestamp': time.time(),
                    'frame_id': frame_id,
                    'metrics': pose_msg.get('metrics'),  # Cache metrics for metrics panel
                    'performance': pose_msg.get('performance')  # Cache performance metrics (FPS, latency)
                }
                messages_drained += 1
                if cam_id == cam_idx:
                    latest_frame_id = frame_id
        except:
            pass  # Queue empty

        # Queue drain activity tracked silently (no per-frame logs)

        # FIX: Check cache staleness and clear if >1 second old
        cache_entry = self._cached_pose_data.get(cam_idx)
        if cache_entry:
            age = time.time() - cache_entry['timestamp']
            if age > 1.0:
                # Cache is stale - clear it to prevent frozen overlay
                if not hasattr(self, '_pose_cache_stale_logged'):
                    self._pose_cache_stale_logged = {}
                if cam_idx not in self._pose_cache_stale_logged or time.time() - self._pose_cache_stale_logged[cam_idx] > 5.0:
                    logger.warning(f"[POSE CACHE] Camera {cam_idx}: Clearing stale pose data (age={age:.1f}s, frame_id={cache_entry['frame_id']})")
                    self._pose_cache_stale_logged[cam_idx] = time.time()
                return []
            return cache_entry['poses']

        # No cache entry
        return []

    def _update_3d_skeleton_display(self):
        """
        Update the 3D skeleton visualization canvas with latest pose data.
        Called periodically to render 3D skeleton plot from pose estimation.

        Features:
        - Real-time 3D skeleton rendering from pose data
        - Test mode with synthetic pose data for verification
        - Debug logging to track data availability
        - Visual feedback when no data is present
        """
        try:
            # Enable test mode to verify visualization works (set to False for production)
            TEST_MODE = False  # Set to False for production (use real pose data)

            # Get latest pose data for selected camera (with intelligent fallback)
            if self._current_camera_index is not None and self._current_camera_index in self.active_camera_procs:
                cam_idx = self._current_camera_index
            elif self.active_camera_procs:
                # Fallback to first active camera if selection invalid
                cam_idx = list(self.active_camera_procs.keys())[0]
                logger.warning(f"[3D SKELETON] Selected camera invalid, using camera {cam_idx}")
            else:
                # No active cameras
                if not hasattr(self, '_no_data_count'):
                    self._no_data_count = 0
                self._no_data_count += 1
                if self._no_data_count % 10 == 0:
                    logger.debug(f"[3D SKELETON] No active cameras ({self._no_data_count/10:.1f}s elapsed)")
                self.after(100, self._update_3d_skeleton_display)
                return

            # Log camera being displayed (only when it changes)
            if not hasattr(self, '_skeleton_display_camera') or self._skeleton_display_camera != cam_idx:
                logger.info(f"[3D SKELETON] Now displaying pose data from camera {cam_idx}")
                self._skeleton_display_camera = cam_idx

            poses = self._get_latest_poses(cam_idx)

            # Debug logging (only log state changes, not every frame)
            if not hasattr(self, '_last_pose_state'):
                self._last_pose_state = None
                self._no_data_count = 0

            current_state = len(poses) if poses else 0

            # Use test data if in TEST_MODE or if no real data available
            if TEST_MODE or not poses or len(poses) == 0:
                # Increment no-data counter
                self._no_data_count += 1

                # Log state change or periodic reminder
                if self._last_pose_state != 0:
                    if TEST_MODE:
                        logger.info("[3D SKELETON] TEST_MODE enabled - using animated test data instead of real poses")
                    else:
                        logger.info("[3D SKELETON] No pose data available - waiting for pose estimation...")
                    self._last_pose_state = 0
                elif self._no_data_count % 100 == 0:  # Log every 10 seconds (100 frames at 10 FPS)
                    if TEST_MODE:
                        logger.debug(f"[3D SKELETON] Still in TEST_MODE - showing dummy animation ({self._no_data_count/10:.1f}s elapsed)")
                    else:
                        logger.debug(f"[3D SKELETON] Still waiting for pose data... ({self._no_data_count/10:.1f}s elapsed)")

                if TEST_MODE:
                    # Generate realistic test skeleton data for visualization verification
                    poses = [self._generate_test_skeleton_data()]
                else:
                    # Show "waiting for data" message on canvas
                    self._show_skeleton_waiting_message()
                    self.after(100, self._update_3d_skeleton_display)
                    return
            else:
                # Reset counter and log when data becomes available
                if self._last_pose_state == 0:
                    logger.info(f"[3D SKELETON] ✓ Pose data received! Rendering REAL POSE from {len(poses)} person(s)")
                    self._last_pose_state = len(poses)
                    self._no_data_count = 0
                elif self._last_pose_state != len(poses):
                    # Log when number of detected persons changes
                    logger.info(f"[3D SKELETON] Person count changed: {self._last_pose_state} -> {len(poses)}")
                    self._last_pose_state = len(poses)

            # Get first person's pose data (we're only showing one person)
            pose_data = poses[0]

            # Extract keypoints (already numpy array with shape (133, 4))
            # Format: pose_data is dict with 'keypoints' numpy array [x, y, z, confidence]
            if 'keypoints' not in pose_data:
                logger.warning("[3D SKELETON] Pose data missing 'keypoints' key")
                self.after(100, self._update_3d_skeleton_display)
                return

            # Keypoints is already numpy array (133, 4)
            keypoints = pose_data['keypoints']  # Numpy array (133, 4)

            # Validate data
            if keypoints.shape[0] < 17:
                logger.warning(f"[3D SKELETON] Insufficient keypoints: {keypoints.shape[0]} keypoints")
                self.after(100, self._update_3d_skeleton_display)
                return

            # Split into coordinates and scores using numpy slicing
            keypoints_3d = keypoints[:, :3].astype(np.float32)  # (133, 3) - x, y, z
            scores = keypoints[:, 3].astype(np.float32)          # (133,) - confidence

            logger.debug(f"[3D SKELETON] Rendering skeleton: {keypoints_3d.shape[0]} keypoints, avg confidence: {scores.mean():.2f}")

            # Force canvas to realize its geometry (critical for accurate dimensions)
            self.skeleton_canvas.update_idletasks()

            # Check if canvas is visible (mapped to screen)
            if not self.skeleton_canvas.winfo_viewable():
                logger.debug("[3D SKELETON] Canvas not viewable yet, forcing update")
                self.skeleton_canvas.update()

            # Get canvas dimensions for dynamic sizing (AFTER realization)
            canvas_w = self.skeleton_canvas.winfo_width()
            canvas_h = self.skeleton_canvas.winfo_height()

            # Validate canvas dimensions - ABORT if invalid (don't use fallback)
            if canvas_w <= 1 or canvas_h <= 1:
                logger.debug(f"[3D SKELETON] Canvas not ready: {canvas_w}x{canvas_h}, skipping frame")
                self.after(100, self._update_3d_skeleton_display)
                return

            # Calculate target size (square aspect ratio for 3D plot, use smaller dimension)
            # Use 90% of canvas size to leave some padding
            target_size = int(min(canvas_w, canvas_h) * 0.9)

            # Apply reasonable limits to prevent memory issues and maintain quality
            MIN_SIZE = 400
            MAX_SIZE = 1200
            target_size = max(MIN_SIZE, min(MAX_SIZE, target_size))

            # Calculate matplotlib figsize and DPI
            # Use fixed DPI for consistent quality, adjust figsize to match target pixels
            target_dpi = 100
            target_figsize_inches = target_size / target_dpi

            logger.info(f"[3D SKELETON] Canvas: {canvas_w}x{canvas_h}, Target: {target_size}x{target_size}, Figsize: {target_figsize_inches:.1f}in @ {target_dpi} DPI")

            # Generate 3D skeleton plot as PIL Image with dynamic sizing
            # Get current view angle from interactive control panel
            current_view_angle = self.skeleton_3d_controls.get_view_angle()

            # Get ear_to_nose_drop_ratio from centralized config (single source of truth)
            ear_to_nose_ratio = self.config.get('metrics_settings.ear_to_nose_drop_ratio', 0.20)

            skeleton_img = plot_3d_skeleton(
                keypoints=keypoints_3d,
                scores=scores,
                min_confidence=0.3,
                view_angle=current_view_angle,
                figsize=(target_figsize_inches, target_figsize_inches),
                dpi=target_dpi,
                ear_to_nose_drop_ratio=ear_to_nose_ratio
            )

            # Defensive resize: Ensure image fits within target bounds (handle matplotlib margins)
            img_w, img_h = skeleton_img.size
            if img_w > target_size or img_h > target_size:
                logger.debug(f"[3D SKELETON] Matplotlib margins detected - resizing {img_w}x{img_h} → {target_size}x{target_size}")
                skeleton_img = skeleton_img.resize(
                    (target_size, target_size),
                    Image.Resampling.LANCZOS
                )

            # Create canvas-sized background image (matches camera feed approach)
            # This guarantees PhotoImage = Canvas size, preventing overflow
            canvas_img_full = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            # Convert PIL Image to numpy array for pasting
            skeleton_np = np.array(skeleton_img)
            plot_h, plot_w = skeleton_np.shape[:2]

            # Calculate centering offsets (can be negative if plot > canvas)
            x_offset_raw = (canvas_w - plot_w) // 2
            y_offset_raw = (canvas_h - plot_h) // 2

            # Handle cases where plot is larger than canvas
            # Source region (plot): which part of the plot to use
            src_x_start = max(0, -x_offset_raw)
            src_y_start = max(0, -y_offset_raw)
            src_x_end = min(plot_w, canvas_w + src_x_start)
            src_y_end = min(plot_h, canvas_h + src_y_start)

            # Destination region (canvas): where to paste on canvas
            dst_x_start = max(0, x_offset_raw)
            dst_y_start = max(0, y_offset_raw)
            dst_x_end = dst_x_start + (src_x_end - src_x_start)
            dst_y_end = dst_y_start + (src_y_end - src_y_start)

            # Paste skeleton plot centered on black background
            canvas_img_full[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                skeleton_np[src_y_start:src_y_end, src_x_start:src_x_end]

            # Convert canvas-sized image to PIL then PhotoImage
            pil_img = Image.fromarray(canvas_img_full, mode='RGB')
            photo = ImageTk.PhotoImage(pil_img)

            # Store reference to prevent garbage collection
            if not hasattr(self, '_skeleton_photo_ref'):
                self._skeleton_photo_ref = None
            self._skeleton_photo_ref = photo

            # Update canvas
            if not hasattr(self, '_skeleton_canvas_image_id'):
                self._skeleton_canvas_image_id = None

            # Clear placeholder text on first update
            if not hasattr(self, '_skeleton_placeholder_cleared'):
                self.skeleton_canvas.delete('placeholder')
                self.skeleton_canvas.delete('waiting_msg')  # Also clear waiting message
                self._skeleton_placeholder_cleared = True
                logger.info("[3D SKELETON] ✓ Canvas initialized and rendering")

            # Delete old image before creating new (prevents staleness)
            if hasattr(self, '_skeleton_canvas_image_id') and self._skeleton_canvas_image_id is not None:
                try:
                    self.skeleton_canvas.delete(self._skeleton_canvas_image_id)
                except:
                    pass  # Image might already be deleted

            # Create new image at canvas center (use dimensions from line 4418-4419)
            self._skeleton_canvas_image_id = self.skeleton_canvas.create_image(
                canvas_w // 2, canvas_h // 2,
                image=photo,
                anchor='center'
            )

        except Exception as e:
            logger.error(f"[3D SKELETON] Error updating 3D skeleton display: {e}")
            import traceback
            traceback.print_exc()

        # Schedule next update (10 FPS = 100ms interval for quality rendering)
        self.after(100, self._update_3d_skeleton_display)

    def _show_skeleton_waiting_message(self):
        """Show a waiting message on the skeleton canvas when no data is available."""
        # Clear previous waiting message if it exists
        self.skeleton_canvas.delete('waiting_msg')

        # Get canvas dimensions
        canvas_w = self.skeleton_canvas.winfo_width() if self.skeleton_canvas.winfo_width() > 1 else 800
        canvas_h = self.skeleton_canvas.winfo_height() if self.skeleton_canvas.winfo_height() > 1 else 600

        # Create waiting message
        self.skeleton_canvas.create_text(
            canvas_w // 2, canvas_h // 2,
            text="Waiting for pose data...\n\nStart pose estimation to see 3D visualization",
            fill='#888888',
            font=('Arial', 12),
            justify='center',
            tags='waiting_msg'
        )

    def _generate_test_skeleton_data(self):
        """
        Generate realistic test skeleton data for visualization verification.

        Returns a pose dict with anatomically plausible 3D coordinates.
        Used to verify that the 3D rendering pipeline is working correctly.
        """
        import math

        # Animate the test skeleton with a breathing motion
        if not hasattr(self, '_test_skeleton_time'):
            self._test_skeleton_time = 0
        self._test_skeleton_time += 0.1

        breathe = math.sin(self._test_skeleton_time) * 0.05  # Breathing motion
        sway = math.cos(self._test_skeleton_time * 0.5) * 0.1  # Gentle sway

        # Base position (person standing at ~5m depth, centered)
        base_z = 5.0

        # COCO-17 body keypoints in anatomically correct positions
        # Format: [x (left-right), y (up-down), z (depth)]
        landmarks = [
            # Head (0-4)
            (0.0 + sway, 1.6, base_z),           # 0: nose
            (0.05 + sway, 1.65, base_z - 0.05),  # 1: left eye
            (-0.05 + sway, 1.65, base_z - 0.05), # 2: right eye
            (0.08 + sway, 1.55, base_z + 0.05),  # 3: left ear
            (-0.08 + sway, 1.55, base_z + 0.05), # 4: right ear

            # Upper body (5-6)
            (0.2 + sway, 1.3 + breathe, base_z),  # 5: left shoulder
            (-0.2 + sway, 1.3 + breathe, base_z), # 6: right shoulder

            # Arms (7-10)
            (0.3 + sway, 1.0, base_z + 0.1),      # 7: left elbow
            (-0.3 + sway, 1.0, base_z + 0.1),     # 8: right elbow
            (0.35 + sway, 0.7, base_z + 0.15),    # 9: left wrist
            (-0.35 + sway, 0.7, base_z + 0.15),   # 10: right wrist

            # Torso (11-12)
            (0.15 + sway, 0.9 + breathe, base_z), # 11: left hip
            (-0.15 + sway, 0.9 + breathe, base_z),# 12: right hip

            # Legs (13-16)
            (0.15 + sway, 0.5, base_z),           # 13: left knee
            (-0.15 + sway, 0.5, base_z),          # 14: right knee
            (0.15 + sway, 0.1, base_z),           # 15: left ankle
            (-0.15 + sway, 0.1, base_z),          # 16: right ankle
        ]

        # Extend to 33 keypoints (add feet and hand points with lower confidence)
        # Feet keypoints
        for i in range(6):
            landmarks.append((sway, 0.0, base_z))  # Foot points at ground level

        # Hand keypoints (10 points each hand = 20 total)
        # Left hand
        for i in range(10):
            offset = i * 0.02
            landmarks.append((0.35 + sway + offset, 0.7 - offset * 0.5, base_z + 0.2))

        # Right hand
        for i in range(10):
            offset = i * 0.02
            landmarks.append((-0.35 + sway - offset, 0.7 - offset * 0.5, base_z + 0.2))

        # High confidence for body keypoints, medium for extremities
        visibility = (
            [0.95] * 17 +      # High confidence for main body (COCO-17)
            [0.7] * 6 +        # Medium for feet
            [0.6] * 10 +       # Medium-low for left hand
            [0.6] * 10         # Medium-low for right hand
        )

        return {
            'landmarks': landmarks[:33],  # Ensure exactly 33 keypoints
            'visibility': visibility[:33],
            'centroid': (0.0, 1.3),
            'pose_resolution': (640, 480)
        }

    def _read_display_ready_data(self, cam_idx: int) -> Optional[Dict[str, Any]]:
        """
        Read display-ready data from worker's display buffer.

        Returns processed frames with overlays if available, None otherwise.
        Worker health monitoring handles crashes/freezes separately (no fallback needed).

        SIMPLIFIED: Removed staleness check and fallback logic to eliminate bugs.
        Worker restart is handled by _monitor_worker_health() which runs every 5 seconds.
        """
        try:
            # DIAGNOSTIC: Check if display_buffer_manager exists and is valid
            if not hasattr(self, 'display_buffer_manager'):
                logger.error(f"[DISPLAY DATA] Camera {cam_idx}: display_buffer_manager attribute does NOT EXIST!")
                return None

            if self.display_buffer_manager is None:
                logger.error(f"[DISPLAY DATA] Camera {cam_idx}: display_buffer_manager is None!")
                return None

            # DIAGNOSTIC: Check if camera is in display_readers
            if hasattr(self.display_buffer_manager, 'display_readers'):
                if cam_idx not in self.display_buffer_manager.display_readers:
                    logger.error(f"[DISPLAY DATA] Camera {cam_idx}: NOT in display_readers! Keys: {list(self.display_buffer_manager.display_readers.keys())}")
                    return None

            # Log successful pre-checks (every 30 calls)
            if not hasattr(self, '_display_data_call_count'):
                self._display_data_call_count = {}
            if cam_idx not in self._display_data_call_count:
                self._display_data_call_count[cam_idx] = 0
            self._display_data_call_count[cam_idx] += 1

            # Read from display buffer (worker output)
            if hasattr(self, 'display_buffer_manager') and self.display_buffer_manager:
                display_data = self.display_buffer_manager.get_display_data(cam_idx)

                if display_data and display_data.frame_bgr is not None:
                    # SUCCESS: Worker has processed data available
                    faces_data = []
                    if display_data.faces:
                        for face in display_data.faces:
                            faces_data.append({
                                'bbox': face.get('bbox', [0, 0, 0, 0]),
                                'landmarks': face.get('landmarks'),
                                'label': face.get('label', ''),
                                'track_id': face.get('track_id', -1),
                                'participant_id': face.get('participant_id', -1),  # CRITICAL FIX: Include participant_id for enrollment state lookup
                                'display_flags': face.get('display_flags', 0)      # CRITICAL FIX: Include display_flags for enrollment state encoding
                            })

                    all_poses = self._get_latest_poses(cam_idx)

                    # Pose drawing tracked silently (no per-frame logs)

                    return {
                        'frame': display_data.frame_bgr,
                        'faces': faces_data,
                        'frame_id': display_data.frame_id,
                        'timestamp': display_data.timestamp,
                        'all_poses': all_poses
                    }

            # No data available yet (startup or worker temporarily behind)
            return None

        except Exception as e:
            logger.error(f"Error reading display data for camera {cam_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _deserialize_display_faces(self, buffer: bytes, num_faces: int) -> List[Dict]:
        """Deserialize pre-processed face data for display."""
        import struct
        faces = []
        offset = 0
        
        for i in range(num_faces):
            try:
                if offset + 28 > len(buffer):  # Basic face data size
                    break
                    
                # Unpack basic face data
                track_id, participant_id, x1, y1, x2, y2, label_len = struct.unpack(
                    'IIffffI', buffer[offset:offset + 28]
                )
                offset += 28
                
                # Read label string
                if offset + label_len > len(buffer):
                    break
                label = buffer[offset:offset + label_len].decode('utf-8')
                offset += label_len
                
                face_data = {
                    'track_id': track_id,
                    'participant_id': participant_id,
                    'bbox': [x1, y1, x2, y2],
                    'label': label
                }
                
                # Check for landmark data (478 landmarks * 3 coords * 4 bytes = 5736 bytes)
                # Updated: GUI Processing Worker now serializes as 478x3 (x,y,z) including iris
                expected_landmark_size = 478 * 3 * 4
                if offset + expected_landmark_size <= len(buffer):
                    landmark_bytes = buffer[offset:offset + expected_landmark_size]
                    # Updated to handle 3D landmarks (478x3)
                    landmarks = np.frombuffer(landmark_bytes, dtype=np.float32).reshape(478, 3)
                    face_data['landmarks'] = landmarks
                    offset += expected_landmark_size
                
                faces.append(face_data)
                
            except Exception as e:
                logger.error(f"Error deserializing face {i}: {e}")
                break
                
        return faces

    def _read_raw_camera_frame(self, cam_idx: int) -> Optional[np.ndarray]:
        """
        FALLBACK: Read raw frame directly from camera frame buffer.
        Used when display buffer unavailable or stale (worker not ready yet).
        Ensures video always displays even without worker processing.

        Args:
            cam_idx: Camera index

        Returns:
            BGR frame as numpy array, or None if unavailable
        """
        try:
            # Check if we have the frame buffer name for this camera
            if not hasattr(self, 'actual_buffer_names') or cam_idx not in self.actual_buffer_names:
                return None

            frame_buffer_name = self.actual_buffer_names[cam_idx].get('frame')
            if not frame_buffer_name:
                return None

            # Connect to frame buffer if not already connected
            if not hasattr(self, '_raw_frame_buffers'):
                self._raw_frame_buffers = {}

            if cam_idx not in self._raw_frame_buffers:
                import multiprocessing.shared_memory as mp_shm
                try:
                    shm = mp_shm.SharedMemory(name=frame_buffer_name)
                    self._raw_frame_buffers[cam_idx] = shm
                    logger.info(f"[FALLBACK] Connected to raw frame buffer for camera {cam_idx}: {frame_buffer_name}")
                except Exception as e:
                    logger.error(f"[FALLBACK] Failed to connect to frame buffer {frame_buffer_name}: {e}")
                    return None

            shm = self._raw_frame_buffers[cam_idx]

            # CRITICAL FIX: Read ACTUAL resolution from buffer header (same as display_buffer_manager)
            # Buffer header layout: [write_index(8)][width(4)][height(4)]
            # This ensures we use the correct resolution even if cache is stale
            actual_res = np.frombuffer(shm.buf, dtype=np.uint32, count=2, offset=8)
            width, height = int(actual_res[0]), int(actual_res[1])

            # Validate resolution is reasonable
            if width <= 0 or height <= 0 or width > 7680 or height > 4320:
                logger.error(f"[FALLBACK] Invalid resolution from buffer header: {width}x{height}")
                return None

            # CRITICAL FIX: Get proper buffer layout from BufferCoordinator
            # This includes metadata offsets that the old code ignored!
            # The camera frame buffer has metadata between frames, not just raw frame data
            layout = self.buffer_coordinator.get_buffer_layout(cam_idx, (width, height))
            if not layout:
                logger.error(f"[FALLBACK] No layout returned from coordinator for camera {cam_idx}")
                return None

            # Use detection_frame_offsets from layout (NOT simple offset calculation!)
            detection_frame_offsets = layout.get('detection_frame_offsets', [])
            if not detection_frame_offsets:
                logger.error(f"[FALLBACK] No detection_frame_offsets in layout for camera {cam_idx}")
                return None

            # Read write_index to find latest frame
            write_index_bytes = bytes(shm.buf[0:8])
            write_index = struct.unpack('Q', write_index_bytes)[0]

            if write_index == 0:
                return None  # No frames written yet

            # Calculate slot position in ring buffer
            ring_buffer_size = len(detection_frame_offsets)
            slot = (write_index - 1) % ring_buffer_size

            # Use CORRECT offset from layout (includes metadata space!)
            frame_offset = detection_frame_offsets[slot]
            frame_size = width * height * 3  # BGR

            # Validate bounds before reading
            if frame_offset + frame_size > shm.size:
                logger.error(f"[FALLBACK] Frame read would exceed buffer: offset={frame_offset}, size={frame_size}, buffer={shm.size}")
                return None

            # Read frame data
            frame_bytes = bytes(shm.buf[frame_offset:frame_offset + frame_size])
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))

            # Return copy to avoid buffer lifecycle issues
            return frame.copy()

        except Exception as e:
            logger.error(f"[FALLBACK] Error reading raw frame for camera {cam_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _schedule_optimized_display(self):
        """
        Optimized display using pre-computed data from processing worker.
        This should achieve 30+ FPS by only doing rendering, no processing.
        """
        # DIAGNOSTIC: Track method calls to confirm loop is running
        if not hasattr(self, '_display_method_calls'):
            self._display_method_calls = 0
        self._display_method_calls += 1

        # DIAGNOSTIC DISABLED: Uncomment to debug display loop
        # if self._display_method_calls % 30 == 0:
        #     logger.info(f"[DISPLAY METHOD] _schedule_optimized_display() called #{self._display_method_calls}")

        # Reschedule next update first (60 FPS target)
        if getattr(self, '_optimized_display_active', True) and not getattr(self, 'shutdown_flag', False):
            self.after(16, self._schedule_optimized_display)
        else:
            logger.info("[DISPLAY] Optimized display loop stopped")
            return

        # SIMPLIFIED: No longer wait for worker ready flag
        # Display loop starts immediately and shows raw frames
        # Overlays appear progressively when worker makes them available

        # Track timing for performance monitoring
        preview_start_time = time.time()
        cameras_processed = 0

        # DIAGNOSTIC: Log display loop entry and active cameras
        active_cams = list(self.active_camera_procs.keys())

        # DIAGNOSTIC DISABLED: Uncomment to debug active cameras
        # if self._display_method_calls % 30 == 0:
        #     logger.info(f"[DISPLAY METHOD] active_cams = {active_cams}, len = {len(active_cams)}")

        if not hasattr(self, '_display_loop_diagnostic_logged'):
            # DIAGNOSTIC DISABLED: Uncomment to debug display loop initialization
            # logger.info(f"[DISPLAY DIAGNOSTIC] Display loop executing")
            # logger.info(f"[DISPLAY DIAGNOSTIC] Active cameras: {active_cams}")
            # logger.info(f"[DISPLAY DIAGNOSTIC] Number of frames: {len(self.frames)}")
            # CRITICAL FIX: Do NOT call combo.get() from within after() callback
            # Calling .get() on Tkinter widgets from within event loop can corrupt the callback queue
            # and prevent subsequent iterations from executing (causes display loop to stop)
            self._display_loop_diagnostic_logged = True

        # FIX: Process active cameras directly instead of relying on combo box values
        # Active cameras are tracked in self.active_camera_procs which is reliable
        # Make defensive copy to prevent modification during iteration
        active_cams_copy = active_cams.copy()

        for cam_idx in active_cams_copy:
            # DIAGNOSTIC: Track camera loop iterations (log every 30 iterations)
            if not hasattr(self, '_camera_loop_iterations'):
                self._camera_loop_iterations = {}
            if cam_idx not in self._camera_loop_iterations:
                self._camera_loop_iterations[cam_idx] = 0
            self._camera_loop_iterations[cam_idx] += 1

            # DIAGNOSTIC DISABLED: Uncomment to debug camera loop iterations
            # if self._camera_loop_iterations[cam_idx] % 30 == 0:
            #     logger.info(f"[CAMERA LOOP] Cam {cam_idx}: iteration #{self._camera_loop_iterations[cam_idx]}")

            # Keep old diagnostic flag for compatibility with other diagnostics
            if not hasattr(self, '_display_cam_diagnostic_logged'):
                self._display_cam_diagnostic_logged = {cam_idx: True}
            elif cam_idx not in self._display_cam_diagnostic_logged:
                self._display_cam_diagnostic_logged[cam_idx] = True

            # Find the corresponding frame for this active camera
            frame_info = None
            frame_idx = None

            # Simple lookup using persistent camera-to-frame map (set when camera starts)
            if cam_idx in self.camera_to_frame_map:
                frame_idx = self.camera_to_frame_map[cam_idx]
                if 0 <= frame_idx < len(self.frames):
                    frame_info = self.frames[frame_idx]
                else:
                    logger.error(f"[DISPLAY] Frame index {frame_idx} out of bounds for camera {cam_idx}")
                    continue
            else:
                logger.warning(f"[DISPLAY] Camera {cam_idx} not in frame map (not started yet?)")
                continue

            if frame_info is None:
                logger.warning(f"[DISPLAY EXIT] Cam {cam_idx}: No frame_info found (iteration {self._camera_loop_iterations.get(cam_idx, 0)})")
                continue
                
            try:
                canvas = frame_info['canvas']
                cameras_processed += 1

                # CRITICAL FIX: Ensure canvas is ready before trying to render
                # This prevents rendering to uninitialized canvases
                if not canvas.winfo_exists():
                    logger.warning(f"[DISPLAY EXIT] Cam {cam_idx}: Canvas doesn't exist (iteration {self._camera_loop_iterations.get(cam_idx, 0)})")
                    continue

                # Force canvas to be visible if it's not
                if not canvas.winfo_viewable():
                    logger.debug(f"[GUI] Canvas for camera {cam_idx} not viewable, forcing update")
                    canvas.update_idletasks()
                    # Give it one more chance
                    if not canvas.winfo_viewable():
                        logger.warning(f"[GUI] Canvas for camera {cam_idx} still not viewable after update")

                # DIAGNOSTIC DISABLED: Uncomment to debug display data reading
                # if cam_idx in getattr(self, '_display_cam_diagnostic_logged', {}):
                #     logger.info(f"[DISPLAY DIAGNOSTIC] About to read display data for camera {cam_idx}")

                # HEARTBEAT: Track display loop iterations to confirm it's running
                if not hasattr(self, '_display_loop_heartbeat'):
                    self._display_loop_heartbeat = {}
                if cam_idx not in self._display_loop_heartbeat:
                    self._display_loop_heartbeat[cam_idx] = 0
                self._display_loop_heartbeat[cam_idx] += 1

                # DIAGNOSTIC DISABLED: Uncomment to debug display loop heartbeat
                # if self._display_loop_heartbeat[cam_idx] % 30 == 0:
                #     logger.info(f"[DISPLAY LOOP HEARTBEAT] Camera {cam_idx}: iteration #{self._display_loop_heartbeat[cam_idx]}")

                # FIX #2: Defensive check - ensure display reader is connected before attempting read
                if not hasattr(self, 'display_buffer_manager') or not self.display_buffer_manager:
                    logger.warning(f"[DISPLAY] Camera {cam_idx}: display_buffer_manager not initialized")
                    continue

                if cam_idx not in self.display_buffer_manager.display_readers:
                    # Track occurrences for rate-limited logging
                    if not hasattr(self, '_reader_missing_logged'):
                        self._reader_missing_logged = {}
                    if cam_idx not in self._reader_missing_logged:
                        self._reader_missing_logged[cam_idx] = 0
                    self._reader_missing_logged[cam_idx] += 1

                    # Log first 5 occurrences, then every 30 occurrences
                    if self._reader_missing_logged[cam_idx] <= 5 or self._reader_missing_logged[cam_idx] % 30 == 0:
                        connected_cameras = list(self.display_buffer_manager.display_readers.keys())
                        logger.warning(f"[DISPLAY DIAGNOSTIC] ❌ Camera {cam_idx} NOT in display_readers")
                        logger.warning(f"[DISPLAY DIAGNOSTIC]   Cameras that ARE connected: {connected_cameras}")
                        logger.warning(f"[DISPLAY DIAGNOSTIC]   Miss count: {self._reader_missing_logged[cam_idx]}")
                        logger.warning(f"[DISPLAY] Camera {cam_idx}: display reader not connected yet "
                                      f"(count: {self._reader_missing_logged[cam_idx]}) - waiting for worker")
                    continue  # Skip this camera until reader connected

                # CRITICAL FIX: Read display-ready data from GUI buffer
                # This replaces the broken dual-source approach
                display_data = self._read_display_ready_data(cam_idx)

                # DIAGNOSTIC DISABLED: Uncomment to debug display data read results
                # if cam_idx in getattr(self, '_display_cam_diagnostic_logged', {}):
                #     logger.info(f"[DISPLAY DIAGNOSTIC] Read display data result: {display_data is not None}")
                #     if display_data:
                #         logger.info(f"[DISPLAY DIAGNOSTIC] Display data keys: {list(display_data.keys())}")

                if not display_data:
                    # CRITICAL DIAGNOSTIC: Track how often we hit this exit point
                    if not hasattr(self, '_no_display_data_count'):
                        self._no_display_data_count = {}
                    if cam_idx not in self._no_display_data_count:
                        self._no_display_data_count[cam_idx] = 0
                    self._no_display_data_count[cam_idx] += 1

                    # Log every 30 times to monitor this exit point
                    if self._no_display_data_count[cam_idx] % 30 == 0:
                        logger.info(f"[DISPLAY EXIT] Cam {cam_idx}: No display_data (count: {self._no_display_data_count[cam_idx]}, iteration: {self._camera_loop_iterations.get(cam_idx, 0)})")
                    continue  # No new data
                
                # Extract frame and faces from display-ready data
                frame_bgr = display_data.get('frame')
                display_faces = display_data.get('faces', [])

                # DIAGNOSTIC: Log frame fetch status to debug frozen display
                if not hasattr(self, '_frame_fetch_logged'):
                    self._frame_fetch_logged = {}
                if cam_idx not in self._frame_fetch_logged or self._frame_fetch_logged[cam_idx] < 5:
                    if cam_idx not in self._frame_fetch_logged:
                        self._frame_fetch_logged[cam_idx] = 0
                    self._frame_fetch_logged[cam_idx] += 1
                    logger.info(f"[GUI DEBUG] Camera {cam_idx}: display_data exists={display_data is not None}, "
                              f"frame_bgr is {'NOT None' if frame_bgr is not None else 'None'}, "
                              f"frame_id={display_data.get('frame_id') if display_data else 'N/A'}")
                    if frame_bgr is None and display_data:
                        logger.warning(f"[GUI DEBUG] Camera {cam_idx}: Frame is None despite display_data existing! "
                                     f"Keys in display_data: {list(display_data.keys()) if display_data else 'N/A'}")

                # FALLBACK: Create black frame if frame fetch failed but we have display data
                # This ensures canvas updates even when frame buffer read fails
                if frame_bgr is None and display_data:
                    logger.info(f"[GUI FALLBACK] Creating black frame for camera {cam_idx} (frame fetch failed)")
                    resolution = self.camera_resolutions.get(cam_idx, (1280, 720))
                    frame_bgr = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                    # Add text to indicate waiting state
                    cv2.putText(frame_bgr, "Waiting for camera data...", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame_bgr, f"Frame ID: {display_data.get('frame_id', 'N/A')}", (50, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

                # FIX #2: Validate frame exists AND has valid dimensions
                if frame_bgr is not None and frame_bgr.shape[0] > 0 and frame_bgr.shape[1] > 0:
                    # Auto-detect camera resolution from frame
                    frame_res = (frame_bgr.shape[1], frame_bgr.shape[0])  # (width, height)
                    if cam_idx not in self.camera_resolutions:
                        self.camera_resolutions[cam_idx] = frame_res
                        # SYNC FIX: Also update BufferCoordinator's resolution tracking
                        self.buffer_coordinator.camera_resolutions[cam_idx] = frame_res
                        logger.info(f"[GUI] Detected camera {cam_idx} resolution: {frame_res[0]}x{frame_res[1]} (synced to BufferCoordinator)")

                    # Track if this camera has ever shown face detection results
                    if not hasattr(self, '_camera_faces_detected'):
                        self._camera_faces_detected = {}

                    # Show initialization status overlay if no faces have been detected yet
                    if not display_faces and cam_idx not in self._camera_faces_detected:
                        # Add "Face detection initializing..." overlay
                        h, w = frame_bgr.shape[:2]
                        text = "Face detection initializing..."
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        thickness = 2
                        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                        text_x = (w - text_size[0]) // 2
                        text_y = h // 2

                        # Draw semi-transparent background
                        padding = 10
                        overlay = frame_bgr.copy()
                        cv2.rectangle(overlay,
                                    (text_x - padding, text_y - text_size[1] - padding),
                                    (text_x + text_size[0] + padding, text_y + padding),
                                    (50, 50, 50), -1)
                        cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)

                        # Draw text
                        cv2.putText(frame_bgr, text, (text_x, text_y),
                                  font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    elif display_faces:
                        # Mark that this camera has detected faces
                        self._camera_faces_detected[cam_idx] = True

                    # Draw overlays on frame BEFORE rendering to canvas
                    if display_faces:
                        # Convert display_faces to the format expected by draw_overlays_combined
                        faces_for_overlay = []
                        for face_idx, face in enumerate(display_faces):
                            bbox_value = face.get('bbox', [0, 0, 0, 0])
                            face_data = {
                                'id': face.get('track_id', 0),
                                'participant_id': face.get('participant_id', -1),  # FIX: Add participant_id for enrollment state lookup
                                'bbox': bbox_value,
                                'landmarks': face.get('landmarks'),
                                'display_flags': face.get('display_flags', 0),  # CRITICAL FIX: Include display_flags for enrollment overlay colors
                            }
                            faces_for_overlay.append(face_data)

                            # DEBUG: Log bbox for first face every 30 frames
                            if face_idx == 0 and not hasattr(self, '_bbox_debug_count'):
                                self._bbox_debug_count = 0
                            if face_idx == 0:
                                self._bbox_debug_count += 1
                                if self._bbox_debug_count % 30 == 0:
                                    logger.debug(f"[BBOX DEBUG] Camera {cam_idx} Face {face_idx}: bbox={bbox_value}, "
                                               f"has_bbox={'bbox' in face}, track_id={face.get('track_id', 'N/A')}")

                        # Create labels dict for participant names
                        labels = {}
                        for face in display_faces:
                            participant_id = face.get('participant_id')
                            if participant_id and participant_id > 0:
                                labels[participant_id] = face.get('label', f'P{participant_id}')

                        # Event-driven enrollment states (updated by queue consumer)
                        # No lock needed - O(1) dict read from cache populated by 'enrollment_state_update' events
                        enrollment_states = self.enrollment_state_cache

                        # Get pose data before drawing overlays
                        all_poses = display_data.get('all_poses', [])

                        # Draw overlays directly on frame using MediaPipe drawing utilities
                        frame_bgr = draw_overlays_combined(
                            frame_bgr,
                            faces=faces_for_overlay,
                            labels=labels,
                            face_mesh=False,
                            face_contours=True,
                            face_points=False,
                            pose_lines=True,  # Enable centralized pose drawing with transparency
                            enrollment_states=enrollment_states,
                            all_poses=all_poses,
                            config=self.config
                        )

                    # Use detected resolution or fallback
                    original_res = self.camera_resolutions.get(cam_idx, (1920, 1080))

                    # DIAGNOSTIC: Log before render attempt
                    if not hasattr(self, '_render_attempt_logged'):
                        self._render_attempt_logged = {}
                    if cam_idx not in self._render_attempt_logged:
                        logger.info(f"[GUI] About to call render_frame_to_canvas for camera {cam_idx}")
                        logger.info(f"[GUI] Canvas size: {canvas.winfo_width()}x{canvas.winfo_height()}")
                        logger.info(f"[GUI] Frame shape: {frame_bgr.shape}")
                        logger.info(f"[GUI] Original resolution: {original_res}")
                        self._render_attempt_logged[cam_idx] = True

                    # ALWAYS render frames if available (even during initialization)
                    # Transparent overlay will sit on top and show status messages
                    # This allows users to see live video throughout warmup process
                    photo = self.drawingmanager.render_frame_to_canvas(
                        frame_bgr, canvas, frame_idx,
                        original_resolution=original_res
                    )

                    # Keep overlay on top if it exists
                    if canvas.find_withtag('loading_overlay'):
                        canvas.tag_raise('loading_overlay')

                    # DIAGNOSTIC: Track successful renders
                    if not hasattr(self, '_successful_renders'):
                        self._successful_renders = {}
                    if cam_idx not in self._successful_renders:
                        self._successful_renders[cam_idx] = 0
                    self._successful_renders[cam_idx] += 1

                    # Log every 30 successful renders
                    if self._successful_renders[cam_idx] % 30 == 0:
                        logger.info(f"[RENDER SUCCESS] Cam {cam_idx}: Rendered frame to canvas (render #{self._successful_renders[cam_idx]})")

                    # DIAGNOSTIC: Log render result
                    if cam_idx in self._render_attempt_logged and photo is None:
                        logger.warning(f"[GUI] render_frame_to_canvas returned None for camera {cam_idx}")
                else:
                    # FIX #2: Log when frame is skipped due to invalid dimensions
                    if frame_bgr is not None:
                        logger.warning(f"[GUI] Skipping frame for camera {cam_idx}: invalid dimensions {frame_bgr.shape}")
                    # Continue to next camera without processing this invalid frame
                    continue

                # Track successful preview update
                if hasattr(self, 'reliability_monitor'):
                    self.reliability_monitor.track_preview_update()

            except Exception as e:
                logger.error(f"[FATAL] Error in optimized preview for camera {cam_idx} (iteration {self._camera_loop_iterations.get(cam_idx, 0)}): {e}")
                import traceback
                traceback.print_exc()
                # Don't break the loop on exception - continue to next camera
                continue

            # Log successful completion of camera loop iteration (runs after try/except completes successfully)
            if not hasattr(self, '_loop_completion_count'):
                self._loop_completion_count = {}
            if cam_idx not in self._loop_completion_count:
                self._loop_completion_count[cam_idx] = 0
            self._loop_completion_count[cam_idx] += 1

            if self._loop_completion_count[cam_idx] % 30 == 0:
                logger.info(f"[LOOP COMPLETE] Cam {cam_idx}: Completed iteration #{self._camera_loop_iterations.get(cam_idx, 0)} successfully")

        # DIAGNOSTIC DISABLED: Uncomment to debug display loop completion
        # if self._display_method_calls % 30 == 0:
        #     logger.info(f"[DISPLAY METHOD] Completed loop - processed {cameras_processed} cameras")

        # Calculate and report performance
        preview_end_time = time.time()
        preview_duration_ms = (preview_end_time - preview_start_time) * 1000
        
        # Update FPS tracking
        if not hasattr(self, '_fps_frame_times'):
            self._fps_frame_times = []
            
        self._fps_frame_times.append(preview_end_time)
        if len(self._fps_frame_times) > 30:
            self._fps_frame_times.pop(0)
            
        # Report performance every second
        if not hasattr(self, '_last_display_report'):
            self._last_display_report = time.time()
            
        if preview_end_time - self._last_display_report >= 1.0:
            if len(self._fps_frame_times) > 1:
                time_span = self._fps_frame_times[-1] - self._fps_frame_times[0]
                if time_span > 0:
                    actual_fps = (len(self._fps_frame_times) - 1) / time_span
                else:
                    actual_fps = 0
            else:
                actual_fps = 0
                
            # GUI display performance logging removed per user request
            # Focus on face recognition and correlator functions only

            self._last_display_report = preview_end_time

    def _check_processing_worker_status(self):
        """Check and report processing worker status."""
        try:
            while not self.processing_status_queue.empty():
                status = self.processing_status_queue.get_nowait()
                
                if status['type'] == 'performance':
                    pass  # Performance metrics available for monitoring, not logged by default
                elif status['type'] == 'error':
                    logger.error(f"[PROCESSING WORKER] Error: {status['error']}")
                elif status['type'] == 'display_buffer_added':
                    # Handle dynamically added camera display buffer
                    cam_idx = status.get('camera_index')
                    buffer_name = status.get('display_buffer_name')
                    if cam_idx is not None and buffer_name:
                        try:
                            if hasattr(self, 'display_buffer_manager') and self.display_buffer_manager:
                                self.display_buffer_manager.connect_display_buffer(cam_idx, buffer_name)
                                logger.info(f"[GUI DYNAMIC] Connected to display buffer for camera {cam_idx}: {buffer_name}")

                                # Also connect to frame buffer if available
                                if hasattr(self, 'actual_buffer_names') and cam_idx in self.actual_buffer_names:
                                    frame_buffer_name = self.actual_buffer_names[cam_idx].get('frame')
                                    if frame_buffer_name:
                                        self.display_buffer_manager.connect_frame_buffer(cam_idx, frame_buffer_name)
                                        logger.info(f"[GUI DYNAMIC] Connected to frame buffer for camera {cam_idx}: {frame_buffer_name}")
                            else:
                                logger.error(f"[GUI DYNAMIC] display_buffer_manager not available for camera {cam_idx}")
                        except Exception as e:
                            logger.error(f"[GUI DYNAMIC] Failed to connect display buffer for camera {cam_idx}: {e}")
                elif status['type'] == 'camera_display_connected':
                    # DYNAMIC DISPLAY CONNECTION: Worker has connected to camera display buffer
                    # This is the response to our 'connect_camera' control message
                    cam_idx = status.get('camera_index')
                    buffer_name = status.get('display_buffer_name')

                    logger.info(f"[GUI DIAGNOSTIC] 🔔 RECEIVED camera_display_connected for camera {cam_idx}")
                    logger.info(f"[GUI DIAGNOSTIC]   Display buffer: {buffer_name}")
                    logger.info(f"[GUI DIAGNOSTIC]   display_buffer_manager exists: {hasattr(self, 'display_buffer_manager')}")
                    if hasattr(self, 'display_buffer_manager'):
                        logger.info(f"[GUI DIAGNOSTIC]   display_buffer_manager is not None: {self.display_buffer_manager is not None}")
                        if self.display_buffer_manager:
                            logger.info(f"[GUI DIAGNOSTIC]   Current display_readers: {list(self.display_buffer_manager.display_readers.keys())}")

                    logger.info(f"[GUI] 📥 Received camera_display_connected for camera {cam_idx}")
                    logger.info(f"[GUI]   Display buffer: {buffer_name}")

                    if cam_idx is not None and buffer_name:
                        try:
                            if hasattr(self, 'display_buffer_manager') and self.display_buffer_manager:
                                # Connect to the worker's display buffer
                                logger.info(f"[GUI DIAGNOSTIC] 🔗 CALLING connect_display_buffer({cam_idx}, {buffer_name})")
                                self.display_buffer_manager.connect_display_buffer(cam_idx, buffer_name)
                                logger.info(f"[GUI DIAGNOSTIC] ✅ connect_display_buffer() returned successfully")
                                logger.info(f"[GUI DIAGNOSTIC]   Updated display_readers: {list(self.display_buffer_manager.display_readers.keys())}")
                                logger.info(f"[GUI] ✅ Connected to display buffer for camera {cam_idx}: {buffer_name}")

                                # Also connect to frame buffer if available (for direct frame access)
                                if hasattr(self, 'actual_buffer_names') and cam_idx in self.actual_buffer_names:
                                    frame_buffer_name = self.actual_buffer_names[cam_idx].get('frame')
                                    if frame_buffer_name:
                                        self.display_buffer_manager.connect_frame_buffer(cam_idx, frame_buffer_name)
                                        logger.info(f"[GUI] ✅ Connected to frame buffer for camera {cam_idx}: {frame_buffer_name}")

                                logger.info(f"[GUI DIAGNOSTIC] ✅✅✅ Camera {cam_idx} display fully connected - should render now!")
                                logger.info(f"[GUI] ✅✅✅ Camera {cam_idx} display fully connected - should render now!")
                            else:
                                logger.error(f"[GUI DIAGNOSTIC] ❌ display_buffer_manager not available for camera {cam_idx}")
                                logger.error(f"[GUI] display_buffer_manager not available for camera {cam_idx}")
                        except Exception as e:
                            logger.error(f"[GUI DIAGNOSTIC] ❌ EXCEPTION connecting display buffer: {e}")
                            logger.error(f"[GUI] Failed to connect display buffer for camera {cam_idx}: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
        except:
            pass


    def update_participant_names(self):
        """Update participant names in processing worker."""
        if hasattr(self, 'processing_control_queue'):
            self.processing_control_queue.put({
                'type': 'update_names',
                'names': self.participant_names.copy()
            })

    def _monitor_worker_health(self):
        """
        Periodically check if GUI processing worker is alive and restart if needed.

        CRITICAL FIX: Monitors worker health every 5 seconds and attempts automatic restart
        if the worker process has died. This prevents the GUI from hanging indefinitely
        when the worker crashes silently.
        """
        if not hasattr(self, 'gui_processing_worker'):
            # Worker not initialized yet, skip monitoring
            self.after(5000, self._monitor_worker_health)
            return

        if self.gui_processing_worker and not self.gui_processing_worker.is_alive():
            # Worker crashed - restart it
            exitcode = self.gui_processing_worker.exitcode
            logger.error(f"[WORKER MONITOR] GUI processing worker died with exit code: {exitcode}")
            logger.error(f"[WORKER MONITOR] Dead worker PID: {self.gui_processing_worker.pid}")

            logger.info("[WORKER MONITOR] Attempting automatic worker restart...")
            try:
                self.shutdown_display_processing()
                time.sleep(1.0)  # Give time for cleanup

                success = self.initialize_display_processing()
                if success:
                    logger.info("[WORKER MONITOR] Worker restarted successfully")
                else:
                    logger.error("[WORKER MONITOR] Worker restart failed")
            except Exception as restart_error:
                logger.error(f"[WORKER MONITOR] Error during worker restart: {restart_error}")
                import traceback
                logger.error(traceback.format_exc())

        elif self.gui_processing_worker and hasattr(self, 'display_buffer_manager'):
            # Worker is alive - check if it's frozen (not processing frames)
            try:
                # Get latest frame_id from display buffer
                display_data = self.display_buffer_manager.get_display_data(0)  # Camera 0 as reference
                current_frame_id = display_data.frame_id if display_data else None

                if current_frame_id is not None:
                    last_frame_id = getattr(self, '_monitor_last_frame_id', None)

                    if last_frame_id is not None and current_frame_id == last_frame_id:
                        # Frame ID hasn't changed - increment freeze counter
                        self._monitor_freeze_count = getattr(self, '_monitor_freeze_count', 0) + 1

                        # If frozen for 10 checks (50 seconds), restart worker
                        if self._monitor_freeze_count >= 10:
                            logger.error(f"[WORKER MONITOR] Worker frozen at frame_id={current_frame_id} for {self._monitor_freeze_count * 5}s")
                            logger.info("[WORKER MONITOR] Attempting worker restart due to freeze...")
                            try:
                                self.shutdown_display_processing()
                                time.sleep(1.0)
                                success = self.initialize_display_processing()
                                if success:
                                    logger.info("[WORKER MONITOR] Worker restarted successfully")
                                    self._monitor_freeze_count = 0
                                else:
                                    logger.error("[WORKER MONITOR] Worker restart failed")
                            except Exception as restart_error:
                                logger.error(f"[WORKER MONITOR] Error during restart: {restart_error}")
                    else:
                        # Frame ID changed - reset freeze counter
                        self._monitor_freeze_count = 0
                        self._monitor_last_frame_id = current_frame_id
            except Exception as e:
                logger.debug(f"[WORKER MONITOR] Error checking frame_id: {e}")

        # Schedule next health check (every 5 seconds)
        self.after(5000, self._monitor_worker_health)

    def shutdown_display_processing(self):
        """Shutdown display processing and cleanup."""
        logger.info("Shutting down display processing...")

        # Stop LSL streaming in worker first
        if hasattr(self, 'processing_control_queue'):
            try:
                self.processing_control_queue.put({'type': 'stop_lsl_streaming'})
                logger.info("[GUI] Disabled LSL streaming in GUI processing worker")
            except Exception as e:
                logger.warning(f"Failed to disable LSL streaming in worker: {e}")

        # Send shutdown command to worker
        if hasattr(self, 'processing_control_queue'):
            self.processing_control_queue.put({'type': 'shutdown'})
            
        # Wait for worker to stop
        if hasattr(self, 'gui_processing_worker') and self.gui_processing_worker.is_alive():
            self.gui_processing_worker.join(timeout=2.0)
            if self.gui_processing_worker.is_alive():
                logger.warning("Processing worker did not stop gracefully")
                self.gui_processing_worker.terminate()
            # Clear worker reference
            self.gui_processing_worker = None

        # Cleanup display buffer manager connections (preserve instance for reuse)
        if hasattr(self, 'display_buffer_manager'):
            self.display_buffer_manager.cleanup()
            # Keep instance alive - will reconnect to new buffers on worker restart
            logger.info("DisplayBufferManager connections cleaned up (instance preserved for reuse)")

        # Reset display loop flag
        self._optimized_display_active = False

        logger.info("Display processing shutdown complete")
    
    
    def _process_camera_status_updates(self):
        """Process status and metadata updates from camera workers."""
        # DEBUG: Always log that this method is being called
        logger.info(f"[GUI] _process_camera_status_updates() called")

        # Process status updates from camera workers
        status_updates = self.camera_worker_manager.process_status_updates()

        # CRITICAL FIX: Retrieve cached status updates from handshake phase
        # During camera startup, model_loading and model_ready messages are cached
        # in pending_status_updates but never retrieved. This causes overlay to show
        # generic "Initializing Camera..." instead of detailed progress messages.
        if hasattr(self, 'active_camera_procs'):
            for cam_idx in self.active_camera_procs.keys():
                pending = self.camera_worker_manager.get_pending_status_updates(cam_idx)
                if pending:
                    logger.info(f"[GUI] Retrieved {len(pending)} cached status updates for camera {cam_idx}")
                    status_updates.extend(pending)

        logger.info(f"[GUI] Got {len(status_updates)} total status updates (real-time + cached)")

        if status_updates:
            logger.info(f"[GUI] Processing {len(status_updates)} status updates")

        for status in status_updates:
            cam_idx = status.get('camera_index')
            status_type = status.get('type', 'unknown')
            logger.info(f"[GUI] Status update from camera {cam_idx}: type={status_type}")

            # CRITICAL FIX: Ensure camera_health_status entry exists before processing
            if cam_idx is not None:
                # Create entry if it doesn't exist
                if cam_idx not in self.camera_health_status:
                    logger.info(f"[GUI] Creating new camera_health_status entry for camera {cam_idx}")
                    self.camera_health_status[cam_idx] = {}

                # NEW: Handle MediaPipe model initialization statuses
                if status_type == 'model_loading':
                    message = status.get('data', {}).get('message', 'Loading face detection model...')
                    logger.info(f"[GUI] Camera {cam_idx}: {message}")
                    self.camera_health_status[cam_idx]['initialization_stage'] = message
                    self.camera_health_status[cam_idx]['state'] = 'initializing'
                    self.camera_health_status[cam_idx]['state_color'] = '#2196F3'  # Blue
                    # Update overlay immediately with status message
                    self._update_initialization_status(cam_idx, message)
                    # Update UI to show status
                    self._update_camera_health_ui()
                    continue
                elif status_type == 'model_ready':
                    message = status.get('data', {}).get('message', 'Face detection ready')
                    logger.info(f"[GUI] Camera {cam_idx}: {message}")
                    # Update status but keep spinner - waiting for real-time sync
                    self.camera_health_status[cam_idx]['initialization_stage'] = 'Synchronizing real-time video...'
                    self.camera_health_status[cam_idx]['state'] = 'model_ready'
                    self.camera_health_status[cam_idx]['state_color'] = '#FFA500'  # Orange - still syncing
                    # Update overlay to show sync status
                    self._update_initialization_status(cam_idx, 'Synchronizing real-time video...')
                    # Update UI to show status
                    self._update_camera_health_ui()
                    # DON'T clear overlay - wait for realtime_ready
                    continue

                elif status_type == 'realtime_ready':
                    # Real-time sync achieved - show transparent overlay, flush buffers, then dismiss overlay
                    message = status.get('data', {}).get('message', 'Real-time video sync achieved')
                    logger.warning(f"[REALTIME READY] ⚠️⚠️⚠️ Camera {cam_idx}: {message}")
                    self.camera_health_status[cam_idx]['initialization_stage'] = None  # Clear to allow video rendering!
                    self.camera_health_status[cam_idx]['state'] = 'running'
                    self.camera_health_status[cam_idx]['state_color'] = '#4CAF50'  # Green - ready

                    # Track that this camera has achieved realtime sync
                    if not hasattr(self, '_realtime_ready_cameras'):
                        self._realtime_ready_cameras = set()
                    self._realtime_ready_cameras.add(cam_idx)
                    logger.warning(f"[REALTIME READY] ✅ Camera {cam_idx} marked as realtime ready")

                    # IMMEDIATE OVERLAY REMOVAL: Detection is working, show live video
                    canvas_idx = self.camera_to_frame_map.get(cam_idx)
                    if canvas_idx is not None and canvas_idx < len(self.frames):
                        canvas = self.frames[canvas_idx].get('canvas')
                        if canvas:
                            # Delete any existing loading overlay immediately
                            canvas.delete('loading_overlay')
                            # Stop spinner animation
                            if canvas_idx in self._canvas_spinners:
                                after_id = self._canvas_spinners[canvas_idx].get('after_id')
                                if after_id:
                                    self.after_cancel(after_id)
                                del self._canvas_spinners[canvas_idx]
                            # Clean up overlay PhotoImage reference
                            if hasattr(self, '_overlay_images') and canvas_idx in self._overlay_images:
                                del self._overlay_images[canvas_idx]
                            logger.warning(f"[REALTIME READY] ✅ Overlay removed for camera {cam_idx} - showing realtime video")

                        # Update status label to show active state
                        meta_label = self.frames[canvas_idx].get('meta_label')
                        if meta_label:
                            meta_label.config(text=f"Camera {cam_idx} - Active", foreground='green')

                    logger.info(f"[REALTIME READY] Camera {cam_idx} achieved real-time sync")

                    self._update_camera_health_ui()
                    continue

                elif status_type == 'first_face_detected':
                    # First face detected - NOW flush buffers and dismiss overlay
                    message = status.get('data', {}).get('message', 'First face detected')
                    logger.warning(f"[FIRST FACE] 🎉 Camera {cam_idx}: {message}")
                    self.camera_health_status[cam_idx]['initialization_stage'] = None
                    self.camera_health_status[cam_idx]['state'] = 'running'
                    self.camera_health_status[cam_idx]['state_color'] = '#4CAF50'  # Green - fully ready

                    # Track first face detection
                    if not hasattr(self, '_first_face_cameras'):
                        self._first_face_cameras = set()
                    self._first_face_cameras.add(cam_idx)
                    logger.warning(f"[FIRST FACE] ✅ Camera {cam_idx} first face detected, clearing overlay")

                    # Notify GUI processing worker to flush its internal state
                    if hasattr(self, 'gui_processing_worker') and self.gui_processing_worker and self.gui_processing_worker.is_alive():
                        try:
                            self.processing_control_queue.put({
                                'type': 'flush_camera',
                                'camera_index': cam_idx,
                                'reason': 'first_face_detected'
                            })
                            logger.warning(f"[FLUSH WORKER] ⚠️ Sent flush_camera command to worker for camera {cam_idx} (first face)")
                        except Exception as e:
                            logger.error(f"[FLUSH WORKER] Failed to send flush command to worker: {e}")
                    else:
                        logger.warning(f"[FLUSH WORKER] ⚠️ Worker not available to flush camera {cam_idx}")

                    # NOW clear loading overlay (after flush is queued)
                    canvas_idx = self.camera_to_frame_map.get(cam_idx)
                    if canvas_idx is not None and canvas_idx < len(self.frames):
                        canvas = self.frames[canvas_idx].get('canvas')
                        if canvas:
                            canvas.delete('loading_overlay')
                            # Stop spinner animation for this canvas
                            if canvas_idx in self._canvas_spinners:
                                after_id = self._canvas_spinners[canvas_idx].get('after_id')
                                if after_id:
                                    self.after_cancel(after_id)
                                del self._canvas_spinners[canvas_idx]
                            # Clean up transparent overlay PhotoImage reference
                            if canvas_idx in self._overlay_images:
                                del self._overlay_images[canvas_idx]
                            logger.warning(f"[FIRST FACE] ✅ Spinner overlay dismissed for camera {cam_idx}")

                    self._update_camera_health_ui()
                    continue

                # Update status information
                self.camera_health_status[cam_idx].update({
                    'last_update': time.time(),
                    'status': status_type,
                    'data': status.get('data', {})
                })

                # Sync status to frames dict for backward compatibility with legacy checks
                # This fixes mesh toggle and other code that checks self.frames[idx]['status']
                if hasattr(self, 'camera_to_frame_map') and cam_idx in self.camera_to_frame_map:
                    frame_idx = self.camera_to_frame_map[cam_idx]
                    if 0 <= frame_idx < len(self.frames):
                        self.frames[frame_idx]['status'] = status_type
                        logger.debug(f"[GUI] Synced status '{status_type}' to frames[{frame_idx}] for camera {cam_idx}")

                # Capture actual buffer names from 'ready' status - CRITICAL FOR FAST DISPLAY
                if status_type == 'ready' and 'shared_memory' in status.get('data', {}):
                    if not hasattr(self, 'actual_buffer_names'):
                        self.actual_buffer_names = {}
                        logger.info(f"[GUI] Initializing actual_buffer_names dictionary")
                    
                    self.actual_buffer_names[cam_idx] = status['data']['shared_memory']
                    logger.info(f"[GUI] Captured actual buffer names for camera {cam_idx}: {list(status['data']['shared_memory'].keys())}")
                    logger.info(f"[GUI] Total cameras with buffer names: {len(self.actual_buffer_names)}")

                    # Also store the actual camera resolution for buffer calculations
                    if 'resolution' in status['data']:
                        resolution = status['data']['resolution']
                        self.camera_resolutions[cam_idx] = resolution
                        # SYNC FIX: Also update BufferCoordinator's resolution tracking
                        self.buffer_coordinator.camera_resolutions[cam_idx] = resolution
                        logger.info(f"[GUI] Captured camera {cam_idx} resolution: {resolution} (synced to BufferCoordinator)")

                        # CONFIG PROPAGATION FIX: Update per-camera config with detected resolution
                        cam_key = f'camera_{cam_idx}'
                        self.config.set(f'camera_settings.{cam_key}.width', resolution[0])
                        self.config.set(f'camera_settings.{cam_key}.height', resolution[1])
                        logger.info(f"[GUI] Updated config: camera_{cam_idx} width={resolution[0]}, height={resolution[1]}")

                        # Update global ZMQ detected_resolution (for future camera discovery)
                        zmq_enabled = self.config.get('zmq_camera_bridge', {}).get('enabled', False)
                        if zmq_enabled:
                            current_detected = self.config.get('zmq_camera_bridge.detected_resolution')
                            if not current_detected:
                                self.config.set('zmq_camera_bridge.detected_resolution', list(resolution))
                                logger.info(f"[GUI] Set global ZMQ detected_resolution: {resolution[0]}x{resolution[1]}")

                        # Persist config to disk (ensures future runs use correct resolution)
                        try:
                            self.config.save()
                            logger.info(f"[GUI] Saved detected resolution to config file")
                        except Exception as e:
                            logger.warning(f"[GUI] Failed to save config: {e}")

                        # Update GUI resolution display label (if helper method exists)
                        if hasattr(self, '_update_camera_resolution_display'):
                            self._update_camera_resolution_display(cam_idx, resolution)

                        # ZMQ AUTO-DETECTION: If this is camera 0 in ZMQ mode, store as detected resolution
                        zmq_enabled = self.config.get('zmq_camera_bridge', {}).get('enabled', False)
                        if zmq_enabled and cam_idx == 0:
                            detected_res = list(status['data']['resolution'])  # Convert tuple to list for JSON

                            # Check if resolution changed
                            current_detected = self.config.get('zmq_camera_bridge', {}).get('detected_resolution')
                            if current_detected != detected_res:
                                # Store as global ZMQ resolution
                                self.config.set('zmq_camera_bridge.detected_resolution', detected_res)
                                logger.info(f"[GUI] ZMQ resolution detected from camera 0: {detected_res[0]}x{detected_res[1]}")

                                # Update UI to reflect detected resolution
                                self._update_resolution_selector_state()

                                # Apply to all cameras
                                self._apply_detected_resolution_to_all_cameras(detected_res)

                        # PROPAGATION FIX: Send updated resolutions to GUI worker if it's running
                        if hasattr(self, 'gui_processing_worker') and self.gui_processing_worker and self.gui_processing_worker.is_alive():
                            try:
                                self.processing_control_queue.put({
                                    'type': 'buffer_names',
                                    'actual_buffer_names': self.actual_buffer_names,
                                    'camera_resolutions': self.buffer_coordinator.camera_resolutions.copy()
                                })
                                logger.info(f"[GUI] Sent updated camera resolutions to worker after detecting camera {cam_idx}")
                            except Exception as e:
                                logger.error(f"[GUI] Failed to send updated resolutions to worker: {e}")
                    
                    # Track ready cameras (but DON'T back off yet - still need aggressive polling during warmup)
                    self._ready_cameras.add(cam_idx)

                    # FIX: Only back off when ALL cameras achieve realtime_ready (not just 'ready')
                    # This ensures we capture model_loading and model_ready status updates during warmup
                    if not hasattr(self, '_realtime_ready_cameras'):
                        self._realtime_ready_cameras = set()

                    if hasattr(self, 'active_camera_procs') and self.active_camera_procs:
                        # Check if ALL cameras have achieved realtime sync
                        if self._realtime_ready_cameras.issuperset(self.active_camera_procs.keys()):
                            # All active cameras achieved realtime_ready -> back off to 500 ms (2 Hz)
                            self._status_poll_interval_ms = 500
                            logger.warning(f"[STATUS POLL] ⚠️ All cameras realtime ready - backing off to 500ms status polling")
                        else:
                            # Still waiting for realtime_ready - keep aggressive polling
                            pending_cameras = set(self.active_camera_procs.keys()) - self._realtime_ready_cameras
                            logger.info(f"[STATUS POLL] Keeping 100ms polling - waiting for realtime_ready from cameras: {pending_cameras}")
        
        # Process metadata updates
        metadata_updates = self.camera_worker_manager.process_metadata()
        for metadata in metadata_updates:
            cam_idx = metadata.get('camera_index')
            if cam_idx is not None:
                # Update health status with processing info
                if cam_idx in self.camera_health_status:
                    self.camera_health_status[cam_idx].update({
                        'processing_time_ms': metadata.get('processing_time_ms', 0),
                        'frame_id': metadata.get('frame_id', 0),
                        'n_faces': metadata.get('n_faces', 0)
                    })
                    
        # Update health status UI every few frames
        if not hasattr(self, '_health_update_counter'):
            self._health_update_counter = 0
        self._health_update_counter += 1
        
        if self._health_update_counter % 30 == 0:  # Update every second at 30fps
            self._update_camera_health_ui()

    def _get_frame_index_for_camera(self, camera_idx: int):
        """
        Get the frame index (GUI slot) for a given camera index.

        Args:
            camera_idx: Camera index (e.g., 0, 1, 2)

        Returns:
            int or None: Frame index if found, None otherwise
        """
        for frame_idx, info in enumerate(self.frames):
            # Check if this frame is assigned to the camera
            # The camera assignment can be in 'camera_index' or we need to check combo value
            if info.get('camera_index') == camera_idx:
                return frame_idx

            # Also check if combo box selection matches
            combo = info.get('combo')
            if combo:
                try:
                    selected = combo.get()
                    # Parse camera index from combo text (format: "Cam 0: Name")
                    if selected and selected.startswith('Cam '):
                        cam_idx_from_combo = int(selected.split(':')[0].split()[1])
                        if cam_idx_from_combo == camera_idx:
                            return frame_idx
                except:
                    pass

        return None

    def _update_camera_resolution_display(self, camera_idx: int, resolution: tuple):
        """
        Update per-camera resolution display label.

        Shows actual detected resolution underneath each camera canvas.
        For ZMQ cameras, adds "(auto-detected)" indicator in green.

        Args:
            camera_idx: Camera index
            resolution: (width, height) tuple
        """
        try:
            # Find the frame info for this camera
            frame_idx = self._get_frame_index_for_camera(camera_idx)
            if frame_idx is None:
                logger.debug(f"[GUI] Cannot update resolution display - no frame for camera {camera_idx}")
                return

            info = self.frames[frame_idx]
            resolution_label = info.get('resolution_label')

            if resolution_label:
                width, height = resolution

                # Check if ZMQ mode to show "auto-detected" indicator
                zmq_enabled = self.config.get('zmq_camera_bridge', {}).get('enabled', False)

                if zmq_enabled:
                    # ZMQ camera - show green "auto-detected" text
                    text = f"Resolution: {width}x{height} (auto-detected)"
                    resolution_label.config(text=text, foreground='#4CAF50')  # Green
                else:
                    # V4L2 camera - show gray text (no auto-detected indicator)
                    text = f"Resolution: {width}x{height}"
                    resolution_label.config(text=text, foreground='#666666')  # Gray

                logger.info(f"[GUI] Updated resolution display for camera {camera_idx}: {width}x{height}")
            else:
                logger.debug(f"[GUI] No resolution_label found in frame {frame_idx} for camera {camera_idx}")

        except Exception as e:
            logger.error(f"[GUI] Error updating resolution display for camera {camera_idx}: {e}")

    def debug_canvas_state(self, canvas_idx=None):
        """Debug canvas state using drawing manager stats."""
        if canvas_idx is None:
            for idx in range(len(self.frames)):
                self.debug_canvas_state(idx)
            return
        
        if canvas_idx >= len(self.frames):
            return
        
        print(f"\n[Canvas Debug] Canvas {canvas_idx}:")
        stats = self.drawingmanager.get_stats(canvas_idx)
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    def on_participant_name_change(self, slot_idx, new_name):
        """Update participant_names and force overlays to update."""
        # CRITICAL FIX: Strip any locked status text before storing
        if new_name:
            clean_name = new_name.replace(" (Locked, Absent)", "").replace(" (Locked)", "").strip()
            self.participant_names[slot_idx] = clean_name
        else:
            self.participant_names[slot_idx] = f"P{slot_idx+1}"

        # CRITICAL FIX: Immediately propagate name changes to all systems (no 1-second delay)
        # Update the global participant manager with current names
        if hasattr(self, 'global_participant_manager') and self.global_participant_manager is not None:
            self.global_participant_manager.set_participant_names(self.participant_names)

        # If streaming, update all fusion processes immediately
        if self.streaming:
            for cam_idx, pipe in self.participant_mapping_pipes.items():
                try:
                    pipe.send({
                        'type': 'participant_names',
                        'names': self.global_participant_manager.participant_names.copy()
                    })
                except:
                    pass

        # Force overlays to update now
        self.schedule_preview()

    def toggle_participant_lock(self, slot_idx):
        """
        Toggle lock state for a participant.

        Args:
            slot_idx: 0-based participant slot index (converts to 1-based participant_id)
        """
        if not self.global_participant_manager:
            print("[LOCK] Cannot toggle lock: global_participant_manager not available")
            return

        # Convert 0-based slot to 1-based participant_id
        participant_id = slot_idx + 1

        # Check if currently locked
        is_locked = self.global_participant_manager.is_participant_locked(participant_id)

        if is_locked:
            # Unlock participant
            success = self.global_participant_manager.unlock_participant(participant_id)
            if success:
                print(f"[LOCK] Unlocked participant {participant_id}")
                # ARRAY-BASED IPC: Sync lock state to shared array for worker process slot reservation
                self.lock_state_array[participant_id] = 0  # 0 = unlocked
                self.update_lock_button_states()
            else:
                print(f"[LOCK] Failed to unlock participant {participant_id}")
        else:
            # Lock participant - requires ENROLLED state
            success = self.global_participant_manager.lock_participant(participant_id)
            if success:
                print(f"[LOCK] Locked participant {participant_id}")
                # ARRAY-BASED IPC: Sync lock state to shared array for worker process slot reservation
                self.lock_state_array[participant_id] = 1  # 1 = locked
                self.update_lock_button_states()
            else:
                # Lock failed - just log (button should be disabled when not enrolled, so this is unlikely)
                enrollment_state = self.enrollment_state_cache.get(participant_id, 'UNKNOWN')
                print(f"[LOCK] Cannot lock participant {participant_id}: enrollment state is {enrollment_state}")
                # No popup needed - button shows "Enrollment not complete" when disabled

    def update_lock_button_states(self):
        """Update lock button text and state based on enrollment and lock status."""
        if not self.global_participant_manager:
            return

        for slot_idx, lock_btn in enumerate(self.participant_lock_buttons):
            participant_id = slot_idx + 1

            # Check enrollment state
            enrollment_state = self.enrollment_state_cache.get(participant_id, 'UNKNOWN')
            is_enrolled = (enrollment_state == 'ENROLLED')

            # Check lock state
            is_locked = self.global_participant_manager.is_participant_locked(participant_id)

            # Check if participant is active (in frame) with debouncing
            # Cache presence state to avoid acting on temporary False values during worker updates
            if not hasattr(self, '_presence_cache'):
                self._presence_cache = {}

            # ARRAY-BASED IPC: Read presence from array instead of dict
            current_presence = bool(self.participant_presence_array[participant_id]) if participant_id < len(self.participant_presence_array) else False

            # Only update cached state if it's been stable for 2+ consecutive reads (debounce)
            cache_key = f"presence_{participant_id}"
            if cache_key not in self._presence_cache:
                self._presence_cache[cache_key] = {'state': current_presence, 'count': 0}
            else:
                cached = self._presence_cache[cache_key]
                if cached['state'] == current_presence:
                    cached['count'] += 1
                else:
                    # State changed - reset counter
                    cached['state'] = current_presence
                    cached['count'] = 0

            # Use cached state if confirmed (seen 2+ times), otherwise use current
            if self._presence_cache[cache_key]['count'] >= 2:
                is_active = self._presence_cache[cache_key]['state']
            else:
                is_active = current_presence

            # DIAGNOSTIC: Log state for locked participants (only on state changes)
            if is_locked:
                state_key = f"lock_state_{participant_id}"
                current_state = (is_locked, is_active, enrollment_state)
                if not hasattr(self, '_last_lock_states'):
                    self._last_lock_states = {}
                if self._last_lock_states.get(state_key) != current_state:
                    print(f"[LOCK BUTTON] P{participant_id}: locked={is_locked}, present={is_active}, enrollment={enrollment_state}", flush=True)
                    self._last_lock_states[state_key] = current_state

            # Update button appearance with visual styling
            if is_locked:
                # Apply 'locked' style - blue background, clickable to unlock
                style = self.lock_button_styles['locked']
                lock_btn.config(
                    text="Locked",
                    width=10,
                    bg=style['bg'],
                    fg=style['fg'],
                    activebackground=style['activebackground'],
                    state=style['state'],
                    relief=style['relief'],
                    cursor='hand2'  # Hand cursor indicates clickability
                )

                # LOCK SYSTEM: ALWAYS show locked status in entry field
                # Show "(Locked)" when present, "(Locked, Absent)" when absent
                if hasattr(self, 'participant_entries'):
                    entry = self.participant_entries[slot_idx]
                    current_text = entry.get()
                    participant_name = self.participant_names.get(slot_idx, f"P{participant_id}")
                    # CRITICAL FIX: Strip any existing locked status to prevent accumulation
                    participant_name = participant_name.replace(" (Locked, Absent)", "").replace(" (Locked)", "").strip()

                    # Determine desired text based on presence
                    if is_active:
                        desired_text = f"{participant_name} (Locked)"
                    else:
                        desired_text = f"{participant_name} (Locked, Absent)"

                    # DIAGNOSTIC: Show presence state for debugging absence updates
                    print(f"[GUI DEBUG] P{participant_id}: is_active={is_active}, current_text='{current_text}', desired_text='{desired_text}'", flush=True)

                    # Only update if text changed (avoid unnecessary redraws)
                    if current_text != desired_text:
                        print(f"[GUI DEBUG] P{participant_id}: UPDATING entry field from '{current_text}' to '{desired_text}'", flush=True)
                        entry.config(state='normal')  # Must set to normal before editing
                        entry.delete(0, 'end')
                        entry.insert(0, desired_text)
                        entry.config(state='readonly')  # Prevent editing while locked
                        print(f"[GUI DEBUG] P{participant_id}: Entry field updated, new text='{entry.get()}'", flush=True)
            else:
                # Unlocked - show enrollment status as button text with color coding

                # CRITICAL FIX: Clear ANY locked status text when unlocking
                if hasattr(self, 'participant_entries'):
                    entry = self.participant_entries[slot_idx]
                    current_text = entry.get()
                    participant_name = self.participant_names.get(slot_idx, f"P{participant_id}")
                    # Strip any existing locked status from stored name
                    participant_name = participant_name.replace(" (Locked, Absent)", "").replace(" (Locked)", "").strip()

                    # Check if entry contains any locked status
                    if "(Locked" in current_text:  # Matches both "(Locked)" and "(Locked, Absent)"
                        entry.delete(0, 'end')
                        entry.insert(0, participant_name)

                    # CRITICAL FIX: ALWAYS set to normal when unlocked (not just when text has "(Locked)")
                    entry.config(state='normal')  # Re-enable editing

                if enrollment_state in ['ENROLLED', 'IMPROVING']:
                    # Ready to lock! Both states indicate successful enrollment
                    # Apply 'lock_ready' style - green background, clickable
                    style = self.lock_button_styles['lock_ready']
                    lock_btn.config(
                        text="Lock Ready",
                        width=12,
                        bg=style['bg'],
                        fg=style['fg'],
                        activebackground=style['activebackground'],
                        state=style['state'],
                        relief=style['relief'],
                        cursor='hand2'  # Hand cursor indicates clickability
                    )
                elif enrollment_state == 'COLLECTING':
                    # Actively collecting face samples
                    # Apply 'enrolling' style - light grey, disabled
                    style = self.lock_button_styles['enrolling']
                    lock_btn.config(
                        text="Enrolling... (Collecting)",
                        width=25,
                        bg=style['bg'],
                        fg=style['fg'],
                        activebackground=style['activebackground'],
                        state=style['state'],
                        relief=style['relief'],
                        cursor='arrow'  # Normal cursor for disabled button
                    )
                elif enrollment_state == 'VALIDATING':
                    # Validating consistency and stability
                    # Apply 'enrolling' style - light grey, disabled
                    style = self.lock_button_styles['enrolling']
                    lock_btn.config(
                        text="Enrolling... (Validating)",
                        width=25,
                        bg=style['bg'],
                        fg=style['fg'],
                        activebackground=style['activebackground'],
                        state=style['state'],
                        relief=style['relief'],
                        cursor='arrow'  # Normal cursor for disabled button
                    )
                elif enrollment_state == 'FAILED':
                    # Enrollment failed
                    # Apply 'failed' style - red background, disabled
                    style = self.lock_button_styles['failed']
                    lock_btn.config(
                        text="Enrollment Failed",
                        width=18,
                        bg=style['bg'],
                        fg=style['fg'],
                        activebackground=style['activebackground'],
                        state=style['state'],
                        relief=style['relief'],
                        cursor='arrow'  # Normal cursor for disabled button
                    )
                elif enrollment_state == 'UNKNOWN':
                    # No face detected or no enrollment data
                    # Apply 'disabled' style - medium grey, disabled
                    style = self.lock_button_styles['disabled']
                    lock_btn.config(
                        text="Awaiting participant",
                        width=20,
                        bg=style['bg'],
                        fg=style['fg'],
                        activebackground=style['activebackground'],
                        state=style['state'],
                        relief=style['relief'],
                        cursor='arrow'  # Normal cursor for disabled button
                    )
                else:
                    # Fallback for unexpected states
                    # Apply 'disabled' style - medium grey, disabled
                    style = self.lock_button_styles['disabled']
                    lock_btn.config(
                        text="Enrollment not complete",
                        width=20,
                        bg=style['bg'],
                        fg=style['fg'],
                        activebackground=style['activebackground'],
                        state=style['state'],
                        relief=style['relief'],
                        cursor='arrow'  # Normal cursor for disabled button
                    )

                # Ensure entry is editable when unlocked
                if hasattr(self, 'participant_entries'):
                    entry = self.participant_entries[slot_idx]
                    if entry.cget('state') == 'readonly':
                        entry.config(state='normal')

    def get_label_for_face(self, global_id):
        """
        Given a global participant ID (integer), return display label:
        - Use GUI participant name if set, else default to 'P#'.
        - Mapping is 1-based (global_id starts at 1).
        """
        idx = global_id - 1  # zero-based for lists/dicts
        name = self.participant_names.get(idx)
        if name and name.strip() and not name.strip().startswith('P'):
            return name.strip()
        return f"P{global_id}"

    def update_gui_health(self):
        """
        Independent GUI health monitoring - maintains GUI responsiveness tracking
        without being tied to camera preview lifecycle.
        """
        # Reschedule every 100ms for consistent GUI health tracking
        if not self._shutdown_in_progress:
            self.after(100, self.update_gui_health)
        
        # Update GUI responsiveness timestamp for reliability monitoring
        if hasattr(self, 'reliability_monitor'):
            if hasattr(self.reliability_monitor, 'update_gui_timestamp'):
                self.reliability_monitor.update_gui_timestamp()

    def _start_preview_for_camera(self, cam_idx):
        """Start preview loop when camera becomes active"""
        # Cleanup 3: Defensive check and simplify for single-camera mode
        if not self.frames or len(self.frames) == 0:
            logger.warning("[GUI] No frames available for preview")
            return

        # Clear loading overlay from single canvas when preview starts
        info = self.frames[0]
        canvas = info.get('canvas')
        if canvas:
            canvas.delete('loading_overlay')

            # Stop spinner animation for this canvas
            if 0 in self._canvas_spinners:
                after_id = self._canvas_spinners[0].get('after_id')
                if after_id:
                    self.after_cancel(after_id)
                del self._canvas_spinners[0]

                # Clean up transparent overlay PhotoImage reference
                if 0 in self._overlay_images:
                    del self._overlay_images[0]

        # Mark camera as ready to show (dismiss "initializing..." splash)
        if not hasattr(self, '_camera_faces_detected'):
            self._camera_faces_detected = {}
        self._camera_faces_detected[cam_idx] = True

        if not hasattr(self, '_preview_active') or not self._preview_active:
            self._preview_active = True
            self.after(16, self.schedule_preview)  # Start 60 FPS loop
            print(f"[GUI] Started preview loop for camera {cam_idx}")
        else:
            print(f"[GUI] Preview loop already active for camera {cam_idx}")

    def _stop_preview_for_camera(self, cam_idx):
        """Stop preview loop when no cameras are active"""
        if hasattr(self, 'active_camera_procs'):
            active_cameras = len([k for k in self.active_camera_procs.keys() if k != cam_idx])
            if active_cameras == 0:  # This camera being stopped is the last one
                self._preview_active = False
                print(f"[GUI] Stopped preview loop - camera {cam_idx} was last active camera")
            else:
                print(f"[GUI] Preview loop continues - {active_cameras} other cameras still active")

    def continuous_correlation_monitor(self):
        """Monitor correlation from shared buffer - SAFE VERSION"""
        # Check shutdown flag before rescheduling
        if not self._shutdown_in_progress:
            self.after(33, self.continuous_correlation_monitor)

        # Safe shared memory access with validation
        if hasattr(self, 'corr_array') and self.corr_array is not None:
            try:
                # Validate array before access
                if hasattr(self.corr_array, 'copy'):
                    corr = self.corr_array.copy()
                    # CRITICAL FIX: Always update plot, even with zero values
                    # Bar graph needs to show zeros during silent periods
                    self.update_plot(corr)
            except (AttributeError, ValueError, RuntimeError) as e:
                # Array was deleted during cleanup - this is expected during shutdown
                if not self._shutdown_in_progress:
                    logger.debug(f"Correlation monitor access error: {e}")
                
    def _get_theme_colors(self):
        """Get colors based on current theme"""
        if self.current_theme == "dark":
            return {
                'canvas_bg': '#1e1e1e',      # Darker background for better contrast
                'text_fg': '#ffffff',        # White text
                'center_line': '#4a4a4a',    # Subtle gray line
                'bar_green_start': '#00ff88',  # Bright green
                'bar_green_end': '#00aa55',    # Darker green
                'bar_red_start': '#ff4466',    # Bright red
                'bar_red_end': '#cc0033',      # Darker red
                'value_text': '#ffffff',     # White value text
                # ECG-specific colors
                'ecg_fig_bg': '#1e1e1e',     # Match main canvas background
                'ecg_axes_bg': '#2a2a2a',    # Slightly lighter for depth
                'ecg_line': '#00ff88',       # Bright green (match bar_green_start)
                'ecg_grid': '#3a3a3a',       # Subtle grid
                'ecg_text': '#ffffff',       # White text
                'ecg_axis_label': '#aaaaaa', # Gray axis labels
                'ecg_bpm_bg': '#2a2a2a',     # Modern indicator background
                'ecg_bpm_accent': '#ff4466', # Heart icon color (match bar_red)
                'ecg_battery_high': '#00ff88',   # >50% (bright green)
                'ecg_battery_mid': '#ffaa00',    # 20-50% (amber)
                'ecg_battery_low': '#ff4466'     # <20% (red)
            }
        else:  # light theme
            return {
                'canvas_bg': '#ffffff',      # White background
                'text_fg': '#000000',        # Black text
                'center_line': '#cccccc',    # Light gray line
                'bar_green_start': '#00dd66',  # Bright green
                'bar_green_end': '#009944',    # Darker green
                'bar_red_start': '#ff3355',    # Bright red
                'bar_red_end': '#bb0022',      # Darker red
                'value_text': '#000000',     # Black value text
                # ECG-specific colors
                'ecg_fig_bg': '#ffffff',     # White background
                'ecg_axes_bg': '#f5f5f5',    # Light gray for depth
                'ecg_line': '#0088cc',       # Blue ECG line
                'ecg_grid': '#dddddd',       # Light grid
                'ecg_text': '#000000',       # Black text
                'ecg_axis_label': '#666666', # Dark gray axis labels
                'ecg_bpm_bg': '#f5f5f5',     # Light indicator background
                'ecg_bpm_accent': '#ff3355', # Heart icon color (match bar_red)
                'ecg_battery_high': '#00aa44',   # >50% (green)
                'ecg_battery_mid': '#ff9900',    # 20-50% (orange)
                'ecg_battery_low': '#ff3355'     # <20% (red)
            }

    def _interpolate_color(self, color1, color2, factor):
        """Interpolate between two hex colors"""
        # Convert hex to RGB
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)

        # Interpolate
        r = int(r1 + (r2 - r1) * factor)
        g = int(g1 + (g2 - g1) * factor)
        b = int(b1 + (b2 - b1) * factor)

        return f'#{r:02x}{g:02x}{b:02x}'

    # _draw_gradient_bar method removed (was for co-modulation bars)
    # update_plot method removed (was for co-modulation bars)
    # Metrics panel updates will be implemented in Phase 2


    # _gui_update_loop removed (was for co-modulation plot updates)

    def on_record_toggle(self):
        """Handle recording toggle state change"""
        enabled = self.record_video.get()
        state = 'normal' if enabled else 'disabled'
        
        # Enable/disable recording controls
        self.dir_entry.config(state=state)
        self.name_entry.config(state=state)
        
        # Save preference
        self.config.set('video_recording.enabled', enabled)
        
        # Update browse button state
        for widget in self.dir_entry.master.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.config(state=state)
    
    def browse_save_directory(self):
        """Browse for save directory"""
        directory = filedialog.askdirectory(
            initialdir=self.save_dir.get(),
            title="Select Recording Directory"
        )
        if directory:
            self.save_dir.set(directory)
            self.config.set('video_recording.save_directory', directory)

    def _start_frame_production_for_recording(self):
        """
        Start a thread that copies frames from display buffers to recording queues.
        This is the PRODUCER that feeds frames to recording threads (the CONSUMERS).
        """
        def frame_producer():
            """Copy display data to recording queues for all active cameras"""
            logger.info("[FRAME PRODUCER] Started frame production thread for recording")
            logger.info(f"[FRAME PRODUCER] Found {len(self.recording_queues)} recording queues: {list(self.recording_queues.keys())}")

            if not self.recording_queues:
                logger.warning("[FRAME PRODUCER] No recording queues available - recording will not work!")
                logger.warning("[FRAME PRODUCER] Cameras must be started before recording begins")

            while self.recording_active:
                try:
                    # Get all cameras that have recording queues
                    for camera_index, recording_queue in self.recording_queues.items():
                        # Read display data from display buffer manager
                        if hasattr(self, 'display_buffer_manager') and self.display_buffer_manager:
                            display_data = self.display_buffer_manager.get_display_data(camera_index)

                            if display_data and display_data.frame_bgr is not None:
                                # Prepare data dict matching what recording thread expects
                                frame_data = {
                                    'frame_bgr': display_data.frame_bgr,
                                    'faces': display_data.faces if display_data.faces else [],
                                    'all_poses': display_data.poses if hasattr(display_data, 'poses') else []
                                }

                                # Non-blocking put to avoid blocking if queue is full
                                try:
                                    recording_queue.put_nowait(frame_data)
                                except:
                                    # Queue full - skip this frame (older frames will be used)
                                    pass

                    # Sleep briefly to avoid tight loop (aim for ~30 fps production)
                    time.sleep(0.033)  # ~30 fps

                except Exception as e:
                    logger.error(f"[FRAME PRODUCER] Error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)  # Avoid rapid error loop

            logger.info("[FRAME PRODUCER] Frame production thread stopped")

        # Start the producer thread
        self.frame_producer_thread = threading.Thread(target=frame_producer, daemon=True)
        self.frame_producer_thread.start()
        logger.info("[FRAME PRODUCER] Frame production thread started")

    def _start_video_recording(self, active_workers):
        """Start video recording with optional audio for all active workers"""
        # Set immediate recording flag if not already recording
        if not self.recording_active:
            self.immediate_recording = True
            
        if not self.record_video.get() and not self.immediate_recording:
            return
        self.recording_active = True

        # Start frame production thread to feed recording queues
        self._start_frame_production_for_recording()

        # Save recording state for recovery protection
        if hasattr(self, 'recording_protection'):
            self.recording_protection._save_emergency_state()
            
        # Create save directory if it doesn't exist
        save_path = Path(self.save_dir.get())
        try:
            save_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Recording Error", f"Cannot create directory: {e}")
            return
        
        # Check if we should record audio with video
        record_audio_with_video = (
            self.audio_enabled.get() and 
            self.audio_mode.get() == "with_video"
        )
        
        # Save current settings
        self.config.set('video_recording.save_directory', str(save_path))
        self.config.set('video_recording.filename_template', self.filename_template.get())
        
        # Create recorder for each active worker
        self.video_recorders.clear()
        recording_count = 0

        # Check if we have any active cameras
        if not active_workers or len(active_workers) == 0:
            logger.warning("[GUI] No active cameras for recording")
            messagebox.showerror("Recording Error", "No cameras running")
            return

        logger.info(f"[VIDEO RECORDING] Starting recording for {len(active_workers)} active camera(s)")

        # Loop through all active cameras and create a recorder for each
        for worker_info in active_workers:
            idx = worker_info.get('camera_index', 0)
            info = worker_info

            logger.info(f"[VIDEO RECORDING] Processing camera {idx} for recording")

            # Get frame dimensions from stored resolution
            resolution = info.get('resolution', (640, 480))
            actual_width, actual_height = resolution

            try:
                recorder = self._create_video_recorder(idx, record_audio_with_video)
                if recorder is None:
                    logger.error(f"[GUI] Failed to create recorder for cam {idx}")
                    continue  # Skip this camera, try next one

                # Store recorder info
                self.video_recorders[idx] = {
                    'recorder': recorder,
                    'capture_thread': None,
                    'stop_flag': None
                }

                # Define the recording thread function with proper scope
                def create_recording_thread(recorder, recording_queue, width, height, idx, participant_names):
                    stop_flag = threading.Event()
                    
                    def add_recording_info_overlay(frame, frame_count, lsl_timestamp, recording_fps=None):
                        """
                        Add recording information overlay to frame.
                        
                        Args:
                            frame: The video frame to overlay on
                            frame_count: Current frame number
                            lsl_timestamp: LSL timestamp (seconds since epoch)
                            recording_fps: Optional actual recording FPS
                        """
                        h, w = frame.shape[:2]
                        
                        # Format timestamp for display
                        dt = datetime.fromtimestamp(lsl_timestamp)
                        time_str = dt.strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
                        
                        # Prepare text lines
                        text_lines = [
                            f"Frame: {frame_count:06d}",
                            f"Time: {time_str}",
                            f"LSL: {lsl_timestamp:.6f}",
                        ]
                        
                        if recording_fps is not None:
                            text_lines.append(f"FPS: {recording_fps:.1f}")
                        
                        # Style settings
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1
                        text_color = (0, 255, 0)  # Green
                        bg_color = (0, 0, 0)  # Black background
                        padding = 5
                        line_height = 20
                        
                        # Calculate text dimensions
                        max_width = 0
                        for text in text_lines:
                            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                            max_width = max(max_width, text_size[0])
                        
                        # Position in top-right corner
                        x_start = w - max_width - 15
                        y_start = 15
                        
                        # Draw semi-transparent background
                        overlay = frame.copy()
                        cv2.rectangle(overlay,
                                    (x_start - padding, y_start - padding),
                                    (w - 5, y_start + len(text_lines) * line_height + padding),
                                    bg_color, -1)
                        
                        # Blend with original (semi-transparent background)
                        alpha = 0.7
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                        
                        # Draw text lines
                        for i, text in enumerate(text_lines):
                            y = y_start + (i + 1) * line_height
                            cv2.putText(frame, text, (x_start, y),
                                        font, font_scale, text_color, thickness, cv2.LINE_AA)
                        
                        return frame

                    def record_overlay_thread():
                        """Fixed overlay recording thread with proper frame timing"""
                        frame_count = 0
                        last_frame_time = time.time()
                        frame_interval = 1.0 / CAPTURE_FPS  # Target interval between frames
                        
                        while self.recording_active and not stop_flag.is_set():
                            try:
                                # Calculate when next frame should be captured
                                current_time = time.time()
                                time_since_last = current_time - last_frame_time
                                
                                # Skip if we're ahead of schedule
                                if time_since_last < frame_interval:
                                    time.sleep(frame_interval - time_since_last)
                                    continue
                                                            
                                try:
                                    latest = recording_queue.get(timeout=0.1)
                                except:
                                    continue
                                
                                if latest is None:
                                    continue
                                    
                                frame_bgr = latest.get('frame_bgr')
                                if frame_bgr is None:
                                    continue
                                
                                # Make a copy to avoid modifying the preview
                                frame_bgr = frame_bgr.copy()

                                # Get ALL faces data (multiple faces)
                                faces = latest.get('faces', [])
                                
                                # Get ALL poses data (multiple poses)
                                all_poses = latest.get('all_poses', [])
                                
                                # Build labels dictionary for faces using global IDs
                                labels = {}
                                for face in faces:
                                    global_id = face.get('id')
                                    if global_id and (not isinstance(global_id, str) or not str(global_id).startswith('local_')):
                                        # Get participant name from global manager
                                        labels[global_id] = self.global_participant_manager.get_participant_name(global_id)

                                # Event-driven enrollment states (updated by queue consumer)
                                enrollment_states = self.enrollment_state_cache

                                # Draw overlays with all faces and poses
                                overlayed = draw_overlays_combined(
                                    frame_bgr,
                                    faces=faces,  # Pass all faces
                                    pose_landmarks=None,
                                    labels=labels,
                                    face_mesh=False,
                                    face_contours=True,
                                    face_points=False,
                                    pose_lines=True,  # Enable centralized pose drawing with transparency
                                    enrollment_states=enrollment_states,
                                    all_poses=all_poses,
                                    config=self.config
                                )
                                
                                # Draw bboxes from advanced detection
                                for face in faces:
                                    global_id = face.get('id')
                                    if global_id and (idx, global_id) in self.participant_bboxes:
                                        bbox = self.participant_bboxes[(idx, global_id)]
                                        if bbox and len(bbox) == 4:
                                            x1, y1, x2, y2 = [int(v) for v in bbox]
                                            # Draw green bbox
                                            cv2.rectangle(overlayed, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            # Draw label with background using PIL (Unicode support)
                                            label = labels.get(global_id, f"P{global_id}")
                                            overlayed = _draw_pil_text_with_background(
                                                overlayed, label, (x1+5, y1),
                                                font_size=20, text_color=(0, 0, 0), bg_color=(0, 255, 0)
                                            )

                                # Add recording info overlay
                                overlayed = add_recording_info_overlay(
                                    overlayed, 
                                    frame_count, 
                                    current_time,
                                    1.0 / (current_time - last_frame_time) if last_frame_time else None
                                )
                                
                                # Get actual frame dimensions
                                actual_h, actual_w = overlayed.shape[:2]
                                
                                # Only resize if dimensions don't match recorder expectations
                                if (actual_w, actual_h) != (width, height):
                                    print(f"[Recorder Thread] Resizing from {actual_w}x{actual_h} to {width}x{height}")
                                    overlayed = cv2.resize(overlayed, (width, height), interpolation=cv2.INTER_LINEAR)
                                
                                # Add frame to recorder with global frame ID for synchronization
                                global_frame_id = self.get_next_global_frame_id()
                                if recorder.add_frame(overlayed, frame_id=global_frame_id):
                                    frame_count += 1
                                    last_frame_time = current_time  # Update last frame time
                                    
                                    if frame_count % CAPTURE_FPS == 0:  # Log every second
                                        actual_fps = CAPTURE_FPS / (current_time - last_frame_time + frame_interval * CAPTURE_FPS)
                                        print(f"[Recorder Thread {idx}] {frame_count} frames, actual FPS: {actual_fps:.1f}")
                                
                            except Exception as e:
                                print(f"[Recorder Thread {idx}] Error: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        print(f"[Recorder Thread {idx}] Stopped after {frame_count} frames")
                    
                    # Start the thread
                    thread = threading.Thread(target=record_overlay_thread, daemon=True)
                    thread.start()
                    return thread, stop_flag
                # Now call the function to start the recording thread
                print(f"[GUI] Starting overlay recording thread for cam {idx}")
                recording_queue = self.recording_queues.get(idx)
                if not recording_queue:
                    logger.error(f"[VIDEO RECORDING] No recording queue for camera {idx}")
                    continue  # Skip this camera, process next one

                capture_thread, stop_flag = create_recording_thread(
                    recorder,
                    recording_queue,
                    actual_width,
                    actual_height,
                    idx,
                    self.participant_names,
                )

                self.video_recorders[idx]['capture_thread'] = capture_thread
                self.video_recorders[idx]['stop_flag'] = stop_flag
                print(f"[GUI] Overlay capture thread started for cam {idx}")
                print(f"[GUI] Started recording video stream for camera {idx}")
                recording_count += 1

            except Exception as e:
                logger.error(f"[GUI] Failed to start recorder for cam {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue  # Skip this camera, try next one

        # Update status based on recording count
        if recording_count > 0:
            save_path = Path(self.save_dir.get())
            self.record_status.config(
                text=f"Recording {recording_count} camera(s) to {save_path.name}/",
                foreground='red'
            )
            logger.info(f"[VIDEO RECORDING] Successfully started recording {recording_count}/{len(active_workers)} camera(s)")
        else:
            self.record_status.config(text="Recording failed - no cameras started", foreground='red')
            logger.error("[VIDEO RECORDING] Failed to start recording any cameras")

        # Start standalone audio recorders if enabled
        if self.audio_enabled.get() and self.audio_mode.get() == "standalone":
            self._start_standalone_audio_recording()



    def _create_video_recorder(self, idx, record_audio):
        """Create a video recorder process - dimensions auto-detected"""
        try:
            # Generate filename based on template
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            participant_name = self.participant_names.get(idx, f"P{idx+1}")
            
            filename = self.filename_template.get().format(
                participant=participant_name,
                camera=f"cam{idx}",
                timestamp=timestamp
            )
            
            # Ensure .avi extension
            if not filename.endswith('.avi'):
                filename += '.avi'
            
            save_path = Path(self.save_dir.get())
            
            # Create the recorder process
            recorder = VideoRecorderProcess(
                output_dir=str(save_path),
                codec='MJPG',
                fps=CAPTURE_FPS,
                camera_index=idx,
                config=self.config.config
            )
            
            # Start recording - dimensions will be auto-detected, with timestamp overlay enabled
            if recorder.start_recording(participant_name, filename, annotate_frames=True):
                print(f"[GUI] Created recorder for {participant_name}: {filename} (timestamp overlay enabled)")
                recorder.participant = participant_name
                return recorder
            else:
                print(f"[GUI] Failed to start recorder for {participant_name}")
                return None
                
        except Exception as e:
            print(f"[GUI] Error creating recorder for camera {idx}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _stop_video_recording(self):
        """Stop all process recorders + capture threads."""
        if not self.recording_active:
            return
            
        print("[GUI] Stopping video recording...")
        self.recording_active = False  # Signal capture threads to exit
        
        # First, wait for capture threads to finish
        for idx, rec_info in list(self.video_recorders.items()):
            if 'capture_thread' in rec_info and rec_info['capture_thread']:
                print(f"[GUI] Waiting for capture thread {idx}...")
                rec_info['capture_thread'].join(timeout=2.0)
            if 'stop_flag' in rec_info and rec_info['stop_flag']:
                rec_info['stop_flag'].set()
        
        # Give recorders time to process remaining frames
        time.sleep(0.5)
        
        # Then stop recorders
        for idx, rec_info in list(self.video_recorders.items()):
            try:
                recorder = rec_info['recorder']
                if hasattr(recorder, 'get_frame_count'):
                    frame_count = recorder.get_frame_count()
                    print(f"[GUI] Stopping recorder {idx} with {frame_count} frames...")
                else:
                    print(f"[GUI] Stopping recorder {idx}...")
                
                recorder.stop_recording()
                
                # Verify recording was saved
                save_path = Path(self.save_dir.get())
                participant = getattr(recorder, 'participant', f'P{idx+1}')
                recordings = list(save_path.glob(f"{participant}_*.avi"))
                if recordings:
                    latest = max(recordings, key=lambda p: p.stat().st_mtime)
                    size_mb = latest.stat().st_size / (1024 * 1024)
                    print(f"[GUI] Saved {latest.name} ({size_mb:.1f} MB)")
                    
            except Exception as e:
                print(f"[GUI] Error stopping recorder {idx}: {e}")
                import traceback
                traceback.print_exc()
        
        self.video_recorders.clear()
        self.immediate_recording = False  # Reset immediate recording flag
        self.record_status.config(text="Recording stopped", foreground='gray')


    def on_audio_toggle(self):
        """Handle audio recording toggle"""
        enabled = self.audio_enabled.get()
        state = 'normal' if enabled else 'disabled'
        
        # Enable/disable audio controls
        for widget in self.audio_mode_frame.winfo_children():
            widget.config(state=state)
        self.audio_device_btn.config(state=state)
        
        # Save preference
        self.config.set('audio_recording.enabled', enabled)
        
        # Update status
        if enabled:
            self.refresh_audio_status()
        else:
            self.audio_status.config(text="Audio: Disabled", foreground='gray')
    
    def on_audio_mode_change(self):
        """Handle audio mode change"""
        mode = self.audio_mode.get()
        self.config.set('audio_recording.standalone_audio', mode == "standalone")
        self.config.set('audio_recording.audio_with_video', mode == "with_video")
        
        # Update status
        self.refresh_audio_status()

    def start_audio_recording(self):
        """Start standalone audio recording"""
        if self.audio_mode.get() != "standalone":
            messagebox.showinfo("Audio Mode", "Audio recording is set to 'with video' mode. Use video recording controls.")
            return
            
        if not self.audio_device_assignments:
            messagebox.showwarning("No Devices", "Please configure audio devices first")
            return
            
        self._start_standalone_audio_recording()
        self.audio_recording_active = True
        self.start_audio_btn.config(state='disabled')
        self.stop_audio_btn.config(state='normal')
        self.audio_status.config(text=self.audio_status.cget("text") + " (recording)", foreground='red')

    def stop_audio_recording(self):
        """Stop standalone audio recording"""
        self.audio_recording_active = False
        
        # Stop all audio recorders
        for assignment, recorder in self.audio_recorders.items():
            recorder.stop_recording()
            print(f"[GUI] Stopped audio recording for {assignment}")
        
        self.audio_recorders.clear()
        self.start_audio_btn.config(state='normal' if self.audio_enabled.get() else 'disabled')
        self.stop_audio_btn.config(state='disabled')
        self.refresh_audio_status()

    def on_audio_toggle(self):
        """Handle audio recording toggle"""
        enabled = self.audio_enabled.get()
        state = 'normal' if enabled else 'disabled'
        
        # Enable/disable audio controls
        for widget in self.audio_mode_frame.winfo_children():
            widget.config(state=state)
        self.audio_device_btn.config(state=state)
        
        # Handle button states based on mode
        if enabled and self.audio_mode.get() == "standalone":
            self.start_audio_btn.config(state='normal' if not self.audio_recording_active else 'disabled')
            self.stop_audio_btn.config(state='normal' if self.audio_recording_active else 'disabled')
        else:
            self.start_audio_btn.config(state='disabled')
            self.stop_audio_btn.config(state='disabled')
        
        # Save preference
        self.config.set('audio_recording.enabled', enabled)
        
        # Update status
        if enabled:
            self.refresh_audio_status()
        else:
            self.audio_status.config(text="Audio: Disabled", foreground='gray')
    
    def refresh_audio_devices(self):
        """Refresh list of available audio devices"""
        try:
            self.available_audio_devices = AudioDeviceManager.list_audio_devices()
            print(f"[GUI] Found {len(self.available_audio_devices)} audio input devices")
        except Exception as e:
            print(f"[GUI] Error listing audio devices: {e}")
            self.available_audio_devices = []
    
    def refresh_audio_status(self):
        """Update audio status label"""
        if not self.audio_enabled.get():
            self.audio_status.config(text="Audio: Disabled", foreground='gray')
            return
            
        mode = self.audio_mode.get()
        assigned_count = len(self.audio_device_assignments)
        
        if assigned_count == 0:
            self.audio_status.config(text="Audio: No devices assigned", foreground='orange')
        else:
            mode_text = "standalone" if mode == "standalone" else "with video"
            self.audio_status.config(
                text=f"Audio: {assigned_count} device(s) assigned ({mode_text})",
                foreground='green'
            )
    
    def configure_audio_devices(self):
        """Open audio device configuration dialog"""
        dialog = tk.Toplevel(self)
        dialog.title("Configure Audio Devices")
        dialog.geometry("600x400")
        dialog.transient(self)
        dialog.grab_set()
        
        # Instructions
        ttk.Label(
            dialog,
            text="Assign audio devices to participants:",
            font=('Arial', 10, 'bold')
        ).pack(pady=10)
        
        # Create scrollable frame
        canvas = tk.Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Device assignment widgets
        device_vars = {}
        
        # Create assignment for each participant
        for i in range(self.participant_count.get()):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill='x', padx=10, pady=5)
            
            # Get participant name from the entry field
            participant_name = self.participant_names.get(i, f"P{i+1}")
            
            ttk.Label(frame, text=f"{participant_name}:", width=20).pack(side='left')
            
            # Store by participant index
            var = tk.StringVar(value=self.audio_device_assignments.get(f"participant{i}", "None"))
            device_vars[f"participant{i}"] = var
            
            devices = ["None"] + [f"{d['index']}: {d['name']}" for d in self.available_audio_devices]
            combo = ttk.Combobox(frame, textvariable=var, values=devices, state='readonly', width=40)
            combo.pack(side='left', padx=(10, 0))
            
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0))
        scrollbar.pack(side="right", fill="y")
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', pady=10)
        
        def save_assignments():
            # Save device assignments
            self.audio_device_assignments.clear()
            for key, var in device_vars.items():
                value = var.get()
                if value != "None":
                    # Extract device index from string
                    try:
                        device_index = int(value.split(":")[0])
                        self.audio_device_assignments[key] = device_index
                    except:
                        pass
            
            # Save to config
            self.config.set('audio_devices', self.audio_device_assignments)
            self.refresh_audio_status()
            dialog.destroy()
        
        ttk.Button(button_frame, text="Save", command=save_assignments).pack(side='right', padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side='right', padx=5)
        
        # Refresh devices button
        ttk.Button(
            button_frame,
            text="Refresh Devices",
            command=lambda: [self.refresh_audio_devices(), dialog.destroy(), self.configure_audio_devices()]
        ).pack(side='left', padx=5)
        
    def _start_standalone_audio_recording(self):
        """Start standalone audio recording for assigned devices"""
        save_path = Path(self.save_dir.get())
        
        for assignment, device_index in self.audio_device_assignments.items():
            if device_index is None:
                continue
                
            # Determine participant ID from assignment
            if assignment.startswith("participant"):
                participant_num = int(assignment.replace("participant", ""))
                participant_id = self.participant_names.get(participant_num, f"P{participant_num+1}")
            else:
                # Fallback for old format
                participant_id = assignment
            
            # Create audio recorder
            recorder = AudioRecorder(
                device_index=device_index,
                sample_rate=self.config.get('audio_recording.sample_rate', 44100),
                channels=self.config.get('audio_recording.channels', 1),
                output_dir=str(save_path)
            )
            
            # Start recording
            if recorder.start_recording(participant_id):
                self.audio_recorders[assignment] = recorder
                print(f"[GUI] Started audio recording for {participant_id}")

    # ===== Bluetooth Device Management Methods =====

    def _create_single_ecg_strip(self):
        """
        Phase 2: Create single ECG strip for P1 (single-participant adaptation).

        Simplified version of _create_ecg_strips() for one participant.
        """
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        from collections import deque

        # Get theme colors
        theme_colors = self._get_theme_colors()

        # Initialize storage (single participant)
        self.ecg_strips = []
        self.ecg_buffers = {}
        self.ecg_dirty_flags = {}
        self.ecg_heart_rates = {}

        participant_id = self.single_participant_id

        # Create matplotlib figure with theme colors
        fig = Figure(figsize=(10, 1.5), dpi=100, facecolor=theme_colors['ecg_fig_bg'])
        ax = fig.add_subplot(111, facecolor=theme_colors['ecg_axes_bg'])

        # Configure axis with theme colors
        ax.set_xlim(0, 650)  # 5 seconds @ 130 Hz
        ax.set_ylim(-500, 500)  # Will auto-scale later
        ax.set_xlabel('Time (samples)', fontsize=9, color=theme_colors['ecg_axis_label'])
        ax.set_ylabel('μV', fontsize=9, color=theme_colors['ecg_axis_label'])
        ax.tick_params(labelsize=8, colors=theme_colors['ecg_axis_label'])
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=theme_colors['ecg_grid'])

        # Set spine colors
        for spine in ax.spines.values():
            spine.set_edgecolor(theme_colors['ecg_grid'])

        # Create empty line plot with theme color
        line, = ax.plot([], [], '-', linewidth=1.5, color=theme_colors['ecg_line'])

        # Add BPM indicator
        text_bpm = ax.text(
            0.02, 0.95, '❤ -- BPM',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            color=theme_colors['ecg_text'],
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor=theme_colors['ecg_bpm_bg'],
                edgecolor=theme_colors['ecg_bpm_accent'],
                alpha=0.9,
                linewidth=1.5
            )
        )

        # Add battery indicator
        text_battery = ax.text(
            0.98, 0.95, '🔋 -- %',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            color=theme_colors['ecg_text'],
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor=theme_colors['ecg_bpm_bg'],
                edgecolor=theme_colors['ecg_grid'],
                alpha=0.9,
                linewidth=1.5
            )
        )

        # Tight layout
        fig.tight_layout()

        # Embed canvas in frame
        canvas = FigureCanvasTkAgg(fig, self.timeseries_frame)
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        canvas.draw()

        # Store references
        self.ecg_strips.append({
            'frame': self.timeseries_frame,
            'figure': fig,
            'ax': ax,
            'canvas': canvas,
            'line': line,
            'text_bpm': text_bpm,
            'text_battery': text_battery,
            'participant_id': participant_id
        })

        # Initialize buffer for P1 (5 seconds @ 130 Hz = 650 samples)
        self.ecg_buffers[participant_id] = deque(maxlen=650)
        self.ecg_dirty_flags[participant_id] = False
        self.ecg_heart_rates[participant_id] = 0.0

        print(f"[ECG] Created single-participant ECG strip for {participant_id}")

        # Start update loop (reuse existing _refresh_ecg_strips)
        if not hasattr(self, '_ecg_refresh_scheduled') or not self._ecg_refresh_scheduled:
            self._ecg_refresh_scheduled = True
            self.after(50, self._refresh_ecg_strips)  # 20 Hz refresh rate

    def _create_ecg_strips(self):
        """
        Create ECG strip widgets for each participant with matplotlib canvas.

        Each strip shows a 5-second rolling window of ECG data with quality metrics.
        Uses theme-aware styling for dark/light mode support.
        """
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        from collections import deque

        # Get theme colors
        theme_colors = self._get_theme_colors()

        # Create ECG strip frame (right side of row1_container)
        self.ecg_strip_frame = ttk.Frame(self.row1_container, relief='flat', borderwidth=0)
        self.ecg_strip_frame.grid(row=0, column=1, sticky='nsew')

        # Initialize storage
        self.ecg_strips = []  # List of {frame, figure, ax, canvas, line, text_bpm, text_battery}
        self.ecg_buffers = {}  # {participant_id: deque(maxlen=650)}
        self.ecg_dirty_flags = {}  # {participant_id: bool} - track which strips need update
        self.ecg_heart_rates = {}  # {participant_id: float} - cached BPM calculations

        print(f"[ECG DEBUG] Creating ECG strips for {self.participant_count.get()} participant(s) with {self.current_theme} theme")

        # Create strip for each participant
        participant_count = self.participant_count.get()
        for i in range(participant_count):
            participant_id = self.participant_names.get(i, f"P{i+1}")

            # Create label frame for this participant's ECG
            strip_frame = ttk.LabelFrame(
                self.ecg_strip_frame,
                text=f"{participant_id} ECG",
                padding=5
            )
            strip_frame.grid(row=i, column=0, sticky='nsew', padx=5, pady=2)
            self.ecg_strip_frame.grid_rowconfigure(i, weight=1)

            # Create matplotlib figure with theme colors
            fig = Figure(figsize=(4.5, 1.2), dpi=100, facecolor=theme_colors['ecg_fig_bg'])
            ax = fig.add_subplot(111, facecolor=theme_colors['ecg_axes_bg'])

            # Configure axis with theme colors
            ax.set_xlim(0, 650)  # 5 seconds @ 130 Hz
            ax.set_ylim(-500, 500)  # Will auto-scale later
            ax.set_xlabel('Time (samples)', fontsize=8, color=theme_colors['ecg_axis_label'])
            ax.set_ylabel('μV', fontsize=8, color=theme_colors['ecg_axis_label'])
            ax.tick_params(labelsize=7, colors=theme_colors['ecg_axis_label'])
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=theme_colors['ecg_grid'])

            # Set spine colors
            for spine in ax.spines.values():
                spine.set_edgecolor(theme_colors['ecg_grid'])

            # Create empty line plot with theme color
            line, = ax.plot([], [], '-', linewidth=1.5, color=theme_colors['ecg_line'])

            # Add modern BPM indicator with theme styling
            text_bpm = ax.text(
                0.02, 0.95, '❤ -- BPM',
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                color=theme_colors['ecg_text'],
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor=theme_colors['ecg_bpm_bg'],
                    edgecolor=theme_colors['ecg_bpm_accent'],
                    alpha=0.9,
                    linewidth=1.5
                )
            )

            # Add modern battery indicator with theme styling
            text_battery = ax.text(
                0.98, 0.95, '🔋 -- %',
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                color=theme_colors['ecg_text'],
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor=theme_colors['ecg_bpm_bg'],
                    edgecolor=theme_colors['ecg_grid'],
                    alpha=0.9,
                    linewidth=1.5
                )
            )

            # Tight layout
            fig.tight_layout()

            # Embed canvas in frame
            canvas = FigureCanvasTkAgg(fig, strip_frame)
            canvas.get_tk_widget().pack(fill='both', expand=True)
            canvas.draw()

            # Store references
            self.ecg_strips.append({
                'frame': strip_frame,
                'figure': fig,
                'ax': ax,
                'canvas': canvas,
                'line': line,
                'text_bpm': text_bpm,
                'text_battery': text_battery,
                'participant_id': participant_id
            })

            # Initialize buffer for this participant (5 seconds @ 130 Hz = 650 samples)
            self.ecg_buffers[participant_id] = deque(maxlen=650)
            self.ecg_dirty_flags[participant_id] = False
            self.ecg_heart_rates[participant_id] = 0.0
            print(f"[ECG DEBUG] Created themed ECG strip and buffer for {participant_id}")

        print(f"[ECG DEBUG] Created buffers for: {list(self.ecg_buffers.keys())}")

        # Start update loop
        self._refresh_ecg_strips()

    def _refresh_ecg_strips(self):
        """
        Update ECG strip plots for participants with dirty flags.

        Called every 100ms to refresh visualizations.
        """
        if not hasattr(self, 'ecg_strips'):
            return

        # Initialize refresh counter (for heart rate calculation throttling)
        if not hasattr(self, '_ecg_refresh_counter'):
            self._ecg_refresh_counter = 0

        self._ecg_refresh_counter += 1

        # Calculate heart rate every 10 refreshes (1 second)
        calculate_hr = (self._ecg_refresh_counter % 10 == 0)

        # Update each strip that has new data
        for strip in self.ecg_strips:
            participant_id = strip['participant_id']

            # Check if this strip needs update
            if not self.ecg_dirty_flags.get(participant_id, False):
                continue

            # Get buffered data
            buffer = self.ecg_buffers.get(participant_id)
            if not buffer or len(buffer) == 0:
                if self._ecg_refresh_counter % 50 == 0:  # Print every 5 seconds
                    print(f"[ECG DEBUG] Strip {participant_id}: buffer empty or missing")
                continue

            # Log buffer status periodically
            if self._ecg_refresh_counter % 50 == 0:  # Every 5 seconds
                print(f"[ECG DEBUG] Strip {participant_id}: {len(buffer)} samples, dirty={self.ecg_dirty_flags.get(participant_id)}")

            # Convert deque to list
            data = list(buffer)
            x_data = list(range(len(data)))

            # Update line plot
            strip['line'].set_data(x_data, data)

            # Auto-scale Y-axis with padding
            if len(data) > 0:
                y_min = min(data)
                y_max = max(data)
                y_range = y_max - y_min
                if y_range > 0:
                    padding = y_range * 0.1
                    strip['ax'].set_ylim(y_min - padding, y_max + padding)

            # Update battery level and heart rate from bluetooth manager
            if hasattr(self, 'bluetooth_manager'):
                # Get theme colors for styling
                theme_colors = self._get_theme_colors()

                assignments = self.bluetooth_manager.get_all_assignments()
                for assignment in assignments:
                    if assignment.participant_id == participant_id:
                        status = self.bluetooth_manager.get_device_status(
                            participant_id,
                            assignment.mac_address
                        )

                        # Update battery level with theme-aware colors
                        battery = status.get('battery')
                        if battery is not None:
                            strip['text_battery'].set_text(f'🔋 {battery}%')
                            # Color code battery level with theme colors
                            if battery > 50:
                                edge_color = theme_colors['ecg_battery_high']
                            elif battery > 20:
                                edge_color = theme_colors['ecg_battery_mid']
                            else:
                                edge_color = theme_colors['ecg_battery_low']

                            strip['text_battery'].set_bbox(
                                dict(
                                    boxstyle='round,pad=0.5',
                                    facecolor=theme_colors['ecg_bpm_bg'],
                                    edgecolor=edge_color,
                                    alpha=0.9,
                                    linewidth=1.5
                                )
                            )
                            strip['text_battery'].set_color(theme_colors['ecg_text'])
                        else:
                            strip['text_battery'].set_text('🔋 -- %')
                            strip['text_battery'].set_bbox(
                                dict(
                                    boxstyle='round,pad=0.5',
                                    facecolor=theme_colors['ecg_bpm_bg'],
                                    edgecolor=theme_colors['ecg_grid'],
                                    alpha=0.9,
                                    linewidth=1.5
                                )
                            )
                            strip['text_battery'].set_color(theme_colors['ecg_text'])

                        # Calculate heart rate every second
                        if calculate_hr:
                            bpm = self._calculate_heart_rate(participant_id)
                            if bpm > 0:
                                self.ecg_heart_rates[participant_id] = bpm

                        # Update heart rate display with theme-aware styling
                        bpm = self.ecg_heart_rates.get(participant_id, 0.0)
                        if bpm > 0:
                            strip['text_bpm'].set_text(f'❤ {int(bpm)} BPM')
                        else:
                            strip['text_bpm'].set_text('❤ -- BPM')

                        strip['text_bpm'].set_color(theme_colors['ecg_text'])

                        break

            # Update canvas
            strip['canvas'].draw_idle()

            # Clear dirty flag
            self.ecg_dirty_flags[participant_id] = False

        # Schedule next update (10 Hz = every 100ms)
        self.after(100, self._refresh_ecg_strips)

    def _calculate_heart_rate(self, participant_id):
        """
        Calculate heart rate (BPM) from ECG buffer using simple R-peak detection.

        Uses scipy.signal.find_peaks with adaptive threshold.

        Args:
            participant_id: Participant ID to calculate BPM for

        Returns:
            float: Heart rate in BPM, or 0.0 if calculation fails
        """
        try:
            from scipy.signal import find_peaks

            # Get buffer
            buffer = self.ecg_buffers.get(participant_id)
            if not buffer or len(buffer) < 260:  # Need at least 2 seconds @ 130 Hz
                return 0.0

            data = list(buffer)

            # Adaptive threshold: 60% of max amplitude
            threshold = 0.6 * max(data) if max(data) > 0 else 0

            # Find R-peaks
            # distance: Minimum 390ms between peaks (154 samples @ 130 Hz) = max 154 BPM
            # height: Above threshold to avoid noise
            peaks, _ = find_peaks(data, height=threshold, distance=50)

            if len(peaks) < 2:
                return 0.0

            # Calculate average R-R interval
            rr_intervals = []
            for i in range(1, len(peaks)):
                interval_samples = peaks[i] - peaks[i-1]
                rr_intervals.append(interval_samples)

            if not rr_intervals:
                return 0.0

            # Average R-R interval in samples
            avg_rr_samples = sum(rr_intervals) / len(rr_intervals)

            # Convert to BPM (130 Hz sampling rate)
            # BPM = (60 seconds * 130 samples/second) / (avg_rr_samples)
            bpm = (60.0 * 130.0) / avg_rr_samples

            # Sanity check: 30-200 BPM range
            if 30 <= bpm <= 200:
                return bpm
            else:
                return 0.0

        except Exception as e:
            print(f"[GUI] Heart rate calculation error for {participant_id}: {e}")
            return 0.0

    def _bluetooth_data_callback(self, participant_id, stream_type, samples):
        """
        Callback for Bluetooth device data - sends to LSL via data queue.

        Args:
            participant_id: Participant ID (e.g., "P1")
            stream_type: Stream type (e.g., "ecg", "heartrate")
            samples: List of sample values
        """
        # Suppressed high-frequency debug output
        # print(f"[ECG DEBUG] Callback: participant_id={participant_id}, stream_type={stream_type}, samples={len(samples)} values")
        # if len(samples) > 0:
        #     print(f"[ECG DEBUG] First 5 samples: {samples[:5]}")

        # Send to LSL data queue
        if hasattr(self, 'lsl_data_queue'):
            self.lsl_data_queue.put({
                'type': 'bluetooth_data',
                'participant_id': participant_id,
                'stream_type': stream_type,
                'samples': samples
            })

        # Update ECG visualization if strips exist
        if hasattr(self, 'ecg_buffers'):
            if participant_id in self.ecg_buffers:
                self.ecg_buffers[participant_id].extend(samples)
                self.ecg_dirty_flags[participant_id] = True
                # Suppressed high-frequency debug output
                # print(f"[ECG DEBUG] Added {len(samples)} samples to buffer for {participant_id}, total={len(self.ecg_buffers[participant_id])}")
            else:
                print(f"[ECG DEBUG] WARNING: participant_id '{participant_id}' not in ecg_buffers keys: {list(self.ecg_buffers.keys())}")
        else:
            print(f"[ECG DEBUG] WARNING: ecg_buffers not created yet!")

    def _restore_bluetooth_assignments(self):
        """Restore Bluetooth device assignments from config"""
        bluetooth_config = self.config.get('bluetooth_devices', {})

        for participant_key, devices in bluetooth_config.items():
            # Skip comment keys and example entries
            if participant_key.startswith('_'):
                continue

            # Handle both old format (participant_id: device_dict) and new format (participant_id: [device_list])
            if isinstance(devices, dict):
                # Single device (old format)
                devices = [devices]

            # Skip if devices is not a list (malformed config)
            if not isinstance(devices, list):
                print(f"[GUI] Skipping malformed Bluetooth config for {participant_key}: {type(devices)}")
                continue

            for device in devices:
                # Skip if device is not a dict
                if not isinstance(device, dict):
                    continue

                mac = device.get('mac')
                device_type = device.get('type')
                enabled = device.get('enabled', True)

                if mac and device_type:
                    # Extract participant ID
                    if participant_key.startswith("participant"):
                        participant_num = int(participant_key.replace("participant", ""))
                        participant_id = self.participant_names.get(participant_num, f"P{participant_num+1}")
                    else:
                        participant_id = participant_key

                    # Add assignment to manager
                    self.bluetooth_manager.add_device_assignment(
                        participant_id=participant_id,
                        mac_address=mac,
                        device_type=device_type,
                        enabled=enabled
                    )
                    print(f"[ECG DEBUG] Device assigned: participant_id={participant_id}, mac={mac}")
                    if hasattr(self, 'ecg_buffers'):
                        print(f"[ECG DEBUG] Current ECG buffer keys: {list(self.ecg_buffers.keys())}")

    def _update_bluetooth_status(self):
        """Update Bluetooth status label"""
        assignments = self.bluetooth_manager.get_all_assignments()
        assigned_count = len([a for a in assignments if a.enabled])

        if assigned_count == 0:
            self.bluetooth_status.config(
                text="Bluetooth: Not configured",
                foreground='gray'
            )
        else:
            self.bluetooth_status.config(
                text=f"Bluetooth: {assigned_count} device(s) assigned",
                foreground='green'
            )

    def configure_bluetooth_devices(self):
        """Open Bluetooth device configuration dialog with live preview"""
        import asyncio
        import threading
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        dialog = tk.Toplevel(self)
        dialog.title("Configure Bluetooth Devices")
        dialog.geometry("800x600")
        dialog.transient(self)
        dialog.grab_set()

        # Instructions
        ttk.Label(
            dialog,
            text="Scan for Bluetooth devices and assign to participants:",
            font=('Arial', 10, 'bold')
        ).pack(pady=10)

        # Scan button frame
        scan_frame = ttk.Frame(dialog)
        scan_frame.pack(fill='x', padx=10, pady=5)

        scan_button = ttk.Button(scan_frame, text="Scan for Devices")
        scan_button.pack(side='left', padx=5)

        scan_status = ttk.Label(scan_frame, text="Ready to scan", foreground='gray')
        scan_status.pack(side='left', padx=10)

        # Discovered devices list
        devices_frame = ttk.LabelFrame(dialog, text="Discovered Devices", padding=10)
        devices_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Scrollable device list
        devices_canvas = tk.Canvas(devices_frame)
        devices_scrollbar = ttk.Scrollbar(devices_frame, orient="vertical", command=devices_canvas.yview)
        devices_scrollable = ttk.Frame(devices_canvas)

        devices_scrollable.bind(
            "<Configure>",
            lambda e: devices_canvas.configure(scrollregion=devices_canvas.bbox("all"))
        )

        devices_canvas.create_window((0, 0), window=devices_scrollable, anchor="nw")
        devices_canvas.configure(yscrollcommand=devices_scrollbar.set)

        devices_canvas.pack(side="left", fill="both", expand=True)
        devices_scrollbar.pack(side="right", fill="y")

        # Current assignments display
        assignments_frame = ttk.LabelFrame(dialog, text="Current Assignments", padding=10)
        assignments_frame.pack(fill='x', padx=10, pady=5)

        assignments_text = tk.Text(assignments_frame, height=5, state='disabled')
        assignments_text.pack(fill='x')

        # Device assignment storage
        discovered_devices = []
        assignment_widgets = {}  # {mac: {participant_var, preview_canvas, ...}}

        # Auto-load ZMQ discovered devices from config
        zmq_bridge_config = self.config.get('bluetooth_bridge', {})
        if zmq_bridge_config.get('enabled', False):
            discovered_zmq = zmq_bridge_config.get('discovered_devices', {})
            if discovered_zmq:
                # Convert discovered_devices dict to DeviceInfo list
                from core.data_streaming.bluetooth_backends.base import DeviceInfo
                for mac, dev_info in discovered_zmq.items():
                    if mac.startswith('_'):  # Skip comment keys
                        continue
                    zmq_device = DeviceInfo(
                        mac_address=mac,
                        name=dev_info['name'],
                        device_type='zmq_bluetooth',
                        rssi=None,  # N/A for ZMQ
                        metadata={'source': 'auto-discovered', 'port': dev_info['port'], 'topic': dev_info['topic']}
                    )
                    discovered_devices.append(zmq_device)

                print(f"[GUI] Auto-loaded {len(discovered_devices)} ZMQ Bluetooth device(s) from config")

        def update_assignments_display():
            """Update the current assignments text"""
            assignments = self.bluetooth_manager.get_all_assignments()

            assignments_text.config(state='normal')
            assignments_text.delete('1.0', tk.END)

            if not assignments:
                assignments_text.insert('1.0', "No devices assigned")
            else:
                for assignment in assignments:
                    status = self.bluetooth_manager.get_device_status(
                        assignment.participant_id,
                        assignment.mac_address
                    )
                    status_icon = "🟢" if status['connected'] else "🔴"
                    battery = f"{status['battery']}%" if status['battery'] is not None else "N/A"

                    assignments_text.insert(
                        tk.END,
                        f"{status_icon} {assignment.participant_id}: {assignment.mac_address} "
                        f"({assignment.device_type}) - Battery: {battery}\n"
                    )

            assignments_text.config(state='disabled')

        def scan_for_devices():
            """Scan for Bluetooth devices in background thread"""
            scan_button.config(state='disabled')
            scan_status.config(text="Scanning...", foreground='blue')

            def async_scan():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    # Discover from all backends
                    devices = loop.run_until_complete(
                        self.bluetooth_manager.discover_devices(timeout=10.0)
                    )

                    # Update GUI in main thread
                    dialog.after(0, lambda: display_discovered_devices(devices))

                except Exception as e:
                    dialog.after(0, lambda: scan_status.config(
                        text=f"Scan failed: {e}",
                        foreground='red'
                    ))
                finally:
                    dialog.after(0, lambda: scan_button.config(state='normal'))
                    loop.close()

            threading.Thread(target=async_scan, daemon=True).start()

        def display_discovered_devices(devices):
            """Display discovered devices in the list"""
            nonlocal discovered_devices
            discovered_devices = devices

            # Clear previous widgets
            for widget in devices_scrollable.winfo_children():
                widget.destroy()

            if not devices:
                ttk.Label(
                    devices_scrollable,
                    text="No devices found. Make sure devices are powered on and in pairing mode.",
                    foreground='gray'
                ).pack(pady=20)
                scan_status.config(text="No devices found", foreground='gray')
                return

            scan_status.config(text=f"Found {len(devices)} device(s)", foreground='green')

            # Create widget for each device
            for device in devices:
                device_frame = ttk.Frame(devices_scrollable, relief='raised', borderwidth=1)
                device_frame.pack(fill='x', padx=5, pady=5)

                # Device info
                info_frame = ttk.Frame(device_frame)
                info_frame.pack(fill='x', padx=10, pady=5)

                ttk.Label(
                    info_frame,
                    text=f"📱 {device.name}",
                    font=('Arial', 10, 'bold')
                ).pack(anchor='w')

                ttk.Label(
                    info_frame,
                    text=f"MAC: {device.mac_address} | Type: {device.device_type or 'Unknown'}",
                    foreground='gray'
                ).pack(anchor='w')

                if device.rssi is not None:
                    ttk.Label(
                        info_frame,
                        text=f"Signal: {device.rssi} dBm",
                        foreground='gray'
                    ).pack(anchor='w')

                # Assignment controls
                assign_frame = ttk.Frame(device_frame)
                assign_frame.pack(fill='x', padx=10, pady=5)

                ttk.Label(assign_frame, text="Assign to:").pack(side='left', padx=5)

                # Participant selection
                participant_var = tk.StringVar(value="None")
                participant_names = ["None"] + [
                    self.participant_names.get(i, f"P{i+1}")
                    for i in range(self.participant_count.get())
                ]
                participant_combo = ttk.Combobox(
                    assign_frame,
                    textvariable=participant_var,
                    values=participant_names,
                    state='readonly',
                    width=15
                )
                participant_combo.pack(side='left', padx=5)

                # Assign button
                def make_assign_callback(dev, var):
                    def assign():
                        pid = var.get()
                        if pid and pid != "None":
                            self.bluetooth_manager.add_device_assignment(
                                participant_id=pid,
                                mac_address=dev.mac_address,
                                device_type=dev.device_type,
                                enabled=True
                            )
                            update_assignments_display()
                            print(f"[GUI] Assigned {dev.mac_address} to {pid}")
                    return assign

                ttk.Button(
                    assign_frame,
                    text="Assign",
                    command=make_assign_callback(device, participant_var)
                ).pack(side='left', padx=5)

                # Remove button
                def make_remove_callback(dev):
                    def remove():
                        # Find and remove assignment
                        assignments = self.bluetooth_manager.get_all_assignments()
                        for assignment in assignments:
                            if assignment.mac_address == dev.mac_address:
                                self.bluetooth_manager.remove_device_assignment(
                                    assignment.participant_id,
                                    dev.mac_address
                                )
                                update_assignments_display()
                                print(f"[GUI] Removed {dev.mac_address}")
                                break
                    return remove

                ttk.Button(
                    assign_frame,
                    text="Remove",
                    command=make_remove_callback(device)
                ).pack(side='left', padx=5)

        scan_button.config(command=scan_for_devices)

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', pady=10)

        def save_and_close():
            """Save assignments to config and close dialog"""
            # Build config structure
            bluetooth_config = {}

            assignments = self.bluetooth_manager.get_all_assignments()
            for assignment in assignments:
                # Convert participant ID back to participant index
                participant_key = None
                for i in range(self.participant_count.get()):
                    if self.participant_names.get(i, f"P{i+1}") == assignment.participant_id:
                        participant_key = f"participant{i}"
                        break

                if participant_key:
                    if participant_key not in bluetooth_config:
                        bluetooth_config[participant_key] = []

                    bluetooth_config[participant_key].append({
                        'mac': assignment.mac_address,
                        'type': assignment.device_type,
                        'enabled': assignment.enabled
                    })

            # Save to config
            self.config.set('bluetooth_devices', bluetooth_config)
            self.bluetooth_device_assignments = bluetooth_config
            self._update_bluetooth_status()

            print(f"[GUI] Saved Bluetooth assignments: {bluetooth_config}")
            dialog.destroy()

        ttk.Button(button_frame, text="Save", command=save_and_close).pack(side='right', padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side='right', padx=5)

        # Initial display
        update_assignments_display()

        # If ZMQ devices were auto-loaded, display them immediately
        if discovered_devices:
            display_discovered_devices(discovered_devices)
            scan_status.config(
                text=f"Auto-loaded {len(discovered_devices)} ZMQ device(s) from config",
                foreground='green'
            )

    # ===== End Bluetooth Management Methods =====

    def start_stream(self):
        """Start unified streaming (LSL only, correlation already running)"""
        if self.streaming:
            print("[GUI] Already streaming, returning")
            return

        # Check for active cameras using new CameraWorkerManager architecture
        active_cameras = list(self.active_camera_procs.keys()) if hasattr(self, 'active_camera_procs') else []
        print(f"[GUI] Found {len(active_cameras)} active cameras: {active_cameras}")

        if not active_cameras:
            print("[GUI] No active cameras found - cannot start streaming")
            return
        
        # Re-initialize LSL helper if needed
        if not hasattr(self, 'lsl_helper_proc') or self.lsl_helper_proc is None or not self.lsl_helper_proc.is_alive():
            print("[GUI] LSL helper not running - initializing now...")
            self._initialize_lsl_helper()

            # Double-check the process started successfully
            if not self.lsl_helper_proc.is_alive():
                print("[GUI] ERROR: Failed to start LSL helper process - cannot stream")
                return
        else:
            print(f"[GUI] LSL helper already running with PID: {self.lsl_helper_proc.pid}")
        
        self.streaming = True

        # Start performance monitoring
        self.after(1000, self.update_performance_stats)

        # CRITICAL: Send initial mesh state to LSL process for each active camera
        # This ensures LSL process knows the mesh setting from startup
        current_mesh_state = self.enable_mesh.get()
        # Get expected camera count from config to match LSL initialization
        expected_cameras = self.config.get('startup_mode', {}).get('camera_count', len(self.frames))

        for cam_idx in range(expected_cameras):
            # Check if camera process is alive (reliable check, not dependent on status)
            if cam_idx in self.active_camera_procs and self.active_camera_procs[cam_idx].is_alive():
                self.lsl_command_queue.put({
                    'type': 'config_update',
                    'camera_index': cam_idx,
                    'mesh_enabled': current_mesh_state
                })
        print(f"[GUI] Sent initial mesh state ({current_mesh_state}) to LSL process for all active cameras")

        # Enable LSL streaming in GUI processing worker (for correlator)
        # Note: lsl_data_queue is now passed during worker creation, not via control message
        if hasattr(self, 'processing_control_queue'):
            try:
                self.processing_control_queue.put({
                    'type': 'start_lsl_streaming'
                    # lsl_data_queue NO LONGER PASSED HERE - already passed at process creation
                })
                logger.info("[GUI] Enabled LSL streaming in GUI processing worker")

                # Send current participant names to GUI worker
                # This locks names at stream creation time for LSL stream naming
                self.processing_control_queue.put({
                    'type': 'set_participant_names',
                    'names': self.participant_names.copy()  # Send copy of current names dict
                })
                logger.info(f"[GUI] Sent participant names to GUI worker: {self.participant_names}")
            except Exception as e:
                logger.warning(f"Failed to enable LSL streaming in worker: {e}")

        # NOTE: In MediaPipe architecture, streaming state is managed by GUI processing worker
        # No need to send to individual camera workers - they just capture frames
        print("[GUI] Streaming state managed by GUI processing worker")
        
        # Co-modulation stream removed
        
        # NOTE: Pose streams are now created dynamically when pose data is first detected
        # This prevents creating empty streams for non-existent participants
        # Pose stream creation happens in LSLHelper when type='pose_data' is received

        # ===== Start Bluetooth Devices =====
        # Create LSL streams for all assigned Bluetooth devices
        from core.data_streaming.bluetooth_backends import get_registry
        registry = get_registry()

        assignments = self.bluetooth_manager.get_all_assignments()
        for assignment in assignments:
            if assignment.enabled:
                # Get backend to retrieve stream info
                backend = registry.create_backend(assignment.device_type)
                if backend:
                    stream_info = backend.get_stream_info()

                    # Create LSL stream via command queue
                    self.lsl_command_queue.put({
                        'type': 'create_bluetooth_stream',
                        'participant_id': assignment.participant_id,
                        'stream_type': stream_info.stream_type,
                        'channel_count': stream_info.channel_count,
                        'nominal_srate': stream_info.nominal_srate,
                        'channel_names': stream_info.channel_names,
                        'channel_units': stream_info.channel_units,
                        'manufacturer': stream_info.manufacturer,
                        'model': stream_info.model
                    })

                    print(f"[GUI] Created Bluetooth stream for {assignment.participant_id}: "
                          f"{stream_info.stream_type} ({assignment.device_type})")

        # Start Bluetooth device connections and streaming BEFORE streaming_started
        # This ensures devices are connecting before LSL expects data
        self.bluetooth_manager.start_all()
        print(f"[GUI] Started {len(assignments)} Bluetooth device(s)")
        # ===== End Bluetooth Devices =====

        # NOW notify LSL helper that streaming started (after Bluetooth devices started)
        self.lsl_command_queue.put({
            'type': 'streaming_started',
            'max_participants': self.participant_count.get()
        })
        print("[GUI] Sent 'streaming_started' command to LSL helper")

        # Auto-start recording if "Start recording with LSL stream" toggle is enabled
        # This must be here (not in _initialize_lsl_helper) because LSL helper may already be running
        has_var = hasattr(self, 'record_with_lsl_var')
        toggle_enabled = has_var and self.record_with_lsl_var.get()
        not_recording = not self.recording_active

        logger.info(f"[LSL AUTO-START] Checking conditions: has_var={has_var}, toggle_enabled={toggle_enabled}, not_recording={not_recording}")

        if has_var and toggle_enabled and not_recording:
            logger.info("[LSL AUTO-START] All conditions met - scheduling video recording start")
            # Schedule recording start after a short delay to ensure LSL is fully initialized
            self.after(500, self._auto_start_lsl_synchronized_recording)
        else:
            if not has_var:
                logger.warning("[LSL AUTO-START] record_with_lsl_var not found")
            elif not toggle_enabled:
                logger.info("[LSL AUTO-START] Toggle not enabled (record_with_lsl_var=False)")
            elif not not_recording:
                logger.info("[LSL AUTO-START] Already recording (recording_active=True)")

        # Update UI to show streaming is active
        for info in self.frames:
            if info.get('meta_label'):
                info['meta_label'].config(text="LSL → ON", foreground='green')

        print(f"[GUI] Started LSL streaming - streams will be created dynamically with mesh={'enabled' if self.enable_mesh.get() else 'disabled'}")
    
    def _extract_shape_from_landmarks(self, landmarks):
        """Extract participant shape vector from landmarks for recognition."""
        if landmarks is None:
            return None
        
        try:
            # Convert landmarks to numpy array if needed
            if isinstance(landmarks, list):
                landmarks = np.array(landmarks)
            
            # Check if we have valid landmarks array
            if not isinstance(landmarks, np.ndarray) or landmarks.size == 0:
                return None
            
            # Use stable landmarks for participant recognition (same as in participantmanager)
            stable_indices = [33, 133, 362, 263, 168, 6, 10, 234, 454, 152]  # Key facial points
            
            if landmarks.shape[0] > max(stable_indices) and landmarks.shape[1] >= 2:
                # Extract stable landmarks - keep as 2D array for Procrustes
                stable_landmarks = landmarks[stable_indices, :2]  # Take only x,y coordinates
                # Return as 2D array (10, 2) not flattened
                return stable_landmarks
        except Exception as e:
            # Return None if shape extraction fails - will use position-based matching
            pass
        
        return None
    
    def _shutdown_all_processes(self, timeout=3.0):
        """Shutdown with camera worker manager - protected with try-except for robustness"""
        print("XXXXXX DEBUG: _shutdown_all_processes is being called! XXXXXX")
        print("\n[GUI] === COMPREHENSIVE SHUTDOWN INITIATED ===")
        print("[DEBUG] _shutdown_all_processes called - printing stack trace")
        import sys
        sys.stdout.flush()
        import traceback
        traceback.print_stack()
        sys.stdout.flush()

        # CRITICAL: Set shutdown flag FIRST to prevent error floods
        self._shutdown_in_progress = True

        # Cancel all pending callbacks first
        try:
            self._cancel_all_pending_callbacks()
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to cancel callbacks: {e}")

        # Stop reliability monitoring first
        try:
            if hasattr(self, 'reliability_monitor'):
                # Only call emergency_cleanup if not already called by signal handler
                if not getattr(self, '_emergency_cleanup_completed', False):
                    self.reliability_monitor.emergency_cleanup()
                    self._emergency_cleanup_completed = True
                self.reliability_monitor.stop_monitoring()
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to stop reliability monitor: {e}")

        # 1. Stop streaming flag first
        try:
            self.streaming = False
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to set streaming flag: {e}")

        # 2. Stop all recordings
        try:
            if self.recording_active:
                print("[GUI] Stopping video recording...")
                self._stop_video_recording()
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to stop video recording: {e}")

        try:
            if self.audio_recording_active:
                print("[GUI] Stopping audio recording...")
                self.stop_audio_recording()
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to stop audio recording: {e}")

        # 2.5. Shutdown display processing worker
        try:
            if hasattr(self, 'display_buffer_manager'):
                print("[GUI] Shutting down display processing worker...")
                self.shutdown_display_processing()
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to shutdown display processing: {e}")

        # 2.6. CRITICAL: Stop enrollment worker BEFORE camera workers
        # This prevents new register_track_participant messages from being queued
        try:
            if hasattr(self, 'global_participant_manager'):
                print("[GUI] Stopping enrollment worker thread...")
                self.global_participant_manager.stop_enrollment_worker(timeout=2.0)
                print("[GUI] Enrollment worker stopped")
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to stop enrollment worker: {e}")

        # 2.7. CRITICAL: Stop participant update thread BEFORE camera workers
        # This prevents processing of queued messages that would try to access cleaned-up command buffers
        try:
            if hasattr(self, 'participant_update_thread') and self.participant_update_thread.is_alive():
                print("[GUI] Stopping participant update thread...")
                if hasattr(self, 'participant_update_queue'):
                    # Send shutdown signal
                    self.participant_update_queue.put(None)
                # Wait for thread to finish processing
                self.participant_update_thread.join(timeout=2.0)
                if self.participant_update_thread.is_alive():
                    print("[SHUTDOWN WARNING] Participant update thread did not stop within timeout")
                else:
                    print("[GUI] Participant update thread stopped")
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to stop participant update thread: {e}")

        # 2.8. Clear any remaining messages from participant update queue
        try:
            if hasattr(self, 'participant_update_queue'):
                cleared_count = 0
                while not self.participant_update_queue.empty():
                    try:
                        self.participant_update_queue.get_nowait()
                        cleared_count += 1
                    except:
                        break
                if cleared_count > 0:
                    print(f"[GUI] Cleared {cleared_count} stale messages from participant update queue")
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to clear participant update queue: {e}")

        # 3. Stop all camera workers (NOW SAFE - no threads will try to access command buffers)
        try:
            print("[GUI] Stopping camera workers...")
            if hasattr(self, 'camera_worker_manager'):
                self.camera_worker_manager.stop_all()
                print("[GUI] All camera workers stopped")
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to stop camera workers: {e}")

        # Clear active camera processes and health status
        try:
            self.active_camera_procs.clear()
            self.camera_health_status.clear()
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to clear camera tracking: {e}")

        # 4. Close all participant mapping pipes (legacy cleanup)
        try:
            print("[GUI] Closing participant mapping pipes...")
            for cam_idx, pipe in list(self.participant_mapping_pipes.items()):
                try:
                    pipe.send({'type': 'shutdown'})
                    pipe.close()
                except:
                    pass
            self.participant_mapping_pipes.clear()
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to close mapping pipes: {e}")

        # Continue with the rest of the shutdown sequence
        try:
            self._complete_shutdown_sequence()
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed during complete shutdown sequence: {e}")
    
    def _create_transparent_overlay(self, width, height, opacity=0.75):
        """
        Create a semi-transparent black overlay image using PIL.

        Args:
            width: Width of the overlay in pixels
            height: Height of the overlay in pixels
            opacity: Opacity level (0.0 = fully transparent, 1.0 = opaque). Default: 0.75

        Returns:
            PIL.ImageTk.PhotoImage: PhotoImage suitable for canvas display
        """
        try:
            # Create RGBA image with semi-transparent black
            alpha_value = int(255 * opacity)
            overlay_img = Image.new('RGBA', (width, height), (0, 0, 0, alpha_value))

            # Convert to PhotoImage for Tkinter
            return ImageTk.PhotoImage(overlay_img)
        except Exception as e:
            logger.error(f"Failed to create transparent overlay: {e}")
            return None

    def _display_loading_overlay(self, canvas, message, canvas_idx=None, transparent=False, opacity=0.75):
        """
        Display loading message overlay with animated spinner on canvas during model initialization.

        Args:
            canvas: Tkinter canvas widget
            message: Status message to display
            canvas_idx: Index of canvas (for spinner animation tracking)
            transparent: If True, use semi-transparent overlay (requires video feed underneath)
            opacity: Opacity level for transparent overlay (0.0-1.0). Default: 0.75
        """
        try:
            # Delete any existing loading overlay
            canvas.delete('loading_overlay')

            # Clean up old PhotoImage reference if exists
            if canvas_idx is not None and canvas_idx in self._overlay_images:
                del self._overlay_images[canvas_idx]

            # Get canvas dimensions
            canvas.update_idletasks()
            width = canvas.winfo_width()
            height = canvas.winfo_height()

            if width <= 1 or height <= 1:
                return

            # Create overlay background (transparent or solid)
            if transparent:
                # Create semi-transparent overlay using PIL
                photo = self._create_transparent_overlay(width, height, opacity)
                if photo:
                    canvas.create_image(
                        0, 0,
                        image=photo,
                        anchor='nw',
                        tags='loading_overlay'
                    )
                    # Store reference to prevent garbage collection
                    if canvas_idx is not None:
                        self._overlay_images[canvas_idx] = photo
                else:
                    # Fallback to solid if PIL fails
                    canvas.create_rectangle(
                        0, 0, width, height,
                        fill='black',
                        tags='loading_overlay'
                    )
            else:
                # Solid black overlay (opaque)
                canvas.create_rectangle(
                    0, 0, width, height,
                    fill='black',
                    tags='loading_overlay'
                )

            # Create animated loading spinner (clock symbol)
            canvas.create_text(
                width // 2, height // 2 - 50,
                text="◷",  # Initial spinner character
                fill='white',
                font=('DejaVu Sans', 32),
                tags=('loading_overlay', 'loading_spinner')
            )

            # Create loading message text
            canvas.create_text(
                width // 2, height // 2,
                text=message,
                fill='white',
                font=('Arial', 14, 'bold'),
                tags=('loading_overlay', 'loading_message')
            )

            # FIX: Don't show redundant "Please wait..." - the main message is enough
            # The main message already conveys the current status dynamically

            # Start spinner animation if canvas_idx provided
            if canvas_idx is not None:
                # Stop any existing animation for this canvas
                if canvas_idx in self._canvas_spinners:
                    old_after_id = self._canvas_spinners[canvas_idx].get('after_id')
                    if old_after_id:
                        self.after_cancel(old_after_id)

                # Initialize spinner state
                self._canvas_spinners[canvas_idx] = {
                    'frame_count': 0,
                    'after_id': None
                }

                # Start animation
                self._animate_canvas_spinner(canvas_idx)

            # Keep overlay above any background images/frames
            canvas.tag_raise('loading_overlay')

            # FIX: Force canvas to repaint immediately to show overlay
            try:
                canvas.update_idletasks()
            except Exception as update_error:
                logger.warning(f"[OVERLAY] Failed to update canvas: {update_error}")

        except Exception as e:
            logger.error(f"Error displaying loading overlay: {e}")

    def _animate_canvas_spinner(self, canvas_idx):
        """
        Animate the loading spinner on a canvas.

        Args:
            canvas_idx: Index of the canvas in self.frames
        """
        # Check if spinner is still active
        if canvas_idx not in self._canvas_spinners:
            return

        # Check if canvas still exists
        if canvas_idx >= len(self.frames):
            # Canvas was removed, stop animation
            if canvas_idx in self._canvas_spinners:
                del self._canvas_spinners[canvas_idx]
            return

        info = self.frames[canvas_idx]
        canvas = info.get('canvas')

        if not canvas:
            # Canvas removed, stop animation
            if canvas_idx in self._canvas_spinners:
                del self._canvas_spinners[canvas_idx]
            return

        try:
            # Check if loading_spinner tag still exists
            spinner_items = canvas.find_withtag('loading_spinner')
            if not spinner_items:
                # Spinner overlay was removed, stop animation
                if canvas_idx in self._canvas_spinners:
                    del self._canvas_spinners[canvas_idx]
                return

            # Rotating spinner characters (clock faces)
            spinner_chars = ['◷', '◶', '◵', '◴']  # Unicode clock symbols (U+25F7-U+25F4)

            # Get current frame count
            frame_count = self._canvas_spinners[canvas_idx]['frame_count']
            char_index = frame_count % len(spinner_chars)

            # Update spinner character
            for item_id in spinner_items:
                canvas.itemconfig(item_id, text=spinner_chars[char_index])

            # Increment frame count
            self._canvas_spinners[canvas_idx]['frame_count'] = frame_count + 1

            # FIX: Force canvas to repaint after spinner update
            try:
                canvas.update_idletasks()
            except:
                pass  # Don't break animation on update error

            # Schedule next animation frame (200ms for smooth rotation)
            after_id = self.after(200, lambda: self._animate_canvas_spinner(canvas_idx))
            self._canvas_spinners[canvas_idx]['after_id'] = after_id

        except Exception as e:
            # On error, clean up spinner state
            if canvas_idx in self._canvas_spinners:
                del self._canvas_spinners[canvas_idx]
            logger.error(f"Error animating canvas spinner: {e}")

    def _update_initialization_status(self, camera_idx, message):
        """
        Update the initialization overlay message for a specific camera.

        Args:
            camera_idx: Camera index
            message: Status message to display
        """
        # GUARD: Prevent recreating overlay if camera already detected first face
        # This prevents the obsolete status polling path from interfering after camera is fully ready
        if hasattr(self, '_first_face_cameras') and camera_idx in self._first_face_cameras:
            logger.debug(f"[GUI] Skipping overlay update for camera {camera_idx} - first face already detected")
            return

        try:
            # Find the canvas for this camera
            canvas_idx = self.camera_to_frame_map.get(camera_idx)
            if canvas_idx is None:
                return

            if canvas_idx >= len(self.frames):
                return

            info = self.frames[canvas_idx]
            canvas = info.get('canvas')

            if not canvas:
                return

            # Check if overlay exists
            overlay_items = canvas.find_withtag('loading_overlay')
            if not overlay_items:
                # No overlay exists, create one (transparent to show live video underneath)
                self._display_loading_overlay(canvas, message, canvas_idx=canvas_idx, transparent=True, opacity=0.75)
                return

            # Overlay exists, update the text
            message_updated = False
            for item_id in overlay_items:
                # Check if this is the message text (has loading_message tag)
                item_tags = canvas.gettags(item_id)
                if 'loading_message' in item_tags:
                    # This is the message text - update it
                    try:
                        # DIAGNOSTIC: Log overlay text changes for validation
                        old_text = canvas.itemcget(item_id, 'text')
                        canvas.itemconfig(item_id, text=message)
                        new_text = canvas.itemcget(item_id, 'text')
                        logger.warning(f"[OVERLAY UPDATE] ⚠️ Camera {camera_idx}: Changed overlay text "
                                     f"from '{old_text}' to '{new_text}'")
                        message_updated = True
                        break
                    except Exception as e:
                        logger.error(f"[OVERLAY UPDATE ERROR] Camera {camera_idx}: {e}")
                        pass

            # Ensure overlay stays on top after updating
            canvas.tag_raise('loading_overlay')

            # FIX: Force canvas to repaint after text update
            if message_updated:
                try:
                    canvas.update_idletasks()
                except Exception as canvas_update_error:
                    logger.warning(f"[OVERLAY] Failed to update canvas after text change: {canvas_update_error}")
                logger.debug(f"[GUI] Updated overlay message for camera {camera_idx}: {message}")

        except Exception as e:
            logger.error(f"[GUI] Error updating initialization status: {e}")

    def _update_camera_health_ui(self):
        """Update camera health status in the UI"""
        current_time = time.time()

        for idx, info in enumerate(self.frames):
            if not info.get('combo'):
                continue

            sel = info['combo'].get()
            if not sel:
                continue

            try:
                cam_idx = int(sel.split(":", 1)[0])
            except (ValueError, IndexError):
                continue

            if cam_idx not in self.camera_health_status:
                continue

            health = self.camera_health_status[cam_idx]
            last_update = health.get('last_update', 0)
            time_since_update = current_time - last_update

            # Determine status and color
            # NEW: Check for initialization stage first (takes priority)
            if 'initialization_stage' in health:
                status_text = f"Status: {health['initialization_stage']}"
                color = health.get('state_color', '#2196F3')  # Default to blue

                # FIX: Don't recreate overlay here - it's managed by _update_initialization_status()
                # The overlay creation/update should ONLY happen via _update_initialization_status()
                # which is called when status messages arrive. This prevents race conditions where
                # the periodic _update_camera_health_ui() recreates the overlay with stale messages
                # before _update_initialization_status() can update it.
                # The overlay will be properly created/updated by the status handler in _handle_status_heartbeat()
                pass
            elif cam_idx not in self.active_camera_procs:
                status_text = "Status: Idle"
                color = 'gray'
            elif time_since_update > 5.0:  # No update in 5 seconds
                status_text = "Status: Disconnected"
                color = 'red'
            elif health.get('status') == 'error':
                status_text = f"Status: Error - {health.get('data', {}).get('message', 'Unknown')}"
                color = 'red'
            elif health.get('status') == 'heartbeat':
                # Show performance info
                processing_time = health.get('processing_time_ms', 0)
                n_faces = health.get('n_faces', 0)
                frame_id = health.get('frame_id', 0)
                
                if processing_time > 0:
                    status_text = f"Status: Active ({processing_time:.1f}ms, {n_faces} face{'s' if n_faces != 1 else ''}, #{frame_id})"
                else:
                    status_text = "Status: Active"
                color = 'green'
            else:
                status_text = f"Status: {health.get('status', 'Unknown')}"
                color = 'orange'
            
            # Update UI
            info['health_label'].config(text=status_text, foreground=color)
    
    def _complete_shutdown_sequence(self):
        """Complete the shutdown sequence - protected with try-except for robustness"""
        # 5. Send shutdown to LSL helper
        try:
            if hasattr(self, 'lsl_command_queue'):
                print("[GUI] Shutting down LSL helper...")
                try:
                    for _ in range(3):
                        self.lsl_command_queue.put({'type': 'stop'})
                except:
                    pass
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to shutdown LSL helper: {e}")

        # 6. Clear all queues - FIXED: Handle both dict and list cases
        try:
            print("[GUI] Clearing all queues...")
            queues_to_clear = [
                ('participant_update_queue', self.participant_update_queue),
                ('lsl_command_queue', getattr(self, 'lsl_command_queue', None)),
                ('lsl_data_queue', getattr(self, 'lsl_data_queue', None))
            ]

            for name, q in queues_to_clear:
                if q:
                    try:
                        q.put(None)
                        while not q.empty():
                            try:
                                q.get_nowait()
                            except:
                                break
                        print(f"[GUI] Cleared {name}")
                    except:
                        pass
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to clear queues: {e}")

        # 7. Terminate worker processes
        try:
            print("[GUI] Terminating worker processes...")
            for idx, info in enumerate(self.frames):
                if info.get('proc'):
                    proc = info['proc']
                    try:
                        if proc.is_alive():
                            print(f"[GUI] Terminating worker {idx}...")
                            proc.terminate()
                            proc.join(timeout=1.0)
                            if proc.is_alive():
                                print(f"[GUI] Force killing worker {idx}...")
                                proc.kill()
                                proc.join(timeout=0.5)
                    except Exception as e:
                        print(f"[SHUTDOWN ERROR] Failed to terminate worker {idx}: {e}")
                    info['proc'] = None
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed during worker process termination: {e}")

        # 8. Terminate LSL helper process
        try:
            if hasattr(self, 'lsl_helper_proc') and self.lsl_helper_proc:
                if self.lsl_helper_proc.is_alive():
                    print("[GUI] Terminating LSL helper...")
                    self.lsl_helper_proc.terminate()
                    self.lsl_helper_proc.join(timeout=1.0)
                    if self.lsl_helper_proc.is_alive():
                        print("[GUI] Force killing LSL helper...")
                        self.lsl_helper_proc.kill()
                        self.lsl_helper_proc.join(timeout=0.5)
                self.lsl_helper_proc = None
                # CRITICAL: Also clean up LSL queues to force reinitialization on next start
                self.lsl_data_queue = None
                self.lsl_command_queue = None
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to terminate LSL helper: {e}")

        # 9. Clean up shared memory
        try:
            print("[GUI] Cleaning up shared memory...")
            # Score buffers
            for info in self.frames:
                if info.get('score_buffer'):
                    try:
                        info['score_buffer'].cleanup()
                    except:
                        pass
                    info['score_buffer'] = None
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to cleanup score buffers: {e}")

        # Correlation buffer - CRITICAL: Delete numpy view before closing
        try:
            if hasattr(self, 'corr_array'):
                try:
                    # Delete the numpy array view to prevent access violations
                    del self.corr_array
                    logger.info("[GUI] Deleted correlation array numpy view")
                except Exception as e:
                    logger.warning(f"[GUI] Error deleting corr_array: {e}")

            if hasattr(self, 'correlation_buffer'):
                try:
                    self.correlation_buffer.close()
                    self.correlation_buffer.unlink()
                except:
                    pass
                self.correlation_buffer = None
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to cleanup correlation buffer: {e}")

        # 11. Terminate active camera processes
        try:
            print("[GUI] Terminating active camera processes...")
            for cam_idx, proc in list(self.active_camera_procs.items()):
                try:
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=1.0)
                        if proc.is_alive():
                            proc.kill()
                            proc.join(timeout=0.5)
                except:
                    pass
            self.active_camera_procs.clear()
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to terminate camera processes: {e}")

        # 12. Reset state variables
        try:
            self.streaming = False
            self.recording_active = False
            self.immediate_recording = False
            self.audio_recording_active = False
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed to reset state variables: {e}")

        # 13. Final cleanup - ensure all child processes are gone
        try:
            import psutil
            try:
                current_process = psutil.Process()
                children = current_process.children(recursive=True)
                if children:
                    print(f"[GUI] Found {len(children)} remaining child processes, terminating...")
                    for child in children:
                        try:
                            child.terminate()
                        except:
                            pass
                    # Give them time to terminate
                    gone, alive = psutil.wait_procs(children, timeout=1)
                    # Force kill any remaining
                    for p in alive:
                        try:
                            print(f"[GUI] Force killing process {p.pid}")
                            p.kill()
                        except:
                            pass
            except ImportError:
                print("[GUI] psutil not available, skipping child process cleanup")
            except Exception as e:
                print(f"[GUI] Error during child process cleanup: {e}")
        except Exception as e:
            print(f"[SHUTDOWN ERROR] Failed during final cleanup: {e}")

        print("[GUI] === SHUTDOWN COMPLETE ===\n")


    def _stop_lsl_streaming(self):
        """Stop LSL streaming only, keeping all processes alive for quick restart"""
        print("\n[GUI] === STOPPING LSL STREAMING (processes remain active) ===")

        # 1. Set streaming flag to false
        self.streaming = False

        # 2. Stop LSL streaming in GUI processing worker
        if hasattr(self, 'processing_control_queue'):
            try:
                self.processing_control_queue.put({
                    'type': 'stop_lsl_streaming'
                })
                print("[GUI] Sent stop_lsl_streaming command to GUI processing worker")
            except Exception as e:
                print(f"[GUI WARNING] Failed to stop LSL streaming in worker: {e}")

        # 3. Stop streaming in LSL helper (stop accepting new data)
        if hasattr(self, 'lsl_command_queue'):
            try:
                self.lsl_command_queue.put({
                    'type': 'streaming_stopped'
                })
                print("[GUI] Sent streaming_stopped command to LSL helper")
            except Exception as e:
                print(f"[GUI WARNING] Failed to stop streaming in LSL helper: {e}")

        # Co-modulation stream removed

        # 4.5. Stop Bluetooth devices
        if hasattr(self, 'bluetooth_manager'):
            try:
                self.bluetooth_manager.stop_all()
                print("[GUI] Stopped all Bluetooth devices")
            except Exception as e:
                print(f"[GUI WARNING] Failed to stop Bluetooth devices: {e}")

        # 5. Destroy all LSL outlets for clean restart
        if hasattr(self, 'lsl_command_queue'):
            try:
                self.lsl_command_queue.put({
                    'type': 'close_all_streams'
                })
                print("[GUI] Sent close_all_streams command to LSL helper")
            except Exception as e:
                print(f"[GUI WARNING] Failed to close streams: {e}")

        # 6. Update UI elements
        for info in self.frames:
            if info.get('meta_label'):
                info['meta_label'].config(text="LSL → OFF")

        # 7. Stop performance monitoring (optional)
        # The update_performance_stats will check self.streaming flag

        print("[GUI] LSL streaming stopped and outlets destroyed (ready for fresh restart)\n")


    def stop_stream(self):
        """Stop LSL data streaming and finalize video recordings"""
        print("[GUI] stop_stream() called")
        if not self.streaming:
            print("[GUI] Not currently streaming, nothing to stop")
            return

        print("\n[GUI] === STOPPING DATA STREAM ===")

        # Stop and finalize video recording if active
        if self.recording_active:
            print("[GUI] Finalizing video recordings...")
            self._stop_video_recording()
            print("[GUI] Video recordings finalized and saved")

        # Stop LSL streaming (lightweight operation)
        self._stop_lsl_streaming()

        # Update button states
        if self.record_video.get():
            active_workers = [info for info in self.frames if info.get('proc') and info['proc'].is_alive()]
            if active_workers:
                self.record_now_btn.config(state='normal')
        self.stop_record_btn.config(state='disabled')

        print("[GUI] Data stream stopped and recordings finalized (ready to restart)\n")


    def reset(self):
        """Complete reset with comprehensive cleanup"""
        print("XXXXXX DEBUG: reset() called! XXXXXX")
        print("\n[GUI] === FULL RESET INITIATED ===")
        
        # 1. Comprehensive shutdown first
        print("XXXXXX DEBUG: reset() calling _shutdown_all_processes XXXXXX")
        self._shutdown_all_processes()

        # 1.5. Clean up shared memory infrastructure
        print("[GUI] Cleaning up shared memory buffers...")

        # Clean up existing correlation buffer
        if hasattr(self, 'correlation_buffer') and self.correlation_buffer:
            try:
                self.correlation_buffer.close()
                self.correlation_buffer.unlink()
                print("[GUI] Cleaned up correlation buffer")
            except Exception as e:
                print(f"[GUI] Error cleaning up correlation buffer: {e}")
            self.correlation_buffer = None
            self.corr_array = None

        # Clean up all BufferCoordinator buffers
        if hasattr(self, 'buffer_coordinator') and self.buffer_coordinator:
            try:
                self.buffer_coordinator.cleanup_all_buffers()
                print("[GUI] Cleaned up all BufferCoordinator buffers")
            except Exception as e:
                print(f"[GUI] Error cleaning up BufferCoordinator: {e}")
            # Don't set to None yet - we'll recreate it below

        # 2. Participant update thread already stopped in _shutdown_all_processes()
        # (No need to stop again - it was stopped before camera workers shutdown)

        # 3. Reset global participant manager
        if hasattr(self, 'global_participant_manager'):
            self.global_participant_manager.reset()
        
        # 4. Clear all data structures
        self.frames.clear()
        self.worker_procs.clear()
        self.preview_queues.clear()
        self.score_queues.clear()
        self.recording_queues.clear()
        self.participant_mapping_pipes.clear()
        self.video_recorders.clear()
        self.audio_recorders.clear()

        # 4.5. Clear actual buffer names from previous session
        if hasattr(self, 'actual_buffer_names'):
            self.actual_buffer_names.clear()
            print("[GUI] Cleared actual_buffer_names")

        # Clear display-related flags
        self._optimized_display_active = False
        print("[GUI] Reset display flags")

        # 5. Clear drawing manager state
        for idx in range(len(self.frames)):
            self.drawingmanager .cleanup_canvas(idx)
        
        # Bar plot removed (was for co-modulation)
        
        # 7. Reset all flags
        self.streaming = False
        self.recording_active = False
        self.immediate_recording = False
        self.audio_recording_active = False
        self.monitoring_active = False
        self.latest_correlation = None
        
        # 8. Update UI states
        self.record_status.config(text="Not recording", foreground='gray')
        self.record_now_btn.config(
            text="Start Video Recording", 
            state='normal' if self.record_video.get() else 'disabled'
        )
        self.stop_record_btn.config(state='disabled')
        self.start_audio_btn.config(
            state='normal' if self.audio_enabled.get() and self.audio_mode.get() == "standalone" else 'disabled'
        )
        self.stop_audio_btn.config(state='disabled')
        self.audio_status.config(text=self.audio_status.cget("text").replace(" (recording)", ""))
        
        # 9. Recreate essential structures
        self.participant_update_queue = MPQueue()

        # 9.5 Recreate BufferCoordinator and related structures
        print("[GUI] Recreating BufferCoordinator and buffers...")
        max_cameras = self.config.get('system.max_cameras', 10)  # Single source of truth for max camera count

        # Recreate BufferCoordinator with fresh state
        self.buffer_coordinator = BufferCoordinator(camera_count=max_cameras, config=self.config.config)
        print("[GUI] BufferCoordinator recreated")

        # Recreate recovery buffers
        self.recovery_buffer_names = {}
        for i in range(max_cameras):
            try:
                buffer_name = self.buffer_coordinator.create_recovery_buffer(i)
                self.recovery_buffer_names[i] = buffer_name
                print(f"[GUI] Created recovery buffer for camera {i}: {buffer_name}")
            except Exception as e:
                print(f"[ERROR] Failed to create recovery buffer for camera {i}: {e}")

        # Recreate CameraWorkerManager with fresh BufferCoordinator
        self.camera_worker_manager = CameraWorkerManager(
            config=self.config.config,
            participant_update_queue=self.participant_update_queue,
            buffer_coordinator=self.buffer_coordinator,
            gui_buffer_manager=None  # Will be set later if needed
        )
        print("[GUI] CameraWorkerManager recreated")

        # 10. Restart participant update processor (using the new method for consistency)
        print("[GUI] Restarting participant update thread after reset...")
        self._start_participant_update_thread()

        # 11. Note: Correlation buffer will be recreated in _initialize_lsl_helper() called by build_frames()
        # Don't create it here to avoid duplication

        # 12. Rebuild frames
        self.build_frames()
        
        print("[GUI] === RESET COMPLETE ===\n")
        
    def save_current_settings(self):
        """Save all current GUI settings to configuration file"""
        # Save all current settings
        self.config.set('camera_settings.target_fps', self.desired_fps.get())
        self.config.set('camera_settings.resolution', self.res_choice.get())
        self.config.set('startup_mode.participant_count', self.participant_count.get())
        self.config.set('startup_mode.camera_count', self.camera_count.get())
        self.config.set('startup_mode.enable_mesh', self.enable_mesh.get())
        self.config.set('video_recording.enabled', self.record_video.get())
        self.config.set('video_recording.save_directory', self.save_dir.get())
        self.config.set('video_recording.filename_template', self.filename_template.get())
        self.config.set('audio_recording.enabled', self.audio_enabled.get())
        self.config.set('audio_recording.standalone_audio', self.audio_mode.get() == "standalone")
        self.config.set('audio_recording.audio_with_video', self.audio_mode.get() == "with_video")
        self.config.set('audio_devices', self.audio_device_assignments)
        
        # Show confirmation
        messagebox.showinfo("Settings Saved", "Current settings have been saved to configuration file")

if __name__ == '__main__':
    # Enable fault handler to catch segfaults
    import faulthandler
    import signal
    faulthandler.enable()

    # NOTE: multiprocessing.set_start_method('spawn') moved to module level (line 40)
    # This ensures semaphores are created in spawn mode BEFORE BufferCoordinator __init__

    # CRITICAL: Initialize ONNX CUDA context BEFORE GUI creation
    # This must happen before YouQuantiPyGUI() because something in GUI.__init__()
    # breaks CUDA accessibility (cudaGetDeviceCount returns error 100).
    # By initializing ONNX first, it captures the working CUDA context.
    print("\n[GUI] Initializing ONNX CUDA context for GPU-accelerated face recognition...")
    print("[GUI] ⚠️  NOTE: Initializing BEFORE GUI creation to avoid CUDA context corruption")

    print(f"[GUI]   LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')[:150]}...")
    print(f"[GUI]   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    print(f"[GUI]   CUDA_PATH: {os.environ.get('CUDA_PATH', 'NOT SET')}")
    print(f"[GUI]   YOUQUANTIPY_CUDA_CONFIGURED: {os.environ.get('YOUQUANTIPY_CUDA_CONFIGURED', 'NOT SET')}")

    from core.gui_cuda_init import initialize_onnx_cuda_context

    # Use ArcFace model for initialization (same model used by face recognition)
    onnx_cuda_status = initialize_onnx_cuda_context(
        model_path="/home/canoz/Projects/youquantipy_mediapipe/models/arcface.onnx"
    )

    if onnx_cuda_status['success']:
        print(f"[GUI] ✅ ONNX CUDA initialized successfully!")
        print(f"[GUI]   Provider: {onnx_cuda_status['provider']}")
        print(f"[GUI]   Latency: {onnx_cuda_status['latency_ms']:.1f}ms per face")
        print(f"[GUI]   Face recognition will use GPU acceleration (~8-20ms per face)")
    else:
        print(f"[GUI] ⚠️  ONNX CUDA initialization failed")
        if onnx_cuda_status.get('error'):
            print(f"[GUI]   Error: {onnx_cuda_status['error']}")
        print(f"[GUI]   Face recognition will use CPU (~200ms per face)")
    print()

    # Create GUI instance
    # NOTE: Something in __init__() may break CUDA, but ONNX already has its context
    # NOTE: X server and window manager warmup now handled in run.sh for better reliability
    print("[GUI] Creating GUI instance...")
    app = YouQuantiPyGUI()
    print("[GUI] GUI instance created successfully")

    # Register atexit handler as last-resort cleanup (catches kill -9, crashes, etc.)
    def emergency_exit_cleanup():
        """Last-resort cleanup called by atexit when program terminates."""
        print("\n[ATEXIT] Emergency cleanup triggered...")
        try:
            if hasattr(app, '_shutdown_all_processes'):
                app._shutdown_all_processes()
        except Exception as e:
            print(f"[ATEXIT] Error during emergency cleanup: {e}")

    atexit.register(emergency_exit_cleanup)

    # Signal handler for graceful shutdown on Ctrl+C
    def signal_handler(sig, frame):
        print(f"\n[SIGNAL] Received signal {sig}, initiating graceful shutdown...")
        try:
            # Call the comprehensive shutdown sequence
            if hasattr(app, '_shutdown_all_processes'):
                app._shutdown_all_processes()
            app.destroy()
        except Exception as e:
            print(f"[SIGNAL] Error during shutdown: {e}")
        finally:
            import sys
            sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command

    # Window is already visible (not hidden during init)
    # Just ensure it's properly positioned and focused
    print("[DEBUG] Finalizing window setup...")
    sys.stdout.flush()

    try:
        # Ensure widgets are laid out
        app.update_idletasks()

        # Platform-specific focus handling
        platform = PlatformContext.detect(app.config.config)
        if platform['display_mode'] == 'wslg':
            # WSLg: Ensure window has focus (but don't use -topmost which interferes with maximize)
            app.lift()                 # Bring to front
            app.focus_force()          # Grab keyboard focus
            print("[DEBUG] WSLg: Window focused")
            sys.stdout.flush()
        else:
            # Standard X11/Wayland: Just ensure it's raised
            app.lift()
            print(f"[DEBUG] Display mode: {platform['display_mode']} - window raised")
            sys.stdout.flush()

        print("[DEBUG] ✅ Window setup complete")
        sys.stdout.flush()

    except Exception as e:
        print(f"[DEBUG] ⚠️ Warning during window setup: {e}")
        sys.stdout.flush()

    print("[DEBUG] About to call app.mainloop()...")
    sys.stdout.flush()

    # Schedule build_frames() AFTER mainloop starts to avoid threading issues
    # with background camera discovery calling self.after() before event loop is ready
    app.after(100, app.build_frames)

    app.mainloop()
    print("[DEBUG] app.mainloop() returned (window closed)")
    sys.stdout.flush()
    
