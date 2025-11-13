"""
V4L2 Camera Source Implementation

Handles camera capture using Video4Linux2 backend (Linux).
Extracted and refactored from camera_worker_enhanced.py for modularity.

Platform-aware: Adapts behavior for WSL2 vs native Linux.
"""

import cv2
import numpy as np
import subprocess
import time
import logging
from typing import Tuple, Optional, Dict
from .base import CameraSource
from core.platform_detection import PlatformContext

logger = logging.getLogger('V4L2CameraSource')


class V4L2CameraSource(CameraSource):
    """
    V4L2-based camera source for Linux/WSL2.

    Features:
    - WSL2-optimized MJPEG format handling
    - Auto-detection of camera type (regular/IR)
    - Manual exposure control support
    - Automatic process cleanup for blocked devices
    - Optimal buffer sizing for FPS
    """

    def __init__(self, config: Dict, camera_index: int):
        """
        Initialize V4L2 camera source.

        Args:
            config: Camera configuration dictionary
            camera_index: Camera device index
        """
        self.config = config
        self.camera_index = camera_index
        self.cap = None
        self.backend_name = "V4L2"
        self.actual_resolution = None
        self.actual_fps = None

        # Parse configuration
        camera_key = f"camera_{camera_index}"
        camera_settings = config.get('camera_settings', {}).get(camera_key, {})

        self.device_path = camera_settings.get('device_path', f'/dev/video{camera_index}')
        self.target_width = int(camera_settings.get('width', 1280))
        self.target_height = int(camera_settings.get('height', 720))
        self.target_fps = int(camera_settings.get('fps', 30))
        self.buffersize = int(camera_settings.get('buffersize', 2))  # 2 is optimal for WSL2
        self.camera_type = camera_settings.get('camera_type', 'auto')
        self.retry_count = int(camera_settings.get('retry_count', 3))
        self.exposure_control = camera_settings.get('exposure_control', 'auto')
        self.camera_settings = camera_settings

        # NEW: Mapped pinned memory optimization
        self._use_mapped_memory = camera_settings.get('use_mapped_memory', True)
        self._mapped_buffer = None
        self._frame_view = None

        # LIFO frame capture: Drain kernel buffer to get latest frame (eliminates startup lag)
        # When True, discards old buffered frames and returns the MOST RECENT frame
        # When False, uses default FIFO behavior (returns oldest buffered frame)
        self.drain_kernel_buffer = camera_settings.get('drain_kernel_buffer', True)

    def open(self) -> bool:
        """
        Open the V4L2 camera device and configure settings.

        Platform-aware: Adapts MJPEG and buffer size settings for WSL2 vs native Linux.

        Returns:
            bool: True if opened successfully, False otherwise
        """
        logger.info(f"Initializing camera {self.camera_index} at {self.device_path}")

        # Detect platform mode for conditional tuning
        platform = PlatformContext.detect(self.config)
        is_wsl2 = (platform['mode'] == 'wsl2')

        logger.info(f"Platform mode: {platform['mode']}")

        # Auto-detect camera type if needed
        if self.camera_type == 'auto':
            self.camera_type = self._detect_camera_type()

        # Determine if we should use MJPEG
        # WSL2: MJPEG required for regular cameras (YUYV @ 720p = ~10fps max)
        # Native Linux: Test and determine optimal format (may work with YUYV)
        if is_wsl2:
            use_mjpeg = (self.camera_type == 'regular')
            logger.info(f"WSL2 mode: MJPEG enforcement = {use_mjpeg} (camera type: {self.camera_type})")
        else:
            # Native Linux: MJPEG may still be beneficial for bandwidth, but not strictly required
            use_mjpeg = (self.camera_type == 'regular')
            logger.info(f"Native Linux mode: MJPEG suggested = {use_mjpeg} (camera type: {self.camera_type})")

        # Retry loop with cleanup
        for attempt in range(self.retry_count):
            if attempt > 0:
                logger.info(f"Retry attempt {attempt + 1}/{self.retry_count}")
                self._cleanup_camera_processes()
                time.sleep(1)

            try:
                # Open camera with V4L2 backend
                logger.info(f"  Opening {self.device_path} with V4L2 backend...")
                self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)

                if not self.cap.isOpened():
                    logger.error(f"  Failed to open {self.device_path}")
                    if self.cap:
                        self.cap.release()
                    continue

                # CRITICAL: Set MJPEG format FIRST for regular cameras in WSL2
                # This must be done before setting resolution/FPS
                if use_mjpeg:
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                    logger.info("  Set FOURCC=MJPG for regular camera")

                # Set resolution
                logger.info(f"  Requesting resolution: {self.target_width}x{self.target_height} @ {self.target_fps} FPS")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

                # Set FPS
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

                # Set buffer size (platform-aware tuning)
                # WSL2: Buffer size 2 optimal (size 1 = ~15fps bug, size 2+ = ~29fps)
                # Native Linux: May differ, test for optimal settings
                if is_wsl2:
                    optimal_buffersize = max(self.buffersize, 2)  # Ensure at least 2 for WSL2
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, optimal_buffersize)
                    logger.info(f"  Set buffer size={optimal_buffersize} (WSL2 optimization: prevents 15fps bug)")
                else:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffersize)
                    logger.info(f"  Set buffer size={self.buffersize} (Native Linux)")

                # CRITICAL: Check what resolution was actually negotiated BEFORE test frame
                # This helps diagnose if camera doesn't support requested resolution
                negotiated_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                negotiated_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                negotiated_fps = self.cap.get(cv2.CAP_PROP_FPS)
                logger.info(f"  Camera negotiated: {negotiated_w}x{negotiated_h} @ {negotiated_fps} FPS")

                if (negotiated_w != self.target_width) or (negotiated_h != self.target_height):
                    logger.warning(f"  ⚠️  Resolution mismatch! Requested {self.target_width}x{self.target_height}, got {negotiated_w}x{negotiated_h}")
                    logger.warning(f"      Camera may not support requested resolution - attempting to use negotiated resolution")

                # Verify with a test frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.error("  Failed to capture test frame")
                    self.cap.release()
                    continue

                # Success! Get actual settings
                actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

                self.actual_resolution = (actual_w, actual_h)

                # Verify FOURCC (actual format negotiated)
                actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = "".join([chr((actual_fourcc >> 8*i) & 0xFF) for i in range(4)])

                logger.info(f"Camera {self.camera_index} initialized successfully:")
                logger.info(f"  Device: {self.device_path}")
                logger.info(f"  Type: {self.camera_type}")
                logger.info(f"  Resolution: {actual_w}x{actual_h}")
                logger.info(f"  FPS: {self.actual_fps}")
                logger.info(f"  Format: {fourcc_str} (requested: {'MJPEG' if use_mjpeg else 'Default'})")
                logger.info(f"  Backend: V4L2")

                # CRITICAL: Verify MJPEG was actually set
                if use_mjpeg and fourcc_str != 'MJPG':
                    logger.warning(f"  ⚠️  FOURCC mismatch! Camera using {fourcc_str} instead of MJPEG")
                    logger.warning(f"      This will likely result in low FPS (YUYV @ 720p = ~10fps max)")

                # Apply manual exposure control if configured
                if self.exposure_control == 'manual':
                    self._apply_manual_exposure()

                return True

            except Exception as e:
                logger.error(f"  Error opening camera: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None

        logger.error(f"Failed to open camera {self.camera_index} after {self.retry_count} attempts")
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the V4L2 camera with optimized mapped pinned memory.

        When mapped_memory is enabled, frames are copied to GPU-accessible memory
        for faster MediaPipe upload (4× speedup vs regular memory).

        When drain_kernel_buffer is enabled (default), uses LIFO behavior to get
        the MOST RECENT frame by draining old buffered frames. This eliminates
        startup lag and ensures true real-time display.

        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - success: True if frame read successfully
                - frame: BGR numpy array (H, W, 3) in mapped pinned memory or regular memory
        """
        if not self.cap or not self.cap.isOpened():
            return False, None

        # LIFO OPTIMIZATION: Drain kernel buffer to get latest frame (eliminates startup lag)
        # This changes behavior from FIFO (oldest frame) to LIFO (newest frame)
        if self.drain_kernel_buffer:
            # Get kernel buffer size (typically 2 in WSL2)
            buffer_size = int(self.cap.get(cv2.CAP_PROP_BUFFERSIZE))

            # Drain old frames from kernel buffer (keep only latest)
            # grab() is fast - no MJPEG decode, just discards old frames
            for _ in range(max(0, buffer_size - 1)):
                self.cap.grab()

            # Now retrieve the most recent frame (decodes MJPEG)
            ret = self.cap.grab()  # Grab latest frame
            if not ret:
                return False, None
            ret, frame = self.cap.retrieve()  # Decode it
            if not ret or frame is None:
                return False, None
        else:
            # Standard FIFO behavior: cap.read() returns oldest buffered frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False, None

        # Optimize: Copy to mapped pinned memory if available
        if self._use_mapped_memory:
            # Initialize mapped buffer on first frame
            if self._mapped_buffer is None:
                self._init_mapped_memory(frame.shape)

            # Copy to mapped pinned memory (fast DMA transfer)
            if self._mapped_buffer is not None and self._frame_view is not None:
                self._frame_view[:] = frame
                return True, self._frame_view

        # Fallback: Return regular memory
        return True, frame

    def _init_mapped_memory(self, shape):
        """Initialize mapped pinned memory buffer for frame storage."""
        try:
            from core.buffer_management.pinned_memory_ctypes import get_allocator
            allocator = get_allocator()

            if not allocator.available:
                logger.info(f"  ℹ️  Camera {self.camera_index}: Mapped memory not available, using regular memory")
                self._use_mapped_memory = False
                return

            # Check if mapped memory is supported
            if not hasattr(allocator, 'allocate_mapped'):
                logger.info(f"  ℹ️  Camera {self.camera_index}: Mapped memory API not available")
                self._use_mapped_memory = False
                return

            h, w, c = shape
            frame_size = h * w * c

            # Allocate mapped pinned memory
            self._mapped_buffer = allocator.allocate_mapped(frame_size, dtype=np.uint8)
            self._frame_view = self._mapped_buffer.as_array().reshape((h, w, c))

            logger.info(f"  ✅ Camera {self.camera_index} mapped pinned memory allocated: {frame_size/1024/1024:.1f} MB")
            logger.info(f"     CPU ptr: {hex(self._mapped_buffer.get_cpu_pointer())}")
            logger.info(f"     GPU ptr: {hex(self._mapped_buffer.get_gpu_pointer())}")
            logger.info(f"     Memory is accessible from BOTH CPU and GPU!")
            logger.info(f"     Expected MediaPipe upload speedup: ~4×")

        except Exception as e:
            logger.warning(f"  ⚠️  Camera {self.camera_index}: Failed to allocate mapped memory: {e}")
            logger.warning(f"      Falling back to regular memory")
            self._use_mapped_memory = False
            self._mapped_buffer = None
            self._frame_view = None

    def release(self):
        """Release the V4L2 camera device."""
        if self.cap:
            self.cap.release()
            self.cap = None
            logger.info(f"Camera {self.camera_index} released")

    def get_resolution(self) -> Tuple[int, int]:
        """Get the actual camera resolution."""
        if self.actual_resolution:
            return self.actual_resolution
        return (self.target_width, self.target_height)

    def get_fps(self) -> float:
        """Get the actual camera FPS."""
        if self.actual_fps:
            return self.actual_fps
        return float(self.target_fps)

    def is_opened(self) -> bool:
        """Check if the camera is currently opened."""
        return self.cap is not None and self.cap.isOpened()

    def get_backend_name(self) -> str:
        """Get the backend name."""
        return self.backend_name

    def get_camera_name(self) -> str:
        """
        Get the camera friendly name.

        For V4L2 cameras, returns the device path as the name.

        Returns:
            str: Camera device path (e.g., "/dev/video0")
        """
        return self.device_path

    def _detect_camera_type(self) -> str:
        """
        Auto-detect if camera is IR or regular based on capabilities.

        Returns:
            "regular", "ir", or "unknown"
        """
        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', self.device_path, '--list-formats-ext'],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                output = result.stdout
                # IR cameras typically support GREY format
                if 'GREY' in output and '640x360' in output:
                    logger.info(f"{self.device_path} detected as IR camera")
                    return "ir"
                else:
                    logger.info(f"{self.device_path} detected as regular camera")
                    return "regular"
        except Exception as e:
            logger.warning(f"Camera type detection failed: {e}, assuming regular")

        return "regular"

    def _cleanup_camera_processes(self) -> int:
        """
        Kill any processes currently accessing the camera device.
        Critical for WSL2 where zombie processes can block camera access.

        Returns:
            Number of processes killed
        """
        killed_count = 0

        try:
            # Use fuser to find processes using the device
            result = subprocess.run(
                ['fuser', self.device_path],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split()
                logger.info(f"Cleaning up {len(pids)} process(es) blocking {self.device_path}: {pids}")

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

    def _apply_manual_exposure(self):
        """
        Apply manual exposure control using v4l2-ctl to prevent auto-exposure from limiting FPS.
        """
        logger.info(f"Applying manual exposure control to {self.device_path}...")

        # Get exposure settings from config
        target_shutter_fps = self.camera_settings.get('target_shutter_fps', self.target_fps)
        gain = self.camera_settings.get('gain', 0)

        try:
            # Step 1: Set exposure mode to manual (1 = manual on most UVC cameras)
            result = subprocess.run(
                ['v4l2-ctl', '-d', self.device_path, '--set-ctrl=exposure_auto=1'],
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
                    ['v4l2-ctl', '-d', self.device_path, f'--set-ctrl=exposure_absolute={exposure_absolute}'],
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
                    ['v4l2-ctl', '-d', self.device_path, f'--set-ctrl=gain={gain}'],
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
                    ['v4l2-ctl', '-d', self.device_path, '--get-ctrl=exposure_auto'],
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
