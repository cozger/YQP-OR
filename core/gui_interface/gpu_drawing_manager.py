"""
GPU-Accelerated Drawing Manager for YouQuantiPy
Provides CUDA-accelerated frame processing and overlay rendering for maximum GUI performance.

Author: YouQuantiPy Team
Performance Target: 100-200 FPS (vs 30-60 FPS CPU baseline)
"""

# CRITICAL: Initialize CUDA environment BEFORE importing cv2
import os
import sys
# Add parent directory to path to import cuda_init
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda_init import setup_cuda_environment, initialize_cuda_context
setup_cuda_environment()

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any
import time

logger = logging.getLogger(__name__)


class GPUDrawingManager:
    """
    Manages GPU-accelerated drawing operations for camera feeds and overlays.
    Uses OpenCV CUDA module for hardware acceleration on NVIDIA GPUs.
    """

    def __init__(self, enable_gpu: bool = True, profile_performance: bool = False):
        """
        Initialize GPU drawing manager with CUDA context.

        Args:
            enable_gpu: Enable GPU acceleration (falls back to CPU if unavailable)
            profile_performance: Enable detailed performance profiling
        """
        self.gpu_available = False
        self.gpu_enabled = enable_gpu
        self.profile_performance = profile_performance

        # Performance tracking
        self.perf_stats = {
            'upload_time': [],
            'resize_time': [],
            'cvt_color_time': [],
            'overlay_time': [],
            'download_time': [],
            'total_time': []
        }

        # GPU memory pools (pre-allocated buffers for common resolutions)
        self.gpu_frame_cache = {}  # {(width, height, channels): [GpuMat, GpuMat, ...]}
        self.cache_size_per_resolution = 4  # Number of buffers to cache per resolution

        # CUDA stream for asynchronous operations
        self.cuda_stream = None

        # Initialize GPU context
        self._initialize_gpu()

    def _initialize_gpu(self):
        """Initialize CUDA context and verify GPU availability."""
        logger.info("[GPU INIT] Starting GPU initialization...")
        logger.info(f"[GPU INIT] Config setting gpu_enabled={self.gpu_enabled}")

        if not self.gpu_enabled:
            logger.info("[GPU] GPU acceleration disabled via config")
            return

        # Ensure CUDA environment is properly configured
        logger.info("[GPU INIT] Ensuring CUDA environment configuration...")
        from cuda_init import ensure_cuda_initialized
        cuda_status = ensure_cuda_initialized()
        logger.info(f"[GPU INIT] CUDA environment status: OpenCV={cuda_status.get('opencv_cuda', False)}, "
                   f"Devices={cuda_status.get('opencv_devices', 0)}")

        try:
            # Step 1: Check if cv2.cuda module exists
            logger.info("[GPU INIT] Step 1: Checking cv2.cuda module availability...")
            if not hasattr(cv2, 'cuda'):
                logger.warning("[GPU] OpenCV not compiled with CUDA support (cv2.cuda module missing) - falling back to CPU")
                self.gpu_available = False
                return
            logger.info("[GPU INIT] ✓ cv2.cuda module found")

            # Step 2: Check if getCudaEnabledDeviceCount is callable
            logger.info("[GPU INIT] Step 2: Checking getCudaEnabledDeviceCount function...")
            if not callable(getattr(cv2.cuda, 'getCudaEnabledDeviceCount', None)):
                logger.warning("[GPU] OpenCV CUDA module incomplete (getCudaEnabledDeviceCount missing) - falling back to CPU")
                self.gpu_available = False
                return
            logger.info("[GPU INIT] ✓ getCudaEnabledDeviceCount function found")

            # Step 3: Check for available CUDA devices
            logger.info("[GPU INIT] Step 3: Querying CUDA device count...")
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            logger.info(f"[GPU INIT] Found {device_count} CUDA-enabled device(s)")

            if device_count == 0:
                logger.warning("[GPU] No CUDA-capable devices found - falling back to CPU")
                logger.warning("[GPU] Possible causes: Driver issues, WSL2 D3D12 mapping, or no NVIDIA GPU")
                self.gpu_available = False
                return

            # Step 4: Set GPU device
            logger.info("[GPU INIT] Step 4: Setting GPU device 0...")
            cv2.cuda.setDevice(0)
            logger.info("[GPU INIT] ✓ GPU device 0 selected")

            # Step 5: Get device info (using OpenCV 4.10.0 API)
            logger.info("[GPU INIT] Step 5: Querying device information...")

            # Print device summary (includes name) to stdout - visible in logs
            logger.info("[GPU] Device information:")
            cv2.cuda.printShortCudaDeviceInfo(0)  # Prints: "Device 0: NVIDIA GeForce RTX 4070 Laptop GPU..."

            # Log structured properties using correct API
            device_info = cv2.cuda.DeviceInfo(0)
            logger.info(f"[GPU] Compute capability: {device_info.majorVersion()}.{device_info.minorVersion()}")
            logger.info(f"[GPU] Total memory: {device_info.totalMemory() / (1024**3):.2f} GB")
            logger.info(f"[GPU] Multiprocessor count: {device_info.multiProcessorCount()}")
            logger.info(f"[GPU] Clock rate: {device_info.clockRate() / 1000:.0f} MHz")

            # Step 6: Create CUDA stream
            logger.info("[GPU INIT] Step 6: Creating CUDA stream...")
            self.cuda_stream = cv2.cuda.Stream()
            logger.info("[GPU] Created CUDA stream for async operations")

            self.gpu_available = True
            logger.info("[GPU INIT] ✓✓✓ GPU acceleration ENABLED and ready ✓✓✓")

        except Exception as e:
            logger.error(f"[GPU] Failed to initialize GPU: {e}")
            logger.error(f"[GPU] Exception type: {type(e).__name__}")
            logger.error(f"[GPU] Exception details: {str(e)}")
            import traceback
            logger.error(f"[GPU] Traceback:\n{traceback.format_exc()}")
            logger.info("[GPU] Falling back to CPU operations")
            self.gpu_available = False

    def _get_cached_gpu_buffer(self, width: int, height: int, channels: int) -> Optional[cv2.cuda.GpuMat]:
        """Get or create a cached GPU buffer for the given dimensions."""
        if not self.gpu_available:
            return None

        cache_key = (width, height, channels)

        if cache_key not in self.gpu_frame_cache:
            self.gpu_frame_cache[cache_key] = []

        cache = self.gpu_frame_cache[cache_key]

        # Return existing buffer if available
        if cache:
            return cache.pop()

        # Create new buffer
        try:
            gpu_mat = cv2.cuda.GpuMat(height, width, cv2.CV_8UC3 if channels == 3 else cv2.CV_8UC1)
            return gpu_mat
        except Exception as e:
            logger.error(f"[GPU] Failed to allocate GPU buffer {width}x{height}x{channels}: {e}")
            return None

    def _return_gpu_buffer(self, gpu_mat: cv2.cuda.GpuMat):
        """Return a GPU buffer to the cache for reuse."""
        if not self.gpu_available or gpu_mat is None:
            return

        try:
            height, width = gpu_mat.size()
            channels = gpu_mat.channels()
            cache_key = (width, height, channels)

            if cache_key not in self.gpu_frame_cache:
                self.gpu_frame_cache[cache_key] = []

            cache = self.gpu_frame_cache[cache_key]

            # Only cache up to limit
            if len(cache) < self.cache_size_per_resolution:
                cache.append(gpu_mat)
        except Exception as e:
            logger.debug(f"[GPU] Failed to return buffer to cache: {e}")

    def process_frame_gpu(self, frame_bgr: np.ndarray, target_size: Tuple[int, int],
                         convert_to_rgb: bool = True) -> Optional[np.ndarray]:
        """
        Process frame on GPU: upload → resize → color convert → download.

        Args:
            frame_bgr: Input frame in BGR format (numpy array)
            target_size: Target (width, height) for resizing
            convert_to_rgb: Convert from BGR to RGB

        Returns:
            Processed frame as numpy array, or None on error
        """
        if not self.gpu_available:
            # Fallback to CPU
            return self._process_frame_cpu(frame_bgr, target_size, convert_to_rgb)

        t_start = time.perf_counter()

        try:
            # Upload to GPU
            t0 = time.perf_counter()
            gpu_frame = cv2.cuda.GpuMat()
            gpu_frame.upload(frame_bgr, self.cuda_stream)
            t_upload = (time.perf_counter() - t0) * 1000

            # Resize on GPU
            t0 = time.perf_counter()
            target_w, target_h = target_size

            # FIX #3: Validate target size before GPU resize to prevent driver crash
            if target_w <= 0 or target_h <= 0:
                logger.error(f"[GPU] Invalid target size: {target_w}x{target_h} - falling back to CPU processing")
                return self._process_frame_cpu(frame_bgr, target_size, convert_to_rgb)

            gpu_resized = cv2.cuda.resize(gpu_frame, (target_w, target_h),
                                        interpolation=cv2.INTER_LINEAR,
                                        stream=self.cuda_stream)
            t_resize = (time.perf_counter() - t0) * 1000

            # Color conversion on GPU (if requested)
            t0 = time.perf_counter()
            if convert_to_rgb:
                gpu_rgb = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2RGB, stream=self.cuda_stream)
            else:
                gpu_rgb = gpu_resized
            t_cvt = (time.perf_counter() - t0) * 1000

            # Download from GPU
            t0 = time.perf_counter()
            result = gpu_rgb.download(self.cuda_stream)

            # Wait for stream to complete
            self.cuda_stream.waitForCompletion()
            t_download = (time.perf_counter() - t0) * 1000

            # Track performance
            t_total = (time.perf_counter() - t_start) * 1000

            if self.profile_performance:
                self.perf_stats['upload_time'].append(t_upload)
                self.perf_stats['resize_time'].append(t_resize)
                self.perf_stats['cvt_color_time'].append(t_cvt)
                self.perf_stats['download_time'].append(t_download)
                self.perf_stats['total_time'].append(t_total)

                # Keep only recent stats (last 100 frames)
                for key in self.perf_stats:
                    if len(self.perf_stats[key]) > 100:
                        self.perf_stats[key] = self.perf_stats[key][-100:]

            return result

        except Exception as e:
            logger.error(f"[GPU] Frame processing failed: {e}")
            # Fallback to CPU
            return self._process_frame_cpu(frame_bgr, target_size, convert_to_rgb)

    def _process_frame_cpu(self, frame_bgr: np.ndarray, target_size: Tuple[int, int],
                          convert_to_rgb: bool = True) -> np.ndarray:
        """CPU fallback for frame processing."""
        t_start = time.perf_counter()

        # Resize
        target_w, target_h = target_size
        resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Color convert
        if convert_to_rgb:
            result = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            result = resized

        if self.profile_performance:
            t_total = (time.perf_counter() - t_start) * 1000
            self.perf_stats['total_time'].append(t_total)

        return result

    def draw_landmarks_gpu(self, frame: np.ndarray, landmarks: np.ndarray,
                          connections: List[Tuple[int, int]],
                          color: Tuple[int, int, int] = (0, 255, 0),
                          thickness: int = 2) -> np.ndarray:
        """
        Draw landmark connections on frame using GPU acceleration.

        Args:
            frame: Input frame (numpy array)
            landmarks: Nx3 array of (x, y, z) landmarks in pixel coordinates
            connections: List of (start_idx, end_idx) tuples defining connections
            color: Line color (B, G, R)
            thickness: Line thickness

        Returns:
            Frame with landmarks drawn
        """
        if not self.gpu_available or len(landmarks) == 0:
            # Fallback to CPU
            return self._draw_landmarks_cpu(frame, landmarks, connections, color, thickness)

        try:
            # For now, use CPU for drawing (GPU drawing of lines is not well-supported in OpenCV CUDA)
            # Future optimization: Use custom CUDA kernel or OpenGL for overlay rendering
            return self._draw_landmarks_cpu(frame, landmarks, connections, color, thickness)

        except Exception as e:
            logger.error(f"[GPU] Landmark drawing failed: {e}")
            return frame

    def _draw_landmarks_cpu(self, frame: np.ndarray, landmarks: np.ndarray,
                           connections: List[Tuple[int, int]],
                           color: Tuple[int, int, int], thickness: int) -> np.ndarray:
        """CPU implementation of landmark drawing."""
        result = frame.copy()

        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                pt1 = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                pt2 = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))

                # Skip connections with (0,0) coordinates
                if (pt1[0] == 0 and pt1[1] == 0) or (pt2[0] == 0 and pt2[1] == 0):
                    continue

                cv2.line(result, pt1, pt2, color, thickness, cv2.LINE_AA)

        return result

    def batch_process_frames(self, frames: List[np.ndarray], target_size: Tuple[int, int],
                            convert_to_rgb: bool = True) -> List[np.ndarray]:
        """
        Batch process multiple frames on GPU for multi-camera setups.

        Args:
            frames: List of input frames
            target_size: Target (width, height) for all frames
            convert_to_rgb: Convert from BGR to RGB

        Returns:
            List of processed frames
        """
        if not self.gpu_available or len(frames) == 0:
            # Process sequentially on CPU
            return [self._process_frame_cpu(f, target_size, convert_to_rgb) for f in frames]

        try:
            # Process all frames using GPU
            # Note: Current OpenCV CUDA doesn't support true batching,
            # but we can pipeline uploads/downloads for better performance
            results = []

            for frame in frames:
                processed = self.process_frame_gpu(frame, target_size, convert_to_rgb)
                if processed is not None:
                    results.append(processed)
                else:
                    # Fallback
                    results.append(self._process_frame_cpu(frame, target_size, convert_to_rgb))

            return results

        except Exception as e:
            logger.error(f"[GPU] Batch processing failed: {e}")
            return [self._process_frame_cpu(f, target_size, convert_to_rgb) for f in frames]

    def get_performance_stats(self) -> Dict[str, float]:
        """Get average performance statistics (in milliseconds)."""
        if not self.profile_performance:
            return {}

        stats = {}
        for key, times in self.perf_stats.items():
            if times:
                stats[f'{key}_avg'] = sum(times) / len(times)
                stats[f'{key}_max'] = max(times)
                stats[f'{key}_min'] = min(times)

        return stats

    def print_performance_report(self):
        """Print detailed performance report."""
        if not self.profile_performance:
            logger.info("[GPU] Performance profiling not enabled")
            return

        stats = self.get_performance_stats()

        if not stats:
            logger.info("[GPU] No performance data available")
            return

        # Single-line GPU performance report
        if 'total_time_avg' in stats:
            fps_capability = 1000.0 / stats['total_time_avg'] if stats['total_time_avg'] > 0 else 0
            logger.info(f"[GPU] {stats['total_time_avg']:.2f}ms avg | {fps_capability:.0f} FPS capability | "
                       f"GPU: {'Enabled' if self.gpu_available else 'Disabled'}")

    def cleanup(self):
        """Clean up GPU resources."""
        try:
            # Clear cache
            self.gpu_frame_cache.clear()

            # Release CUDA stream
            if self.cuda_stream is not None:
                self.cuda_stream = None

            logger.info("[GPU] Cleaned up GPU resources")

        except Exception as e:
            logger.error(f"[GPU] Cleanup error: {e}")


# Singleton instance for global access
_gpu_manager_instance = None


def get_gpu_manager(enable_gpu: bool = True, profile_performance: bool = False) -> GPUDrawingManager:
    """Get or create the global GPU drawing manager instance."""
    global _gpu_manager_instance

    if _gpu_manager_instance is None:
        logger.info(f"[GPU SINGLETON] Creating new GPU manager instance (enable_gpu={enable_gpu}, profile={profile_performance})")
        _gpu_manager_instance = GPUDrawingManager(enable_gpu, profile_performance)
    else:
        logger.info(f"[GPU SINGLETON] Returning existing GPU manager instance (gpu_available={_gpu_manager_instance.gpu_available})")
        logger.info(f"[GPU SINGLETON] Request parameters: enable_gpu={enable_gpu}, existing instance gpu_enabled={_gpu_manager_instance.gpu_enabled}")

    return _gpu_manager_instance
