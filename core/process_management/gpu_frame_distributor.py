"""
GPU-First Frame Distributor
CRITICAL: This replaces the entire frame distribution system.
All GPU processing happens in threads, not processes, to maintain GPU context.
"""

import cv2
import numpy as np
import cupy as cp
import threading
import time
from queue import Queue, Empty
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import warnings

# Suppress warnings that clutter output
warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class StreamConfig:
    """Configuration for each processing stream"""
    name: str
    needs_gpu: bool
    resolution: Optional[Tuple[int, int]]  # (width, height)
    format: str  # 'rgb' or 'bgr'
    queue: Queue
    gpu_processor: Optional[Callable] = None  # For GPU pipelines


class GPUFrameDistributor:
    """
    Captures frames using pinned memory for fast GPU transfer.
    Distributes to GPU pipelines (threads) and CPU consumers (processes).
    
    CRITICAL DESIGN DECISIONS:
    1. Uses threads for GPU processing (shares CUDA context)
    2. Uses pinned memory for 2-3x faster CPU→GPU transfer
    3. Pre-allocates all GPU buffers to avoid runtime allocation
    4. Computes common resolutions once on GPU
    """
    
    def __init__(self, camera_index: int, resolution: Tuple[int, int] = (1920, 1080), 
                 fps: int = 30, gpu_device: int = 0):
        """
        Initialize GPU-first frame distributor.
        
        Args:
            camera_index: Camera device index
            resolution: Target resolution (width, height)
            fps: Target frame rate
            gpu_device: GPU device ID
        """
        self.camera_index = camera_index
        self.target_resolution = resolution
        self.fps = fps
        self.gpu_device = gpu_device
        
        # Set GPU device
        cp.cuda.Device(gpu_device).use()
        print(f"[GPU Distributor] Using GPU device {gpu_device}")
        
        # Initialize camera - MUST be done first to get actual dimensions
        self._init_camera()
        
        # Pre-allocate pinned memory for zero-copy transfer
        self._init_pinned_memory()
        
        # Pre-allocate GPU buffers
        self._init_gpu_buffers()
        
        # Streams configuration
        self.streams: Dict[str, StreamConfig] = {}
        
        # Control
        self.is_running = False
        self.capture_thread = None
        self.stats = {
            'frames_captured': 0,
            'gpu_transfer_ms': [],
            'gpu_processing_ms': [],
            'drops': {}
        }
        
        # CUDA stream for async operations
        self.cuda_stream = cp.cuda.Stream(non_blocking=True)
        
        print(f"[GPU Distributor] Initialized for {self.actual_width}x{self.actual_height} capture")
    
    def _init_camera(self):
        """Initialize camera with optimal settings"""
        # Try DSHOW backend on Windows for better performance
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends:
            self.cap = cv2.VideoCapture(self.camera_index, backend)
            if self.cap.isOpened():
                print(f"[GPU Distributor] Using camera backend: {backend}")
                break
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Optimal: buffer=1 limits to 15fps, buffer=2+ achieves 29fps
        
        # Get resolution from camera properties
        prop_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        prop_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[GPU Distributor] Camera properties report: {prop_width}x{prop_height}")
        
        # Read a test frame to get ACTUAL dimensions
        ret, test_frame = self.cap.read()
        if ret:
            print(f"[GPU Distributor] Test frame shape: {test_frame.shape}, dtype: {test_frame.dtype}")
            self.actual_height, self.actual_width = test_frame.shape[:2]
            print(f"[GPU Distributor] Using actual frame dimensions: {self.actual_width}x{self.actual_height}")
        else:
            # Fallback to camera properties if test frame fails
            print(f"[GPU Distributor] WARNING: Could not read test frame, using camera properties")
            self.actual_width = prop_width
            self.actual_height = prop_height
        
        if (self.actual_width, self.actual_height) != self.target_resolution:
            print(f"[GPU Distributor] Warning: Camera resolution {self.actual_width}x{self.actual_height} "
                  f"differs from requested {self.target_resolution}")
    
    def _init_pinned_memory(self):
        """
        Allocate pinned memory for fast CPU→GPU transfers.
        Pinned memory provides 2-3x faster transfers than pageable memory.
        """
        # Calculate buffer size based on actual dimensions
        buffer_size = self.actual_height * self.actual_width * 3  # BGR
        
        # Debug info
        print(f"[GPU Distributor] Camera actual resolution: {self.actual_width}x{self.actual_height}")
        print(f"[GPU Distributor] Required buffer size: {buffer_size} bytes ({buffer_size / 1024 / 1024:.1f}MB)")
        
        # Allocate pinned memory
        self.pinned_buffer = cp.cuda.alloc_pinned_memory(buffer_size)
        
        # Create NumPy view for OpenCV to write into (as uint8)
        # The view will be exactly the size we need
        self.pinned_frame = np.frombuffer(self.pinned_buffer, dtype=np.uint8, count=buffer_size).reshape((self.actual_height, self.actual_width, 3))
        
        print(f"[GPU Distributor] Pinned memory ready for {self.actual_width}x{self.actual_height} frames")
    
    def _init_gpu_buffers(self):
        """Pre-allocate all GPU buffers to avoid runtime allocation"""
        # Primary capture buffers
        self.gpu_frame_bgr = cp.zeros((self.actual_height, self.actual_width, 3), dtype=cp.uint8)
        self.gpu_frame_rgb = cp.zeros((self.actual_height, self.actual_width, 3), dtype=cp.uint8)
        
        # Common resolutions (pre-computed)
        self.gpu_buffers = {
            'detection_640': cp.zeros((640, 640, 3), dtype=cp.uint8),
            'preview_540p': cp.zeros((540, 960, 3), dtype=cp.uint8),
            'pose_480p': cp.zeros((480, 640, 3), dtype=cp.uint8)
        }
        
        # Detection frame parameters
        self.detection_scale = min(640.0 / self.actual_width, 640.0 / self.actual_height)
        self.detection_new_w = int(self.actual_width * self.detection_scale)
        self.detection_new_h = int(self.actual_height * self.detection_scale)
        self.detection_offset_x = (640 - self.detection_new_w) // 2
        self.detection_offset_y = (640 - self.detection_new_h) // 2
        
        print(f"[GPU Distributor] Pre-allocated GPU buffers")
    
    def add_gpu_stream(self, name: str, gpu_processor: Callable, output_queue: Queue):
        """
        Add a GPU processing stream (runs in thread to share CUDA context).
        
        Args:
            name: Stream identifier
            gpu_processor: Function that processes GPU arrays
            output_queue: Queue for results (can be consumed by other processes)
        """
        if name in self.streams:
            raise ValueError(f"Stream '{name}' already exists")
        
        self.streams[name] = StreamConfig(
            name=name,
            needs_gpu=True,
            resolution=None,
            format='rgb',
            queue=output_queue,
            gpu_processor=gpu_processor
        )
        
        print(f"[GPU Distributor] Added GPU stream: {name}")
    
    def add_cpu_stream(self, name: str, output_queue: Queue, 
                      resolution: Optional[Tuple[int, int]] = None,
                      format: str = 'rgb'):
        """
        Add a CPU consumer stream (typically for GUI or pose).
        
        Args:
            name: Stream identifier
            output_queue: Queue for frames
            resolution: Desired resolution (None = original)
            format: 'rgb' or 'bgr'
        """
        if name in self.streams:
            raise ValueError(f"Stream '{name}' already exists")
        
        self.streams[name] = StreamConfig(
            name=name,
            needs_gpu=False,
            resolution=resolution,
            format=format,
            queue=output_queue
        )
        
        # Pre-allocate GPU buffer for this resolution if custom
        if resolution and resolution not in [(960, 540), (640, 480)]:
            key = f"custom_{resolution[0]}x{resolution[1]}"
            self.gpu_buffers[key] = cp.zeros((resolution[1], resolution[0], 3), dtype=cp.uint8)
        
        print(f"[GPU Distributor] Added CPU stream: {name} @ {resolution}")
    
    def _capture_and_distribute(self):
        """
        Main capture and distribution loop.
        CRITICAL: This runs in a thread to maintain GPU context.
        """
        frame_count = 0
        last_frame_time = time.time()
        frame_interval = 1.0 / self.fps
        
        # For FPS calculation
        fps_calc_frames = 0
        fps_calc_start = time.time()
        
        while self.is_running:
            loop_start = time.time()
            
            # Frame rate limiting
            elapsed = loop_start - last_frame_time
            if elapsed < frame_interval:
                time.sleep(0.001)
                continue
            
            # ========== CAPTURE PHASE ==========
            # Read frame from camera
            ret, frame = self.cap.read()
            if not ret:
                print("[GPU Distributor] Failed to capture frame")
                continue
            
            # Ensure frame matches expected dimensions
            if frame.shape != (self.actual_height, self.actual_width, 3):
                print(f"[GPU Distributor] Warning: Frame shape {frame.shape} doesn't match expected "
                      f"({self.actual_height}, {self.actual_width}, 3)")
                # Resize if needed
                if frame.shape[0] != self.actual_height or frame.shape[1] != self.actual_width:
                    frame = cv2.resize(frame, (self.actual_width, self.actual_height))
            
            # Copy to pinned memory
            np.copyto(self.pinned_frame, frame)
            
            frame_count += 1
            fps_calc_frames += 1
            capture_timestamp = time.time()
            
            # ========== GPU TRANSFER PHASE ==========
            transfer_start = time.time()
            
            # Async transfer to GPU using CUDA stream
            with self.cuda_stream:
                # Transfer to GPU (async) - use asarray to transfer from pinned memory
                self.gpu_frame_bgr[:] = cp.asarray(self.pinned_frame)
                
                # Convert BGR to RGB on GPU
                self.gpu_frame_rgb[:] = self.gpu_frame_bgr[:, :, ::-1]
                
                # Pre-compute common formats
                self._prepare_standard_formats()
            
            # Wait for GPU operations to complete
            self.cuda_stream.synchronize()
            
            transfer_time = time.time() - transfer_start
            self.stats['gpu_transfer_ms'].append(transfer_time * 1000)
            
            # ========== DISTRIBUTION PHASE ==========
            distribution_start = time.time()
            
            # Process each stream
            for stream_name, stream in self.streams.items():
                if stream.queue.full():
                    self.stats['drops'][stream_name] = self.stats['drops'].get(stream_name, 0) + 1
                    continue
                
                try:
                    if stream.needs_gpu:
                        # GPU processing stream
                        self._process_gpu_stream(stream, frame_count, capture_timestamp)
                    else:
                        # CPU consumer stream
                        self._process_cpu_stream(stream, frame_count, capture_timestamp)
                
                except Exception as e:
                    print(f"[GPU Distributor] Error in stream '{stream_name}': {e}")
                    import traceback
                    traceback.print_exc()
            
            distribution_time = time.time() - distribution_start
            self.stats['gpu_processing_ms'].append(distribution_time * 1000)
            
            # Update stats
            self.stats['frames_captured'] = frame_count
            last_frame_time = loop_start
            
            # Print FPS every 5 seconds
            if time.time() - fps_calc_start >= 5.0:
                actual_fps = fps_calc_frames / (time.time() - fps_calc_start)
                avg_transfer = np.mean(self.stats['gpu_transfer_ms'][-30:]) if self.stats['gpu_transfer_ms'] else 0
                avg_process = np.mean(self.stats['gpu_processing_ms'][-30:]) if self.stats['gpu_processing_ms'] else 0
                
                print(f"\n[GPU Distributor] Performance Report:")
                print(f"  Actual FPS: {actual_fps:.1f}")
                print(f"  GPU Transfer: {avg_transfer:.2f}ms")
                print(f"  GPU Processing: {avg_process:.2f}ms")
                print(f"  Total GPU time: {avg_transfer + avg_process:.2f}ms")
                print(f"  Drops: {self.stats['drops']}")
                
                fps_calc_frames = 0
                fps_calc_start = time.time()
    
    def _prepare_standard_formats(self):
        """Pre-compute standard formats on GPU"""
        # Detection frame (640x640 padded)
        self.gpu_buffers['detection_640'].fill(0)
        resized = self._gpu_resize(self.gpu_frame_rgb, 
                                  (self.detection_new_h, self.detection_new_w))
        self.gpu_buffers['detection_640'][
            self.detection_offset_y:self.detection_offset_y + self.detection_new_h,
            self.detection_offset_x:self.detection_offset_x + self.detection_new_w
        ] = resized
        
        # Preview 540p
        self._gpu_resize_into(self.gpu_frame_bgr, self.gpu_buffers['preview_540p'])
        
        # Pose 480p RGB
        self._gpu_resize_into(self.gpu_frame_rgb, self.gpu_buffers['pose_480p'])
    
    def _gpu_resize(self, src: cp.ndarray, new_size: Tuple[int, int]) -> cp.ndarray:
        """
        Efficient GPU resize using CuPy.
        Note: This is bilinear interpolation, not as high quality as cv2.resize
        but much faster on GPU.
        """
        from cupyx.scipy import ndimage
        
        h_new, w_new = new_size
        h_old, w_old = src.shape[:2]
        
        zoom_factors = [h_new / h_old, w_new / w_old, 1]
        
        # Use order=1 for bilinear interpolation (fast)
        # Use order=3 for bicubic (slower but better quality)
        resized = ndimage.zoom(src, zoom_factors, order=1)
        
        return resized.astype(cp.uint8)
    
    def _gpu_resize_into(self, src: cp.ndarray, dst: cp.ndarray):
        """Resize directly into pre-allocated buffer"""
        h_dst, w_dst = dst.shape[:2]
        resized = self._gpu_resize(src, (h_dst, w_dst))
        cp.copyto(dst, resized)
    
    def _process_gpu_stream(self, stream: StreamConfig, frame_id: int, timestamp: float):
        """Process GPU stream - NO CPU TRANSFER"""
        if stream.gpu_processor is None:
            return
        
        # Prepare inputs based on stream needs
        if stream.name == 'face':
            # Face processing gets detection frame and full resolution
            result = stream.gpu_processor(
                detection_frame=self.gpu_buffers['detection_640'],
                full_frame=self.gpu_frame_rgb,
                frame_id=frame_id,
                timestamp=timestamp,
                detection_params={
                    'scale': self.detection_scale,
                    'offset_x': self.detection_offset_x,
                    'offset_y': self.detection_offset_y,
                    'original_width': self.actual_width,
                    'original_height': self.actual_height
                }
            )
        else:
            # Other GPU processors get full frame
            result = stream.gpu_processor(
                gpu_frame=self.gpu_frame_rgb,
                frame_id=frame_id,
                timestamp=timestamp
            )
        
        # Add metadata
        result['frame_id'] = frame_id
        result['timestamp'] = timestamp
        
        # Send results (only final results are on CPU)
        stream.queue.put_nowait(result)
    
    def _process_cpu_stream(self, stream: StreamConfig, frame_id: int, timestamp: float):
        """Process CPU stream - transfer only what's needed"""
        # Select appropriate GPU buffer
        if stream.resolution == (960, 540) and stream.format == 'bgr':
            gpu_buffer = self.gpu_buffers['preview_540p']
        elif stream.resolution == (640, 480) and stream.format == 'rgb':
            gpu_buffer = self.gpu_buffers['pose_480p']
        elif stream.resolution:
            # Custom resolution
            key = f"custom_{stream.resolution[0]}x{stream.resolution[1]}"
            if key in self.gpu_buffers:
                gpu_src = self.gpu_frame_bgr if stream.format == 'bgr' else self.gpu_frame_rgb
                self._gpu_resize_into(gpu_src, self.gpu_buffers[key])
                gpu_buffer = self.gpu_buffers[key]
            else:
                # Fallback to resizing on CPU
                gpu_buffer = self.gpu_frame_bgr if stream.format == 'bgr' else self.gpu_frame_rgb
        else:
            # Original resolution
            gpu_buffer = self.gpu_frame_bgr if stream.format == 'bgr' else self.gpu_frame_rgb
        
        # Transfer to CPU
        cpu_frame = cp.asnumpy(gpu_buffer)
        
        # Send frame
        stream.queue.put_nowait({
            'frame': cpu_frame,
            'frame_id': frame_id,
            'timestamp': timestamp,
            'resolution': (self.actual_width, self.actual_height),  # Original capture resolution
            'preview_resolution': (cpu_frame.shape[1], cpu_frame.shape[0]),  # Actual frame resolution
            'format': stream.format
        })
    
    def start(self):
        """Start capture and distribution thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.capture_thread = threading.Thread(
            target=self._capture_and_distribute,
            name="GPUFrameDistributor",
            daemon=True
        )
        self.capture_thread.start()
        print("[GPU Distributor] Started capture thread")
    
    def stop(self):
        """Stop capture and cleanup"""
        print("[GPU Distributor] Stopping...")
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        # Print final stats
        if self.stats['gpu_transfer_ms']:
            print(f"\n[GPU Distributor] Final Statistics:")
            print(f"  Total frames: {self.stats['frames_captured']}")
            print(f"  Avg GPU transfer: {np.mean(self.stats['gpu_transfer_ms']):.2f}ms")
            print(f"  Avg GPU processing: {np.mean(self.stats['gpu_processing_ms']):.2f}ms")
            print(f"  Final drops: {self.stats['drops']}")
        
        print("[GPU Distributor] Stopped")