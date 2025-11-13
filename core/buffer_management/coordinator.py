"""
Buffer Coordinator - Centralized buffer management for multi-process system.

This module provides the BufferCoordinator class which serves as the single authority
for all buffer management across the system, coordinating shared memory allocation,
buffer discovery, and cleanup.
"""

import logging
import time
import atexit
import os
import struct
import multiprocessing as mp
from multiprocessing import Value, Semaphore, Lock, Event, shared_memory
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Import CommandBuffer from the new location
from .command_buffer import CommandBuffer
# Import buffer layout dataclasses
from .layouts import (
    FrameBufferLayout,
    Pose3DBufferLayout,
    ResultsBufferLayout,
    ROIBufferLayout,
    DetectionBufferLayout,
    EmbeddingBufferLayout,
    DisplayBufferLayout,
    create_layout
)
# Import ParticipantQueryBuffer for recovery buffer creation
from .sharedbuffer import ParticipantQueryBuffer


logger = logging.getLogger('BufferSystem')


class BufferCoordinator:
    """
    Centralized buffer coordination with atomic synchronization.
    Single authority for all buffer management across the system.
    """
    
    def __init__(self, camera_count: int, config: Dict[str, Any], create_coordinator_info: bool = True,
                 external_semaphores: Optional[Dict[int, Semaphore]] = None,
                 external_pose_semaphores: Optional[Dict[int, Semaphore]] = None):
        """
        Initialize BufferCoordinator.

        Args:
            camera_count: Number of cameras to support
            config: Configuration dictionary
            create_coordinator_info: Whether to create coordinator info file for face recognition discovery.
                                    Should be False for child processes to prevent overwriting parent's file.
            external_semaphores: Optional dict of pre-created frame_ready semaphores {camera_index: Semaphore}.
                               If provided, uses these instead of creating new ones.
                               CRITICAL for process spawn chains to share the SAME semaphore instance.
            external_pose_semaphores: Optional dict of pre-created pose_frame_ready semaphores {camera_index: Semaphore}.
                                     If provided, uses these instead of creating new ones.
                                     CRITICAL for process spawn chains to share the SAME semaphore instance.
        """
        self.camera_count = camera_count
        self.config = config
        
        # Centralized buffer calculations
        self.max_faces = self._calculate_max_faces()
        self.ring_buffer_size = self._get_ring_buffer_size()
        self.roi_buffer_size = self._get_roi_buffer_size()
        self.gui_buffer_size = config.get('buffer_settings', {}).get('ring_buffers', {}).get('gui_display', 8)
        
        # Global sequence counter for ordering validation
        self.sequence_counter = Value('L', 0)

        # Frame ready semaphores for event-driven synchronization (replaces polling)
        # One semaphore per camera - posted by camera worker, waited by MediaPipe
        if external_semaphores is not None:
            # Use provided semaphores (child process using parent's semaphores)
            self.frame_ready_semaphores = external_semaphores
            logger.info(f"Using {len(external_semaphores)} external frame_ready semaphores (child process)")
        else:
            # Create new semaphores (parent process)
            self.frame_ready_semaphores = {i: Semaphore(0) for i in range(camera_count)}

        # Pose frame ready semaphores for event-driven pose frame synchronization
        # One semaphore per camera - posted by camera worker when writing pose frames, waited by PoseProcess
        if external_pose_semaphores is not None:
            # Use provided semaphores (child process using parent's semaphores)
            self.pose_frame_ready_semaphores = external_pose_semaphores
            logger.info(f"Using {len(external_pose_semaphores)} external pose_frame_ready semaphores (child process)")
        else:
            # Create new semaphores (parent process)
            self.pose_frame_ready_semaphores = {i: Semaphore(0) for i in range(camera_count)}

        # Track created buffers for cleanup
        self.created_buffers = {}  # {name: shm_object}
        self.buffer_registry = {}  # {camera_index: {buffer_type: name}}

        # SCRFD parallel processing buffers
        self.detection_buffers = {}  # {camera_index: detection_buffer_info}
        self.embedding_buffers = {}  # {camera_index: embedding_buffer_info}

        # SCRFD parallel processing - Event-driven synchronization (Phase 1.3)
        # Separate semaphores prevent signal stealing between MediaPipe and ArcFace
        self.mediapipe_ready_semaphores = {}  # {camera_index: Semaphore(0)} - SCRFD→MediaPipe
        self.arcface_ready_semaphores = {}  # {camera_index: Semaphore(0)} - SCRFD→ArcFace
        self.mediapipe_done_events = {}  # {camera_index: Event()} - MediaPipe signals completion
        self.arcface_done_events = {}  # {camera_index: Event()} - ArcFace signals completion
        self.detection_ref_count_locks = {}  # {camera_index: Lock()} for atomic ref_count decrement

        for i in range(camera_count):
            self.mediapipe_ready_semaphores[i] = Semaphore(0)  # Start at 0, SCRFD posts to signal
            self.arcface_ready_semaphores[i] = Semaphore(0)  # Start at 0, SCRFD posts to signal
            self.mediapipe_done_events[i] = mp.Event()  # Event for MediaPipe completion
            self.arcface_done_events[i] = mp.Event()  # Event for ArcFace completion
            self.detection_ref_count_locks[i] = Lock()  # Shared lock for atomic ref_count decrement

        # Clean up stale buffers from previous runs (prevents buffer size mismatches)
        self._cleanup_stale_buffers()

        # Track camera resolutions for dynamic buffer size calculations
        self.camera_resolutions = {}  # {camera_index: (width, height)}

        # Pre-populate camera resolutions from config (avoids timing issues with worker startup)
        camera_settings = config.get('camera_settings', {})
        for i in range(camera_count):
            cam_key = f'camera_{i}'
            if cam_key in camera_settings:
                cam_config = camera_settings[cam_key]
                if cam_config.get('enabled', False):
                    width = cam_config.get('width', 1920)
                    height = cam_config.get('height', 1080)
                    self.camera_resolutions[i] = (width, height)

        # Adaptive sizing tracking
        self.face_count_history = {}  # {camera_index: [recent_face_counts]}
        self.buffer_usage_stats = {}  # {camera_index: usage_metrics}
        
        # Face recognition integration
        face_rec_config = config.get('face_recognition_process', {})
        self.face_recognition_enabled = face_rec_config.get('enabled', True)
        
        # Write coordinator info file for discovery (only if requested and enabled)
        if self.face_recognition_enabled and create_coordinator_info:
            try:
                self.coordinator_info_file = self.write_coordinator_info_file()
                atexit.register(self._cleanup_coordinator_info_file)
            except Exception as e:
                logger.warning(f"Failed to write coordinator info file: {e}")
                self.coordinator_info_file = None
        else:
            self.coordinator_info_file = None
        
        # Register cleanup on exit
        atexit.register(self.cleanup_all_buffers)

        logger.debug(f"BufferCoordinator initialized: {camera_count} cameras, max_faces={self.max_faces}")
    
    def _calculate_max_faces(self) -> int:
        """Single source of truth for max faces calculation."""
        # Use centralized buffer_settings configuration
        participant_count = self.config.get('buffer_settings', {}).get('persons', {}).get('participant_count', 1)
        max_faces_config = self.config.get('buffer_settings', {}).get('faces', {}).get('max_faces_per_frame', 10)
        gpu_max_batch = self.config.get('advanced_detection', {}).get('gpu_settings', {}).get('max_batch_size', 8)

        max_faces = min(
            participant_count,      # Use participant_count directly
            max_faces_config,       # Configuration limit
            gpu_max_batch,          # GPU hardware limit
            8                       # Absolute maximum for memory safety
        )

        return max_faces

    # ============ CENTRALIZED BUFFER CONSTANTS ============

    def get_landmark_count(self) -> int:
        """MediaPipe FaceLandmarker outputs 478 landmarks per face."""
        return 478

    def get_blendshape_count(self) -> int:
        """MediaPipe FaceLandmarker outputs 52 blendshapes per face."""
        return 52

    def get_frame_size(self, resolution: Tuple[int, int]) -> int:
        """Calculate frame size in bytes for given resolution (BGR format)."""
        return resolution[0] * resolution[1] * 3  # BGR format

    def get_face_data_size(self) -> int:
        """
        Calculate face data structure size for display buffer.
        Must match DisplayBufferReader expectations.
        """
        fixed_fields_size = 40  # IIIfffffBI struct
        label_size = 64         # MAX_LABEL_LENGTH
        landmarks_size = self.get_landmark_count() * 3 * 4  # xyz, float32
        return fixed_fields_size + label_size + landmarks_size

    def get_frame_metadata_dtype(self) -> np.dtype:
        """
        Get frame metadata structured dtype with correct alignment.
        Ensures consistent metadata size across all components.
        """
        return np.dtype([
            ('frame_id', 'int64'),
            ('timestamp', 'int64'),
            ('is_detection', 'int64'),
            ('ready', 'int64'),
            ('padding', 'int64', 4)  # Pad to 64 bytes total
        ], align=True)

    def get_frame_metadata_size(self) -> int:
        """Get frame metadata size in bytes."""
        return self.get_frame_metadata_dtype().itemsize  # Should be 64 bytes

    def _get_ring_buffer_size(self) -> int:
        """Get ring buffer size, ensuring power of 2 for fast modulo operations."""
        # Use centralized buffer_settings configuration
        size = self.config.get('buffer_settings', {}).get('ring_buffers', {}).get('frame_detection', 16)

        # Ensure power of 2
        if size & (size - 1) != 0:
            size = 1 << (size - 1).bit_length()
            logger.info(f"Ring buffer size rounded to power of 2: {size}")

        return size
    
    def _get_roi_buffer_size(self) -> int:
        """Get ROI buffer size, ensuring power of 2."""
        # Use centralized buffer_settings configuration
        size = self.config.get('buffer_settings', {}).get('ring_buffers', {}).get('roi_processing', 8)

        # Ensure power of 2
        if size & (size - 1) != 0:
            size = 1 << (size - 1).bit_length()
            logger.info(f"ROI buffer size rounded to power of 2: {size}")

        return size

    # ============ CENTRALIZED LAYOUT API ============

    def get_layout(self, buffer_type: str, camera_index: int = None, **kwargs):
        """
        Single entry point for all buffer layouts.
        Returns type-safe dataclass instances for buffer layouts.

        Args:
            buffer_type: Type of buffer ('frame', 'results', 'pose', 'detection', 'embedding', 'display', 'roi')
            camera_index: Optional camera index for resolution-dependent layouts
            **kwargs: Additional parameters passed to layout constructor

        Returns:
            Buffer layout dataclass instance (FrameBufferLayout, ResultsBufferLayout, etc.)

        Example:
            layout = coordinator.get_layout('results', max_faces=8)
            landmarks_offset = layout.landmarks_offset
        """
        # Default parameters from coordinator config
        if buffer_type == 'frame':
            if camera_index is None:
                raise ValueError("camera_index required for frame buffer layout")

            # Try to get resolution from coordinator, or use kwargs override
            resolution = self.camera_resolutions.get(camera_index)
            if resolution is None:
                # Allow manual override via kwargs (for external processes)
                if 'frame_width' in kwargs and 'frame_height' in kwargs:
                    frame_width = kwargs.pop('frame_width')
                    frame_height = kwargs.pop('frame_height')
                else:
                    raise ValueError(f"No resolution found for camera {camera_index}")
            else:
                frame_width = resolution[0]
                frame_height = resolution[1]

            # Pop ring_buffer_size from kwargs to avoid duplicate argument error
            ring_buffer_size = kwargs.pop('ring_buffer_size', self.ring_buffer_size)

            return FrameBufferLayout(
                ring_buffer_size=ring_buffer_size,
                frame_width=frame_width,
                frame_height=frame_height,
                **kwargs
            )

        elif buffer_type == 'results':
            max_faces = kwargs.get('max_faces', self.max_faces)
            return ResultsBufferLayout(
                max_faces=max_faces,
                **{k: v for k, v in kwargs.items() if k != 'max_faces'}
            )

        elif buffer_type == 'pose':
            # Get pose-specific settings from centralized buffer_settings
            pose_max_persons = self.config.get('buffer_settings', {}).get('persons', {}).get('max_persons', 1)
            pose_ring_buffer_size = self.config.get('buffer_settings', {}).get('ring_buffers', {}).get('pose_estimation', 4)
            # Use existing Pose3DBufferLayout
            return Pose3DBufferLayout(
                max_persons=kwargs.get('max_persons', pose_max_persons),
                ring_buffer_size=kwargs.get('ring_buffer_size', pose_ring_buffer_size),
                **{k: v for k, v in kwargs.items() if k not in ['max_persons', 'ring_buffer_size']}
            )

        elif buffer_type == 'detection':
            max_faces = kwargs.get('max_faces', self.max_faces)
            return DetectionBufferLayout(
                max_faces=max_faces,
                **{k: v for k, v in kwargs.items() if k != 'max_faces'}
            )

        elif buffer_type == 'embedding':
            max_faces = kwargs.get('max_faces', self.max_faces)
            return EmbeddingBufferLayout(
                max_faces=max_faces,
                **{k: v for k, v in kwargs.items() if k != 'max_faces'}
            )

        elif buffer_type in ['display', 'gui']:
            if camera_index is None:
                raise ValueError("camera_index required for display buffer layout")
            resolution = self.camera_resolutions.get(camera_index)
            if resolution is None:
                raise ValueError(f"No resolution found for camera {camera_index}")
            return DisplayBufferLayout(
                ring_buffer_size=kwargs.get('ring_buffer_size', 4),
                frame_width=resolution[0],
                frame_height=resolution[1],
                max_faces=kwargs.get('max_faces', self.max_faces),
                **{k: v for k, v in kwargs.items() if k not in ['ring_buffer_size', 'max_faces']}
            )

        elif buffer_type == 'roi':
            enable_preprocessing = self.config.get('process_separation', {}).get('enable_preprocessing_process', False)
            return ROIBufferLayout(
                max_faces=kwargs.get('max_faces', self.max_faces),
                roi_buffer_size=kwargs.get('roi_buffer_size', self.roi_buffer_size),
                enable_preprocessing=kwargs.get('enable_preprocessing', enable_preprocessing),
                **{k: v for k, v in kwargs.items() if k not in ['max_faces', 'roi_buffer_size', 'enable_preprocessing']}
            )

        else:
            raise ValueError(f"Unknown buffer type: {buffer_type}")

    def create_camera_buffers(self, camera_index: int, actual_resolution: Tuple[int, int]) -> Dict[str, str]:
        """
        Create all buffers for a camera with proper sizing.
        MediaPipe native - no ROI buffer needed.
        Returns shared memory names for all buffers.
        """
        frame_w, frame_h = actual_resolution
        frame_size = frame_w * frame_h * 3  # RGB

        # Store resolution for dynamic buffer calculations
        self.camera_resolutions[camera_index] = tuple(actual_resolution)
        logger.info(f"[BufferCoordinator] Stored camera {camera_index} resolution: {actual_resolution[0]}x{actual_resolution[1]}")

        # Validate resolution is supported
        common_resolutions = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
        if tuple(actual_resolution) not in common_resolutions:
            logger.warning(f"Camera {camera_index} using non-standard resolution {actual_resolution}. "
                          f"Supported resolutions: {common_resolutions}. Buffer calculations may need review.")

        # Calculate buffer sizes
        detection_metadata_size = 64 * self.ring_buffer_size  # 16 int32 per slot

        # Add header with write index and resolution info at the beginning
        # Header layout: [write_index(8)][width(4)][height(4)] = 16 bytes total
        header_size = 16  # 8 bytes for write index + 4 bytes width + 4 bytes height

        frame_buffer_size = (
            header_size +  # Header with write index and resolution
            frame_size * self.ring_buffer_size + detection_metadata_size  # Detection ring only
        )

        # Results buffer: use centralized layout calculation (includes blendshapes)
        results_layout = self.get_results_buffer_layout()
        results_size = results_layout['total_size']
        landmarks_size = results_layout['landmarks_size']  # For GUI buffer calculation below

        # GUI-specific buffer (separate from processing pipeline)
        # Layout: [header(16)][frames][metadata]
        # Header: [write_index(8)][width(4)][height(4)]
        gui_buffer_size = header_size + (frame_size * self.gui_buffer_size) + landmarks_size + 2048  # Header + frames + landmarks + metadata

        # Pose buffer layout
        pose_layout = self.get_pose_buffer_layout(camera_index)
        pose_size = pose_layout['total_size']

        # Create unique buffer names with process PID to avoid conflicts
        import os
        pid = os.getpid()

        buffer_names = {
            'frame': f'yq_frame_{camera_index}_{pid}',
            'results': f'yq_results_{camera_index}_{pid}',
            'gui': f'yq_gui_{camera_index}_{pid}',
            'pose': f'yq_pose_{camera_index}_{pid}',
            'detection': f'yq_detection_{camera_index}_{pid}',     # SCRFD detection buffer
            'embedding': f'yq_embedding_{camera_index}_{pid}'      # ArcFace embedding buffer
        }

        # Create shared memory segments
        try:
            # Frame buffer
            frame_shm = shared_memory.SharedMemory(
                create=True,
                name=buffer_names['frame'],
                size=frame_buffer_size
            )
            self.created_buffers[buffer_names['frame']] = frame_shm

            # Initialize frame buffer header with resolution
            # Header layout: [write_index(8)][width(4)][height(4)]
            np.frombuffer(frame_shm.buf, dtype=np.uint32, count=2, offset=8)[:] = [frame_w, frame_h]

            # Initialize ring buffer metadata slots to prevent garbage frame IDs
            # Zero out all metadata slots (64 bytes each) - no structured view needed
            detection_metadata_start = header_size + frame_size * self.ring_buffer_size
            for i in range(self.ring_buffer_size):
                metadata_offset = detection_metadata_start + i * 64
                # Zero out the entire 64-byte metadata slot
                frame_shm.buf[metadata_offset:metadata_offset + 64] = b'\x00' * 64
            logger.info(f"[BufferCoordinator] Initialized {self.ring_buffer_size} ring buffer metadata slots for camera {camera_index}")

            # Results buffer
            results_shm = shared_memory.SharedMemory(
                create=True,
                name=buffer_names['results'],
                size=results_size
            )
            self.created_buffers[buffer_names['results']] = results_shm

            # GUI buffer
            gui_shm = shared_memory.SharedMemory(
                create=True,
                name=buffer_names['gui'],
                size=gui_buffer_size
            )
            self.created_buffers[buffer_names['gui']] = gui_shm

            # Initialize GUI buffer header with resolution
            np.frombuffer(gui_shm.buf, dtype=np.uint32, count=2, offset=8)[:] = [frame_w, frame_h]

            # Initialize GUI buffer write index to 0
            gui_write_index = np.ndarray((1,), dtype=np.uint64, buffer=gui_shm.buf[0:8])
            gui_write_index[0] = 0

            # Pose buffer
            pose_shm = shared_memory.SharedMemory(
                create=True,
                name=buffer_names['pose'],
                size=pose_size
            )
            self.created_buffers[buffer_names['pose']] = pose_shm

            # Initialize pose buffer write index to 0
            pose_write_index = np.ndarray((1,), dtype=np.uint64, buffer=pose_shm.buf[0:8])
            pose_write_index[0] = 0

            # SCRFD detection buffer (with zero-copy crops for parallel MediaPipe + ArcFace)
            detection_info = self.create_detection_buffer(camera_index)
            buffer_names['detection'] = detection_info['buffer_name']  # Update with actual name
            detection_size = detection_info['layout']['total_size']
            self.detection_buffers[camera_index] = detection_info  # Store for child process access

            # ArcFace embedding buffer
            embedding_info = self.create_embedding_buffer(camera_index)
            buffer_names['embedding'] = embedding_info['buffer_name']  # Update with actual name
            embedding_size = embedding_info['layout']['total_size']
            self.embedding_buffers[camera_index] = embedding_info  # Store for child process access

            # Store in registry
            self.buffer_registry[camera_index] = buffer_names.copy()

            logger.info(f"Camera {camera_index} buffers created:")
            logger.info(f"  Frame buffer: {frame_buffer_size} bytes ({buffer_names['frame']})")
            logger.info(f"  Results buffer: {results_size} bytes ({buffer_names['results']})")
            logger.info(f"  GUI buffer: {gui_buffer_size} bytes ({buffer_names['gui']})")
            logger.info(f"  Pose buffer: {pose_size} bytes ({buffer_names['pose']})")
            logger.info(f"  Detection buffer: {detection_size} bytes ({buffer_names['detection']})")
            logger.info(f"  Embedding buffer: {embedding_size} bytes ({buffer_names['embedding']})")

            # Validate buffer layouts to catch size mismatches early
            self.validate_buffer_layout(buffer_names['results'], results_size, 'results')
            self.validate_buffer_layout(buffer_names['pose'], pose_size, 'pose')

            # Log detailed layout information for debugging
            if camera_index == 0:  # Only log once to avoid spam
                self.log_buffer_layout_info()

            return buffer_names

        except Exception as e:
            logger.error(f"Failed to create buffers for camera {camera_index}: {e}")
            # Clean up any created buffers
            self._cleanup_camera_buffers(camera_index)
            raise

    def destroy_camera_buffers(self, camera_index: int):
        """
        Public API to destroy all buffers for a camera.
        Used for dynamic resolution switching - cleanly removes buffers
        so they can be recreated with new dimensions.

        Args:
            camera_index: Camera index to destroy buffers for
        """
        logger.info(f"[BufferCoordinator] Destroying all buffers for camera {camera_index}")

        # CRITICAL FIX: Invalidate cached layouts for this camera
        # Without this, get_buffer_layout() may return stale dimensions (e.g. 1080p when switching to 480p)
        if hasattr(self, '_layout_cache'):
            keys_to_remove = [key for key in self._layout_cache.keys() if key[0] == camera_index]
            for key in keys_to_remove:
                del self._layout_cache[key]
            if keys_to_remove:
                logger.info(f"[BufferCoordinator] Invalidated {len(keys_to_remove)} cached layouts for camera {camera_index}")

        # Clean up all shared memory buffers
        self._cleanup_camera_buffers(camera_index)

        # Remove resolution tracking
        if camera_index in self.camera_resolutions:
            old_resolution = self.camera_resolutions[camera_index]
            del self.camera_resolutions[camera_index]
            logger.info(f"[BufferCoordinator] Removed resolution tracking for camera {camera_index} (was {old_resolution[0]}x{old_resolution[1]})")

        logger.info(f"[BufferCoordinator] Camera {camera_index} buffers destroyed successfully")

    def create_preprocessing_buffer(self, camera_idx: int) -> str:
        """
        Create preprocessed buffer for MediaPipe preprocessing process.
        
        Args:
            camera_idx: Camera index
            
        Returns:
            Name of the created preprocessed buffer
        """
        try:
            buffer_name = f"yq_preprocessed_{os.getpid()}_{camera_idx}"
            
            # Calculate buffer size for preprocessed tensors
            # Preprocessed tensors: 256x256x3 float32 = 786,432 bytes per ROI
            preprocessed_tensor_size = 256 * 256 * 3 * 4  # float32
            metadata_size = 1024  # Same metadata structure as ROI buffer
            
            # Total buffer size
            total_preprocessed_size = (
                8 +  # write index
                (preprocessed_tensor_size * self.max_faces * self.roi_buffer_size) +  # preprocessed data
                (metadata_size * self.roi_buffer_size)  # metadata
            )
            
            # Create shared memory
            from multiprocessing import shared_memory
            shm = shared_memory.SharedMemory(create=True, size=total_preprocessed_size, name=buffer_name)
            
            # Initialize buffer
            shm.buf[:8] = np.array([0], dtype=np.int64).tobytes()  # write index at offset 0
            shm.buf[8:] = b'\x00' * (total_preprocessed_size - 8)  # zero out rest
            
            # Register buffer for cleanup
            self.created_buffers[buffer_name] = shm
            self.buffer_registry[buffer_name] = {
                'type': 'preprocessed',
                'camera_index': camera_idx,
                'size': total_preprocessed_size,
                'shm': shm
            }
            
            logger.info(f"Created preprocessed buffer {buffer_name} for camera {camera_idx} "
                       f"(size: {total_preprocessed_size / 1024 / 1024:.1f} MB)")
            
            return buffer_name
            
        except Exception as e:
            logger.error(f"Failed to create preprocessed buffer for camera {camera_idx}: {e}")
            raise
    
    def get_gui_buffer_names(self) -> Dict[int, str]:
        """Get GUI-specific buffer names for each camera."""
        gui_names = {}
        for camera_index, buffers in self.buffer_registry.items():
            gui_names[camera_index] = buffers['gui']
        return gui_names

    def get_frame_ready_semaphore(self, camera_index: int) -> Semaphore:
        """
        Get frame ready semaphore for a camera.

        Used for event-driven frame synchronization between camera worker and MediaPipe.
        Camera worker posts to semaphore after writing frame.
        MediaPipe waits on semaphore instead of busy-wait polling.

        Args:
            camera_index: Camera index

        Returns:
            Semaphore instance for this camera
        """
        return self.frame_ready_semaphores[camera_index]

    def get_pose_frame_ready_semaphore(self, camera_index: int) -> Semaphore:
        """
        Get pose frame ready semaphore for a camera.

        Used for event-driven pose frame synchronization between camera worker and PoseProcess.
        Camera worker posts to semaphore after writing pose frame.
        PoseProcess waits on semaphore instead of polling.

        Args:
            camera_index: Camera index

        Returns:
            Semaphore instance for this camera's pose frames
        """
        return self.pose_frame_ready_semaphores[camera_index]

    def get_mediapipe_ready_semaphore(self, camera_index: int) -> Semaphore:
        """
        Get MediaPipe ready semaphore for SCRFD → MediaPipe signaling.

        Event-driven synchronization pattern:
        - SCRFD posts to semaphore after writing detections
        - MediaPipe waits on semaphore instead of polling

        Args:
            camera_index: Camera index

        Returns:
            Semaphore instance for MediaPipe detection events
        """
        return self.mediapipe_ready_semaphores[camera_index]

    def get_arcface_ready_semaphore(self, camera_index: int) -> Semaphore:
        """
        Get ArcFace ready semaphore for SCRFD → ArcFace signaling.

        Event-driven synchronization pattern:
        - SCRFD posts to semaphore after writing detections
        - ArcFace waits on semaphore instead of polling

        Args:
            camera_index: Camera index

        Returns:
            Semaphore instance for ArcFace detection events
        """
        return self.arcface_ready_semaphores[camera_index]

    def get_mediapipe_done_event(self, camera_index: int):
        """
        Get MediaPipe completion event for SCRFD to wait on.

        Args:
            camera_index: Camera index

        Returns:
            Event instance that MediaPipe sets when done processing
        """
        return self.mediapipe_done_events[camera_index]

    def get_arcface_done_event(self, camera_index: int):
        """
        Get ArcFace completion event for SCRFD to wait on.

        Args:
            camera_index: Camera index

        Returns:
            Event instance that ArcFace sets when done processing
        """
        return self.arcface_done_events[camera_index]

    def get_detection_ref_count_lock(self, camera_index: int) -> Lock:
        """
        Get shared lock for atomic ref_count decrement.

        Used by MediaPipe and ArcFace to atomically decrement ref_count
        to prevent race conditions.

        Args:
            camera_index: Camera index

        Returns:
            Lock instance for atomic ref_count operations
        """
        return self.detection_ref_count_locks[camera_index]

    def create_face_recognition_buffers(self, camera_indices: List[int]) -> Dict[str, str]:
        """
        Create command buffers specifically for face recognition process.
        
        Args:
            camera_indices: List of camera indices
            
        Returns:
            Dictionary mapping camera_idx -> command_buffer_name
        """
        import os
        pid = os.getpid()
        
        command_buffers = {}
        
        for camera_idx in camera_indices:
            # Create command buffer for this camera
            buffer_name = f'yq_face_cmd_{camera_idx}_{pid}'
            
            try:
                cmd_buffer = CommandBuffer(
                    name=buffer_name,
                    buffer_size=32  # Optimized size for face recognition commands
                )
                
                self.created_buffers[buffer_name] = cmd_buffer.shm
                command_buffers[camera_idx] = buffer_name
                
                logger.info(f"Created face recognition command buffer: {buffer_name}")
                
            except Exception as e:
                logger.error(f"Failed to create face recognition command buffer for camera {camera_idx}: {e}")
                # Clean up any created buffers
                self._cleanup_partial_face_recognition_buffers(command_buffers)
                raise
        
        return command_buffers
    
    def _cleanup_partial_face_recognition_buffers(self, command_buffers: Dict[int, str]):
        """Clean up partially created face recognition buffers on failure."""
        for camera_idx, buffer_name in command_buffers.items():
            try:
                if buffer_name in self.created_buffers:
                    self.created_buffers[buffer_name].close()
                    self.created_buffers[buffer_name].unlink()
                    del self.created_buffers[buffer_name]
            except Exception as e:
                logger.error(f"Error cleaning up face recognition buffer {buffer_name}: {e}")
    
    def write_coordinator_info_file(self) -> str:
        """
        Write coordinator information to discoverable file.
        
        Returns:
            Path to written info file
        """
        import os
        import json
        import time
        import tempfile
        
        coordinator_info = {
            'pid': os.getpid(),
            'timestamp': time.time(),
            'camera_count': self.camera_count,
            'buffer_config': self.get_buffer_sizes(),
            'buffer_registry': self.buffer_registry.copy(),
            'version': '1.0'
        }
        
        # Write to multiple locations for reliability
        info_paths = []
        
        try:
            # Primary location - Windows compatible
            temp_dir = tempfile.gettempdir()
            primary_path = os.path.join(temp_dir, "youquantipy_coordinator.pid")
            with open(primary_path, 'w') as f:
                json.dump(coordinator_info, f, indent=2)
            info_paths.append(primary_path)
            
            # Secondary location - user home directory (cross-platform)
            home_dir = os.path.expanduser("~")
            backup_dir = os.path.join(home_dir, ".youquantipy")
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, "coordinator.pid")
            with open(backup_path, 'w') as f:
                json.dump(coordinator_info, f, indent=2)
            info_paths.append(backup_path)
            
            logger.info(f"Coordinator info written to: {info_paths}")
            return primary_path
            
        except Exception as e:
            logger.error(f"Failed to write coordinator info file: {e}")
            # Try fallback to current directory
            try:
                fallback_path = os.path.join(os.getcwd(), "youquantipy_coordinator.pid")
                with open(fallback_path, 'w') as f:
                    json.dump(coordinator_info, f, indent=2)
                logger.warning(f"Used fallback coordinator info path: {fallback_path}")
                return fallback_path
            except Exception as fallback_error:
                logger.error(f"Fallback coordinator info write failed: {fallback_error}")
                raise

    def update_coordinator_registry(self, camera_index: int, buffer_names: Dict[str, str]):
        """
        Update coordinator PID file with buffer names (thread-safe with file locking).

        This allows child processes (like MediaPipe) to register their buffer names
        so that other processes (like face recognition) can discover them.

        Args:
            camera_index: Camera index
            buffer_names: Dict of buffer_type -> buffer_name
                         e.g., {'frame': 'yq_frame_0_12345', 'results': 'yq_results_0_12345'}
        """
        import json
        import tempfile
        import os
        import fcntl

        pid_file_path = os.path.join(tempfile.gettempdir(), "youquantipy_coordinator.pid")

        if not os.path.exists(pid_file_path):
            logger.warning(f"Coordinator PID file not found at {pid_file_path}, skipping registry update")
            return

        try:
            with open(pid_file_path, 'r+') as f:
                # Acquire exclusive lock (blocks until available, auto-releases on crash)
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                try:
                    # Read current data
                    f.seek(0)
                    coordinator_info = json.load(f)

                    # Update registry
                    if 'buffer_registry' not in coordinator_info:
                        coordinator_info['buffer_registry'] = {}

                    coordinator_info['buffer_registry'][str(camera_index)] = buffer_names.copy()

                    # Atomic write via temp file + rename
                    temp_path = pid_file_path + '.tmp'
                    with open(temp_path, 'w') as temp_f:
                        json.dump(coordinator_info, temp_f, indent=2)
                        temp_f.flush()
                        os.fsync(temp_f.fileno())

                    os.rename(temp_path, pid_file_path)  # Atomic on POSIX

                    logger.info(f"[BufferCoordinator] Registered buffers for camera {camera_index}: {buffer_names}")

                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        except Exception as e:
            logger.error(f"Failed to update coordinator registry for camera {camera_index}: {e}")
            # Non-fatal - MediaPipe can continue without face recognition

    def _cleanup_coordinator_info_file(self):
        """Clean up coordinator info file on exit."""
        if hasattr(self, 'coordinator_info_file') and self.coordinator_info_file:
            try:
                import os
                if os.path.exists(self.coordinator_info_file):
                    os.remove(self.coordinator_info_file)
                    logger.debug(f"Removed coordinator info file: {self.coordinator_info_file}")
                    
                # Also try to remove backup
                backup_path = "/dev/shm/yq_coordinator.info"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                    
            except Exception as e:
                logger.debug(f"Error removing coordinator info file: {e}")
    
    def get_buffer_sizes(self) -> Dict[str, int]:
        """Get standardized buffer sizes for all processes."""
        return {
            'max_faces': self.max_faces,
            'ring_buffer_size': self.ring_buffer_size,
            'roi_buffer_size': self.roi_buffer_size,
            'gui_buffer_size': self.gui_buffer_size
        }

    def get_camera_resolution(self, camera_index: int) -> Tuple[int, int]:
        """
        Get the actual resolution for a camera.

        Args:
            camera_index: Camera identifier

        Returns:
            Tuple of (width, height), defaults to (1280, 720) if not available
        """
        return self.camera_resolutions.get(camera_index, (1280, 720))

    def get_results_buffer_layout(self) -> Dict[str, int]:
        """
        Get centralized results buffer layout for consistent offset calculations.
        MediaPipe native: landmarks + blendshapes + bbox metadata + frame metadata

        Returns:
            Dictionary with offsets and sizes for landmarks, blendshapes, roi_metadata, and metadata sections

        Note: This method now uses the centralized ResultsBufferLayout dataclass.
              Returns dict for backward compatibility.
        """
        layout = self.get_layout('results')
        return layout.to_dict()

    def get_pose_buffer_layout(self, camera_idx: int) -> Dict[str, int]:
        """
        Get pose buffer layout for MediaPipe pose detection.

        Buffer structure:
        - write_index (8 bytes)
        - pose_data (max_poses × 33 landmarks × 4 values × float32)
        - metadata (64 bytes)

        Returns:
            Dictionary with offsets and sizes for pose buffer

        Note: This method now uses the centralized PoseBufferLayout dataclass.
              Returns dict for backward compatibility.
        """
        layout = self.get_layout('pose')
        return layout.to_dict()

    def create_pose_frame_buffer(self, camera_index: int, resolution: Tuple[int, int] = (640, 480)) -> Dict[str, Any]:
        """
        Create dedicated frame buffer for pose processing at reduced resolution.

        Args:
            camera_index: Camera identifier
            resolution: (width, height) for pose frames (default 640×480)

        Returns:
            Dict with buffer_name, shm object, and layout info
        """
        width, height = resolution
        pid = os.getpid()
        buffer_name = f'yq_pose_frame_{camera_index}_{pid}'

        # Ring buffer configuration
        ring_size = 16  # Match main frame buffer

        # Calculate sizes
        write_index_size = 8  # uint64
        metadata_header_size = 16  # width(4) + height(4) + frame_ready(4) + padding(4)
        frame_size = width * height * 3  # BGR
        slot_metadata_size = 64  # frame_id, timestamp, etc.
        slot_size = frame_size + slot_metadata_size

        total_size = write_index_size + metadata_header_size + (ring_size * slot_size)

        logger.info(f"[BUFFER] Creating pose frame buffer: {buffer_name}")
        logger.info(f"[BUFFER]   Resolution: {width}×{height}")
        logger.info(f"[BUFFER]   Frame size: {frame_size / 1024:.1f} KB")
        logger.info(f"[BUFFER]   Ring slots: {ring_size}")
        logger.info(f"[BUFFER]   Total size: {total_size / 1024 / 1024:.2f} MB")

        # CRITICAL: Clean up any stale buffer from previous crashed runs
        # This prevents EPERM (errno 1) errors when buffer name already exists
        try:
            existing_shm = shared_memory.SharedMemory(name=buffer_name)
            existing_shm.close()
            existing_shm.unlink()
            logger.warning(f"[BUFFER] Cleaned up stale buffer: {buffer_name}")
        except FileNotFoundError:
            # No stale buffer - this is the expected case
            pass
        except Exception as e:
            # Log but continue - creation might still work
            logger.debug(f"[BUFFER] Could not cleanup stale buffer {buffer_name}: {e}")

        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, name=buffer_name, size=total_size)

        # Initialize write index
        write_index_view = np.ndarray((1,), dtype=np.uint64, buffer=shm.buf[0:8])
        write_index_view[0] = 0

        # Initialize metadata header (width, height)
        metadata_view = np.ndarray((4,), dtype=np.uint32, buffer=shm.buf[8:24])
        metadata_view[0] = width
        metadata_view[1] = height
        metadata_view[2] = 0  # frame_ready flag
        metadata_view[3] = 0  # padding

        # Track buffer
        self.created_buffers[buffer_name] = shm

        logger.info(f"✅ [BUFFER] Pose frame buffer created: {buffer_name}")

        return {
            'buffer_name': buffer_name,
            'shm': shm,
            'layout': {
                'write_index_offset': 0,
                'metadata_offset': 8,
                'ring_start_offset': 24,
                'ring_size': ring_size,
                'slot_size': slot_size,
                'frame_size': frame_size,
                'width': width,
                'height': height,
                'total_size': total_size
            }
        }

    def create_detection_buffer(self, camera_index: int, max_faces: int = None) -> Dict[str, Any]:
        """
        Create SCRFD detection buffer with zero-copy crop storage for parallel MediaPipe + ArcFace access.

        Buffer Layout:
        - Header (64B): write_index, ref_count, n_detections, frame_id, timestamp
        - Detection metadata (max_faces × 128B): bbox, keypoints, confidence, detection_index, crop_offset
        - Crop buffer (max_faces × 192×192×3): Zero-copy crops for parallel consumer access

        Args:
            camera_index: Camera identifier
            max_faces: Maximum faces to track (default: from config)

        Returns:
            Dict with buffer_name, shm, layout, and semaphores for parallel consumers
        """
        if max_faces is None:
            max_faces = self.max_faces

        CROP_SIZE = 192 * 192 * 3  # 110,592 bytes per face @ 192×192 RGB
        DETECTION_META_SIZE = 128   # Per-face metadata
        HEADER_SIZE = 64

        total_size = (
            HEADER_SIZE +
            max_faces * DETECTION_META_SIZE +
            max_faces * CROP_SIZE
        )

        pid = os.getpid()
        buffer_name = f'yq_detection_{camera_index}_{pid}'

        logger.info(f"[BUFFER] Creating detection buffer: {buffer_name}")
        logger.info(f"[BUFFER]   Max faces: {max_faces}")
        logger.info(f"[BUFFER]   Crop size per face: {CROP_SIZE / 1024:.1f} KB (192×192 RGB)")
        logger.info(f"[BUFFER]   Total size: {total_size / 1024 / 1024:.2f} MB")

        # Clean up stale buffer
        try:
            existing_shm = shared_memory.SharedMemory(name=buffer_name)
            existing_shm.close()
            existing_shm.unlink()
            logger.warning(f"[BUFFER] Cleaned up stale detection buffer: {buffer_name}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"[BUFFER] Could not cleanup stale buffer {buffer_name}: {e}")

        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, name=buffer_name, size=total_size)

        # Initialize header
        write_index_view = np.ndarray((1,), dtype=np.uint64, buffer=shm.buf[0:8])
        write_index_view[0] = 0

        ref_count_view = np.ndarray((1,), dtype=np.int32, buffer=shm.buf[8:12])
        ref_count_view[0] = 0  # No consumers reading yet

        n_detections_view = np.ndarray((1,), dtype=np.int32, buffer=shm.buf[12:16])
        n_detections_view[0] = 0

        # CRITICAL: Initialize frame_id and timestamp to prevent garbage values
        # These fields are used for frame synchronization in the visualizer
        frame_id_view = np.ndarray((1,), dtype=np.int64, buffer=shm.buf[16:24])
        frame_id_view[0] = -1  # Initialize to -1 (no frame processed yet)

        timestamp_view = np.ndarray((1,), dtype=np.int64, buffer=shm.buf[24:32])
        timestamp_view[0] = 0  # Initialize to 0

        # Track buffer
        self.created_buffers[buffer_name] = shm

        logger.info(f"✅ [BUFFER] Detection buffer created: {buffer_name}")

        layout = {
            'total_size': total_size,
            'header_size': HEADER_SIZE,
            'detection_metadata_offset': HEADER_SIZE,
            'crop_buffer_offset': HEADER_SIZE + max_faces * DETECTION_META_SIZE,
            'crop_size_per_face': CROP_SIZE,
            'detection_meta_size': DETECTION_META_SIZE,
            'max_faces': max_faces,
            'crop_width': 192,
            'crop_height': 192
        }

        # Note: No longer including semaphores in return dict (using Value-based polling)
        return {
            'buffer_name': buffer_name,
            'shm': shm,
            'layout': layout
        }

    def create_embedding_buffer(self, camera_index: int, max_faces: int = None, embedding_dim: int = 512) -> Dict[str, Any]:
        """
        Create ArcFace embedding buffer for participant recognition.

        Buffer Layout:
        - write_index (8B)
        - frame_id (8B)
        - n_embeddings (4B)
        - padding (4B)
        - embeddings (max_faces × embedding_dim × float32)

        Args:
            camera_index: Camera identifier
            max_faces: Maximum faces (default: from config)
            embedding_dim: Embedding dimension (default: 512 for ArcFace)

        Returns:
            Dict with buffer_name, shm, and layout
        """
        if max_faces is None:
            max_faces = self.max_faces

        HEADER_SIZE = 24  # write_index(8) + frame_id(8) + n_embeddings(4) + padding(4)
        EMBEDDING_SIZE = max_faces * embedding_dim * 4  # float32

        total_size = HEADER_SIZE + EMBEDDING_SIZE

        pid = os.getpid()
        buffer_name = f'yq_embedding_{camera_index}_{pid}'

        logger.info(f"[BUFFER] Creating embedding buffer: {buffer_name}")
        logger.info(f"[BUFFER]   Max faces: {max_faces}")
        logger.info(f"[BUFFER]   Embedding dim: {embedding_dim}")
        logger.info(f"[BUFFER]   Total size: {total_size / 1024:.1f} KB")

        # Clean up stale buffer
        try:
            existing_shm = shared_memory.SharedMemory(name=buffer_name)
            existing_shm.close()
            existing_shm.unlink()
            logger.warning(f"[BUFFER] Cleaned up stale embedding buffer: {buffer_name}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"[BUFFER] Could not cleanup stale buffer {buffer_name}: {e}")

        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, name=buffer_name, size=total_size)

        # Initialize header
        write_index_view = np.ndarray((1,), dtype=np.uint64, buffer=shm.buf[0:8])
        write_index_view[0] = 0

        frame_id_view = np.ndarray((1,), dtype=np.int64, buffer=shm.buf[8:16])
        frame_id_view[0] = 0

        n_embeddings_view = np.ndarray((1,), dtype=np.int32, buffer=shm.buf[16:20])
        n_embeddings_view[0] = 0

        # Track buffer
        self.created_buffers[buffer_name] = shm

        logger.info(f"✅ [BUFFER] Embedding buffer created: {buffer_name}")

        return {
            'buffer_name': buffer_name,
            'shm': shm,
            'layout': {
                'total_size': total_size,
                'header_size': HEADER_SIZE,
                'embedding_offset': HEADER_SIZE,
                'embedding_dim': embedding_dim,
                'max_faces': max_faces
            }
        }

    def get_display_buffer_layout(self, camera_idx: int) -> Dict[str, Any]:
        """
        Get display buffer layout for GUI rendering (RING BUFFER).

        RING BUFFER FIX: Display buffer now uses 4-slot ring buffer to decouple
        read/write frequencies. This eliminates polling frequency mismatch that
        caused display freeze (writer at 30 FPS, reader at 60 FPS).

        OPTIMIZATION: Display buffer NO LONGER stores full frame data.
        - Stores only frame_id reference (GUI fetches frame from camera buffer)
        - Reduces buffer size from ~3MB to ~50KB per camera
        - Eliminates 2.7MB memory copy per frame (10x faster Display Write)

        Args:
            camera_idx: Camera index

        Returns:
            Dictionary with all display buffer offsets and sizes
        """
        resolution = self.camera_resolutions.get(camera_idx, (1280, 720))

        # RING BUFFER CONSTANTS
        ring_size = 4  # 4-slot ring buffer for read/write decoupling
        write_index_size = 8  # int64
        frame_metadata_size = 64  # Fixed metadata size (includes frame_id reference)
        face_data_size = self.get_face_data_size()

        # Calculate slot size (metadata + all faces)
        slot_size = frame_metadata_size + (self.max_faces * face_data_size)

        # RING BUFFER LAYOUT: [write_index][slot0][slot1][slot2][slot3]
        layout = {
            'ring_size': ring_size,
            'slot_size': slot_size,
            'write_index_offset': 0,
            'write_index_size': write_index_size,
            'frame_metadata_size': frame_metadata_size,
            # Legacy fields for compatibility (but frame NOT stored)
            'frame_size': 0,  # Frame not stored in display buffer
            'frame_width': resolution[0],
            'frame_height': resolution[1],
            'face_data_size': face_data_size,
            'max_faces': self.max_faces,
            'landmark_count': self.get_landmark_count(),
            'total_size': write_index_size + (ring_size * slot_size)
        }

        single_slot_size = write_index_size + frame_metadata_size + (self.max_faces * face_data_size)
        logger.debug(f"Display buffer layout for camera {camera_idx} (RING BUFFER - 4 slots):")
        logger.debug(f"  Ring size: {ring_size} slots")
        logger.debug(f"  Slot size: {slot_size} bytes")
        logger.debug(f"  Metadata per slot: {frame_metadata_size} bytes")
        logger.debug(f"  Face data size: {face_data_size} bytes per face ({self.max_faces} max)")
        logger.debug(f"  Total size: {layout['total_size']} bytes (single slot would be {single_slot_size} bytes)")

        return layout

    def validate_buffer_layout(self, buffer_name: str, actual_size: int, buffer_type: str) -> bool:
        """
        Validate that a buffer's actual size matches expected layout requirements.

        Args:
            buffer_name: Name of the buffer being validated
            actual_size: Actual size of the buffer in bytes
            buffer_type: Type of buffer ('results' or 'roi')

        Returns:
            True if layout is valid, False otherwise
        """
        try:
            if buffer_type == 'results':
                layout = self.get_results_buffer_layout()
                expected_size = layout['total_size']
                logger.info(f"[BUFFER VALIDATION] {buffer_name} (results): actual={actual_size}, expected={expected_size}")

                if actual_size < expected_size:
                    logger.error(f"[BUFFER VALIDATION] {buffer_name} too small: {actual_size} < {expected_size}")
                    logger.error(f"  Landmarks: offset={layout['landmarks_offset']}, size={layout['landmarks_size']}")
                    logger.error(f"  ROI metadata: offset={layout['roi_metadata_offset']}, size={layout['roi_metadata_size']}")
                    logger.error(f"  Metadata: offset={layout['metadata_offset']}, size={layout['metadata_size']}")
                    return False

            elif buffer_type == 'roi':
                layout = self.get_roi_buffer_layout(0)  # Use camera 0 for validation (buffer size is consistent across cameras)
                expected_size = layout['total_size']
                logger.info(f"[BUFFER VALIDATION] {buffer_name} (roi): actual={actual_size}, expected={expected_size}")

                if actual_size < expected_size:
                    logger.error(f"[BUFFER VALIDATION] {buffer_name} too small: {actual_size} < {expected_size}")
                    logger.error(f"  Write index: offset={layout['write_index_offset']}, size={layout['write_index_size']}")
                    logger.error(f"  ROI data: offset={layout['roi_data_offset']}, size={layout['roi_data_size']}")
                    logger.error(f"  Metadata base: offset={layout['metadata_offset_base']}, per_slot={layout['metadata_per_slot']}")
                    return False

            logger.info(f"[BUFFER VALIDATION] {buffer_name} layout valid ✓")
            return True

        except Exception as e:
            logger.error(f"[BUFFER VALIDATION] Error validating {buffer_name}: {e}")
            return False

    def log_buffer_layout_info(self):
        """Log detailed buffer layout information for debugging."""
        results_layout = self.get_results_buffer_layout()
        roi_layout = self.get_roi_buffer_layout(0)  # Use camera 0 for logging (layout is same for all cameras)

        logger.info("=== BUFFER LAYOUT INFO ===")
        logger.info(f"Max faces: {self.max_faces}")
        logger.info(f"Ring buffer size: {self.ring_buffer_size}")
        logger.info(f"ROI buffer size: {self.roi_buffer_size}")
        logger.info(f"GUI buffer size: {self.gui_buffer_size}")

        logger.info("--- Results Buffer Layout ---")
        for key, value in results_layout.items():
            logger.info(f"  {key}: {value}")

        logger.info("--- ROI Buffer Layout ---")
        for key, value in roi_layout.items():
            logger.info(f"  {key}: {value}")
        logger.info("=== END BUFFER LAYOUT INFO ===")
    
    def create_recovery_buffer(self, camera_index: int) -> str:
        """
        Create participant recovery buffer for a camera.
        
        Returns:
            Shared memory name for recovery buffer
        """
        import os
        pid = os.getpid()
        recovery_name = f'yq_recovery_{camera_index}_{pid}'
        
        try:
            # Create recovery buffer using ParticipantQueryBuffer
            recovery_buffer = ParticipantQueryBuffer(
                name=recovery_name,
                max_queries=16
            )
            
            # Store the buffer object itself for proper access and cleanup
            if not hasattr(self, 'recovery_buffers'):
                self.recovery_buffers = {}
            self.recovery_buffers[recovery_name] = recovery_buffer
            
            # Also store in created_buffers for cleanup tracking
            self.created_buffers[recovery_name] = recovery_buffer.shm
            
            logger.info(f"Created recovery buffer for camera {camera_index}: {recovery_name}")
            return recovery_name
            
        except Exception as e:
            logger.error(f"Failed to create recovery buffer for camera {camera_index}: {e}")
            raise
    
    def create_display_buffer(self, camera_index: int) -> str:
        """
        Create display buffer for GUI rendering.
        
        Returns:
            Shared memory name for display buffer
        """
        import os
        buffer_name = f'yq_display_{camera_index}_{os.getpid()}'
        
        try:
            # Display buffer size calculation
            # Structure: write_index + metadata + face_data_array
            write_index_size = 8  # int64
            frame_metadata_size = 64  # frame_id, timestamp, n_faces, etc.
            max_faces = self.max_faces
            face_data_size = 3856  # Per face (see gui_processing_worker.py)
            
            total_size = (
                write_index_size +
                frame_metadata_size +
                (face_data_size * max_faces)
            )
            
            # Create shared memory
            display_shm = shared_memory.SharedMemory(
                create=True,
                name=buffer_name,
                size=total_size
            )
            
            # Initialize write index to 0
            display_shm.buf[0:8] = struct.pack('Q', 0)
            
            # Store reference
            if not hasattr(self, 'display_buffers'):
                self.display_buffers = {}
            self.display_buffers[camera_index] = display_shm
            self.created_buffers[buffer_name] = display_shm
            
            # Update registry
            if camera_index not in self.buffer_registry:
                self.buffer_registry[camera_index] = {}
            self.buffer_registry[camera_index]['display'] = buffer_name
            
            logger.info(f"Created display buffer for camera {camera_index}: {buffer_name} ({total_size} bytes)")
            return buffer_name
            
        except Exception as e:
            logger.error(f"Failed to create display buffer for camera {camera_index}: {e}")
            raise
    
    def get_display_buffer_info(self) -> Dict[int, Dict[str, Any]]:
        """
        Get display buffer information for all cameras.
        
        Returns:
            Dictionary mapping camera index to buffer info
        """
        display_info = {}
        
        for camera_index, buffers in self.buffer_registry.items():
            if 'display' in buffers:
                display_info[camera_index] = {
                    'buffer_name': buffers['display'],
                    'max_faces': self.max_faces,
                    'write_index_offset': 0,
                    'metadata_offset': 8,
                    'face_data_offset': 72,  # After write index + metadata
                    'face_data_size': 3856
                }
                
        return display_info
    
    def increment_sequence(self) -> int:
        """Thread-safe sequence number increment."""
        with self.sequence_counter.get_lock():
            self.sequence_counter.value += 1
            return self.sequence_counter.value
    
    def get_current_sequence(self) -> int:
        """Get current sequence number."""
        return self.sequence_counter.value
    
    def _cleanup_camera_buffers(self, camera_index: int):
        """Clean up buffers for specific camera."""
        if camera_index not in self.buffer_registry:
            return
            
        buffer_names = self.buffer_registry[camera_index]
        for buffer_type, name in buffer_names.items():
            if name in self.created_buffers:
                try:
                    shm = self.created_buffers[name]
                    shm.close()
                    shm.unlink()
                    del self.created_buffers[name]
                    logger.info(f"Cleaned up {buffer_type} buffer for camera {camera_index}: {name}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup buffer {name}: {e}")
        
        del self.buffer_registry[camera_index]
    
    def cleanup_display_buffer(self, camera_index: int):
        """Clean up display buffer for specific camera."""
        try:
            # Clean up from display_buffers dict
            if hasattr(self, 'display_buffers') and camera_index in self.display_buffers:
                display_shm = self.display_buffers[camera_index]
                display_shm.close()
                display_shm.unlink()
                del self.display_buffers[camera_index]
                logger.info(f"Cleaned up display buffer for camera {camera_index}")
            
            # Clean up from registry if it exists
            if camera_index in self.buffer_registry and 'display' in self.buffer_registry[camera_index]:
                buffer_name = self.buffer_registry[camera_index]['display']
                if buffer_name in self.created_buffers:
                    try:
                        shm = self.created_buffers[buffer_name]
                        shm.close()
                        shm.unlink()
                        del self.created_buffers[buffer_name]
                        logger.info(f"Cleaned up display buffer from registry: {buffer_name}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup display buffer from registry {buffer_name}: {e}")
                del self.buffer_registry[camera_index]['display']
                
        except Exception as e:
            logger.warning(f"Error during display buffer cleanup for camera {camera_index}: {e}")

    def cleanup_preprocessed_buffers(self):
        """Clean up preprocessed buffers."""
        try:
            preprocessed_buffers = [name for name in self.created_buffers if name.startswith('yq_preprocessed_')]
            
            for buffer_name in preprocessed_buffers:
                try:
                    if buffer_name in self.buffer_registry:
                        shm = self.buffer_registry[buffer_name].get('shm')
                        if shm:
                            shm.close()
                            shm.unlink()
                        del self.buffer_registry[buffer_name]
                    self.created_buffers.remove(buffer_name)
                    logger.info(f"Cleaned up preprocessed buffer: {buffer_name}")
                except Exception as e:
                    logger.error(f"Error cleaning up preprocessed buffer {buffer_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in preprocessed buffer cleanup: {e}")
    
    def create_command_buffer(self, source: str, destination: str) -> str:
        """
        Create command buffer between two processes.
        
        Args:
            source: Source process identifier (e.g., 'cam_0', 'detection', 'gui')
            destination: Destination process identifier
            
        Returns:
            Shared memory name for the command buffer
        """
        import os
        pid = os.getpid()
        buffer_name = f'yq_cmd_{source}_{destination}_{pid}'
        
        try:
            command_buffer = CommandBuffer(name=buffer_name, buffer_size=8)
            
            # Store the buffer object itself for proper cleanup
            self.command_buffers = getattr(self, 'command_buffers', {})
            self.command_buffers[buffer_name] = command_buffer
            self.created_buffers[buffer_name] = command_buffer.shm
            
            logger.info(f"Created command buffer {source} -> {destination}: {buffer_name}")
            return buffer_name
            
        except Exception as e:
            logger.error(f"Failed to create command buffer {source} -> {destination}: {e}")
            raise
    
    def create_camera_command_buffers(self, camera_index: int) -> Dict[str, str]:
        """
        Create all command buffers for a camera.
        
        Args:
            camera_index: Camera identifier
            
        Returns:
            Dictionary mapping buffer purpose to shared memory name
        """
        camera_id = f'cam_{camera_index}'
        
        try:
            buffers = {
                'camera_to_detection': self.create_command_buffer(camera_id, 'detection'),
                'detection_to_camera': self.create_command_buffer('detection', camera_id),
                'gui_to_camera': self.create_command_buffer('gui', camera_id),
                'camera_to_gui': self.create_command_buffer(camera_id, 'gui')
            }
            
            # Store in registry for cleanup
            if not hasattr(self, 'command_buffer_registry'):
                self.command_buffer_registry = {}
            self.command_buffer_registry[camera_index] = buffers
            
            logger.info(f"Created command buffers for camera {camera_index}: {list(buffers.keys())}")
            return buffers
            
        except Exception as e:
            logger.error(f"Failed to create command buffers for camera {camera_index}: {e}")
            # Clean up any created buffers
            self._cleanup_camera_command_buffers(camera_index)
            raise
    
    def _cleanup_camera_command_buffers(self, camera_index: int):
        """Clean up command buffers for specific camera."""
        if not hasattr(self, 'command_buffer_registry'):
            return
            
        if camera_index not in self.command_buffer_registry:
            return
        
        buffer_names = self.command_buffer_registry[camera_index]
        for buffer_purpose, buffer_name in buffer_names.items():
            if buffer_name in getattr(self, 'command_buffers', {}):
                try:
                    command_buffer = self.command_buffers[buffer_name]
                    command_buffer.cleanup()
                    del self.command_buffers[buffer_name]
                    
                    if buffer_name in self.created_buffers:
                        del self.created_buffers[buffer_name]
                    
                    logger.info(f"Cleaned up command buffer for camera {camera_index}: {buffer_purpose}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup command buffer {buffer_name}: {e}")
        
        del self.command_buffer_registry[camera_index]

    def _cleanup_stale_buffers(self):
        """
        Clean up stale shared memory buffers from previous runs.

        This prevents buffer size mismatches when configuration changes between runs
        (e.g., resolution or ring_buffer_size changes).

        Removes buffers from dead processes (PID doesn't exist) and logs cleanup actions.
        """
        import os
        import re

        if not os.path.exists('/dev/shm'):
            return

        cleaned_count = 0
        current_pid = os.getpid()

        # logger.info("[BufferCoordinator] Checking for stale shared memory buffers...")

        for filename in os.listdir('/dev/shm'):
            # Only clean YouQuantiPy buffers
            if not filename.startswith('yq_'):
                continue

            try:
                # Extract PID from buffer name (e.g., yq_frame_0_12345 -> 12345)
                match = re.search(r'_(\d+)$', filename)
                if not match:
                    continue

                buffer_pid = int(match.group(1))

                # Skip buffers from current process
                if buffer_pid == current_pid:
                    continue

                # Check if process is still alive
                try:
                    os.kill(buffer_pid, 0)  # Signal 0 = check existence
                    # Process exists, don't clean
                    continue
                except OSError:
                    # Process doesn't exist, clean up buffer
                    filepath = f'/dev/shm/{filename}'
                    try:
                        os.unlink(filepath)
                        cleaned_count += 1
                        logger.debug(f"  Removed stale buffer from dead PID {buffer_pid}: {filename}")
                    except Exception as e:
                        logger.warning(f"  Failed to remove stale buffer {filename}: {e}")

            except Exception as e:
                logger.debug(f"  Error processing {filename}: {e}")
                continue

        if cleaned_count > 0:
            logger.info(f"[BufferCoordinator] Cleaned up {cleaned_count} stale buffer(s)")
        # else:
        #     logger.info("[BufferCoordinator] No stale buffers found")

    def cleanup_all_buffers(self):
        """Clean up all shared memory segments."""
        logger.debug("Cleaning up all BufferCoordinator buffers...")
        
        # Clean up recovery buffers first
        if hasattr(self, 'recovery_buffers'):
            for name, buffer in list(self.recovery_buffers.items()):
                try:
                    buffer.cleanup()
                    logger.info(f"Cleaned up recovery buffer: {name}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup recovery buffer {name}: {e}")
            self.recovery_buffers.clear()
        
        # Clean up preprocessed buffers
        self.cleanup_preprocessed_buffers()
        
        # Clean up command buffers first
        if hasattr(self, 'command_buffer_registry'):
            for camera_index in list(self.command_buffer_registry.keys()):
                self._cleanup_camera_command_buffers(camera_index)
        
        # Clean up any remaining command buffers
        if hasattr(self, 'command_buffers'):
            for name, buffer in list(self.command_buffers.items()):
                try:
                    buffer.cleanup()
                    logger.info(f"Cleaned up remaining command buffer: {name}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup command buffer {name}: {e}")
            self.command_buffers.clear()
        
        # Clean up display buffers
        if hasattr(self, 'display_buffers'):
            for camera_index, display_shm in list(self.display_buffers.items()):
                try:
                    display_shm.close()
                    display_shm.unlink()
                    logger.info(f"Cleaned up display buffer for camera {camera_index}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup display buffer for camera {camera_index}: {e}")
            self.display_buffers.clear()
        
        # Clean up data buffers
        for camera_index in list(self.buffer_registry.keys()):
            self._cleanup_camera_buffers(camera_index)
        
        # Clean up any remaining buffers
        for name, obj in list(self.created_buffers.items()):
            try:
                # Skip non-SharedMemory objects (like dictionaries)
                if hasattr(obj, 'close') and hasattr(obj, 'unlink'):
                    obj.close()
                    obj.unlink()
                    logger.info(f"Cleaned up remaining buffer: {name}")
                else:
                    # Handle dictionary entries (like 'command_buffers')
                    if isinstance(obj, dict):
                        logger.debug(f"Skipping dictionary entry during cleanup: {name}")
                    else:
                        logger.warning(f"Unknown object type in created_buffers: {name} - {type(obj)}")
            except Exception as e:
                logger.warning(f"Failed to cleanup remaining buffer {name}: {e}")
        
        self.created_buffers.clear()
        logger.debug("BufferCoordinator cleanup complete")
    
    def validate_buffer_integrity(self, camera_index: int) -> Dict[str, Any]:
        """Validate buffer integrity and provide diagnostic information."""
        if camera_index not in self.buffer_registry:
            return {'status': 'error', 'message': 'Camera not registered'}
        
        validation_results = {
            'status': 'ok',
            'camera_index': camera_index,
            'validations': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            buffer_names = self.buffer_registry[camera_index]
            
            # Test each buffer
            for buffer_type, buffer_name in buffer_names.items():
                try:
                    # Try to connect to buffer
                    test_shm = shared_memory.SharedMemory(name=buffer_name)
                    
                    validation_results['validations'][buffer_type] = {
                        'name': buffer_name,
                        'size': test_shm.size,
                        'accessible': True
                    }
                    
                    test_shm.close()
                    
                except Exception as e:
                    validation_results['errors'].append(f"{buffer_type} buffer '{buffer_name}' not accessible: {e}")
                    validation_results['validations'][buffer_type] = {
                        'name': buffer_name,
                        'accessible': False,
                        'error': str(e)
                    }
            
            # Check buffer sizes consistency
            expected_sizes = self.get_buffer_sizes()
            validation_results['buffer_sizes'] = expected_sizes
            
        except Exception as e:
            validation_results['status'] = 'error'
            validation_results['errors'].append(f"Validation failed: {e}")
        
        if validation_results['errors']:
            validation_results['status'] = 'error'
        elif validation_results['warnings']:
            validation_results['status'] = 'warning'
            
        return validation_results
    
    def get_memory_usage_mb(self) -> float:
        """Calculate total memory usage in MB."""
        total_bytes = 0
        for name, shm in self.created_buffers.items():
            total_bytes += shm.size
        return total_bytes / (1024 * 1024)
    
    def update_face_count_stats(self, camera_index: int, face_count: int):
        """Update face count statistics for adaptive sizing."""
        if camera_index not in self.face_count_history:
            self.face_count_history[camera_index] = []
        
        history = self.face_count_history[camera_index]
        history.append(face_count)
        
        # Keep only recent history (last 100 frames)
        if len(history) > 100:
            history.pop(0)
    
    def get_adaptive_buffer_size(self, camera_index: int, buffer_type: str) -> int:
        """
        Calculate optimal buffer size based on actual usage patterns.
        
        Args:
            camera_index: Camera identifier
            buffer_type: Type of buffer ('ring', 'roi', 'gui')
            
        Returns:
            Optimized buffer size
        """
        base_sizes = {
            'ring': self.ring_buffer_size,
            'roi': self.roi_buffer_size,
            'gui': self.gui_buffer_size
        }
        
        base_size = base_sizes.get(buffer_type, self.ring_buffer_size)
        
        # If no adaptive sizing enabled, return base size
        if not self.config.get('process_separation', {}).get('enable_adaptive_batch_size', False):
            return base_size
        
        # Get face count history for this camera
        history = self.face_count_history.get(camera_index, [1])
        if not history:
            return base_size
        
        # Calculate statistics
        avg_faces = sum(history) / len(history)
        max_faces = max(history)
        
        # Adjust buffer size based on face detection patterns
        if buffer_type == 'roi':
            # ROI buffer should scale with face count
            if avg_faces < 2:
                return max(4, base_size // 2)  # Smaller for single face
            elif avg_faces > 4:
                return min(32, base_size * 2)  # Larger for many faces
        
        elif buffer_type == 'ring':
            # Ring buffer scales with detection frequency
            if max_faces > 6:
                return min(32, base_size * 2)  # More buffering for complex scenes
            elif max_faces < 2:
                return max(8, base_size // 2)  # Less buffering for simple scenes
        
        elif buffer_type == 'gui':
            # GUI buffer can be smaller for single face scenarios
            if avg_faces < 2:
                return max(4, base_size // 2)
        
        return base_size
    
    def get_buffer_layout(self, camera_index: int, resolution: Tuple[int, int]) -> Dict:
        """
        Provide authoritative buffer layout for all processes.
        This ensures consistent buffer access across all components.
        Includes write index at offset 0 for direct access.
        Architecture B: Detection-only layout (no integrated GUI sections)
        """
        # Create cache key from parameters
        cache_key = (camera_index, resolution[0], resolution[1])

        # Initialize cache if needed
        if not hasattr(self, '_layout_cache'):
            self._layout_cache = {}

        # Return cached layout if available
        if cache_key in self._layout_cache:
            return self._layout_cache[cache_key]

        frame_w, frame_h = resolution
        frame_size = frame_w * frame_h * 3
        header_size = 16  # Header: [write_index(8)][width(4)][height(4)]

        # Detection ring layout (after header)
        detection_frame_offsets = [header_size + i * frame_size for i in range(self.ring_buffer_size)]
        detection_metadata_start = header_size + frame_size * self.ring_buffer_size
        detection_metadata_offsets = [detection_metadata_start + i * 64 for i in range(self.ring_buffer_size)]

        # Total size is just detection ring + metadata (no GUI sections)
        total_size = detection_metadata_start + 64 * self.ring_buffer_size

        # Only log on first calculation (not on cache hits)
        logger.info(f"Buffer layout calculated and cached for camera {camera_index} ({frame_w}x{frame_h}):")
        logger.info(f"  Frame size: {frame_size}, Total: {total_size}")
        # Removed repetitive detailed logging

        layout = {
            'write_index_offset': 0,  # Write index always at start
            'resolution_offset': 8,  # Resolution info starts at byte 8
            'header_size': header_size,  # Total header size
            'frame_size': frame_size,
            'detection_frame_offsets': detection_frame_offsets,
            'detection_metadata_offsets': detection_metadata_offsets,
            # GUI sections removed for Architecture B
            'total_size': total_size
        }

        # Cache the layout
        self._layout_cache[cache_key] = layout

        return layout
    
    def get_roi_buffer_layout(self, camera_idx: int) -> Dict[str, Any]:
        """
        Get ROI buffer layout with offsets for centralized access.
        Ensures consistency between writer (detection process) and readers.

        ROI size depends on preprocessing mode:
        - Preprocessing disabled: 256×256×3 uint8 = 196,608 bytes/ROI
        - Preprocessing enabled: 256×256×3 float32 = 786,432 bytes/ROI (4× larger)

        Note: This method now uses the centralized ROIBufferLayout dataclass.
              Returns dict for backward compatibility.
        """
        # Get layout from centralized dataclass
        layout = self.get_layout('roi')

        # Enhanced debug logging for ROI buffer layout
        logger.info(f"[ROI BUFFER LAYOUT] camera={camera_idx}")
        logger.info(f"[ROI BUFFER LAYOUT] preprocessing={layout.enable_preprocessing}")
        logger.info(f"[ROI BUFFER LAYOUT] max_faces={layout.max_faces}")
        logger.info(f"[ROI BUFFER LAYOUT] roi_buffer_size={layout.roi_buffer_size}")
        logger.info(f"[ROI BUFFER LAYOUT] roi_size={layout.roi_size} bytes/ROI")
        logger.info(f"[ROI BUFFER LAYOUT] roi_data_size={layout.roi_data_size}")
        logger.info(f"[ROI BUFFER LAYOUT] metadata_size={layout.metadata_per_slot} per slot")
        logger.info(f"[ROI BUFFER LAYOUT] total_size={layout.total_size} bytes ({layout.total_size/1024/1024:.2f} MB)")

        return layout.to_dict()

    def get_preprocessed_buffer_layout(self, camera_idx: int) -> Dict[str, Any]:
        """
        Get preprocessed buffer layout with offsets for centralized access.
        """
        preprocessed_tensor_size = 256 * 256 * 3 * 4  # float32 preprocessed tensors
        metadata_size = 1024  # Same metadata structure

        # Calculate total preprocessed data size
        preprocessed_data_size = preprocessed_tensor_size * self.max_faces * self.roi_buffer_size

        return {
            'write_index_offset': 0,  # Write index always at offset 0
            'write_index_size': 8,    # Write index size (int64)
            'preprocessed_data_offset': 8,  # Preprocessed data starts after write index
            'preprocessed_tensor_size': preprocessed_tensor_size,
            'preprocessed_data_size': preprocessed_data_size,  # Total preprocessed data size
            'preprocessed_buffer_size': self.roi_buffer_size,  # Reuse roi_buffer_size
            'metadata_offset_base': 8 + preprocessed_data_size,  # Base metadata offset
            'metadata_per_slot': metadata_size,  # Metadata size per slot
            'metadata_offset': 8 + preprocessed_data_size,  # Keep for backward compatibility
            'metadata_size': metadata_size,
            'total_size': 8 + preprocessed_data_size + (self.roi_buffer_size * metadata_size)
        }
    
    def get_memory_optimization_recommendations(self) -> Dict[str, Any]:
        """Generate memory optimization recommendations based on usage patterns."""
        recommendations = {
            'current_usage_mb': self.get_memory_usage_mb(),
            'camera_recommendations': {}
        }
        
        total_potential_savings = 0
        
        for camera_index in self.buffer_registry.keys():
            camera_rec = {
                'current_sizes': {
                    'ring': self.ring_buffer_size,
                    'roi': self.roi_buffer_size,
                    'gui': self.gui_buffer_size
                },
                'recommended_sizes': {},
                'potential_savings_mb': 0
            }
            
            # Calculate recommended sizes
            for buffer_type in ['ring', 'roi', 'gui']:
                current_size = camera_rec['current_sizes'][buffer_type]
                recommended_size = self.get_adaptive_buffer_size(camera_index, buffer_type)
                camera_rec['recommended_sizes'][buffer_type] = recommended_size
                
                # Calculate potential memory savings (rough estimate)
                # Use actual camera resolution instead of hardcoded 1080p
                resolution = self.get_camera_resolution(camera_index)
                frame_size = resolution[0] * resolution[1] * 3

                if buffer_type == 'ring':
                    savings = (current_size - recommended_size) * frame_size
                elif buffer_type == 'roi':
                    roi_size = 256 * 256 * 3 * self.max_faces
                    savings = (current_size - recommended_size) * roi_size
                else:  # gui
                    savings = (current_size - recommended_size) * frame_size
                
                if savings > 0:
                    camera_rec['potential_savings_mb'] += savings / (1024 * 1024)
            
            total_potential_savings += camera_rec['potential_savings_mb']
            recommendations['camera_recommendations'][camera_index] = camera_rec
        
        recommendations['total_potential_savings_mb'] = total_potential_savings
        recommendations['optimization_enabled'] = self.config.get('process_separation', {}).get('enable_adaptive_batch_size', False)
        
        return recommendations
    
    def create_command_buffer(self, source: str, destination: str) -> str:
        """
        Create command buffer between two processes.
        
        Args:
            source: Source process identifier (e.g., 'cam_0', 'gui')
            destination: Destination process identifier (e.g., 'detection', 'cam_0')
            
        Returns:
            Shared memory name for the command buffer
        """
        import os
        buffer_name = f"yq_cmd_{source}_{destination}_{os.getpid()}"
        
        try:
            # Create CommandBuffer through existing system
            command_buffer = CommandBuffer(name=buffer_name, buffer_size=32)
            
            # Track for cleanup
            if 'command_buffers' not in self.created_buffers:
                self.created_buffers['command_buffers'] = {}
            self.created_buffers['command_buffers'][buffer_name] = command_buffer
            
            logger.info(f"Created command buffer: {buffer_name} ({source} -> {destination})")
            return buffer_name
            
        except Exception as e:
            logger.error(f"Failed to create command buffer {buffer_name}: {e}")
            raise
    
    def create_camera_command_buffers(self, camera_index: int) -> Dict[str, str]:
        """
        Create all command buffers for a camera.
        
        Args:
            camera_index: Camera index
            
        Returns:
            Dictionary mapping buffer types to shared memory names
        """
        try:
            buffers = {}
            
            # Camera to detection process command buffer
            buffers['camera_to_detection'] = self.create_command_buffer(
                f'cam_{camera_index}', 'detection'
            )
            
            # Detection to camera process response buffer (optional for future use)
            buffers['detection_to_camera'] = self.create_command_buffer(
                'detection', f'cam_{camera_index}'
            )
            
            # GUI to camera command buffer (for control commands)
            buffers['gui_to_camera'] = self.create_command_buffer(
                'gui', f'cam_{camera_index}'
            )
            
            # Camera to GUI response buffer (for status updates)
            buffers['camera_to_gui'] = self.create_command_buffer(
                f'cam_{camera_index}', 'gui'
            )
            
            logger.info(f"Created command buffers for camera {camera_index}: {list(buffers.keys())}")
            return buffers
            
        except Exception as e:
            logger.error(f"Failed to create command buffers for camera {camera_index}: {e}")
            raise
    
    def cleanup_command_buffers(self):
        """Cleanup all command buffers created by this coordinator."""
        try:
            command_buffers = self.created_buffers.get('command_buffers', {})
            
            for buffer_name, command_buffer in command_buffers.items():
                try:
                    command_buffer.cleanup()
                    logger.info(f"Cleaned up command buffer: {buffer_name}")
                except Exception as e:
                    logger.error(f"Failed to cleanup command buffer {buffer_name}: {e}")
            
            # Clear the registry
            if 'command_buffers' in self.created_buffers:
                del self.created_buffers['command_buffers']
                
        except Exception as e:
            logger.error(f"Error during command buffer cleanup: {e}")
    
    def discover_roi_buffers(self, camera_indices: List[int]) -> Dict[int, str]:
        """
        Discover ROI buffer names for face recognition processes.
        
        This method helps face recognition processes find ROI buffers created
        by detection processes, even when PIDs don't match.
        
        Args:
            camera_indices: List of camera indices to find buffers for
            
        Returns:
            Dictionary mapping camera_index -> buffer_name
        """
        import glob
        import os
        
        discovered_buffers = {}
        
        for camera_idx in camera_indices:
            try:
                # Try to find ROI buffer by pattern matching
                pattern = f"yq_roi_{camera_idx}_*"
                
                # In Linux/WSL, shared memory is in /dev/shm/
                if os.path.exists('/dev/shm'):
                    shm_files = glob.glob(f'/dev/shm/{pattern}')
                    if shm_files:
                        # Take the most recent one (by modification time)
                        latest_file = max(shm_files, key=os.path.getmtime)
                        buffer_name = os.path.basename(latest_file)
                        discovered_buffers[camera_idx] = buffer_name
                        logger.info(f"Discovered ROI buffer for camera {camera_idx}: {buffer_name}")
                        continue
                
                # Fallback: try registered buffers
                if camera_idx in self.buffer_registry:
                    roi_buffer_name = self.buffer_registry[camera_idx].get('roi')
                    if roi_buffer_name:
                        # Test if buffer still exists
                        try:
                            test_shm = shared_memory.SharedMemory(name=roi_buffer_name)
                            test_shm.close()
                            discovered_buffers[camera_idx] = roi_buffer_name
                            logger.info(f"Found registered ROI buffer for camera {camera_idx}: {roi_buffer_name}")
                            continue
                        except:
                            pass
                
                logger.warning(f"Could not discover ROI buffer for camera {camera_idx}")
                
            except Exception as e:
                logger.error(f"Error discovering ROI buffer for camera {camera_idx}: {e}")
        
        return discovered_buffers
    
    def get_roi_buffer_info(self, camera_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get ROI buffer information for a specific camera.
        
        Args:
            camera_idx: Camera index
            
        Returns:
            Dictionary with ROI buffer information or None if not found
        """
        try:
            if camera_idx not in self.buffer_registry:
                return None
            
            roi_buffer_name = self.buffer_registry[camera_idx].get('roi')
            if not roi_buffer_name:
                return None
            
            # Calculate buffer layout information
            roi_size = 256 * 256 * 3  # 256x256 RGB image
            total_roi_space = self.roi_buffer_size * roi_size * self.max_faces
            metadata_space = 1024  # Space for metadata
            
            return {
                'buffer_name': roi_buffer_name,
                'max_faces': self.max_faces,
                'roi_buffer_size': self.roi_buffer_size,
                'roi_size': roi_size,
                'total_roi_space': total_roi_space,
                'metadata_offset': total_roi_space,
                'metadata_space': metadata_space,
                'total_buffer_size': total_roi_space + metadata_space
            }
            
        except Exception as e:
            logger.error(f"Error getting ROI buffer info for camera {camera_idx}: {e}")
            return None

