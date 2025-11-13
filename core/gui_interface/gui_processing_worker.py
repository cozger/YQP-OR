"""
GUI Processing Worker - Dedicated process for all GUI-related data processing.
Separates complex processing logic from display rendering to achieve 30+ FPS.
"""

import multiprocessing as mp
import queue  # For queue.Full exception handling
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass
import struct
import os
import sys

# CRITICAL FIX: Configure logging at MODULE level (before any imports that might fail)
# This ensures import errors are visible even if worker crashes during module load
# Without this, worker crashes are silent (no logs, no error messages)
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s',
    stream=sys.stderr,
    force=True
)

logger = logging.getLogger('GUIProcessingWorker')
sys.stderr.flush()

logger.info("[WORKER MODULE] Loading gui_processing_worker.py...")
sys.stderr.flush()

# Wrap project imports with error handling to expose import failures
try:
    from ..buffer_management.coordinator import BufferCoordinator
    from ..buffer_management.sharedbuffer import AtomicRingBuffer, CommandBuffer
    from ..participant_management import SingleParticipantManager
    from ..process_management.confighandler import ConfigHandler
    from ..pose_processing.pose_smoother import PoseLandmarkSmoother
    from ..visualization.rtmpose_visualizer import RTMPoseVisualizer
    from .coordinate_transform import CoordinateSystem, create_coordinate_system
    from ..metrics import PoseMetricsCalculator, MetricsConfig
    logger.info("[WORKER MODULE] ✅ All imports successful")
    sys.stderr.flush()
except Exception as import_error:
    logger.error(f"[WORKER MODULE] ❌ FATAL: Import failed: {import_error}")
    import traceback
    logger.error(traceback.format_exc())
    sys.stderr.flush()
    raise  # Re-raise to prevent worker from starting with broken imports


# Display flags bit encoding for enrollment states
# Used to determine overlay colors in GUI without querying enrollment manager
DISPLAY_FLAG_ENROLLED = 0x01       # Bit 0: Fully enrolled (green overlay)
DISPLAY_FLAG_COLLECTING = 0x02     # Bit 1: Collecting samples (yellow overlay)
DISPLAY_FLAG_VALIDATING = 0x04     # Bit 2: Validating (orange overlay)
DISPLAY_FLAG_FAILED = 0x08         # Bit 3: Enrollment failed (red overlay)


@dataclass
class DisplayFaceData:
    """Pre-computed face data ready for display."""
    face_id: int
    participant_id: int
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in screen coords
    landmarks: Optional[np.ndarray]  # 478x3 screen x,y,z coordinates
    label: str  # Pre-formatted label like "P1 ✓ (E)"
    confidence: float
    display_flags: int  # Bit flags for enrollment state (DISPLAY_FLAG_*)


class DisplayBuffer:
    """
    Specialized shared memory buffer for display-ready data.
    Optimized for fast GUI rendering with zero processing needed.
    Now includes frame data for complete display buffer.
    """

    def __init__(self, camera_idx: int, buffer_name: str, buffer_coordinator, create: bool = False):
        self.camera_idx = camera_idx
        self.buffer_name = buffer_name
        self.coordinator = buffer_coordinator

        # Get layout from BufferCoordinator (single source of truth)
        layout = self.coordinator.get_display_buffer_layout(camera_idx)

        # Set all constants from layout
        self.RING_SIZE = layout['ring_size']
        self.SLOT_SIZE = layout['slot_size']
        self.WRITE_INDEX_SIZE = layout['write_index_size']
        self.FRAME_METADATA_SIZE = layout['frame_metadata_size']
        self.MAX_FACES = layout['max_faces']
        self.FACE_DATA_SIZE = layout['face_data_size']
        self.MAX_LABEL_LENGTH = 64  # Still constant
        self.FRAME_WIDTH = layout['frame_width']
        self.FRAME_HEIGHT = layout['frame_height']
        self.frame_size = layout['frame_size']
        self.landmark_count = layout['landmark_count']

        # Calculate total buffer size including frame data
        self.total_size = layout['total_size']

        # Ring buffer write position (rotates 0 → 1 → 2 → 3 → 0)
        self.write_position = 0

        if create:
            # Create new shared memory buffer
            self.shm = mp.shared_memory.SharedMemory(
                name=buffer_name,
                create=True,
                size=self.total_size
            )
            # LIFECYCLE LOG: Track shared memory creation for debugging buffer issues
            logger.info(f"[SHM LIFECYCLE] Created display buffer: {buffer_name}, size={self.total_size}, pid={mp.current_process().pid}, camera={camera_idx}")
            # Initialize write index
            self.shm.buf[0:8] = struct.pack('Q', 0)
        else:
            # Connect to existing buffer
            self.shm = mp.shared_memory.SharedMemory(name=buffer_name)
            # LIFECYCLE LOG: Track shared memory connection for debugging buffer issues
            logger.info(f"[SHM LIFECYCLE] Connected to display buffer: {buffer_name}, pid={mp.current_process().pid}, camera={camera_idx}")

        # Create numpy view for write index
        self.write_index_view = np.ndarray(
            (1,), dtype=np.uint64, buffer=self.shm.buf[0:8]
        )

        # CRITICAL FIX: Track numpy views for proper cleanup to prevent BufferError
        # BufferError occurs when SharedMemory.close() is called while numpy views still exist
        # We must delete all views before closing shared memory
        self.active_views = [self.write_index_view]

        # Monotonic write counter for display buffer
        # ALWAYS increments on write, ensuring write_index always changes
        # Ring buffer allows reader to scan slots and find latest frame_id
        self.write_counter = 0

    def write_display_data(self, frame_id: int, timestamp: float,
                          faces: List[DisplayFaceData], frame_bgr: np.ndarray) -> bool:
        """
        Write display-ready data to ring buffer slot.

        RING BUFFER FIX: Writes to rotating slots (0→1→2→3→0) to decouple
        read/write frequencies. GUI can poll at any rate and find latest frame_id.

        OPTIMIZATION: Display buffer NO LONGER stores full frame data.
        - Stores only frame_id reference (GUI fetches frame from camera buffer)
        - Eliminates 2.7MB memory copy per frame
        - Reduces Display Write time from 2.16ms to 0.2ms (10x faster)
        """
        try:
            # CRITICAL FIX: Validate frame_id before packing
            # Negative frame_id (-1) means no valid frame yet
            # Convert to 0 to allow buffer write to succeed during initialization
            if frame_id < 0:
                # Use 0 as placeholder for invalid frame_id
                # This allows GUI to initialize properly even before frames arrive
                if not hasattr(self, '_invalid_frame_logged') or not self._invalid_frame_logged.get(self.camera_idx, False):
                    logger.debug(f"[DisplayBuffer] Camera {self.camera_idx}: Invalid frame_id={frame_id}, using 0 as placeholder")
                    if not hasattr(self, '_invalid_frame_logged'):
                        self._invalid_frame_logged = {}
                    self._invalid_frame_logged[self.camera_idx] = True
                frame_id = 0  # Use 0 instead of -1 to satisfy unsigned format

            # RING BUFFER: Calculate offset for current write position
            # Layout: [write_index(8)][slot0][slot1][slot2][slot3]
            slot_offset = self.WRITE_INDEX_SIZE + (self.write_position * self.SLOT_SIZE)

            # Prepare metadata (frame_id is the reference to camera buffer frame)
            metadata = struct.pack(
                'QdI44x',  # frame_id (unsigned), timestamp, n_faces, padding
                frame_id,
                timestamp,
                len(faces)
            )

            # Bounds check for metadata write
            metadata_offset = slot_offset
            if metadata_offset + self.FRAME_METADATA_SIZE > self.shm.size:
                logger.error(f"[RING BUFFER OVERFLOW] Camera {self.camera_idx} metadata write would exceed buffer: "
                           f"slot={self.write_position}, offset={metadata_offset}, size={self.FRAME_METADATA_SIZE}, "
                           f"buffer_size={self.shm.size}, frame_id={frame_id}")
                return False

            # Write metadata to current ring slot
            self.shm.buf[metadata_offset:metadata_offset + self.FRAME_METADATA_SIZE] = metadata

            # OPTIMIZATION: Skip frame write entirely - GUI reads directly from camera buffer
            # Face data now written immediately after metadata (no frame storage)
            face_offset = metadata_offset + self.FRAME_METADATA_SIZE

            # Bounds check for face data writes
            total_face_size = len(faces[:self.MAX_FACES]) * self.FACE_DATA_SIZE
            if face_offset + total_face_size > self.shm.size:
                logger.error(f"[RING BUFFER OVERFLOW] Camera {self.camera_idx} face data write would exceed buffer: "
                           f"slot={self.write_position}, offset={face_offset}, total_face_size={total_face_size}, "
                           f"n_faces={len(faces)}, buffer_size={self.shm.size}, frame_id={frame_id}")
                return False

            for i, face in enumerate(faces[:self.MAX_FACES]):
                # Pack face data
                face_bytes = self._pack_face_data(face)
                start = face_offset + (i * self.FACE_DATA_SIZE)
                end = start + self.FACE_DATA_SIZE

                # Additional per-face bounds check
                if end > self.shm.size:
                    logger.error(f"[RING BUFFER OVERFLOW] Camera {self.camera_idx} face {i} write would exceed buffer: "
                               f"slot={self.write_position}, start={start}, end={end}, "
                               f"buffer_size={self.shm.size}, frame_id={frame_id}")
                    return False

                self.shm.buf[start:end] = face_bytes

            # RING BUFFER: Advance write position (wraps around at RING_SIZE)
            self.write_position = (self.write_position + 1) % self.RING_SIZE

            # Update write index with monotonic counter (signals new data available)
            # Increments on every write - GUI scans ring slots to find latest frame_id
            self.write_counter += 1
            self.write_index_view[0] = self.write_counter

            return True

        except Exception as e:
            # Enhanced error logging with buffer state
            logger.error(f"[FATAL] Error writing display data for camera {self.camera_idx}: {e}")
            logger.error(f"[FATAL] Buffer state: size={self.shm.size}, total_size={self.total_size}, "
                        f"write_position={self.write_position}, frame_id={frame_id}, n_faces={len(faces)}")
            logger.error(f"[FATAL] Layout: RING_SIZE={self.RING_SIZE}, SLOT_SIZE={self.SLOT_SIZE}, "
                        f"WRITE_INDEX_SIZE={self.WRITE_INDEX_SIZE}, FRAME_METADATA_SIZE={self.FRAME_METADATA_SIZE}, "
                        f"frame_size={self.frame_size}, FACE_DATA_SIZE={self.FACE_DATA_SIZE}")
            import traceback
            traceback.print_exc()
            return False
            
    def _pack_face_data(self, face: DisplayFaceData) -> bytes:
        """Pack face data into bytes for shared memory."""
        # Start with fixed-size fields (removed obsolete is_tracking field)
        # Format: 3 SIGNED ints (face_id, participant_id, track_id) + 5 floats (bbox×4, confidence) + 1 unsigned int (display_flags)
        # NOTE: Changed from 'I' to 'i' for first 3 fields to support -1 sentinel values
        packed = struct.pack(
            'iiifffffI',
            face.face_id,
            face.participant_id,
            face.track_id,
            face.bbox[0], face.bbox[1], face.bbox[2], face.bbox[3],
            face.confidence,
            face.display_flags
        )
        
        # Add label (fixed 64 bytes)
        label_bytes = face.label.encode('utf-8')[:self.MAX_LABEL_LENGTH]
        label_bytes += b'\x00' * (self.MAX_LABEL_LENGTH - len(label_bytes))
        packed += label_bytes
        
        # Add landmarks if available (478 * 3 * 4 = 5736 bytes for 3D coordinates)
        if face.landmarks is not None and hasattr(face.landmarks, 'shape'):
            # Ensure correct shape and type (478x3 for MediaPipe)
            if len(face.landmarks.shape) >= 2 and face.landmarks.shape[0] == self.landmark_count:
                if face.landmarks.shape[1] == 2:
                    # If we have 2D landmarks, pad with zeros for z-coordinate
                    landmarks_3d = np.zeros((self.landmark_count, 3), dtype=np.float32)
                    landmarks_3d[:, :2] = face.landmarks
                    landmarks_flat = landmarks_3d.reshape(-1)
                elif face.landmarks.shape[1] == 3:
                    # Already 3D
                    landmarks_flat = face.landmarks.reshape(-1).astype(np.float32)
                else:
                    # Unexpected shape - fill with zeros
                    packed += b'\x00' * (self.landmark_count * 3 * 4)
                    logger.warning(f"Unexpected landmark shape: {face.landmarks.shape}")
                if 'landmarks_flat' in locals():
                    packed += landmarks_flat.tobytes()
            else:
                # Invalid shape - fill with zeros
                packed += b'\x00' * (self.landmark_count * 3 * 4)
        else:
            # No landmarks - fill with zeros
            packed += b'\x00' * (self.landmark_count * 3 * 4)
            
        # Add padding to reach FACE_DATA_SIZE
        current_size = len(packed)
        if current_size < self.FACE_DATA_SIZE:
            packed += b'\x00' * (self.FACE_DATA_SIZE - current_size)
            
        return packed[:self.FACE_DATA_SIZE]  # Ensure exact size
        
    def cleanup(self):
        """Clean up shared memory.

        CRITICAL FIX: Delete numpy views BEFORE closing shared memory to prevent BufferError.
        BufferError is raised when SharedMemory.close() is called while numpy views still exist.
        This can cause silent process crashes if it occurs in __del__() or background cleanup.
        """
        try:
            # Step 1: Delete all numpy views to release references to shared memory buffer
            if hasattr(self, 'active_views'):
                for view in self.active_views:
                    try:
                        del view
                    except Exception as e:
                        logger.warning(f"[DisplayBuffer] Error deleting view: {e}")
                self.active_views.clear()

            # Step 2: Delete the write_index_view explicitly (may still be referenced)
            if hasattr(self, 'write_index_view'):
                try:
                    del self.write_index_view
                except Exception as e:
                    logger.warning(f"[DisplayBuffer] Error deleting write_index_view: {e}")

            # Step 3: Now it's safe to close shared memory
            if hasattr(self, 'shm'):
                buffer_name = self.shm.name if hasattr(self.shm, 'name') else 'unknown'
                self.shm.close()
                # LIFECYCLE LOG: Track shared memory close for debugging buffer issues
                logger.info(f"[SHM LIFECYCLE] Closed display buffer: {buffer_name}, pid={mp.current_process().pid}, camera={self.camera_idx}")

        except Exception as e:
            logger.error(f"[DisplayBuffer] Error during cleanup for camera {self.camera_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())


class GUIProcessingWorker:
    """
    Dedicated process for all GUI-related data processing.
    Reads from detection/landmark buffers, produces display-ready data.
    """
    
    def __init__(self, buffer_coordinator: BufferCoordinator,
                 config: Dict[str, Any],
                 camera_indices: List[int],
                 recovery_buffer_names: Dict[int, str],
                 control_queue: mp.Queue,
                 status_queue: mp.Queue,
                 participant_update_queue: mp.Queue,
                 enrollment_state_array: 'multiprocessing.sharedctypes.SynchronizedArray',
                 lock_state_array: 'multiprocessing.sharedctypes.SynchronizedArray',
                 participant_presence_array: 'multiprocessing.sharedctypes.SynchronizedArray',
                 enrollment_states_mapping: Dict[str, int]):
        """
        Initialize the GUI processing worker.

        Args:
            buffer_coordinator: Centralized buffer management (includes actual_buffer_names)
            config: Configuration dictionary
            camera_indices: List of active camera indices
            recovery_buffer_names: Recovery buffer names for each camera
            control_queue: Queue for receiving control commands
            status_queue: Queue for sending status updates
            participant_update_queue: Queue for enrollment state updates (event-driven)
            enrollment_state_array: Shared array for reading enrollment state from GUI process [participant_id] = state_code
            lock_state_array: Shared array for reading lock state from GUI process [participant_id] = 0/1
            participant_presence_array: Shared array for writing presence state to GUI process [participant_id] = 0/1
            enrollment_states_mapping: Dict mapping state names to integer codes
        """
        self.buffer_coordinator = buffer_coordinator
        self.config = config
        self.camera_indices = camera_indices
        self.recovery_buffer_names = recovery_buffer_names
        self.control_queue = control_queue
        self.status_queue = status_queue
        self.participant_update_queue = participant_update_queue

        # ARRAY-BASED IPC: Shared arrays instead of Manager dicts
        self.enrollment_state_array = enrollment_state_array  # Shared enrollment state from GUI process
        self.lock_state_array = lock_state_array  # Shared lock state from GUI process
        self.participant_presence_array = participant_presence_array  # Shared presence state to GUI process

        # State name <-> code mappings
        self.ENROLLMENT_STATES = enrollment_states_mapping
        self.ENROLLMENT_STATES_REVERSE = {v: k for k, v in enrollment_states_mapping.items()}
        
        # Extract actual buffer names from coordinator info
        # NOTE: actual_buffer_names will be set after initialization by worker_main
        self.actual_buffer_names = {}

        # Track actual camera resolutions for buffer calculations
        # NOTE: camera_resolutions will be set after initialization by worker_main
        self.camera_resolutions = {}

        # Coordinate system for landmark transformation (crop → frame → canvas)
        # Will be initialized per-camera when resolution is known
        self.coordinate_systems = {}  # {cam_idx: CoordinateSystem}

        # Processing components (moved from GUI)
        # Use centralized buffer_settings configuration
        self.max_participants = config.get('buffer_settings', {}).get('persons', {}).get('participant_count', 1)  # Single participant mode

        # Initialize unified participant manager (replaces GlobalParticipantManager + GridParticipantManager)
        self.participant_manager = SingleParticipantManager(
            config=config,
            enable_query_buffer=False,  # Query buffer managed separately if needed
            face_recognition_callback=None  # No face recognition in worker process
        )

        # Legacy attributes for backward compatibility
        self.global_participant_manager = self.participant_manager

        # ARCHITECTURAL CONSOLIDATION: EnrollmentManager removed from worker process
        # Enrollment is now ONLY managed by GUI process (GlobalParticipantManager.enrollment_manager)
        # This worker sends embedding samples to GUI for enrollment processing
        # OLD CODE REMOVED:
        # self.enrollment_manager = EnrollmentManager(...)
        # self.enrollment_manager.on_state_change = ...

        # Display buffers for each camera - worker creates these for processed overlay data
        self.display_buffers = {}
        self._create_display_buffers()
        
        # Processing state
        self.running = False
        self.last_frame_ids = {cam: -1 for cam in camera_indices}
        # FIX #1: Track last write_index from results buffer to detect stale data
        # Prevents frozen overlay when MediaPipe stops writing new results
        self.last_results_write_index = {cam: -1 for cam in camera_indices}
        # FIX #1b: Cache last valid result to prevent flickering when write_index unchanged
        # GUI runs at 60 FPS, MediaPipe writes at 30 FPS, so most reads see unchanged write_index
        # Cache allows smooth display while still detecting truly stale data (via timeout)
        self.cached_landmarks = {}  # {cam_idx: {'data': result_dict, 'timestamp': time.time()}}
        # NOTE: tracker_id_to_slot/slot_to_tracker_id removed - GridParticipantManager handles assignment
        self.participant_last_seen = {}
        self.participant_names = {i: f"P{i+1}" for i in range(self.max_participants)}
        
        # Performance tracking
        self.processing_times = []
        self.last_stats_print = time.time()

        # Occlusion state tracking for recovery mechanism
        # Tracks whether each camera is in occlusion, recovery, or normal state
        self._occlusion_state = {}  # {cam_idx: {'in_occlusion': bool, 'recovery_frames_left': int}}
        self._consecutive_no_pose = {}  # {cam_idx: int} - counts consecutive frames with no pose

        # Embedding frame sync diagnostics (to monitor ArcFace lag)
        self._embedding_frame_mismatch_count = 0
        self._embedding_sync_success_count = 0

        # Debug frame counter for participant assignment logs
        self.debug_frame_counter = 0
        
        # Buffer connections (will be set after initialization)
        self.buffer_registry = {}
        self.buffer_coordinator_info = {}
        self.camera_buffers = {}  # {camera_index: {buffer_type: shared_memory}}

        # LSL streaming support (for correlator)
        self.lsl_data_queue = None
        self.lsl_streaming_active = False
        self.mesh_enabled = config.get('startup_mode', {}).get('enable_mesh', False)
        self.verbose_debug = config.get('logging', {}).get('verbose_debug', False)

        # Correlation calculation for bar graph display (independent of LSL)
        from core.data_streaming.correlator import ChannelCorrelator
        self.correlator = ChannelCorrelator(
            window_size=60,
            fps=30,
            config=config
        )
        self.correlation_buffer = None  # Will be connected via control message
        self.participant_scores = {}  # Store latest scores per participant for correlation

        # Pose landmark smoothing (reduces jitter)
        self.pose_smoother = PoseLandmarkSmoother(config)
        logger.info("[WORKER] Pose smoother initialized")

        # Pose metrics calculator (head orientation, neck angle, shoulder shrug)
        metrics_config_dict = config.get('metrics_settings', {})
        self.metrics_config = MetricsConfig.from_dict(metrics_config_dict) if metrics_config_dict else MetricsConfig()
        self.metrics_calculator = PoseMetricsCalculator(self.metrics_config)
        logger.info(f"[WORKER] Pose metrics calculator initialized (min_confidence={self.metrics_config.min_confidence})")

        # Frame synchronization buffer for pose-face matching
        # Store recent display_faces with frame_ids for relaxed synchronization (±20 frames)
        self._display_faces_buffer = {}  # {cam_idx: deque of (frame_id, display_faces)}
        from collections import deque
        for cam_idx in camera_indices:
            self._display_faces_buffer[cam_idx] = deque(maxlen=30)  # Last 30 frames (±20 tolerance + 10 margin)

        # Pose LSL transmission diagnostics
        self._pose_lsl_stats = {}  # {cam_idx: {'dropped_no_faces': 0, 'sent': 0, 'frame_mismatches': 0}}
        self._pose_lsl_last_report = time.time()

        # Latest pose metrics (for GUI access)
        self._latest_pose_metrics = {}  # {cam_idx: PoseMetrics}

        # Latest performance metrics (for GUI access)
        # Format: {cam_idx: {'detection_fps': float, 'pose_fps': float, 'latency_ms': float}}
        self._latest_performance_metrics = {}

        # Performance metric tracking (per-camera rolling windows)
        self._detection_timing_window = {}  # {cam_idx: deque of (timestamp, processing_time_ms)}
        self._pose_timing_window = {}  # {cam_idx: deque of (timestamp, det_time_ms, pose_time_ms)}
        self._latency_window = {}  # {cam_idx: deque of latency_ms values}

    def get_latest_pose_metrics(self, cam_idx: int) -> Optional['PoseMetrics']:
        """
        Get the latest calculated pose metrics for a camera.

        Called by GUI to retrieve current metrics for display.

        Args:
            cam_idx: Camera index

        Returns:
            PoseMetrics instance or None if no metrics available
        """
        return self._latest_pose_metrics.get(cam_idx)

    def get_latest_performance_metrics(self, cam_idx: int) -> Optional[dict]:
        """
        Get the latest performance metrics for a camera.

        Called by GUI to retrieve current FPS and latency metrics for display.

        Args:
            cam_idx: Camera index

        Returns:
            Dict with keys: 'detection_fps', 'pose_fps', 'latency_ms'
            Returns None if no metrics available
        """
        return self._latest_performance_metrics.get(cam_idx)

    # ARCHITECTURAL CONSOLIDATION: Enrollment callbacks removed
    # Enrollment now handled entirely in GUI process
    # Worker sends embedding samples instead of managing enrollment state
    # OLD METHODS REMOVED:
    # - _on_enrollment_state_change()
    # - _on_enrollment_complete()
    # - _on_enrollment_failed()

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

    def _create_display_buffers(self):
        """Create display buffers for processed overlay data output."""
        for cam_idx in self.camera_indices:
            buffer_name = f"yq_display_{cam_idx}_{mp.current_process().pid}"
            try:
                display_buffer = DisplayBuffer(cam_idx, buffer_name, self.buffer_coordinator, create=True)
                self.display_buffers[cam_idx] = display_buffer
                logger.info(f"[DUAL BUFFER] Created display buffer for camera {cam_idx}: {buffer_name}")
            except Exception as e:
                logger.error(f"[DUAL BUFFER] Failed to create display buffer for camera {cam_idx}: {e}")
                
    def _connect_to_camera_gui_buffers(self):
        """Connect to camera processing buffers for reading and GUI buffer for writing (data bridge)."""
        import multiprocessing.shared_memory as mp_shm
        
        for cam_idx in self.camera_indices:
            self.camera_buffers[cam_idx] = {}

            # Initialize performance tracking for this camera
            from collections import deque
            self._detection_timing_window[cam_idx] = deque(maxlen=30)  # Last 30 frames (~1 sec at 30 FPS)
            self._pose_timing_window[cam_idx] = deque(maxlen=30)
            self._latency_window[cam_idx] = deque(maxlen=30)

            # Check if we have actual buffer names for this camera
            if cam_idx in self.actual_buffer_names:
                # Use the actual buffer names broadcast by camera worker
                actual_names = self.actual_buffer_names[cam_idx]

                # CRITICAL FIX: Connect to camera PROCESSING buffers for INPUT
                # Camera sends buffer names via shared_memory status dict
                # Map: {'frame': 'yq_frame_X_PID', 'results': 'yq_results_X_PID', 'gui': 'yq_gui_X_PID'}
                # MediaPipe native - results buffer contains landmarks + blendshapes + bboxes

                # INPUT buffers - read raw processing data from these
                input_mapping = {
                    'frame': 'frame',      # Camera frame buffer (yq_frame_X) - raw frames
                    'results': 'results',  # Results buffer (yq_results_X) - landmarks + blendshapes + bboxes
                    'pose': 'pose',        # Pose buffer (yq_pose_X) - body pose landmarks
                    'detection': 'detection',  # SCRFD detection buffer (yq_detection_X) - face bboxes + crops
                    'embedding': 'embedding',  # ArcFace embedding buffer (yq_embedding_X) - 512-dim embeddings
                }
                
                for buffer_type, camera_key in input_mapping.items():
                    if camera_key in actual_names:
                        buffer_name = actual_names[camera_key]
                        try:
                            shm = mp_shm.SharedMemory(name=buffer_name)
                            self.camera_buffers[cam_idx][buffer_type] = shm
                            logger.info(f"[DATA BRIDGE INPUT] Connected to {buffer_type} buffer for camera {cam_idx}: {buffer_name}")
                        except Exception as e:
                            logger.error(f"[DATA BRIDGE INPUT] Failed to connect to {buffer_type} buffer {buffer_name}: {e}")
                    else:
                        logger.warning(f"[DATA BRIDGE INPUT] No {camera_key} buffer name for camera {cam_idx}")
                
                # OUTPUT buffer - write display-ready data here
                if 'gui' in actual_names:
                    gui_buffer_name = actual_names['gui']
                    try:
                        gui_shm = mp_shm.SharedMemory(name=gui_buffer_name)
                        self.camera_buffers[cam_idx]['gui_output'] = gui_shm
                        logger.info(f"[DATA BRIDGE OUTPUT] Connected to GUI buffer for camera {cam_idx}: {gui_buffer_name}")
                    except Exception as e:
                        logger.error(f"[DATA BRIDGE OUTPUT] Failed to connect to GUI buffer {gui_buffer_name}: {e}")
                else:
                    logger.error(f"[DATA BRIDGE OUTPUT] No GUI buffer name for camera {cam_idx}")
                    
                logger.info(f"[DATA BRIDGE] Camera {cam_idx} bridge established: " +
                           f"INPUT from {list(input_mapping.keys())}, OUTPUT to gui_output")
            else:
                logger.error(f"[DATA BRIDGE] No actual buffer names available for camera {cam_idx} - cannot establish data bridge")

    def _connect_camera_buffers_for_index(self, cam_idx: int):
        """Connect to camera processing buffers for a specific camera index."""
        import multiprocessing.shared_memory as mp_shm

        if cam_idx not in self.camera_buffers:
            self.camera_buffers[cam_idx] = {}

        # Initialize performance tracking for this camera
        from collections import deque
        if cam_idx not in self._detection_timing_window:
            self._detection_timing_window[cam_idx] = deque(maxlen=30)  # Last 30 frames (~1 sec at 30 FPS)
            self._pose_timing_window[cam_idx] = deque(maxlen=30)
            self._latency_window[cam_idx] = deque(maxlen=30)

        # Check if we have actual buffer names for this camera
        if cam_idx in self.actual_buffer_names:
            actual_names = self.actual_buffer_names[cam_idx]
            
            # INPUT buffers - read raw processing data from these
            # MediaPipe native - results buffer contains landmarks + blendshapes + bboxes
            input_mapping = {
                'frame': 'frame',      # Camera frame buffer (yq_frame_X) - raw frames
                'results': 'results',  # Results buffer (yq_results_X) - landmarks + blendshapes + bboxes
                'pose': 'pose',        # Pose buffer (yq_pose_X) - body pose landmarks
                'detection': 'detection',  # SCRFD detection buffer (yq_detection_X) - face bboxes + crops
                'embedding': 'embedding',  # ArcFace embedding buffer (yq_embedding_X) - 512-dim embeddings
            }
            
            for buffer_type, camera_key in input_mapping.items():
                if camera_key in actual_names:
                    buffer_name = actual_names[camera_key]
                    try:
                        shm = mp_shm.SharedMemory(name=buffer_name)
                        self.camera_buffers[cam_idx][buffer_type] = shm
                        logger.info(f"[DYNAMIC CONNECT] Connected to {buffer_type} buffer for camera {cam_idx}: {buffer_name}")
                    except Exception as e:
                        logger.error(f"[DYNAMIC CONNECT] Failed to connect to {buffer_type} buffer {buffer_name}: {e}")
                else:
                    logger.warning(f"[DYNAMIC CONNECT] No {camera_key} buffer name for camera {cam_idx}")
            
            # OUTPUT buffer - write display-ready data here
            if 'gui' in actual_names:
                gui_buffer_name = actual_names['gui']
                try:
                    gui_shm = mp_shm.SharedMemory(name=gui_buffer_name)
                    self.camera_buffers[cam_idx]['gui_output'] = gui_shm
                    logger.info(f"[DYNAMIC CONNECT] Connected to GUI buffer for camera {cam_idx}: {gui_buffer_name}")
                except Exception as e:
                    logger.error(f"[DYNAMIC CONNECT] Failed to connect to GUI buffer {gui_buffer_name}: {e}")
            else:
                logger.error(f"[DYNAMIC CONNECT] No GUI buffer name for camera {cam_idx}")
            
            logger.info(f"[DYNAMIC CONNECT] Camera {cam_idx} buffers connected dynamically")
        else:
            logger.error(f"[DYNAMIC CONNECT] No actual buffer names for camera {cam_idx}")
                
    def _connect_to_dual_buffers(self):
        """Connect to both input (camera GUI) and output (display) buffers."""
        logger.info("[DUAL BUFFER] Connecting to input buffers (camera GUI buffers)...")
        self._connect_to_camera_gui_buffers()
        logger.info("[DUAL BUFFER] Input buffer connections completed")
        
        logger.info("[DUAL BUFFER] Display buffers already created during initialization")
        
        # Validate dual buffer setup
        for cam_idx in self.camera_indices:
            input_connected = cam_idx in self.camera_buffers and 'gui_output' in self.camera_buffers[cam_idx]
            output_created = cam_idx in self.display_buffers
            
            if input_connected and output_created:
                logger.info(f"[DUAL BUFFER] Camera {cam_idx} fully connected - INPUT: {self.camera_buffers[cam_idx]['gui_output'].name}, OUTPUT: {self.display_buffers[cam_idx].buffer_name}")
            else:
                logger.error(f"[DUAL BUFFER] Camera {cam_idx} setup incomplete - INPUT: {input_connected}, OUTPUT: {output_created}")
    
    def run(self):
        """Main processing loop.

        CRITICAL FIX: Comprehensive exception handling to prevent silent crashes.
        All exceptions are caught, logged with full stack traces, and cleanup is guaranteed.
        """
        try:
            self.running = True

            # Process early control messages (shutdown, correlation buffer)
            # NOTE: actual_buffer_names is already set by gui_processing_worker_main (line 3105)
            # No need to wait for buffer_names via control queue - they're passed as function args
            logger.info("[WORKER STARTUP] Processing early control messages...")
            try:
                # Process messages in queue (non-blocking, single pass - no wait)
                while not self.control_queue.empty():
                    msg = self.control_queue.get_nowait()
                    if msg['type'] == 'shutdown':
                        logger.info("[WORKER STARTUP] Received early shutdown command")
                        self.running = False
                        return
                    elif msg['type'] == 'connect_correlation_buffer':
                        # CRITICAL: Process correlation buffer connection during startup
                        # GUI sends this message before worker is ready, so we must handle it here
                        buffer_name = msg.get('buffer_name')
                        try:
                            import multiprocessing.shared_memory as mp_shm
                            # CRITICAL: Store SharedMemory reference to prevent garbage collection
                            self.correlation_shm = mp_shm.SharedMemory(name=buffer_name)
                            self.correlation_buffer = np.ndarray((52,), dtype=np.float32, buffer=self.correlation_shm.buf)
                            logger.info(f"[WORKER STARTUP] Connected to correlation buffer: {buffer_name}")
                        except Exception as e:
                            logger.error(f"[WORKER STARTUP] Failed to connect to correlation buffer: {e}")
                            import traceback
                            traceback.print_exc()
                    elif msg['type'] == 'connect_camera':
                        # CRITICAL: Process camera connection during startup (BLACK SCREEN FIX)
                        # GUI sends connect_camera when camera becomes ready, which happens AFTER worker starts
                        # We must handle this in early processing so camera is registered BEFORE _connect_to_dual_buffers()
                        cam_idx = msg.get('camera_index')
                        buffer_names = msg.get('buffer_names', {})
                        resolution = msg.get('resolution')

                        logger.info(f"[WORKER STARTUP] 📡 Received connect_camera (early) for camera {cam_idx}")
                        logger.info(f"[WORKER STARTUP]   Buffer names: {list(buffer_names.keys())}")
                        logger.info(f"[WORKER STARTUP]   Resolution: {resolution}")

                        try:
                            # 1. Register camera for processing BEFORE dual buffer connection
                            if cam_idx not in self.actual_buffer_names:
                                self.actual_buffer_names[cam_idx] = buffer_names
                                logger.info(f"[WORKER STARTUP] Registered buffer names for camera {cam_idx}")

                            if resolution and cam_idx not in self.camera_resolutions:
                                self.camera_resolutions[cam_idx] = resolution
                                logger.info(f"[WORKER STARTUP] Registered resolution for camera {cam_idx}: {resolution}")

                            if cam_idx not in self.camera_indices:
                                self.camera_indices.append(cam_idx)
                                logger.info(f"[WORKER STARTUP] Added camera {cam_idx} to camera_indices: {self.camera_indices}")

                            # 2. Create display buffer for this camera (will be connected in _connect_to_dual_buffers)
                            if cam_idx not in self.display_buffers:
                                buffer_name = f"yq_display_{cam_idx}_{mp.current_process().pid}"
                                try:
                                    display_buffer = DisplayBuffer(cam_idx, buffer_name, self.buffer_coordinator, create=True)
                                    self.display_buffers[cam_idx] = display_buffer
                                    logger.info(f"[WORKER STARTUP] ✅ Created display buffer for camera {cam_idx}: {buffer_name}")

                                    # Queue notification for after startup completes (status_queue not safe during early init)
                                    if not hasattr(self, '_pending_display_notifications'):
                                        self._pending_display_notifications = []
                                    self._pending_display_notifications.append({
                                        'type': 'camera_display_connected',
                                        'camera_index': cam_idx,
                                        'display_buffer_name': buffer_name
                                    })
                                    logger.info(f"[WORKER STARTUP] Queued display buffer notification for camera {cam_idx}")

                                except Exception as e:
                                    logger.error(f"[WORKER STARTUP] Failed to create display buffer for camera {cam_idx}: {e}")
                                    import traceback
                                    traceback.print_exc()

                            # 3. Initialize tracking state early
                            if cam_idx not in self.last_frame_ids:
                                self.last_frame_ids[cam_idx] = -1

                            # 4. Initialize pose-face matching buffer early
                            from collections import deque
                            if cam_idx not in self._display_faces_buffer:
                                self._display_faces_buffer[cam_idx] = deque(maxlen=30)

                            logger.info(f"[WORKER STARTUP] ✅ Camera {cam_idx} fully registered and ready")

                        except Exception as e:
                            logger.error(f"[WORKER STARTUP] Failed to register camera {cam_idx}: {e}")
                            import traceback
                            traceback.print_exc()
                    # Ignore buffer_names messages - already received via function args
            except Exception as e:
                logger.error(f"[WORKER STARTUP] Error processing early control messages: {e}")

            logger.info("[WORKER STARTUP] Early message processing complete")
            
            # Connect to dual buffer system (input: camera GUI buffers, output: display buffers)
            logger.info("[WORKER STARTUP] Connecting to dual buffers...")
            self._connect_to_dual_buffers()
            logger.info("[WORKER STARTUP] ✅ Dual buffer connection complete")

            # Report ready status with display buffer names for GUI to connect to
            logger.info("[WORKER STARTUP] Preparing to send 'ready' status to GUI...")
            
            # Report the display buffers that GUI should connect to (OUTPUT buffers)
            display_buffer_info = {
                cam: self.display_buffers[cam].buffer_name 
                for cam in self.display_buffers
            }
            
            # Also report input buffer connections for debugging
            input_buffer_info = {}
            for cam_idx in self.camera_indices:
                if cam_idx in self.camera_buffers and 'gui_output' in self.camera_buffers[cam_idx]:
                    gui_shm = self.camera_buffers[cam_idx]['gui_output']
                    input_buffer_info[cam_idx] = gui_shm.name
            
            
            ready_status = {
                'type': 'ready',
                'display_buffers': display_buffer_info,  # GUI connects to these (OUTPUT)
                'input_buffers': input_buffer_info       # For debugging only
            }
    
            # CRITICAL FIX: Use non-blocking put with error handling to prevent queue overflow crashes
            # If queue is full, log error but don't crash the worker
            try:
                self.status_queue.put(ready_status, block=False)
                logger.info("[WORKER] ✅ Sent 'ready' status to GUI - display should start soon")
                sys.stderr.flush()  # CRITICAL: Flush to make logs visible immediately
            except queue.Full:
                logger.error("[WORKER ERROR] Status queue full! Cannot send ready status. GUI may not initialize properly.")
                sys.stderr.flush()
            except Exception as e:
                logger.error(f"[WORKER ERROR] Failed to send ready status: {e}")
                sys.stderr.flush()

            # Send any pending display buffer notifications that were queued during early processing
            if hasattr(self, '_pending_display_notifications'):
                logger.info(f"[WORKER STARTUP] Sending {len(self._pending_display_notifications)} pending display notifications...")
                for notification in self._pending_display_notifications:
                    try:
                        self.status_queue.put(notification, block=False)
                        cam_idx = notification.get('camera_index')
                        logger.info(f"[WORKER STARTUP] 📤 Sent camera_display_connected for camera {cam_idx}")
                    except queue.Full:
                        logger.error(f"[WORKER STARTUP] Status queue full! Cannot send display notification for camera {notification.get('camera_index')}")
                    except Exception as e:
                        logger.error(f"[WORKER STARTUP] Failed to send display notification: {e}")
                # Clear the pending list
                self._pending_display_notifications = []

            while self.running:
                try:
                    # Check for control commands (non-blocking)
                    self._process_control_commands()
    
                    # Process each camera with individual exception handling
                    for cam_idx in self.camera_indices:
                        try:
                            self._process_camera(cam_idx)
                        except Exception as cam_error:
                            logger.error(f"[WORKER ERROR] Exception in _process_camera({cam_idx}): {cam_error}")
                            import traceback
                            traceback.print_exc()
                            # Continue with other cameras instead of breaking entire loop
    
                    # Performance monitoring
                    self._update_performance_stats()
    
                    # FIX: Adjusted polling rate to match camera frame rate (30 FPS = ~33ms)
                    # OPTIMIZED: Reduced polling rate for lower latency (10ms = ~100 Hz polling)
                    # Previously 30ms (33 FPS polling) - now 10ms for better pose responsiveness
                    # Previous 5ms polling caused 82% skip rate due to checking too frequently
                    time.sleep(0.010)  # 10ms - improves pose overlay latency by ~20ms
    
                except Exception as e:
                    logger.error(f"[WORKER ERROR] Exception in main processing loop: {e}")
                    import traceback
                    traceback.print_exc()
                    # import sys removed - using module-level import (line 15)
                    sys.stdout.flush()
                    sys.stderr.flush()
    
            logger.info("GUI Processing Worker stopped")

        except Exception as fatal_error:
            # CRITICAL: Catch ANY unhandled exception in the worker
            # This prevents silent crashes and ensures we log the root cause
            logger.critical(f"[WORKER FATAL] Unhandled exception in worker main loop: {fatal_error}")
            import traceback
            logger.critical(f"[WORKER FATAL] Full traceback:\n{traceback.format_exc()}")
            # import sys removed - using module-level import (line 15)
            sys.stdout.flush()
            sys.stderr.flush()

            # Try to notify GUI of the error (best effort, may fail if queue issues caused crash)
            try:
                self.status_queue.put({
                    'type': 'fatal_error',
                    'error': str(fatal_error),
                    'traceback': traceback.format_exc()
                }, block=False, timeout=1.0)
            except:
                logger.critical("[WORKER FATAL] Failed to notify GUI of fatal error (queue unavailable)")

            # Re-raise to ensure process exits with error code
            raise

        finally:
            # CRITICAL: Cleanup is ALWAYS executed, even on exception or early return
            # This prevents resource leaks and orphaned shared memory
            logger.info("[WORKER CLEANUP] Starting cleanup in finally block...")
            try:
                self._cleanup()
                logger.info("[WORKER CLEANUP] Cleanup completed successfully")
            except Exception as cleanup_error:
                logger.error(f"[WORKER CLEANUP] Error during cleanup: {cleanup_error}")
                import traceback
                logger.error(traceback.format_exc())
        
    def _process_camera(self, cam_idx: int):
        """Complete data bridge: Read processing data → Process → Write display data."""
        overall_start = time.time()

        # Initialize profiling structures
        if not hasattr(self, '_perf_stats'):
            self._perf_stats = {}
        if cam_idx not in self._perf_stats:
            self._perf_stats[cam_idx] = {
                'total_calls': 0,
                'frames_processed': 0,
                'frames_skipped_none': 0,
                'frames_skipped_duplicate': 0,
                'timing': {
                    'read_frame_data': [],
                    'participant_assign': [],
                    'face_recognition': [],
                    'lsl_streaming': [],
                    'display_write': [],
                    'total_processing': []
                }
            }

        stats = self._perf_stats[cam_idx]
        stats['total_calls'] += 1

        # Debug entry every 1000 calls (~1 second for camera 0)
        if not hasattr(self, 'process_camera_call_count'):
            self.process_camera_call_count = {}

        if cam_idx not in self.process_camera_call_count:
            self.process_camera_call_count[cam_idx] = 0

        self.process_camera_call_count[cam_idx] += 1

        try:
            # STEP 1: Check latest frame_id BEFORE reading expensive frame data
            # This ensures we always skip to the absolute latest frame, eliminating "tail-gating" lag
            latest_frame_id = self._get_latest_frame_id(cam_idx)
            if latest_frame_id is None:
                stats['frames_skipped_none'] += 1

                # FALLBACK: If display buffer hasn't been updated in >50ms, write a heartbeat frame
                # This ensures GUI displays smooth video even when landmarks aren't available/synchronized
                # Provides minimum 20 FPS display rate when normal data flow is blocked
                if not hasattr(self, '_last_display_write_time'):
                    self._last_display_write_time = {}

                last_write = self._last_display_write_time.get(cam_idx, 0)
                time_since_write = (time.time() - last_write) * 1000  # ms

                if time_since_write > 50:  # 50ms threshold (~20 FPS minimum display rate)
                    # Fetch latest frame directly from camera buffer for heartbeat write
                    heartbeat_frame = self._fetch_latest_frame_for_heartbeat(cam_idx)
                    if heartbeat_frame is not None:
                        # CRITICAL: Always update timestamp BEFORE write attempt to prevent heartbeat stalls
                        # If we only update on successful write, failed writes cause heartbeat loop with no progress
                        current_time = time.time()
                        self._last_display_write_time[cam_idx] = current_time

                        # Write heartbeat with empty faces to keep GUI alive
                        success = self._write_display_data(cam_idx, heartbeat_frame['frame_id'],
                                                         current_time, [], heartbeat_frame['frame'])

                        # Track heartbeat activity regardless of write success
                        if not hasattr(self, '_heartbeat_count'):
                            self._heartbeat_count = {}
                        if cam_idx not in self._heartbeat_count:
                            self._heartbeat_count[cam_idx] = 0
                        self._heartbeat_count[cam_idx] += 1

                        # Track failures separately for diagnostics
                        if not success:
                            if not hasattr(self, '_heartbeat_failures'):
                                self._heartbeat_failures = {}
                            if cam_idx not in self._heartbeat_failures:
                                self._heartbeat_failures[cam_idx] = 0
                            self._heartbeat_failures[cam_idx] += 1

                        # Log heartbeat activity occasionally (every 100 heartbeats)
                        if self._heartbeat_count[cam_idx] % 100 == 1:
                            failures = self._heartbeat_failures.get(cam_idx, 0) if hasattr(self, '_heartbeat_failures') else 0
                            success_rate = ((self._heartbeat_count[cam_idx] - failures) / self._heartbeat_count[cam_idx] * 100) if self._heartbeat_count[cam_idx] > 0 else 0
                            logger.info(f"[HEARTBEAT] Camera {cam_idx}: Active (count={self._heartbeat_count[cam_idx]}, "
                                      f"success_rate={success_rate:.1f}%)")

                return

            # Deduplicate: Skip if same as last processed frame
            # This check happens BEFORE reading expensive frame data for performance
            if latest_frame_id == self.last_frame_ids[cam_idx]:
                stats['frames_skipped_duplicate'] += 1
                return  # Already processed this frame

            # DIAGNOSTIC: Log frame skipping behavior
            if not hasattr(self, '_frame_process_log_counter'):
                self._frame_process_log_counter = {}
            if cam_idx not in self._frame_process_log_counter:
                self._frame_process_log_counter[cam_idx] = 0
            self._frame_process_log_counter[cam_idx] += 1


            # STEP 2: Read full frame data for latest frame ONLY
            # This happens AFTER dedup check to avoid expensive read for duplicate frames
            read_start = time.time()
            frame_data = self._read_frame_data(cam_idx)
            read_time = (time.time() - read_start) * 1000  # Convert to ms
            # FIX: Defensive check before accessing nested dict keys
            if 'timing' in stats and 'read_frame_data' in stats['timing']:
                stats['timing']['read_frame_data'].append(read_time)

            if not frame_data:
                # Should not happen since _get_latest_frame_id() succeeded, but handle gracefully
                stats['frames_skipped_none'] += 1
                return

            # Frame will be processed - increment counter
            stats['frames_processed'] += 1

            # STEP 3: Extract and process data
            frame_bgr = frame_data.get('frame')
            landmarks_data = frame_data.get('landmarks')
            pose_data = frame_data.get('pose')  # List of poses

            # DIAGNOSTIC: Log pose data extracted from frame_data
            if pose_data is not None and len(pose_data) > 0:
                logger.debug(f"[POSE PIPELINE-2] Camera {cam_idx}, frame {frame_data.get('frame_id')}: "
                           f"Extracted {len(pose_data)} pose(s) from frame_data dict")
            elif pose_data is not None and len(pose_data) == 0:
                logger.debug(f"[POSE PIPELINE-2] Camera {cam_idx}: pose_data=[] in frame_data")
            else:
                logger.debug(f"[POSE PIPELINE-2] Camera {cam_idx}: pose_data=None in frame_data")

            # Get frame dimensions for coordinate conversion
            if frame_bgr is not None:
                frame_height, frame_width = frame_bgr.shape[:2]
            else:
                # Fallback to camera's actual resolution from camera_resolutions
                if cam_idx in self.camera_resolutions:
                    frame_width, frame_height = self.camera_resolutions[cam_idx]
                else:
                    # Last resort: use 720p default
                    frame_width, frame_height = 1280, 720
                    logger.warning(f"[WORKER] No resolution info for camera {cam_idx}, using 720p default")

            # Build unified face list from landmarks data
            faces = self._build_unified_face_list(landmarks_data, cam_idx, frame_width, frame_height)


            # STEP 3: Create display-ready face data with all processing
            display_faces = []

            if faces:
                # Perform participant assignment (grid + embedding matching)
                assign_start = time.time()
                participant_assignments = self._assign_participants(faces, cam_idx, frame_data['frame_id'])
                assign_time = (time.time() - assign_start) * 1000
                # FIX: Defensive check before accessing nested dict keys
                if 'timing' in stats and 'participant_assign' in stats['timing']:
                    stats['timing']['participant_assign'].append(assign_time)

                # Process face recognition (if available)
                recog_start = time.time()
                recognition_updates = self._process_face_recognition(faces, cam_idx)
                recog_time = (time.time() - recog_start) * 1000
                # FIX: Defensive check before accessing nested dict keys
                if 'timing' in stats and 'face_recognition' in stats['timing']:
                    stats['timing']['face_recognition'].append(recog_time)
                
                for i, face in enumerate(faces):
                    participant_id = participant_assignments[i] if i < len(participant_assignments) else None

                    # Add face index for recognition_updates lookup
                    face['face_idx'] = i

                    # Create pre-formatted label
                    label = self._format_face_label(face, participant_id, recognition_updates)
                    
                    # Transform landmarks from crop space to frame space using centralized coordinate system
                    display_landmarks = None
                    if face.get('landmarks') is not None:
                        landmarks = face['landmarks']
                        bbox = face.get('bbox', [0, 0, 100, 100])

                        try:
                            # Validate landmark array structure (expecting 478x3 from MediaPipe)
                            if isinstance(landmarks, np.ndarray) and len(landmarks.shape) >= 2:
                                if landmarks.shape[0] == 478 and landmarks.shape[1] >= 2:
                                    # Initialize coordinate system for this camera if needed
                                    if cam_idx not in self.coordinate_systems:
                                        # Get frame resolution for this camera
                                        frame_width, frame_height = self.camera_resolutions.get(
                                            cam_idx, (1280, 720)
                                        )
                                        self.coordinate_systems[cam_idx] = create_coordinate_system(
                                            self.config, frame_width, frame_height
                                        )

                                    coord_sys = self.coordinate_systems[cam_idx]

                                    # Check if landmarks are in crop space (192×192) or already in frame space
                                    x_coords = landmarks[:, 0]
                                    y_coords = landmarks[:, 1]
                                    max_x, max_y = np.max(x_coords), np.max(y_coords)

                                    # Landmarks from MediaPipe are in 192×192 crop space
                                    # If max_x/max_y are around 192, they need transformation
                                    # If they're already large (e.g., 1280), they're in frame space (legacy path)
                                    if max_x <= coord_sys.crop_size * 1.5:  # Allow 50% margin for crop space detection
                                        # Transform from 192×192 crop space to frame space
                                        display_landmarks = coord_sys.transform_landmarks_crop_to_frame(
                                            landmarks, bbox
                                        )

                                        # Debug logging (first 5 frames only)
                                        if not hasattr(self, '_transform_debug_count'):
                                            self._transform_debug_count = 0
                                        if self._transform_debug_count < 5:
                                            logger.info(
                                                f"[COORD TRANSFORM] Camera {cam_idx}: "
                                                f"Crop→Frame transformation applied. "
                                                f"Crop max: ({max_x:.1f}, {max_y:.1f}), "
                                                f"Bbox: {bbox}, "
                                                f"Frame: {coord_sys.frame_width}×{coord_sys.frame_height}"
                                            )
                                            self._transform_debug_count += 1
                                    else:
                                        # Already in frame space (legacy path for backward compatibility)
                                        display_landmarks = landmarks

                                        # One-time warning
                                        if not hasattr(self, '_frame_space_warning_logged'):
                                            logger.warning(
                                                f"[COORD TRANSFORM] Landmarks already in frame space "
                                                f"(max_x={max_x:.1f}, expected ~{coord_sys.crop_size}). "
                                                f"This is unexpected with SCRFD pipeline."
                                            )
                                            self._frame_space_warning_logged = True
                                else:
                                    # Invalid shape - use as-is
                                    display_landmarks = landmarks
                                    logger.warning(f"[COORD TRANSFORM] Invalid landmark shape: {landmarks.shape}")
                            else:
                                display_landmarks = landmarks
                        except Exception as e:
                            logger.error(f"[FATAL] Error transforming landmarks: {e}")
                            import traceback
                            traceback.print_exc()
                            display_landmarks = landmarks

                    # Encode enrollment state in display flags for overlay colors
                    display_flags = 0
                    face_idx = face.get('face_idx', -1)
                    if face_idx >= 0 and face_idx in recognition_updates:
                        rec_info = recognition_updates[face_idx]
                        enrollment_state = rec_info.get('enrollment_state', 'UNKNOWN')

                        if enrollment_state == 'ENROLLED':
                            display_flags |= DISPLAY_FLAG_ENROLLED
                        elif enrollment_state == 'COLLECTING':
                            display_flags |= DISPLAY_FLAG_COLLECTING
                        elif enrollment_state == 'VALIDATING':
                            display_flags |= DISPLAY_FLAG_VALIDATING
                        elif enrollment_state == 'FAILED':
                            display_flags |= DISPLAY_FLAG_FAILED

                    # Create display face data structure
                    display_face = DisplayFaceData(
                        face_id=face.get('id', -1),
                        participant_id=participant_id or -1,
                        track_id=face.get('track_id', -1),
                        bbox=face.get('bbox', (0, 0, 0, 0)),
                        landmarks=display_landmarks,
                        label=label,
                        confidence=face.get('confidence', 0.0),
                        display_flags=display_flags  # Encoded enrollment state
                    )
                    display_faces.append(display_face)

            # Store display_faces in frame synchronization buffer for pose matching
            # This allows pose data from nearby frames to find matching faces (±3 frame tolerance)
            if cam_idx in self._display_faces_buffer:
                self._display_faces_buffer[cam_idx].append((frame_data['frame_id'], display_faces))

                # DIAGNOSTIC: Track display_faces buffer population
                if not hasattr(self, '_display_faces_buffer_diagnostic'):
                    self._display_faces_buffer_diagnostic = {}
                if cam_idx not in self._display_faces_buffer_diagnostic:
                    self._display_faces_buffer_diagnostic[cam_idx] = {'total': 0, 'with_faces': 0, 'empty': 0}

                self._display_faces_buffer_diagnostic[cam_idx]['total'] += 1
                if display_faces:
                    self._display_faces_buffer_diagnostic[cam_idx]['with_faces'] += 1
                else:
                    self._display_faces_buffer_diagnostic[cam_idx]['empty'] += 1


            # CRITICAL FIX: Match poses to participants AFTER faces are in buffer
            # This ensures current frame's faces are available for matching
            # (Previously done in _read_pose_data(), but buffer was empty/stale)
            if pose_data:
                logger.debug(f"[POSE PIPELINE-3] Camera {cam_idx}, frame {frame_data['frame_id']}: "
                           f"Before matching: {len(pose_data)} pose(s)")
                pose_data = self._match_poses_to_participants(pose_data, cam_idx, frame_data['frame_id'])
                logger.debug(f"[POSE PIPELINE-3] Camera {cam_idx}, frame {frame_data['frame_id']}: "
                           f"After matching: {len(pose_data) if pose_data else 0} pose(s)")
            else:
                logger.debug(f"[POSE PIPELINE-3] Camera {cam_idx}: Skipping pose matching (pose_data is None or empty)")

            # STEP 3.4: Calculate pose metrics (head orientation, neck angle, shoulder shrug)
            pose_metrics = None
            if pose_data and len(pose_data) > 0:
                # For single-person tracking, calculate metrics for first person
                # TODO: Support multi-person metrics
                first_person_pose = pose_data[0] if isinstance(pose_data, list) else pose_data
                pose_metrics = self.metrics_calculator.calculate_metrics(first_person_pose)

                if pose_metrics and self.verbose_debug:
                    logger.debug(f"[METRICS] Camera {cam_idx}: {pose_metrics.get_head_orientation_summary()}")

            # STEP 3.5: Calculate correlation for bar graph display (ALWAYS active, independent of LSL)
            # Read latest blendshapes independently (no frame sync check needed for correlation)
            correlation_data = self._read_latest_blendshapes_for_correlation(cam_idx)

            if correlation_data and len(faces) > 0:
                blendshapes_array = correlation_data['blendshapes']

                # Update participant scores for correlation
                for i, face in enumerate(faces):
                    participant_id = participant_assignments[i] if i < len(participant_assignments) else None
                    if participant_id is not None and i < len(blendshapes_array):
                        pid_key = f"P{participant_id}"
                        self.participant_scores[pid_key] = np.array(blendshapes_array[i])

                # Calculate correlation if we have 2+ participants
                if len(self.participant_scores) >= 2 and self.correlation_buffer is not None:
                    pids = sorted(self.participant_scores.keys())[:2]

                    # Measure correlator execution time
                    corr_start_time = time.time()
                    corr = self.correlator.update(
                        self.participant_scores[pids[0]],
                        self.participant_scores[pids[1]]
                    )
                    corr_duration_ms = (time.time() - corr_start_time) * 1000

                    # Report correlator activity periodically
                    if not hasattr(self, '_last_correlator_report'):
                        self._last_correlator_report = 0
                    if time.time() - self._last_correlator_report > 5.0:
                        max_corr = np.max(np.abs(corr)) if corr is not None else 0
                        logger.info(f"[Correlator] {pids[0]} vs {pids[1]} | max: {max_corr:.3f} | {corr_duration_ms:.1f}ms")
                        self._last_correlator_report = time.time()

                    if corr is not None:
                        if self.correlation_buffer is not None:
                            try:
                                # Explicit dtype conversion to avoid shared memory hang
                                self.correlation_buffer[:] = corr.astype(np.float32, copy=False)
                            except Exception as e:
                                logger.error(f"Correlation buffer write failed: {e}")
                                raise
                        else:
                            # Log once per session
                            if not hasattr(self, '_corr_buffer_warning_logged'):
                                logger.warning("Correlation buffer not connected")
                                self._corr_buffer_warning_logged = True

            # Send blendshapes to LSL for streaming (if streaming active)
            lsl_start = time.time()
            if self.lsl_streaming_active and self.lsl_data_queue and landmarks_data and 'blendshapes' in landmarks_data:
                blendshapes_array = landmarks_data['blendshapes']

                # DIAGNOSTIC: Log LSL streaming state (only once per session to avoid spam)
                if not hasattr(self, '_lsl_mesh_diagnostic_logged'):
                    logger.info(f"[MESH DIAGNOSTIC] LSL streaming active, mesh_enabled={self.mesh_enabled}")
                    self._lsl_mesh_diagnostic_logged = True

                if len(blendshapes_array) > 0:
                    lsl_faces = []
                    for i, face in enumerate(faces):
                        participant_id = participant_assignments[i] if i < len(participant_assignments) else None

                        # GRID ASSIGNMENT: participant_id is now primary identifier
                        # track_id from MediaPipe is obsolete with SCRFD, use participant_id instead
                        track_id = participant_id if participant_id is not None else face.get('track_id', -1)

                        # Only send if we have a valid participant ID and blendshapes
                        if participant_id is not None and i < len(blendshapes_array):
                            # Use custom participant name if available, otherwise fallback to "P{id}" format
                            # participant_id is 1-based, dict is 0-based
                            custom_name = self.participant_names.get(participant_id - 1)
                            if custom_name and custom_name.strip() and not custom_name.strip().startswith('P'):
                                participant_label = custom_name.strip()
                            else:
                                participant_label = f"P{participant_id}"

                            face_dict = {
                                'participant_id': participant_label,  # Uses custom name for LSL stream naming
                                'track_id': track_id,  # Legacy field: now equals participant_id for grid-based assignment
                                'blend_scores': blendshapes_array[i].tolist()
                            }

                            # Add landmarks for quality estimation and mesh streaming
                            if 'landmarks' in landmarks_data and i < len(landmarks_data['landmarks']):
                                landmarks_478x3 = landmarks_data['landmarks'][i]
                                face_dict['landmarks'] = landmarks_478x3.tolist()

                                # Add flattened mesh data if enabled
                                if self.mesh_enabled:
                                    mesh_data_flat = landmarks_478x3.flatten().tolist()
                                    face_dict['mesh_data'] = mesh_data_flat
                                    # DIAGNOSTIC: Log mesh data addition (throttled to first 3 occurrences)
                                    if not hasattr(self, '_mesh_add_count'):
                                        self._mesh_add_count = 0
                                    if self._mesh_add_count < 3:
                                        logger.info(f"[MESH DIAGNOSTIC] Added mesh_data for P{participant_id} "
                                                  f"({len(mesh_data_flat)} values, shape was {landmarks_478x3.shape})")
                                        self._mesh_add_count += 1
                                else:
                                    # DIAGNOSTIC: Log when mesh is skipped (throttled)
                                    if not hasattr(self, '_mesh_skip_count'):
                                        self._mesh_skip_count = 0
                                    if self._mesh_skip_count < 3:
                                        logger.warning(f"[MESH DIAGNOSTIC] Skipping mesh_data for P{participant_id} "
                                                     f"(mesh_enabled={self.mesh_enabled})")
                                        self._mesh_skip_count += 1
                            else:
                                # DIAGNOSTIC: Log when landmarks are missing
                                if not hasattr(self, '_landmarks_missing_logged'):
                                    logger.warning(f"[MESH DIAGNOSTIC] No landmarks in landmarks_data for face {i}")
                                    self._landmarks_missing_logged = True

                            # Add bbox and frame info for quality estimation
                            if 'bbox' in face:
                                face_dict['bbox'] = face['bbox']
                            if 'frame_shape' in frame_data:
                                face_dict['frame_shape'] = frame_data['frame_shape']

                            lsl_faces.append(face_dict)

                    # Send to LSL helper via queue
                    if lsl_faces:
                        try:
                            self.lsl_data_queue.put_nowait({
                                'type': 'participant_data',
                                'camera_index': cam_idx,
                                'frame_id': frame_data['frame_id'],
                                'faces': lsl_faces
                            })
                        except Exception as e:
                            logger.warning(f"Failed to send to LSL queue: {e}")

            # Send pose data to LSL if enabled and available (independent of face detection)
            # DIAGNOSTIC: Track why pose LSL might not be called
            if not hasattr(self, '_pose_lsl_condition_diagnostic'):
                self._pose_lsl_condition_diagnostic = {
                    'checked': 0,
                    'lsl_inactive': 0,
                    'no_pose_data': 0,
                    'no_queue': 0,
                    'success': 0
                }

            self._pose_lsl_condition_diagnostic['checked'] += 1

            # Check each condition separately for diagnostics
            lsl_active = self.lsl_streaming_active
            has_pose_data = pose_data is not None and len(pose_data) > 0
            has_queue = self.lsl_data_queue is not None

            if not lsl_active:
                self._pose_lsl_condition_diagnostic['lsl_inactive'] += 1
            if not has_pose_data:
                self._pose_lsl_condition_diagnostic['no_pose_data'] += 1
            if not has_queue:
                self._pose_lsl_condition_diagnostic['no_queue'] += 1

            # ALWAYS-ON diagnostic: Log state every 100 frames
            if self._pose_lsl_condition_diagnostic['checked'] % 100 == 0:
                logger.info(f"[LSL DIAGNOSTIC] Frame {frame_data['frame_id']}: "
                          f"lsl_active={lsl_active}, "
                          f"has_pose_data={has_pose_data} ({len(pose_data) if pose_data else 0} poses), "
                          f"has_queue={has_queue}, "
                          f"diagnostic_counts={self._pose_lsl_condition_diagnostic}")

            if pose_data and self.verbose_debug:
                logger.info(f"[POSE LSL DEBUG] Checking send condition for cam {cam_idx}: "
                          f"lsl_active={lsl_active}, "
                          f"pose_data={len(pose_data)} poses, "
                          f"queue={'exists' if has_queue else 'None'}, "
                          f"display_faces={len(display_faces)} faces")

            if lsl_active and has_pose_data and has_queue:
                self._pose_lsl_condition_diagnostic['success'] += 1
                if self.verbose_debug:
                    logger.info(f"[POSE LSL DEBUG] Condition passed! Calling _send_pose_data_to_lsl for camera {cam_idx}, frame {frame_data['frame_id']}")
                try:
                    self._send_pose_data_to_lsl(cam_idx, frame_data['frame_id'],
                                               pose_data, display_faces)
                except Exception as e:
                    logger.warning(f"Failed to send pose data to LSL queue: {e}")
                    import traceback
                    traceback.print_exc()

            # Record LSL streaming time
            lsl_time = (time.time() - lsl_start) * 1000
            # FIX: Defensive check before accessing nested dict keys
            if 'timing' in stats and 'lsl_streaming' in stats['timing']:
                stats['timing']['lsl_streaming'].append(lsl_time)

            # Write display-ready data to GUI buffer
            write_start = time.time()
            success = self._write_display_data(cam_idx, frame_data['frame_id'],
                                             time.time(), display_faces, frame_bgr, pose_data, pose_metrics)
            write_time = (time.time() - write_start) * 1000
            # FIX: Defensive check before accessing nested dict keys
            if 'timing' in stats and 'display_write' in stats['timing']:
                stats['timing']['display_write'].append(write_time)
            
            if success:
                self.last_frame_ids[cam_idx] = frame_data['frame_id']
                # Track display write time for heartbeat fallback monitoring
                if not hasattr(self, '_last_display_write_time'):
                    self._last_display_write_time = {}
                self._last_display_write_time[cam_idx] = time.time()
                # FIX: Update last processed time for heartbeat logic
                if not hasattr(self, '_last_processed_frame'):
                    self._last_processed_frame = {}
                self._last_processed_frame[cam_idx] = time.time()
            else:
                logger.error(f"Failed to write display data for camera {cam_idx}")

            # Track total processing time
            total_time = (time.time() - overall_start) * 1000  # ms
            # FIX: Defensive check before accessing nested dict keys
            if 'timing' in stats and 'total_processing' in stats['timing']:
                stats['timing']['total_processing'].append(total_time)
            self.processing_times.append(total_time)  # Keep for backward compatibility

            # Track per-camera detection timing for FPS calculation
            current_time = time.time()
            if cam_idx in self._detection_timing_window:
                self._detection_timing_window[cam_idx].append((current_time, total_time))

        except Exception as e:
            logger.error(f"[DATA BRIDGE ERROR] Processing camera {cam_idx}: {e}")
            import traceback
            traceback.print_exc()

    def _clear_cached_layout_for_camera(self, cam_idx: int):
        """Clear any cached layouts for a specific camera to force recalculation."""
        if hasattr(self.buffer_coordinator, '_layout_cache'):
            # Clear layouts for this camera from BufferCoordinator's cache
            keys_to_remove = [key for key in self.buffer_coordinator._layout_cache.keys()
                            if key[0] == cam_idx]
            for key in keys_to_remove:
                del self.buffer_coordinator._layout_cache[key]
                logger.debug(f"[DATA BRIDGE] Cleared cached layout for camera {cam_idx}, key: {key}")

    def _read_resolution_from_buffer(self, shm, cam_idx: int) -> Optional[Tuple[int, int]]:
        """Read resolution from buffer header as ultimate source of truth."""
        try:
            # Resolution is stored at offset 8-16 in header (after write_index)
            # Layout: [write_index(8)][width(4)][height(4)]
            if shm.size < 16:
                logger.error(f"[DATA BRIDGE] Buffer too small to contain header: {shm.size} bytes")
                return None

            resolution_data = np.frombuffer(shm.buf, dtype=np.uint32, count=2, offset=8)
            width, height = int(resolution_data[0]), int(resolution_data[1])

            # Validate resolution is reasonable
            if width > 0 and height > 0 and width <= 7680 and height <= 4320:  # Max 8K resolution
                logger.debug(f"[DATA BRIDGE] Read resolution from buffer header for camera {cam_idx}: {width}x{height}")
                return (width, height)
            else:
                logger.warning(f"[DATA BRIDGE] Invalid resolution in buffer header: {width}x{height}")
                return None

        except Exception as e:
            logger.error(f"[DATA BRIDGE] Failed to read resolution from buffer: {e}")
            return None

    def _fetch_latest_frame_for_heartbeat(self, cam_idx: int) -> Optional[Dict[str, Any]]:
        """
        Fetch latest frame from camera buffer for display heartbeat (GUI freeze prevention).

        This is a lightweight frame fetch that bypasses full processing pipeline.
        Used when normal frame read returns None but display buffer needs updating.

        Returns:
            Dict with 'frame_id' and 'frame' (BGR numpy array), or None if fetch fails
        """
        try:
            # Validate camera buffer exists
            if cam_idx not in self.camera_buffers:
                return None

            cam_buffers = self.camera_buffers[cam_idx]
            if 'frame' not in cam_buffers:
                return None

            frame_shm = cam_buffers['frame']

            # CRITICAL FIX: Read ACTUAL resolution from buffer header (same pattern as _read_frame_data)
            # This prevents "buffer is too small" errors when camera outputs different resolution than configured
            # (e.g., OBS virtual camera outputting 1280x720 while config specifies 1920x1080)
            buffer_resolution = self._read_resolution_from_buffer(frame_shm, cam_idx)
            if not buffer_resolution:
                return None
            frame_w, frame_h = buffer_resolution

            # Get buffer layout
            layout = self.buffer_coordinator.get_buffer_layout(cam_idx, (frame_w, frame_h))

            # Read write index to find latest frame position
            write_index_offset = layout.get('write_index_offset', 0)
            write_index = np.ndarray((1,), dtype=np.uint64,
                                   buffer=frame_shm.buf[write_index_offset:write_index_offset+8])
            current_write_pos = int(write_index[0])

            # Get detection frame offsets
            detection_frame_offsets = layout.get('detection_frame_offsets', [])
            detection_metadata_offsets = layout.get('detection_metadata_offsets', [])

            if not detection_frame_offsets:
                return None

            # Calculate latest position (write_pos - 1 to get most recent complete frame)
            buffer_size = len(detection_frame_offsets)
            latest_pos = (current_write_pos - 1) % buffer_size

            # Read frame data
            frame_offset = detection_frame_offsets[latest_pos]
            frame_size = frame_w * frame_h * 3

            # Bounds check
            if frame_offset + frame_size > frame_shm.size:
                return None

            frame_data = np.ndarray((frame_h, frame_w, 3), dtype=np.uint8,
                                  buffer=frame_shm.buf[frame_offset:frame_offset + frame_size])
            frame_bgr = frame_data.copy()

            # Read frame_id from metadata
            metadata_offset = detection_metadata_offsets[latest_pos]
            metadata_view = np.ndarray((4,), dtype=np.int32,
                                     buffer=frame_shm.buf[metadata_offset:metadata_offset + 16])
            frame_id = int(metadata_view[0])

            return {
                'frame_id': frame_id,
                'frame': frame_bgr
            }

        except Exception as e:
            logger.error(f"[HEARTBEAT] Failed to fetch frame for camera {cam_idx}: {e}")
            return None

    def _read_frame_data(self, cam_idx: int) -> Optional[Dict[str, Any]]:
        """Read raw processing data from camera buffers (INPUT side of data bridge)."""
        # Debug entry every 1000 calls
        if not hasattr(self, 'read_frame_call_count'):
            self.read_frame_call_count = {}
        
        if cam_idx not in self.read_frame_call_count:
            self.read_frame_call_count[cam_idx] = 0
        
        self.read_frame_call_count[cam_idx] += 1

        # Validate camera buffer exists
        if cam_idx not in self.camera_buffers:
            logger.error(f"[DATA BRIDGE ERROR] Camera {cam_idx} not in camera_buffers. Available: {list(self.camera_buffers.keys())}")
            return None
            
        cam_buffers = self.camera_buffers[cam_idx]
        
        # Validate frame buffer exists
        if 'frame' not in cam_buffers:
            logger.error(f"[DATA BRIDGE ERROR] Camera {cam_idx} no frame buffer. Available buffers: {list(cam_buffers.keys())}")
            return None
            
        frame_shm = cam_buffers['frame']
        
        # Validate shared memory is still accessible
        try:
            # Test access to shared memory
            _ = frame_shm.buf[0]
        except Exception as e:
            logger.error(f"[DATA BRIDGE ERROR] Camera {cam_idx} frame buffer not accessible: {e}")
            return None
        
        if self._should_log_for_camera(cam_idx) and self.read_frame_call_count[cam_idx] % 1000 == 0:
            logger.debug(f"[DATA BRIDGE] Camera {cam_idx} using frame buffer: {frame_shm.name} (size: {frame_shm.size})")

        try:
            # Validate buffer coordinator exists
            if not hasattr(self, 'buffer_coordinator') or self.buffer_coordinator is None:
                logger.error(f"[DATA BRIDGE ERROR] No buffer coordinator available for layout")
                return None

            # PRIMARY FIX: Read resolution from buffer header FIRST (source of truth)
            # Buffer header contains the ACTUAL resolution used when creating the buffer,
            # which may differ from config if the camera doesn't support the requested resolution
            buffer_resolution = self._read_resolution_from_buffer(frame_shm, cam_idx)

            if not buffer_resolution:
                # Buffer header not readable - skip processing
                return None

            # Check if resolution has changed (resolution mismatch detection)
            cached_resolution = self.camera_resolutions.get(cam_idx)
            if cached_resolution and cached_resolution != buffer_resolution:
                logger.warning(f"[RESOLUTION MISMATCH] Camera {cam_idx}: cached={cached_resolution}, actual={buffer_resolution}")
                logger.warning(f"[RESOLUTION MISMATCH] Clearing layout cache and updating resolution")
                # Clear cached layouts for this camera
                self._clear_cached_layout_for_camera(cam_idx)

            # Update resolution caches to match actual buffer
            if cached_resolution != buffer_resolution:
                logger.info(f"[DATA BRIDGE] Camera {cam_idx} resolution from buffer header: {buffer_resolution}")
                self.camera_resolutions[cam_idx] = buffer_resolution
                # Sync to BufferCoordinator for consistency
                if hasattr(self.buffer_coordinator, 'camera_resolutions'):
                    self.buffer_coordinator.camera_resolutions[cam_idx] = buffer_resolution

            cam_resolution = buffer_resolution

            # Use centralized layout calculation with validated resolution
            layout = self.buffer_coordinator.get_buffer_layout(cam_idx, tuple(cam_resolution))
            
            # Read write index from frame buffer (camera writes here)
            write_index_offset = layout.get('write_index_offset', 0)
            
            # Validate write index offset is within buffer bounds
            if write_index_offset + 8 > frame_shm.size:
                logger.error(f"[DATA BRIDGE ERROR] Write index offset {write_index_offset} exceeds buffer size {frame_shm.size}")
                return None
                
            write_index = np.ndarray((1,), dtype=np.uint64,
                                   buffer=frame_shm.buf[write_index_offset:write_index_offset+8])
            current_write_pos = int(write_index[0])

            # OPTIMIZATION: Smart write_pos tracking to avoid reading same slot repeatedly
            # Track BOTH frame buffer write_pos AND results buffer write_index
            # This prevents 33× redundant buffer reads per frame (1ms polling vs 33ms camera)
            # BUT still reads data when landmarks change (even if frame doesn't)
            if not hasattr(self, '_last_write_state'):
                self._last_write_state = {}  # {cam_idx: frame_write_pos}
            if not hasattr(self, '_last_results_write_index'):
                self._last_results_write_index = {}  # {cam_idx: results_write_index}

            last_write_pos = self._last_write_state.get(cam_idx, -1)

            # Read results buffer write_index to detect landmark updates
            current_results_write_index = -1
            if cam_idx in self.camera_buffers and 'results' in self.camera_buffers[cam_idx]:
                results_shm = self.camera_buffers[cam_idx]['results']
                try:
                    # Results buffer write_index is at offset 0 (8 bytes, uint64)
                    results_write_index_view = np.ndarray((1,), dtype=np.uint64,
                                                          buffer=results_shm.buf[0:8])
                    current_results_write_index = int(results_write_index_view[0])
                except Exception as e:
                    # If we can't read results buffer, ignore it (results might not be ready yet)
                    pass

            last_results_write_index = self._last_results_write_index.get(cam_idx, -1)

            # Check if BOTH frame and results are unchanged
            frame_unchanged = (current_write_pos == last_write_pos and last_write_pos >= 0)
            results_unchanged = (current_results_write_index == last_results_write_index and last_results_write_index >= 0)

            # FIX: Less aggressive skip logic to prevent display buffer starvation
            # Track if we've processed data for display at least once
            if not hasattr(self, '_last_processed_frame'):
                self._last_processed_frame = {}

            # Skip ONLY if BOTH frame and results are unchanged AND we've processed recently
            # This optimization reduces CPU usage while ensuring we read when:
            # - Camera writes new frame (frame_unchanged = False)
            # - MediaPipe writes new results (results_unchanged = False)
            # - We haven't written to display buffer recently (prevents starvation)
            if frame_unchanged and results_unchanged:
                # Check if we've never processed data or it's been too long
                if last_write_pos < 0 or last_results_write_index < 0:
                    # No data available yet - safe to skip
                    if not hasattr(self, '_frame_skip_stats'):
                        self._frame_skip_stats = {}
                    if cam_idx not in self._frame_skip_stats:
                        self._frame_skip_stats[cam_idx] = {
                            'skips': 0, 'reads': 0,
                            'frame_only_changes': 0, 'results_only_changes': 0, 'both_changes': 0,
                            'last_log': time.time()
                        }
                    self._frame_skip_stats[cam_idx]['skips'] += 1
                    return None  # Skip buffer read entirely

                # Check if we need to force processing for display buffer heartbeat
                current_time = time.time()
                last_processed = self._last_processed_frame.get(cam_idx, 0)

                # Force processing at least every 100ms to prevent display buffer starvation
                if current_time - last_processed > 0.1:
                    logger.debug(f"[HEARTBEAT] Camera {cam_idx}: Forcing processing after {(current_time - last_processed)*1000:.1f}ms")
                    # Fall through to process the current data
                else:
                    # Safe to skip - we've processed recently
                    if not hasattr(self, '_frame_skip_stats'):
                        self._frame_skip_stats = {}
                    if cam_idx not in self._frame_skip_stats:
                        self._frame_skip_stats[cam_idx] = {
                            'skips': 0, 'reads': 0,
                            'frame_only_changes': 0, 'results_only_changes': 0, 'both_changes': 0,
                            'last_log': time.time()
                        }
                    self._frame_skip_stats[cam_idx]['skips'] += 1
                    return None  # Skip buffer read entirely

            # Update tracking for both buffers
            self._last_write_state[cam_idx] = current_write_pos
            if current_results_write_index >= 0:
                self._last_results_write_index[cam_idx] = current_results_write_index

            # Track successful reads and classify what changed
            if hasattr(self, '_frame_skip_stats') and cam_idx in self._frame_skip_stats:
                self._frame_skip_stats[cam_idx]['reads'] += 1

                # Classify what triggered this read
                if not frame_unchanged and not results_unchanged:
                    self._frame_skip_stats[cam_idx]['both_changes'] += 1
                elif not frame_unchanged:
                    self._frame_skip_stats[cam_idx]['frame_only_changes'] += 1
                elif not results_unchanged:
                    self._frame_skip_stats[cam_idx]['results_only_changes'] += 1

                # Log skip rate and change triggers every 10 seconds
                if time.time() - self._frame_skip_stats[cam_idx]['last_log'] >= 10.0:
                    stats = self._frame_skip_stats[cam_idx]
                    total = stats['skips'] + stats['reads']
                    skip_rate = (stats['skips'] / max(total, 1)) * 100
                    logger.info(f"[BUFFER SKIP STATS] Camera {cam_idx}: {skip_rate:.1f}% skipped "
                              f"({stats['skips']} skips, {stats['reads']} reads) | "
                              f"Changes: frame_only={stats['frame_only_changes']}, "
                              f"results_only={stats['results_only_changes']}, "
                              f"both={stats['both_changes']}")
                    # Reset counters
                    stats['skips'] = 0
                    stats['reads'] = 0
                    stats['frame_only_changes'] = 0
                    stats['results_only_changes'] = 0
                    stats['both_changes'] = 0
                    stats['last_log'] = time.time()

            # Debug logging for write position
            if self._should_log_for_camera(cam_idx) and self.read_frame_call_count[cam_idx] % 100 == 0:
                logger.debug(f"[DATA BRIDGE] Camera {cam_idx} frame buffer write_pos: {current_write_pos}")

            # FIX: Only skip position 0 if buffer is TRULY uninitialized (never written)
            # Position 0 is VALID when ring buffer wraps from position 15 → 0
            if not hasattr(self, '_buffer_initialized'):
                self._buffer_initialized = {}

            if current_write_pos == 0 and not self._buffer_initialized.get(cam_idx, False):
                # First time seeing this camera, and write_pos is still 0 (not yet written)
                if self._should_log_for_camera(cam_idx) and self.read_frame_call_count[cam_idx] % 1000 == 0:
                    logger.debug(f"[DATA BRIDGE] Camera {cam_idx} write_pos=0, buffer not yet initialized")
                return None  # No data written yet
            elif current_write_pos > 0:
                # Mark buffer as initialized once we see write_pos advance
                self._buffer_initialized[cam_idx] = True

            # Architecture B: Read from detection ring buffer (camera writes here)
            # Use detection frame offsets from layout
            detection_frame_offsets = layout.get('detection_frame_offsets', [])
            detection_metadata_offsets = layout.get('detection_metadata_offsets', [])

            # CRITICAL FIX: Use actual buffer size from layout (not stale config)
            # The layout contains the TRUE buffer size used when buffer was created
            detection_buffer_size = len(detection_frame_offsets)

            if not detection_frame_offsets:
                logger.error(f"[DATA BRIDGE ERROR] No detection frame offsets in layout")
                return None

            # LIFO OPTIMIZATION: Scan all ring buffer slots to find the one with the highest frame_id
            # This matches the pattern used in DisplayBufferManager and eliminates overlay lag
            # by always processing the most recent frame instead of sequential frames
            latest_frame_id = -1
            latest_pos = 0
            metadata_size = 64

            for slot in range(detection_buffer_size):
                # Get metadata offset for this slot
                if slot >= len(detection_metadata_offsets):
                    continue

                metadata_offset = detection_metadata_offsets[slot]

                # Bounds check
                if metadata_offset + metadata_size > frame_shm.size:
                    continue

                # Read metadata to get frame_id and ready flag
                try:
                    metadata_bytes = frame_shm.buf[metadata_offset:metadata_offset + metadata_size]
                    metadata = np.frombuffer(metadata_bytes, dtype=np.int64, count=8)
                    slot_frame_id = int(metadata[0])
                    ready_flag = int(metadata[3])

                    # Skip uninitialized or not-ready slots
                    if slot_frame_id < 0 or ready_flag != 1:
                        continue

                    # Update latest if this slot has newer frame
                    if slot_frame_id > latest_frame_id:
                        latest_frame_id = slot_frame_id
                        latest_pos = slot

                except Exception as e:
                    # Skip slots with read errors
                    continue

            # DIAGNOSTIC: Log LIFO scan result
            if not hasattr(self, '_lifo_scan_log_counter'):
                self._lifo_scan_log_counter = {}
            if cam_idx not in self._lifo_scan_log_counter:
                self._lifo_scan_log_counter[cam_idx] = 0
            self._lifo_scan_log_counter[cam_idx] += 1


            # If no valid slot found, return None
            if latest_frame_id < 0:
                return None

            if latest_pos >= len(detection_frame_offsets):
                logger.error(f"[DATA BRIDGE ERROR] Position {latest_pos} exceeds offsets array size {len(detection_frame_offsets)}")
                return None
            
            # Read frame from detection section with bounds checking
            frame_offset = detection_frame_offsets[latest_pos]
            cam_resolution = self.camera_resolutions.get(cam_idx)
            if not cam_resolution:
                logger.error(f"[DATA BRIDGE ERROR] No resolution for camera {cam_idx} (available: {self.camera_resolutions.keys()})")
                return None
            frame_w, frame_h = cam_resolution[0], cam_resolution[1]
            frame_size = layout.get('frame_size', frame_w * frame_h * 3)  # Use layout's frame_size

            # DEFENSIVE VALIDATION: Enhanced frame bounds check with detailed diagnostics
            if frame_offset + frame_size > frame_shm.size:
                logger.error(f"[DATA BRIDGE ERROR] Frame data ({frame_offset}+{frame_size}) exceeds buffer size {frame_shm.size}")
                logger.error(f"  Camera {cam_idx} resolution: {cam_resolution} ({frame_w}x{frame_h})")
                logger.error(f"  Latest position: {latest_pos}, Write position: {current_write_pos}")
                logger.error(f"  Expected frame size: {frame_size} bytes")
                logger.error(f"  Frame offset calculation: detection_frame_offsets[{latest_pos}] = {frame_offset}")
                logger.error(f"  Buffer coordinator resolution: {self.buffer_coordinator.camera_resolutions.get(cam_idx)}")
                # Clear any cached layouts that might be wrong
                self._clear_cached_layout_for_camera(cam_idx)
                return None

            try:
                # CONSISTENCY CHECK: Read write_index BEFORE reading frame data
                write_index_before = current_write_pos

                # Read frame data from shared memory
                frame_data = np.ndarray((frame_h, frame_w, 3), dtype=np.uint8,
                                      buffer=frame_shm.buf[frame_offset:frame_offset + frame_size])
                frame_bgr = frame_data.copy()  # Make a copy to avoid shared memory issues

                # CRITICAL FIX: Validate frame shape and contiguity to detect buffer artifacts
                expected_shape = (frame_h, frame_w, 3)
                if frame_bgr.shape != expected_shape:
                    logger.error(
                        f"[READER VALIDATION] Camera {cam_idx}: Frame shape mismatch! "
                        f"Expected {expected_shape}, got {frame_bgr.shape}. "
                        f"Buffer offset artifacts detected. Skipping frame {latest_frame_id}."
                    )
                    return None

                # Check for obviously corrupted data (coordinates > 10,000 or all zeros)
                if frame_bgr.size > 0:
                    # Check if frame is all zeros (completely black - likely corruption)
                    if np.all(frame_bgr == 0):
                        if not hasattr(self, '_zero_frame_count'):
                            self._zero_frame_count = {}
                        self._zero_frame_count[cam_idx] = self._zero_frame_count.get(cam_idx, 0) + 1
                        if self._zero_frame_count[cam_idx] % 100 == 1:
                            logger.warning(
                                f"[READER VALIDATION] Camera {cam_idx}: Frame {latest_frame_id} is all zeros. "
                                f"Possible buffer corruption or uninitialized memory."
                            )

                # Diagnostic logging for frame properties
                if latest_frame_id % 100 == 0:
                    logger.debug(
                        f"[READER VALIDATION] Camera {cam_idx} Frame {latest_frame_id}: "
                        f"shape={frame_bgr.shape}, dtype={frame_bgr.dtype}, "
                        f"contiguous={frame_bgr.flags['C_CONTIGUOUS']}, "
                        f"min={frame_bgr.min()}, max={frame_bgr.max()}"
                    )

                # CONSISTENCY CHECK: Read write_index AFTER reading frame data
                write_index_after = int(np.ndarray((1,), dtype=np.uint64,
                                                  buffer=frame_shm.buf[write_index_offset:write_index_offset+8])[0])

                # Verify no write occurred during read (atomic write pattern check)
                if write_index_after != write_index_before:
                    # Write occurred during read - frame data may be inconsistent
                    if not hasattr(self, '_consistency_check_failures'):
                        self._consistency_check_failures = {}
                    self._consistency_check_failures[cam_idx] = self._consistency_check_failures.get(cam_idx, 0) + 1

                    # Log occasionally to avoid spam
                    if self._consistency_check_failures[cam_idx] % 100 == 1:
                        logger.warning(f"Camera {cam_idx}: Write index changed during read ({write_index_before} → {write_index_after}), rejecting frame")

                    return None  # Reject potentially inconsistent frame

            except Exception as e:
                logger.error(f"[DATA BRIDGE ERROR] Failed to create frame array: {e}")
                return None
            
            # Use frame_id from LIFO scan (already validated as ready)
            # The LIFO scan above already checked ready_flag, so we can use latest_frame_id directly
            frame_id = latest_frame_id

            # FIX: Detect actual buffer overrun (ring buffer wraparound)
            # Frame ID going BACKWARDS indicates camera wrapped buffer before we read
            # Note: frame_id == last_frame_id is now prevented by write_pos tracking above
            if frame_id < self.last_frame_ids.get(cam_idx, -1):
                # Frame went BACKWARDS = camera wrapped ring buffer (serious issue!)
                if not hasattr(self, '_buffer_overrun_count'):
                    self._buffer_overrun_count = {}
                self._buffer_overrun_count[cam_idx] = self._buffer_overrun_count.get(cam_idx, 0) + 1

                if self._buffer_overrun_count[cam_idx] % 10 == 1:  # Log every 10 overruns
                    logger.error(f"[BUFFER OVERRUN] Camera {cam_idx} buffer wrapped! "
                               f"Missed frames: last_processed={self.last_frame_ids.get(cam_idx)}, "
                               f"current_frame={frame_id}, "
                               f"write_pos={current_write_pos}, buffer_pos={latest_pos} "
                               f"- increase ring_buffer_size in config!")
                return None  # Skip backwards frame
            # Note: Removed == case - duplicate frame_id now impossible due to write_pos tracking

            # Read landmarks from results buffer
            landmarks_data = self._read_landmarks(cam_idx, frame_id)

            # Read pose data from pose buffer
            pose_data = self._read_pose_data(cam_idx, frame_id)

            # DIAGNOSTIC: Log pose read result immediately
            if pose_data is not None and len(pose_data) > 0:
                logger.debug(f"[POSE PIPELINE-1] Camera {cam_idx}, frame {frame_id}: "
                           f"Successfully read {len(pose_data)} pose(s) from pose buffer")
            elif pose_data is not None and len(pose_data) == 0:
                logger.debug(f"[POSE PIPELINE-1] Camera {cam_idx}, frame {frame_id}: "
                           f"Read pose buffer but no poses detected (pose_data=[])")
            else:
                logger.debug(f"[POSE PIPELINE-1] Camera {cam_idx}, frame {frame_id}: "
                           f"Pose buffer read returned None")

            # DIAGNOSTIC: Track pose data reads
            if not hasattr(self, '_pose_read_diagnostic_count'):
                self._pose_read_diagnostic_count = {}
            if cam_idx not in self._pose_read_diagnostic_count:
                self._pose_read_diagnostic_count[cam_idx] = {'success': 0, 'none': 0, 'empty': 0}

            if pose_data is None:
                self._pose_read_diagnostic_count[cam_idx]['none'] += 1
            elif len(pose_data) == 0:
                self._pose_read_diagnostic_count[cam_idx]['empty'] += 1
            else:
                self._pose_read_diagnostic_count[cam_idx]['success'] += 1

            # Log every 30 frames
            total = sum(self._pose_read_diagnostic_count[cam_idx].values())
            if total % 30 == 0:
                stats = self._pose_read_diagnostic_count[cam_idx]
                logger.debug(f"[POSE READ DIAGNOSTIC] Camera {cam_idx}: success={stats['success']}, "
                           f"empty={stats['empty']}, none={stats['none']}")

            return {
                'frame_id': frame_id,
                'frame': frame_bgr,
                'landmarks': landmarks_data,
                'pose': pose_data,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"[DATA BRIDGE ERROR] Failed reading frame data for camera {cam_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_latest_frame_id(self, cam_idx: int) -> Optional[int]:
        """
        Get the absolute latest frame_id from camera buffer (LIFO optimization).
        Used to determine if current frame is stale and should be skipped.

        Returns:
            Latest frame_id available, or None if unable to read
        """
        try:
            if cam_idx not in self.camera_buffers:
                return None

            if 'frame' not in self.camera_buffers[cam_idx]:
                return None

            frame_shm = self.camera_buffers[cam_idx]['frame']

            # FIX: Use same layout retrieval as _read_frame_data() (was calling non-existent _get_layout_for_camera)
            cam_resolution = self.camera_resolutions.get(cam_idx)
            if not cam_resolution:
                return None

            layout = self.buffer_coordinator.get_buffer_layout(cam_idx, tuple(cam_resolution))
            if not layout:
                return None

            # Get detection metadata offsets from layout
            detection_metadata_offsets = layout.get('detection_metadata_offsets', [])
            if not detection_metadata_offsets:
                return None

            # LIFO SCAN: Find the slot with the highest frame_id (same as _read_frame_data)
            latest_frame_id = -1
            metadata_size = 64

            for slot in range(len(detection_metadata_offsets)):
                metadata_offset = detection_metadata_offsets[slot]

                # Bounds check
                if metadata_offset + metadata_size > frame_shm.size:
                    continue

                # Read metadata to get frame_id and ready flag
                try:
                    metadata_bytes = frame_shm.buf[metadata_offset:metadata_offset + metadata_size]
                    metadata = np.frombuffer(metadata_bytes, dtype=np.int64, count=8)
                    slot_frame_id = int(metadata[0])
                    ready_flag = int(metadata[3])

                    # Skip uninitialized or not-ready slots
                    if slot_frame_id < 0 or ready_flag != 1:
                        continue

                    # Update latest if this slot has newer frame
                    if slot_frame_id > latest_frame_id:
                        latest_frame_id = slot_frame_id

                except Exception:
                    # Skip slots with read errors
                    continue

            # DIAGNOSTIC: Log what _get_latest_frame_id() found (2nd LIFO scan)
            if not hasattr(self, '_get_latest_log_counter'):
                self._get_latest_log_counter = {}
            if cam_idx not in self._get_latest_log_counter:
                self._get_latest_log_counter[cam_idx] = 0
            self._get_latest_log_counter[cam_idx] += 1


            # Return latest frame_id found, or None if no valid frame
            return latest_frame_id if latest_frame_id >= 0 else None

        except Exception as e:
            # Log exceptions to catch silent failures (like missing _get_layout_for_camera bug)
            if not hasattr(self, '_get_latest_error_logged'):
                self._get_latest_error_logged = {}
            if cam_idx not in self._get_latest_error_logged:
                logger.error(f"[GET_LATEST ERROR] Camera {cam_idx} _get_latest_frame_id() failed: {e}")
                import traceback
                traceback.print_exc()
                self._get_latest_error_logged[cam_idx] = True  # Only log once per camera
            return None

    def _read_landmarks(self, cam_idx: int, target_frame_id: int) -> Optional[Dict]:
        """Read landmark data from results buffer."""
        if cam_idx not in self.camera_buffers:
            # DIAGNOSTIC: Unconditional log for camera 0
            if cam_idx == 0:
                logger.warning(f"❌ [DIAGNOSTIC] Camera {cam_idx} NOT in camera_buffers! Available: {list(self.camera_buffers.keys())}")
            elif self._should_log_for_camera(cam_idx, level="warning"):
                logger.warning(f"[LANDMARK READ] Camera {cam_idx} not in camera_buffers")
            return None

        if 'results' not in self.camera_buffers[cam_idx]:
            # DIAGNOSTIC: Unconditional log for camera 0
            if cam_idx == 0:
                logger.warning(f"❌ [DIAGNOSTIC] No 'results' buffer for camera {cam_idx}! Available buffers: {list(self.camera_buffers[cam_idx].keys())}")
            elif self._should_log_for_camera(cam_idx, level="warning"):
                logger.warning(f"[LANDMARK READ] No results buffer for camera {cam_idx}")
            return None

        results_shm = self.camera_buffers[cam_idx]['results']

        try:
            # Get actual buffer size
            actual_buffer_size = results_shm.size

            # DYNAMIC LAYOUT FIX: Call BufferCoordinator method directly instead of using cached layout
            # This ensures correct layout even when cameras are added dynamically
            results_layout = self.buffer_coordinator.get_results_buffer_layout()
            if not results_layout:
                logger.error("Failed to get results_buffer_layout from BufferCoordinator")
                return None

            # Get max_faces from BufferCoordinator directly (single source of truth)
            max_faces = self.buffer_coordinator.max_faces

            # DIAGNOSTIC: Log layout info for troubleshooting (first 5 reads only)
            if not hasattr(self, '_layout_diagnostic_count'):
                self._layout_diagnostic_count = {}
            if cam_idx not in self._layout_diagnostic_count:
                self._layout_diagnostic_count[cam_idx] = 0

            if self._layout_diagnostic_count[cam_idx] < 5:
                self._layout_diagnostic_count[cam_idx] += 1
                logger.info(f"[LAYOUT DIAGNOSTIC] Camera {cam_idx}:")
                logger.info(f"  max_faces: {max_faces}")
                logger.info(f"  actual_buffer_size: {actual_buffer_size}")
                logger.info(f"  expected_size: {results_layout['total_size']}")
                logger.info(f"  landmarks_size: {results_layout['landmarks_size']} (= {max_faces} * 478 * 3 * 4)")
                if cam_idx in self.camera_resolutions:
                    logger.info(f"  camera_resolution (from coordinator): {self.camera_resolutions[cam_idx]}")
                else:
                    logger.warning(f"  camera_resolution NOT in self.camera_resolutions! Available: {list(self.camera_resolutions.keys())}")
            landmarks_size = results_layout['landmarks_size']
            roi_metadata_offset = results_layout['roi_metadata_offset']
            roi_metadata_size = results_layout['roi_metadata_size']
            metadata_offset = results_layout['metadata_offset']
            metadata_size = results_layout['metadata_size']

            # DEBUG: Log the offset being used
            # if cam_idx == 0:
            #     print(f"[METADATA DEBUG] Metadata buffer offset: {metadata_offset}")

            # Validate buffer size against expected layout
            expected_size = results_layout['total_size']
            if actual_buffer_size < expected_size:
                logger.error(f"[BUFFER MISMATCH] Results buffer for camera {cam_idx}: actual={actual_buffer_size} < expected={expected_size}")
                logger.error(f"  Buffer name: {results_shm.name}")
                logger.error(f"  max_faces used: {max_faces}")
                logger.error(f"  landmarks_size: {landmarks_size} (max_faces={max_faces} * 478 * 3 * 4)")
                logger.error(f"  roi_metadata_size: {roi_metadata_size} (max_faces={max_faces} * roi_metadata_itemsize)")
                logger.error(f"  metadata_size: {metadata_size}")
                logger.error(f"  Layout from coordinator_info: {results_layout}")
                return None

            # Metadata structure (for reference - using centralized size)
            metadata_dtype = np.dtype([
                ('frame_id', 'int32'),
                ('timestamp_ms', 'int64'),
                ('n_faces', 'int32'),
                ('ready', 'int8'),
                ('is_detection_frame', 'int8'),
                ('track_count', 'int32'),
                ('processing_time_ms', 'float32'),
                ('gpu_memory_used_mb', 'float32'),
            ], align=True)  # CRITICAL: Must match BufferCoordinator alignment
            
            # Validate offset is within buffer bounds
            if metadata_offset + metadata_size > actual_buffer_size:
                logger.error(f"Metadata offset {metadata_offset} + size {metadata_size} exceeds buffer size {actual_buffer_size}")
                return None

            # FIX #1: Read write_index FIRST to detect stale data (CRITICAL for frozen overlay bug)
            # Results buffer is single-slot (not ring buffer), so write_index increments on each write
            # If write_index hasn't changed, data is stale (MediaPipe stopped writing)
            write_index = np.frombuffer(
                results_shm.buf[0:8],
                dtype=np.uint64,
                count=1
            )[0]

            # Check if write_index has changed since last read
            # Use .get() to handle dynamic camera addition (camera added after worker init)
            last_write_index = self.last_results_write_index.get(cam_idx, -1)
            if write_index <= last_write_index:
                # FIX #1b: Write index unchanged - return CACHED result to prevent flickering
                # GUI runs at 60 FPS, MediaPipe at 30 FPS, so this is NORMAL
                # Cache never expires - overlays remain visible during MediaPipe initialization/lag
                if cam_idx in self.cached_landmarks:
                    return self.cached_landmarks[cam_idx]['data']
                else:
                    # No cache yet - return None (initialization)
                    return None

            # Read metadata to check if data is ready
            metadata_view = np.ndarray(
                1, dtype=metadata_dtype,
                buffer=results_shm.buf[metadata_offset:metadata_offset + metadata_size]
            )[0]

            # Check if data is ready
            if metadata_view['ready'] != 1:
                return None

            frame_id = int(metadata_view['frame_id'])
            n_faces = int(metadata_view['n_faces'])

            # Frame synchronization: track lag but DON'T reject stale results (async rendering)
            frame_lag = target_frame_id - frame_id  # Positive = results behind, Negative = results ahead
            MAX_FRAME_LAG_AHEAD = 1   # Reject if results are >1 frame ahead
            MAX_FRAME_LAG_BEHIND = 10  # Mark as stale if >10 frames behind, but still render

            if frame_lag < -MAX_FRAME_LAG_AHEAD:
                # Results are ahead of camera frame - reject (future data)
                return None

            # Mark results as stale if significantly behind, but continue processing
            is_stale = frame_lag > MAX_FRAME_LAG_BEHIND

            if is_stale:
                # Track stale result usage for monitoring
                if not hasattr(self, '_stale_results_count'):
                    self._stale_results_count = {}
                self._stale_results_count[cam_idx] = self._stale_results_count.get(cam_idx, 0) + 1

                # Log occasionally for monitoring (every 30 stale results)
                if self._stale_results_count[cam_idx] % 30 == 1:
                    logger.info(f"[FRAME SYNC] Camera {cam_idx}: Using stale results "
                               f"(camera frame={target_frame_id}, results frame={frame_id}, lag={frame_lag} frames)")
            else:
                is_stale = False

            if n_faces == 0:
                # FIX #1: Update write_index even when no faces detected
                self.last_results_write_index[cam_idx] = write_index

                # Prepare result
                result = {
                    'frame_id': frame_id,
                    'n_faces': 0,
                    'landmarks': np.array([]),
                    'roi_metadata': [],
                    'timestamp': time.time(),
                    'is_stale': is_stale
                }

                # FIX #1b: Update cache even when no faces (prevents flickering)
                self.cached_landmarks[cam_idx] = {
                    'data': result,
                    'timestamp': time.time()
                }

                return result
            
            # Read landmarks (478 points, not 468!)
            # CRITICAL FIX: Skip 8-byte write index at offset 0
            landmarks_array = np.ndarray(
                (max_faces, 478, 3), dtype=np.float32,
                buffer=results_shm.buf[8:8 + landmarks_size]
            )

            # Read blendshapes array (52 blendshapes per face for correlator)
            blendshapes_offset = results_layout.get('blendshapes_offset', 0)
            blendshapes_size = results_layout.get('blendshapes_size', 0)
            blendshapes_array = None
            if blendshapes_offset > 0 and blendshapes_size > 0:
                try:
                    blendshapes_array = np.ndarray(
                        (max_faces, 52), dtype=np.float32,
                        buffer=results_shm.buf[blendshapes_offset:blendshapes_offset + blendshapes_size]
                    )
                except Exception as e:
                    logger.warning(f"Failed to read blendshapes for camera {cam_idx}: {e}")
                    blendshapes_array = None

            # Read ROI metadata using structured dtype for proper alignment
            # CRITICAL FIX: Now includes detection_index to map reordered faces back to original detection order
            # MediaPipe writes: [x1, y1, x2, y2, confidence, detection_index]
            roi_metadata_dtype = results_layout.get('roi_metadata_dtype')
            if roi_metadata_dtype is None:
                # Fallback dtype if not in layout (shouldn't happen with updated layout)
                roi_metadata_dtype = np.dtype([
                    ('x1', 'float32'),
                    ('y1', 'float32'),
                    ('x2', 'float32'),
                    ('y2', 'float32'),
                    ('confidence', 'float32'),
                    ('detection_index', 'int32')
                ], align=True)

            roi_metadata_raw = np.ndarray(
                (max_faces,), dtype=roi_metadata_dtype,
                buffer=results_shm.buf[roi_metadata_offset:roi_metadata_offset + roi_metadata_size]
            )

            # Extract only the valid faces
            all_landmarks = []
            all_blendshapes = []
            roi_metadata = []
            for i in range(n_faces):
                # Get MediaPipe landmarks (all 478 points including iris, keeping all 3 dimensions)
                # IMPORTANT: Keep the 3D shape (478x3) for consistency
                landmark_data = landmarks_array[i].copy()  # Shape: (478, 3) - no truncation
                all_landmarks.append(landmark_data)

                # Get blendshapes for this face (52 values for correlator)
                if blendshapes_array is not None:
                    blendshape_data = blendshapes_array[i].copy()  # Shape: (52,)
                    all_blendshapes.append(blendshape_data)

                # Get ROI metadata for this face
                # CRITICAL FIX: Now includes detection_index for correct embedding mapping
                # Format: [x1, y1, x2, y2, confidence, detection_index]
                roi_meta = {
                    'x1': float(roi_metadata_raw[i]['x1']),
                    'y1': float(roi_metadata_raw[i]['y1']),
                    'scale_x': float(roi_metadata_raw[i]['x2']),  # Actually x2 (not scale)
                    'scale_y': float(roi_metadata_raw[i]['y2']),  # Actually y2 (not scale)
                    'confidence': float(roi_metadata_raw[i]['confidence']),
                    'detection_index': int(roi_metadata_raw[i]['detection_index'])  # Original detection order
                }
                roi_metadata.append(roi_meta)

            # FIX #1: Update last_results_write_index after successful read
            # This prevents re-reading the same data on next call
            self.last_results_write_index[cam_idx] = write_index

            # Prepare result dictionary
            result = {
                'frame_id': frame_id,
                'n_faces': n_faces,
                'landmarks': np.array(all_landmarks) if all_landmarks else np.array([]),
                'blendshapes': np.array(all_blendshapes) if all_blendshapes else np.array([]),  # Add blendshapes for LSL/correlator
                'roi_metadata': roi_metadata,  # Include ROI metadata for coordinate transformation
                'timestamp': time.time(),
                'is_stale': is_stale
            }

            # FIX #1b: Update cache with new data (prevents flickering)
            self.cached_landmarks[cam_idx] = {
                'data': result,
                'timestamp': time.time()
            }

            return result
            
        except Exception as e:
            logger.error(f"[LANDMARK READ ERROR] Failed reading landmarks for camera {cam_idx}: {e}")
            logger.error(f"  Buffer: {results_shm.name if results_shm else 'None'}")
            logger.error(f"  Target frame: {target_frame_id}")
            import traceback
            traceback.print_exc()
            return None

    def _read_pose_data(self, cam_idx: int, target_frame_id: int) -> Optional[List[Dict]]:
        """Read pose data from pose buffer."""
        # DIAGNOSTIC: Track _read_pose_data calls
        if not hasattr(self, '_read_pose_data_call_count'):
            self._read_pose_data_call_count = {}
        if cam_idx not in self._read_pose_data_call_count:
            self._read_pose_data_call_count[cam_idx] = 0
        self._read_pose_data_call_count[cam_idx] += 1


        if cam_idx not in self.camera_buffers:
            if not hasattr(self, '_pose_buffer_missing_logged'):
                self._pose_buffer_missing_logged = set()
            if cam_idx not in self._pose_buffer_missing_logged:
                logger.warning(f"[POSE DEBUG] Camera {cam_idx} not in camera_buffers (pose disabled?)")
                logger.warning(f"[POSE DEBUG] Available cameras in camera_buffers: {list(self.camera_buffers.keys())}")
                self._pose_buffer_missing_logged.add(cam_idx)
            return None

        if 'pose' not in self.camera_buffers[cam_idx]:
            # Pose is optional, log once per camera
            if not hasattr(self, '_pose_not_available_logged'):
                self._pose_not_available_logged = set()
            if cam_idx not in self._pose_not_available_logged:
                logger.warning(f"[POSE DEBUG] No pose buffer for camera {cam_idx} (pose detection not enabled)")
                logger.warning(f"[POSE DEBUG] Available buffer types for camera {cam_idx}: {list(self.camera_buffers[cam_idx].keys())}")
                self._pose_not_available_logged.add(cam_idx)
            return None

        pose_shm = self.camera_buffers[cam_idx]['pose']

        try:
            # Get pose buffer layout
            pose_layout = self.buffer_coordinator.get_pose_buffer_layout(cam_idx)

            write_index_offset = pose_layout['write_index_offset']
            write_index_size = pose_layout['write_index_size']
            ring_buffer_size = pose_layout['ring_buffer_size']
            pose_offsets = pose_layout['pose_offsets']
            pose_slot_size = pose_layout['pose_slot_size']
            metadata_offsets = pose_layout['metadata_offsets']
            metadata_size = pose_layout['metadata_size']
            max_poses = pose_layout['max_persons']
            landmarks_per_pose = pose_layout['keypoints_per_person']
            values_per_landmark = pose_layout['values_per_keypoint']

            # STEP 1: Read write_index FIRST (before reading data)
            write_idx_bytes = bytes(pose_shm.buf[write_index_offset:write_index_offset + write_index_size])
            write_idx_before = int.from_bytes(write_idx_bytes, byteorder='little')

            # Calculate ring buffer slot index
            # CRITICAL: Writer increments write_idx AFTER writing data, so we need to subtract 1
            # to get the most recently written slot (prevents reading from empty/future slots)
            slot_idx = (write_idx_before - 1) % ring_buffer_size

            # STEP 2: Read metadata from ring buffer slot
            # Force copy with bytes() to prevent memoryview caching (fixes frozen skeleton bug)
            metadata_dtype = pose_layout['metadata_dtype']
            metadata_offset = metadata_offsets[slot_idx]
            metadata_bytes = bytes(pose_shm.buf[metadata_offset:metadata_offset + metadata_dtype.itemsize])
            metadata = np.frombuffer(metadata_bytes, dtype=metadata_dtype)[0]

            frame_id = metadata['frame_id']
            timestamp_ms = metadata['timestamp_ms']
            n_poses = metadata['n_persons']

            # Extract timing data from metadata for performance metrics
            processing_time_ms = float(metadata['processing_time_ms'])
            detection_time_ms = float(metadata['detection_time_ms'])
            pose_time_ms = float(metadata['pose_time_ms'])

            # Track pose timing for FPS calculation
            current_time = time.time()
            if cam_idx in self._pose_timing_window:
                self._pose_timing_window[cam_idx].append(
                    (current_time, detection_time_ms, pose_time_ms)
                )

            # Get actual camera resolution (native resolution, not hardcoded 480p)
            pose_w, pose_h = self.camera_resolutions.get(cam_idx, (1920, 1080))

            # FRAME SYNCHRONIZATION: Check if pose results match camera frame (reduce lag)
            # Same pattern as face landmarks - reject future/very stale data
            frame_lag = target_frame_id - frame_id  # Positive = results behind, Negative = results ahead
            MAX_FRAME_LAG_AHEAD = 1    # Reject if results are >1 frame ahead
            MAX_FRAME_LAG_BEHIND = 10  # Mark as stale if >10 frames behind, but still render

            # OCCLUSION RECOVERY: Stateful tracking instead of frame_lag-based detection
            # Frame_lag-based recovery was broken because ring buffer (4 slots) gets overwritten every 0.13s at 30 FPS
            # So frame_lag is always 0-4 frames, never >100, and recovery never triggered
            # Solution: Track occlusion state across frames based on consecutive no-pose detections

            # Initialize occlusion state for this camera
            if cam_idx not in self._occlusion_state:
                self._occlusion_state[cam_idx] = {'in_occlusion': False, 'recovery_frames_left': 0}
            if cam_idx not in self._consecutive_no_pose:
                self._consecutive_no_pose[cam_idx] = 0

            # Track detection state and update occlusion/recovery status
            OCCLUSION_ENTRY_THRESHOLD = 10  # Enter occlusion after 10 consecutive no-pose frames
            RECOVERY_DURATION_FRAMES = 30  # Stay in recovery mode for 30 frames (1 second at 30 FPS)

            if n_poses == 0:
                # No pose detected - increment consecutive counter
                self._consecutive_no_pose[cam_idx] += 1

                # Enter occlusion state after threshold consecutive no-pose frames
                if self._consecutive_no_pose[cam_idx] >= OCCLUSION_ENTRY_THRESHOLD:
                    if not self._occlusion_state[cam_idx]['in_occlusion']:
                        logger.info(f"[OCCLUSION] Camera {cam_idx}: Entering occlusion state "
                                   f"(no poses for {self._consecutive_no_pose[cam_idx]} consecutive frames)")
                        self._occlusion_state[cam_idx]['in_occlusion'] = True
            else:
                # Pose detected!
                if self._occlusion_state[cam_idx]['in_occlusion']:
                    # Just exited occlusion → enter recovery mode
                    logger.info(f"[POSE RECOVERY] Camera {cam_idx}: Exiting occlusion, starting {RECOVERY_DURATION_FRAMES}-frame recovery period "
                               f"(was occluded for {self._consecutive_no_pose[cam_idx]} frames)")
                    self._occlusion_state[cam_idx]['in_occlusion'] = False
                    self._occlusion_state[cam_idx]['recovery_frames_left'] = RECOVERY_DURATION_FRAMES

                # Reset consecutive counter when pose detected
                self._consecutive_no_pose[cam_idx] = 0

            # Determine if currently in recovery mode
            recovering_from_occlusion = (self._occlusion_state[cam_idx]['recovery_frames_left'] > 0)

            # Decrement recovery counter if in recovery mode
            if recovering_from_occlusion:
                recovery_frame_num = RECOVERY_DURATION_FRAMES - self._occlusion_state[cam_idx]['recovery_frames_left'] + 1
                if recovery_frame_num <= 5:  # Log first 5 recovery frames
                    logger.info(f"[POSE RECOVERY] Camera {cam_idx}: Recovery frame {recovery_frame_num}/{RECOVERY_DURATION_FRAMES}")
                self._occlusion_state[cam_idx]['recovery_frames_left'] -= 1

            if frame_lag < -MAX_FRAME_LAG_AHEAD:
                # Results are ahead of camera frame - reject (future data, likely race condition)
                if not hasattr(self, '_pose_future_reject_count'):
                    self._pose_future_reject_count = {}
                self._pose_future_reject_count[cam_idx] = self._pose_future_reject_count.get(cam_idx, 0) + 1
                if self._pose_future_reject_count[cam_idx] % 30 == 1:
                    logger.debug(f"[POSE SYNC] Camera {cam_idx}: Rejecting future pose data "
                                f"(camera frame={target_frame_id}, pose frame={frame_id}, lag={frame_lag} frames)")
                return None

            # Mark results as stale if significantly behind (but continue processing for smoothness)
            is_stale = frame_lag > MAX_FRAME_LAG_BEHIND

            if is_stale:
                # Track stale result usage for monitoring
                if not hasattr(self, '_pose_stale_results_count'):
                    self._pose_stale_results_count = {}
                self._pose_stale_results_count[cam_idx] = self._pose_stale_results_count.get(cam_idx, 0) + 1

                # Log occasionally for monitoring (every 30 stale results)
                if self._pose_stale_results_count[cam_idx] % 30 == 1:
                    logger.info(f"[POSE SYNC] Camera {cam_idx}: Using stale pose results "
                               f"(camera frame={target_frame_id}, pose frame={frame_id}, lag={frame_lag} frames)")

            # DEBUG: Track frame_id changes to detect freeze
            if not hasattr(self, '_last_pose_frame_id'):
                self._last_pose_frame_id = {}
            last_frame_id = self._last_pose_frame_id.get(cam_idx, -1)
            frame_changed = (frame_id != last_frame_id)
            self._last_pose_frame_id[cam_idx] = frame_id

            # Debug log for pose resolution (first time only)
            if not hasattr(self, '_pose_resolution_logged'):
                logger.info(f"[GUI Worker] Pose metadata includes resolution: {pose_w}×{pose_h}")
                self._pose_resolution_logged = True

            # Debug logging: track no-pose state
            # CRITICAL: During occlusion recovery, bypass this check to avoid reading stale metadata
            # The ring buffer may have old n_persons=0 in slots that haven't been overwritten yet
            if n_poses == 0 and not recovering_from_occlusion:
                if not hasattr(self, '_no_pose_logged'):
                    self._no_pose_logged = {}
                if cam_idx not in self._no_pose_logged:
                    logger.info(f"[GUI Worker] No poses detected for camera {cam_idx}")
                    self._no_pose_logged[cam_idx] = True
                return []  # No poses detected (but allow through during recovery)

            # During recovery, log that we're bypassing n_poses check even if metadata shows 0
            if recovering_from_occlusion and n_poses == 0:
                if not hasattr(self, '_recovery_bypass_logged'):
                    self._recovery_bypass_logged = {}
                if cam_idx not in self._recovery_bypass_logged or self._recovery_bypass_logged.get(cam_idx, 0) < 3:
                    logger.info(f"[POSE RECOVERY] Camera {cam_idx}: Bypassing n_poses=0 check during recovery "
                               f"(frame_lag={frame_lag} frames, reading stale metadata from ring buffer slot)")
                    self._recovery_bypass_logged[cam_idx] = self._recovery_bypass_logged.get(cam_idx, 0) + 1
                # Continue processing to force reader to cycle through slots and find fresh data

            # Clear no-pose flag when poses detected
            if hasattr(self, '_no_pose_logged') and cam_idx in self._no_pose_logged:
                del self._no_pose_logged[cam_idx]
                logger.info(f"[GUI Worker] ✅ Pose detection resumed for camera {cam_idx}")

            # Log first successful pose detection
            if not hasattr(self, '_first_pose_detected'):
                self._first_pose_detected = set()
            if cam_idx not in self._first_pose_detected:
                logger.info(f"✅ [POSE] First pose detection for camera {cam_idx}: {n_poses} pose(s) in frame {frame_id}")
                self._first_pose_detected.add(cam_idx)

            # Periodic debug logging (every 50 frames)
            if not hasattr(self, '_pose_read_count'):
                self._pose_read_count = {}
            if cam_idx not in self._pose_read_count:
                self._pose_read_count[cam_idx] = 0

            self._pose_read_count[cam_idx] += 1
            if self._pose_read_count[cam_idx] % 50 == 0:
                logger.info(f"[GUI Worker] Read pose data: camera {cam_idx}, {n_poses} pose(s), frame {frame_id}")

            # STEP 3: Read pose data from ring buffer slot
            # Force copy with bytes() to prevent memoryview caching (fixes frozen skeleton bug)
            pose_data_offset = pose_offsets[slot_idx]
            pose_data_bytes = bytes(pose_shm.buf[pose_data_offset:pose_data_offset + pose_slot_size])
            pose_array = np.frombuffer(pose_data_bytes, dtype=np.float32)

            # STEP 4: Consistency check - re-read write_index to verify data wasn't modified during read
            write_idx_bytes_after = bytes(pose_shm.buf[write_index_offset:write_index_offset + write_index_size])
            write_idx_after = int.from_bytes(write_idx_bytes_after, byteorder='little')

            # Initialize consecutive failure tracking for recovery mechanism
            if not hasattr(self, '_pose_consistency_fail_streak'):
                self._pose_consistency_fail_streak = {}
            if not hasattr(self, '_pose_consistency_fail_count'):
                self._pose_consistency_fail_count = {}

            # CONSISTENCY CHECK BYPASS: Skip during occlusion recovery
            # recovering_from_occlusion was calculated earlier (line 2222) based on frame_lag
            # During long occlusion, consistency check often fails because reader/writer are desynchronized
            # Solution: Skip consistency check during recovery - prioritize showing ANY pose data over waiting
            if recovering_from_occlusion:
                # Skip consistency check during occlusion recovery
                if not hasattr(self, '_occlusion_recovery_logged'):
                    self._occlusion_recovery_logged = {}
                if cam_idx not in self._occlusion_recovery_logged or self._occlusion_recovery_logged.get(cam_idx, 0) < 5:
                    logger.info(f"[POSE RECOVERY] Camera {cam_idx}: Skipping consistency check during occlusion recovery "
                               f"(frame_lag={frame_lag} frames, write_index: {write_idx_before} -> {write_idx_after})")
                    self._occlusion_recovery_logged[cam_idx] = self._occlusion_recovery_logged.get(cam_idx, 0) + 1
                # Reset streak counter since we're bypassing check
                self._pose_consistency_fail_streak[cam_idx] = 0
                # Continue processing without consistency check
            # If write_index advanced by more than 1, data may be inconsistent - reject this frame
            # Note: Allow +1 advance since writer may legitimately write one more frame during our read (~1-2ms)
            # The 4-slot ring buffer provides sufficient collision protection
            elif write_idx_after > write_idx_before + 1:
                self._pose_consistency_fail_count[cam_idx] = self._pose_consistency_fail_count.get(cam_idx, 0) + 1
                self._pose_consistency_fail_streak[cam_idx] = self._pose_consistency_fail_streak.get(cam_idx, 0) + 1

                # Recovery mechanism: After 10 consecutive failures, force accept data to prevent permanent freeze
                # This prevents freeze bug after occlusion where reader can never catch up to writer
                if self._pose_consistency_fail_streak[cam_idx] >= 10:
                    logger.warning(f"[POSE RECOVERY] Camera {cam_idx}: Forcing pose data after "
                                 f"{self._pose_consistency_fail_streak[cam_idx]} consecutive consistency failures "
                                 f"(write_index: {write_idx_before} -> {write_idx_after})")
                    # Reset streak counter and accept data (better stale data than frozen visualization)
                    self._pose_consistency_fail_streak[cam_idx] = 0
                    # Continue processing despite inconsistency
                else:
                    # Standard rejection with debug logging
                    if self._pose_consistency_fail_count[cam_idx] % 30 == 1:
                        logger.debug(f"[POSE CONSISTENCY] Camera {cam_idx}: Rejecting inconsistent pose data "
                                    f"(write_index jumped: {write_idx_before} -> {write_idx_after}, "
                                    f"consecutive failures: {self._pose_consistency_fail_streak[cam_idx]})")
                    return None
            else:
                # Reset consecutive failure streak on successful read
                if cam_idx in self._pose_consistency_fail_streak:
                    if self._pose_consistency_fail_streak[cam_idx] > 0:
                        logger.debug(f"[POSE RECOVERY] Camera {cam_idx}: Consistency check passed, "
                                   f"reset streak after {self._pose_consistency_fail_streak[cam_idx]} failures")
                    self._pose_consistency_fail_streak[cam_idx] = 0

                # Reset occlusion recovery logs when no longer in recovery mode
                if hasattr(self, '_occlusion_recovery_logged') and cam_idx in self._occlusion_recovery_logged:
                    if not recovering_from_occlusion:
                        del self._occlusion_recovery_logged[cam_idx]

                # Also reset n_poses bypass log when no longer in recovery mode
                if hasattr(self, '_recovery_bypass_logged') and cam_idx in self._recovery_bypass_logged:
                    if not recovering_from_occlusion:
                        del self._recovery_bypass_logged[cam_idx]

            # Parse poses
            poses = []
            for i in range(n_poses):
                start_idx = i * landmarks_per_pose * values_per_landmark
                end_idx = start_idx + (landmarks_per_pose * values_per_landmark)
                pose_values = pose_array[start_idx:end_idx]

                # Reshape to (33, 4) for landmarks
                landmarks = pose_values.reshape((landmarks_per_pose, values_per_landmark))

                # Extract centroid from first two values (stored by PoseProcess)
                centroid = (float(landmarks[0, 0]), float(landmarks[0, 1]))

                # Keep as numpy array (133, 4) - STANDARD FORMAT [x, y, z, confidence]
                poses.append({
                    'keypoints': landmarks,  # Shape (133, 4) numpy array
                    'centroid': centroid,
                    'pose_resolution': (pose_w, pose_h)
                })

            # DIAGNOSTIC: Check shoulder coordinates RAW from buffer (before smoothing/matching)
            if n_poses > 0:
                if not hasattr(self, '_pose_raw_shoulder_log_count'):
                    self._pose_raw_shoulder_log_count = {}
                if cam_idx not in self._pose_raw_shoulder_log_count:
                    self._pose_raw_shoulder_log_count[cam_idx] = 0
                self._pose_raw_shoulder_log_count[cam_idx] += 1

                # Removed: False positive pose anomaly detection logging
                # The pose skeleton is rendered correctly, confirming landmarks are valid.
                # These checks were triggering on normal poses at camera angles.

            # NOTE: Pose-to-participant matching moved to _process_camera() (after face buffer population)
            # to ensure faces from current frame are available in _display_faces_buffer.
            # Previously matching happened here, but buffer was empty/stale.

            # Apply temporal smoothing to reduce jitter
            # NOTE: Smoothing now happens BEFORE participant matching (in _process_camera)
            # This is suboptimal but necessary due to execution order constraints.
            # TODO: Revisit smoothing order if accuracy issues arise.
            if poses and hasattr(self, 'pose_smoother'):
                poses = self.pose_smoother.smooth(poses, cam_idx, frame_id)

                # Removed: Post-smoothing pose anomaly detection logging
                # (See above for why these checks are false positives)

            return poses

        except Exception as e:
            logger.error(f"[POSE READ ERROR] Failed reading pose for camera {cam_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _match_poses_to_participants(self, poses: List[Dict], cam_idx: int,
                                     frame_id: int) -> List[Dict]:
        """
        Match poses to participants by head centroid proximity BEFORE smoothing.

        CRITICAL FIX: This prevents EMA from smoothing across different people when
        MediaPipe detection order changes between frames.

        Args:
            poses: List of pose dicts with 'landmarks', 'centroid', 'visibility'
            cam_idx: Camera index
            frame_id: Current frame ID

        Returns:
            List of pose dicts with added 'participant_id' field (or None if unmatched)
        """
        if not poses:
            return poses

        # Get configuration for matching
        match_config = self.config.get('pose_to_participant_matching', {})
        max_distance = match_config.get('max_head_distance_px', 150)
        frame_tolerance = match_config.get('frame_tolerance', 3)

        # EMERGENCY DIAGNOSTIC: Verify function entry and config loading (PERIODIC)
        if not hasattr(self, '_emergency_diagnostic_logged'):
            self._emergency_diagnostic_logged = {}
        if cam_idx not in self._emergency_diagnostic_logged:
            self._emergency_diagnostic_logged[cam_idx] = 0
        self._emergency_diagnostic_logged[cam_idx] += 1


        # Find display_faces from nearby frames (relaxed frame sync)
        matched_faces = []
        matched_frame_id = frame_id

        # DIAGNOSTIC: Track matching start (log first 3 calls)
        if not hasattr(self, '_pose_match_diagnostic_count'):
            self._pose_match_diagnostic_count = {}
        if cam_idx not in self._pose_match_diagnostic_count:
            self._pose_match_diagnostic_count[cam_idx] = 0
        self._pose_match_diagnostic_count[cam_idx] += 1


        if cam_idx in self._display_faces_buffer:
            best_match = None
            best_distance = float('inf')
            newest_faces_fallback = None  # Track newest non-empty faces as fallback


            # Smart search: Skip empty frames (cache artifacts), prioritize closest non-empty frame
            for buffered_frame_id, buffered_faces in self._display_faces_buffer[cam_idx]:
                # CRITICAL FIX: Skip empty face lists (65% of frames are empty due to cache staleness)
                if not buffered_faces:
                    continue

                # Track newest non-empty faces for fallback (regardless of tolerance)
                if newest_faces_fallback is None or buffered_frame_id > newest_faces_fallback[0]:
                    newest_faces_fallback = (buffered_frame_id, buffered_faces)

                # Check if within tolerance window
                distance = abs(buffered_frame_id - frame_id)
                if distance <= frame_tolerance:
                    if distance < best_distance:
                        best_distance = distance
                        best_match = (buffered_frame_id, buffered_faces)

            # Use best match within tolerance, or fallback to newest faces
            if best_match:
                matched_frame_id, matched_faces = best_match
            elif newest_faces_fallback:
                # Fallback: No faces within ±20 frames, use newest available faces (ignore tolerance)
                matched_frame_id, matched_faces = newest_faces_fallback


        # If no faces found, mark all poses as unmatched
        if not matched_faces:
            for pose in poses:
                pose['participant_id'] = None
            return poses

        # Match each pose to nearest participant by head centroid
        # FIX: Track used participant IDs to prevent duplicate assignments
        used_participant_ids = set()

        for pose_idx, pose in enumerate(poses):
            # Calculate HEAD centroid from pose landmarks (nose + eyes for tight face center)
            landmarks = pose['keypoints']  # 33 MediaPipe Pose landmarks
            head_indices = [0, 1, 2, 4, 5]  # Nose + left eye (inner, center) + right eye (inner, center)
            # Tightened from [0-8] (nose+eyes+ears) to exclude ears for better face center alignment

            try:
                head_points = [landmarks[i] for i in head_indices if i < len(landmarks)]
                if not head_points:
                    pose['participant_id'] = None
                    if self._pose_match_diagnostic_count[cam_idx] % 30 == 1 and pose_idx == 0:
                        logger.warning(f"[POSE MATCH] Pose {pose_idx}: No head_points extracted")
                    continue

                # Calculate centroid in NORMALIZED space [0,1]
                head_centroid_norm = (
                    sum(p[0] for p in head_points) / len(head_points),
                    sum(p[1] for p in head_points) / len(head_points)
                )

                # CRITICAL: Convert from normalized to PIXEL space using CAMERA resolution
                # Pose landmarks are normalized [0,1], must convert to same space as face bbox
                # Face bbox and pose buffer now both use native camera resolution
                camera_resolution = self.camera_resolutions.get(cam_idx, (1920, 1080))
                camera_w, camera_h = camera_resolution

                head_centroid = (
                    head_centroid_norm[0] * camera_w,  # Convert using camera resolution
                    head_centroid_norm[1] * camera_h   # Now matches face coordinate space!
                )

                # DIAGNOSTIC: Log first pose's head centroid
                if self._pose_match_diagnostic_count[cam_idx] % 30 == 1 and pose_idx == 0:
                    logger.warning(f"[POSE MATCH] Pose {pose_idx}: head_centroid=({head_centroid[0]:.1f}, {head_centroid[1]:.1f}) pixels "
                                 f"(normalized=({head_centroid_norm[0]:.3f}, {head_centroid_norm[1]:.3f})), "
                                 f"from {len(head_points)} head points, camera_resolution={camera_w}x{camera_h}")
            except Exception as e:
                logger.warning(f"[POSE MATCH] Failed to calculate head centroid: {e}")
                pose['participant_id'] = None
                continue

            # Find nearest face by centroid distance
            best_participant_id = None
            min_distance = float('inf')

            for face_idx, face in enumerate(matched_faces):
                # FIX: Skip faces already matched to other poses (one-to-one validation)
                if face.participant_id in used_participant_ids:
                    continue

                # Calculate face centroid from bbox
                face_centroid = (
                    (face.bbox[0] + face.bbox[2]) / 2.0,  # (x1 + x2) / 2
                    (face.bbox[1] + face.bbox[3]) / 2.0   # (y1 + y2) / 2
                )
                distance = np.sqrt((head_centroid[0] - face_centroid[0])**2 +
                                 (head_centroid[1] - face_centroid[1])**2)

                # DIAGNOSTIC: Log first face comparison
                if self._pose_match_diagnostic_count[cam_idx] % 30 == 1 and pose_idx == 0 and face_idx == 0:
                    logger.warning(f"[POSE MATCH] Face {face_idx}: participant_id={face.participant_id}, "
                                 f"bbox={face.bbox}, "
                                 f"face_centroid=({face_centroid[0]:.1f}, {face_centroid[1]:.1f}), "
                                 f"distance={distance:.1f}px")

                if distance < min_distance:
                    min_distance = distance
                    best_participant_id = face.participant_id

            # DIAGNOSTIC: Log matching result (first pose only)
            if self._pose_match_diagnostic_count[cam_idx] % 30 == 1 and pose_idx == 0:
                logger.warning(f"[POSE MATCH RESULT] Pose {pose_idx}: "
                             f"best_participant_id={best_participant_id}, "
                             f"min_distance={min_distance:.1f}px, "
                             f"compared {len(matched_faces)} faces")

            # Assign participant_id to nearest face (no distance threshold - always match)
            pose['participant_id'] = best_participant_id

            # FIX: Mark this participant_id as used to prevent duplicate assignments
            if best_participant_id is not None:
                used_participant_ids.add(best_participant_id)

        # DIAGNOSTIC: Check for shallow copy bug (poses sharing landmarks arrays)
        if len(poses) > 1:
            landmark_ids = [id(p['keypoints']) for p in poses]
            unique_ids = set(landmark_ids)
            if len(landmark_ids) != len(unique_ids):
                logger.error(f"🚨 [SHALLOW COPY BUG] Camera {cam_idx} frame {frame_id}: "
                           f"Multiple poses share same landmarks array! "
                           f"landmark_ids={landmark_ids}, unique={len(unique_ids)}/{len(landmark_ids)}")
                # Log which poses share arrays
                from collections import Counter
                id_counts = Counter(landmark_ids)
                for lm_id, count in id_counts.items():
                    if count > 1:
                        sharing_indices = [i for i, pid in enumerate(landmark_ids) if pid == lm_id]
                        participant_ids = [poses[i].get('participant_id', 'None') for i in sharing_indices]
                        logger.error(f"  Landmarks array {lm_id} shared by poses {sharing_indices} "
                                   f"(participant_ids={participant_ids})")

        return poses

    def _send_pose_data_to_lsl(self, cam_idx: int, frame_id: int,
                               pose_data: List[Dict], display_faces: List) -> None:
        """
        Send pose data to LSL process using pre-assigned participant IDs.

        NOTE: Poses already have participant_id assigned by _match_poses_to_participants()
        called upstream. This method simply formats and sends to LSL.

        Args:
            cam_idx: Camera index
            frame_id: Current frame ID from pose detection
            pose_data: List of pose dicts with 'landmarks', 'centroid', 'visibility', 'participant_id'
            display_faces: List of DisplayFaceData objects (unused, kept for backward compat)
        """
        # DIAGNOSTIC: Track function entry
        if not hasattr(self, '_pose_lsl_entry_count'):
            self._pose_lsl_entry_count = 0
        self._pose_lsl_entry_count += 1

        if self._pose_lsl_entry_count % 30 == 1:
            logger.info(f"[POSE LSL ENTRY] _send_pose_data_to_lsl called {self._pose_lsl_entry_count} times")

        if self.verbose_debug:
            logger.info(f"[POSE LSL DEBUG] _send_pose_data_to_lsl ENTERED: "
                      f"cam={cam_idx}, frame={frame_id}, "
                      f"poses={len(pose_data) if pose_data else 0}, "
                      f"faces={len(display_faces) if display_faces else 0}")

        if not pose_data:
            if self.verbose_debug:
                logger.info(f"[POSE LSL DEBUG] Early return: pose_data is empty")
            return

        # DIAGNOSTIC: Track participant_id assignment status
        valid_ids = sum(1 for p in pose_data if p.get('participant_id') is not None)
        none_ids = len(pose_data) - valid_ids

        if not hasattr(self, '_pose_participant_id_diagnostic'):
            self._pose_participant_id_diagnostic = {}
        if cam_idx not in self._pose_participant_id_diagnostic:
            self._pose_participant_id_diagnostic[cam_idx] = {'total': 0, 'with_id': 0, 'without_id': 0}

        self._pose_participant_id_diagnostic[cam_idx]['total'] += len(pose_data)
        self._pose_participant_id_diagnostic[cam_idx]['with_id'] += valid_ids
        self._pose_participant_id_diagnostic[cam_idx]['without_id'] += none_ids

        # Log every 30 calls
        if not hasattr(self, '_pose_pid_diagnostic_count'):
            self._pose_pid_diagnostic_count = {}
        if cam_idx not in self._pose_pid_diagnostic_count:
            self._pose_pid_diagnostic_count[cam_idx] = 0
        self._pose_pid_diagnostic_count[cam_idx] += 1

        if self._pose_pid_diagnostic_count[cam_idx] % 30 == 0:
            stats_pid = self._pose_participant_id_diagnostic[cam_idx]
            logger.info(f"[POSE PARTICIPANT_ID] Camera {cam_idx}: "
                      f"total_poses={stats_pid['total']}, "
                      f"with_id={stats_pid['with_id']}, "
                      f"without_id={stats_pid['without_id']}")

        # Initialize diagnostics for this camera
        if cam_idx not in self._pose_lsl_stats:
            self._pose_lsl_stats[cam_idx] = {
                'dropped_no_faces': 0,
                'sent': 0,
                'frame_mismatches': 0,
                'last_pose_frame_id': -1,
                'last_face_frame_id': -1
            }

        # NO FACE DETECTION: Stream poses directly without face synchronization
        # Face detection removed - poses are assigned default participant_id

        # Stream all detected poses (no face detection in this version)
        for pose in pose_data:
            # Assign default participant_id if not set (no face detection)
            pose_participant_id = pose.get('participant_id')
            if pose_participant_id is None:
                pose_participant_id = 1  # Default to first participant

                # Log first occurrence per camera
                if not hasattr(self, '_pose_default_id_logged'):
                    self._pose_default_id_logged = {}
                if cam_idx not in self._pose_default_id_logged:
                    logger.info(f"[POSE LSL] Camera {cam_idx}: Using default participant_id=1 "
                               f"(no face detection in this version)")
                    self._pose_default_id_logged[cam_idx] = True

            # Format participant_id as string (with custom name if available)
            # pose_participant_id is 1-based (1, 2, ...), dict is 0-based
            custom_name = self.participant_names.get(pose_participant_id - 1)
            if custom_name and custom_name.strip() and not custom_name.strip().startswith('P'):
                participant_id = custom_name.strip()
            else:
                participant_id = f"P{pose_participant_id}"

            # Format pose data for LSL
            # Keypoints: 133 keypoints × 4 values (x, y, z, confidence) = 532 floats
            keypoints = pose['keypoints']  # Numpy array (133, 4)

            # Flatten to [x1, y1, z1, v1, x2, y2, z2, v2, ...] using numpy
            pose_flat = keypoints.flatten().tolist()

            # Calculate pose metrics (head orientation, neck, shoulders)
            try:
                # Prepare pose_data dict for metrics calculator
                # metrics_calculator expects: {'keypoints': ndarray(133, 4), ...}
                pose_dict_for_metrics = {
                    'keypoints': keypoints,  # Already (133, 4) ndarray
                    'frame_id': frame_id,
                    'timestamp_ms': int(time.time() * 1000)
                }

                metrics = self.metrics_calculator.calculate_metrics(pose_dict_for_metrics)

                # Store latest metrics for GUI access
                self._latest_pose_metrics[cam_idx] = metrics

                # Send metrics to LSL (if enabled and metrics valid)
                if metrics and metrics.has_valid_metrics():
                    try:
                        # LIFO FRAME-ID TRACKING: Prevent duplicate queueing (same pattern as pose)
                        # Track last queued frame_id per participant to ensure each unique frame sent exactly once
                        if not hasattr(self, '_lsl_last_queued_metrics_frame_id'):
                            self._lsl_last_queued_metrics_frame_id = {}

                        key = f"{cam_idx}_{participant_id}"

                        # Check if this frame was already queued
                        if key in self._lsl_last_queued_metrics_frame_id and frame_id <= self._lsl_last_queued_metrics_frame_id[key]:
                            # Skip queueing duplicate or older frame (LIFO behavior)
                            if self.verbose_debug:
                                logger.debug(f"[METRICS LSL SKIP] Frame {frame_id} already queued for {participant_id} "
                                           f"(last={self._lsl_last_queued_metrics_frame_id[key]})")
                            # Track skipped metrics (initialize stats if needed)
                            if not hasattr(self, '_metrics_lsl_stats'):
                                self._metrics_lsl_stats = {}
                            if cam_idx not in self._metrics_lsl_stats:
                                self._metrics_lsl_stats[cam_idx] = {'sent': 0, 'skipped': 0}
                            self._metrics_lsl_stats[cam_idx]['skipped'] += 1
                        else:
                            # Get camera FPS from config for dynamic stream creation
                            camera_fps = self.config.get('process_separation', {}).get('camera_fps', 30)

                            self.lsl_data_queue.put_nowait({
                                'type': 'metrics_data',
                                'participant_id': participant_id,
                                'camera_index': cam_idx,
                                'frame_id': frame_id,
                                'fps': camera_fps,  # Dynamic FPS for stream creation
                                'metrics': metrics.to_dict()
                            })

                            # Update last queued frame_id (LIFO tracking)
                            self._lsl_last_queued_metrics_frame_id[key] = frame_id

                            # Track successful transmission
                            if not hasattr(self, '_metrics_lsl_stats'):
                                self._metrics_lsl_stats = {}
                            if cam_idx not in self._metrics_lsl_stats:
                                self._metrics_lsl_stats[cam_idx] = {'sent': 0, 'skipped': 0}
                            self._metrics_lsl_stats[cam_idx]['sent'] += 1

                            # Diagnostic: Log frame_id for verification
                            if self.verbose_debug:
                                logger.debug(f"[METRICS LSL QUEUED] Frame {frame_id} queued for {participant_id}")

                    except Exception as e:
                        logger.warning(f"[METRICS LSL] Failed to queue metrics data for {participant_id}: {e}")

            except Exception as e:
                logger.warning(f"[METRICS] Failed to calculate/send metrics for {participant_id}: {e}")

            # Send pose keypoints to LSL queue
            try:
                # LIFO FRAME-ID TRACKING: Prevent duplicate queueing
                # Track last queued frame_id per participant to ensure each unique frame sent exactly once
                if not hasattr(self, '_lsl_last_queued_frame_id'):
                    self._lsl_last_queued_frame_id = {}

                key = f"{cam_idx}_{participant_id}"

                # Check if this frame was already queued
                if key in self._lsl_last_queued_frame_id and frame_id <= self._lsl_last_queued_frame_id[key]:
                    # Skip queueing duplicate or older frame (LIFO behavior)
                    if self.verbose_debug:
                        logger.debug(f"[POSE LSL SKIP] Frame {frame_id} already queued for {participant_id} "
                                   f"(last={self._lsl_last_queued_frame_id[key]})")
                    self._pose_lsl_stats[cam_idx]['skipped'] = self._pose_lsl_stats[cam_idx].get('skipped', 0) + 1
                else:
                    # Get camera FPS from config for dynamic stream creation
                    camera_fps = self.config.get('process_separation', {}).get('camera_fps', 30)

                    self.lsl_data_queue.put_nowait({
                        'type': 'pose_data',
                        'participant_id': participant_id,
                        'camera_index': cam_idx,
                        'frame_id': frame_id,
                        'fps': camera_fps,  # Dynamic FPS for stream creation
                        'pose_data': pose_flat  # 532 floats (133 keypoints × 4 values)
                    })

                    # Update last queued frame_id (LIFO tracking)
                    self._lsl_last_queued_frame_id[key] = frame_id

                    # Track successful transmission
                    self._pose_lsl_stats[cam_idx]['sent'] += 1

                    # Log first successful pose transmission per participant
                    if not hasattr(self, '_pose_lsl_logged'):
                        self._pose_lsl_logged = set()
                    if key not in self._pose_lsl_logged:
                        logger.info(f"✅ [POSE LSL] First pose data sent for camera {cam_idx}, {participant_id}")
                        self._pose_lsl_logged.add(key)

                    # Diagnostic: Log frame_id for verification
                    if self.verbose_debug:
                        logger.debug(f"[POSE LSL QUEUED] Frame {frame_id} queued for {participant_id}")

            except Exception as e:
                logger.warning(f"[POSE LSL] Failed to queue pose data for {participant_id}: {e}")

        # DIAGNOSTIC: Log summary of processed poses (first 5 calls)
        if not hasattr(self, '_pose_summary_count'):
            self._pose_summary_count = {}
        if cam_idx not in self._pose_summary_count:
            self._pose_summary_count[cam_idx] = 0
        self._pose_summary_count[cam_idx] += 1

        if self._pose_summary_count[cam_idx] <= 5:
            sent_this_frame = self._pose_lsl_stats[cam_idx]['sent'] - self._pose_lsl_stats[cam_idx].get('_last_sent', 0)
            self._pose_lsl_stats[cam_idx]['_last_sent'] = self._pose_lsl_stats[cam_idx]['sent']
            logger.info(f"[POSE LSL SUMMARY] Camera {cam_idx} frame {frame_id}: "
                      f"received={len(pose_data)} poses, "
                      f"sent={sent_this_frame}")

        # Periodic diagnostic report (every 30 seconds)
        current_time = time.time()
        if current_time - self._pose_lsl_last_report >= 30.0:
            self._print_pose_lsl_diagnostics()
            self._pose_lsl_last_report = current_time

    def _print_pose_lsl_diagnostics(self) -> None:
        """Print diagnostic summary of pose LSL transmission health."""
        if not self._pose_lsl_stats:
            return

        logger.info("=" * 60)
        logger.info("[POSE LSL DIAGNOSTICS] 30-second summary:")
        for cam_idx, stats in self._pose_lsl_stats.items():
            sent = stats['sent']
            dropped = stats['dropped_no_faces']
            mismatches = stats['frame_mismatches']
            total = sent + dropped

            if total > 0:
                success_rate = (sent / total) * 100 if total > 0 else 0
                logger.info(
                    f"  Camera {cam_idx}: "
                    f"Sent={sent}, Dropped(no faces)={dropped}, "
                    f"Success={success_rate:.1f}%, Frame mismatches={mismatches}, "
                    f"Last pose frame={stats['last_pose_frame_id']}, "
                    f"Last face frame={stats['last_face_frame_id']}"
                )
            else:
                logger.info(f"  Camera {cam_idx}: No pose data processed")

        logger.info("=" * 60)

    def _read_latest_blendshapes_for_correlation(self, cam_idx: int) -> Optional[Dict]:
        """
        Read the most recent blendshapes available for correlation calculation.
        Unlike _read_landmarks(), this IGNORES frame sync to ensure correlation
        continues even with slight processing lag.

        Returns dict with 'blendshapes' and 'n_faces' or None if no data ready.
        """
        if cam_idx not in self.camera_buffers or 'results' not in self.camera_buffers[cam_idx]:
            if self._should_log_for_camera(cam_idx):
                logger.debug(f"[CORRELATION BLENDSHAPES] Camera {cam_idx} buffer not ready")
            return None

        try:
            results_shm = self.camera_buffers[cam_idx]['results']
            # DYNAMIC LAYOUT FIX: Call BufferCoordinator method directly instead of using cached layout
            # This ensures correct layout even when cameras are added dynamically
            results_layout = self.buffer_coordinator.get_results_buffer_layout()

            if not results_layout:
                logger.error(f"[CORRELATION BLENDSHAPES] Failed to get results_buffer_layout from BufferCoordinator")
                return None

            # Read metadata to get latest frame info
            metadata_offset = results_layout.get('metadata_offset', 0)
            metadata_size = results_layout.get('metadata_size', 0)

            if metadata_offset == 0 or metadata_size == 0:
                logger.error(f"[CORRELATION BLENDSHAPES] Invalid metadata layout: offset={metadata_offset}, size={metadata_size}")
                return None

            metadata_dtype = np.dtype([
                ('frame_id', 'int32'),
                ('timestamp_ms', 'int64'),
                ('n_faces', 'int32'),
                ('ready', 'int8'),
                ('is_detection_frame', 'int8'),
                ('track_count', 'int32'),
                ('processing_time_ms', 'float32'),
                ('gpu_memory_used_mb', 'float32'),
            ], align=True)

            metadata_view = np.ndarray(
                1, dtype=metadata_dtype,
                buffer=results_shm.buf[metadata_offset:metadata_offset + metadata_size]
            )[0]

            # Skip if data not ready
            if metadata_view['ready'] != 1:
                return None

            n_faces = int(metadata_view['n_faces'])
            if n_faces == 0:
                return None

            # Read blendshapes (NO frame sync check - use whatever is latest)
            blendshapes_offset = results_layout.get('blendshapes_offset', 0)
            blendshapes_size = results_layout.get('blendshapes_size', 0)

            # Defensive check: ensure blendshapes section exists
            if blendshapes_offset == 0 or blendshapes_size == 0:
                if self._should_log_for_camera(cam_idx, level="warning"):
                    logger.warning(f"[CORRELATION BLENDSHAPES] Blendshapes not available in buffer layout")
                return None

            max_faces = self.buffer_coordinator.max_faces

            blendshapes_view = np.ndarray(
                (max_faces, 52), dtype=np.float32,
                buffer=results_shm.buf[blendshapes_offset:blendshapes_offset + blendshapes_size]
            )

            # Debug log (first read only)
            if self._should_log_for_camera(cam_idx) and not hasattr(self, '_blendshapes_debug_logged'):
                logger.info(f"[CORRELATION BLENDSHAPES] Successfully reading blendshapes: shape={blendshapes_view[:n_faces].shape}")
                self._blendshapes_debug_logged = True

            # Return blendshapes for detected faces only
            return {
                'blendshapes': blendshapes_view[:n_faces].copy(),
                'n_faces': n_faces
            }

        except Exception as e:
            logger.error(f"[CORRELATION BLENDSHAPES] Error reading from camera {cam_idx}: {e}", exc_info=True)
            return None

    def _read_embedding_for_face(self, camera_idx: int, face_index: int, frame_id: int) -> Optional[np.ndarray]:
        """
        Read ArcFace embedding from embedding buffer for a specific face.

        Args:
            camera_idx: Camera index
            face_index: Face index (0-based)
            frame_id: Frame ID to match (for synchronization)

        Returns:
            512-dim embedding array (float32) or None if not available
        """
        if camera_idx not in self.camera_buffers:
            return None

        if 'embedding' not in self.camera_buffers[camera_idx]:
            # Embedding buffer not connected (normal if ArcFace disabled or not ready yet)
            return None

        try:
            emb_shm = self.camera_buffers[camera_idx]['embedding']

            # Read header from EmbeddingBufferLayout
            # Layout: [write_index(8)][frame_id(8)][n_embeddings(4)][padding(4)][embeddings...]
            # Header size: 24 bytes

            # CRITICAL FIX: Track frame_id (not write_index) to allow multi-face reads
            # Problem: Multiple faces per frame need to read from same write_index
            # Original bug: Tracking write_index blocked Face 1 when Face 0 already read
            # Solution: Track frame_id only - allows multiple faces to read same frame,
            #           but prevents reading older frames (stale data)

            # Read frame_id FIRST to check for stale data
            buffer_frame_id_bytes = bytes(emb_shm.buf[8:16])
            buffer_frame_id = struct.unpack('Q', buffer_frame_id_bytes)[0]

            # Read n_embeddings
            n_embeddings_bytes = bytes(emb_shm.buf[16:20])
            n_embeddings = struct.unpack('I', n_embeddings_bytes)[0]

            # Track last frame_id processed per camera (not write_index!)
            if not hasattr(self, '_last_embedding_frame_id'):
                self._last_embedding_frame_id = {}

            last_frame = self._last_embedding_frame_id.get(camera_idx, -1)

            # DIAGNOSTIC: Log buffer state for every face read
            logger.debug(
                f"[EMBEDDING READ] Cam{camera_idx} Face{face_index}: "
                f"buffer_frame={buffer_frame_id}, requested_frame={frame_id}, "
                f"n_embeddings={n_embeddings}, last_frame={last_frame}"
            )

            # Skip ONLY if buffer has OLDER frame than last read (truly stale)
            # Allow re-reading SAME frame (for Face 0 and Face 1) or NEWER frame
            if buffer_frame_id < last_frame:
                # Buffer rewound to older frame - stale data, reject
                logger.debug(
                    f"[EMBEDDING READ] REJECTED (Stale): Cam{camera_idx} Face{face_index}: "
                    f"buffer_frame={buffer_frame_id} < last_frame={last_frame}"
                )
                return None

            # Update tracking to current buffer frame
            # This allows: Face 0 reads frame 920 → Face 1 reads frame 920 (same frame, OK!)
            # But prevents: Reading frame 919 after already reading frame 920 (stale)
            self._last_embedding_frame_id[camera_idx] = buffer_frame_id

            # Frame sync check with tolerance (allow ArcFace to lag slightly behind MediaPipe)
            # PERFORMANCE FIX: ArcFace runs in parallel with MediaPipe and may lag 1-2 frames
            # due to natural processing latency variance. Strict equality (old code) rejected
            # 100% of embeddings, preventing enrollment. New tolerance-based check allows
            # ArcFace to lag by up to 6 frames (~200ms at 30 FPS).
            # NOTE: Batch inference (v2) typically reduces lag to 1-2 frames, but tolerance
            # of 6 provides safety margin for occasional processing spikes.
            FRAME_TOLERANCE = 6  # Allow 6-frame lag (200ms at 30 FPS) - batch reduces to 1-2 frames typical
            frame_lag = frame_id - buffer_frame_id

            # Diagnostic logging (every 30 frames to avoid spam)
            if frame_lag != 0:
                self._embedding_frame_mismatch_count += 1
                if self._embedding_frame_mismatch_count % 30 == 1:
                    logger.debug(
                        f"[EMBEDDING SYNC] Cam{camera_idx} frame lag: {frame_lag} frames "
                        f"(buffer={buffer_frame_id}, requested={frame_id})"
                    )
            else:
                self._embedding_sync_success_count += 1

            # Reject only if stale (>100ms old) or future frame (time travel not possible!)
            if abs(frame_lag) > FRAME_TOLERANCE:
                logger.debug(
                    f"[EMBEDDING READ] REJECTED (Frame Lag): Cam{camera_idx} Face{face_index}: "
                    f"lag={frame_lag} frames (buffer={buffer_frame_id}, requested={frame_id}, "
                    f"tolerance={FRAME_TOLERANCE})"
                )
                return None

            # Validate face index
            if face_index >= n_embeddings:
                # Face index out of range - ArcFace detected fewer faces than MediaPipe
                logger.debug(
                    f"[EMBEDDING READ] REJECTED (Index OOB): Cam{camera_idx} Face{face_index}: "
                    f"face_index={face_index} >= n_embeddings={n_embeddings} "
                    f"(frame={frame_id})"
                )
                return None

            # Calculate embedding offset: header(24) + face_index * embedding_dim(512) * sizeof(float32)(4)
            EMBEDDING_DIM = 512
            embedding_offset = 24 + face_index * EMBEDDING_DIM * 4
            embedding_end = embedding_offset + EMBEDDING_DIM * 4

            # Read embedding as numpy array
            embedding_view = np.ndarray(
                (EMBEDDING_DIM,), dtype=np.float32,
                buffer=emb_shm.buf[embedding_offset:embedding_end]
            )

            # Return copy to avoid shared memory lifetime issues
            embedding = embedding_view.copy()

            # DIAGNOSTIC: Log successful read
            logger.debug(
                f"[EMBEDDING READ] SUCCESS: Cam{camera_idx} Face{face_index}: "
                f"embedding shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}"
            )

            return embedding

        except Exception as e:
            if self._should_log_for_camera(camera_idx, level="warning"):
                logger.warning(f"[EMBEDDING READ] Error reading embedding for camera {camera_idx}, face {face_index}: {e}")
            return None

    def _build_unified_face_list(self, landmarks_data: Optional[Dict],
                                cam_idx: int,
                                frame_width: int,
                                frame_height: int) -> List[Dict]:
        """Build unified face list from landmarks data.

        Args:
            landmarks_data: Landmark data with roi_metadata containing PIXEL bbox coords
            cam_idx: Camera index
            frame_width: Frame width in pixels (for validation/debugging)
            frame_height: Frame height in pixels (for validation/debugging)

        Returns:
            List of face dicts with bbox and centroid in PIXEL coordinates
        """
        faces = []

        # Use landmark data if available
        if landmarks_data and landmarks_data.get('n_faces', 0) > 0:
            landmarks = landmarks_data.get('landmarks', [])
            n_faces = landmarks_data.get('n_faces', 0)
            roi_metadata = landmarks_data.get('roi_metadata', None)
            
            for face_idx in range(n_faces):
                if face_idx < len(landmarks):
                    face_landmarks = landmarks[face_idx]

                    # CRITICAL FIX: Filter out zero landmarks (MediaPipe failed to detect partial faces)
                    # Zero landmarks are written when MediaPipe doesn't detect a face during entry/exit
                    # This prevents overlays from appearing at (0,0) during transitionary frames
                    # Matches the zero-check logic in display_buffer_manager.py:235
                    if np.all(face_landmarks[:, :2] == 0):
                        continue  # Skip this face - no valid landmarks detected

                    # Get metadata (defaults if not available)
                    confidence = 1.0
                    transform = None
                    detection_index = face_idx  # Default: assume no reordering

                    if roi_metadata is not None and face_idx < len(roi_metadata):
                        meta = roi_metadata[face_idx]
                        confidence = float(meta['confidence'])
                        transform = {
                            'x1': float(meta['x1']),
                            'y1': float(meta['y1']),
                            'scale_x': float(meta['scale_x']),
                            'scale_y': float(meta['scale_y'])
                        }
                        # CRITICAL FIX: Extract detection_index to map reordered faces back to original detection order
                        # This enables correct embedding lookup (ArcFace processes faces in original order)
                        detection_index = meta.get('detection_index', face_idx)

                    # Create face dict (participant_id will be assigned by GridParticipantManager)
                    face = {
                        'landmarks': face_landmarks,
                        'confidence': confidence,
                        'transform': transform,
                        'centroid': (0.5, 0.5),
                        'source': 'landmarks',
                        'detection_index': detection_index  # Preserve original detection order for embedding lookup
                    }
                    
                    # Generate bbox from transform
                    # CRITICAL: scale_x and scale_y are actually x2/y2 coordinates (not scales!)
                    # MediaPipe writes bbox coordinates directly to roi_metadata in PIXEL coordinates
                    # NO CONVERSION NEEDED - they are already in pixel space
                    if transform and all(k in transform for k in ['x1', 'y1', 'scale_x', 'scale_y']):
                        # Extract PIXEL coordinates (already in pixel space from MediaPipe)
                        x1_px = transform['x1']
                        y1_px = transform['y1']
                        x2_px = transform['scale_x']  # Actually x2 coordinate (not scale)
                        y2_px = transform['scale_y']  # Actually y2 coordinate (not scale)

                        # Set bbox (already in PIXEL coordinates, no scaling needed)
                        face['bbox'] = [x1_px, y1_px, x2_px, y2_px]
                        # Set centroid in PIXEL coordinates
                        face['centroid'] = ((x1_px + x2_px) / 2.0, (y1_px + y2_px) / 2.0)

                    faces.append(face)

        return faces
        
    def _assign_participants(self, faces: List[Dict], cam_idx: int, frame_id: int) -> List[int]:
        """
        Assign participant ID to detected faces (single participant mode).

        In single participant mode, we assign the first/best face to participant_id=1.
        All other faces are ignored.

        Args:
            faces: List of face dicts with 'bbox', 'centroid', optional 'embedding'
            cam_idx: Camera index
            frame_id: Frame ID for embedding synchronization

        Returns:
            List of participant IDs (integers) - always [1] for first face, empty if no faces
        """
        if not faces:
            # No faces detected - mark participant as absent
            if hasattr(self, 'participant_presence_array') and len(self.participant_presence_array) > 1:
                self.participant_presence_array[1] = 0  # Participant 1 absent
            return []

        # SINGLE PARTICIPANT MODE: Use only the first/best face
        # In the future, could select "best" based on size, confidence, etc.
        best_face = faces[0]
        detection_index = best_face.get('detection_index', 0)

        # Read embedding for the selected face (needed for enrollment)
        embedding = self._read_embedding_for_face(
            camera_idx=cam_idx,
            face_index=detection_index,
            frame_id=frame_id
        )

        # Add embedding to face dict for enrollment processing
        best_face['embedding'] = embedding

        # Assign to participant 1
        participant_id = 1
        best_face['participant_id'] = participant_id
        best_face['is_new_participant'] = False  # No concept of "new" in single participant mode

        # Update last seen time
        self.participant_last_seen[participant_id] = time.time()

        # Mark participant 1 as present in presence array
        if hasattr(self, 'participant_presence_array') and len(self.participant_presence_array) > 1:
            if self.participant_presence_array[1] != 1:
                self.participant_presence_array[1] = 1  # Participant 1 present
                logger.debug(f"[PRESENCE] Cam{cam_idx}: Participant 1 present")

        # Return list with only first face assigned (ignore remaining faces)
        return [participant_id]
        
    def _process_face_recognition(self, faces: List[Dict], cam_idx: int) -> Dict:
        """
        Process face recognition and enrollment for detected faces.

        Integrates grid-based participant assignments with enrollment system.
        Uses embeddings from ArcFace process for enrollment validation.

        Args:
            faces: List of face dicts with participant_id, bbox, embedding (optional)
            cam_idx: Camera index

        Returns:
            Dict mapping face index to enrollment/recognition status
        """
        recognition_updates = {}

        # ARCHITECTURAL CONSOLIDATION: Enrollment manager removed from worker
        # Worker sends samples to GUI for enrollment processing
        # (No check needed - we always send samples if participant_update_queue exists)

        # Initialize enrollment diagnostic counter
        if not hasattr(self, '_enrollment_diagnostic_counter'):
            self._enrollment_diagnostic_counter = 0
            self._enrollment_skipped_no_pid = 0
            self._enrollment_skipped_no_embedding = 0
            self._enrollment_processed = 0

        self._enrollment_diagnostic_counter += 1

        for face_idx, face in enumerate(faces):
            participant_id = face.get('participant_id')
            confidence = face.get('confidence', 0.0)

            # DIAGNOSTIC: Log when participant assignment missing
            if participant_id is None or participant_id < 0:
                self._enrollment_skipped_no_pid += 1
                # Log first 10 occurrences to avoid spam
                if self._enrollment_skipped_no_pid <= 10:
                    logger.warning(f"[ENROLLMENT DIAG] Skipped face {face_idx}: no participant_id "
                                  f"(confidence={confidence:.3f})")
                continue

            # Get embedding for this face (may be None if ArcFace is lagging)
            embedding = face.get('embedding')

            # DIAGNOSTIC: Log when embedding missing (CRITICAL for enrollment)
            if embedding is None:
                self._enrollment_skipped_no_embedding += 1
                # Log periodically to show embedding lag issue
                if self._enrollment_skipped_no_embedding % 30 == 1:  # First and every 30th
                    logger.warning(f"[ENROLLMENT DIAG] Skipped P{participant_id} face {face_idx}: "
                                  f"embedding NOT available (ArcFace lagging?). "
                                  f"Total skipped: {self._enrollment_skipped_no_embedding}")
                    logger.warning(f"  → Confidence={confidence:.3f}, can't start enrollment without embedding")
                continue  # Wait for embedding to be available

            # Calculate quality score from detection confidence
            # (In real system, this could be based on pose, blur, lighting, etc.)
            confidence = face.get('confidence', 1.0)
            quality_score = confidence

            # ARCHITECTURAL CONSOLIDATION: Send enrollment sample to GUI
            # GUI's GlobalParticipantManager.enrollment_manager will process it
            if self.participant_update_queue:
                try:
                    enrollment_sample = {
                        'type': 'enrollment_sample',  # NEW event type
                        'participant_id': participant_id,
                        'track_id': participant_id,
                        'embedding': embedding.tolist(),  # Convert numpy array to list for serialization
                        'quality_score': quality_score,
                        'camera_idx': cam_idx,
                        'timestamp': time.time(),
                        'frame_id': face.get('frame_id', -1)
                    }
                    self.participant_update_queue.put_nowait(enrollment_sample)

                    self._enrollment_processed += 1
                    if self._enrollment_processed <= 10:
                        logger.info(f"[WORKER→GUI] Sent enrollment sample for P{participant_id} (quality={quality_score:.3f})")

                except queue.Full:
                    logger.warning(f"[WORKER→GUI] participant_update_queue full, dropped enrollment sample for P{participant_id}")

            # ARRAY-BASED IPC: Read enrollment state from GUI's shared array for real-time overlay colors
            # Replaces hardcoded 'UNKNOWN' with actual state (COLLECTING, VALIDATING, ENROLLED, etc.)
            if 1 <= participant_id < len(self.enrollment_state_array):
                state_code = self.enrollment_state_array[participant_id]
                enrollment_state = self.ENROLLMENT_STATES_REVERSE.get(state_code, 'unknown')
            else:
                enrollment_state = 'unknown'

            # DIAGNOSTIC: Log enrollment state changes (first 10 occurrences per participant)
            if not hasattr(self, '_enrollment_state_log_count'):
                self._enrollment_state_log_count = {}
            if participant_id not in self._enrollment_state_log_count:
                self._enrollment_state_log_count[participant_id] = 0

            if self._enrollment_state_log_count[participant_id] < 10 or enrollment_state != 'UNKNOWN':
                logger.debug(f"[WORKER←GUI] Read enrollment state for P{participant_id}: {enrollment_state}")
                self._enrollment_state_log_count[participant_id] += 1

            recognition_updates[face_idx] = {
                'participant_id': participant_id,
                'similarity': 0.0,  # Will be calculated by GUI's enrollment manager
                'quality_score': quality_score,
                'enrollment_state': enrollment_state  # Read from GUI's shared dict (real-time state)
            }

        # DIAGNOSTIC: Periodic summary (every 60 frames = 2 seconds at 30 FPS)
        if self._enrollment_diagnostic_counter % 60 == 0:
            logger.info(f"[WORKER→GUI ENROLLMENT SUMMARY] Frame {self._enrollment_diagnostic_counter}:")
            logger.info(f"  Skipped (no participant_id): {self._enrollment_skipped_no_pid}")
            logger.info(f"  Skipped (no embedding): {self._enrollment_skipped_no_embedding}")
            logger.info(f"  Sent to GUI for enrollment: {self._enrollment_processed}")
            logger.info(f"  (Enrollment state tracked by GUI's GlobalParticipantManager)")

        return recognition_updates
        
    def _format_face_label(self, face: Dict, participant_id: Optional[int],
                          recognition_updates: Dict) -> str:
        """Format display label for face with enrollment status."""

        # Handle rejected participants (None due to cap reached)
        if participant_id is None:
            return "Cap Reached"

        if participant_id and participant_id <= self.max_participants:
            label = self.participant_names.get(participant_id - 1, f"P{participant_id}")

            # Add enrollment/recognition status from recognition_updates (keyed by face_idx)
            # recognition_updates is now keyed by face_idx (not track_id)
            face_idx = face.get('face_idx', -1)
            if face_idx >= 0 and face_idx in recognition_updates:
                rec_info = recognition_updates[face_idx]
                enrollment_state = rec_info.get('enrollment_state', 'UNKNOWN')

                # Add enrollment state indicator (Unicode symbols, NOT emoji)
                # Uses DejaVu Sans / Segoe UI Symbol compatible characters
                if enrollment_state == 'ENROLLED':
                    label += " ●"  # U+25CF Black circle - Fully enrolled
                elif enrollment_state == 'COLLECTING':
                    label += " ◴"  # U+25F4 Circle with upper right quadrant - Collecting samples
                elif enrollment_state == 'VALIDATING':
                    label += " ◵"  # U+25F5 Circle with upper left quadrant - Validating
                elif enrollment_state == 'IMPROVING':
                    label += " ○"  # U+25CB White circle - Continuous improvement
                elif enrollment_state == 'FAILED':
                    label += " ✗"  # U+2717 Ballot X - Enrollment failed
                elif enrollment_state == 'UNKNOWN':
                    label += " ?"  # U+003F Question mark - New participant awaiting enrollment

                # Add similarity indicator if enrolled
                if enrollment_state in ['ENROLLED', 'IMPROVING']:
                    similarity = rec_info.get('similarity', 0)
                    if similarity > 0.8:
                        label += f" ({similarity:.2f})"

            return label
        else:
            return "Unknown"
            
    def _extract_shape_from_landmarks(self, landmarks: np.ndarray) -> Optional[np.ndarray]:
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
            
            # Use stable landmarks for participant recognition
            stable_indices = [33, 133, 362, 263, 168, 6, 10, 234, 454, 152]  # Key facial points
            
            if landmarks.shape[0] > max(stable_indices) and landmarks.shape[1] >= 2:
                # Extract stable landmarks for quality estimation
                stable_landmarks = landmarks[stable_indices, :2]  # Take only x,y coordinates
                return stable_landmarks  # Return as 2D array (10, 2) not flattened
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error extracting shape from landmarks: {e}")
            return None
        
    def _write_display_data(self, cam_idx: int, frame_id: int, timestamp: float,
                           display_faces: List[DisplayFaceData], frame_bgr: np.ndarray,
                           pose_data: Optional[List[Dict]] = None,
                           pose_metrics: Optional['PoseMetrics'] = None):
        """Write processed display-ready data to display buffer (OUTPUT side of data bridge)."""

        # Store latest pose metrics for GUI access (per-camera)
        if not hasattr(self, '_latest_pose_metrics'):
            self._latest_pose_metrics = {}
        if pose_metrics is not None:
            self._latest_pose_metrics[cam_idx] = pose_metrics

        # FIX: Write to display buffer, not GUI buffer!
        # The GUI reads from display buffers created by this worker
        if cam_idx not in self.display_buffers:
            logger.error(f"[DATA BRIDGE] No display buffer for camera {cam_idx}")
            return False

        try:
            display_buffer = self.display_buffers[cam_idx]

            # FIX: Store and send pose data ALWAYS (even when empty) to clear stale cache
            # Store pose data for GUI access (DisplayBuffer doesn't support pose in write_display_data yet)
            if not hasattr(self, 'latest_pose_data'):
                self.latest_pose_data = {}
            self.latest_pose_data[cam_idx] = {
                'frame_id': frame_id,
                'poses': pose_data if pose_data else []  # Store empty list if None
            }

            # Send pose data to GUI via queue (ALWAYS, even when empty)
            if hasattr(self, 'pose_data_queue') and self.pose_data_queue:
                try:
                    # DEBUG: Track when pose data is sent to GUI
                    if not hasattr(self, '_last_pose_queue_frame_id'):
                        self._last_pose_queue_frame_id = {}
                    last_sent = self._last_pose_queue_frame_id.get(cam_idx, -1)
                    n_poses = len(pose_data) if pose_data else 0

                    # Get performance metrics for this camera
                    perf_metrics = self._latest_performance_metrics.get(cam_idx)

                    self.pose_data_queue.put({
                        'camera_idx': cam_idx,
                        'frame_id': frame_id,
                        'poses': pose_data if pose_data else [],  # Send empty list to clear cache
                        'metrics': pose_metrics.to_dict() if pose_metrics else None,  # Add metrics for GUI metrics panel
                        'performance': perf_metrics  # Add performance metrics (FPS, latency)
                    }, block=False)

                    # Track last sent frame_id (no verbose logging)
                    if frame_id != last_sent:
                        self._last_pose_queue_frame_id[cam_idx] = frame_id
                except Exception as queue_error:
                    logger.warning(f"[POSE] Queue write failed for cam={cam_idx}: {queue_error}")

            # CLEAN ARCHITECTURE: No skeleton overlays on camera feed
            # Display shows clean native resolution frames from camera frame_buffer
            # Pose data is processed separately for metrics/analysis

            # Write to display buffer using its write method
            # The DisplayBuffer class handles the memory layout
            success = display_buffer.write_display_data(
                frame_id=frame_id,
                timestamp=timestamp,
                faces=display_faces,
                frame_bgr=frame_bgr
            )

            # FIX: Add logging to verify display buffer writes
            if success:
                # Track successful writes for debugging
                if not hasattr(self, '_display_write_count'):
                    self._display_write_count = {}
                self._display_write_count[cam_idx] = self._display_write_count.get(cam_idx, 0) + 1

                # Log first few writes and then periodically
                if self._display_write_count[cam_idx] <= 5 or self._display_write_count[cam_idx] % 100 == 0:
                    logger.info(f"[DISPLAY WRITE SUCCESS] Camera {cam_idx}: "
                              f"Wrote frame_id={frame_id} with {len(display_faces)} faces "
                              f"(write #{self._display_write_count[cam_idx]})")
            else:
                logger.warning(f"[DISPLAY WRITE FAILED] Camera {cam_idx}: Failed to write frame_id={frame_id}")

            return success
            
        except Exception as e:
            logger.error(f"[DATA BRIDGE ERROR] Failed writing to display buffer for camera {cam_idx}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _serialize_display_faces(self, faces: List[DisplayFaceData]) -> bytes:
        """Serialize face data optimized for GUI rendering."""
        if not faces:
            return b''
        
        import struct
        serialized_faces = []
        
        for face in faces:
            try:
                # DisplayFaceData is a dataclass - access attributes directly
                track_id = face.track_id
                participant_id = face.participant_id
                bbox = face.bbox
                label = face.label
                    
                label_bytes = label.encode('utf-8')[:63]  # Limit label length
                
                # Pack basic face data (28 bytes + label)
                face_data = struct.pack('IIffffI',
                                       int(track_id),
                                       int(participant_id),
                                       float(bbox[0]),
                                       float(bbox[1]),
                                       float(bbox[2]),
                                       float(bbox[3]),
                                       len(label_bytes))
                face_data += label_bytes
                
                # Add landmark data if available (already in display coordinates)
                landmarks = face.landmarks
                    
                if landmarks is not None:
                    # Convert to float32 array and serialize
                    if not isinstance(landmarks, np.ndarray):
                        landmarks = np.array(landmarks, dtype=np.float32)
                    else:
                        landmarks = landmarks.astype(np.float32)
                    
                    # Flatten to 1D and add to face data
                    landmarks_flat = landmarks.flatten()
                    landmarks_bytes = landmarks_flat.tobytes()
                    face_data += landmarks_bytes
                
                serialized_faces.append(face_data)
                
            except Exception as e:
                logger.error(f"[DATA BRIDGE] Error serializing face: {e}")
                continue
        
        return b''.join(serialized_faces)
            
    def _process_control_commands(self):
        """Process control commands from GUI."""
        try:
            while not self.control_queue.empty():
                command = self.control_queue.get_nowait()
                
                if command['type'] == 'shutdown':
                    self.running = False
                elif command['type'] == 'update_names':
                    self.participant_names = command['names']
                elif command['type'] == 'buffer_names':
                    # Handle dynamic buffer name updates (e.g., when cameras are added)
                    new_buffer_names = command.get('actual_buffer_names', {})

                    # CRITICAL FIX: Detect removed cameras (cameras in camera_indices but not in new_buffer_names)
                    removed_cameras = set(self.camera_indices) - set(new_buffer_names.keys())
                    for cam_idx in removed_cameras:
                        logger.info(f"[WORKER] Camera {cam_idx} removed - cleaning up resources")

                        # 1. Remove from processing loop
                        if cam_idx in self.camera_indices:
                            self.camera_indices.remove(cam_idx)
                            logger.info(f"[WORKER] Removed camera {cam_idx} from camera_indices: {self.camera_indices}")

                        # 2. Close and delete display buffer
                        if cam_idx in self.display_buffers:
                            try:
                                display_buf = self.display_buffers[cam_idx]
                                if hasattr(display_buf, 'shm') and display_buf.shm:
                                    display_buf.shm.close()
                                del self.display_buffers[cam_idx]
                                logger.info(f"[WORKER] Closed display buffer for camera {cam_idx}")
                            except Exception as e:
                                logger.error(f"[WORKER] Error closing display buffer for camera {cam_idx}: {e}")

                        # 3. Close and delete camera buffers (input connections)
                        if cam_idx in self.camera_buffers:
                            try:
                                for buf_type, shm in self.camera_buffers[cam_idx].items():
                                    if hasattr(shm, 'close'):
                                        shm.close()
                                del self.camera_buffers[cam_idx]
                                logger.info(f"[WORKER] Closed camera buffers for camera {cam_idx}")
                            except Exception as e:
                                logger.error(f"[WORKER] Error closing camera buffers for camera {cam_idx}: {e}")

                        # 4. Clean up tracking state
                        if cam_idx in self.last_frame_ids:
                            del self.last_frame_ids[cam_idx]

                        # 5. Remove from actual_buffer_names
                        if cam_idx in self.actual_buffer_names:
                            del self.actual_buffer_names[cam_idx]

                        # 6. Clean up pose-face matching buffer
                        if cam_idx in self._display_faces_buffer:
                            self._display_faces_buffer[cam_idx].clear()
                            del self._display_faces_buffer[cam_idx]
                            logger.info(f"[WORKER] Cleared _display_faces_buffer for camera {cam_idx}")

                        logger.info(f"[WORKER] Camera {cam_idx} cleanup complete")

                    # Update buffer names (add new cameras)
                    self.actual_buffer_names.update(new_buffer_names)

                    # Also update camera resolutions
                    new_camera_resolutions = command.get('camera_resolutions', {})
                    self.camera_resolutions.update(new_camera_resolutions)

                    logger.info(f"[WORKER] Updated buffer names for {len(new_buffer_names)} cameras")
                    logger.info(f"[WORKER] Updated camera resolutions: {self.camera_resolutions}")

                    # Reconnect buffers if needed
                    for cam_idx in new_buffer_names:
                        if cam_idx not in self.camera_buffers:
                            # New camera added dynamically
                            logger.info(f"[DYNAMIC CAMERA] Adding camera {cam_idx} to worker")

                            # 1. Add to camera_indices for processing loop
                            if cam_idx not in self.camera_indices:
                                self.camera_indices.append(cam_idx)
                                logger.info(f"[DYNAMIC CAMERA] Added camera {cam_idx} to camera_indices: {self.camera_indices}")

                            # 2. Create display buffer
                            buffer_name = f"yq_display_{cam_idx}_{mp.current_process().pid}"
                            try:
                                display_buffer = DisplayBuffer(cam_idx, buffer_name, self.buffer_coordinator, create=True)
                                self.display_buffers[cam_idx] = display_buffer
                                logger.info(f"[DYNAMIC CAMERA] Created display buffer for camera {cam_idx}: {buffer_name}")

                                # 5. Notify GUI about new display buffer
                                try:
                                    # Use non-blocking put to prevent queue overflow crashes
                                    self.status_queue.put({
                                        'type': 'display_buffer_added',
                                        'camera_index': cam_idx,
                                        'display_buffer_name': buffer_name
                                    }, block=False)
                                    logger.info(f"[DYNAMIC CAMERA] Notified GUI about display buffer for camera {cam_idx}")
                                except queue.Full:
                                    logger.error(f"[DYNAMIC CAMERA] Status queue full! Cannot notify GUI about camera {cam_idx}")
                                except Exception as e:
                                    logger.error(f"[DYNAMIC CAMERA] Failed to notify GUI: {e}")
                            except Exception as e:
                                logger.error(f"[DYNAMIC CAMERA] Failed to create display buffer for camera {cam_idx}: {e}")

                            # 3. Initialize tracking state
                            self.last_frame_ids[cam_idx] = -1
                            logger.info(f"[DYNAMIC CAMERA] Initialized tracking state for camera {cam_idx}")

                            # 4. Connect input buffers
                            self._connect_camera_buffers_for_index(cam_idx)

                            # 5. Initialize pose-face matching buffer for new camera
                            from collections import deque
                            if cam_idx not in self._display_faces_buffer:
                                self._display_faces_buffer[cam_idx] = deque(maxlen=30)

                elif command['type'] == 'selected_camera_changed':
                    # Handle camera selection broadcast from GUI
                    new_cam_idx = command['camera_index']
                    prev_cam_idx = command.get('previous_camera_index')

                    logger.info(f"[WORKER] 📡 Received camera selection change: {prev_cam_idx} → {new_cam_idx}")

                    # Update camera_indices to only process selected camera
                    old_cameras = self.camera_indices.copy() if hasattr(self, 'camera_indices') else []
                    self.camera_indices = [new_cam_idx]

                    logger.info(f"[WORKER] Updated camera_indices: {old_cameras} → [{new_cam_idx}]")

                    # Clear cached data for non-selected cameras
                    cameras_cleared = []

                    # Clear pose data cache (if exists)
                    if hasattr(self, '_cached_pose_data'):
                        for cam in list(self._cached_pose_data.keys()):
                            if cam != new_cam_idx:
                                del self._cached_pose_data[cam]
                                cameras_cleared.append(cam)

                    # Clear facial landmarks cache (if exists)
                    if hasattr(self, 'facial_landmarks_cache'):
                        for cam in list(self.facial_landmarks_cache.keys()):
                            if cam != new_cam_idx:
                                del self.facial_landmarks_cache[cam]

                    # Clear face centroids cache (if exists)
                    if hasattr(self, 'face_centroids_cache'):
                        for cam in list(self.face_centroids_cache.keys()):
                            if cam != new_cam_idx:
                                del self.face_centroids_cache[cam]

                    # Clear display faces buffer for non-selected cameras
                    if hasattr(self, '_display_faces_buffer'):
                        for cam in list(self._display_faces_buffer.keys()):
                            if cam != new_cam_idx:
                                self._display_faces_buffer[cam].clear()
                                del self._display_faces_buffer[cam]

                    if cameras_cleared:
                        logger.info(f"[WORKER] 🧹 Cleared cached data for cameras: {cameras_cleared}")

                    logger.info(f"[WORKER] ✅ Now processing only camera {new_cam_idx}")

                elif command['type'] == 'update_config':
                    # Handle config updates
                    pass
                elif command['type'] == 'start_lsl_streaming':
                    # Enable LSL streaming (queue already set during initialization)
                    # Note: lsl_data_queue is now passed during process creation, not via control message
                    logger.info("[WORKER] ⚡ Received 'start_lsl_streaming' command")
                    if self.lsl_data_queue is None:
                        logger.error("[WORKER] ❌ Cannot enable LSL streaming - lsl_data_queue is None!")
                    else:
                        self.lsl_streaming_active = True
                        logger.info(f"[WORKER] ✅ LSL streaming ENABLED! lsl_streaming_active={self.lsl_streaming_active}, queue={self.lsl_data_queue}")
                elif command['type'] == 'stop_lsl_streaming':
                    # Disable LSL streaming (keep queue reference, just stop using it)
                    self.lsl_streaming_active = False
                    # Note: Don't set lsl_data_queue to None - it's a shared resource
                    logger.info("[WORKER] LSL streaming disabled")
                elif command['type'] == 'set_participant_names':
                    # Update participant names for LSL stream naming
                    # These names are locked at stream creation time
                    names_dict = command.get('names', {})
                    if names_dict:
                        self.participant_names.update(names_dict)
                        logger.info(f"[WORKER] Updated participant names: {self.participant_names}")
                    else:
                        logger.warning("[WORKER] Received set_participant_names with empty names dict")
                elif command['type'] == 'set_mesh':
                    # Handle mesh data sending toggle
                    old_value = self.mesh_enabled
                    new_value = command.get('enabled', False)
                    self.mesh_enabled = new_value

                    # ENHANCED DIAGNOSTIC: Log state change with verification
                    logger.info(f"[WORKER] Mesh toggle command received: {old_value} → {new_value}")
                    logger.info(f"[WORKER] Mesh data sending {'enabled' if self.mesh_enabled else 'disabled'}")

                    # Verify state was actually changed
                    if self.mesh_enabled != new_value:
                        logger.error(f"[WORKER] CRITICAL: mesh_enabled state mismatch! "
                                   f"Expected {new_value}, got {self.mesh_enabled}")
                    else:
                        logger.info(f"[WORKER] Mesh state verified: mesh_enabled={self.mesh_enabled}")

                    # Reset diagnostic counters to log next mesh additions
                    if hasattr(self, '_mesh_add_count'):
                        delattr(self, '_mesh_add_count')
                    if hasattr(self, '_mesh_skip_count'):
                        delattr(self, '_mesh_skip_count')
                    if hasattr(self, '_lsl_mesh_diagnostic_logged'):
                        delattr(self, '_lsl_mesh_diagnostic_logged')
                elif command['type'] == 'connect_correlation_buffer':
                    # Connect to correlation buffer for bar graph display
                    buffer_name = command.get('buffer_name')
                    try:
                        import multiprocessing.shared_memory as mp_shm
                        # CRITICAL: Store SharedMemory reference to prevent garbage collection
                        self.correlation_shm = mp_shm.SharedMemory(name=buffer_name)
                        self.correlation_buffer = np.ndarray((52,), dtype=np.float32, buffer=self.correlation_shm.buf)
                        logger.info(f"[WORKER] Connected to correlation buffer: {buffer_name}")
                    except Exception as e:
                        logger.error(f"[WORKER] Failed to connect to correlation buffer: {e}")
                        import traceback
                        traceback.print_exc()
                elif command['type'] == 'connect_camera':
                    # DYNAMIC DISPLAY CONNECTION: Connect to camera display buffer when camera becomes ready
                    # This solves the black screen issue where GUI worker starts before camera 0 exists
                    cam_idx = command.get('camera_index')
                    buffer_names = command.get('buffer_names', {})
                    resolution = command.get('resolution')

                    logger.info(f"[WORKER] 📡 Received connect_camera command for camera {cam_idx}")
                    logger.info(f"[WORKER]   Buffer names: {list(buffer_names.keys())}")
                    logger.info(f"[WORKER]   Resolution: {resolution}")

                    try:
                        # 1. Add to actual_buffer_names for buffer connection
                        if cam_idx not in self.actual_buffer_names:
                            self.actual_buffer_names[cam_idx] = buffer_names
                            logger.info(f"[WORKER] Added buffer names for camera {cam_idx}")

                        # 2. Update camera resolution
                        if resolution and cam_idx not in self.camera_resolutions:
                            self.camera_resolutions[cam_idx] = resolution
                            logger.info(f"[WORKER] Set resolution for camera {cam_idx}: {resolution}")

                        # 3. Add to camera_indices for processing loop (if not already present)
                        if cam_idx not in self.camera_indices:
                            self.camera_indices.append(cam_idx)
                            logger.info(f"[WORKER] Added camera {cam_idx} to camera_indices: {self.camera_indices}")

                        # 4. Create display buffer for this camera (output buffer for GUI rendering)
                        if cam_idx not in self.display_buffers:
                            buffer_name = f"yq_display_{cam_idx}_{mp.current_process().pid}"
                            try:
                                display_buffer = DisplayBuffer(cam_idx, buffer_name, self.buffer_coordinator, create=True)
                                self.display_buffers[cam_idx] = display_buffer
                                logger.info(f"[WORKER] ✅ Created display buffer for camera {cam_idx}: {buffer_name}")

                                # Notify GUI about new display buffer
                                logger.info(f"[WORKER DIAGNOSTIC] 🔔 ATTEMPTING to send camera_display_connected for camera {cam_idx}")
                                logger.info(f"[WORKER DIAGNOSTIC]   Buffer name: {buffer_name}")
                                logger.info(f"[WORKER DIAGNOSTIC]   Status queue size: {self.status_queue.qsize()}")
                                try:
                                    self.status_queue.put({
                                        'type': 'camera_display_connected',
                                        'camera_index': cam_idx,
                                        'display_buffer_name': buffer_name
                                    }, block=False)
                                    logger.info(f"[WORKER DIAGNOSTIC] ✅ SUCCESS: Sent camera_display_connected for camera {cam_idx}")
                                    logger.info(f"[WORKER] 📤 Notified GUI: display buffer ready for camera {cam_idx}")
                                except queue.Full:
                                    logger.error(f"[WORKER DIAGNOSTIC] ❌ QUEUE FULL! Cannot notify GUI about camera {cam_idx}")
                                    logger.error(f"[WORKER] Status queue full! Cannot notify GUI about camera {cam_idx}")
                                except Exception as e:
                                    logger.error(f"[WORKER DIAGNOSTIC] ❌ EXCEPTION: Failed to notify GUI: {e}")
                                    logger.error(f"[WORKER] Failed to notify GUI: {e}")
                            except Exception as e:
                                logger.error(f"[WORKER] Failed to create display buffer for camera {cam_idx}: {e}")
                                import traceback
                                traceback.print_exc()

                        # 5. Initialize tracking state (frame ID tracking)
                        if cam_idx not in self.last_frame_ids:
                            self.last_frame_ids[cam_idx] = -1
                            logger.info(f"[WORKER] Initialized tracking state for camera {cam_idx}")

                        # 6. Connect input buffers (frame, results, pose, detection, embedding)
                        logger.info(f"[WORKER] Connecting input buffers for camera {cam_idx}...")
                        self._connect_camera_buffers_for_index(cam_idx)

                        # 7. Initialize pose-face matching buffer for new camera
                        from collections import deque
                        if cam_idx not in self._display_faces_buffer:
                            self._display_faces_buffer[cam_idx] = deque(maxlen=30)
                            logger.info(f"[WORKER] Initialized pose-face buffer for camera {cam_idx}")

                        logger.info(f"[WORKER] ✅✅✅ Camera {cam_idx} fully connected and ready for display!")

                    except Exception as e:
                        logger.error(f"[WORKER] Failed to connect camera {cam_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                # flush_camera command removed - LIFO scan handles frame skipping automatically

        except Exception as e:
            logger.error(f"[WORKER] Error processing control command: {e}")
            import traceback
            traceback.print_exc()
            
    def _update_performance_stats(self):
        """Update and report performance statistics, calculate FPS and latency metrics."""
        current_time = time.time()
        if current_time - self.last_stats_print >= 1.0:  # Every second
            # Legacy processing times tracking (GUI worker overall processing)
            avg_time = 0.0
            frame_count = 0

            if self.processing_times:
                avg_time = np.mean(self.processing_times)
                max_time = np.max(self.processing_times)
                frame_count = len(self.processing_times)

                # Use non-blocking put to prevent queue overflow crashes
                try:
                    self.status_queue.put({
                        'type': 'performance',
                        'avg_processing_ms': avg_time,
                        'max_processing_ms': max_time,
                        'fps_capability': 1000.0 / avg_time if avg_time > 0 else 0
                    }, block=False)
                except queue.Full:
                    # Don't log every time - performance stats are not critical
                    pass
                except Exception as e:
                    logger.warning(f"[WORKER] Failed to send performance stats: {e}")

                logger.info(f"Processing performance: avg={avg_time:.1f}ms, max={max_time:.1f}ms")

                # Reset stats
                self.processing_times = []

            # Calculate per-camera FPS and latency metrics from timing windows
            for cam_idx in self.camera_indices:
                metrics = {}

                # Calculate detection FPS from per-camera detection timing window
                # This represents the final realized throughput of the detection pipeline for this camera
                if cam_idx in self._detection_timing_window and len(self._detection_timing_window[cam_idx]) > 0:
                    detection_window = list(self._detection_timing_window[cam_idx])

                    # Filter to last 1 second of data
                    recent_detections = [
                        (ts, proc_time) for ts, proc_time in detection_window
                        if current_time - ts <= 1.0
                    ]

                    if len(recent_detections) > 0:
                        # Realized detection FPS = count of frames processed in last second
                        detection_fps = len(recent_detections) / 1.0
                        metrics['detection_fps'] = detection_fps
                    else:
                        metrics['detection_fps'] = 0.0
                else:
                    metrics['detection_fps'] = 0.0

                # Calculate pose estimation FPS from pose timing window
                if cam_idx in self._pose_timing_window and len(self._pose_timing_window[cam_idx]) > 0:
                    pose_window = list(self._pose_timing_window[cam_idx])

                    # Filter to last 1 second of data
                    recent_poses = [
                        (ts, det_time, pose_time) for ts, det_time, pose_time in pose_window
                        if current_time - ts <= 1.0
                    ]

                    if len(recent_poses) > 1:
                        # Realized pose FPS = count of poses in last second
                        pose_fps = len(recent_poses) / 1.0
                        metrics['pose_fps'] = pose_fps

                        # Average timing breakdown
                        avg_det_time = np.mean([det for _, det, _ in recent_poses])
                        avg_pose_time = np.mean([pose for _, _, pose in recent_poses])
                        metrics['avg_detection_ms'] = avg_det_time
                        metrics['avg_pose_ms'] = avg_pose_time
                    else:
                        metrics['pose_fps'] = 0.0
                        metrics['avg_detection_ms'] = 0.0
                        metrics['avg_pose_ms'] = 0.0
                else:
                    metrics['pose_fps'] = 0.0
                    metrics['avg_detection_ms'] = 0.0
                    metrics['avg_pose_ms'] = 0.0

                # Calculate end-to-end latency
                # Latency = time from frame capture to display (estimated from processing times)
                if cam_idx in self._latency_window and len(self._latency_window[cam_idx]) > 0:
                    avg_latency = np.mean(list(self._latency_window[cam_idx]))
                    metrics['latency_ms'] = avg_latency
                else:
                    # Estimate latency from total processing time if no explicit latency tracking
                    if avg_time > 0:
                        metrics['latency_ms'] = avg_time
                    else:
                        metrics['latency_ms'] = 0.0

                # Update cache for GUI access
                if cam_idx not in self._latest_performance_metrics:
                    self._latest_performance_metrics[cam_idx] = {}
                self._latest_performance_metrics[cam_idx].update(metrics)

            self.last_stats_print = current_time
                
    def _cleanup(self):
        """Clean up resources."""
        # Clean up display buffers
        for buffer in self.display_buffers.values():
            buffer.cleanup()

        # Clean up correlation shared memory
        if hasattr(self, 'correlation_shm'):
            try:
                self.correlation_shm.close()
                logger.info("[WORKER] Closed correlation shared memory")
            except Exception as e:
                logger.warning(f"[WORKER] Error closing correlation shared memory: {e}")


def gui_processing_worker_main(buffer_coordinator_info: Dict,
                              config: Dict,
                              camera_indices: List[int],
                              recovery_buffer_names: Dict[int, str],
                              control_queue: mp.Queue,
                              status_queue: mp.Queue,
                              lsl_data_queue: mp.Queue,
                              pose_data_queue: mp.Queue,
                              participant_update_queue: mp.Queue,
                              enrollment_state_array: 'multiprocessing.sharedctypes.SynchronizedArray',
                              lock_state_array: 'multiprocessing.sharedctypes.SynchronizedArray',
                              participant_presence_array: 'multiprocessing.sharedctypes.SynchronizedArray',
                              enrollment_states_mapping: Dict[str, int]):
    """
    Main entry point for GUI processing worker process.

    Args:
        buffer_coordinator_info: Info to reconstruct BufferCoordinator (includes actual_buffer_names)
        config: Configuration dictionary
        camera_indices: List of active camera indices
        recovery_buffer_names: Recovery buffer names
        control_queue: Queue for receiving commands
        status_queue: Queue for sending status
        lsl_data_queue: Queue for sending LSL data to LSL helper process (must be passed at creation)
        pose_data_queue: Queue for sending pose data to GUI (must be passed at creation)
        participant_update_queue: Queue for enrollment state updates from EnrollmentWorker (event-driven)
        enrollment_state_array: Shared array for reading enrollment state from GUI process [participant_id] = state_code
        lock_state_array: Shared array for reading lock state from GUI process [participant_id] = 0/1
        participant_presence_array: Shared array for writing presence state to GUI process [participant_id] = 0/1
        enrollment_states_mapping: Dict mapping state names to integer codes
    """
    # NOTE: Logging is now configured at module-level (lines 20-31)
    # This ensures import errors are visible even if worker crashes during module load

    try:
        # CRITICAL: Log function entry IMMEDIATELY using direct stderr print
        # This bypasses logging framework to ensure visibility even if logging fails
        print("[WORKER ENTRY] gui_processing_worker_main() called", file=sys.stderr, flush=True)

        logger.info("[WORKER DEBUG] GUI processing worker starting...")
        sys.stderr.flush()  # Flush after critical startup message

        # Import BufferCoordinator here to avoid circular imports
        logger.info("[WORKER DEBUG] Importing BufferCoordinator...")
        sys.stderr.flush()

        from core.buffer_management.coordinator import BufferCoordinator

        logger.info("[WORKER DEBUG] BufferCoordinator imported successfully")
        sys.stderr.flush()
        
        # Store buffer registry and actual buffer names for direct connections
        buffer_registry = buffer_coordinator_info.get('buffer_registry', {})
        actual_buffer_names = buffer_coordinator_info.get('actual_buffer_names', {})
        
        logger.info(f"[WORKER DEBUG] Worker received buffer registry for {len(buffer_registry)} cameras")
        logger.info(f"[WORKER DEBUG] Worker received actual buffer names for {len(actual_buffer_names)} cameras")
        logger.info(f"[WORKER DEBUG] Camera indices: {camera_indices}")
        logger.info(f"[WORKER DEBUG] Recovery buffer names keys: {list(recovery_buffer_names.keys())}")
        
        # Reconstruct BufferCoordinator from info
        # Use create_coordinator_info=False to prevent child process from overwriting parent's file
        logger.info("[WORKER DEBUG] Reconstructing BufferCoordinator...")
        camera_count = buffer_coordinator_info.get('camera_count', 4)
        buffer_coordinator = BufferCoordinator(
            camera_count=camera_count,
            config=config,
            create_coordinator_info=False  # Don't create info file in child process
        )
        
        # Restore buffer registry and other state from parent
        buffer_coordinator.buffer_registry = buffer_registry

        # Override calculated values with those from parent to ensure consistency
        original_max_faces = buffer_coordinator.max_faces  # The value _calculate_max_faces() returned
        # CRITICAL FIX: Read from top level of buffer_coordinator_info, not from non-existent buffer_config sub-dict
        buffer_coordinator.max_faces = buffer_coordinator_info.get('max_faces', 2)  # Default 2 = participant_count (2)
        buffer_coordinator.ring_buffer_size = buffer_coordinator_info.get('ring_buffer_size', 16)
        buffer_coordinator.roi_buffer_size = buffer_coordinator_info.get('roi_buffer_size', 8)
        buffer_coordinator.gui_buffer_size = buffer_coordinator_info.get('gui_buffer_size', 8)

        # CRITICAL: Restore camera resolutions from parent
        buffer_coordinator.camera_resolutions = buffer_coordinator_info.get('camera_resolutions', {})

        logger.info(f"[WORKER DEBUG] BufferCoordinator reconstructed: cameras={camera_count}")
        logger.info(f"[WORKER DEBUG] max_faces: calculated={original_max_faces}, from_parent={buffer_coordinator.max_faces}")
        logger.info(f"[WORKER DEBUG] ring_buffer_size={buffer_coordinator.ring_buffer_size}")
        logger.info(f"[WORKER DEBUG] roi_buffer_size={buffer_coordinator.roi_buffer_size}")
        logger.info(f"[WORKER DEBUG] gui_buffer_size={buffer_coordinator.gui_buffer_size}")
        
        # Create and run worker with reconstructed BufferCoordinator
        logger.info("[WORKER DEBUG] Creating GUIProcessingWorker instance...")
        worker = GUIProcessingWorker(
            buffer_coordinator=buffer_coordinator,
            config=config,
            camera_indices=camera_indices,
            recovery_buffer_names=recovery_buffer_names,
            control_queue=control_queue,
            status_queue=status_queue,
            participant_update_queue=participant_update_queue,
            enrollment_state_array=enrollment_state_array,  # ARRAY-BASED IPC: Shared enrollment state for overlays
            lock_state_array=lock_state_array,  # ARRAY-BASED IPC: Shared lock state for slot reservation
            participant_presence_array=participant_presence_array,  # ARRAY-BASED IPC: Shared presence state for Absent/Present display
            enrollment_states_mapping=enrollment_states_mapping  # State name <-> code mapping
        )
        logger.info("[WORKER DEBUG] GUIProcessingWorker instance created successfully")
        
        # Pass buffer registry and actual buffer names to worker for connection
        logger.info("[WORKER DEBUG] Setting worker buffer connections...")
        worker.buffer_registry = buffer_registry
        worker.actual_buffer_names = actual_buffer_names
        worker.camera_resolutions = buffer_coordinator.camera_resolutions  # Pass actual camera resolutions
        worker.buffer_coordinator_info = buffer_coordinator_info

        # CRITICAL: Assign LSL data queue directly (passed via args, not control message)
        worker.lsl_data_queue = lsl_data_queue
        logger.info("[WORKER DEBUG] Assigned lsl_data_queue to worker")

        # Assign pose data queue for sending pose data to GUI
        worker.pose_data_queue = pose_data_queue
        logger.info("[WORKER DEBUG] Assigned pose_data_queue to worker")

        logger.info("[WORKER DEBUG] Worker buffer connections set")
        logger.info(f"[WORKER DEBUG] Camera resolutions: {worker.camera_resolutions}")

        logger.info("[WORKER DEBUG] Starting worker.run()...")
        worker.run()
        
    except Exception as e:
        logger.error(f"[WORKER DEBUG] GUI processing worker failed: {e}")
        import traceback
        logger.error(f"[WORKER DEBUG] Worker failure traceback: {traceback.format_exc()}")
        try:
            # Use non-blocking put to prevent queue overflow crashes
            status_queue.put({
                'type': 'error',
                'error': str(e)
            }, block=False)
        except queue.Full:
            logger.error(f"[WORKER DEBUG] Status queue full! Cannot send error status to GUI")
        except Exception as queue_error:
            logger.error(f"[WORKER DEBUG] Failed to send error status: {queue_error}")