"""
Single Participant Manager - Unified Participant Management System

This unified class consolidates all participant management functionality:
- Multi-camera data fusion (from ParticipantDataMerger)
- IPC query buffer communication (from ParticipantQueryBuffer)
- Simplified enrollment state management
- Single participant tracking

Designed specifically for single-participant mode.
"""

import numpy as np
import time
import logging
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from ..data_processing.quality_estimator import StabilityTracker, estimate_capture_quality

logger = logging.getLogger(__name__)


class EnrollmentState(Enum):
    """Enrollment states for single participant."""
    IDLE = "idle"
    ENROLLING = "enrolling"
    ENROLLED = "enrolled"


@dataclass
class CameraData:
    """Data from a single camera for the participant."""
    camera_idx: int
    landmarks: np.ndarray  # (478, 3) MediaPipe landmarks
    blendshapes: np.ndarray  # (52,) MediaPipe blendshape coefficients
    bbox: np.ndarray  # (4,) [x1, y1, x2, y2]
    quality: float  # Overall quality score 0.0-1.0
    quality_breakdown: Dict[str, float]  # Individual metric scores


@dataclass
class ParticipantInfo:
    """Information about the enrolled participant."""
    participant_id: int
    name: str
    embedding: Optional[np.ndarray] = None  # Face embedding if available
    enrolled_at: float = field(default_factory=time.time)


class SingleParticipantManager:
    """
    Unified manager for single participant tracking and data management.

    Features:
    1. Multi-camera data fusion with quality-weighted averaging
    2. Shared memory query buffer for IPC communication
    3. Simple enrollment state management
    4. Participant identification and tracking

    Usage:
        # Initialize
        manager = SingleParticipantManager(config)

        # Enrollment
        manager.start_enrollment()
        # ... collect face data ...
        manager.complete_enrollment("John Doe", embedding)

        # Data fusion (each frame)
        for camera in cameras:
            manager.add_camera_data(camera_idx, landmarks, blendshapes, bbox, frame_shape)

        merged_landmarks, merged_blends = manager.get_merged_data()
        manager.clear_frame()

        # Query buffer (for recovery queries)
        query_id = manager.submit_query(camera_idx, bbox, shape, embedding)
        response = manager.get_response(query_id)
    """

    def __init__(self, config: Optional[Dict] = None, enable_query_buffer: bool = False,
                 query_buffer_name: Optional[str] = None, face_recognition_callback=None):
        """
        Initialize the unified participant manager.

        Args:
            config: Configuration dict
            enable_query_buffer: Whether to enable shared memory query buffer
            query_buffer_name: Name for shared memory (e.g., 'yq_recovery_0')
            face_recognition_callback: Optional callback for face recognition
        """
        self.config = config or {}
        self.face_recognition_callback = face_recognition_callback

        # ===== ENROLLMENT STATE =====
        self.enrollment_state = EnrollmentState.IDLE
        self.participant_info: Optional[ParticipantInfo] = None
        self.enrollment_embeddings = []  # Collect multiple embeddings during enrollment

        # ===== DATA FUSION CONFIGURATION =====
        merging_config = self.config.get('multi_camera_merging', {})

        # Quality estimation config
        quality_config = merging_config.get('quality_estimation', {})
        self.quality_weights = quality_config.get('weights', {
            'size': 0.4,
            'frontal': 0.3,
            'stability': 0.3
        })
        stability_window = quality_config.get('stability_window_size', 5)
        self.min_quality_threshold = quality_config.get('min_quality_threshold', 0.1)

        # Merging strategy
        self.merging_enabled = merging_config.get('enabled', True)
        self.merging_strategy = merging_config.get('merging_strategy', 'weighted_average')
        self.log_quality_scores = merging_config.get('log_quality_scores', False)

        # Per-frame accumulator for data fusion
        self.camera_data_accumulator: List[CameraData] = []

        # Temporal stability tracker (persists across frames)
        self.stability_tracker = StabilityTracker(window_size=stability_window)

        # Statistics
        self.merge_stats = {
            'total_merges': 0,
            'single_camera': 0,
            'multi_camera': 0,
            'quality_improvements': 0
        }

        # ===== QUERY BUFFER (IPC) =====
        self.query_buffer_enabled = enable_query_buffer
        self.query_buffer_name = query_buffer_name
        self.shm = None
        self._created_shm = False

        if enable_query_buffer and query_buffer_name:
            self._init_query_buffer()

        logger.info(f"[SingleParticipantManager] Initialized (query_buffer={'enabled' if enable_query_buffer else 'disabled'})")

    # ==================== ENROLLMENT MANAGEMENT ====================

    def start_enrollment(self, participant_id: int = 1) -> bool:
        """
        Start enrollment for the participant.

        Args:
            participant_id: ID to assign (default: 1 for single participant)

        Returns:
            True if enrollment started successfully
        """
        if self.enrollment_state == EnrollmentState.ENROLLED:
            logger.warning("[Enrollment] Already enrolled, cannot start new enrollment")
            return False

        self.enrollment_state = EnrollmentState.ENROLLING
        self.enrollment_embeddings = []
        logger.info(f"[Enrollment] Started enrollment for participant {participant_id}")
        return True

    def add_enrollment_embedding(self, embedding: np.ndarray):
        """
        Add a face embedding during enrollment.

        Args:
            embedding: Face embedding vector (512,)
        """
        if self.enrollment_state != EnrollmentState.ENROLLING:
            logger.warning("[Enrollment] Not in enrolling state, cannot add embedding")
            return

        self.enrollment_embeddings.append(embedding.copy())
        logger.debug(f"[Enrollment] Added embedding {len(self.enrollment_embeddings)}")

    def complete_enrollment(self, name: str, embedding: Optional[np.ndarray] = None,
                           participant_id: int = 1) -> bool:
        """
        Complete enrollment with participant information.

        Args:
            name: Participant name
            embedding: Optional final face embedding (if not provided, average of collected embeddings)
            participant_id: ID to assign (default: 1)

        Returns:
            True if enrollment completed successfully
        """
        if self.enrollment_state != EnrollmentState.ENROLLING:
            logger.warning("[Enrollment] Not in enrolling state, cannot complete")
            return False

        # Use provided embedding or average of collected embeddings
        final_embedding = None
        if embedding is not None:
            final_embedding = embedding.copy()
        elif len(self.enrollment_embeddings) > 0:
            final_embedding = np.mean(self.enrollment_embeddings, axis=0)
            logger.info(f"[Enrollment] Averaged {len(self.enrollment_embeddings)} embeddings")

        self.participant_info = ParticipantInfo(
            participant_id=participant_id,
            name=name,
            embedding=final_embedding
        )

        self.enrollment_state = EnrollmentState.ENROLLED
        self.enrollment_embeddings = []  # Clear temp embeddings

        logger.info(f"[Enrollment] Completed for '{name}' (ID={participant_id})")
        return True

    def cancel_enrollment(self):
        """Cancel ongoing enrollment."""
        if self.enrollment_state == EnrollmentState.ENROLLING:
            self.enrollment_state = EnrollmentState.IDLE
            self.enrollment_embeddings = []
            logger.info("[Enrollment] Cancelled")

    def get_enrollment_state(self) -> str:
        """Get current enrollment state."""
        return self.enrollment_state.value

    def get_participant_id(self) -> Optional[int]:
        """Get the enrolled participant ID."""
        return self.participant_info.participant_id if self.participant_info else None

    def get_participant_name(self, participant_id: int = 1) -> str:
        """
        Get participant name.

        Args:
            participant_id: Participant ID (ignored in single participant mode)

        Returns:
            Participant name or generic name if not enrolled
        """
        if self.participant_info and self.participant_info.participant_id == participant_id:
            return self.participant_info.name
        return f"Person {participant_id}"

    def reset(self):
        """Reset participant state (clear enrollment)."""
        self.enrollment_state = EnrollmentState.IDLE
        self.participant_info = None
        self.enrollment_embeddings = []
        self.camera_data_accumulator = []
        self.stability_tracker.clear_all()
        logger.info("[SingleParticipantManager] Reset participant state")

    # ==================== MULTI-CAMERA DATA FUSION ====================

    def add_camera_data(
        self,
        camera_idx: int,
        landmarks: np.ndarray,
        blendshapes: np.ndarray,
        bbox: np.ndarray,
        frame_shape: Tuple[int, int],
        participant_id: int = 1
    ):
        """
        Add data from a camera for the participant.

        Args:
            camera_idx: Camera index (0, 1, 2, etc.)
            landmarks: Face landmarks (478, 3) - x,y,z pixel coordinates
            blendshapes: Blendshape coefficients (52,) - 0.0-1.0 range
            bbox: Bounding box [x1, y1, x2, y2] in pixels
            frame_shape: (height, width) of source frame
            participant_id: Participant ID (default: 1 for single participant)
        """
        if not self.merging_enabled:
            # Merging disabled - just store first camera's data
            if len(self.camera_data_accumulator) == 0:
                self.camera_data_accumulator.append(CameraData(
                    camera_idx=camera_idx,
                    landmarks=landmarks,
                    blendshapes=blendshapes,
                    bbox=bbox,
                    quality=1.0,
                    quality_breakdown={'size': 1.0, 'frontal': 1.0, 'stability': 1.0}
                ))
            return

        # Estimate quality for this camera's detection
        quality, breakdown = estimate_capture_quality(
            bbox=bbox,
            landmarks=landmarks,
            frame_shape=frame_shape,
            stability_tracker=self.stability_tracker,
            participant_id=participant_id,
            camera_idx=camera_idx,
            weights=self.quality_weights
        )

        # Filter out very low quality detections
        if quality < self.min_quality_threshold:
            logger.debug(f"[MERGE] Filtered low quality: cam{camera_idx} "
                        f"quality={quality:.3f} < threshold={self.min_quality_threshold}")
            return

        # Store camera data
        camera_data = CameraData(
            camera_idx=camera_idx,
            landmarks=landmarks.copy(),
            blendshapes=blendshapes.copy(),
            bbox=bbox.copy(),
            quality=quality,
            quality_breakdown=breakdown
        )

        self.camera_data_accumulator.append(camera_data)

        if self.log_quality_scores:
            logger.info(f"[MERGE] cam{camera_idx}: quality={quality:.3f} "
                       f"(size={breakdown['size']:.2f}, "
                       f"frontal={breakdown['frontal']:.2f}, "
                       f"stability={breakdown['stability']:.2f})")

    def get_merged_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get merged landmarks and blendshapes from all cameras.

        Returns:
            (merged_landmarks, merged_blendshapes) or (None, None) if no data
        """
        if len(self.camera_data_accumulator) == 0:
            return None, None

        if len(self.camera_data_accumulator) == 1:
            # Single camera - no merging needed
            self.merge_stats['single_camera'] += 1
            cam_data = self.camera_data_accumulator[0]
            return cam_data.landmarks, cam_data.blendshapes

        # Multi-camera merge
        self.merge_stats['multi_camera'] += 1
        self.merge_stats['total_merges'] += 1

        if self.merging_strategy == 'weighted_average':
            merged_landmarks, merged_blends = self._weighted_average_merge()
        elif self.merging_strategy == 'best_camera':
            merged_landmarks, merged_blends = self._best_camera_merge()
        else:
            logger.warning(f"[MERGE] Unknown strategy '{self.merging_strategy}', using weighted_average")
            merged_landmarks, merged_blends = self._weighted_average_merge()

        return merged_landmarks, merged_blends

    def _weighted_average_merge(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge using quality-weighted average.

        Higher quality cameras contribute more to the final result.
        """
        cameras = self.camera_data_accumulator

        # Calculate total quality for normalization
        total_quality = sum(cam.quality for cam in cameras)

        if total_quality < 0.001:
            # Degenerate case - use first camera
            logger.warning("[MERGE] Zero total quality, using first camera")
            return cameras[0].landmarks, cameras[0].blendshapes

        # Weighted merge
        merged_landmarks = np.zeros_like(cameras[0].landmarks, dtype=np.float32)
        merged_blendshapes = np.zeros_like(cameras[0].blendshapes, dtype=np.float32)

        weights_str = []
        for cam in cameras:
            weight = cam.quality / total_quality
            merged_landmarks += weight * cam.landmarks
            merged_blendshapes += weight * cam.blendshapes
            weights_str.append(f"cam{cam.camera_idx}={weight:.2f}")

        if self.log_quality_scores:
            logger.info(f"[MERGE] {len(cameras)} cameras, weights=[{', '.join(weights_str)}]")

        return merged_landmarks, merged_blendshapes

    def _best_camera_merge(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge by selecting the best camera (highest quality).

        Simple strategy that avoids averaging artifacts but may have discontinuities.
        """
        cameras = self.camera_data_accumulator
        best_camera = max(cameras, key=lambda c: c.quality)

        if self.log_quality_scores:
            logger.info(f"[MERGE] Best camera = {best_camera.camera_idx} "
                       f"(quality={best_camera.quality:.3f})")

        return best_camera.landmarks, best_camera.blendshapes

    def clear_frame(self):
        """Clear accumulator for next frame. Call at end of each frame."""
        self.camera_data_accumulator.clear()

    def get_merge_stats(self) -> Dict:
        """Get merging statistics."""
        stats = self.merge_stats.copy()
        if stats['total_merges'] > 0:
            stats['multi_camera_percentage'] = (
                100.0 * stats['multi_camera'] / stats['total_merges']
            )
        else:
            stats['multi_camera_percentage'] = 0.0
        return stats

    def reset_merge_stats(self):
        """Reset statistics."""
        self.merge_stats = {
            'total_merges': 0,
            'single_camera': 0,
            'multi_camera': 0,
            'quality_improvements': 0
        }

    # ==================== QUERY BUFFER (IPC) ====================

    def _init_query_buffer(self, max_queries: int = 16):
        """
        Initialize shared memory query buffer.

        Args:
            max_queries: Maximum concurrent queries
        """
        self.max_queries = max_queries

        # Query structure size (in bytes)
        # camera_idx (4) + bbox (16) + shape (478*3*4) + embedding (512*4) + metadata (64)
        self.query_size = 4 + 16 + 478*3*4 + 512*4 + 64  # ~7.9KB per query
        self.response_size = 128  # participant_id + scores + metadata

        # Total buffer size
        total_size = (self.query_size + self.response_size) * max_queries + 1024  # coordination data

        try:
            # Create shared memory
            self.shm = shared_memory.SharedMemory(create=True, name=self.query_buffer_name, size=total_size)
            self._created_shm = True

            # Create numpy views for structured access
            self._init_buffer_views()

            # Initialize atomic indices IN SHARED MEMORY
            self.coordination[0] = 0  # query_write_idx
            self.coordination[1] = 0  # query_read_idx
            self.coordination[2] = 0  # response_write_idx
            self.coordination[3] = 0  # response_read_idx
            self.coordination[4] = max_queries  # max_queries for validation
            self.coordination[5] = int(time.time())  # creation timestamp

            logger.info(f"[QueryBuffer] Created '{self.query_buffer_name}' with {max_queries} slots")

        except Exception as e:
            logger.error(f"[QueryBuffer] Failed to create: {e}")
            self.query_buffer_enabled = False

    def _init_buffer_views(self):
        """Initialize structured numpy views into shared memory."""
        # Coordination area (first 1024 bytes)
        self.coordination = np.frombuffer(self.shm.buf, dtype=np.int32, count=256, offset=0)

        # Query area
        query_offset = 1024
        self.query_camera_idx = np.frombuffer(
            self.shm.buf, dtype=np.int32,
            count=self.max_queries,
            offset=query_offset
        )

        query_offset += self.max_queries * 4
        self.query_bboxes = np.frombuffer(
            self.shm.buf, dtype=np.float32,
            count=self.max_queries * 4,
            offset=query_offset
        ).reshape(self.max_queries, 4)

        query_offset += self.max_queries * 16
        self.query_shapes = np.frombuffer(
            self.shm.buf, dtype=np.float32,
            count=self.max_queries * 478 * 3,
            offset=query_offset
        ).reshape(self.max_queries, 478, 3)

        query_offset += self.max_queries * 478 * 3 * 4
        self.query_embeddings = np.frombuffer(
            self.shm.buf, dtype=np.float32,
            count=self.max_queries * 512,
            offset=query_offset
        ).reshape(self.max_queries, 512)

        # Response area
        response_offset = 1024 + self.query_size * self.max_queries
        self.response_participant_ids = np.frombuffer(
            self.shm.buf, dtype=np.int32,
            count=self.max_queries,
            offset=response_offset
        )

        response_offset += self.max_queries * 4
        self.response_scores = np.frombuffer(
            self.shm.buf, dtype=np.float32,
            count=self.max_queries * 16,
            offset=response_offset
        ).reshape(self.max_queries, 16)

    def submit_query(self, camera_idx: int, bbox: np.ndarray,
                    shape: Optional[np.ndarray] = None,
                    embedding: Optional[np.ndarray] = None,
                    zombie_track_id: Optional[int] = None,
                    zombie_participant_id: Optional[int] = None) -> Optional[int]:
        """
        Submit a participant recovery query.

        Args:
            camera_idx: Camera index
            bbox: Bounding box [x1, y1, x2, y2]
            shape: Optional face landmarks (478, 3)
            embedding: Optional face embedding (512,)
            zombie_track_id: Optional zombie track ID
            zombie_participant_id: Optional zombie participant ID

        Returns:
            Query ID for retrieving response, or None if query buffer disabled
        """
        if not self.query_buffer_enabled or self.shm is None:
            return None

        # Atomic increment of write index in shared memory
        query_write_idx = int(self.coordination[0])
        slot = query_write_idx % self.max_queries
        query_id = query_write_idx
        self.coordination[0] = query_write_idx + 1  # Atomic write

        # Write query data
        self.query_camera_idx[slot] = camera_idx
        self.query_bboxes[slot] = bbox

        if shape is not None:
            self.query_shapes[slot] = shape
        else:
            self.query_shapes[slot] = 0  # Clear

        if embedding is not None:
            self.query_embeddings[slot] = embedding
        else:
            self.query_embeddings[slot] = 0  # Clear

        # Store zombie track info in coordination array
        if zombie_track_id is not None:
            self.coordination[24 + slot] = zombie_track_id
        else:
            self.coordination[24 + slot] = -1  # Use -1 for None

        if zombie_participant_id is not None:
            self.coordination[40 + slot] = zombie_participant_id
        else:
            self.coordination[40 + slot] = -1  # Use -1 for None

        # Write sequence/timestamp to signal data is ready
        timestamp_us = int(time.time() * 1000000) & 0x7FFFFFFF  # Limit to 31-bit signed int
        self.coordination[8 + slot] = timestamp_us  # Mark slot as ready with timestamp

        logger.debug(f"[QueryBuffer] Submitted query {query_id} to slot {slot}")
        return query_id

    def get_query(self) -> Optional[Dict[str, Any]]:
        """
        Get next query from buffer (non-blocking with polling).

        Returns:
            Query data or None if no query available
        """
        if not self.query_buffer_enabled or self.shm is None:
            return None

        current_read_idx = int(self.coordination[1])
        current_write_idx = int(self.coordination[0])

        # Check if there are unread queries
        if current_read_idx >= current_write_idx:
            return None  # No new queries

        # Get next slot to read
        slot = current_read_idx % self.max_queries

        # Check if this slot has data ready (timestamp > 0)
        slot_timestamp = self.coordination[8 + slot]
        if slot_timestamp == 0:
            return None  # Slot not ready

        # Read query data
        query_id = current_read_idx
        query = {
            'query_id': query_id,
            'camera_idx': int(self.query_camera_idx[slot]),
            'bbox': self.query_bboxes[slot].copy(),
            'shape': self.query_shapes[slot].copy() if np.any(self.query_shapes[slot]) else None,
            'embedding': self.query_embeddings[slot].copy() if np.any(self.query_embeddings[slot]) else None,
            'zombie_track_id': int(self.coordination[24 + slot]) if self.coordination[24 + slot] >= 0 else None,
            'zombie_participant_id': int(self.coordination[40 + slot]) if self.coordination[40 + slot] >= 0 else None
        }

        # Mark slot as consumed and advance read index
        self.coordination[8 + slot] = 0  # Clear timestamp
        self.coordination[1] = current_read_idx + 1  # Advance read index

        logger.debug(f"[QueryBuffer] Retrieved query {query_id} from slot {slot}")
        return query

    def submit_response(self, query_id: int, participant_id: Optional[int],
                       scores: Dict[str, float]):
        """
        Submit response to a query.

        Args:
            query_id: Query ID from get_query()
            participant_id: Matched participant ID or None
            scores: Matching scores dict
        """
        if not self.query_buffer_enabled or self.shm is None:
            return

        slot = query_id % self.max_queries

        # Write response
        self.response_participant_ids[slot] = participant_id if participant_id else -1

        # Pack scores into response array
        score_array = np.zeros(16, dtype=np.float32)
        score_array[0] = scores.get('embedding', 0.0)
        score_array[1] = scores.get('shape', 0.0)
        score_array[2] = scores.get('position', 0.0)
        score_array[3] = scores.get('combined', 0.0)
        self.response_scores[slot] = score_array

        # Signal response available by setting timestamp in coordination
        response_slot_offset = 8 + self.max_queries + slot  # After query timestamps
        timestamp_us = int(time.time() * 1000000) & 0x7FFFFFFF
        self.coordination[response_slot_offset] = timestamp_us

        logger.debug(f"[QueryBuffer] Submitted response for query {query_id}, participant_id={participant_id}")

    def get_response(self, query_id: int, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        Get response for a query with polling.

        Args:
            query_id: Query ID from submit_query()
            timeout: Timeout in seconds

        Returns:
            Response dict or None if timeout
        """
        if not self.query_buffer_enabled or self.shm is None:
            return None

        slot = query_id % self.max_queries
        response_slot_offset = 8 + self.max_queries + slot

        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if response is available (timestamp > 0)
            if self.coordination[response_slot_offset] > 0:
                # Read response data
                participant_id = int(self.response_participant_ids[slot])
                if participant_id == -1:
                    participant_id = None

                scores = {
                    'embedding': float(self.response_scores[slot, 0]),
                    'shape': float(self.response_scores[slot, 1]),
                    'position': float(self.response_scores[slot, 2]),
                    'combined': float(self.response_scores[slot, 3])
                }

                # Clear response timestamp to mark as consumed
                self.coordination[response_slot_offset] = 0

                logger.debug(f"[QueryBuffer] Retrieved response for query {query_id}, participant_id={participant_id}")
                return {
                    'participant_id': participant_id,
                    'scores': scores
                }

            # Brief sleep to avoid busy waiting
            time.sleep(0.001)

        return None  # Timeout

    @classmethod
    def connect(cls, name: str, config: Optional[Dict] = None, max_queries: int = 16):
        """
        Connect to an existing SingleParticipantManager's query buffer.

        Args:
            name: Shared memory name to connect to
            config: Configuration dict
            max_queries: Expected number of query slots

        Returns:
            SingleParticipantManager instance connected to existing memory
        """
        instance = cls(config=config, enable_query_buffer=False)
        instance.query_buffer_enabled = True
        instance.query_buffer_name = name
        instance.max_queries = max_queries
        instance._created_shm = False

        # Connect to existing shared memory
        instance.shm = shared_memory.SharedMemory(name=name)

        # Initialize buffer views
        instance.query_size = 4 + 16 + 478*3*4 + 512*4 + 64
        instance.response_size = 128
        instance._init_buffer_views()

        # Validate connection by checking coordination data
        stored_max_queries = int(instance.coordination[4])
        creation_time = int(instance.coordination[5])

        if stored_max_queries != max_queries:
            logger.warning(f"[QueryBuffer] Max queries mismatch: expected {max_queries}, found {stored_max_queries}")

        logger.info(f"[QueryBuffer] Connected to '{name}' (created at {creation_time}, max_queries={stored_max_queries})")
        return instance

    # ==================== COMPATIBILITY METHODS ====================
    # Methods to maintain compatibility with old stub interfaces

    @property
    def accumulator(self):
        """
        Backward compatibility property for ParticipantDataMerger.accumulator.

        The old ParticipantDataMerger stored data as {participant_id: [CameraData, ...]}.
        Our SingleParticipantManager just stores [CameraData, ...] directly.
        This property wraps it in the old format for compatibility.
        """
        # Return dict format that LSLHelper expects
        if len(self.camera_data_accumulator) > 0:
            # Assume participant_id=1 for single participant mode
            return {1: self.camera_data_accumulator}
        return {}

    def update(self, *args, **kwargs):
        """Compatibility stub for update() method."""
        pass

    def get_global_id(self, *args, **kwargs):
        """Compatibility stub - always returns participant ID."""
        return self.get_participant_id()

    def register_participant(self, *args, **kwargs):
        """Compatibility stub for registration."""
        return self.get_participant_id()

    def get_all_states(self) -> Dict:
        """Get enrollment states for all participants (single participant version)."""
        if self.participant_info:
            return {
                self.participant_info.participant_id: self.enrollment_state.value
            }
        return {}

    # ==================== CLEANUP ====================

    def cleanup(self):
        """Cleanup resources."""
        # Clear data
        self.camera_data_accumulator.clear()
        self.stability_tracker.clear_all()

        # Cleanup shared memory if enabled
        if self.query_buffer_enabled and self.shm is not None:
            try:
                # Delete numpy array views to release references
                if hasattr(self, 'coordination'):
                    del self.coordination
                if hasattr(self, 'query_camera_idx'):
                    del self.query_camera_idx
                if hasattr(self, 'query_bboxes'):
                    del self.query_bboxes
                if hasattr(self, 'query_shapes'):
                    del self.query_shapes
                if hasattr(self, 'query_embeddings'):
                    del self.query_embeddings
                if hasattr(self, 'response_participant_ids'):
                    del self.response_participant_ids
                if hasattr(self, 'response_scores'):
                    del self.response_scores

                # Close shared memory
                self.shm.close()

                # Only unlink if we created the buffer
                if self._created_shm:
                    self.shm.unlink()
                    logger.info(f"[QueryBuffer] Cleaned up and unlinked '{self.query_buffer_name}'")
                else:
                    logger.info(f"[QueryBuffer] Disconnected from '{self.query_buffer_name}'")

            except Exception as e:
                logger.error(f"[QueryBuffer] Failed to cleanup: {e}")

        logger.info("[SingleParticipantManager] Cleanup complete")

    def __del__(self):
        """Destructor - ensure cleanup on garbage collection."""
        # Explicit cleanup handled via cleanup() method
        pass
