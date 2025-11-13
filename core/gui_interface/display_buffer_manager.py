"""
Display Buffer Manager - Fast read-only access to display-ready data for GUI rendering.
Part of the GUI process separation architecture for achieving 30+ FPS.
"""

import multiprocessing as mp
import numpy as np
import struct
import time
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger('DisplayBufferManager')


@dataclass 
class DisplayData:
    """Display-ready data for immediate rendering."""
    frame_id: int
    timestamp: float
    n_faces: int
    faces: List[Dict[str, Any]]
    frame_bgr: Optional[np.ndarray] = None


class DisplayBufferReader:
    """
    Fast reader for display buffers created by GUIProcessingWorker.
    Optimized for zero-copy reads with minimal processing.
    Now reads frame data from display buffer.
    """

    def __init__(self, camera_idx: int, buffer_name: str, buffer_coordinator):
        self.camera_idx = camera_idx
        self.buffer_name = buffer_name
        self.last_frame_id = -1  # Track frame_id to detect duplicate reads
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

        try:
            # Connect to existing display buffer
            self.shm = mp.shared_memory.SharedMemory(name=buffer_name)

            # Create numpy view for write index
            self.write_index_view = np.ndarray(
                (1,), dtype=np.uint64, buffer=self.shm.buf[0:8]
            )

            # Track whether we've done first scan diagnostic
            self._first_scan_logged = False

            logger.info(f"Connected to display buffer for camera {camera_idx}: {buffer_name} (ring size: {self.RING_SIZE})")

        except Exception as e:
            logger.error(f"Failed to connect to display buffer {buffer_name}: {e}")
            raise
            
    def read_display_data_atomic(self) -> Optional[DisplayData]:
        """
        Read display data using LIFO ring buffer scan.

        RING BUFFER FIX: Always scans all slots to find the one with the highest frame_id
        (most recent frame). This completely decouples read/write frequencies - writer can
        write at 30 FPS while reader polls at 60 FPS without missing data.

        Returns None only if frame_id hasn't changed since last read (true duplicate).
        """
        try:
            # LIFO RING SCAN: Find the slot with the highest frame_id (most recent)
            latest_frame_id = -1
            latest_slot = -1
            latest_timestamp = 0.0
            latest_n_faces = 0

            # DIAGNOSTIC DISABLED: Initialize scan counter (kept for compatibility)
            if not hasattr(self, '_scan_count'):
                self._scan_count = 0
            self._scan_count += 1

            # DIAGNOSTIC DISABLED: Uncomment to debug ring buffer scans
            # should_log_diagnostic = (not self._first_scan_logged) or (self._scan_count % 30 == 0)
            # if should_log_diagnostic and not self._first_scan_logged:
            #     logger.info(f"[DISPLAY BUFFER DIAGNOSTIC] Camera {self.camera_idx}: Starting first display metadata scan")
            #     logger.info(f"[DISPLAY BUFFER DIAGNOSTIC] Display ring buffer size: {self.RING_SIZE}")
            #     logger.info(f"[DISPLAY BUFFER DIAGNOSTIC] Buffer name: {self.buffer_name}")
            #     logger.info(f"[DISPLAY BUFFER DIAGNOSTIC] Buffer size: {self.shm.size}")
            #     self._first_scan_logged = True
            # elif should_log_diagnostic:
            #     logger.info(f"[DISPLAY BUFFER SCAN #{self._scan_count}] Camera {self.camera_idx}: Periodic ring buffer scan")
            if not self._first_scan_logged:
                self._first_scan_logged = True

            for slot in range(self.RING_SIZE):
                # Calculate offset for this slot
                slot_offset = self.WRITE_INDEX_SIZE + (slot * self.SLOT_SIZE)
                metadata_offset = slot_offset

                # Read metadata header to get frame_id
                metadata_bytes = self.shm.buf[metadata_offset:metadata_offset + self.FRAME_METADATA_SIZE]
                frame_id, timestamp, n_faces = struct.unpack_from('QdI', metadata_bytes, 0)

                # DIAGNOSTIC DISABLED: Uncomment to debug slot scanning
                # if should_log_diagnostic:
                #     logger.info(f"[RING SCAN] Camera {self.camera_idx} Slot {slot}: "
                #               f"frame_id={frame_id}, timestamp={timestamp:.3f}, n_faces={n_faces}")

                # Skip uninitialized slots (frame_id = 0 from initialization)
                # Also skip slots with frame_id == 0 if we already have a valid frame
                if frame_id == 0 and latest_frame_id > 0:
                    continue

                # Update latest if this slot has newer frame
                if frame_id > latest_frame_id:
                    latest_frame_id = frame_id
                    latest_slot = slot
                    latest_timestamp = timestamp
                    latest_n_faces = n_faces

            # DIAGNOSTIC DISABLED: Uncomment to debug scan results
            # if should_log_diagnostic:
            #     logger.info(f"[RING SCAN RESULT] Camera {self.camera_idx}: "
            #               f"latest_frame_id={latest_frame_id}, latest_slot={latest_slot}, "
            #               f"last_frame_id={self.last_frame_id}")

            # If no valid slot found, return None
            if latest_slot < 0:
                # Always log this condition as it indicates a problem
                logger.warning(f"[RING BUFFER ERROR] Camera {self.camera_idx}: No valid slot found! "
                             f"(scan #{self._scan_count}, all slots may be uninitialized or corrupted)")
                return None

            # Check if this is the same frame_id as last read (true duplicate)
            if latest_frame_id == self.last_frame_id:
                # Track duplicate count for debugging (kept for compatibility)
                if not hasattr(self, '_duplicate_count'):
                    self._duplicate_count = 0
                self._duplicate_count += 1

                # DIAGNOSTIC DISABLED: Uncomment to debug duplicate frames
                # if self._duplicate_count <= 100 or self._duplicate_count % 30 == 0:
                #     logger.info(f"[DUPLICATE FRAME] Camera {self.camera_idx}: Duplicate frame_id={latest_frame_id} "
                #                f"(duplicate #{self._duplicate_count}, scan #{self._scan_count}) - "
                #                f"reader polling faster than writer OR writer stalled")
                return None  # No new frame - reader polling faster than writer

            # Track successful reads (kept for compatibility)
            if not hasattr(self, '_successful_read_count'):
                self._successful_read_count = 0
            self._successful_read_count += 1

            # DIAGNOSTIC DISABLED: Uncomment to debug buffer progress
            # if self._successful_read_count % 100 == 0:
            #     # Re-scan all slots to show current ring buffer state
            #     slot_states = []
            #     for s in range(self.RING_SIZE):
            #         s_offset = self.WRITE_INDEX_SIZE + (s * self.SLOT_SIZE)
            #         s_bytes = self.shm.buf[s_offset:s_offset + self.FRAME_METADATA_SIZE]
            #         s_frame_id, s_ts, s_faces = struct.unpack_from('QdI', s_bytes, 0)
            #         slot_states.append(f"slot{s}={s_frame_id}")
            #     logger.info(f"[DISPLAY BUFFER PROGRESS] Camera {self.camera_idx}: Read #{self._successful_read_count}, "
            #               f"frame_id={latest_frame_id}, [{', '.join(slot_states)}]")

            # Read face data from latest slot
            slot_offset = self.WRITE_INDEX_SIZE + (latest_slot * self.SLOT_SIZE)
            metadata_offset = slot_offset
            faces = []
            face_offset = metadata_offset + self.FRAME_METADATA_SIZE

            for i in range(min(latest_n_faces, self.MAX_FACES)):
                face_data = self._read_face_data(face_offset + (i * self.FACE_DATA_SIZE))
                if face_data:
                    faces.append(face_data)

            # Update last frame_id to prevent returning same frame again
            self.last_frame_id = latest_frame_id

            return DisplayData(
                frame_id=latest_frame_id,
                timestamp=latest_timestamp,
                n_faces=latest_n_faces,
                faces=faces,
                frame_bgr=None  # Frame fetched separately from camera buffer
            )

        except Exception as e:
            logger.error(f"[RING BUFFER READ ERROR] Camera {self.camera_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _read_face_data(self, offset: int) -> Optional[Dict[str, Any]]:
        """Read and unpack face data from buffer."""
        try:
            # Read face data bytes
            face_bytes = self.shm.buf[offset:offset + self.FACE_DATA_SIZE]
            
            # Unpack fixed fields (removed obsolete is_tracking field)
            # Format: 3 SIGNED ints (face_id, participant_id, track_id) + 5 floats (bbox×4, confidence) + 1 unsigned int (display_flags)
            # NOTE: Changed from 'I' to 'i' for first 3 fields to support -1 sentinel values
            (face_id, participant_id, track_id,
             x1, y1, x2, y2, confidence,
             display_flags) = struct.unpack_from(
                'iiifffffI', face_bytes, 0
            )

            # Read label
            label_offset = struct.calcsize('iiifffffI')
            label_bytes = face_bytes[label_offset:label_offset + self.MAX_LABEL_LENGTH]
            # Convert memoryview to bytes for split operation
            label = bytes(label_bytes).split(b'\x00')[0].decode('utf-8', errors='ignore')
            
            # Read landmarks (MediaPipe outputs 478 landmarks)
            landmarks_offset = label_offset + self.MAX_LABEL_LENGTH
            landmarks_bytes = face_bytes[landmarks_offset:landmarks_offset + (self.landmark_count * 3 * 4)]
            landmarks = np.frombuffer(landmarks_bytes, dtype=np.float32).reshape((self.landmark_count, 3))
            
            # Check if landmarks are valid (not all zeros)
            # Check only x,y coordinates for validity (z can be 0)
            if np.all(landmarks[:, :2] == 0):
                landmarks = None
                
            return {
                'face_id': face_id,
                'participant_id': participant_id,
                'track_id': track_id,
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'display_flags': display_flags,
                'label': label,
                'landmarks': landmarks
            }
            
        except Exception as e:
            logger.error(f"Error reading face data: {e}")
            return None
            
    def cleanup(self):
        """Clean up shared memory connection."""
        try:
            self.shm.close()
        except:
            pass


class DisplayBufferManager:
    """
    Manager for all display buffers in the GUI process.
    Provides fast, zero-copy access to display-ready data.
    """

    def __init__(self, buffer_coordinator):
        self.coordinator = buffer_coordinator
        self.display_readers = {}  # camera_index -> DisplayBufferReader
        self.frame_buffers = {}    # camera_index -> frame buffer connection
        self.pose_frame_buffers = {}  # camera_index -> pose frame buffer connection (contains skeleton overlays)
        self.last_update_times = {}
        self.performance_stats = {
            'read_times': [],
            'frame_counts': {},
            'last_report': time.time()
        }
        
    def connect_display_buffer(self, camera_idx: int, buffer_name: str) -> bool:
        """
        Connect to a display buffer created by GUIProcessingWorker.

        Args:
            camera_idx: Camera index
            buffer_name: Shared memory name of display buffer

        Returns:
            Success status
        """
        try:
            reader = DisplayBufferReader(camera_idx, buffer_name, self.coordinator)
            self.display_readers[camera_idx] = reader
            self.last_update_times[camera_idx] = time.time()
            logger.info(f"Connected to display buffer for camera {camera_idx}")

            # Show if this is initial connection or reconnection
            if len(self.display_readers) == 1:
                logger.info(f"[BUFFER MANAGER] First camera connected (initial startup)")
            else:
                logger.info(f"[BUFFER MANAGER] Camera {camera_idx} reconnected (total cameras: {len(self.display_readers)})")

            return True
            
        except Exception as e:
            logger.error(f"Failed to connect display buffer for camera {camera_idx}: {e}")
            return False
            
    def connect_frame_buffer(self, camera_idx: int, frame_buffer_name: str) -> bool:
        """
        Connect to frame buffer for raw frame data.

        Args:
            camera_idx: Camera index
            frame_buffer_name: Name of frame shared memory buffer

        Returns:
            Success status
        """
        try:
            # Connect to existing frame buffer
            frame_shm = mp.shared_memory.SharedMemory(name=frame_buffer_name)
            self.frame_buffers[camera_idx] = frame_shm
            logger.info(f"Connected to frame buffer for camera {camera_idx}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect frame buffer for camera {camera_idx}: {e}")
            return False

    def connect_pose_frame_buffer(self, camera_idx: int, pose_frame_buffer_name: str) -> bool:
        """
        Connect to pose frame buffer for skeleton-overlaid frame data.

        Pose frame buffers (DISABLED/LEGACY) would contain frames with skeleton overlays.
        Currently unused - system uses native resolution frame buffers instead.

        Args:
            camera_idx: Camera index
            pose_frame_buffer_name: Name of pose frame shared memory buffer (e.g., 'yq_pose_frame_2_5409')

        Returns:
            Success status
        """
        try:
            # Connect to existing pose frame buffer
            pose_frame_shm = mp.shared_memory.SharedMemory(name=pose_frame_buffer_name)
            self.pose_frame_buffers[camera_idx] = pose_frame_shm
            logger.info(f"[POSE FRAME] Connected to pose frame buffer for camera {camera_idx}: {pose_frame_buffer_name}")
            return True

        except Exception as e:
            logger.error(f"[POSE FRAME] Failed to connect pose frame buffer for camera {camera_idx}: {e}")
            return False

    def get_display_data(self, camera_idx: int) -> Optional[DisplayData]:
        """
        Get latest display data for a camera (OPTIMIZED).

        OPTIMIZATION: Fetches frame separately from camera buffer using frame_id.
        Display buffer now contains only metadata + face data (no 2.7MB frame copy).

        This is the main method called by schedule_preview() for rendering.
        Optimized for minimal latency.

        Args:
            camera_idx: Camera index

        Returns:
            DisplayData if new data available, None otherwise
        """
        # DIAGNOSTIC DISABLED: Track method calls (kept for compatibility)
        if not hasattr(self, '_get_display_data_calls'):
            self._get_display_data_calls = {}
        if camera_idx not in self._get_display_data_calls:
            self._get_display_data_calls[camera_idx] = 0
        self._get_display_data_calls[camera_idx] += 1

        # DIAGNOSTIC DISABLED: Uncomment to debug method calls
        # if self._get_display_data_calls[camera_idx] % 30 == 0:
        #     logger.info(f"[GET DISPLAY DATA] Camera {camera_idx}: Method called (call #{self._get_display_data_calls[camera_idx]})")

        start_time = time.time()

        if camera_idx not in self.display_readers:
            # DIAGNOSTIC ENABLED: Debug missing display readers
            if self._get_display_data_calls[camera_idx] % 30 == 0:
                logger.error(f"[GET DISPLAY DATA ERROR] Camera {camera_idx}: NOT in display_readers! Keys: {list(self.display_readers.keys())}")
            return None

        try:
            # Read display data atomically (metadata + faces only, no frame)
            display_data = self.display_readers[camera_idx].read_display_data_atomic()

            if display_data:
                # CLEAN ARCHITECTURE: Direct fetch from camera frame buffer (native resolution, no overlays)
                # Single path: Camera Worker → frame_buffer → Display
                if camera_idx in self.frame_buffers:
                    frame_bgr = self._fetch_frame_from_camera_buffer(camera_idx, display_data.frame_id)
                    display_data.frame_bgr = frame_bgr
                else:
                    logger.warning(f"Camera {camera_idx} not in frame_buffers, cannot fetch frame")

                # Track performance
                read_time = (time.time() - start_time) * 1000  # ms
                self.performance_stats['read_times'].append(read_time)

                # Update frame count
                if camera_idx not in self.performance_stats['frame_counts']:
                    self.performance_stats['frame_counts'][camera_idx] = 0
                self.performance_stats['frame_counts'][camera_idx] += 1

                # DIAGNOSTIC DISABLED: Track display read count (kept for compatibility)
                if not hasattr(self, '_display_read_count'):
                    self._display_read_count = {}
                self._display_read_count[camera_idx] = self._display_read_count.get(camera_idx, 0) + 1

                # DIAGNOSTIC DISABLED: Uncomment to debug display read success
                # if self._display_read_count[camera_idx] <= 5 or self._display_read_count[camera_idx] % 100 == 0:
                #     has_frame = display_data.frame_bgr is not None
                #     logger.info(f"[DISPLAY READ SUCCESS] Camera {camera_idx}: "
                #               f"Read frame_id={display_data.frame_id} with {display_data.n_faces} faces, "
                #               f"frame={'present' if has_frame else 'missing'} "
                #               f"(read #{self._display_read_count[camera_idx]})")

                self.last_update_times[camera_idx] = time.time()

            return display_data

        except Exception as e:
            logger.error(f"Error getting display data for camera {camera_idx}: {e}")
            return None

    def _fetch_frame_from_camera_buffer(self, camera_idx: int, frame_id: int) -> Optional[np.ndarray]:
        """
        Fetch frame from camera buffer using LIFO mode (always latest frame).

        LIFO OPTIMIZATION: Instead of trying to match frame_id from landmarks,
        this method scans the ring buffer to find the slot with the highest frame_id
        (most recent frame) and fetches that. This eliminates the 10-16 frame lag
        caused by ring buffer race conditions.

        Tradeoff: Landmarks may be 1-3 frames behind video (acceptable for real-time display).

        Args:
            camera_idx: Camera index
            frame_id: Frame ID from landmarks (IGNORED in LIFO mode, kept for interface compatibility)

        Returns:
            Frame as numpy array or None
        """
        if camera_idx not in self.frame_buffers:
            logger.error(f"[FRAME FETCH] Camera {camera_idx} not in frame_buffers dict")
            return None

        try:
            frame_shm = self.frame_buffers[camera_idx]

            # CRITICAL: Read ACTUAL resolution from buffer header (same pattern as MediaPipe/Pose)
            # This supports mixed-resolution cameras and eliminates timing dependencies
            # Buffer header layout: [write_index(8)][width(4)][height(4)]
            actual_res = np.frombuffer(frame_shm.buf, dtype=np.uint32, count=2, offset=8)
            frame_w, frame_h = int(actual_res[0]), int(actual_res[1])

            # Validate resolution is reasonable
            if frame_w <= 0 or frame_h <= 0 or frame_w > 7680 or frame_h > 4320:
                logger.error(f"Invalid resolution in buffer header for camera {camera_idx}: {frame_w}x{frame_h}")
                return None

            frame_size = frame_w * frame_h * 3

            # Get frame buffer layout from coordinator using ACTUAL resolution
            layout = self.coordinator.get_buffer_layout(camera_idx, (frame_w, frame_h))

            if not layout:
                logger.error(f"[FRAME FETCH] No layout returned from coordinator for camera {camera_idx}")
                return None

            # Get detection frame offsets and metadata offsets
            detection_frame_offsets = layout.get('detection_frame_offsets', [])
            detection_metadata_offsets = layout.get('detection_metadata_offsets', [])

            if not detection_frame_offsets or not detection_metadata_offsets:
                logger.error(f"[FRAME FETCH] No detection offsets in layout for camera {camera_idx}. "
                           f"Layout keys: {list(layout.keys()) if layout else 'None'}")
                return None

            ring_buffer_size = len(detection_frame_offsets)
            frame_shm = self.frame_buffers[camera_idx]

            # LIFO MODE: Scan all slots to find the one with highest frame_id (most recent)
            latest_frame_id = -1
            latest_slot = 0

            # DIAGNOSTIC: Track whether we've done the first scan logging
            if not hasattr(self, '_lifo_scan_logged'):
                self._lifo_scan_logged = {}

            # Only do diagnostic logging on the very first scan for this camera
            should_log_diagnostic = camera_idx not in self._lifo_scan_logged

            if should_log_diagnostic:
                # Mark that we've done the diagnostic for this camera (never delete this flag!)
                self._lifo_scan_logged[camera_idx] = True

            for slot in range(ring_buffer_size):
                metadata_offset = detection_metadata_offsets[slot]

                # Read metadata (32 bytes = 4 int64 values, matching camera_worker_enhanced.py line 694)
                # metadata[0] = frame_id, metadata[1] = timestamp, metadata[2] = detection_flag, metadata[3] = ready_flag
                metadata_view = np.ndarray(
                    (4,), dtype=np.int64,
                    buffer=frame_shm.buf[metadata_offset:metadata_offset + 32]
                )

                slot_frame_id = int(metadata_view[0])


                # Skip invalid negative frame IDs (uninitialized slots)
                if slot_frame_id < 0:
                    continue

                # Update latest if this slot has newer frame
                if slot_frame_id > latest_frame_id:
                    latest_frame_id = slot_frame_id
                    latest_slot = slot


            # Use the slot with the latest frame
            frame_position = latest_slot
            frame_offset = detection_frame_offsets[frame_position]

            # Validate bounds before reading
            if frame_offset + frame_size > frame_shm.size:
                logger.error(f"Frame read would exceed buffer: offset={frame_offset}, size={frame_size}, buffer={frame_shm.size}")
                return None

            # Log exact parameters before numpy array creation
            logger.debug(f"[FRAME FETCH] Creating numpy array: offset={frame_offset}, size={frame_size}, "
                        f"shape=({frame_h}, {frame_w}, 3), buffer_size={frame_shm.size}")

            # CRITICAL FIX: Use np.frombuffer() instead of np.ndarray() with buffer slice
            # This handles memoryview slices more reliably
            try:
                # First create a flat array from the buffer slice
                frame_flat = np.frombuffer(
                    frame_shm.buf,
                    dtype=np.uint8,
                    count=frame_size,
                    offset=frame_offset
                )

                # Then reshape to the correct dimensions
                frame_data = frame_flat.reshape((frame_h, frame_w, 3))

                logger.debug(f"[FRAME FETCH] Successfully created frame array with shape {frame_data.shape}")

            except Exception as e:
                logger.error(f"[FRAME FETCH] Failed to create numpy array: {e}")
                logger.error(f"[FRAME FETCH] Parameters: offset={frame_offset}, size={frame_size}, "
                           f"shape=({frame_h}, {frame_w}, 3), buffer_size={frame_shm.size}")
                import traceback
                traceback.print_exc()
                return None

            # DIAGNOSTIC DISABLED: Track LIFO counter (kept for compatibility)
            if not hasattr(self, '_lifo_log_counter'):
                self._lifo_log_counter = {}
            if camera_idx not in self._lifo_log_counter:
                self._lifo_log_counter[camera_idx] = 0

            self._lifo_log_counter[camera_idx] += 1
            # DIAGNOSTIC DISABLED: Uncomment to debug LIFO frame selection
            # if self._lifo_log_counter[camera_idx] % 100 == 0:
            #     frame_desync = abs(latest_frame_id - frame_id)
            #     logger.info(f"[LIFO] Camera {camera_idx}: Fetched frame {latest_frame_id} "
            #               f"(landmark frame_id={frame_id}, desync={frame_desync} frames)")

            # DIAGNOSTIC DISABLED: Track frame fetch count (kept for compatibility)
            if not hasattr(self, '_frame_fetch_success_count'):
                self._frame_fetch_success_count = {}
            if camera_idx not in self._frame_fetch_success_count:
                self._frame_fetch_success_count[camera_idx] = 0
            self._frame_fetch_success_count[camera_idx] += 1
            # DIAGNOSTIC DISABLED: Uncomment to debug frame fetch success
            # if self._frame_fetch_success_count[camera_idx] <= 5 or self._frame_fetch_success_count[camera_idx] % 100 == 0:
            #     logger.info(f"[FRAME FETCH SUCCESS] Camera {camera_idx}: Fetched frame {latest_frame_id} "
            #               f"from slot {latest_slot}, shape={frame_data.shape}")

            # Return copy to avoid shared memory issues
            return frame_data.copy()

        except Exception as e:
            logger.error(f"[FRAME FETCH ERROR] Error fetching frame from camera buffer {camera_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _fetch_frame_from_pose_frame_buffer(self, camera_idx: int, frame_id: int) -> Optional[np.ndarray]:
        """
        LEGACY/DISABLED: Fetch skeleton-overlaid frame from pose frame buffer.

        This method is not currently used as pose frame buffers are disabled.
        System now uses native resolution frame buffers without downsampling.

        Args:
            camera_idx: Camera index
            frame_id: Frame ID (IGNORED in LIFO mode, kept for interface compatibility)

        Returns:
            BGR frame with skeleton overlays, or None if unavailable (currently always None)
        """
        if camera_idx not in self.pose_frame_buffers:
            logger.debug(f"[POSE FRAME FETCH] Camera {camera_idx} not in pose_frame_buffers dict")
            return None

        try:
            pose_frame_shm = self.pose_frame_buffers[camera_idx]

            # LEGACY: Hardcoded 640×480 resolution (this code path is disabled)
            pose_w, pose_h = 640, 480
            frame_size = pose_w * pose_h * 3

            # Pose frame buffer layout (from camera_worker_enhanced.py line ~677):
            # [ring_buffer_size(8)][frame_0][frame_1]...[frame_N][metadata_0][metadata_1]...[metadata_N]
            # Ring buffer size: 16 slots (default for pose frames)

            # Read ring buffer size from header
            ring_size_bytes = bytes(pose_frame_shm.buf[0:8])
            ring_buffer_size = int.from_bytes(ring_size_bytes, byteorder='little')

            if ring_buffer_size <= 0 or ring_buffer_size > 32:
                logger.error(f"[POSE FRAME FETCH] Invalid ring_buffer_size={ring_buffer_size} for camera {camera_idx}")
                return None

            # Calculate offsets
            header_size = 8  # ring_buffer_size
            frame_offset_base = header_size
            metadata_offset_base = frame_offset_base + (ring_buffer_size * frame_size)
            metadata_size = 32  # [frame_id(8)][timestamp(8)][reserved(8)][reserved(8)]

            # LIFO MODE: Scan all slots to find the one with highest frame_id (most recent)
            latest_frame_id = -1
            latest_slot = 0

            for slot in range(ring_buffer_size):
                metadata_offset = metadata_offset_base + (slot * metadata_size)

                # Read metadata
                metadata_bytes = bytes(pose_frame_shm.buf[metadata_offset:metadata_offset + metadata_size])
                slot_frame_id = int.from_bytes(metadata_bytes[0:8], byteorder='little', signed=True)

                # Skip invalid/uninitialized slots
                if slot_frame_id < 0:
                    continue

                # Update latest if this slot has newer frame
                if slot_frame_id > latest_frame_id:
                    latest_frame_id = slot_frame_id
                    latest_slot = slot

            # If no valid frames found, return None
            if latest_frame_id < 0:
                logger.debug(f"[POSE FRAME FETCH] No valid frames found in pose frame buffer for camera {camera_idx}")
                return None

            # Read frame from latest slot
            frame_offset = frame_offset_base + (latest_slot * frame_size)

            # Validate bounds
            if frame_offset + frame_size > pose_frame_shm.size:
                logger.error(f"[POSE FRAME FETCH] Frame read would exceed buffer: "
                           f"offset={frame_offset}, size={frame_size}, buffer={pose_frame_shm.size}")
                return None

            # Read frame using np.frombuffer (zero-copy)
            frame_flat = np.frombuffer(
                pose_frame_shm.buf,
                dtype=np.uint8,
                count=frame_size,
                offset=frame_offset
            )

            # Reshape to image dimensions
            frame_data = frame_flat.reshape((pose_h, pose_w, 3))

            # Log first successful fetch per camera
            if not hasattr(self, '_pose_frame_fetch_logged'):
                self._pose_frame_fetch_logged = set()
            if camera_idx not in self._pose_frame_fetch_logged:
                logger.info(f"[POSE FRAME FETCH] ✅ Camera {camera_idx}: First successful fetch "
                          f"(frame_id={latest_frame_id}, slot={latest_slot}, shape={frame_data.shape})")
                self._pose_frame_fetch_logged.add(camera_idx)

            # Return copy to avoid shared memory issues
            return frame_data.copy()

        except Exception as e:
            logger.error(f"[POSE FRAME FETCH ERROR] Error fetching pose frame for camera {camera_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_frame_for_display(self, camera_idx: int, frame_id: int, 
                             resolution: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Get frame data for display.
        
        Args:
            camera_idx: Camera index
            frame_id: Frame ID to retrieve
            resolution: Frame resolution (width, height)
            
        Returns:
            Frame as numpy array or None
        """
        if camera_idx not in self.frame_buffers:
            return None
            
        try:
            # Calculate frame location in GUI buffer
            # This assumes the GUI buffer layout from BufferCoordinator
            frame_w, frame_h = resolution
            frame_size = frame_w * frame_h * 3
            
            # TODO: Get actual GUI buffer layout from BufferCoordinator
            # For now, assume simple layout
            gui_buffer_offset = 0  # This needs proper calculation
            
            # Create numpy view of frame data
            frame_flat = np.ndarray(
                (frame_size,), 
                dtype=np.uint8,
                buffer=self.frame_buffers[camera_idx].buf[gui_buffer_offset:gui_buffer_offset + frame_size]
            )
            
            # Reshape to image
            frame_bgr = frame_flat.reshape((frame_h, frame_w, 3))
            
            return frame_bgr.copy()  # Return copy for safety
            
        except Exception as e:
            logger.error(f"Error getting frame for camera {camera_idx}: {e}")
            return None
            
    def get_buffer(self, camera_idx: int):
        """
        Get the display buffer reader for a camera.
        
        Args:
            camera_idx: Camera index
            
        Returns:
            DisplayBufferReader instance or None if not connected
        """
        return self.display_readers.get(camera_idx)
    
    def get_buffer_name(self, camera_idx: int) -> Optional[str]:
        """
        Get the display buffer name for a camera.
        
        Args:
            camera_idx: Camera index
            
        Returns:
            Buffer name or None if not connected
        """
        reader = self.display_readers.get(camera_idx)
        return reader.buffer_name if reader else None
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        current_time = time.time()
        elapsed = current_time - self.performance_stats['last_report']
        
        if elapsed > 0 and self.performance_stats['read_times']:
            avg_read_time = np.mean(self.performance_stats['read_times'])
            max_read_time = np.max(self.performance_stats['read_times'])
            
            # Calculate FPS per camera
            camera_fps = {}
            for cam_idx, count in self.performance_stats['frame_counts'].items():
                camera_fps[cam_idx] = count / elapsed
                
            stats = {
                'avg_read_time_ms': avg_read_time,
                'max_read_time_ms': max_read_time,
                'camera_fps': camera_fps,
                'total_frames': sum(self.performance_stats['frame_counts'].values())
            }
            
            # Reset counters
            self.performance_stats['read_times'] = []
            self.performance_stats['frame_counts'] = {}
            self.performance_stats['last_report'] = current_time
            
            return stats
        else:
            return {
                'avg_read_time_ms': 0,
                'max_read_time_ms': 0,
                'camera_fps': {},
                'total_frames': 0
            }
            
    def flush_display_buffer(self, camera_idx: int):
        """
        Flush display buffer for a camera to discard old frames.

        Called when camera achieves realtime_ready status to ensure that
        only fresh frames (post-warmup) are displayed, eliminating startup lag.

        Implementation: Scans ring buffer to find current highest frame_id,
        then sets last_frame_id to that value PLUS safety margin. This causes
        all buffered frames to be skipped, and only NEW frames (written after
        this flush) will be returned by subsequent reads.

        Args:
            camera_idx: Camera index to flush
        """
        # DIAGNOSTIC: High-visibility logging for flush execution
        logger.warning(f"[DISPLAY BUFFER FLUSH] ⚠️⚠️⚠️ FLUSH CALLED for camera {camera_idx}")
        logger.warning(f"[DISPLAY BUFFER FLUSH] display_readers keys: {list(self.display_readers.keys())}")

        if camera_idx not in self.display_readers:
            logger.warning(f"[DISPLAY BUFFER FLUSH] Camera {camera_idx} not in display_readers, cannot flush")
            return

        try:
            reader = self.display_readers[camera_idx]
            old_last_frame_id = reader.last_frame_id

            # Scan ring buffer to find current highest frame_id
            highest_frame_id = -1
            valid_frame_count = 0  # FIX 3: Count valid frames for adaptive margin
            for slot in range(reader.RING_SIZE):
                slot_offset = reader.WRITE_INDEX_SIZE + (slot * reader.SLOT_SIZE)
                metadata_offset = slot_offset

                # Read frame_id from metadata
                metadata_bytes = reader.shm.buf[metadata_offset:metadata_offset + reader.FRAME_METADATA_SIZE]
                frame_id = struct.unpack_from('Q', metadata_bytes, 0)[0]

                if frame_id > highest_frame_id:
                    highest_frame_id = frame_id

                # FIX 3: Count valid frames (frame_id > 0 means real frame, not flushed slot)
                if frame_id > 0:
                    valid_frame_count += 1

            # FLUSH: Reset to accept ANY next frame
            # During startup, display buffer contains OLD frames (from warmup period)
            # But GUI worker has flushed to FRESH frames (from camera buffer)
            # Solution: Reset last_frame_id = -1 to accept whatever worker writes next
            # Worker will write fresh frames immediately (no gap, no lag!)
            new_last_frame_id = -1

            logger.warning(f"[DISPLAY BUFFER FLUSH] ⚠️ Camera {camera_idx} BEFORE flush: "
                         f"last_frame_id={old_last_frame_id}, highest_found={highest_frame_id} (OLD backlog), "
                         f"valid_frames={valid_frame_count}")
            logger.warning(f"[DISPLAY BUFFER FLUSH] 🧹 Resetting last_frame_id to -1 (accept any next frame)")
            logger.warning(f"[DISPLAY BUFFER FLUSH] ⏳ Worker will write FRESH frames immediately")

            # Reset last_frame_id to accept any frame (worker provides fresh frames)
            reader.last_frame_id = new_last_frame_id

            logger.warning(f"[DISPLAY BUFFER FLUSH] ✅ Camera {camera_idx}: Flush complete - "
                         f"ready to accept fresh frames from worker")

        except Exception as e:
            logger.error(f"[DISPLAY BUFFER FLUSH] Error flushing buffer for camera {camera_idx}: {e}")
            import traceback
            traceback.print_exc()

    def cleanup(self):
        """Clean up all connections."""
        # Clean up display readers
        for reader in self.display_readers.values():
            reader.cleanup()
        self.display_readers.clear()

        # Clean up frame buffer connections
        for frame_shm in self.frame_buffers.values():
            try:
                frame_shm.close()
            except:
                pass
        self.frame_buffers.clear()

        # Clean up pose frame buffer connections
        for pose_frame_shm in self.pose_frame_buffers.values():
            try:
                pose_frame_shm.close()
            except:
                pass
        self.pose_frame_buffers.clear()

        logger.info("DisplayBufferManager cleaned up")