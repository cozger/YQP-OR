"""
GUI Buffer Manager - Dedicated buffer management for GUI display data
Provides isolated buffer management for GUI display to decouple from processing pipeline
"""

import numpy as np
import time
import logging
import threading
from multiprocessing import shared_memory, Value
from typing import Dict, List, Optional, Tuple, Any
from core.buffer_management.coordinator import BufferCoordinator

logger = logging.getLogger('GUIBufferManager')


class GUIBufferManager:
    """
    Dedicated buffer manager for GUI display data.
    Manages separate ring buffers optimized for display performance.
    """
    
    def __init__(self, buffer_coordinator: BufferCoordinator):
        self.coordinator = buffer_coordinator
        self.gui_buffers = {}  # camera_index -> GUIBuffer
        self.last_sequence_numbers = {}  # track ordering for each camera
        self.update_lock = threading.Lock()
        
        # Performance tracking
        self.update_counts = {}
        self.last_update_times = {}
        
        logger.info("GUIBufferManager initialized")
    
    def register_camera(self, camera_index: int, gui_buffer_name: str = None) -> bool:
        """Register a camera for GUI buffer management."""
        try:
            # Use provided buffer name or try to get from coordinator
            if gui_buffer_name is None:
                gui_buffer_names = self.coordinator.get_gui_buffer_names()
                if camera_index not in gui_buffer_names:
                    logger.error(f"No GUI buffer found for camera {camera_index}")
                    return False
                gui_buffer_name = gui_buffer_names[camera_index]

            # Get actual camera resolution from coordinator
            resolution = self.coordinator.get_camera_resolution(camera_index)

            gui_buffer = GUIBuffer(
                camera_index=camera_index,
                buffer_name=gui_buffer_name,
                buffer_sizes=self.coordinator.get_buffer_sizes(),
                resolution=resolution
            )
            
            self.gui_buffers[camera_index] = gui_buffer
            self.last_sequence_numbers[camera_index] = 0
            self.update_counts[camera_index] = 0
            self.last_update_times[camera_index] = time.time()
            
            logger.info(f"Registered camera {camera_index} for GUI buffer management")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register camera {camera_index}: {e}")
            return False
    
    def update_gui_data(self, camera_index: int, frame_data: np.ndarray, 
                       landmark_data: Dict, sequence_num: int) -> bool:
        """
        Thread-safe GUI data update with sequence validation.
        
        Args:
            camera_index: Camera identifier
            frame_data: Frame image data (numpy array)
            landmark_data: Structured landmark data with metadata
            sequence_num: Sequence number for ordering validation
            
        Returns:
            True if update was successful, False otherwise
        """
        if camera_index not in self.gui_buffers:
            logger.warning(f"Camera {camera_index} not registered")
            return False
        
        # Check sequence ordering to prevent stale data
        if sequence_num <= self.last_sequence_numbers[camera_index]:
            logger.debug(f"Ignoring stale data for camera {camera_index}: "
                        f"seq {sequence_num} <= {self.last_sequence_numbers[camera_index]}")
            return False
        
        try:
            with self.update_lock:
                gui_buffer = self.gui_buffers[camera_index]
                success = gui_buffer.write_display_data(frame_data, landmark_data, sequence_num)
                
                if success:
                    self.last_sequence_numbers[camera_index] = sequence_num
                    self.update_counts[camera_index] += 1
                    self.last_update_times[camera_index] = time.time()
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to update GUI data for camera {camera_index}: {e}")
            return False
    
    def get_latest_display_data(self, camera_index: int) -> Optional[Dict]:
        """
        Get latest data for GUI display.
        
        Returns:
            Dictionary with frame, landmarks, and metadata or None if no data
        """
        if camera_index not in self.gui_buffers:
            return None
        
        try:
            gui_buffer = self.gui_buffers[camera_index]
            return gui_buffer.read_display_data()
            
        except Exception as e:
            logger.error(f"Failed to get display data for camera {camera_index}: {e}")
            return None
    
    def get_buffer_status(self, camera_index: int) -> Dict[str, Any]:
        """Get buffer status and performance metrics."""
        if camera_index not in self.gui_buffers:
            return {'status': 'not_registered'}
        
        current_time = time.time()
        last_update = self.last_update_times.get(camera_index, 0)
        update_count = self.update_counts.get(camera_index, 0)
        
        return {
            'status': 'active',
            'last_update_age_ms': (current_time - last_update) * 1000,
            'total_updates': update_count,
            'last_sequence': self.last_sequence_numbers.get(camera_index, 0),
            'buffer_health': self.gui_buffers[camera_index].get_health_status()
        }
    
    def cleanup_camera(self, camera_index: int):
        """Clean up resources for a specific camera."""
        if camera_index in self.gui_buffers:
            try:
                self.gui_buffers[camera_index].cleanup()
                del self.gui_buffers[camera_index]
                del self.last_sequence_numbers[camera_index]
                del self.update_counts[camera_index]
                del self.last_update_times[camera_index]
                logger.info(f"Cleaned up GUI buffer for camera {camera_index}")
            except Exception as e:
                logger.error(f"Error cleaning up camera {camera_index}: {e}")
    
    def cleanup_all(self):
        """Clean up all GUI buffers."""
        for camera_index in list(self.gui_buffers.keys()):
            self.cleanup_camera(camera_index)
        logger.info("All GUI buffers cleaned up")


class GUIBuffer:
    """
    Individual GUI buffer for a single camera.
    Manages dedicated ring buffer for display data.
    """
    
    def __init__(self, camera_index: int, buffer_name: str, buffer_sizes: Dict[str, int],
                 resolution: Tuple[int, int] = (1920, 1080)):
        self.camera_index = camera_index
        self.buffer_name = buffer_name
        self.buffer_sizes = buffer_sizes
        self.resolution = resolution  # Store actual camera resolution for buffer calculations

        # Connect to shared memory
        try:
            self.shm = shared_memory.SharedMemory(name=buffer_name)
            logger.info(f"Connected to GUI buffer for camera {camera_index}: {buffer_name} (resolution: {resolution[0]}x{resolution[1]})")
        except Exception as e:
            logger.error(f"Failed to connect to GUI buffer {buffer_name}: {e}")
            raise

        # Ring buffer management
        self.ring_size = buffer_sizes['gui_buffer_size']
        self.ring_mask = self.ring_size - 1  # Power of 2 optimization

        # Atomic write index
        self.write_idx = Value('L', 0)

        # Performance tracking
        self.write_count = 0
        self.last_write_time = 0
        self.errors = 0
    
    def write_display_data(self, frame_data: np.ndarray, landmark_data: Dict, sequence_num: int) -> bool:
        """
        Write display data to ring buffer.
        
        Args:
            frame_data: Frame image data
            landmark_data: Landmark data with metadata
            sequence_num: Sequence number for ordering
            
        Returns:
            True if write successful
        """
        try:
            # Calculate buffer slot
            with self.write_idx.get_lock():
                slot = self.write_idx.value & self.ring_mask
                self.write_idx.value += 1
            
            # Write frame data to shared memory
            frame_size = frame_data.size * frame_data.itemsize
            frame_offset = slot * frame_size
            
            # Ensure we don't exceed buffer bounds
            if frame_offset + frame_size > self.shm.size:
                logger.error(f"Frame data too large for buffer slot {slot}")
                self.errors += 1
                return False
            
            # Copy frame data
            frame_view = np.frombuffer(
                self.shm.buf,
                dtype=frame_data.dtype,
                count=frame_data.size,
                offset=frame_offset
            )
            frame_view[:] = frame_data.flat
            
            # Write metadata (sequence number, timestamp, landmark info)
            metadata_offset = self.ring_size * frame_size + slot * 64  # 64 bytes per metadata slot
            metadata_view = np.frombuffer(
                self.shm.buf,
                dtype=np.int32,
                count=16,
                offset=metadata_offset
            )
            
            # Pack metadata: [sequence, timestamp_hi, timestamp_lo, n_faces, ...landmark_info...]
            timestamp = int(time.time() * 1000000)  # microseconds
            metadata_view[0] = sequence_num
            metadata_view[1] = timestamp >> 32
            metadata_view[2] = timestamp & 0xFFFFFFFF
            metadata_view[3] = landmark_data.get('n_faces', 0)
            
            # Pack first few landmark track IDs for quick access
            if 'roi_metadata' in landmark_data and landmark_data['roi_metadata']:
                roi_metadata = landmark_data['roi_metadata']
                for i in range(min(10, len(roi_metadata))):  # Pack up to 10 track IDs
                    if i + 4 < 16:
                        metadata_view[i + 4] = roi_metadata[i].get('track_id', -1)
            
            self.write_count += 1
            self.last_write_time = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write display data to camera {self.camera_index}: {e}")
            self.errors += 1
            return False
    
    def read_display_data(self) -> Optional[Dict]:
        """
        Read latest display data from ring buffer.
        
        Returns:
            Dictionary with frame data and metadata or None
        """
        try:
            # Get latest written slot
            current_write_idx = self.write_idx.value
            if current_write_idx == 0:
                return None
            
            # Read from most recent slot
            slot = (current_write_idx - 1) & self.ring_mask

            # Calculate frame size using actual camera resolution
            frame_size = self.resolution[0] * self.resolution[1] * 3
            metadata_offset = self.ring_size * frame_size + slot * 64
            
            # Check if metadata offset is within bounds
            if metadata_offset + 64 > self.shm.size:
                logger.error(f"Metadata offset {metadata_offset} exceeds buffer size {self.shm.size}")
                return None
            
            metadata_view = np.frombuffer(
                self.shm.buf,
                dtype=np.int32,
                count=16,
                offset=metadata_offset
            )
            
            sequence_num = metadata_view[0]
            timestamp = (metadata_view[1] << 32) | metadata_view[2]
            n_faces = metadata_view[3]
            
            # Read frame data using actual resolution
            frame_offset = slot * frame_size
            if frame_offset + frame_size > self.shm.size:
                logger.error(f"Frame offset {frame_offset} exceeds buffer size {self.shm.size}")
                return None
            
            # For now, return metadata only (frame reading would need actual dimensions)
            return {
                'sequence_num': sequence_num,
                'timestamp': timestamp,
                'n_faces': n_faces,
                'camera_index': self.camera_index,
                'slot': slot
            }
            
        except Exception as e:
            logger.error(f"Failed to read display data from camera {self.camera_index}: {e}")
            return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get buffer health metrics."""
        current_time = time.time()
        return {
            'write_count': self.write_count,
            'last_write_age_ms': (current_time - self.last_write_time) * 1000 if self.last_write_time > 0 else -1,
            'errors': self.errors,
            'current_slot': self.write_idx.value & self.ring_mask,
            'buffer_size_mb': self.shm.size / (1024 * 1024)
        }
    
    def cleanup(self):
        """Clean up buffer resources."""
        try:
            if self.shm:
                self.shm.close()
                logger.info(f"GUI buffer for camera {self.camera_index} closed")
        except Exception as e:
            logger.error(f"Error closing GUI buffer for camera {self.camera_index}: {e}")