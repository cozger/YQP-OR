"""
ProcessBufferManager - Child process interface to centralized buffer system.

This module provides efficient buffer discovery and access for child processes
in the YouQuantiPy architecture. It replaces manual buffer scanning with
direct index-based access for O(1) performance.

Key Features:
- Systematic buffer discovery with multiple fallback strategies
- Direct write-index-based buffer access (no scanning)
- Consistent buffer configuration across all processes
- CommandBuffer integration for efficient IPC
"""

import os
import re
import time
import json
import logging
import numpy as np
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass

from .command_protocol import CommandProtocol
from .sharedbuffer import CommandBuffer, AtomicRingBuffer

logger = logging.getLogger(__name__)


@dataclass
class BufferLayout:
    """
    LEGACY: Simple metadata layout for index lookups.

    NOTE: For comprehensive buffer layouts with all offsets and sizes, use the new
    dataclasses in layouts.py (FrameBufferLayout, ResultsBufferLayout, etc.) which
    provide complete buffer structure information.

    This class is kept for metadata parsing in child processes where only
    metadata field indices are needed (ready_flag_index, frame_id_index, etc.).
    """
    metadata_size: int
    metadata_dtype: type
    metadata_count: int
    ready_flag_index: int
    frame_id_index: int
    write_index_offset: int = 0  # Always at start


# Standard buffer layouts for metadata parsing (not comprehensive layouts)
# For full buffer layouts, use BufferCoordinator.get_layout() or import from layouts.py
BUFFER_LAYOUTS = {
    'frame': BufferLayout(64, np.int32, 16, 3, 0),  # Frame buffer: raw camera frames
    'roi': BufferLayout(1024, np.float32, 256, 253, 0),  # FIXED: Updated for 5-point alignment metadata
    'results': BufferLayout(128, np.float32, 32, 15, 0),  # Results buffer: landmarks + blendshapes + bboxes
    'gui': BufferLayout(64, np.int32, 16, 3, 0),
    'pose': BufferLayout(64, np.float32, 16, 15, 0),  # Pose buffer: 64 bytes metadata, float32 dtype
}


class ProcessBufferManager:
    """
    Child process interface to centralized buffer system.
    Discovers and connects to buffers created by BufferCoordinator.
    Provides O(1) direct access instead of O(n) scanning.
    """
    
    def __init__(self, process_type: str, camera_indices: List[int], config: Dict):
        """
        Initialize ProcessBufferManager.
        
        Args:
            process_type: Type of process ('face_recognition', 'detection', 'landmark')
            camera_indices: List of camera indices this process handles
            config: Configuration dictionary from main process
        """
        self.process_type = process_type
        self.camera_indices = camera_indices
        self.config = config
        
        # Buffer connections
        self.roi_buffers = {}  # camera_idx -> buffer info
        self.command_buffers = {}  # camera_idx -> CommandBuffer
        self.frame_buffers = {}  # camera_idx -> frame buffer info
        self.detection_buffers = {}  # camera_idx -> detection buffer info (SCRFD detections)
        self.gui_buffers = {}  # camera_idx -> GUI buffer info
        self.pose_buffers = {}  # camera_idx -> pose buffer info
        self.results_buffers = {}  # camera_idx -> results buffer info (landmarks + blendshapes + bboxes)
        
        # Track numpy views for proper cleanup - CRITICAL for preventing BufferError
        self.active_numpy_views = []  # List of numpy arrays that need cleanup
        
        # Configuration from coordinator
        self.buffer_config = {}
        self.coordinator_pid = None
        self.coordinator_info = None
        
        # Performance metrics
        self.discovery_time = 0
        self.access_stats = {
            'direct_reads': 0,
            'fallback_scans': 0,
            'read_errors': 0
        }
        
    def discover_and_connect(self, timeout_seconds: float = 10.0) -> bool:
        """
        Discover BufferCoordinator and connect to all required buffers.
        Uses systematic discovery rather than manual PID scanning.

        Args:
            timeout_seconds: Maximum time to wait for discovery

        Returns:
            True if successfully connected to all required buffers
        """
        start_time = time.time()
        last_error = None
        attempt = 0

        logger.info(f"   Timeout: {timeout_seconds}s")

        # Try discovery with timeout
        while time.time() - start_time < timeout_seconds:
            attempt += 1

            try:
                # Attempt discovery
                discovery_result = self._discover_coordinator_buffers()

                if discovery_result:
                    self.coordinator_info = discovery_result

                    # Connect to discovered buffers
                    if self._connect_to_buffers(discovery_result):
                        self.discovery_time = time.time() - start_time
                        logger.info(f"✅ Buffer discovery successful in {self.discovery_time:.2f}s")
                        logger.info(f"   Coordinator PID: {self.coordinator_pid}")
                        logger.info(f"   Connected buffers: Frame={len(self.frame_buffers)}, "
                                   f"Detection={len(self.detection_buffers)}, Results={len(self.results_buffers)}")
                        return True

                # Log progress every 2 seconds
                if attempt % 20 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"⏳ Still searching for buffers... ({elapsed:.1f}s elapsed)")

                # Wait before retry
                time.sleep(0.1)

            except Exception as e:
                last_error = e
                logger.debug(f"Discovery attempt {attempt} failed: {e}")
                time.sleep(0.5)

        # Discovery failed - provide detailed diagnostics
        elapsed = time.time() - start_time
        logger.error(f"❌ Buffer discovery failed after {elapsed:.1f}s ({attempt} attempts)")
        if last_error:
            logger.error(f"   Last error: {last_error}")

        # Log comprehensive diagnostics
        self._log_discovery_diagnostics()

        return False
    
    def _discover_coordinator_buffers(self) -> Optional[Dict[str, Any]]:
        """
        Multi-strategy buffer discovery algorithm.
        
        Returns:
            Dictionary with buffer information or None if not found
        """
        strategies = [
            self._discover_via_pid_file,      # Primary strategy
            self._discover_via_recent_buffers # Fallback only
        ]
        
        for strategy in strategies:
            try:
                result = strategy()
                if result and self._validate_buffer_set(result):
                    logger.debug(f"[ProcessBufferManager] Discovery succeeded with {strategy.__name__}")
                    return result
            except Exception as e:
                logger.debug(f"[ProcessBufferManager] Discovery strategy {strategy.__name__} failed: {e}")
                continue
        
        return None
    
    def _discover_via_pid_file(self) -> Optional[Dict[str, Any]]:
        """Look for coordinator PID file with buffer information."""
        import tempfile
        temp_dir = tempfile.gettempdir()
        home_dir = os.path.expanduser("~")
        
        pid_file_locations = [
            os.path.join(temp_dir, "youquantipy_coordinator.pid"),
            os.path.join(home_dir, ".youquantipy", "coordinator.pid"),
            os.path.join(os.getcwd(), "youquantipy_coordinator.pid")  # Fallback
        ]
        
        for pid_file_path in pid_file_locations:
            try:
                if os.path.exists(pid_file_path):
                    with open(pid_file_path, 'r') as f:
                        coordinator_info = json.load(f)
                    
                    # Validate coordinator info
                    required_keys = ['pid', 'timestamp', 'camera_count', 'buffer_config']
                    if all(key in coordinator_info for key in required_keys):
                        # Check if process is still running
                        try:
                            pid = coordinator_info['pid']
                            os.kill(pid, 0)  # Send null signal to check if process exists
                            logger.info(f"[ProcessBufferManager] Found coordinator info at {pid_file_path}")
                            return coordinator_info
                        except OSError:
                            logger.warning(f"[ProcessBufferManager] Coordinator PID {pid} not running, skipping {pid_file_path}")
                            continue
                    else:
                        logger.warning(f"[ProcessBufferManager] Invalid coordinator info format in {pid_file_path}")
                        
            except Exception as e:
                logger.debug(f"[ProcessBufferManager] Failed to read {pid_file_path}: {e}")
                continue
        
        logger.debug("[ProcessBufferManager] No valid coordinator PID file found")
        return None
    
    def _discover_via_recent_buffers(self) -> Optional[Dict[str, Any]]:
        """Find most recently created buffer set by timestamp."""
        logger.info("[ProcessBufferManager] Trying discovery via recent buffers...")
        if not os.path.exists('/dev/shm'):
            logger.info("[ProcessBufferManager] /dev/shm does not exist")
            return None

        # Group buffers by PID only (not by modification time)
        # Store (pid, max_mtime, buffer_list) for sorting
        buffer_groups = defaultdict(lambda: {'buffers': [], 'max_mtime': 0})

        for filename in os.listdir('/dev/shm'):
            if filename.startswith('yq_'):
                try:
                    filepath = f'/dev/shm/{filename}'
                    stat_info = os.stat(filepath)

                    # Extract PID from filename
                    match = re.search(r'_(\d+)$', filename)
                    if match:
                        pid = match.group(1)
                        buffer_groups[pid]['buffers'].append(filename)
                        # Track most recent modification time for this PID's buffers
                        buffer_groups[pid]['max_mtime'] = max(buffer_groups[pid]['max_mtime'], stat_info.st_mtime)
                except:
                    continue

        # Sort by most recent buffer modification time (most recent first)
        sorted_groups = sorted(
            [(pid, info['max_mtime'], info['buffers']) for pid, info in buffer_groups.items()],
            key=lambda x: x[1],
            reverse=True
        )

        logger.info(f"[ProcessBufferManager] Found {len(sorted_groups)} buffer groups in /dev/shm")

        for pid, mtime, buffer_list in sorted_groups:
            logger.info(f"[ProcessBufferManager] Checking buffer set with PID {pid}: {len(buffer_list)} buffers")
            logger.info(f"[ProcessBufferManager] Buffer list: {buffer_list[:5]}...")  # Show first 5
            is_complete = self._is_complete_buffer_set(buffer_list)
            logger.info(f"[ProcessBufferManager] Is complete buffer set: {is_complete}")
            if is_complete:
                logger.info(f"[ProcessBufferManager] Found complete buffer set with PID {pid}")
                built_info = self._build_buffer_info_from_list(buffer_list, pid)
                logger.info(f"[ProcessBufferManager] Built buffer info: {built_info.get('buffers', {})}")
                return built_info

        logger.info(f"[ProcessBufferManager] No complete buffer sets found among {len(sorted_groups)} groups")
        return None
    
    def _validate_coordinator_info(self, info: Dict) -> bool:
        """Validate coordinator info structure."""
        required_fields = ['pid', 'timestamp', 'camera_count', 'buffer_config']
        return all(field in info for field in required_fields)
    
    def _validate_buffer_set(self, buffer_info: Dict) -> bool:
        """
        Validate that we have all required buffers for our cameras.
        Supports both MediaPipe native (ROI buffers) and SCRFD (detection + results buffers) pipelines.
        """
        buffers = buffer_info.get('buffers', {})

        for cam_idx in self.camera_indices:
            # Check for EITHER:
            # 1. ROI buffers (MediaPipe native pipeline)
            # 2. Detection + Results buffers (SCRFD pipeline)
            roi_key = f'roi_{cam_idx}'
            detection_key = f'detection_{cam_idx}'
            results_key = f'results_{cam_idx}'

            has_roi = roi_key in buffers
            has_scrfd = (detection_key in buffers and results_key in buffers)

            if not (has_roi or has_scrfd):
                logger.debug(f"[ProcessBufferManager] Camera {cam_idx} missing required buffers: "
                           f"roi={has_roi}, detection+results={has_scrfd}")
                return False

        return True
    
    def _is_complete_buffer_set(self, buffer_list: List[str]) -> bool:
        """
        Check if buffer list contains a complete set for at least one camera.
        Supports both MediaPipe native and SCRFD pipelines.
        """
        # Look for essential buffer types
        has_roi = any('_roi_' in buf for buf in buffer_list)
        has_detection = any('_detection_' in buf for buf in buffer_list)
        has_results = any('_results_' in buf for buf in buffer_list)
        has_frame = any('_frame_' in buf for buf in buffer_list)
        has_gui = any('_gui_' in buf for buf in buffer_list)
        has_cmd = any('_cmd_' in buf for buf in buffer_list)

        # Valid buffer sets:
        # 1. MediaPipe native: ROI buffers
        # 2. SCRFD pipeline: Detection + Results + Frame buffers
        # 3. Legacy: GUI + CMD buffers
        return has_roi or (has_detection and has_results and has_frame) or (has_gui and has_cmd)
    
    def _build_buffer_info_from_coordinator(self, coordinator_info: Dict) -> Dict[str, Any]:
        """Build buffer information from coordinator info."""
        pid = coordinator_info['pid']
        buffer_config = coordinator_info['buffer_config']
        
        result = {
            'pid': pid,
            'buffer_config': buffer_config,
            'buffers': {}
        }
        
        # Build buffer names for each camera
        for cam_idx in self.camera_indices:
            result['buffers'][f'roi_{cam_idx}'] = f'yq_roi_{cam_idx}_{pid}'
            result['buffers'][f'gui_{cam_idx}'] = f'yq_gui_{cam_idx}_{pid}'
            result['buffers'][f'cmd_{cam_idx}'] = f'yq_cmd_{cam_idx}_{pid}'
            result['buffers'][f'frame_{cam_idx}'] = f'yq_frame_{cam_idx}_{pid}'
            result['buffers'][f'pose_{cam_idx}'] = f'yq_pose_{cam_idx}_{pid}'
            result['buffers'][f'results_{cam_idx}'] = f'yq_results_{cam_idx}_{pid}'
        
        return result
    
    def _build_buffer_info_from_list(self, buffer_list: List[str], pid: str) -> Dict[str, Any]:
        """Build buffer information from discovered buffer list."""
        result = {
            'pid': pid,
            'buffer_config': self._infer_buffer_config(),
            'buffers': {}
        }
        
        # Parse buffer names
        for buffer_name in buffer_list:
            # Extract buffer type and camera index
            match = re.match(r'yq_(\w+)_(\d+)_\d+$', buffer_name)
            if match:
                buf_type = match.group(1)
                cam_idx = int(match.group(2))
                
                if cam_idx in self.camera_indices:
                    key = f'{buf_type}_{cam_idx}'
                    result['buffers'][key] = buffer_name
        
        return result
    
    def _infer_buffer_config(self) -> Dict[str, int]:
        """Infer buffer configuration from config when coordinator info unavailable."""
        ps_config = self.config.get('process_separation', {})
        return {
            'ring_buffer_size': ps_config.get('ring_buffer_size', 16),
            'roi_buffer_size': ps_config.get('roi_buffer_size', 8),
            'gui_buffer_size': ps_config.get('gui_buffer_size', 8),
            'max_faces': ps_config.get('max_faces_per_frame', 8)
        }
    
    def _connect_to_buffers(self, buffer_info: Dict) -> bool:
        """Connect to discovered shared memory buffers."""
        try:
            self.coordinator_pid = buffer_info['pid']
            self.buffer_config = buffer_info['buffer_config']

            # Connect to buffers for each camera
            for cam_idx in self.camera_indices:
                # Frame buffer (for visualizers and external processes)
                frame_name = buffer_info['buffers'].get(f'frame_{cam_idx}')
                if frame_name and self.process_type in ['visualizer', 'external', 'gui']:
                    if not self._connect_frame_buffer(cam_idx, frame_name):
                        logger.warning(f"Failed to connect to frame buffer for camera {cam_idx}")

                # Detection buffer (for visualizers and external processes)
                detection_name = buffer_info['buffers'].get(f'detection_{cam_idx}')
                if detection_name and self.process_type in ['visualizer', 'external', 'gui']:
                    if not self._connect_detection_buffer(cam_idx, detection_name):
                        logger.warning(f"Failed to connect to detection buffer for camera {cam_idx}")

                # ROI buffer (required for face recognition)
                roi_name = buffer_info['buffers'].get(f'roi_{cam_idx}')
                if roi_name and self.process_type in ['face_recognition', 'landmark']:
                    if not self._connect_roi_buffer(cam_idx, roi_name):
                        logger.error(f"Failed to connect to ROI buffer for camera {cam_idx}")
                        return False

                # Command buffer (if available)
                cmd_name = buffer_info['buffers'].get(f'cmd_{cam_idx}')
                if cmd_name:
                    if not self._connect_command_buffer(cam_idx, cmd_name):
                        logger.warning(f"Failed to connect to command buffer for camera {cam_idx}")

                # GUI buffer (if needed)
                gui_name = buffer_info['buffers'].get(f'gui_{cam_idx}')
                if gui_name and self.process_type == 'face_recognition':
                    self._connect_gui_buffer(cam_idx, gui_name)

                # Pose buffer (if available and needed)
                pose_name = buffer_info['buffers'].get(f'pose_{cam_idx}')
                if pose_name and self.process_type in ['face_recognition', 'pose']:
                    self._connect_pose_buffer(cam_idx, pose_name)

                # Results buffer (if available and needed)
                results_name = buffer_info['buffers'].get(f'results_{cam_idx}')
                if results_name and self.process_type in ['face_recognition', 'gui', 'lsl', 'visualizer', 'external']:
                    if not self._connect_results_buffer(cam_idx, results_name):
                        logger.warning(f"Failed to connect to results buffer for camera {cam_idx}")

            # Log connection summary
            logger.info(f"[ProcessBufferManager] Connected buffers for {self.process_type}:")
            logger.info(f"  Frame: {len(self.frame_buffers)}, Detection: {len(self.detection_buffers)}, "
                       f"Results: {len(self.results_buffers)}")

            return True

        except Exception as e:
            logger.error(f"[ProcessBufferManager] Failed to connect to buffers: {e}")
            return False
    
    def _connect_roi_buffer(self, camera_idx: int, buffer_name: str) -> bool:
        """Connect to ROI shared memory buffer."""
        try:
            shm = shared_memory.SharedMemory(name=buffer_name)
            
            # Calculate buffer layout
            roi_size = 256 * 256 * 3
            roi_buffer_size = self.buffer_config.get('roi_buffer_size', 8)
            max_faces = self.buffer_config.get('max_faces', 8)
            
            # Buffer structure: write_index (8 bytes) + data + metadata
            self.roi_buffers[camera_idx] = {
                'shm': shm,
                'name': buffer_name,
                'write_index_offset': 0,
                'data_offset': 8,
                'roi_size': roi_size,
                'buffer_size': roi_buffer_size,
                'max_faces': max_faces,
                'metadata_size': 1024,  # FIXED: Updated for 5-point alignment metadata
                'metadata_offset': 8 + (roi_size * roi_buffer_size * max_faces)
            }
            
            logger.debug(f"[ProcessBufferManager] Connected to ROI buffer {buffer_name}")
            return True
            
        except Exception as e:
            logger.error(f"[ProcessBufferManager] Failed to connect to ROI buffer {buffer_name}: {e}")
            return False
    
    def _connect_command_buffer(self, camera_idx: int, buffer_name: str) -> bool:
        """Connect to command buffer."""
        try:
            cmd_buffer = CommandBuffer.connect(buffer_name)
            self.command_buffers[camera_idx] = cmd_buffer
            logger.debug(f"[ProcessBufferManager] Connected to command buffer {buffer_name}")
            return True
        except Exception as e:
            logger.error(f"[ProcessBufferManager] Failed to connect to command buffer {buffer_name}: {e}")
            return False
    
    def _connect_gui_buffer(self, camera_idx: int, buffer_name: str) -> bool:
        """Connect to GUI buffer for additional data."""
        try:
            shm = shared_memory.SharedMemory(name=buffer_name)

            # Get resolution from coordinator if available, otherwise from config
            if self.coordinator and camera_idx in self.coordinator.camera_resolutions:
                width, height = self.coordinator.camera_resolutions[camera_idx]
                logger.debug(f"[ProcessBufferManager] Using coordinator resolution for camera {camera_idx}: {width}x{height}")
            else:
                # Fallback to per-camera config
                cam_key = f'camera_{camera_idx}'
                cam_config = self.config.get('camera_settings', {}).get(cam_key, {})
                width = cam_config.get('width', 1280)
                height = cam_config.get('height', 720)
                logger.debug(f"[ProcessBufferManager] Using config resolution for camera {camera_idx}: {width}x{height}")

            frame_size = width * height * 3
            gui_buffer_size = self.buffer_config.get('gui_buffer_size', 8)

            self.gui_buffers[camera_idx] = {
                'shm': shm,
                'name': buffer_name,
                'write_index_offset': 0,
                'data_offset': 8,
                'frame_size': frame_size,
                'width': width,
                'height': height,
                'buffer_size': gui_buffer_size,
                'metadata_offset': 8 + (frame_size * gui_buffer_size)
            }

            logger.debug(f"[ProcessBufferManager] Connected to GUI buffer {buffer_name}")
            return True
            
        except Exception as e:
            logger.error(f"[ProcessBufferManager] Failed to connect to GUI buffer {buffer_name}: {e}")
            return False

    def _connect_pose_buffer(self, camera_idx: int, buffer_name: str) -> bool:
        """Connect to pose shared memory buffer."""
        try:
            shm = shared_memory.SharedMemory(name=buffer_name)

            # Get pose buffer layout from BUFFER_LAYOUTS
            pose_layout = BUFFER_LAYOUTS['pose']

            self.pose_buffers[camera_idx] = {
                'shm': shm,
                'name': buffer_name,
                'write_index_offset': 0,
                'metadata_offset': 8,  # After write_index (8 bytes)
                'metadata_size': pose_layout.metadata_size,
                'layout': pose_layout
            }

            logger.debug(f"[ProcessBufferManager] Connected to pose buffer {buffer_name}")
            return True

        except Exception as e:
            logger.error(f"[ProcessBufferManager] Failed to connect to pose buffer {buffer_name}: {e}")
            return False

    def _connect_results_buffer(self, camera_idx: int, buffer_name: str) -> bool:
        """Connect to results shared memory buffer (landmarks + blendshapes + bboxes)."""
        try:
            shm = shared_memory.SharedMemory(name=buffer_name)

            # Results buffer uses dynamic layout from BufferCoordinator
            # Cannot use simple BUFFER_LAYOUTS since it has variable sections
            # Store basic info; actual layout queried from BufferCoordinator when needed

            self.results_buffers[camera_idx] = {
                'shm': shm,
                'name': buffer_name,
                'write_index_offset': 0,
                'data_offset': 8,  # After write_index
                # Complex layout handled by BufferCoordinator.get_results_buffer_layout()
            }

            logger.debug(f"[ProcessBufferManager] Connected to results buffer {buffer_name}")
            return True

        except Exception as e:
            logger.error(f"[ProcessBufferManager] Failed to connect to results buffer {buffer_name}: {e}")
            return False

    def _connect_frame_buffer(self, camera_idx: int, buffer_name: str) -> bool:
        """Connect to frame buffer with dynamic resolution support."""
        try:
            shm = shared_memory.SharedMemory(name=buffer_name)

            # Read ACTUAL resolution from buffer header (not config!)
            # Buffer header layout: [write_index(8)][width(4)][height(4)]
            # Resolution stored at bytes 8-15 as uint32
            resolution_data = np.frombuffer(shm.buf, dtype=np.uint32, count=2, offset=8)
            width = int(resolution_data[0])
            height = int(resolution_data[1])

            # Validate resolution is non-zero
            if width == 0 or height == 0:
                logger.warning(f"[ProcessBufferManager] Buffer header has zero resolution for {buffer_name}, using config fallback")
                cam_key = f'camera_{camera_idx}'
                cam_config = self.config.get('camera_settings', {}).get(cam_key, {})
                width = cam_config.get('width', 1280)
                height = cam_config.get('height', 720)

            frame_size = width * height * 3  # BGR format

            # Get ring buffer size from config
            ring_buffer_size = self.config.get('process_separation', {}).get('ring_buffer_size', 16)

            self.frame_buffers[camera_idx] = {
                'shm': shm,
                'name': buffer_name,
                'write_index_offset': 0,
                'data_offset': 8,  # After write_index
                'frame_size': frame_size,
                'width': width,
                'height': height,
                'ring_buffer_size': ring_buffer_size,
                # Frame buffer layout handled by BufferCoordinator.get_layout('frame')
            }

            logger.debug(f"[ProcessBufferManager] Connected to frame buffer {buffer_name} ({width}x{height}) [read from buffer header]")
            return True

        except Exception as e:
            logger.error(f"[ProcessBufferManager] Failed to connect to frame buffer {buffer_name}: {e}")
            return False

    def _connect_detection_buffer(self, camera_idx: int, buffer_name: str) -> bool:
        """Connect to SCRFD detection buffer."""
        try:
            shm = shared_memory.SharedMemory(name=buffer_name)

            # Get max faces from config
            max_faces = self.config.get('process_separation', {}).get('max_faces_per_frame', 8)

            # Detection buffer layout:
            # Header (64B): write_index (8B), ref_count (4B), n_detections (4B), frame_id (8B), timestamp (8B), ...
            # Metadata: 128 bytes per detection (bbox, keypoints, confidence, etc.)
            # Crops: 192x192x3 face crops (zero-copy storage)

            HEADER_SIZE = 64
            DETECTION_META_SIZE = 128
            CROP_SIZE = 192 * 192 * 3

            self.detection_buffers[camera_idx] = {
                'shm': shm,
                'name': buffer_name,
                'write_index_offset': 0,
                'header_size': HEADER_SIZE,
                'metadata_offset': HEADER_SIZE,  # Metadata starts after header
                'detection_meta_size': DETECTION_META_SIZE,
                'crop_size': CROP_SIZE,
                'max_faces': max_faces,
                # Complex layout handled by BufferCoordinator.create_detection_buffer()
            }

            # Store in detection_buffers dict (not a typo - different from self.detection_shm)
            logger.debug(f"[ProcessBufferManager] Connected to detection buffer {buffer_name} (max_faces={max_faces})")
            return True

        except Exception as e:
            logger.error(f"[ProcessBufferManager] Failed to connect to detection buffer {buffer_name}: {e}")
            return False

    def get_buffer_config(self) -> Dict[str, int]:
        """Get standardized buffer configuration from coordinator."""
        return self.buffer_config.copy()
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration for this process type."""
        # Get process-specific configuration
        if self.process_type == 'face_recognition':
            face_config = self.config.get('face_recognition_process', {})
            return {
                'max_faces': self.buffer_config.get('max_faces', 8),
                'frame_skip': face_config.get('frame_skip', 2),
                'quality_threshold': face_config.get('quality_threshold', 0.5),
                'processing_interval_ms': face_config.get('processing_interval_ms', 1),
                'batch_size': self.buffer_config.get('roi_buffer_size', 8) // 2
            }
        else:
            return self.buffer_config
    
    def get_roi_data_batch(self, camera_indices: Optional[List[int]] = None) -> Dict[int, List[Dict]]:
        """
        Get ROI data for multiple cameras efficiently using direct index access.
        
        Args:
            camera_indices: List of camera indices to fetch, or None for all
            
        Returns:
            Dictionary mapping camera_idx -> list of ROI dictionaries
        """
        if camera_indices is None:
            camera_indices = self.camera_indices
        
        results = {}
        
        for cam_idx in camera_indices:
            if cam_idx in self.roi_buffers:
                roi_data = self._get_latest_roi_data_direct(cam_idx)
                if roi_data:
                    results[cam_idx] = roi_data
        
        return results
    
    def _get_latest_roi_data_direct(self, camera_idx: int) -> Optional[List[Dict]]:
        """
        Get latest ROI data using direct write-index access (O(1)).
        No scanning required - direct read from current write position.
        """
        try:
            buffer_info = self.roi_buffers.get(camera_idx)
            if not buffer_info:
                return None
            
            # Read write index atomically
            write_idx = self._read_atomic_index(buffer_info['shm'], buffer_info['write_index_offset'])
            
            if write_idx < 0:
                return None
            
            # Direct read from current write position
            buffer_idx = write_idx % buffer_info['buffer_size']
            
            # Read metadata for this buffer position
            metadata = self._read_roi_metadata(buffer_info, buffer_idx)
            
            if not metadata or not metadata.get('ready'):
                # Fallback to scanning if direct read fails
                self.access_stats['fallback_scans'] += 1
                return self._get_roi_data_fallback_scan(camera_idx)
            
            # Read ROI data for all faces
            roi_list = []
            num_faces = int(metadata.get('num_faces', 0))
            
            for face_idx in range(min(num_faces, buffer_info['max_faces'])):
                roi_data = self._read_roi_at_position(buffer_info, buffer_idx, face_idx)
                if roi_data:
                    roi_list.append(roi_data)
            
            self.access_stats['direct_reads'] += 1
            return roi_list if roi_list else None
            
        except Exception as e:
            logger.error(f"[ProcessBufferManager] Error reading ROI data for camera {camera_idx}: {e}")
            self.access_stats['read_errors'] += 1
            return None
    
    def _read_atomic_index(self, shm: shared_memory.SharedMemory, offset: int) -> int:
        """Read atomic write index from shared memory."""
        try:
            # Create numpy view for atomic read and track it for cleanup
            index_array = np.ndarray((1,), dtype=np.int64, buffer=shm.buf[offset:offset+8])
            self.active_numpy_views.append(index_array)
            
            # Read value and immediately clean up view to prevent BufferError
            value = int(index_array[0])
            self.active_numpy_views.remove(index_array)
            del index_array
            return value
        except Exception as e:
            logger.error(f"[ProcessBufferManager] Failed to read atomic index: {e}")
            self._cleanup_views_on_error()
            return -1
    
    def _read_roi_metadata(self, buffer_info: Dict, buffer_idx: int) -> Optional[Dict]:
        """Read metadata for specific ROI buffer position."""
        try:
            metadata_offset = buffer_info['metadata_offset']
            metadata_size = buffer_info['metadata_size']
            
            # Calculate position in metadata array
            meta_start = metadata_offset + (buffer_idx * metadata_size)
            
            # Read metadata and track numpy view for cleanup
            metadata_array = np.ndarray(
                (metadata_size // 4,), 
                dtype=np.float32, 
                buffer=buffer_info['shm'].buf[meta_start:meta_start + metadata_size]
            )
            self.active_numpy_views.append(metadata_array)
            
            # Parse metadata (layout from BUFFER_LAYOUTS)
            layout = BUFFER_LAYOUTS.get('roi', BUFFER_LAYOUTS['roi'])
            
            result = {
                'ready': bool(metadata_array[layout.ready_flag_index]),
                'frame_id': int(metadata_array[layout.frame_id_index]) if layout.frame_id_index < len(metadata_array) else 0,
                'num_faces': int(metadata_array[1]) if len(metadata_array) > 1 else 0,
                'timestamp': float(metadata_array[2]) if len(metadata_array) > 2 else 0
            }
            
            # Clean up numpy view immediately after use to prevent BufferError
            self.active_numpy_views.remove(metadata_array) 
            del metadata_array
            return result
            
        except Exception as e:
            logger.error(f"[ProcessBufferManager] Failed to read ROI metadata: {e}")
            self._cleanup_views_on_error()
            return None
    
    def _read_roi_at_position(self, buffer_info: Dict, buffer_idx: int, face_idx: int) -> Optional[Dict]:
        """Read specific ROI data at buffer position and face index."""
        try:
            roi_size = buffer_info['roi_size']
            max_faces = buffer_info['max_faces']
            
            # Calculate position in buffer
            offset = buffer_info['data_offset'] + (buffer_idx * max_faces + face_idx) * roi_size
            
            # Read ROI data and track numpy views for cleanup
            roi_array = np.ndarray(
                (256, 256, 3), 
                dtype=np.uint8,
                buffer=buffer_info['shm'].buf[offset:offset + roi_size]
            )
            self.active_numpy_views.append(roi_array)
            
            # Read face metadata from metadata section
            metadata_offset = buffer_info['metadata_offset'] + buffer_idx * buffer_info['metadata_size']
            face_meta_offset = metadata_offset + 16 + face_idx * 32  # Skip header, 32 bytes per face
            
            face_metadata = np.ndarray(
                (8,), 
                dtype=np.float32,
                buffer=buffer_info['shm'].buf[face_meta_offset:face_meta_offset + 32]
            )
            self.active_numpy_views.append(face_metadata)
            
            # Extract data before cleaning up views to prevent BufferError
            result = {
                'roi': roi_array.copy(),
                'track_id': int(face_metadata[0]),
                'confidence': float(face_metadata[1]),
                'bbox': face_metadata[2:6].tolist(),
                'buffer_idx': buffer_idx,
                'face_idx': face_idx
            }
            
            # Clean up numpy views immediately to prevent BufferError
            self.active_numpy_views.remove(roi_array)
            self.active_numpy_views.remove(face_metadata)
            del roi_array
            del face_metadata
            
            return result
            
        except Exception as e:
            logger.error(f"[ProcessBufferManager] Failed to read ROI at position: {e}")
            # Ensure cleanup on error to prevent BufferError
            self._cleanup_views_on_error()
            return None
    
    def _get_roi_data_fallback_scan(self, camera_idx: int) -> Optional[List[Dict]]:
        """Fallback scanning method when direct index read fails."""
        buffer_info = self.roi_buffers.get(camera_idx)
        if not buffer_info:
            return None
        
        # Scan all buffer positions to find latest
        latest_frame_id = -1
        latest_data = None
        
        for buffer_idx in range(buffer_info['buffer_size']):
            metadata = self._read_roi_metadata(buffer_info, buffer_idx)
            
            if metadata and metadata.get('ready') and metadata.get('frame_id', -1) > latest_frame_id:
                # Found newer data
                latest_frame_id = metadata['frame_id']
                
                # Read ROI data
                roi_list = []
                num_faces = int(metadata.get('num_faces', 0))
                
                for face_idx in range(min(num_faces, buffer_info['max_faces'])):
                    roi_data = self._read_roi_at_position(buffer_info, buffer_idx, face_idx)
                    if roi_data:
                        roi_list.append(roi_data)
                
                if roi_list:
                    latest_data = roi_list
        
        return latest_data
    
    def send_command_to_coordinator(self, command: Dict) -> bool:
        """
        Send command using optimized CommandBuffer protocol.
        
        Args:
            command: Command dictionary to send
            
        Returns:
            True if command sent successfully
        """
        try:
            # Determine target camera or broadcast
            camera_idx = command.get('camera_idx', 0)
            
            if camera_idx in self.command_buffers:
                cmd_buffer = self.command_buffers[camera_idx]
                return cmd_buffer.write_command(command)
            else:
                # Broadcast to all available command buffers
                success = False
                for cmd_buffer in self.command_buffers.values():
                    if cmd_buffer.write_command(command):
                        success = True
                return success
                
        except Exception as e:
            logger.error(f"[ProcessBufferManager] Failed to send command: {e}")
            return False
    
    def get_latest_data_direct(self, camera_idx: int, buffer_type: str) -> Optional[Any]:
        """
        Get latest data using write-index for O(1) access.
        No scanning required - direct read from current write position.
        
        Args:
            camera_idx: Camera index
            buffer_type: Type of buffer ('roi', 'gui', 'frame')
            
        Returns:
            Latest data from buffer or None
        """
        if buffer_type == 'roi':
            roi_data = self._get_latest_roi_data_direct(camera_idx)
            return roi_data[0] if roi_data else None
        elif buffer_type == 'gui':
            return self._get_latest_gui_frame_direct(camera_idx)
        else:
            logger.warning(f"[ProcessBufferManager] Unsupported buffer type: {buffer_type}")
            return None
    
    def _get_latest_gui_frame_direct(self, camera_idx: int) -> Optional[np.ndarray]:
        """Get latest GUI frame using direct index access."""
        try:
            buffer_info = self.gui_buffers.get(camera_idx)
            if not buffer_info:
                return None
            
            # Read write index
            write_idx = self._read_atomic_index(buffer_info['shm'], buffer_info['write_index_offset'])
            
            if write_idx < 0:
                return None
            
            # Direct read from current position
            buffer_idx = write_idx % buffer_info['buffer_size']
            
            # Read frame data
            frame_size = buffer_info['frame_size']
            offset = buffer_info['data_offset'] + buffer_idx * frame_size
            
            frame_array = np.ndarray(
                (buffer_info['frame_size'] // 3, 3),
                dtype=np.uint8,
                buffer=buffer_info['shm'].buf[offset:offset + frame_size]
            )
            self.active_numpy_views.append(frame_array)

            self.access_stats['direct_reads'] += 1

            # Create result and clean up view to prevent BufferError
            # Use dynamic resolution from buffer_info
            height = buffer_info['height']
            width = buffer_info['width']
            result = frame_array.reshape((height, width, 3)).copy()
            self.active_numpy_views.remove(frame_array)
            del frame_array
            return result
            
        except Exception as e:
            logger.error(f"[ProcessBufferManager] Failed to read GUI frame: {e}")
            self.access_stats['read_errors'] += 1
            self._cleanup_views_on_error()
            return None
    
    def get_buffer_health(self) -> Dict[str, Any]:
        """Get buffer access health metrics."""
        total_accesses = (self.access_stats['direct_reads'] +
                         self.access_stats['fallback_scans'])

        return {
            'discovery_time_ms': self.discovery_time * 1000,
            'connected_buffers': {
                'roi': len(self.roi_buffers),
                'command': len(self.command_buffers),
                'gui': len(self.gui_buffers),
                'pose': len(self.pose_buffers),
                'results': len(self.results_buffers),
                'frame': len(self.frame_buffers),
                'detection': len(self.detection_buffers)
            },
            'access_stats': self.access_stats.copy(),
            'direct_read_ratio': (self.access_stats['direct_reads'] / total_accesses
                                 if total_accesses > 0 else 0),
            'coordinator_pid': self.coordinator_pid,
            'active_numpy_views': len(self.active_numpy_views)
        }

    # ============ PUBLIC ACCESSOR METHODS FOR EXTERNAL PROCESSES ============

    def get_frame_buffer(self, camera_idx: int) -> Optional[Dict]:
        """
        Get frame buffer info for external access (e.g., visualizers).

        Args:
            camera_idx: Camera index

        Returns:
            Dictionary with buffer info (shm, name, width, height, etc.) or None
        """
        return self.frame_buffers.get(camera_idx)

    def get_detection_buffer(self, camera_idx: int) -> Optional[Dict]:
        """
        Get detection buffer info for external access (e.g., visualizers).

        Args:
            camera_idx: Camera index

        Returns:
            Dictionary with buffer info (shm, name, metadata_offset, etc.) or None
        """
        return self.detection_buffers.get(camera_idx)

    def get_results_buffer(self, camera_idx: int) -> Optional[Dict]:
        """
        Get results buffer info for external access (e.g., visualizers).

        Args:
            camera_idx: Camera index

        Returns:
            Dictionary with buffer info (shm, name, etc.) or None
        """
        return self.results_buffers.get(camera_idx)

    def get_all_buffers(self, camera_idx: int) -> Dict[str, Optional[Dict]]:
        """
        Get all buffer references for a camera (convenience method for external processes).

        Args:
            camera_idx: Camera index

        Returns:
            Dictionary mapping buffer type to buffer info
        """
        return {
            'frame': self.get_frame_buffer(camera_idx),
            'detection': self.get_detection_buffer(camera_idx),
            'results': self.get_results_buffer(camera_idx),
            'roi': self.roi_buffers.get(camera_idx),
            'gui': self.gui_buffers.get(camera_idx),
            'pose': self.pose_buffers.get(camera_idx),
            'command': self.command_buffers.get(camera_idx)
        }
    
    def _cleanup_views_on_error(self):
        """Clean up any remaining numpy views on error to prevent BufferError."""
        try:
            views_cleaned = len(self.active_numpy_views)
            for view in list(self.active_numpy_views):
                del view
            self.active_numpy_views.clear()
            if views_cleaned > 0:
                logger.debug(f"[ProcessBufferManager] Cleaned up {views_cleaned} numpy views on error")
        except Exception as e:
            logger.warning(f"[ProcessBufferManager] Error during view cleanup: {e}")

    def _log_discovery_diagnostics(self):
        """Log diagnostic information for troubleshooting buffer discovery failures."""
        logger.error("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.error("Buffer Discovery Diagnostics:")
        logger.error("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # 1. Process info
        logger.error(f"  Process Type: {self.process_type}")
        logger.error(f"  Camera Indices: {self.camera_indices}")

        # 2. Check /dev/shm for YouQuantiPy buffers
        if os.path.exists('/dev/shm'):
            yq_buffers = [f for f in os.listdir('/dev/shm') if f.startswith('yq_')]

            if yq_buffers:
                logger.error(f"  Found {len(yq_buffers)} YouQuantiPy buffers in /dev/shm:")

                # Group by camera and buffer type
                from collections import defaultdict
                buffer_groups = defaultdict(list)

                for buf_name in yq_buffers:
                    # Parse: yq_{type}_{camera}_{pid}
                    parts = buf_name.split('_')
                    if len(parts) >= 4:
                        buf_type = parts[1]
                        try:
                            cam_idx = int(parts[2])
                            pid = parts[3]
                            buffer_groups[(cam_idx, pid)].append((buf_type, buf_name))
                        except (ValueError, IndexError):
                            logger.error(f"     - {buf_name} (invalid format)")

                # Display grouped buffers
                for (cam_idx, pid), buffers in sorted(buffer_groups.items()):
                    types = ', '.join([t for t, _ in buffers])
                    logger.error(f"     Camera {cam_idx}, PID {pid}: [{types}]")

                    # Show first few buffer names for this camera
                    for buf_type, buf_name in buffers[:3]:
                        logger.error(f"       - {buf_name}")

                # Check if requested camera buffers exist
                for cam_idx in self.camera_indices:
                    cam_buffers = [b for (c, p), bl in buffer_groups.items() if c == cam_idx for b in bl]
                    if cam_buffers:
                        logger.error(f"  ✅ Camera {cam_idx} has {len(cam_buffers)} buffers")
                    else:
                        logger.error(f"  ❌ Camera {cam_idx} has NO buffers")
            else:
                logger.error("  ❌ No YouQuantiPy buffers found in /dev/shm")
                logger.error("     → Is the main application running?")
                logger.error("     → Did the main application start successfully?")
        else:
            logger.error("  ❌ /dev/shm directory not found!")
            logger.error("     → This is unusual - check your Linux system configuration")

        # 3. Check for PID files
        import tempfile
        temp_dir = tempfile.gettempdir()
        home_dir = os.path.expanduser("~")

        pid_file_locations = [
            os.path.join(temp_dir, "youquantipy_coordinator.pid"),
            os.path.join(home_dir, ".youquantipy", "coordinator.pid"),
            os.path.join(os.getcwd(), "youquantipy_coordinator.pid")
        ]

        pid_file_found = False
        for pid_file_path in pid_file_locations:
            if os.path.exists(pid_file_path):
                try:
                    with open(pid_file_path, 'r') as f:
                        import json
                        coordinator_info = json.load(f)
                    logger.error(f"  ✅ PID file exists: {pid_file_path}")
                    logger.error(f"     PID: {coordinator_info.get('pid')}, Cameras: {coordinator_info.get('camera_count')}")
                    pid_file_found = True
                except Exception as e:
                    logger.error(f"  ⚠️  PID file exists but invalid: {pid_file_path} ({e})")
                    pid_file_found = True

        if not pid_file_found:
            logger.error(f"  ❌ No PID files found in:")
            for path in pid_file_locations:
                logger.error(f"     - {path}")
            logger.error("     → Main application may not have started coordinator")

        # 4. Suggestions
        logger.error("")
        logger.error("Troubleshooting Steps:")
        logger.error("  1. Ensure main application is running (./run.sh)")
        logger.error("  2. Wait a few seconds after starting for buffers to be created")
        logger.error("  3. Check main application logs for errors")
        logger.error("  4. Verify camera indices match enabled cameras in config")
        logger.error("  5. Try increasing timeout_seconds in discover_and_connect()")
        logger.error("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    def cleanup(self):
        """Clean up buffer connections."""
        
        # CRITICAL FIX: Clean up any remaining numpy views first to prevent BufferError
        if self.active_numpy_views:
            logger.info(f"[ProcessBufferManager] Cleaning up {len(self.active_numpy_views)} active numpy views")
            for view in list(self.active_numpy_views):
                try:
                    del view
                except Exception as e:
                    logger.warning(f"[ProcessBufferManager] Failed to delete numpy view: {e}")
            self.active_numpy_views.clear()
        
        # Now safe to close shared memory connections
        for buffer_info in self.roi_buffers.values():
            try:
                buffer_info['shm'].close()
            except Exception as e:
                logger.warning(f"Failed to close ROI buffer {buffer_info.get('name', 'unknown')}: {e}")
        
        for buffer_info in self.gui_buffers.values():
            try:
                buffer_info['shm'].close()
            except Exception as e:
                logger.warning(f"Failed to close GUI buffer {buffer_info.get('name', 'unknown')}: {e}")

        for buffer_info in self.pose_buffers.values():
            try:
                buffer_info['shm'].close()
            except Exception as e:
                logger.warning(f"Failed to close pose buffer {buffer_info.get('name', 'unknown')}: {e}")

        for buffer_info in self.results_buffers.values():
            try:
                buffer_info['shm'].close()
            except Exception as e:
                logger.warning(f"Failed to close results buffer {buffer_info.get('name', 'unknown')}: {e}")

        for buffer_info in self.frame_buffers.values():
            try:
                buffer_info['shm'].close()
            except Exception as e:
                logger.warning(f"Failed to close frame buffer {buffer_info.get('name', 'unknown')}: {e}")
        
        # Close command buffers
        for cmd_buffer in self.command_buffers.values():
            try:
                cmd_buffer.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup command buffer: {e}")
        
        # Clear references to buffer info dictionaries
        self.roi_buffers.clear()
        self.gui_buffers.clear()
        self.pose_buffers.clear()
        self.results_buffers.clear()
        self.frame_buffers.clear()
        self.command_buffers.clear()
        
        logger.info(f"[ProcessBufferManager] Cleaned up buffer connections")