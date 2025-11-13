"""
Integration module for GUI to interface with enhanced camera workers
Handles the bridge between process separation architecture and existing GUI components
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory, Queue, Process
import time
from typing import Dict, List, Optional, Tuple, Any
import logging
import threading
from queue import Empty

from core.buffer_management.camera_worker_enhanced import EnhancedCameraWorker
from core.process_management.confighandler import ConfigHandler
from core.buffer_management.coordinator import BufferCoordinator
from core.buffer_management.sharedbuffer import CommandBuffer
# REMOVED: from core.buffer_management.gui_buffer_manager import GUIBufferManager
# Using only DisplayBufferManager for clean architecture

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('CameraWorkerIntegration')


def get_nested_config(config: Dict, path: str, default=None):
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., 'buffer_management.timeouts.command_retry_delay_ms')
        default: Default value if path not found
        
    Returns:
        Configuration value or default
    """
    if not config:
        return default
        
    keys = path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


class CameraWorkerIntegration:
    """
    Handles integration between GUI and enhanced camera workers.
    Reads results from shared memory and formats for display.
    
    This class provides the interface between the new process separation
    architecture and the existing GUI components.
    """
    
    def __init__(self, camera_index: int, gui_buffer_manager=None):
        self.camera_index = camera_index
        # REMOVED: self.gui_buffer_manager = gui_buffer_manager
        # Using only DisplayBufferManager now
        
        # Cache for last known good results
        self.last_results = None
        self.last_sequence = 0
        
        # Performance tracking
        self.last_update_time = 0
        self.update_count = 0
        
        logger.info(f"Camera {camera_index} integration initialized with GUI buffer manager")

    def get_latest_results(self) -> Optional[Dict]:
        """Get latest results using GUI buffer manager with sequence validation."""
        try:
            # Get latest data from GUI buffer manager
            # REMOVED: GUI buffer manager usage
            # This integration class is deprecated - using DisplayBufferManager directly
            data = None
            
            if not data:
                return self.last_results
            
            # Check sequence ordering to prevent stale data
            if data['sequence_num'] <= self.last_sequence:
                return self.last_results
            
            # Update sequence tracking
            self.last_sequence = data['sequence_num']
            self.update_count += 1
            self.last_update_time = time.time()
            
            # Convert to legacy format for backward compatibility
            legacy_result = {
                'sequence_num': data['sequence_num'],
                'timestamp': data['timestamp'],
                'n_faces': data['n_faces'],
                'camera_index': self.camera_index,
                'faces': [],  # Will be populated from actual landmark data if available
                'slot': data.get('slot', 0)
            }
            
            # Cache the result
            self.last_results = legacy_result
            
            return legacy_result
            
        except Exception as e:
            logger.error(f"Error getting results for camera {self.camera_index}: {e}")
            return self.last_results
    
    def _compute_transform(self, bbox: np.ndarray) -> Dict:
        """Compute coordinate transform from ROI to frame space."""
        try:
            x1, y1, x2, y2 = bbox
            
            # Ensure valid bbox
            if x2 <= x1 or y2 <= y1:
                return {
                    'x1': 0, 'y1': 0,
                    'scale_x': 1.0, 'scale_y': 1.0,
                    'valid': False
                }
            
            return {
                'x1': float(x1),
                'y1': float(y1), 
                'scale_x': float(x2 - x1),  # Width of ROI (landmarks in 0-1 range)
                'scale_y': float(y2 - y1),  # Height of ROI (landmarks in 0-1 range)
                'valid': True
            }
        except Exception as e:
            logger.error(f"Transform computation failed: {e}")
            return {
                'x1': 0, 'y1': 0,
                'scale_x': 1.0, 'scale_y': 1.0,
                'valid': False
            }
    
    def transform_landmarks_to_frame(self, landmarks: np.ndarray, transform: Dict) -> np.ndarray:
        """Transform landmarks from ROI space (256x256) to frame space."""
        # Skip validation check - always transform landmarks
            
        try:
            transformed = landmarks.copy()
            
            # Transform x coordinates
            transformed[:, 0] = (landmarks[:, 0] * transform['scale_x']) + transform['x1']
            
            # Transform y coordinates  
            transformed[:, 1] = (landmarks[:, 1] * transform['scale_y']) + transform['y1']
            
            # Z coordinate remains unchanged (depth relative to face plane)
            
            return transformed
            
        except Exception as e:
            logger.error(f"Landmark transformation failed: {e}")
            return landmarks
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        current_time = time.time()
        elapsed = current_time - self.last_update_time if self.last_update_time > 0 else 1.0
        fps = self.update_count / elapsed if elapsed > 0 else 0
        
        return {
            'camera_index': self.camera_index,
            'update_count': self.update_count,
            'fps': fps,
            'last_frame_id': getattr(self, '_last_frame_ids', {}).get(self.camera_index, -1),
            'has_results': self.last_results is not None,
            'direct_access_stats': getattr(self, '_direct_access_stats', {}),
            'sync_health': self._get_sync_health_status()
        }
    
    def _get_sync_health_status(self) -> Dict:
        """Get frame-overlay synchronization health status."""
        if not hasattr(self, '_last_frame_ids'):
            return {'status': 'no_data'}
            
        frame_id = self._last_frame_ids.get(self.camera_index, -1)
        direct_reads = self._direct_access_stats.get('gui_direct_reads', 0)
        fallbacks = self._direct_access_stats.get('gui_fallbacks', 0)
        
        total_reads = direct_reads + fallbacks
        direct_ratio = direct_reads / total_reads if total_reads > 0 else 0
        
        return {
            'status': 'healthy' if direct_ratio > 0.8 else 'degraded',
            'current_frame_id': frame_id,
            'direct_access_ratio': direct_ratio,
            'total_reads': total_reads
        }
    
    def cleanup(self):
        """Clean up shared memory connections."""
        try:
            if self.results_shm:
                self.results_shm.close()
                logger.info(f"Camera {self.camera_index} integration cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error for camera {self.camera_index}: {e}")


class CameraWorkerManager:
    """Manages multiple camera workers for the application."""
    
    def __init__(self, config: Dict = None, participant_update_queue: Optional[mp.Queue] = None,
                 buffer_coordinator: BufferCoordinator = None, gui_buffer_manager = None):
        """Initialize camera worker manager."""
        self.config = config or ConfigHandler().config
        self.workers = {}  # camera_index -> worker process
        self.control_queues = {}  # camera_index -> control queue
        self.status_queue = mp.Queue()  # Shared status queue for all cameras
        self.shared_memories = {}  # camera_index -> {'frame': shm, 'results': shm}
        self.participant_update_queue = participant_update_queue  # For GlobalParticipantManager

        # Async camera startup callbacks
        self.camera_ready_callbacks = {}  # camera_index -> callback(camera_index, buffer_names)
        self.camera_failed_callbacks = {}  # camera_index -> callback(camera_index, error_message)
        
        # Initialize direct access statistics
        self._direct_access_stats = {
            'gui_direct_reads': 0,
            'gui_fallbacks': 0,
            'roi_direct_reads': 0,
            'roi_fallbacks': 0
        }
        
        # CommandBuffer system for GUI integration (Phase 4)
        self.gui_command_buffers = {}  # Per-camera GUI command buffers
        self.detection_command_buffers = {}  # Per-camera detection command buffers
        # Keep the exact SHM names we created per camera to avoid mismatches
        self._command_buffer_names_by_camera: Dict[int, Dict[str, str]] = {}
        self.max_cameras = 4  # Maximum number of cameras supported
        
        # Buffer coordination
        self.buffer_coordinator = buffer_coordinator
        self.coordinator = buffer_coordinator  # Alias for easier access
        if self.buffer_coordinator is None:
            logger.warning("No BufferCoordinator provided, creating fallback")
            self.buffer_coordinator = BufferCoordinator(camera_count=4, config=self.config)  # 4 cameras max
            self.coordinator = self.buffer_coordinator
            self._owns_coordinator = True
        else:
            self._owns_coordinator = False
        
        # GUI buffer management
        # REMOVED: self.gui_buffer_manager = gui_buffer_manager
        # Using only DisplayBufferManager now
        logger.info("CameraWorkerManager created GUIBufferManager")
        
        # Initialize CommandBuffer system for GUI integration
        self._init_gui_command_buffers()
        
        # Get number of GPUs available
        self.num_gpus = self._get_num_gpus()
        logger.info(f"Camera manager initialized with {self.num_gpus} GPU(s)")
        
    def _init_gui_command_buffers(self):
        """Initialize empty command buffer dictionaries. Buffers created on-demand when cameras selected."""
        try:
            if not self.buffer_coordinator:
                logger.warning("No BufferCoordinator available for GUI command buffer initialization")
                return

            # Initialize empty dictionaries - buffers will be created on-demand
            self.gui_command_buffers = {}
            self.detection_command_buffers = {}
            self._command_buffer_names_by_camera = {}

            logger.info("GUI command buffer system initialized for on-demand camera buffer creation")
            
            # Initialize handshake state tracking for tolerant handshake
            self._handshake = {}  # per-camera: {state, deadline, last_ping, req_id}

        except Exception as e:
            logger.error(f"Failed to initialize GUI command buffer system: {e}")
    
    def _ensure_camera_command_buffers(self, camera_idx: int):
        """Create and connect command buffers for a specific camera if they don't exist."""
        try:
            # Check if buffers already exist for this camera
            if camera_idx in self.gui_command_buffers and camera_idx in self._command_buffer_names_by_camera:
                logger.debug(f"Command buffers already exist for camera {camera_idx}")
                return

            # Create command buffers for this specific camera
            logger.info(f"Creating on-demand command buffers for camera {camera_idx}")
            buffer_names = self.buffer_coordinator.create_camera_command_buffers(camera_idx)
            self._command_buffer_names_by_camera[camera_idx] = buffer_names.copy()

            # Connect manager-side CommandBuffers by exact names
            self.gui_command_buffers[camera_idx] = {}
            if 'camera_to_gui' in buffer_names:
                self.gui_command_buffers[camera_idx]['camera_to_gui'] = CommandBuffer.connect(buffer_names['camera_to_gui'])
            if 'gui_to_camera' in buffer_names:
                self.gui_command_buffers[camera_idx]['gui_to_camera'] = CommandBuffer.connect(buffer_names['gui_to_camera'])

            # Optional detection paths
            self.detection_command_buffers[camera_idx] = {}
            if 'camera_to_detection' in buffer_names:
                self.detection_command_buffers[camera_idx]['camera_to_detection'] = CommandBuffer.connect(buffer_names['camera_to_detection'])
            if 'detection_to_camera' in buffer_names:
                self.detection_command_buffers[camera_idx]['detection_to_camera'] = CommandBuffer.connect(buffer_names['detection_to_camera'])

            # Validate required GUI paths exist
            missing = [k for k in ('camera_to_gui', 'gui_to_camera') if k not in self.gui_command_buffers[camera_idx]]
            if missing:
                logger.error(f"Missing required GUI command buffers for camera {camera_idx}: {missing}")
                # Clean up partial initialization
                if camera_idx in self.gui_command_buffers:
                    del self.gui_command_buffers[camera_idx]
                if camera_idx in self.detection_command_buffers:
                    del self.detection_command_buffers[camera_idx]
                if camera_idx in self._command_buffer_names_by_camera:
                    del self._command_buffer_names_by_camera[camera_idx]
                raise RuntimeError(f"Failed to create required command buffers for camera {camera_idx}")

            logger.info(f"Successfully created command buffers for camera {camera_idx}:")
            logger.info(f"  camera_to_gui -> {self.gui_command_buffers[camera_idx]['camera_to_gui'].name}")
            logger.info(f"  gui_to_camera -> {self.gui_command_buffers[camera_idx]['gui_to_camera'].name}")

        except Exception as e:
            logger.error(f"Failed to create command buffers for camera {camera_idx}: {e}")
            raise

    def send_camera_command(self, camera_idx: int, command_type: str, payload: Dict) -> Optional[int]:
        """
        Send command to camera worker via CommandBuffer.
        
        Args:
            camera_idx: Camera index to send command to
            command_type: Type of command (e.g., 'register_track_participant')
            payload: Command payload data
            
        Returns:
            Command ID if successful, None if failed
        """
        try:
            if camera_idx in self.gui_command_buffers:
                gui_buffer = self.gui_command_buffers[camera_idx]['gui_to_camera']
                command_id = gui_buffer.send_command(command_type, payload)
                if command_id:
                    logger.debug(f"Sent command {command_type} to camera {camera_idx} (id={command_id})")
                    return command_id
                else:
                    logger.warning(f"Failed to send command {command_type} to camera {camera_idx}")
            else:
                logger.error(f"No GUI command buffer available for camera {camera_idx}")
            return None
        except Exception as e:
            logger.error(f"Error sending command {command_type} to camera {camera_idx}: {e}")
            return None
    
    def get_command_status(self, camera_idx: int, command_id: int) -> Optional[Dict]:
        """
        Get status of sent command via acknowledgment system.
        
        Args:
            camera_idx: Camera index
            command_id: Command ID to check status for
            
        Returns:
            Acknowledgment dictionary or None if not available
        """
        try:
            if camera_idx in self.gui_command_buffers:
                gui_buffer = self.gui_command_buffers[camera_idx]['gui_to_camera']
                ack = gui_buffer.get_acknowledgment(command_id, timeout=0.1)
                if ack:
                    logger.debug(f"Got acknowledgment for command {command_id} from camera {camera_idx}: {ack['success']}")
                return ack
            else:
                logger.warning(f"No GUI command buffer available for camera {camera_idx}")
            return None
        except Exception as e:
            logger.error(f"Error getting command status for camera {camera_idx}, command {command_id}: {e}")
            return None
    
    def get_command_statistics(self) -> Dict:
        """
        Get overall command delivery statistics.
        
        Returns:
            Dictionary with command statistics
        """
        try:
            stats = {'sent': 0, 'acknowledged': 0, 'failed': 0, 'cameras': {}}
            
            for camera_idx in self.gui_command_buffers:
                try:
                    gui_buffer = self.gui_command_buffers[camera_idx]['gui_to_camera']
                    buffer_info = gui_buffer.get_buffer_info()
                    
                    camera_stats = {
                        'buffer_name': buffer_info.get('name', 'unknown'),
                        'pending_commands': buffer_info.get('pending_commands', 0),
                        'buffer_size': buffer_info.get('buffer_size', 0)
                    }
                    stats['cameras'][camera_idx] = camera_stats
                    
                    # For overall stats, we'll use buffer info as proxy
                    # This is simplified - full stats would require tracking in CommandBuffer
                    
                except Exception as e:
                    logger.warning(f"Failed to get statistics for camera {camera_idx}: {e}")
                    continue
                    
            return stats
        except Exception as e:
            logger.error(f"Error getting command statistics: {e}")
            return {'error': str(e)}
    
    def is_camera_active(self, camera_idx: int) -> bool:
        """Check if camera is active and has command buffer available."""
        return (camera_idx in self.workers and 
                self.workers[camera_idx].is_alive() and
                camera_idx in self.gui_command_buffers)
    
    def _get_num_gpus(self) -> int:
        """Get number of available GPUs."""
        try:
            import cupy as cp
            device_count = cp.cuda.runtime.getDeviceCount()
            return device_count
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return 1  # Default to 1 if can't detect
    
    def start_camera(self, camera_index: int, display_only: bool = False) -> tuple[bool, Dict[str, str], tuple]:
        """
        Start a camera worker process.

        Args:
            camera_index: Camera device index
            display_only: If True, camera will only display frames without running pose/face detection

        Returns:
            tuple: (success: bool, buffer_names: Dict[str, str], resolution: tuple)
                - success: True if started successfully
                - buffer_names: Dictionary of shared memory buffer names from camera worker
                - resolution: (width, height) tuple from camera, or None if failed
        """
        
        if camera_index in self.workers:
            logger.warning(f"Camera {camera_index} already started")
            return False, {}, None
        
        try:
            # Create control queue for this camera
            control_queue = mp.Queue()
            self.control_queues[camera_index] = control_queue
            
            # Determine GPU assignment (round-robin)
            gpu_id = camera_index % self.num_gpus
            
            
            # Ensure command buffers exist for this camera (on-demand creation)
            self._ensure_camera_command_buffers(camera_index)
            
            # Get the exact command buffer names we created for this camera
            cmd_names = self._command_buffer_names_by_camera.get(camera_index)
            if not cmd_names:
                raise RuntimeError(f"No command buffer names recorded for camera {camera_index}; initialization incomplete")
            # Pass the exact names to the worker
            command_buffer_names = cmd_names.copy()
            
            # Log the exact buffer names being passed
            logger.info(f"[CameraManager] Passing exact buffer names to worker {camera_index}: {command_buffer_names}")
            
            # Validate required buffers exist
            if 'camera_to_gui' not in command_buffer_names:
                raise RuntimeError(f"Missing camera_to_gui buffer name for camera {camera_index}")
            if 'gui_to_camera' not in command_buffer_names:
                raise RuntimeError(f"Missing gui_to_camera buffer name for camera {camera_index}")
            
            # Debug logging to verify buffer objects and names
            if camera_index in self.gui_command_buffers:
                for key, buf in self.gui_command_buffers[camera_index].items():
                    logger.info(f"  {key}: object={buf}, name={buf.name if buf else 'None'}")
            if camera_index in self.detection_command_buffers and self.detection_command_buffers[camera_index]:
                for key, buf in self.detection_command_buffers[camera_index].items():
                    logger.info(f"  {key}: object={buf}, name={buf.name if buf else 'None'}")
            
            # Get frame ready semaphore from coordinator for this camera
            # CRITICAL: Pass the SAME semaphore instance to avoid deadlock
            frame_ready_semaphore = self.buffer_coordinator.get_frame_ready_semaphore(camera_index)
            logger.info(f"[CameraManager] Passing frame_ready_semaphore for camera {camera_index} to worker")

            # Get pose frame ready semaphore from coordinator for this camera
            # CRITICAL: Pass the SAME semaphore instance to avoid deadlock
            pose_frame_ready_semaphore = self.buffer_coordinator.get_pose_frame_ready_semaphore(camera_index)
            logger.info(f"[CameraManager] Passing pose_frame_ready_semaphore for camera {camera_index} to worker")

            # Create worker process
            worker = EnhancedCameraWorker(
                camera_index=camera_index,
                gpu_device_id=gpu_id,
                config=self.config,
                control_queue=control_queue,
                status_queue=self.status_queue,
                participant_update_queue=self.participant_update_queue,
                frame_ready_semaphore=frame_ready_semaphore,  # Pass semaphore from parent
                pose_frame_ready_semaphore=pose_frame_ready_semaphore,  # Pass semaphore from parent
                # buffer_coordinator removed - child will create its own
                recovery_buffer_name=f"yq_recovery_{camera_index}",  # Add recovery buffer name
                command_buffer_names=command_buffer_names,  # Pass the buffer names
                display_only=display_only  # Pass display-only flag
            )
            
            # Start the worker
            worker.start()
            self.workers[camera_index] = worker

            # Tolerant ping/pong handshake to verify connection
            handshake_timeout_s = get_nested_config(self.config, 'camera_timeouts.handshake_timeout_s', 30.0)
            logger.info(f"[HANDSHAKE] Using handshake timeout: {handshake_timeout_s}s for camera {camera_index}")
            print(f"[DEBUG] CameraManager: Performing tolerant ping/pong handshake with camera {camera_index} (timeout: {handshake_timeout_s}s)")
            handshake_ok, ready_payload = self._verify_connection_handshake(camera_index, timeout_seconds=handshake_timeout_s)
            if not handshake_ok:
                logger.error(f"[CameraManager] Tolerant ping/pong handshake failed for camera {camera_index}")
                print(f"[DEBUG] CameraManager: Handshake FAILED for camera {camera_index}")
                return False, {}, None
            
            # Option A: If ready_payload received during handshake, act immediately
            logger.debug(f"[HANDSHAKE] Camera {camera_index} handshake returned: success={handshake_ok}, has_ready_payload={ready_payload is not None}")
            if ready_payload:
                print(f"[DEBUG] CameraManager: Handshake SUCCESS with READY - immediate setup for camera {camera_index}")
                logger.info(f"[CameraManager] Ready payload received during handshake for camera {camera_index}")
                
                # 1) Send handshake acknowledgment
                req_id = ready_payload.get("req_id")
                try:
                    self._send_handshake_ack(camera_index, req_id=req_id)
                except Exception as e:
                    logger.warning(f"[CameraManager] Failed to send handshake_ack cam={camera_index}: {e}")

                # 2) Connect shared memory from payload
                shm = (ready_payload.get("data") or {}).get("shared_memory") or {}
                try:
                    self._connect_shared_memory(camera_index, shm, ready_payload.get("data"))
                    buffer_names = shm
                    print(f"[DEBUG] CameraManager: Camera {camera_index} buffer names: {buffer_names}")
                except Exception as e:
                    logger.error(f"[CameraManager] Failed to connect SHM cam={camera_index}: {e}")
                    return False, {}, None

                # 3) Register GUI buffers
                try:
                    # REMOVED: GUI buffer manager registration
                    # Using DisplayBufferManager instead
                    gui_registration_success = True  # Placeholder
                    logger.info(f"[CameraManager] GUI buffer registration for camera {camera_index}: {'SUCCESS' if gui_registration_success else 'FAILED'}")
                except Exception as e:
                    logger.error(f"[CameraManager] GUI register failed cam={camera_index}: {e}")
                    return False, {}, None

                # 4) Mark ready and return
                hs = self._ensure_handshake_state(camera_index)
                hs["state"] = "done"

                # Extract resolution from ready payload
                resolution = ready_payload.get('data', {}).get('resolution', None)
                logger.info(f"[CameraManager] Camera {camera_index} startup complete via handshake-ready path with resolution: {resolution}")
                print(f"[DEBUG] CameraManager: Camera {camera_index} startup SUCCESS via handshake-ready")
                return True, buffer_names, resolution
            
            # Fallback: we only saw PONG during handshake; wait for READY in post-handshake loop
            ready_status_timeout_s = get_nested_config(self.config, 'camera_timeouts.ready_status_timeout_s', 60.0)
            logger.info(f"[HANDSHAKE] Using ready status timeout: {ready_status_timeout_s}s for camera {camera_index}")
            print(f"[DEBUG] CameraManager: Handshake SUCCESS with PONG - waiting for ready status from camera {camera_index} (timeout: {ready_status_timeout_s}s)")

            # Check if ready status was already received and cached during handshake
            hs = self._ensure_handshake_state(camera_index)
            if hs.get("last_ready_payload"):
                cached_ready = hs["last_ready_payload"]
                logger.info(f"[CameraManager] ✅ Using cached ready payload for camera {camera_index} (received during handshake)")
                print(f"[DEBUG] CameraManager: Found cached ready for camera {camera_index}, skipping fallback wait")

                # Extract buffer names and process immediately
                buffer_names = cached_ready.get('data', {}).get('shared_memory', {})
                print(f"[DEBUG] CameraManager: Camera {camera_index} buffer names from cache: {buffer_names}")

                # Send handshake acknowledgment
                if camera_index in self.gui_command_buffers and 'gui_to_camera' in self.gui_command_buffers[camera_index]:
                    try:
                        gui_to_camera = self.gui_command_buffers[camera_index]['gui_to_camera']
                        ack_id = gui_to_camera.send_command('handshake_ack', {'camera_index': camera_index})
                        logger.info(f"[CameraManager] ✅ Sent handshake acknowledgment to camera {camera_index} (command_id={ack_id}) [from cache]")
                    except Exception as e:
                        logger.error(f"[CameraManager] Failed to send handshake acknowledgment: {e}")

                # Connect to shared memory
                self._connect_shared_memory(camera_index, buffer_names, cached_ready.get('data'))

                # Extract resolution from cached ready
                resolution = cached_ready.get('data', {}).get('resolution', None)
                logger.info(f"Camera {camera_index} started successfully via cached ready with resolution: {resolution}")
                print(f"[DEBUG] CameraManager: Camera {camera_index} startup SUCCESS via cached ready")
                return True, buffer_names, resolution

            # Wait for ready status via CommandBuffer OR check cache (populated by GUI polling)
            timeout = time.time() + ready_status_timeout_s
            status_received = False
            while time.time() < timeout:
                try:
                    # FIRST: Check if GUI thread has populated the cache
                    hs = self._ensure_handshake_state(camera_index)
                    cached_ready = hs.get("last_ready_payload")

                    if cached_ready and cached_ready.get('type') == 'ready':
                        logger.info(f"[CameraManager] ✅ Using cached ready payload for camera {camera_index} (populated by GUI polling)")
                        print(f"[DEBUG] CameraManager: Found cached ready in fallback loop for camera {camera_index}")

                        # Extract buffer names and process immediately
                        buffer_names = cached_ready.get('data', {}).get('shared_memory', {})
                        print(f"[DEBUG] CameraManager: Camera {camera_index} buffer names from cached ready: {buffer_names}")

                        # Send handshake acknowledgment
                        if camera_index in self.gui_command_buffers and 'gui_to_camera' in self.gui_command_buffers[camera_index]:
                            try:
                                gui_to_camera = self.gui_command_buffers[camera_index]['gui_to_camera']
                                ack_id = gui_to_camera.send_command('handshake_ack', {'camera_index': camera_index})
                                logger.info(f"[CameraManager] ✅ Sent handshake acknowledgment to camera {camera_index} (command_id={ack_id}) [from cache in fallback]")
                            except Exception as e:
                                logger.error(f"[CameraManager] Failed to send handshake acknowledgment: {e}")

                        # Connect to shared memory
                        self._connect_shared_memory(camera_index, buffer_names, cached_ready.get('data'))

                        # Extract resolution from cached ready
                        resolution = cached_ready.get('data', {}).get('resolution', None)
                        logger.info(f"Camera {camera_index} started successfully via cached ready in fallback loop with resolution: {resolution}")
                        print(f"[DEBUG] CameraManager: Camera {camera_index} startup SUCCESS via cached ready in fallback")
                        return True, buffer_names, resolution

                    # SECOND: Try to consume new status messages (if cache is still empty)
                    status_updates = self.process_status_updates()

                    for status in status_updates:
                        if status.get('camera_index') == camera_index:
                            status_received = True
                            print(f"[DEBUG] CameraManager: Received status from camera {camera_index}: {status.get('type', 'unknown')}")

                            if status['type'] == 'ready':
                                print(f"[DEBUG] CameraManager: Camera {camera_index} reported ready, connecting to shared memory")
                                # Extract buffer names to return to GUI
                                buffer_names = status['data']['shared_memory']
                                print(f"[DEBUG] CameraManager: Camera {camera_index} buffer names: {buffer_names}")

                                # Send handshake acknowledgment back to worker
                                if camera_index in self.gui_command_buffers and 'gui_to_camera' in self.gui_command_buffers[camera_index]:
                                    try:
                                        gui_to_camera = self.gui_command_buffers[camera_index]['gui_to_camera']
                                        ack_id = gui_to_camera.send_command('handshake_ack', {'camera_index': camera_index})
                                        logger.info(f"[CameraManager] ✅ Sent handshake acknowledgment to camera {camera_index} (command_id={ack_id})")
                                    except Exception as e:
                                        logger.error(f"[CameraManager] Failed to send handshake acknowledgment: {e}")

                                # Connect to shared memory
                                self._connect_shared_memory(camera_index, buffer_names, status['data'])

                                # Register camera with GUI buffer manager
                                # REMOVED: GUI buffer manager registration
                                # Using DisplayBufferManager instead
                                gui_registration_success = True  # Placeholder
                                logger.info(f"GUI buffer registration for camera {camera_index}: {'SUCCESS' if gui_registration_success else 'FAILED'}")

                                # Extract resolution from status
                                resolution = status.get('data', {}).get('resolution', None)
                                logger.info(f"Camera {camera_index} started successfully on GPU {gpu_id} with resolution: {resolution}")
                                print(f"[DEBUG] CameraManager: Camera {camera_index} startup SUCCESS")
                                return True, buffer_names, resolution
                            elif status['type'] == 'error':
                                logger.error(f"Camera {camera_index} error: {status.get('data', {})}")
                                print(f"[DEBUG] CameraManager: Camera {camera_index} reported ERROR: {status.get('data', {})}")
                                return False, {}, None

                    # Brief sleep to avoid busy waiting
                    time.sleep(0.1)
                except Exception as e:
                    logger.debug(f"Error checking ready status: {e}")
                    continue
            
            # Timeout occurred - provide detailed status
            if not status_received:
                print(f"[DEBUG] CameraManager: Camera {camera_index} startup TIMEOUT - No status received")
                logger.error(f"Camera {camera_index} startup timeout - no status received")
            else:
                print(f"[DEBUG] CameraManager: Camera {camera_index} startup TIMEOUT - Status received but not ready")
                logger.error(f"Camera {camera_index} startup timeout - status received but not ready")
            
            # Check if worker process is still alive
            if camera_index in self.workers:
                worker = self.workers[camera_index]
                if worker.is_alive():
                    print(f"[DEBUG] CameraManager: Worker process for camera {camera_index} is still alive, terminating...")
                    worker.terminate()
                    worker.join(timeout=3)
                else:
                    print(f"[DEBUG] CameraManager: Worker process for camera {camera_index} has died")
                del self.workers[camera_index]

            return False, {}, None
            
        except Exception as e:
            print(f"[DEBUG] CameraManager: Exception starting camera {camera_index}: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"Failed to start camera {camera_index}: {e}")
            return False, {}, None
    
    def start_camera_workers(self, camera_indices: List[int]) -> Dict[int, bool]:
        """
        Start multiple camera workers.
        
        Args:
            camera_indices: List of camera device indices to start
            
        Returns:
            Dict[int, bool]: Mapping of camera_index -> success status
        """
        results = {}
        
        for camera_index in camera_indices:
            logger.info(f"Starting camera worker {camera_index}")
            success, buffer_names = self.start_camera(camera_index)
            results[camera_index] = success
            
            if success:
                logger.info(f"Camera {camera_index} started successfully with buffers: {list(buffer_names.keys())}")
            else:
                logger.error(f"Failed to start camera {camera_index}")
        
        return results
    
    def _ensure_handshake_state(self, cam_idx: int, timeout_s: float = 30.0) -> Dict:
        """Ensure handshake state exists for camera. Default timeout increased to 30s for WSL2 compatibility."""
        hs = self._handshake.get(cam_idx)
        if not hs:
            import time
            self._handshake[cam_idx] = hs = {
                "state": "awaiting_pong",
                "deadline": time.monotonic() + timeout_s,
                "last_ping": 0.0,
                "req_id": None,
                "last_ready_payload": None,
                "pending_status_updates": [],  # Cache non-handshake statuses (e.g., model_loading, model_ready)
            }
        return hs

    def _send_ping(self, cam_idx: int) -> bool:
        """Send ping to camera with correlation ID."""
        try:
            hs = self._ensure_handshake_state(cam_idx)
            hs["req_id"] = f"{cam_idx}-{int(time.time()*1000)}"
            payload = {"camera_index": cam_idx, "req_id": hs["req_id"], "timestamp": time.time()}
            
            if cam_idx not in self.gui_command_buffers or 'gui_to_camera' not in self.gui_command_buffers[cam_idx]:
                logger.error(f"[HANDSHAKE] No gui_to_camera buffer for camera {cam_idx}")
                return False
                
            buf = self.gui_command_buffers[cam_idx]["gui_to_camera"]
            ping_id = buf.send_command("ping", payload)
            
            if ping_id:
                hs["last_ping"] = time.monotonic()
                logger.info(f"[HANDSHAKE] Sent ping to camera {cam_idx} (req_id={hs['req_id']}, command_id={ping_id})")
                return True
            else:
                logger.error(f"[HANDSHAKE] Failed to send ping to camera {cam_idx}")
                return False
        except Exception as e:
            logger.error(f"[HANDSHAKE] Exception sending ping to camera {cam_idx}: {e}")
            return False

    def _send_handshake_ack(self, cam_idx: int, req_id: Optional[str] = None) -> bool:
        """Send handshake acknowledgment to camera with optional correlation ID."""
        try:
            if cam_idx not in self.gui_command_buffers or 'gui_to_camera' not in self.gui_command_buffers[cam_idx]:
                logger.error(f"[HANDSHAKE] No gui_to_camera buffer available for camera {cam_idx}")
                return False
            
            payload = {"camera_index": cam_idx}
            if req_id:
                payload["req_id"] = req_id
            
            gui_to_camera = self.gui_command_buffers[cam_idx]['gui_to_camera']
            ack_id = gui_to_camera.send_command('handshake_ack', payload)
            logger.info(f"[HANDSHAKE] ✅ Sent handshake acknowledgment to camera {cam_idx} (command_id={ack_id})")
            return True
            
        except Exception as e:
            logger.error(f"[HANDSHAKE] Exception sending handshake_ack to camera {cam_idx}: {e}")
            return False
    
    def _verify_connection_handshake(self, camera_index: int, timeout_seconds: float = 30.0) -> Tuple[bool, Optional[Dict]]:
        """
        Tolerant handshake with retry ping and correlation IDs.
        If 'ready' is seen during handshake, return it so caller can immediately ACK+connect+register.

        Args:
            camera_index: Camera index to test
            timeout_seconds: Total deadline for handshake completion (default 30s for WSL2 compatibility)

        Returns:
            Tuple[bool, Optional[Dict]]: (success: bool, ready_payload: Optional[Dict])
                - success: True if handshake successful (pong or ready), False otherwise
                - ready_payload: Ready status payload if seen, None if only pong
        """
        try:
            # Initialize handshake state
            hs = self._ensure_handshake_state(camera_index, timeout_seconds)
            logger.info(f"[HANDSHAKE] Starting tolerant handshake for camera {camera_index} (deadline: {timeout_seconds}s)")
            ready_payload: Optional[Dict] = None
            
            # Tolerant handshake loop
            while time.monotonic() < hs["deadline"] and hs["state"] == "awaiting_pong":
                # Re-ping every 500ms
                now = time.monotonic()
                if hs["last_ping"] == 0.0 or (now - hs["last_ping"]) >= 0.5:
                    if not self._send_ping(camera_index):
                        logger.warning(f"[HANDSHAKE] Failed to send ping to camera {camera_index}, will retry")
                
                # Check for responses (non-blocking)
                status_updates = self.process_status_updates()

                # Debug: Log all status updates received
                if status_updates:
                    logger.debug(f"[HANDSHAKE] Camera {camera_index} handshake received {len(status_updates)} status updates")
                    for s in status_updates:
                        logger.debug(f"[HANDSHAKE]   - Status: camera={s.get('camera_index')}, type={s.get('type')}")

                for status in status_updates:
                    cam_idx = status.get('camera_index')
                    status_type = status.get('type')

                    logger.debug(f"[HANDSHAKE] Checking status: cam_idx={cam_idx} (want {camera_index}), type={status_type}")

                    if cam_idx == camera_index:
                        # Process handshake-specific statuses (pong, ready)
                        if status_type in ('pong', 'ready'):
                            # Check correlation ID for pong (ready doesn't need correlation)
                            if status_type == 'pong':
                                req_id = status.get('data', {}).get('req_id')
                                expected_req_id = hs.get('req_id')
                                if req_id != expected_req_id:
                                    logger.debug(f"[HANDSHAKE] Ignoring pong with mismatched req_id: got {req_id}, expected {expected_req_id}")
                                    continue
                                logger.info(f"[HANDSHAKE] ✅ Handshake: PONG from cam {camera_index}")
                            elif status_type == 'ready':
                                logger.info(f"[HANDSHAKE] ✅ Handshake: READY from cam {camera_index}")
                                ready_payload = status

                            # Handshake success!
                            hs["state"] = "done"
                            logger.info(f"[HANDSHAKE] ✅ Handshake complete for camera {camera_index} via {status_type}")
                            return True, ready_payload
                        else:
                            # Cache non-handshake statuses (e.g., model_loading, model_ready) for GUI to process later
                            logger.info(f"[HANDSHAKE] 📦 Caching {status_type} status for camera {camera_index} (will be processed by GUI)")
                            hs["pending_status_updates"].append(status)
                
                time.sleep(0.1)  # Brief sleep to avoid busy waiting

            # Before timeout, check if ready was received and cached during handshake loop
            if not ready_payload and camera_index in self._handshake:
                cached_ready = self._handshake[camera_index].get("last_ready_payload")
                if cached_ready:
                    logger.info(f"[HANDSHAKE] ✅ Found cached ready payload for camera {camera_index} (received during loop but missed)")
                    ready_payload = cached_ready
                    hs["state"] = "done"
                    return True, ready_payload

            # Timeout reached
            logger.error(f"[HANDSHAKE] ❌ Tolerant handshake timeout for camera {camera_index}")
            hs["state"] = "failed"
            return False, None
            
        except Exception as e:
            logger.error(f"[HANDSHAKE] Exception during tolerant handshake with camera {camera_index}: {e}")
            return False, None

    def get_pending_status_updates(self, camera_index: int) -> List[Dict]:
        """
        Get and clear pending status updates that were cached during handshake.

        Args:
            camera_index: Camera index to get pending updates for

        Returns:
            List[Dict]: Cached status updates (e.g., model_loading, model_ready)
        """
        if camera_index not in self._handshake:
            return []

        hs = self._handshake[camera_index]
        pending = hs.get("pending_status_updates", [])

        if pending:
            logger.info(f"[HANDSHAKE] Retrieving {len(pending)} cached status updates for camera {camera_index}")
            # Clear the cache after retrieval
            hs["pending_status_updates"] = []

        return pending

    def _connect_shared_memory(self, camera_index: int, shm_names: Dict[str, str], status_data: Dict = None):
        """Connect to shared memory created by camera worker."""
        try:
            print(f"[SHARED MEMORY DEBUG] Connecting to camera {camera_index} shared memory")
            print(f"[SHARED MEMORY DEBUG] Received shm_names: {shm_names}")
            
            # The shm_names already come from EnhancedCameraWorker status message
            # Keys: 'frame', 'results', 'pose', 'gui'
            gui_shm_names = shm_names.copy()
            print(f"[SHARED MEMORY DEBUG] GUI-ready shm_names: {gui_shm_names}")
            self.shared_memories[camera_index] = {}

            # Connect to frame shared memory (raw camera frames)
            if 'frame' in gui_shm_names:
                preview_shm = shared_memory.SharedMemory(name=gui_shm_names['frame'])
                # Use actual camera resolution from status data
                # The frame buffer contains full resolution frames from camera worker
                if status_data:
                    camera_resolution = status_data.get('resolution', [1920, 1080])
                    width, height = camera_resolution[0], camera_resolution[1]
                else:
                    # Fallback to default
                    width, height = 1920, 1080
                
                frame_size = height * width * 3
                
                # Debug logging
                logger.info(f"Camera {camera_index} buffer connection:")
                logger.info(f"  Expected resolution: {width}x{height}")
                ring_buffer_size = self.config.get('process_separation', {}).get('ring_buffer_size', 32)
                logger.info(f"  Expected frame size: {frame_size}")
                logger.info(f"  Actual buffer size: {len(preview_shm.buf)}")
                logger.info(f"  Ring buffer size: {ring_buffer_size} frames")
                logger.info(f"  Total frames space: {frame_size * ring_buffer_size}")
                logger.info(f"  Metadata space: {len(preview_shm.buf) - (frame_size * ring_buffer_size)}")
                
                # Calculate ring buffer layout
                # ring_buffer_size already calculated above
                total_frame_space = frame_size * ring_buffer_size
                
                logger.info(f"Camera {camera_index} buffer layout:")
                logger.info(f"  Buffer size: {len(preview_shm.buf)}")
                logger.info(f"  Frame size: {frame_size} ({width}x{height}x3)")
                logger.info(f"  Ring buffer frames: {ring_buffer_size}")
                logger.info(f"  Total frame space: {total_frame_space}")
                logger.info(f"  Metadata starts at: {total_frame_space}")
                
                # Create the ring buffer view (all frames at once)
                try:
                    # Create view for the entire ring buffer
                    ring_buffer_array = np.ndarray(
                        (ring_buffer_size, height, width, 3), 
                        dtype=np.uint8, 
                        buffer=preview_shm.buf[:total_frame_space]
                    )
                    logger.info(f"  Ring buffer array created successfully")
                    
                    logger.info(f"  Ring buffer views created successfully")
                    
                except Exception as e:
                    logger.error(f"  Failed to create buffer views: {e}")
                    raise
                
                self.shared_memories[camera_index]['frame'] = {
                    'shm': preview_shm,
                    'ring_buffer': ring_buffer_array,  # Full ring buffer
                    'resolution': (height, width),
                    'ring_size': ring_buffer_size
                }

            # Connect to results shared memory (landmarks + blendshapes + bboxes)
            print(f"[SHARED MEMORY DEBUG] Checking for results key in gui_shm_names...")
            print(f"[SHARED MEMORY DEBUG] 'results' in gui_shm_names: {'results' in gui_shm_names}")
            if 'results' in gui_shm_names:
                print(f"[SHARED MEMORY DEBUG] Found results key, connecting to: {gui_shm_names['results']}")
                try:
                    landmark_shm = shared_memory.SharedMemory(name=gui_shm_names['results'])
                    print(f"[SHARED MEMORY DEBUG] Successfully connected to results shared memory")
                    logger.info(f"  Results buffer size: {landmark_shm.size}")
                    
                    # Define ROI metadata structure to match gpu_pipeline.py
                    roi_metadata_dtype = np.dtype([
                        ('x1', 'float32'),      # ROI top-left x in capture frame coordinates
                        ('y1', 'float32'),      # ROI top-left y in capture frame coordinates
                        ('scale_x', 'float32'), # Scale from ROI to capture frame (x-axis)
                        ('scale_y', 'float32'), # Scale from ROI to capture frame (y-axis)
                        ('confidence', 'float32'), # Detection confidence
                        ('track_id', 'int32'),  # Persistent track ID
                        ('is_detection', 'int8'), # 1 if from detection, 0 if from tracking
                        ('track_age', 'int32')  # Age of track in frames
                    ])
                    
                    # CRITICAL FIX: Use centralized max_faces calculation identical to camera worker
                    max_faces = self._calculate_max_faces(self.config)
                    print(f"[BUFFER SYNC FIX] Using centralized max_faces={max_faces} identical to camera worker")

                    # Use centralized ResultsBufferLayout for consistent size calculations
                    from core.buffer_management.coordinator import BufferCoordinator
                    coordinator = BufferCoordinator(
                        camera_count=1,
                        config=self.config,
                        create_coordinator_info=False
                    )
                    results_layout = coordinator.get_layout('results', max_faces=max_faces)

                    landmark_size = results_layout.landmarks_size  # From ResultsBufferLayout
                    roi_metadata_size = max_faces * roi_metadata_dtype.itemsize
                    total_expected = landmark_size + roi_metadata_size + results_layout.metadata_size
                    
                    logger.info(f"  Expected landmarks space: {landmark_size}")
                    logger.info(f"  Expected ROI metadata: {roi_metadata_size}")
                    logger.info(f"  Expected total: {total_expected}")
                    
                    if landmark_shm.size < total_expected:
                        logger.warning(f"  Landmarks buffer too small: {landmark_shm.size} < {total_expected}, skipping")
                    else:
                        # Create metadata view
                        metadata_view = self._create_metadata_view(landmark_shm.buf[landmark_size + roi_metadata_size:])
                        
                        # DEBUG: Log metadata buffer location and initial state
                        metadata_offset = landmark_size + roi_metadata_size
                        # print(f"[METADATA DEBUG] Metadata buffer offset: {metadata_offset}")
                        # print(f"[METADATA DEBUG] Buffer size: {landmark_shm.size}")
                        # print(f"[METADATA DEBUG] Initial metadata: ready={metadata_view['ready']}, frame_id={metadata_view['frame_id']}")
                        
                        self.shared_memories[camera_index]['results'] = {
                            'shm': landmark_shm,
                            'array': np.ndarray((max_faces, 478, 3), dtype=np.float32,
                                               buffer=landmark_shm.buf[:landmark_size]),
                            'roi_metadata': np.ndarray(max_faces, dtype=roi_metadata_dtype,
                                                     buffer=landmark_shm.buf[landmark_size:landmark_size + roi_metadata_size]),
                            'metadata': metadata_view
                        }
                        logger.info(f"  Results buffer connected successfully")
                        print(f"[SHARED MEMORY DEBUG] Results buffer connected successfully")
                except Exception as landmark_error:
                    logger.error(f"  Failed to connect results buffer: {landmark_error}")
                    print(f"[SHARED MEMORY DEBUG] FAILED to connect results buffer: {landmark_error}")
                    # Continue without results buffer
            else:
                print(f"[SHARED MEMORY DEBUG] No results key in gui_shm_names - results buffer will not be available")
            
            # Connect to ROI shared memory (contains bounding boxes from detection)
            if 'rois' in gui_shm_names:
                try:
                    roi_shm = shared_memory.SharedMemory(name=gui_shm_names['rois'])
                    logger.info(f"  ROI buffer size: {roi_shm.size}")
                    
                    # CRITICAL FIX: Use BufferCoordinator's layout for consistent metadata offset
                    # This ensures detection process and GUI read from the same locations
                    roi_layout = self.coordinator.get_roi_buffer_layout(camera_index)
                    
                    self.shared_memories[camera_index]['rois'] = {
                        'shm': roi_shm,
                        'roi_size': roi_layout['roi_size'],
                        'max_faces': self._calculate_max_faces(self.config),
                        'buffer_size': roi_layout['roi_buffer_size'],
                        'metadata_offset': roi_layout['metadata_offset']  # Use BufferCoordinator's offset
                    }
                    logger.info(f"  ROI buffer connected with metadata_offset={roi_layout['metadata_offset']}")
                except Exception as roi_error:
                    logger.error(f"  Failed to connect ROI buffer: {roi_error}")
                    # Continue without ROI buffer
            
            # Connect to GUI buffer (store buffer name for GUIBufferManager)
            if 'gui' in gui_shm_names:
                try:
                    # Just store the buffer name - GUIBufferManager will handle the actual connection
                    self.shared_memories[camera_index]['gui'] = {
                        'name': gui_shm_names['gui']
                    }
                    logger.info(f"  GUI buffer name stored: {gui_shm_names['gui']}")
                except Exception as gui_error:
                    logger.error(f"  Failed to store GUI buffer info: {gui_error}")
            else:
                logger.warning(f"  No GUI buffer provided for camera {camera_index}")

            # Connect to pose buffer for body pose detection
            if 'pose' in gui_shm_names:
                try:
                    pose_shm = shared_memory.SharedMemory(name=gui_shm_names['pose'])
                    logger.info(f"  Pose buffer size: {pose_shm.size}")

                    # Get pose buffer layout from coordinator
                    pose_layout = self.coordinator.get_pose_buffer_layout(camera_index)

                    self.shared_memories[camera_index]['pose'] = {
                        'shm': pose_shm,
                        'name': gui_shm_names['pose'],
                        'layout': pose_layout
                    }
                    logger.info(f"  Pose buffer connected: {gui_shm_names['pose']}")
                except Exception as pose_error:
                    logger.error(f"  Failed to connect pose buffer: {pose_error}")
                    # Continue without pose buffer - it's optional
            else:
                logger.debug(f"  No pose buffer provided for camera {camera_index} (pose detection may be disabled)")

            # Connect to detection buffer (SCRFD) - for GUI grid debug access
            if 'detection' in gui_shm_names:
                try:
                    detection_name = gui_shm_names['detection']
                    detection_shm = shared_memory.SharedMemory(name=detection_name)
                    self.shared_memories[camera_index]['detection'] = {
                        'shm': detection_shm,
                        'name': detection_name
                    }
                    logger.info(f"  Detection buffer connected: {detection_name}")
                except FileNotFoundError:
                    logger.warning(f"  Detection buffer not found: {detection_name}")
                except Exception as detection_error:
                    logger.error(f"  Failed to connect detection buffer: {detection_error}")
            else:
                logger.debug(f"  No detection buffer provided for camera {camera_index}")

            # Connect to embedding buffer (ArcFace) - for future GUI features
            if 'embedding' in gui_shm_names:
                try:
                    embedding_name = gui_shm_names['embedding']
                    embedding_shm = shared_memory.SharedMemory(name=embedding_name)
                    self.shared_memories[camera_index]['embedding'] = {
                        'shm': embedding_shm,
                        'name': embedding_name
                    }
                    logger.info(f"  Embedding buffer connected: {embedding_name}")
                except FileNotFoundError:
                    logger.warning(f"  Embedding buffer not found: {embedding_name}")
                except Exception as embedding_error:
                    logger.error(f"  Failed to connect embedding buffer: {embedding_error}")
            else:
                logger.debug(f"  No embedding buffer provided for camera {camera_index}")

            logger.info(f"Connected to shared memory for camera {camera_index}")
            
        except Exception as e:
            logger.error(f"Failed to connect shared memory for camera {camera_index}: {e}")
    
    
    def _calculate_max_faces(self, config: Dict[str, Any]) -> int:
        """Centralized max_faces calculation - MUST be identical across all components."""
        participant_count = config.get('startup_mode', {}).get('participant_count', 1)
        max_faces_config = config.get('process_separation', {}).get('max_faces_per_frame', 10)
        gpu_max_batch = config.get('advanced_detection', {}).get('gpu_settings', {}).get('max_batch_size', 8)

        max_faces = min(
            participant_count,      # Use participant_count directly
            max_faces_config,       # Configuration limit
            gpu_max_batch,          # GPU hardware limit
            8                       # Absolute maximum for memory safety
        )

        return max_faces

    def _create_metadata_view(self, buffer):
        """Create structured metadata view with tracking support."""
        metadata_dtype = np.dtype([
            ('frame_id', 'int32'),
            ('timestamp_ms', 'int64'),
            ('n_faces', 'int32'),
            ('ready', 'int8'),
            ('is_detection_frame', 'int8'),  # 1 if full detection, 0 if tracking only
            ('track_count', 'int32'),  # Number of active tracks
            ('processing_time_ms', 'float32'),
            ('gpu_memory_used_mb', 'float32'),
        ])
        return np.ndarray(1, dtype=metadata_dtype, buffer=buffer)[0]

    def start_camera_async(self, camera_index: int,
                          on_ready=None,
                          on_failed=None,
                          gui_after_func=None,
                          display_only: bool = False):
        """
        Start a camera worker asynchronously without blocking GUI thread.

        This method returns immediately after starting the worker process.
        Camera handshake and initialization happens in a background thread.
        Callbacks are invoked on the GUI thread when camera becomes ready or fails.

        Args:
            camera_index: Camera device index
            on_ready: Callback function(camera_index, buffer_names, resolution) called when camera is ready
            on_failed: Callback function(camera_index, error_message) called if startup fails
            gui_after_func: GUI's after() method for scheduling callbacks on main thread
            display_only: If True, camera will only display frames without running pose/face detection

        Returns:
            bool: True if worker process started (not necessarily ready yet)
        """
        import threading

        # Store callbacks
        if on_ready:
            self.camera_ready_callbacks[camera_index] = (on_ready, gui_after_func)
        if on_failed:
            self.camera_failed_callbacks[camera_index] = (on_failed, gui_after_func)

        def handshake_worker():
            """Background thread for camera handshake."""
            try:
                logger.info(f"[ASYNC] Starting handshake for camera {camera_index} in background (display_only={display_only})...")
                success, buffer_names, resolution = self.start_camera(camera_index, display_only=display_only)

                if success:
                    logger.info(f"[ASYNC] Camera {camera_index} ready with buffers: {list(buffer_names.keys())}, resolution: {resolution}")
                    # Invoke ready callback on GUI thread and remove it
                    if camera_index in self.camera_ready_callbacks:
                        callback, after_func = self.camera_ready_callbacks.pop(camera_index)  # FIX #5: Remove after use
                        # Also clean up failed callback since we succeeded
                        self.camera_failed_callbacks.pop(camera_index, None)

                        if after_func:
                            after_func(0, lambda: callback(camera_index, buffer_names, resolution))
                        else:
                            callback(camera_index, buffer_names, resolution)
                else:
                    logger.error(f"[ASYNC] Camera {camera_index} handshake failed")
                    # Invoke failed callback on GUI thread and remove it
                    if camera_index in self.camera_failed_callbacks:
                        callback, after_func = self.camera_failed_callbacks.pop(camera_index)  # FIX #5: Remove after use
                        # Also clean up ready callback since we failed
                        self.camera_ready_callbacks.pop(camera_index, None)

                        error_msg = "Handshake timeout or initialization failed"
                        if after_func:
                            after_func(0, lambda: callback(camera_index, error_msg))
                        else:
                            callback(camera_index, error_msg)

            except Exception as e:
                logger.error(f"[ASYNC] Camera {camera_index} startup exception: {e}", exc_info=True)
                # Invoke failed callback and remove it
                if camera_index in self.camera_failed_callbacks:
                    callback, after_func = self.camera_failed_callbacks.pop(camera_index)  # FIX #5: Remove after use
                    # Also clean up ready callback
                    self.camera_ready_callbacks.pop(camera_index, None)

                    if after_func:
                        after_func(0, lambda: callback(camera_index, str(e)))
                    else:
                        callback(camera_index, str(e))

        # Start background thread
        thread = threading.Thread(target=handshake_worker, daemon=True, name=f"CameraHandshake-{camera_index}")
        thread.start()
        logger.info(f"[ASYNC] Camera {camera_index} handshake thread started")

        return True  # Worker process start initiated

    def stop_camera(self, camera_index: int):
        """
        Stop a camera worker with graceful shutdown.

        IMPROVED: Increased timeout from 5s to 15s to allow camera source cleanup,
        especially important for ZMQ cameras that need to close sockets and terminate contexts.
        """
        if camera_index not in self.workers:
            return

        logger.info(f"Stopping camera {camera_index}...")

        # FIX #5: Clean up any pending async callbacks before stopping
        self.camera_ready_callbacks.pop(camera_index, None)
        self.camera_failed_callbacks.pop(camera_index, None)
        logger.debug(f"Cleaned up pending callbacks for camera {camera_index}")

        # Send stop command
        if camera_index in self.control_queues:
            try:
                self.control_queues[camera_index].put({'command': 'stop'}, timeout=1)
                logger.debug(f"Stop command sent to camera {camera_index}")
            except Exception as e:
                logger.warning(f"Failed to send stop command to camera {camera_index}: {e}")

        # Wait for worker to finish (increased timeout for proper cleanup)
        worker = self.workers[camera_index]
        logger.info(f"Waiting for camera {camera_index} worker to finish (timeout: 15s)...")
        worker.join(timeout=15)  # Increased from 5s to allow ZMQ cleanup

        if worker.is_alive():
            logger.warning(f"Camera {camera_index} did not stop gracefully after 15s, terminating...")
            worker.terminate()
            worker.join(timeout=5)

            if worker.is_alive():
                logger.error(f"Camera {camera_index} did not respond to terminate, killing...")
                worker.kill()
                worker.join(timeout=2)
        else:
            logger.info(f"Camera {camera_index} stopped gracefully")

        # Cleanup shared memory
        if camera_index in self.shared_memories:
            for shm_type, shm_data in self.shared_memories[camera_index].items():
                try:
                    shm_data['shm'].close()
                except Exception as e:
                    logger.debug(f"Error closing shared memory for camera {camera_index}: {e}")
            del self.shared_memories[camera_index]

        # Remove from tracking
        del self.workers[camera_index]
        if camera_index in self.control_queues:
            del self.control_queues[camera_index]

        logger.info(f"Camera {camera_index} stopped and cleaned up")
    
    def register_track_participant(self, camera_index: int, track_id: int, participant_id: int, embedding=None):
        """
        Register track-participant association via hybrid IPC system.
        
        Args:
            camera_index: Camera device index
            track_id: Track ID from detection/tracking
            participant_id: Global participant ID assigned by participant manager
            embedding: Optional face embedding data (triggers legacy mp.Queue for large data)
        """
        try:
            if embedding is not None:
                # Use legacy mp.Queue for large embedding data (>2KB limit)
                if camera_index in self.control_queues:
                    self.control_queues[camera_index].put({
                        'command': 'register_track_participant',
                        'track_id': track_id,
                        'participant_id': participant_id,
                        'embedding': embedding  # Include embedding data
                    })
                    logger.info(f"Sent track registration with embedding via legacy queue: "
                               f"camera={camera_index}, track={track_id}, participant={participant_id}")
                else:
                    logger.error(f"No control queue available for camera {camera_index}")
            else:
                # Use CommandBuffer system for small data without embedding
                command_id = self.send_camera_command(
                    camera_index, 
                    'register_track_participant',
                    {
                        'track_id': track_id,
                        'participant_id': participant_id,
                        'camera_index': camera_index
                    }
                )
                
                if command_id:
                    # Optionally wait for acknowledgment with brief timeout
                    ack = self.get_command_status(camera_index, command_id)
                    if ack and ack['success']:
                        logger.info(f"Track registration confirmed: camera={camera_index}, track={track_id}, participant={participant_id}")
                    else:
                        logger.error(f"Track registration failed via CommandBuffer: camera={camera_index}, track={track_id}")
                else:
                    logger.error(f"Failed to send track registration command for camera {camera_index}")
                
        except Exception as e:
            logger.error(f"Error in track-participant registration: {e}")
    
    def update_participant_embedding(self, camera_index: int, participant_id: int, embedding: np.ndarray):
        """
        Update participant embedding for recovery matching.
        
        Args:
            camera_index: Camera device index
            participant_id: Global participant ID
            embedding: Face embedding vector
        """
        if camera_index not in self.control_queues:
            logger.warning(f"Cannot update embedding for inactive camera {camera_index}")
            return
        
        try:
            # Send embedding update command via legacy mp.Queue (required for large embedding data)
            # CommandBuffer system has 2KB limit, but embeddings are ~11KB with JSON overhead
            # Convert numpy array to list for pickling across processes
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            self.control_queues[camera_index].put({
                'command': 'update_participant_embedding',
                'participant_id': participant_id,
                'embedding': embedding_list
            })
            logger.info(f"Sent participant embedding update to camera {camera_index}: "
                       f"participant {participant_id} (via legacy queue - required for large data)")
        except Exception as e:
            logger.error(f"Failed to send participant embedding update: {e}")
    
    def stop_camera_workers(self, camera_indices: List[int]) -> Dict[int, bool]:
        """
        Stop multiple camera workers.
        
        Args:
            camera_indices: List of camera device indices to stop
            
        Returns:
            Dict[int, bool]: Mapping of camera_index -> success status
        """
        results = {}
        
        for camera_index in camera_indices:
            if camera_index in self.workers:
                self.stop_camera(camera_index)
                results[camera_index] = True
            else:
                results[camera_index] = False
        
        return results
    
    def stop_all(self):
        """Stop all camera workers and cleanup GUI command buffers."""
        camera_indices = list(self.workers.keys())
        for camera_index in camera_indices:
            self.stop_camera(camera_index)
        
        # Cleanup GUI command buffers
        self._cleanup_gui_command_buffers()
    
    def _cleanup_gui_command_buffers(self):
        """Clean up GUI command buffers."""
        try:
            for camera_idx, buffers in self.gui_command_buffers.items():
                try:
                    if 'gui_to_camera' in buffers:
                        buffers['gui_to_camera'].cleanup()
                    if 'camera_to_gui' in buffers:
                        buffers['camera_to_gui'].cleanup()
                    logger.debug(f"Cleaned up GUI command buffers for camera {camera_idx}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup GUI command buffers for camera {camera_idx}: {e}")
            
            self.gui_command_buffers.clear()
            logger.info("GUI command buffers cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up GUI command buffers: {e}")
    
    def get_preview_frame(self, camera_index: int) -> Optional[np.ndarray]:
        """
        Get preview frame from shared memory if available.
        Uses direct write-index access for O(1) performance.
        
        Returns:
            np.ndarray or None: Preview frame if ready
        """
        # Try direct index access first for performance
        frame = self._get_preview_frame_direct(camera_index)
        if frame is not None:
            return frame
            
        # Fallback to scanning method if direct access fails
        return self._get_preview_frame_scan(camera_index)
    
    def _get_preview_frame_direct(self, camera_index: int) -> Optional[np.ndarray]:
        """
        Get preview frame using direct write-index access (O(1) performance).
        This eliminates the need to scan all buffer slots.
        """
        if camera_index not in self.shared_memories:
            return None
            
        if 'frame' not in self.shared_memories[camera_index]:
            return None

        # PERFORMANCE FIX: Cache buffer layout to avoid repeated calculations
        if not hasattr(self, '_cached_layouts'):
            self._cached_layouts = {}

        preview_data = self.shared_memories[camera_index]['frame']
        shm = preview_data['shm']
        
        layout_key = f"{camera_index}_{preview_data['resolution']}"
        
        try:
            # Get buffer layout for consistent access
            height, width = preview_data['resolution']
            
            # Calculate layout using BufferCoordinator (FIXED - no more missing method call)
            if layout_key in self._cached_layouts:
                gui_frame_offsets, gui_metadata_offset, gui_buffer_size = self._cached_layouts[layout_key]
            else:
                # Direct calculation using BufferCoordinator values
                buffer_sizes = self.coordinator.get_buffer_sizes()
                detection_ring_size = buffer_sizes['ring_buffer_size']
                gui_buffer_size = buffer_sizes['gui_buffer_size']
                detection_buffer_size = frame_size * detection_ring_size
                detection_metadata_size = detection_ring_size * 64  # Frame metadata: 8 int64 = 64 bytes
                gui_frame_offsets = [detection_buffer_size + detection_metadata_size + i * frame_size for i in range(gui_buffer_size)]
                gui_metadata_offset = gui_frame_offsets[-1] + frame_size

                # VALIDATION: Check if calculated offsets are valid for actual buffer size
                actual_buffer_size = shm.size
                max_offset = gui_metadata_offset + (gui_buffer_size * 64)  # Metadata for all slots

                if max_offset > actual_buffer_size:
                    logger.error(f"[BUFFER LAYOUT ERROR] Calculated offset {max_offset} exceeds buffer size {actual_buffer_size}")
                    logger.error(f"  Resolution: {resolution}, Frame size: {frame_size}")
                    logger.error(f"  Detection buffer: {detection_buffer_size}, Metadata: {detection_metadata_size}")
                    logger.error(f"  GUI metadata offset: {gui_metadata_offset}")
                    # Clear any wrong cached layouts
                    if layout_key in self._cached_layouts:
                        del self._cached_layouts[layout_key]
                    return None

                self._cached_layouts[layout_key] = (gui_frame_offsets, gui_metadata_offset, gui_buffer_size)
            
            # Read write index from shared memory coordination area
            # Write index is stored at offset 0 in the buffer (BufferCoordinator standard)
            write_idx_offset = 0  # Write index always at offset 0 per BufferCoordinator
            write_idx_array = np.frombuffer(
                shm.buf,
                dtype=np.int64,
                count=1,
                offset=write_idx_offset
            )
            write_idx = int(write_idx_array[0])
            
            if write_idx <= 0:
                return None
                
            # Calculate current slot from write index
            current_slot = (write_idx - 1) % gui_buffer_size
            
            # Read metadata for current slot to verify it's ready
            metadata_offset = gui_metadata_offset + (current_slot * 64)
            metadata = np.frombuffer(
                shm.buf,
                dtype=np.int64,  # FIXED: Use int64 to match BufferCoordinator
                count=8,         # FIXED: 8 int64 values = 64 bytes
                offset=metadata_offset
            )
            
            frame_id = int(metadata[0])
            ready_flag = int(metadata[3])
            
            if ready_flag != 1 or frame_id <= 0:
                return None
            
            # NEW: Store frame_id for overlay synchronization
            if not hasattr(self, '_last_frame_ids'):
                self._last_frame_ids = {}
            self._last_frame_ids[camera_index] = frame_id
                
            # Read frame directly from the current slot
            frame_size = height * width * 3
            frame_offset = gui_frame_offsets[current_slot]
            
            frame_array = np.frombuffer(
                shm.buf,
                dtype=np.uint8,
                count=frame_size,
                offset=frame_offset
            ).reshape(height, width, 3)
            
            # CRITICAL FIX: Create independent copy BEFORE clearing ready flag
            frame = np.array(frame_array, copy=True, order='C')
            
            # Verify the copy is independent
            if not frame.flags.owndata:
                logger.error(f"[MEMORY ERROR] Frame copy failed to create independent memory for camera {camera_index}")
                return None
            
            # NOW safe to clear ready flag after ensuring we have an independent copy
            metadata[3] = 0
            
            # Update access statistics
            if hasattr(self, '_direct_access_stats'):
                self._direct_access_stats['gui_direct_reads'] += 1
            
            # DEBUG: Log frame-overlay coordination
            if not hasattr(self, '_debug_sync_counter'):
                self._debug_sync_counter = {}
            if camera_index not in self._debug_sync_counter:
                self._debug_sync_counter[camera_index] = 0
            self._debug_sync_counter[camera_index] += 1
                
            if self._debug_sync_counter[camera_index] % 30 == 0:
                logger.info(f"[SYNC DEBUG] Camera {camera_index}: frame_id={frame_id}, "
                           f"direct_reads={self._direct_access_stats.get('gui_direct_reads', 0)}, "
                           f"fallbacks={self._direct_access_stats.get('gui_fallbacks', 0)}")
            
            return frame
            
        except Exception as e:
            # Fall back to scanning on any error
            if hasattr(self, '_direct_access_stats'):
                self._direct_access_stats['gui_fallbacks'] += 1
            return None
    
    def get_all_camera_data(self, camera_index: int) -> Optional[Dict]:
        """
        PERFORMANCE FIX: Get all camera data in a single optimized call.
        Reduces sequential buffer access overhead and improves FPS consistency.
        
        Returns:
            Dict with 'frame', 'landmarks', 'detection_boxes' or None if unavailable
        """
        if camera_index not in self.shared_memories:
            return None
            
        try:
            # Get all data in rapid succession to minimize access overhead
            frame = self._get_preview_frame_direct(camera_index)
            landmarks = self.get_landmarks(camera_index) 
            detection_boxes = self._get_detection_boxes_direct(camera_index)
            
            # Fallback to scanning methods if direct access fails
            if frame is None:
                frame = self._get_preview_frame_scan(camera_index)
                if hasattr(self, '_direct_access_stats'):
                    self._direct_access_stats['gui_fallbacks'] += 1
                    
            if detection_boxes is None:
                detection_boxes = self._get_detection_boxes_scan(camera_index)
                if hasattr(self, '_direct_access_stats'):
                    self._direct_access_stats['roi_fallbacks'] += 1
            
            # Return None if no frame available (core requirement)
            if frame is None:
                return None
                
            return {
                'frame': frame,
                'landmarks': landmarks,
                'detection_boxes': detection_boxes,
                'camera_index': camera_index
            }
            
        except Exception as e:
            logger.error(f"Error getting camera data for {camera_index}: {e}")
            # Fallback to individual methods
            return None

    def _get_preview_frame_scan(self, camera_index: int) -> Optional[np.ndarray]:
        """
        Legacy scanning method for preview frames.
        Used as fallback when direct index access is unavailable.
        """
        if camera_index not in self.shared_memories:
            logger.debug(f"Camera {camera_index} not in shared_memories")
            return None
        
        if 'frame' not in self.shared_memories[camera_index]:
            logger.debug(f"No frame in shared_memories for camera {camera_index}")
            return None

        preview_data = self.shared_memories[camera_index]['frame']
        ring_buffer = preview_data['ring_buffer']
        ring_size = preview_data['ring_size']
        shm = preview_data['shm']
        
        # Debug: Log ready status periodically
        if not hasattr(self, '_preview_debug_counter'):
            self._preview_debug_counter = {}
        if camera_index not in self._preview_debug_counter:
            self._preview_debug_counter[camera_index] = 0
        
        self._preview_debug_counter[camera_index] += 1
        
        # Read from preview ring buffer
        height, width = preview_data['resolution']
        frame_size = height * width * 3
        
        # Debug buffer structure
        if camera_index == 0 and self._preview_debug_counter[camera_index] % 100 == 0:
            print(f"[PREVIEW DEBUG] Resolution: {width}x{height}, frame_size: {frame_size}")
            print(f"[PREVIEW DEBUG] preview_data keys: {list(preview_data.keys())}")
        
        # Use BufferCoordinator layout for consistent buffer access
        try:
            layout = self.coordinator.get_buffer_layout(camera_index, (width, height))
            gui_frame_offsets = layout['gui_frame_offsets']
            gui_metadata_offset = layout['gui_metadata_offset']
            gui_buffer_size = len(gui_frame_offsets)
            
            # GUI buffer layout debugging (logging removed)
                
        except Exception as e:
            logger.error(f"Failed to get buffer layout: {e}")
            # Fallback using BufferCoordinator calculations (FIXED - no more hardcoded values)
            buffer_sizes = self.coordinator.get_buffer_sizes()
            detection_ring_size = buffer_sizes['ring_buffer_size']
            gui_buffer_size = buffer_sizes['gui_buffer_size']
            detection_buffer_size = frame_size * detection_ring_size
            # CRITICAL FIX: Use coordinator's metadata size instead of hardcoded 64
            detection_metadata_size = detection_ring_size * 64  # Frame metadata: 8 int64 = 64 bytes
            gui_frame_offsets = [detection_buffer_size + detection_metadata_size + i * frame_size for i in range(gui_buffer_size)]
            gui_metadata_offset = gui_frame_offsets[-1] + frame_size
        
        # Scan preview ring buffer positions to find the latest ready frame
        latest_frame_id = -1
        latest_ready_idx = -1
        
        # Check GUI buffer slots using BufferCoordinator layout  
        if camera_index == 0 and self._preview_debug_counter[camera_index] % 30 == 0:
            print(f"[RING SCAN DEBUG] Scanning {gui_buffer_size} slots (range(0, {gui_buffer_size}))")
            
        for ring_idx in range(gui_buffer_size):
            metadata_offset = gui_metadata_offset + (ring_idx * 64)  # 64 bytes per metadata slot
            
            if camera_index == 0 and self._preview_debug_counter[camera_index] % 30 == 0:
                print(f"[RING SCAN DEBUG] Checking slot {ring_idx}, metadata_offset={metadata_offset}")
            
            try:
                metadata = np.frombuffer(
                    shm.buf,
                    dtype=np.int64,  # FIXED: Use int64 to match BufferCoordinator
                    count=8,   # FIXED: 8 int64 values = 64 bytes
                    offset=metadata_offset
                )
                
                frame_id = int(metadata[0])
                ready_flag = int(metadata[3])  # For preview frames, ready flag is at position 3
                
                # Debug first few slots
                if camera_index == 0 and ring_idx < 3 and self._preview_debug_counter[camera_index] % 30 == 0:
                    print(f"[PREVIEW DEBUG] Slot {ring_idx}: frame_id={frame_id}, ready_flag={ready_flag}, offset={metadata_offset}")
                
                if ready_flag == 1 and frame_id > latest_frame_id:
                    latest_frame_id = frame_id
                    latest_ready_idx = ring_idx
                    
            except Exception as e:
                if ring_idx == 0:  # Only print error for first slot
                    print(f"[PREVIEW ERROR] Failed to read preview metadata slot {ring_idx} for camera {camera_index}: {e}")
                continue
        
        # Debug logging every 100 calls
        if self._preview_debug_counter[camera_index] % 100 == 0:
            logger.info(f"Camera {camera_index} preview scan #{self._preview_debug_counter[camera_index]}: found ready frame {latest_frame_id} at ring_idx {latest_ready_idx}")
        
        # Debug if no ready frame found
        if latest_ready_idx < 0 and camera_index == 0 and self._preview_debug_counter[camera_index] % 30 == 0:
            print(f"[PREVIEW DEBUG] No ready frame found for camera {camera_index}")
        
        # If we found a ready frame, return it
        if latest_ready_idx >= 0:
            # Read frame from GUI buffer at the found index using BufferCoordinator layout
            if latest_ready_idx < len(gui_frame_offsets):
                frame_offset = gui_frame_offsets[latest_ready_idx]
            else:
                # Fallback if index is out of range
                frame_offset = gui_frame_offsets[0]
                logger.warning(f"GUI frame index {latest_ready_idx} out of range, using slot 0")
            try:
                frame_array = np.frombuffer(
                    shm.buf,
                    dtype=np.uint8,
                    count=frame_size,
                    offset=frame_offset
                ).reshape(height, width, 3)
                
                frame = frame_array.copy()
                
                # Debug successful frame read
                if camera_index == 0 and self._preview_debug_counter[camera_index] % 30 == 0:
                    print(f"[PREVIEW DEBUG] Successfully read frame from slot {latest_ready_idx}, offset={frame_offset}, shape: {frame.shape}, min={frame.min()}, max={frame.max()}")
                    
            except Exception as e:
                logger.error(f"Failed to read preview frame: {e}")
                print(f"[PREVIEW ERROR] Failed to read frame at offset {frame_offset}: {e}")
                return None
            
            # Debug: Check frame content
            if self._preview_debug_counter[camera_index] % 100 == 0:
                logger.info(f"Camera {camera_index} returning frame: frame_id={latest_frame_id}, ring_idx={latest_ready_idx}, shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")
            
            # Clear ready flag to indicate frame has been consumed
            # This prevents the GUI from showing the same frame multiple times
            try:
                metadata_offset = gui_metadata_offset + (latest_ready_idx * 64)
                metadata = np.frombuffer(
                    shm.buf,
                    dtype=np.int64,  # FIXED: Use int64 to match BufferCoordinator
                    count=8,         # FIXED: 8 int64 values = 64 bytes
                    offset=metadata_offset
                )
                metadata[3] = 0  # Clear ready flag
            except Exception as e:
                logger.debug(f"Failed to clear ready flag: {e}")
            
            return frame
        
        return None
    
    def get_landmarks(self, camera_index: int) -> Optional[Dict]:
        """
        Get landmark data from shared memory if available.
        
        Returns:
            Dict or None: Landmark data with metadata and tracking info
        """
        # Shared memory status check (debug logging removed)
        
        if camera_index not in self.shared_memories:
            # Camera not in shared memories (debug logging removed)
            return None

        if 'results' not in self.shared_memories[camera_index]:
            # Results not in shared memory (debug logging removed)
            return None

        landmark_data = self.shared_memories[camera_index]['results']
        
        # Initialize landmark cache if needed
        if not hasattr(self, '_landmark_cache'):
            self._landmark_cache = {}
        
        # Check if data is ready
        try:
            ready_flag = landmark_data['metadata']['ready']
            if camera_index == 0:  # Debug for camera 0
                # Access numpy structured array fields directly
                frame_id = landmark_data['metadata']['frame_id']
                n_faces = landmark_data['metadata']['n_faces']
                timestamp = landmark_data['metadata']['timestamp_ms']
                # Landmarks metadata read (debug logging removed)
        except Exception as e:
            if camera_index == 0:
                # Failed to read ready flag (debug logging removed)
                pass
            return None
        
        if ready_flag == 1:
            # Get metadata
            n_faces = landmark_data['metadata']['n_faces']
            frame_id = landmark_data['metadata']['frame_id']
            is_detection_frame = landmark_data['metadata']['is_detection_frame']
            track_count = landmark_data['metadata']['track_count']
            processing_time_ms = landmark_data['metadata']['processing_time_ms']
            
            # Validate frame_id is reasonable (not zero, not negative)
            if frame_id <= 0:
                if camera_index == 0:
                    # Invalid frame_id (debug logging removed)
                    pass
                return self._landmark_cache.get(camera_index)
            
            # Copy landmark data
            landmarks = landmark_data['array'][:n_faces].copy()
            
            # Copy ROI metadata (includes track IDs and transform info)
            roi_metadata = None
            if 'roi_metadata' in landmark_data:
                roi_metadata = landmark_data['roi_metadata'][:n_faces].copy()
            
            # Create landmark data dict
            current_data = {
                'frame_id': frame_id,
                'n_faces': n_faces,
                'landmarks': landmarks,
                'roi_metadata': roi_metadata,
                'is_detection_frame': bool(is_detection_frame),
                'track_count': track_count,
                'processing_time_ms': processing_time_ms
            }
            
            # Check if this is new data compared to our cache
            cached_data = self._landmark_cache.get(camera_index)
            if cached_data is None or cached_data['frame_id'] != frame_id:
                # New data - update cache but DON'T immediately clear ready flag
                # Let the processes manage their own ready flag lifecycle
                self._landmark_cache[camera_index] = current_data
                if camera_index == 0:
                    # New data cached (debug logging removed)
                    pass
                
            return current_data
        else:
            if camera_index == 0:  # Debug for camera 0
                # Data not ready, using cached (debug logging removed)
                pass
        
        # No new data available, return cached data if available
        cached_result = self._landmark_cache.get(camera_index)
        if camera_index == 0:
            pass  # Returning cached data (debug logging removed)
        return cached_result
    
    def process_metadata(self) -> List[Dict]:
        """Metadata processing removed - using CommandBuffer communication only."""
        return []
    
    def _get_command_with_retry(self, buffer, camera_idx: int, max_attempts: int = 1, base_timeout: float = 0.001) -> Dict:
        """
        Get command with retry logic and exponential backoff.
        Implements recommendation from comprehensive code review.
        """
        self._comm_stats['attempts'] += 1
        
        for attempt in range(max_attempts):
            timeout = max(0.0, base_timeout) * (2 ** attempt)  # 0-1 ms typical when active
            logger.debug(f"[CameraWorkerManager] Retry attempt {attempt + 1}/{max_attempts} for camera {camera_idx}, timeout={timeout:.3f}s")
            
            try:
                command = buffer.get_command(timeout=timeout)
                if command is not None:
                    self._comm_stats['successes'] += 1
                    logger.info(f"[CameraWorkerManager] ✓ Command received on attempt {attempt + 1} for camera {camera_idx}")
                    return command
                else:
                    logger.debug(f"[CameraWorkerManager] Timeout on attempt {attempt + 1} for camera {camera_idx}")
                    
            except Exception as e:
                self._comm_stats['errors'] += 1
                logger.error(f"[CameraWorkerManager] Error on attempt {attempt + 1} for camera {camera_idx}: {e}")
                
        self._comm_stats['timeouts'] += 1
        # This is normal operation - no commands available, not an error
        logger.debug(f"[CameraWorkerManager] No commands available for camera {camera_idx} (timeout)")
        return None

    def process_status_updates(self) -> List[Dict]:
        """Process all available status updates from CommandBuffers (replaces broken multiprocessing queue)."""
        status_list = []
        
        # Initialize communication statistics tracking
        if not hasattr(self, '_comm_stats'):
            self._comm_stats = {'attempts': 0, 'successes': 0, 'timeouts': 0, 'errors': 0}
        if not hasattr(self, '_command_log_counter'):
            self._command_log_counter = 0
        
        # logger.info(f"[CameraWorkerManager DEBUG] process_status_updates called")
        # logger.info(f"[CameraWorkerManager DEBUG] gui_command_buffers keys: {list(self.gui_command_buffers.keys()) if hasattr(self, 'gui_command_buffers') else 'NO BUFFERS'}")
        
        # State-aware timeout strategy based on active cameras
        active_cameras = len([cam for cam in self.gui_command_buffers.keys() 
                            if hasattr(self, 'shared_memories') and cam in self.shared_memories])
        
        if active_cameras == 0:
            # Startup: be a bit more forgiving but still snappy
            timeout_strategy = {'max_attempts': 2, 'base_timeout': 0.005}
        else:
            # Normal: fast-fail, caller is rate-limited
            timeout_strategy = {'max_attempts': 1, 'base_timeout': 0.001}
        
        logger.debug(f"[CameraWorkerManager] Using timeout strategy: {timeout_strategy} for {active_cameras} active cameras")

        # Read from CommandBuffers (preferred method)
        if hasattr(self, 'gui_command_buffers') and self.gui_command_buffers:
            # Maximum commands to read per camera to prevent blocking
            MAX_COMMANDS_PER_CAMERA = 10
            
            for camera_idx, buffers in self.gui_command_buffers.items():
                # logger.info(f"[CameraWorkerManager DEBUG] Checking camera {camera_idx}, buffers: {list(buffers.keys()) if buffers else 'EMPTY'}")

                if 'camera_to_gui' in buffers:
                    try:
                        camera_to_gui_buffer = buffers['camera_to_gui']
                        # logger.info(f"[CameraWorkerManager DEBUG] Got buffer object for camera {camera_idx}: {camera_to_gui_buffer}, type: {type(camera_to_gui_buffer)}")
                        
                        # Read limited number of commands to prevent blocking GUI
                        commands_read = 0
                        while commands_read < MAX_COMMANDS_PER_CAMERA:
                            # Use state-aware retry logic
                            logger.debug(f"[CameraWorkerManager DEBUG] Attempting to read command {commands_read} from camera {camera_idx}")
                            command = self._get_command_with_retry(camera_to_gui_buffer, camera_idx, **timeout_strategy)
                            
                            if command is None:
                                logger.debug(f"[CameraWorkerManager DEBUG] No more commands for camera {camera_idx} after {commands_read} reads")
                                break
                            
                            logger.info(f"[CameraWorkerManager DEBUG] Got command from camera {camera_idx}: {command}")
                            logger.info(f"[CameraWorkerManager DEBUG] Command structure - type: {command.get('type')}, has payload: {'payload' in command}")
                            commands_read += 1
                                
                            # Check if this is a status command with schema tolerance
                            status = None
                            cmd_type = command.get("type")
                            
                            if cmd_type == "status" and "payload" in command:
                                # New correct schema: {"type": "status", "payload": {...}}
                                status = command["payload"]
                                logger.info(f"[CameraWorkerManager] ✅ Got status via CommandBuffer (new schema): type={status.get('type', 'unknown')}, camera={status.get('camera_index', 'unknown')}")
                            elif cmd_type in ["ready", "error", "ping", "pong"]:
                                # Legacy schema tolerance: treat direct commands as status payloads
                                status = command
                                logger.info(f"[CameraWorkerManager] ✅ Got status via CommandBuffer (legacy schema): type={status.get('type', 'unknown')}, camera={status.get('camera_index', 'unknown')}")
                                logger.warning(f"[CameraWorkerManager] WARNING: Using legacy schema - worker should send type='status' with payload")
                            
                            # Process valid status messages
                            if status:
                                status_list.append(status)
                                logger.info(f"[CameraWorkerManager DEBUG] Status payload keys: {list(status.keys()) if isinstance(status, dict) else 'Not a dict'}")

                                # Cache model initialization messages for GUI display (even after handshake)
                                status_type = status.get('type')
                                cam_idx = status.get('camera_index')
                                if cam_idx is not None and status_type in ('model_loading', 'model_ready'):
                                    hs = self._ensure_handshake_state(cam_idx)
                                    hs["pending_status_updates"].append(status)
                                    logger.info(f"[CameraWorkerManager] 📦 Caching {status_type} status for camera {cam_idx} (for GUI display)")

                                # Special logging for ready status and capture payload
                                if status.get('type') == 'ready':
                                    logger.info(f"[CameraWorkerManager] 🎉 READY STATUS RECEIVED for camera {status.get('camera_index')}")
                                    logger.info(f"[CameraWorkerManager] Ready status data: {status.get('data', {})}")
                                    # Keep a copy for handshake processing (Option A)
                                    cam_for_ready = status.get('camera_index')
                                    if cam_for_ready is not None:
                                        hs = self._ensure_handshake_state(cam_for_ready)
                                        hs["last_ready_payload"] = status
                                        logger.info(f"[CameraWorkerManager] READY payload captured for cam {cam_for_ready}")
                                
                    except Exception as e:
                        logger.error(f"[CameraWorkerManager] Error reading CommandBuffer for camera {camera_idx}: {e}")
                        import traceback
                        logger.error(f"[CameraWorkerManager] Traceback: {traceback.format_exc()}")
                else:
                    logger.warning(f"[CameraWorkerManager DEBUG] No camera_to_gui buffer for camera {camera_idx}")
        
        # Status communication is now CommandBuffer-only (no fallback queue)
        
        # Enhanced diagnostics: Report communication statistics periodically
        if self._comm_stats['attempts'] > 0 and self._comm_stats['attempts'] % 1000 == 0:
            success_rate = (self._comm_stats['successes'] / self._comm_stats['attempts']) * 100
            logger.info(f"[CameraWorkerManager] Communication Stats: "
                       f"Attempts={self._comm_stats['attempts']}, "
                       f"Success Rate={success_rate:.1f}%, "
                       f"Timeouts={self._comm_stats['timeouts']}, "
                       f"Errors={self._comm_stats['errors']}")
        
        if status_list:
            logger.info(f"[CameraWorkerManager] ✅ Returning {len(status_list)} status updates to GUI")
        else:
            # Enhanced reporting when no status updates received
            active_cameras = len([cam for cam in self.gui_command_buffers.keys() if self.gui_command_buffers[cam]])
            if active_cameras > 0:
                logger.debug(f"[CameraWorkerManager] No status updates from {active_cameras} active cameras")
        
        return status_list
    
    def send_command(self, camera_index: int, command: str, data: Dict = None):
        """Send command to specific camera."""
        if camera_index not in self.control_queues:
            logger.warning(f"Camera {camera_index} not found")
            return
        
        msg = {'command': command}
        if data:
            msg.update(data)
        
        try:
            self.control_queues[camera_index].put_nowait(msg)
        except:
            logger.warning(f"Control queue full for camera {camera_index}")
    
    def pause_camera(self, camera_index: int):
        """Pause a camera."""
        self.send_command(camera_index, 'pause')
    
    def resume_camera(self, camera_index: int):
        """Resume a camera."""
        self.send_command(camera_index, 'resume')
    
    def get_stats(self, camera_index: int = None):
        """Request statistics from camera(s)."""
        if camera_index is not None:
            self.send_command(camera_index, 'get_stats')
        else:
            # Request from all cameras
            for cam_idx in self.control_queues:
                self.send_command(cam_idx, 'get_stats')
    
    def is_camera_active(self, camera_index: int) -> bool:
        """Check if a camera worker is active."""
        return camera_index in self.workers and self.workers[camera_index].is_alive()
    
    def get_camera_health(self, camera_index: int) -> Dict:
        """Get camera health status and performance metrics."""
        if camera_index not in self.shared_memories:
            return {'status': 'disconnected', 'error': 'Camera not connected'}
        
        health_info = {
            'status': 'connected',
            'frame_available': 'frame' in self.shared_memories[camera_index],
            'results_available': 'results' in self.shared_memories[camera_index]
        }

        # Get performance metrics from results metadata if available
        if 'results' in self.shared_memories[camera_index]:
            metadata = self.shared_memories[camera_index]['results']['metadata']
            health_info.update({
                'last_frame_id': int(metadata['frame_id']),
                'processing_time_ms': float(metadata['processing_time_ms']),
                'gpu_memory_used_mb': float(metadata['gpu_memory_used_mb']),
                'active_tracks': int(metadata['track_count'])
            })
        
        return health_info
    
    def is_camera_active(self, camera_index: int) -> bool:
        """Check if camera is actively processing frames."""
        if camera_index not in self.workers:
            return False
        
        worker = self.workers[camera_index]
        return worker.is_alive()
    
    def get_detection_boxes(self, camera_index: int) -> Optional[List[Dict]]:
        """
        Get detection bounding boxes from ROI buffer.
        Uses direct write-index access for O(1) performance.
        
        Returns:
            List[Dict] or None: List of face detections with bounding boxes
        """
        # Try direct index access first for performance
        boxes = self._get_detection_boxes_direct(camera_index)
        if boxes is not None:
            return boxes
            
        # Fallback to scanning method if direct access fails
        return self._get_detection_boxes_scan(camera_index)
    
    def _get_detection_boxes_direct(self, camera_index: int) -> Optional[List[Dict]]:
        """
        Get detection boxes using direct write-index access (O(1) performance).
        This eliminates the need to scan all ROI buffer positions.
        """
        if camera_index not in self.shared_memories:
            return None
            
        if 'rois' not in self.shared_memories[camera_index]:
            return None
            
        roi_data = self.shared_memories[camera_index]['rois']
        roi_shm = roi_data['shm']
        metadata_offset = roi_data['metadata_offset']
        roi_buffer_size = roi_data['buffer_size']
        
        try:
            # Read write index from shared memory
            # Write index is stored at offset 0 in the ROI buffer
            write_idx_offset = 0
            write_idx_array = np.frombuffer(
                roi_shm.buf,
                dtype=np.int64,
                count=1,
                offset=write_idx_offset
            )
            write_idx = int(write_idx_array[0])
            
            if write_idx < 0:
                return None
            
            # NEW: Find detection data that matches current frame_id
            target_frame_id = getattr(self, '_last_frame_ids', {}).get(camera_index, -1)
            
            # Scan recent buffer positions to find frame_id match
            best_match_pos = -1
            best_match_frame_id = -1
            
            # Check last few positions for frame_id coordination
            for offset in range(min(4, roi_buffer_size)):  # Check last 4 positions
                check_pos = (write_idx - 1 - offset) % roi_buffer_size
                check_metadata_pos = metadata_offset + check_pos * 1024
                
                try:
                    check_metadata = np.frombuffer(
                        roi_shm.buf,
                        dtype=np.float32,
                        count=256,
                        offset=check_metadata_pos
                    )
                    
                    check_frame_id = int(check_metadata[0])
                    check_ready = int(check_metadata[253])
                    
                    if check_ready == 1 and check_frame_id > 0:
                        if check_frame_id == target_frame_id:
                            # Exact frame_id match - use this position
                            best_match_pos = check_pos
                            best_match_frame_id = check_frame_id
                            break
                        elif best_match_frame_id < check_frame_id:
                            # Better (more recent) match
                            best_match_pos = check_pos  
                            best_match_frame_id = check_frame_id
                except:
                    continue
            
            if best_match_pos < 0:
                # Fallback to latest position if no good match
                current_pos = (write_idx - 1) % roi_buffer_size
            else:
                current_pos = best_match_pos
            
            # Read metadata from current position
            metadata_pos = metadata_offset + current_pos * 1024
            metadata = np.frombuffer(
                roi_shm.buf,
                dtype=np.float32,
                count=256,
                offset=metadata_pos
            )
            
            frame_id = int(metadata[0])
            num_faces = int(metadata[1])
            ready_flag = int(metadata[253])
            
            if ready_flag != 1 or num_faces == 0 or frame_id <= 0:
                return None
                
            # Extract face bounding boxes
            faces = []
            for i in range(min(num_faces, 8)):
                offset = 4 + i * 9  # 9 values per face
                if offset + 8 < len(metadata):
                    bbox = metadata[offset:offset+4].tolist()
                    confidence = float(metadata[offset+4])
                    track_id = int(metadata[offset+5])
                    hits = int(metadata[offset+6])
                    is_detection = bool(metadata[offset+7])
                    track_age = int(metadata[offset+8])
                    
                    faces.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'track_id': track_id,
                        'hits': hits,
                        'is_detection': is_detection,
                        'track_age': track_age,
                        'frame_id': frame_id
                    })
            
            # Update access statistics
            if hasattr(self, '_direct_access_stats'):
                self._direct_access_stats['roi_direct_reads'] += 1
            
            return faces
            
        except Exception as e:
            # Fall back to scanning on any error
            if hasattr(self, '_direct_access_stats'):
                self._direct_access_stats['roi_fallbacks'] += 1
            return None
    
    def _get_detection_boxes_scan(self, camera_index: int) -> Optional[List[Dict]]:
        """
        Legacy scanning method for detection boxes.
        Used as fallback when direct index access is unavailable.
        """
        if camera_index not in self.shared_memories:
            return None
        
        if 'rois' not in self.shared_memories[camera_index]:
            return None
        
        roi_data = self.shared_memories[camera_index]['rois']
        roi_shm = roi_data['shm']
        metadata_offset = roi_data['metadata_offset']
        
        # Add timing debug
        start_time = time.time()
        
        try:
            # Read metadata from current position
            # ROI buffer metadata layout (from detection_process_gpu.py):
            # [0] = frame_id
            # [1] = num_faces  
            # [2] = timestamp
            # [3] = ready_flag
            # [4+i*8:4+i*8+4] = bbox for face i (x1, y1, x2, y2)
            # [4+i*8+4] = confidence for face i
            # [4+i*8+5] = track_id for face i
            # [4+i*8+6] = hits for face i
            # [4+i*8+7] = is_detection flag for face i
            
            # Scan ROI buffer for latest ready data
            roi_buffer_size = roi_data['buffer_size']
            latest_frame_id = -1
            latest_position = -1
            
            # Debug scanning
            if camera_index == 0 and not hasattr(self, '_roi_scan_count'):
                self._roi_scan_count = 0
            if camera_index == 0:
                self._roi_scan_count += 1
            
            # Scan all positions to find the latest ready frame
            ready_positions = []
            for pos in range(roi_buffer_size):
                metadata_pos = metadata_offset + pos * 1024  # 1024 bytes per metadata block
                try:
                    pos_metadata = np.frombuffer(
                        roi_shm.buf,
                        dtype=np.float32,
                        count=256,
                        offset=metadata_pos
                    )
                    
                    frame_id = int(pos_metadata[0])
                    n_faces = int(pos_metadata[1])
                    ready_flag = int(pos_metadata[253])
                    
                    # ROI position validation (debug logging removed)
                    
                    if ready_flag == 1:
                        ready_positions.append((pos, frame_id))
                        if frame_id > latest_frame_id:
                            latest_frame_id = frame_id
                            latest_position = pos
                except:
                    continue
            
            # Debug output
            if camera_index == 0 and self._roi_scan_count % 30 == 0:
                print(f"[ROI SCAN] Found {len(ready_positions)} ready positions: {ready_positions[:3]}")
                print(f"[ROI SCAN] Latest: pos={latest_position}, frame_id={latest_frame_id}")
            
            if latest_position < 0:
                return None
            
            # Read metadata from the latest position
            metadata_pos = metadata_offset + latest_position * 1024
            
            metadata = np.frombuffer(
                roi_shm.buf,
                dtype=np.float32,
                count=256,
                offset=metadata_pos
            )
            
            frame_id = int(metadata[0])
            num_faces = int(metadata[1])
            ready_flag = int(metadata[253])
            
            # Validate frame_id and ready flag
            if ready_flag != 1 or num_faces == 0 or frame_id <= 0:
                if camera_index == 0 and self._roi_scan_count % 50 == 0:
                    print(f"[ROI DEBUG] Invalid data: ready_flag={ready_flag}, num_faces={num_faces}, frame_id={frame_id}")
                return None
            
            # Don't clear the ready flag - let detection process manage it
            # metadata[253] = 0
            
            # Extract face bounding boxes
            faces = []
            for i in range(min(num_faces, 8)):  # Max 8 faces
                offset = 4 + i * 9  # Updated to 9 values per face (includes age)
                if offset + 8 < len(metadata):
                    bbox = metadata[offset:offset+4].tolist()  # x1, y1, x2, y2
                    confidence = float(metadata[offset+4])
                    track_id = int(metadata[offset+5])
                    hits = int(metadata[offset+6])
                    is_detection = bool(metadata[offset+7])
                    track_age = int(metadata[offset+8])  # New: track age
                    
                    faces.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'track_id': track_id,
                        'hits': hits,
                        'is_detection': is_detection,
                        'track_age': track_age,
                        'frame_id': frame_id
                    })
            
            # Timing debug
            elapsed = time.time() - start_time
            if camera_index == 0 and elapsed > 0.005:  # Log if takes more than 5ms
                print(f"[TIMING] get_detection_boxes took {elapsed:.3f}s for camera {camera_index}")
            
            # Track frame IDs to debug update frequency
            if not hasattr(self, '_last_detection_frame_ids'):
                self._last_detection_frame_ids = {}
            
            last_frame_id = self._last_detection_frame_ids.get(camera_index, -1)
            if camera_index == 0:
                if frame_id != last_frame_id:
                    print(f"[BBOX SYNC] Camera {camera_index}: New detection frame_id={frame_id} (was {last_frame_id}), {len(faces)} faces")
                    self._last_detection_frame_ids[camera_index] = frame_id
                    # Reset stale count on new frame
                    if hasattr(self, '_stale_detection_count') and camera_index in self._stale_detection_count:
                        self._stale_detection_count[camera_index] = 0
                else:
                    # Log every 30 calls if we're getting the same frame_id
                    if not hasattr(self, '_stale_detection_count'):
                        self._stale_detection_count = {}
                    if camera_index not in self._stale_detection_count:
                        self._stale_detection_count[camera_index] = 0
                    self._stale_detection_count[camera_index] += 1
                    if self._stale_detection_count[camera_index] % 30 == 0:
                        print(f"[BBOX SYNC] WARNING: Camera {camera_index} returning same frame_id={frame_id} for {self._stale_detection_count[camera_index]} calls")
            
            logger.info(f"[BBOX DEBUG] Read {len(faces)} faces from ROI buffer for camera {camera_index}")
            for i, face in enumerate(faces[:2]):  # Log first 2 faces
                logger.info(f"[BBOX DEBUG]   Face {i}: bbox={face['bbox']}, track_id={face['track_id']}, confidence={face['confidence']:.3f}")
            
            return faces
            
        except Exception as e:
            logger.error(f"Failed to read detection boxes from ROI buffer: {e}")
            return None
    
    def get_face_rois(self, camera_index: int) -> List[Dict]:
        """
        Get GPU-extracted face ROIs from shared memory buffer.
        
        Returns:
            List of dictionaries with:
                - 'roi': numpy array (256x256x3 uint8) 
                - 'track_id': int
                - 'confidence': float
                - 'quality_score': float (placeholder, needs GPU quality calculation)
        """
        if camera_index not in self.shared_memories:
            return []
            
        try:
            roi_data = self.shared_memories[camera_index]['rois']
            roi_shm = roi_data['shm']
            max_faces = roi_data.get('max_faces', 8)
            roi_buffer_size = roi_data.get('buffer_size', 8)
            
            # First get detection boxes to know which ROIs are valid
            detection_boxes = self.get_detection_boxes(camera_index)
            if not detection_boxes:
                return []
            
            # Calculate ROI size and metadata offset
            roi_size = 256 * 256 * 3  # 256x256 RGB image
            metadata_offset = roi_buffer_size * roi_size * max_faces
            
            # Read current position from metadata
            metadata = np.ndarray(
                (256,), dtype=np.float32,
                buffer=roi_shm.buf[metadata_offset:metadata_offset + 1024]
            )
            
            # Get the latest slot (simplified - in production would track ring buffer position)
            # For now, assume slot 0 contains the latest ROIs
            current_slot = 0
            
            face_rois = []
            for i, det in enumerate(detection_boxes):
                if i >= max_faces:
                    break
                    
                # Calculate offset for this face's ROI
                # Layout: [slot][face_idx] = slot * (max_faces * roi_size) + face_idx * roi_size
                roi_offset = current_slot * (max_faces * roi_size) + i * roi_size
                
                # Bounds check
                if roi_offset + roi_size > metadata_offset:
                    logger.warning(f"ROI offset {roi_offset} exceeds buffer bounds, skipping")
                    continue
                
                # Create numpy view of the ROI data
                roi_array = np.ndarray(
                    (256, 256, 3), dtype=np.uint8,
                    buffer=roi_shm.buf[roi_offset:roi_offset + roi_size]
                )
                
                # Copy the ROI to avoid shared memory issues
                roi_copy = roi_array.copy()
                
                face_rois.append({
                    'roi': roi_copy,
                    'track_id': det['track_id'],
                    'confidence': det['confidence'],
                    'quality_score': 0.8  # Placeholder - should come from GPU quality calculation
                })
            
            return face_rois
            
        except Exception as e:
            logger.error(f"Error getting face ROIs for camera {camera_index}: {e}")
            return []
    
    def get_gui_buffer_name(self, camera_index: int) -> Optional[str]:
        """
        Get the GUI buffer name for a specific camera.
        
        Args:
            camera_index: Camera identifier
            
        Returns:
            str or None: GUI buffer name if available
        """
        try:
            if camera_index in self.shared_memories:
                return self.shared_memories[camera_index].get('gui', {}).get('name')
            return None
        except Exception as e:
            logger.error(f"Failed to get GUI buffer name for camera {camera_index}: {e}")
            return None

    def get_actual_buffer_names(self) -> Dict[int, Dict[str, str]]:
        """
        Get actual shared memory buffer names for all active cameras.
        This is critical for GUI processing worker to connect to the right buffers.

        Returns:
            Dictionary mapping camera_index -> {'frame': name, 'results': name, 'pose': name, 'gui': name}
        """
        actual_names = {}
        
        for camera_index, mem_data in self.shared_memories.items():
            if mem_data:  # Only include cameras with active shared memory
                camera_names = {}
                
                # Extract actual buffer names from shared memory connections
                for buffer_type, buffer_info in mem_data.items():
                    if isinstance(buffer_info, dict) and 'name' in buffer_info:
                        # GUI buffer uses stored name
                        camera_names[buffer_type] = buffer_info['name']
                    elif isinstance(buffer_info, dict) and 'shm' in buffer_info:
                        # Other buffers use shm.name
                        camera_names[buffer_type] = buffer_info['shm'].name
                    
                # Only add camera if it has buffer names
                if camera_names:
                    actual_names[camera_index] = camera_names
                    logger.info(f"[BUFFER NAMES] Camera {camera_index} actual names: {camera_names}")
        
        logger.info(f"[BUFFER NAMES] Collected actual buffer names for {len(actual_names)} cameras")
        return actual_names
    
    def get_all_latest_results(self) -> Dict[int, Optional[Dict]]:
        """
        Get latest results from all active cameras.
        
        Returns:
            Dict[int, Optional[Dict]]: Mapping of camera_index -> results (or None if no results)
        """
        results = {}
        
        for camera_index in self.workers.keys():
            # Use the landmarks data as our "results"
            landmarks_data = self.get_landmarks(camera_index)
            if landmarks_data:
                # Convert to format expected by test
                results[camera_index] = {
                    'frame_id': landmarks_data.get('frame_id', 0),
                    'timestamp': landmarks_data.get('timestamp', 0),
                    'faces': landmarks_data.get('landmarks', []),
                    'n_faces': landmarks_data.get('n_faces', 0),
                    'processing_time_ms': landmarks_data.get('processing_time_ms', 0)
                }
            else:
                results[camera_index] = None
        
        return results
    
    def get_performance_stats(self) -> Dict[int, Dict]:
        """
        Get performance statistics for all cameras.
        
        Returns:
            Dict[int, Dict]: Mapping of camera_index -> performance stats
        """
        stats = {}
        
        for camera_index in self.workers.keys():
            if self.is_camera_active(camera_index):
                # Get basic worker stats
                stats[camera_index] = {
                    'camera_index': camera_index,
                    'active': True,
                    'worker_alive': True,
                    'last_frame_id': 0,
                    'fps': 0.0,
                    'processing_time_ms': 0.0
                }
                
                # Try to get more detailed stats from landmarks
                landmarks_data = self.get_landmarks(camera_index)
                if landmarks_data:
                    stats[camera_index].update({
                        'last_frame_id': landmarks_data.get('frame_id', 0),
                        'processing_time_ms': landmarks_data.get('processing_time_ms', 0.0),
                        'n_faces': landmarks_data.get('n_faces', 0),
                        'direct_access_stats': getattr(self, '_direct_access_stats', {}),
                        'sync_health': self._get_sync_health_status(camera_index)
                    })
            else:
                stats[camera_index] = {
                    'camera_index': camera_index,
                    'active': False,
                    'worker_alive': False
                }
        
        return stats
    
    def _get_sync_health_status(self, camera_index: int) -> Dict:
        """Get frame-overlay synchronization health status."""
        if not hasattr(self, '_last_frame_ids'):
            return {'status': 'no_data'}
            
        frame_id = self._last_frame_ids.get(camera_index, -1)
        direct_reads = self._direct_access_stats.get('gui_direct_reads', 0)
        fallbacks = self._direct_access_stats.get('gui_fallbacks', 0)
        
        total_reads = direct_reads + fallbacks
        direct_ratio = direct_reads / total_reads if total_reads > 0 else 0
        
        return {
            'status': 'healthy' if direct_ratio > 0.8 else 'degraded',
            'current_frame_id': frame_id,
            'direct_access_ratio': direct_ratio,
            'total_reads': total_reads
        }
    
    def get_system_status(self) -> Dict:
        """
        Get overall system status.
        
        Returns:
            Dict: System status including worker counts, memory usage, etc.
        """
        active_workers = sum(1 for camera_index in self.workers.keys() if self.is_camera_active(camera_index))
        
        return {
            'total_workers': len(self.workers),
            'active_workers': active_workers,
            'gpu_count': self.num_gpus,
            'timestamp': time.time(),
            'status': 'healthy' if active_workers > 0 else 'no_active_workers'
        }


def example_usage():
    """Example of using the camera worker manager."""
    # Load config for example
    config = {}
    try:
        import json
        with open('youquantipy_config.json', 'r') as f:
            config = json.load(f)
    except:
        config = {}  # Use defaults if config not available
        
    # Create manager
    manager = CameraWorkerManager()
    
    try:
        # Start cameras
        num_cameras = 2
        for i in range(num_cameras):
            if manager.start_camera(i):
                print(f"Camera {i} started successfully")
            else:
                print(f"Failed to start camera {i}")
        
        # Main loop
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Process metadata
            metadata_list = manager.process_metadata()
            for metadata in metadata_list:
                print(f"Camera {metadata['camera_index']} - "
                      f"Frame {metadata['frame_id']}: "
                      f"{metadata['n_faces']} faces, "
                      f"{metadata['processing_time_ms']:.1f}ms")
            
            # Process status updates
            status_list = manager.process_status_updates()
            for status in status_list:
                if status['type'] == 'heartbeat':
                    data = status['data']
                    print(f"Heartbeat from camera {status['camera_index']}: "
                          f"{data['current_fps']:.1f} FPS, "
                          f"{data['frames_processed']} processed")
            
            # Get preview frames
            for cam_idx in range(num_cameras):
                frame = manager.get_preview_frame(cam_idx)
                if frame is not None:
                    frame_count += 1
                    # Here you would display or process the frame
                    print(f"Got preview frame from camera {cam_idx}")
            
            # Get landmarks
            for cam_idx in range(num_cameras):
                landmarks = manager.get_landmarks(cam_idx)
                if landmarks is not None:
                    print(f"Got landmarks from camera {cam_idx}: "
                          f"{landmarks['n_faces']} faces")
            
            # Print overall stats every 5 seconds
            if time.time() - start_time > 5:
                print(f"\nOverall: {frame_count} frames in 5 seconds")
                frame_count = 0
                start_time = time.time()
                
                # Request detailed stats
                manager.get_stats()
            
            loop_delay = get_nested_config(config, 'buffer_management.timeouts.command_retry_delay_ms', 10) / 1000.0
            time.sleep(loop_delay)  # Configurable loop delay
            
    except KeyboardInterrupt:
        print("\nStopping cameras...")
    finally:
        # Clean shutdown
        manager.stop_all()
        print("All cameras stopped")


if __name__ == '__main__':
    example_usage()