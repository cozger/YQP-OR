"""
RTMPose3D Process - 3D whole-body pose estimation using MMPose.

This process runs RTMDet-M for person detection followed by RTMW3D-L
for 3D whole-body pose estimation (133 keypoints per person).

Pipeline:
    Frame Buffer → RTMDet Person Detection → RTMW3D 3D Pose → Pose Buffer
"""

import multiprocessing as mp
import time
import sys
import os
import numpy as np
from typing import Optional, Dict, Any
import logging

# Add MMPose projects directory to Python path (required for rtmpose3d imports)
# Note: Must add /projects (parent), not /projects/rtmpose3d (child)
# This allows "from rtmpose3d.rtmpose3d import X" to resolve correctly
mmpose_projects_path = "/home/canoz/Projects/surgery/mmpose/projects"
if mmpose_projects_path not in sys.path:
    sys.path.insert(0, mmpose_projects_path)

# CRITICAL FIX: Don't import MMPose/MMDet at module level in spawned subprocess
# The imports require CUDA libraries, but LD_LIBRARY_PATH isn't set yet
# Instead, imports will be done in _init_models() after CUDA environment setup

# Flag to indicate if we should try to import (will be set after CUDA setup)
MMPOSE_AVAILABLE = None  # None = not yet checked, True = available, False = unavailable


def extract_person_bboxes(det_results, conf_threshold=0.5):
    """
    Extract person bounding boxes from detection results.

    Args:
        det_results: MMDet detection results
        conf_threshold: Minimum confidence threshold

    Returns:
        numpy.ndarray: Person bboxes in format [[x1, y1, x2, y2], ...]
    """
    if not hasattr(det_results, 'pred_instances'):
        return np.array([]).reshape(0, 4)

    pred_instances = det_results.pred_instances

    # Filter for person class (class 0 in COCO)
    person_mask = pred_instances.labels == 0

    if not person_mask.any():
        return np.array([]).reshape(0, 4)

    # Get person detections
    person_bboxes = pred_instances.bboxes[person_mask].cpu().numpy()
    person_scores = pred_instances.scores[person_mask].cpu().numpy()

    # Filter by confidence
    conf_mask = person_scores >= conf_threshold

    if not conf_mask.any():
        return np.array([]).reshape(0, 4)

    return person_bboxes[conf_mask]


class RTMPose3DProcess(mp.Process):
    """
    Multiprocessing worker for 3D whole-body pose estimation.

    Reads frames from shared memory buffer, runs RTMDet for person detection,
    then RTMW3D for 3D pose estimation, and writes results to pose buffer.
    """

    def __init__(
        self,
        camera_id: int,
        frame_buffer_name: str,
        frame_layout_dict: Dict[str, Any],
        pose_buffer_name: str,
        pose_layout_dict: Dict[str, Any],
        detector_config: str,
        detector_checkpoint: str,
        pose_config: str,
        pose_checkpoint: str,
        device: str = 'cuda:0',
        det_conf_threshold: float = 0.5,
        pose_conf_threshold: float = 0.3,
        stop_event: Optional[mp.Event] = None,
        ready_event: Optional[mp.Event] = None,
        log_queue: Optional[mp.Queue] = None,
    ):
        """
        Initialize RTMPose3D process.

        Args:
            camera_id: Camera identifier
            frame_buffer_name: Shared memory name for frame buffer
            frame_layout_dict: Frame buffer layout configuration
            pose_buffer_name: Shared memory name for pose buffer
            pose_layout_dict: Pose buffer layout configuration
            detector_config: Path to RTMDet config file
            detector_checkpoint: Path to RTMDet checkpoint
            pose_config: Path to RTMW3D config file
            pose_checkpoint: Path to RTMW3D checkpoint
            device: Inference device ('cuda:0' or 'cpu')
            det_conf_threshold: Detection confidence threshold
            pose_conf_threshold: Pose keypoint confidence threshold
            stop_event: Event to signal process shutdown
            ready_event: Event to signal process initialization complete
            log_queue: Queue for logging messages
        """
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.frame_buffer_name = frame_buffer_name
        self.frame_layout = frame_layout_dict
        self.pose_buffer_name = pose_buffer_name
        self.pose_layout = pose_layout_dict
        self.detector_config = detector_config
        self.detector_checkpoint = detector_checkpoint
        self.pose_config = pose_config
        self.pose_checkpoint = pose_checkpoint
        self.device = device
        self.det_conf_threshold = det_conf_threshold
        self.pose_conf_threshold = pose_conf_threshold
        self.stop_event = stop_event or mp.Event()
        self.ready_event = ready_event or mp.Event()
        self.log_queue = log_queue

        # Will be initialized in run()
        self.detector = None
        self.pose_estimator = None
        self.frame_shm = None
        self.pose_shm = None

    def _log(self, level: str, message: str):
        """Send log message to queue if available, or print to stdout."""
        if self.log_queue:
            try:
                self.log_queue.put_nowait((level, f"[RTMPose3D-{self.camera_id}] {message}"))
            except:
                pass
        else:
            # Fallback: print directly to stdout when no log queue
            print(f"[{level}] [RTMPose3D-{self.camera_id}] {message}", flush=True)

    def _init_models(self):
        """Initialize RTMDet detector and RTMW3D pose estimator."""
        global MMPOSE_AVAILABLE

        # Add MMPose projects directory to Python path for custom modules (rtmpose3d)
        import sys
        import os
        mmpose_projects_dir = '/home/canoz/Projects/surgery/mmpose/projects'
        if os.path.exists(mmpose_projects_dir) and mmpose_projects_dir not in sys.path:
            sys.path.insert(0, mmpose_projects_dir)
            self._log('INFO', f"Added MMPose projects directory to Python path: {mmpose_projects_dir}")

        # Import MMPose/MMDet here (after CUDA environment is set up)
        if MMPOSE_AVAILABLE is None:
            self._log('INFO', "Importing MMPose and MMDet libraries...")

            # Import rtmpose3d classes to register them in MMPose registry
            # Note: rtmpose3d is at projects/rtmpose3d/rtmpose3d/ (nested structure)
            try:
                from rtmpose3d.rtmpose3d.pose_estimator import TopdownPoseEstimator3D
                from rtmpose3d.rtmpose3d.rtmw3d_head import RTMW3DHead
                from rtmpose3d.rtmpose3d.simcc_3d_label import SimCC3DLabel
                from rtmpose3d.rtmpose3d.loss import KLDiscretLossWithWeight
                self._log('INFO', "✅ Imported rtmpose3d classes (TopdownPoseEstimator3D, RTMW3DHead registered)")
            except ImportError as e:
                self._log('ERROR', f"Failed to import rtmpose3d classes: {e}")
                import traceback
                self._log('ERROR', traceback.format_exc())
                raise
            try:
                # Import MMDet FIRST to avoid registry conflicts
                import mmdet
                from mmdet.apis import init_detector, inference_detector

                # Then import MMPose
                from mmpose.apis import init_model, inference_topdown
                from mmpose.utils import adapt_mmdet_pipeline

                # FIX: Register mmdet transforms in mmpose registry to avoid lookup errors
                try:
                    from mmdet.datasets.transforms import PackDetInputs
                    from mmpose.registry import TRANSFORMS
                    if not TRANSFORMS.get('PackDetInputs'):
                        TRANSFORMS.register_module(module=PackDetInputs, force=True)
                except:
                    pass  # Already registered or not needed

                MMPOSE_AVAILABLE = True
                self._log('INFO', "✅ MMPose and MMDet imported successfully")

                # Store imports as instance variables for later use
                self._init_detector = init_detector
                self._init_model = init_model
                self._adapt_mmdet_pipeline = adapt_mmdet_pipeline
                self._inference_detector = inference_detector
                self._inference_topdown = inference_topdown

            except ImportError as e:
                MMPOSE_AVAILABLE = False
                self._log('ERROR', f"Failed to import MMPose/MMDet: {e}")
                import traceback
                self._log('ERROR', f"Traceback: {traceback.format_exc()}")
                raise RuntimeError(f"MMPose not available: {e}")

        if not MMPOSE_AVAILABLE:
            raise RuntimeError("MMPose not installed. Cannot initialize models.")

        self._log('INFO', "Initializing RTMDet person detector...")
        self.detector = self._init_detector(
            self.detector_config,
            self.detector_checkpoint,
            device=self.device
        )

        # CRITICAL: Adapt MMDet pipeline to avoid registry conflicts
        self.detector.cfg = self._adapt_mmdet_pipeline(self.detector.cfg)

        self._log('INFO', "Initializing RTMW3D pose estimator...")
        self.pose_estimator = self._init_model(
            self.pose_config,
            self.pose_checkpoint,
            device=self.device
        )

        self._log('INFO', "Models initialized successfully")

    def _attach_buffers(self):
        """Attach to shared memory buffers."""
        from multiprocessing import shared_memory

        # Attach to frame buffer
        self.frame_shm = shared_memory.SharedMemory(name=self.frame_buffer_name)
        self._log('INFO', f"Attached to frame buffer: {self.frame_buffer_name}")

        # CRITICAL FIX: Read actual resolution from buffer header (single source of truth)
        # The layout dict may contain stale cached values, but the buffer header is always current
        resolution_view = np.frombuffer(self.frame_shm.buf, dtype=np.uint32, count=2, offset=8)
        actual_width = int(resolution_view[0])
        actual_height = int(resolution_view[1])

        # Compare with layout dict to detect mismatches
        expected_width = self.frame_layout.get('frame_width', 0)
        expected_height = self.frame_layout.get('frame_height', 0)

        if actual_width != expected_width or actual_height != expected_height:
            self._log('WARNING', f"Resolution mismatch detected!")
            self._log('WARNING', f"  Expected from layout dict: {expected_width}×{expected_height}")
            self._log('WARNING', f"  Actual from buffer header: {actual_width}×{actual_height}")
            self._log('WARNING', f"  Using actual resolution from buffer header to prevent reshape errors")

            # Override layout dict with actual values from buffer
            self.frame_layout['frame_width'] = actual_width
            self.frame_layout['frame_height'] = actual_height
            self.frame_layout['frame_size'] = actual_width * actual_height * 3
        else:
            self._log('INFO', f"Resolution validated: {actual_width}×{actual_height}")

        # Attach to pose buffer
        self.pose_shm = shared_memory.SharedMemory(name=self.pose_buffer_name)
        self._log('INFO', f"Attached to pose buffer: {self.pose_buffer_name}")

    def _read_frame(self) -> Optional[np.ndarray]:
        """
        Read latest frame from frame buffer.

        Returns:
            Frame as numpy array (H, W, 3) BGR, or None if no frame available
        """
        try:
            # Read write index
            write_idx_bytes = self.frame_shm.buf[
                self.frame_layout['write_index_offset']:
                self.frame_layout['write_index_offset'] + self.frame_layout['write_index_size']
            ]
            write_idx = int.from_bytes(write_idx_bytes, byteorder='little')

            if write_idx == 0:
                return None  # No frames written yet

            # Get current frame index
            frame_idx = (write_idx - 1) % self.frame_layout['ring_buffer_size']

            # Read frame
            frame_offset = self.frame_layout['frame_offsets'][frame_idx]
            frame_size = self.frame_layout['frame_size']

            frame_bytes = self.frame_shm.buf[frame_offset:frame_offset + frame_size]
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                (self.frame_layout['frame_height'], self.frame_layout['frame_width'], 3)
            )

            return frame.copy()  # Return a copy to avoid shared memory issues

        except Exception as e:
            self._log('ERROR', f"Error reading frame: {e}")
            return None

    def _write_pose_results(self, results: list, frame_id: int, timestamp_ms: int,
                           det_time_ms: float, pose_time_ms: float):
        """
        Write pose estimation results to pose buffer using ring buffer with atomic writes.

        Atomic write pattern:
        1. Read current write_index (but don't increment yet)
        2. Calculate slot index
        3. Write pose data to slot
        4. Write metadata to slot
        5. Increment write_index LAST (atomic "data ready" signal)

        This prevents readers from seeing partial updates.

        Args:
            results: List of pose results from MMPose
            frame_id: Frame identifier
            timestamp_ms: Frame timestamp in milliseconds
            det_time_ms: Detection processing time
            pose_time_ms: Pose estimation processing time
        """
        try:
            # Read current write index (don't increment yet - atomic write pattern)
            write_idx_bytes = self.pose_shm.buf[
                self.pose_layout['write_index_offset']:
                self.pose_layout['write_index_offset'] + self.pose_layout['write_index_size']
            ]
            write_idx = int.from_bytes(write_idx_bytes, byteorder='little')

            # Calculate ring buffer slot index
            ring_buffer_size = self.pose_layout['ring_buffer_size']
            slot_idx = write_idx % ring_buffer_size

            # Prepare pose data array
            max_persons = self.pose_layout['max_persons']
            keypoints_per_person = self.pose_layout['keypoints_per_person']
            values_per_keypoint = self.pose_layout['values_per_keypoint']

            pose_data = np.zeros((max_persons, keypoints_per_person, values_per_keypoint), dtype=np.float32)

            # Fill pose data from results
            n_persons = min(len(results), max_persons)
            for person_idx in range(n_persons):
                result = results[person_idx]

                # Extract keypoints from PoseDataSample
                if not hasattr(result, 'pred_instances'):
                    continue

                pred_instances = result.pred_instances

                # Convert to numpy if needed
                keypoints = pred_instances.keypoints
                if hasattr(keypoints, 'cpu'):
                    keypoints = keypoints.cpu().numpy()
                else:
                    keypoints = np.array(keypoints)

                scores = pred_instances.keypoint_scores
                if hasattr(scores, 'cpu'):
                    scores = scores.cpu().numpy()
                else:
                    scores = np.array(scores)

                # Remove batch dimension if present: (1, 133, 3) -> (133, 3)
                if keypoints.ndim == 3 and keypoints.shape[0] == 1:
                    keypoints = keypoints[0]
                if scores.ndim == 2 and scores.shape[0] == 1:
                    scores = scores[0]

                # Combine into (133, 4) format: [x, y, z, confidence]
                for kp_idx in range(min(keypoints_per_person, len(keypoints))):
                    pose_data[person_idx, kp_idx, 0] = keypoints[kp_idx, 0]  # x
                    pose_data[person_idx, kp_idx, 1] = keypoints[kp_idx, 1]  # y
                    pose_data[person_idx, kp_idx, 2] = keypoints[kp_idx, 2]  # z
                    pose_data[person_idx, kp_idx, 3] = scores[kp_idx]  # confidence

            # STEP 1: Write pose data to ring buffer slot FIRST
            pose_offset = self.pose_layout['pose_offsets'][slot_idx]
            pose_bytes = pose_data.tobytes()
            self.pose_shm.buf[pose_offset:pose_offset + len(pose_bytes)] = pose_bytes

            # STEP 2: Write metadata to ring buffer slot SECOND
            metadata_offset = self.pose_layout['metadata_offsets'][slot_idx]
            metadata = np.array([
                (frame_id, timestamp_ms, n_persons, 1, det_time_ms + pose_time_ms, det_time_ms, pose_time_ms)
            ], dtype=self.pose_layout['metadata_dtype'])
            metadata_bytes = metadata.tobytes()
            self.pose_shm.buf[metadata_offset:metadata_offset + len(metadata_bytes)] = metadata_bytes

            # STEP 3: Increment write_index LAST (atomic "data ready" signal)
            # This ensures readers never see partial updates
            write_idx += 1
            self.pose_shm.buf[
                self.pose_layout['write_index_offset']:
                self.pose_layout['write_index_offset'] + self.pose_layout['write_index_size']
            ] = write_idx.to_bytes(8, byteorder='little')

        except Exception as e:
            self._log('ERROR', f"Error writing pose results: {e}")

    def _setup_cuda_environment(self):
        """
        Setup CUDA environment for spawned subprocess.

        In spawn mode, the subprocess doesn't inherit the parent's CUDA setup.
        This method configures LD_LIBRARY_PATH to ensure CUDA libraries are accessible.

        NOTE: PyTorch bundles complete CUDA libraries including nvJitLink with all required
        symbols. We do NOT use LD_PRELOAD as it would override PyTorch's newer libraries
        with older system versions that may be missing required symbols.
        """
        import os

        # Fix CUDA library loading
        # CRITICAL: Dynamic linker search order is:
        #   1. LD_PRELOAD (highest priority)
        #   2. LD_LIBRARY_PATH
        #   3. System paths (/etc/ld.so.conf) ← /usr/lib/wsl/lib is configured here in WSL2!
        #   4. RUNPATH (embedded in binary) ← PyTorch's CUDA libs point here
        #   5. System default paths
        #
        # Problem: /usr/lib/wsl/lib contains OLD nvJitLink (CUDA 12.0, missing __nvJitLinkAddData_12_1)
        #          PyTorch needs COMPLETE nvJitLink (CUDA 12.1+, has all symbols _12_0 through _12_9)
        #
        # Since /usr/lib/wsl/lib is in /etc/ld.so.conf.d/, it's ALWAYS searched before RUNPATH,
        # even when LD_LIBRARY_PATH is unset. This causes the old library to be loaded.
        #
        # SOLUTION: Use LD_PRELOAD to force PyTorch's complete nvJitLink to load first.
        # This is the ONLY way to override system-wide library paths.

        import sys
        site_packages = os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
        pytorch_nvjitlink = os.path.join(site_packages, 'nvidia', 'nvjitlink', 'lib', 'libnvJitLink.so.12')

        if os.path.exists(pytorch_nvjitlink):
            os.environ['LD_PRELOAD'] = pytorch_nvjitlink
            self._log('INFO', f"CUDA environment setup: Preloading PyTorch's nvJitLink via LD_PRELOAD")
            self._log('DEBUG', f"LD_PRELOAD: {pytorch_nvjitlink}")
        else:
            self._log('ERROR', f"PyTorch nvJitLink not found at: {pytorch_nvjitlink}")
            self._log('ERROR', "Cannot override system nvJitLink - torch import will likely fail!")

        # Also clear LD_LIBRARY_PATH to avoid additional conflicts
        if 'LD_LIBRARY_PATH' in os.environ:
            del os.environ['LD_LIBRARY_PATH']
            self._log('DEBUG', "Cleared LD_LIBRARY_PATH")

        # Note: We use LD_PRELOAD instead of relying on RUNPATH because WSL2 has /usr/lib/wsl/lib
        # in /etc/ld.so.conf which is searched BEFORE RUNPATH, causing the wrong library to load.

    def run(self):
        """Main process loop."""
        try:
            self._log('INFO', "Starting RTMPose3D process...")

            # CRITICAL: Fix CUDA library path for spawned subprocess
            # This subprocess doesn't inherit the CUDA environment from parent
            self._setup_cuda_environment()

            # Initialize models (with explicit error handling)
            try:
                self._init_models()
            except Exception as e:
                self._log('ERROR', f"Failed to initialize models: {e}")
                import traceback
                self._log('ERROR', traceback.format_exc())
                return  # Exit process

            # Attach to shared memory buffers
            try:
                self._attach_buffers()
            except Exception as e:
                self._log('ERROR', f"Failed to attach buffers: {e}")
                import traceback
                self._log('ERROR', traceback.format_exc())
                return  # Exit process

            # Signal ready
            self.ready_event.set()
            self._log('INFO', "RTMPose3D process ready")

            frame_id = 0
            last_process_time = time.time()
            fps_interval = 1.0  # Log FPS every second

            while not self.stop_event.is_set():
                try:
                    # Read frame
                    frame = self._read_frame()
                    if frame is None:
                        time.sleep(0.001)  # Brief sleep to avoid busy waiting
                        continue

                    frame_id += 1
                    timestamp_ms = int(time.time() * 1000)

                    # Run person detection
                    det_start = time.time()

                    try:
                        det_results = self._inference_detector(self.detector, frame)
                        person_bboxes = extract_person_bboxes(
                            det_results,
                            conf_threshold=self.det_conf_threshold
                        )

                        # Select closest person (largest bbox by area)
                        if len(person_bboxes) > 1:
                            # Calculate bbox areas
                            areas = (person_bboxes[:, 2] - person_bboxes[:, 0]) * \
                                    (person_bboxes[:, 3] - person_bboxes[:, 1])
                            # Keep only the largest bbox
                            largest_idx = np.argmax(areas)
                            person_bboxes = person_bboxes[largest_idx:largest_idx+1]

                    except Exception as e:
                        self._log('WARNING', f"Detection failed ({e}), using full frame")
                        # Fallback: use full frame as bbox if detection fails
                        h, w = frame.shape[:2]
                        person_bboxes = np.array([[0, 0, w, h]])

                    det_time_ms = (time.time() - det_start) * 1000

                    # Run pose estimation if persons detected
                    pose_results = []
                    pose_time_ms = 0.0
                    if len(person_bboxes) > 0:
                        pose_start = time.time()
                        # Convert bboxes to required format
                        bboxes_xyxy = person_bboxes[:, :4]  # (N, 4)
                        pose_results = self._inference_topdown(
                            self.pose_estimator,
                            frame,
                            bboxes_xyxy
                        )
                        pose_time_ms = (time.time() - pose_start) * 1000

                    # Write results to buffer
                    self._write_pose_results(
                        pose_results,
                        frame_id,
                        timestamp_ms,
                        det_time_ms,
                        pose_time_ms
                    )

                    # FPS logging
                    current_time = time.time()
                    if current_time - last_process_time >= fps_interval:
                        total_time_ms = det_time_ms + pose_time_ms
                        self._log('INFO',
                                 f"Frame {frame_id}: {len(person_bboxes)} persons, "
                                 f"det={det_time_ms:.1f}ms, pose={pose_time_ms:.1f}ms, "
                                 f"total={total_time_ms:.1f}ms")
                        last_process_time = current_time

                except Exception as e:
                    self._log('ERROR', f"Error in processing loop: {e}")
                    import traceback
                    self._log('ERROR', traceback.format_exc())
                    time.sleep(0.1)

        except Exception as e:
            self._log('ERROR', f"Fatal error in RTMPose3D process: {e}")
            import traceback
            self._log('ERROR', traceback.format_exc())

        finally:
            # Cleanup
            self._log('INFO', "Shutting down RTMPose3D process...")
            if self.frame_shm:
                self.frame_shm.close()
            if self.pose_shm:
                self.pose_shm.close()
            self._log('INFO', "RTMPose3D process terminated")
