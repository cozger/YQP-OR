import time
import numpy as np
from multiprocessing import Process, Queue as MPQueue, Value
from pylsl import StreamInfo, StreamOutlet
from collections import OrderedDict
import threading
from core.data_streaming.correlator import ChannelCorrelator
from core.participant_management import SingleParticipantManager
from multiprocessing import shared_memory
import os

class LSLHelper:
    """
    Centralized LSL stream management and comodulation calculations.
    Runs in a separate process for optimal performance.
    """
    def __init__(self, fps=30, config=None):
        self.fps = fps
        self.config = config or {}
        self.streams = {}  # participant_id -> StreamOutlet (face streams)
        self.pose_streams = {}  # participant_id -> pose StreamOutlet
        self.metrics_streams = {}  # participant_id -> metrics StreamOutlet

        # Get correlator window size from config
        correlator_window_size = self.config.get('data_streaming', {}).get('lsl', {}).get('correlator_window_size', 60)
        self.correlator = ChannelCorrelator(window_size=correlator_window_size, fps=fps, config=self.config)
        self.running = False
        
    def create_face_stream(self, participant_id, include_mesh=False, is_merged=False):
        """Create or update LSL stream for a participant

        Args:
            participant_id: Participant identifier (e.g., "P1", "P2")
            include_mesh: Include 478*3 mesh landmark data
            is_merged: Whether this stream contains merged multi-camera data
        """
        stream_name = f"{participant_id}_landmarks"

        # Calculate channel count
        blend_channels = 52
        mesh_channels = 478 * 3 if include_mesh else 0
        quality_channels = 3  # size, frontal, stability
        total_channels = blend_channels + mesh_channels + quality_channels

        # Close existing stream if it exists with different config
        if participant_id in self.streams:
            old_info = self.streams[participant_id]['info']
            if old_info.channel_count() != total_channels:
                # StreamOutlet has no close() method - just delete the reference
                # The outlet will be cleaned up automatically when the dict is deleted
                del self.streams[participant_id]

        # Create new stream if needed
        if participant_id not in self.streams:
            info = StreamInfo(
                name=stream_name,
                type="Landmark",
                channel_count=total_channels,
                nominal_srate=self.fps,
                channel_format="float32",
                source_id=f"{participant_id}_uid"
            )

            # Add metadata to indicate merge status
            desc = info.desc()
            desc.append_child_value("merge_status", "merged" if is_merged else "direct")
            desc.append_child_value("mesh_enabled", "true" if include_mesh else "false")

            # Document channel layout
            channels = desc.append_child("channels")

            # Blendshape channels (52)
            for i in range(52):
                ch = channels.append_child("channel")
                ch.append_child_value("label", f"blend_{i}")
                ch.append_child_value("type", "blendshape")

            # Mesh channels (478*3 if enabled)
            if include_mesh:
                for i in range(478 * 3):
                    ch = channels.append_child("channel")
                    ch.append_child_value("label", f"mesh_{i}")
                    ch.append_child_value("type", "mesh_coordinate")

            # Quality channels (3)
            for label in ["quality_size", "quality_frontal", "quality_stability"]:
                ch = channels.append_child("channel")
                ch.append_child_value("label", label)
                ch.append_child_value("type", "quality_metric")
                ch.append_child_value("unit", "normalized")
                ch.append_child_value("range", "0.0-1.0")

            outlet = StreamOutlet(info)

            self.streams[participant_id] = {
                'outlet': outlet,
                'info': info,
                'include_mesh': include_mesh,
                'is_merged': is_merged,
                'last_sample': None,
                'frame_count': 0,
                'last_fps_time': time.time()
            }
            merge_str = "merged multi-camera" if is_merged else "direct single-camera"
            print(f"[LSL Helper] Created face stream for {participant_id} with {total_channels} channels ({merge_str})")
            # DIAGNOSTIC: Log detailed channel breakdown
            print(f"  Channel breakdown: {blend_channels} blendshapes + {mesh_channels} mesh + {quality_channels} quality")
            print(f"  Mesh {'ENABLED' if include_mesh else 'DISABLED'} (include_mesh={include_mesh})")

    def push_face_data(self, participant_id, blend_scores, mesh_data=None, quality_scores=None, pose_data=None):
        """Push data for a participant

        Args:
            participant_id: Participant identifier
            blend_scores: 52 blendshape coefficients (list or array)
            mesh_data: Optional 478*3 mesh coordinates (if stream has mesh enabled)
            quality_scores: 3-element quality metrics [size, frontal, stability]
            pose_data: (deprecated, kept for compatibility)
        """
        if participant_id not in self.streams:
            return

        stream_info = self.streams[participant_id]

        # Build sample: blendshapes + [mesh] + quality
        sample = blend_scores.copy() if isinstance(blend_scores, list) else blend_scores.tolist()

        # Add mesh data if the stream was created with mesh enabled
        if stream_info['include_mesh']:
            if mesh_data is not None and len(mesh_data) > 0:
                sample.extend(mesh_data if isinstance(mesh_data, list) else mesh_data.tolist())
                # DIAGNOSTIC: Log successful mesh addition (throttled to first 3 per participant)
                if not hasattr(self, '_mesh_success_log'):
                    self._mesh_success_log = {}
                if participant_id not in self._mesh_success_log:
                    self._mesh_success_log[participant_id] = 0
                if self._mesh_success_log[participant_id] < 3:
                    print(f"[LSL MESH DIAGNOSTIC] Added {len(mesh_data)} mesh values to sample for {participant_id}")
                    self._mesh_success_log[participant_id] += 1
            else:
                # Pad with zeros if mesh is expected but not provided
                mesh_size = 478 * 3
                sample.extend([0.0] * mesh_size)
                # ENHANCED DIAGNOSTIC: Always log missing mesh data (critical error)
                print(f"[LSL MESH DIAGNOSTIC] WARNING: Mesh expected but NOT provided for {participant_id}")
                print(f"  Stream include_mesh=True, but mesh_data={mesh_data}")
                print(f"  Padding with {mesh_size} zeros - LSL stream will contain EMPTY mesh data!")
        else:
            # DIAGNOSTIC: Log when mesh is not included in stream (only once)
            if not hasattr(self, '_mesh_not_included_logged'):
                print(f"[LSL MESH DIAGNOSTIC] Stream for {participant_id} created without mesh (include_mesh=False)")
                self._mesh_not_included_logged = True

        # Add quality scores (ALWAYS - 3 channels)
        if quality_scores is not None and len(quality_scores) >= 3:
            # Expect [size, frontal, stability]
            sample.extend(quality_scores[:3])
        else:
            # Default quality scores if not provided
            sample.extend([0.0, 0.0, 0.0])
            print(f"[LSL Helper] Warning: Quality scores not provided for {participant_id}, using zeros")

        # Check for duplicate
        if stream_info['last_sample'] is not None and sample == stream_info['last_sample']:
            return

        # Verify sample size matches expected channel count
        expected_size = stream_info['info'].channel_count()
        if len(sample) != expected_size:
            print(f"[LSL Helper] ERROR: Sample size mismatch for {participant_id}: "
                f"got {len(sample)}, expected {expected_size}")
            return

        stream_info['last_sample'] = sample.copy()
        stream_info['outlet'].push_sample(sample)

        # Update frame count for FPS monitoring
        stream_info['frame_count'] += 1
        
    def get_fps_stats(self):
        """Get FPS statistics for all streams"""
        stats = {}
        current_time = time.time()

        for pid, stream_info in self.streams.items():
            elapsed = current_time - stream_info['last_fps_time']
            if elapsed > 0:
                fps = stream_info['frame_count'] / elapsed
                stats[pid] = fps
                # Reset counters
                stream_info['frame_count'] = 0
                stream_info['last_fps_time'] = current_time

        return stats

    def get_bluetooth_fps_stats(self):
        """Get FPS statistics for Bluetooth streams"""
        stats = {}
        if not hasattr(self, 'bluetooth_streams'):
            return stats

        current_time = time.time()
        for stream_key, stream_info in self.bluetooth_streams.items():
            elapsed = current_time - stream_info['last_fps_time']
            if elapsed > 0:
                fps = stream_info['sample_count'] / elapsed
                stats[stream_key] = fps
                # Reset counters
                stream_info['sample_count'] = 0
                stream_info['last_fps_time'] = current_time

        return stats
    
    def create_pose_stream(self, participant_id, fps=None):
        """Create a dedicated pose stream for participant with full 133 keypoints

        RTMW3D keypoints (133 total):
        - Body (COCO 17): indices 0-16
        - Feet: indices 17-22
        - Face: indices 23-90
        - Left hand: indices 91-111
        - Right hand: indices 112-132

        Each keypoint: [x, y, z, confidence] = 4 channels
        Total: 133 * 4 = 532 channels
        """
        if fps is None:
            fps = self.fps
        stream_name = f"{participant_id}_pose"
        channel_count = 133 * 4  # Full RTMW3D keypoints (was 33 * 4 = 132)

        info = StreamInfo(
            name=stream_name,
            type="Pose",
            channel_count=channel_count,
            nominal_srate=fps,
            channel_format="float32",
            source_id=f"{participant_id}_pose_uid"
        )

        # Add metadata for keypoint structure
        desc = info.desc()
        desc.append_child_value("keypoint_count", "133")
        desc.append_child_value("values_per_keypoint", "4")
        desc.append_child_value("coordinate_system", "camera_space_meters")

        # Document channel layout
        channels = desc.append_child("channels")
        keypoint_names = [
            # Body (COCO 17): 0-16
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
            # Feet: 17-22
            "left_big_toe", "left_small_toe", "left_heel",
            "right_big_toe", "right_small_toe", "right_heel",
        ]
        # Face: 23-90 (68 keypoints)
        keypoint_names.extend([f"face_{i}" for i in range(68)])
        # Left hand: 91-111 (21 keypoints)
        keypoint_names.extend([f"left_hand_{i}" for i in range(21)])
        # Right hand: 112-132 (21 keypoints)
        keypoint_names.extend([f"right_hand_{i}" for i in range(21)])

        # Create 4 channels per keypoint (x, y, z, confidence)
        for kp_idx, kp_name in enumerate(keypoint_names):
            for coord_idx, coord_name in enumerate(["x", "y", "z", "confidence"]):
                ch = channels.append_child("channel")
                ch.append_child_value("label", f"{kp_name}_{coord_name}")
                ch.append_child_value("keypoint_index", str(kp_idx))
                ch.append_child_value("coordinate", coord_name)
                if coord_name in ["x", "y", "z"]:
                    ch.append_child_value("unit", "meters")
                else:
                    ch.append_child_value("unit", "normalized")
                    ch.append_child_value("range", "0.0-1.0")

        outlet = StreamOutlet(info)
        self.pose_streams[participant_id] = {
            'outlet': outlet,
            'info': info,
            'last_sample': None,
            'frame_count': 0,
            'last_fps_time': time.time()
        }
        print(f"[LSL Helper] Created POSE stream for {participant_id} ({channel_count} channels, {fps} FPS)")

    def close_pose_stream(self, participant_id):
        """Close a dedicated pose stream for participant"""
        if participant_id in self.pose_streams:
            # StreamOutlet has no close() method - just delete the reference
            # The outlet will be cleaned up automatically when the dict is deleted
            del self.pose_streams[participant_id]
            print(f"[LSL Helper] Closed POSE stream for {participant_id}")

    def push_pose_data(self, participant_id, pose_data):
        """Push full 133-keypoint pose data to LSL stream

        Args:
            participant_id: Participant identifier
            pose_data: Flattened array of 133 keypoints × 4 values = 532 floats
                      Format: [kp0_x, kp0_y, kp0_z, kp0_conf, kp1_x, ...]
        """
        if participant_id not in self.pose_streams:
            return
        stream_info = self.pose_streams[participant_id]

        # Verify size matches expected 532 channels (133 keypoints × 4)
        expected_size = 133 * 4
        if len(pose_data) != expected_size:
            print(f"[LSL Helper] WARNING: Pose data size mismatch for {participant_id}: "
                  f"got {len(pose_data)}, expected {expected_size}")
            return

        # To avoid flooding, optionally check for duplicate
        if stream_info['last_sample'] == pose_data:
            return
        stream_info['last_sample'] = pose_data.copy()
        stream_info['outlet'].push_sample(pose_data)
        stream_info['frame_count'] += 1

    def close_all_pose_streams(self):
        for pid in list(self.pose_streams.keys()):
            self.close_pose_stream(pid)

    # ===== Pose Metrics Streaming Support =====

    def create_metrics_stream(self, participant_id, fps=None):
        """Create a dedicated pose metrics stream for participant

        Streams 12 clinical metrics + 3 metadata values:
        - Head orientation (3): pitch, yaw, roll
        - Neck & posture (3): flexion_angle, sagittal_angle, forward_translation
        - Shoulder elevation (6): ear_ratio L/R, torso_height L/R, composite L/R
        - Metadata (3): confidence, frame_id, timestamp_ms

        Total: 15 channels
        """
        if fps is None:
            fps = self.fps

        # Initialize metrics_streams dict if not exists
        if not hasattr(self, 'metrics_streams'):
            self.metrics_streams = {}

        stream_name = f"{participant_id}_pose_metrics"
        channel_count = 15  # 12 metrics + 3 metadata

        info = StreamInfo(
            name=stream_name,
            type="PoseMetrics",
            channel_count=channel_count,
            nominal_srate=fps,
            channel_format="float32",
            source_id=f"{participant_id}_metrics_uid"
        )

        # Add metadata
        desc = info.desc()
        desc.append_child_value("metric_count", "12")
        desc.append_child_value("metadata_count", "3")

        # Document channel layout
        channels = desc.append_child("channels")

        # Head orientation metrics (3)
        head_metrics = [
            ("head_pitch", "degrees", "Up/down rotation (0=neutral, -=down, +=up)"),
            ("head_yaw", "degrees", "Left/right rotation (0=forward, -=left, +=right)"),
            ("head_roll", "degrees", "Tilt rotation (0=upright, -=left tilt, +=right tilt)")
        ]

        # Neck & posture metrics (3)
        neck_metrics = [
            ("neck_flexion_angle", "degrees", "Head tilt relative to torso"),
            ("neck_sagittal_angle", "degrees", "Y-Z plane angle (forward/backward)"),
            ("forward_head_translation", "meters", "Depth offset (positive=forward)")
        ]

        # Shoulder elevation metrics (6)
        shoulder_metrics = [
            ("shoulder_elevation_left", "ratio", "Ear-to-shoulder ratio left (lower=elevated, range 1.0-2.5)"),
            ("shoulder_elevation_right", "ratio", "Ear-to-shoulder ratio right (lower=elevated, range 1.0-2.5)"),
            ("torso_height_left", "ratio", "Torso height ratio left (higher=elevated, range 1.3-1.8)"),
            ("torso_height_right", "ratio", "Torso height ratio right (higher=elevated, range 1.3-1.8)"),
            ("shoulder_composite_left", "ratio", "Composite score left (positive=elevated, range -0.5 to +0.5)"),
            ("shoulder_composite_right", "ratio", "Composite score right (positive=elevated, range -0.5 to +0.5)")
        ]

        # Metadata fields (3)
        metadata_fields = [
            ("confidence", "normalized", "Overall pose confidence (0.0-1.0)"),
            ("frame_id", "integer", "Frame identifier"),
            ("timestamp_ms", "milliseconds", "Timestamp in milliseconds")
        ]

        # Add all channels
        for label, unit, description in head_metrics + neck_metrics + shoulder_metrics + metadata_fields:
            ch = channels.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", unit)
            ch.append_child_value("description", description)

        outlet = StreamOutlet(info)
        self.metrics_streams[participant_id] = {
            'outlet': outlet,
            'info': info,
            'last_sample': None,
            'frame_count': 0,
            'last_fps_time': time.time()
        }
        print(f"[LSL Helper] Created POSE METRICS stream for {participant_id} ({channel_count} channels, {fps} FPS)")

    def push_metrics_data(self, participant_id, metrics_dict):
        """Push pose metrics to LSL stream

        Args:
            participant_id: Participant identifier
            metrics_dict: Dictionary with PoseMetrics fields (or PoseMetrics object with to_dict())
        """
        if not hasattr(self, 'metrics_streams'):
            return
        if participant_id not in self.metrics_streams:
            return

        stream_info = self.metrics_streams[participant_id]

        # Convert PoseMetrics object to dict if needed
        if hasattr(metrics_dict, 'to_dict'):
            metrics_dict = metrics_dict.to_dict()

        # Build sample array (15 values)
        # Use 0.0 as default for None values (LSL requires numerical data)
        sample = [
            # Head orientation (3)
            metrics_dict.get('head_pitch') or 0.0,
            metrics_dict.get('head_yaw') or 0.0,
            metrics_dict.get('head_roll') or 0.0,
            # Neck & posture (3)
            metrics_dict.get('neck_flexion_angle') or 0.0,
            metrics_dict.get('neck_sagittal_angle') or 0.0,
            metrics_dict.get('forward_head_translation') or 0.0,
            # Shoulder elevation (6)
            metrics_dict.get('shoulder_elevation_left') or 0.0,
            metrics_dict.get('shoulder_elevation_right') or 0.0,
            metrics_dict.get('torso_height_left') or 0.0,
            metrics_dict.get('torso_height_right') or 0.0,
            metrics_dict.get('shoulder_composite_left') or 0.0,
            metrics_dict.get('shoulder_composite_right') or 0.0,
            # Metadata (3)
            metrics_dict.get('confidence') or 0.0,
            float(metrics_dict.get('frame_id') or 0),
            float(metrics_dict.get('timestamp_ms') or 0)
        ]

        # Verify sample size
        expected_size = 15
        if len(sample) != expected_size:
            print(f"[LSL Helper] ERROR: Metrics sample size mismatch for {participant_id}: "
                  f"got {len(sample)}, expected {expected_size}")
            return

        # Check for duplicate
        if stream_info['last_sample'] is not None and sample == stream_info['last_sample']:
            return

        stream_info['last_sample'] = sample.copy()
        stream_info['outlet'].push_sample(sample)
        stream_info['frame_count'] += 1

    def close_metrics_stream(self, participant_id):
        """Close a dedicated metrics stream for participant"""
        if hasattr(self, 'metrics_streams') and participant_id in self.metrics_streams:
            del self.metrics_streams[participant_id]
            print(f"[LSL Helper] Closed METRICS stream for {participant_id}")

    def close_all_metrics_streams(self):
        """Close all metrics streams"""
        if hasattr(self, 'metrics_streams'):
            for pid in list(self.metrics_streams.keys()):
                self.close_metrics_stream(pid)

    # ===== End Pose Metrics Support =====

    # ===== Bluetooth Streaming Support =====

    def create_bluetooth_stream(self, participant_id, stream_type, channel_count,
                                 nominal_srate, channel_names=None, channel_units=None,
                                 manufacturer="", model=""):
        """
        Create an LSL stream for Bluetooth device data.

        Args:
            participant_id: Participant identifier (e.g., "P1")
            stream_type: Type of data (e.g., "ecg", "heartrate", "accel")
            channel_count: Number of channels
            nominal_srate: Sampling rate in Hz
            channel_names: List of channel names (optional)
            channel_units: List of channel units (optional)
            manufacturer: Device manufacturer (e.g., "Polar")
            model: Device model (e.g., "H10")
        """
        stream_name = f"{participant_id}_{stream_type}"
        stream_key = f"{participant_id}_{stream_type}"  # Unique key for tracking

        # Create stream info
        info = StreamInfo(
            name=stream_name,
            type=f"Bluetooth{stream_type.capitalize()}",  # e.g., "BluetoothEcg"
            channel_count=channel_count,
            nominal_srate=nominal_srate,
            channel_format="float32",
            source_id=f"{participant_id}_{stream_type}_uid"
        )

        # Add metadata
        desc = info.desc()
        desc.append_child_value("manufacturer", manufacturer)
        desc.append_child_value("model", model)
        desc.append_child_value("data_source", "bluetooth")

        # Document channel layout
        if channel_names or channel_units:
            channels = desc.append_child("channels")
            for i in range(channel_count):
                ch = channels.append_child("channel")
                if channel_names and i < len(channel_names):
                    ch.append_child_value("label", channel_names[i])
                else:
                    ch.append_child_value("label", f"ch_{i}")

                if channel_units and i < len(channel_units):
                    ch.append_child_value("unit", channel_units[i])

        outlet = StreamOutlet(info)

        # Store in separate bluetooth_streams dict
        if not hasattr(self, 'bluetooth_streams'):
            self.bluetooth_streams = {}

        self.bluetooth_streams[stream_key] = {
            'outlet': outlet,
            'info': info,
            'last_sample': None,
            'sample_count': 0,
            'last_fps_time': time.time(),
            'fps_report_time': time.time(),  # For periodic FPS reporting
            'nominal_srate': nominal_srate  # Store sampling rate for FPS calculation
        }

        print(f"[LSL Helper] Created Bluetooth stream '{stream_name}' "
              f"({channel_count} ch @ {nominal_srate} Hz)")
        print(f"[LSL Helper DEBUG] Stream key='{stream_key}', ready to receive data")

    def push_bluetooth_data(self, participant_id, stream_type, samples):
        """
        Push Bluetooth device data to LSL stream.

        Args:
            participant_id: Participant identifier
            stream_type: Type of data (e.g., "ecg", "heartrate")
            samples: List of sample values (length must match channel count)
        """
        if not hasattr(self, 'bluetooth_streams'):
            print(f"[LSL Helper DEBUG] ERROR: bluetooth_streams dict not initialized!")
            return

        stream_key = f"{participant_id}_{stream_type}"

        if stream_key not in self.bluetooth_streams:
            print(f"[LSL Helper DEBUG] ERROR: Stream '{stream_key}' not found!")
            print(f"[LSL Helper DEBUG] Available streams: {list(self.bluetooth_streams.keys())}")
            return

        stream_info = self.bluetooth_streams[stream_key]

        # Verify sample count matches channel count
        expected_channels = stream_info['info'].channel_count()
        if len(samples) != expected_channels:
            print(f"[LSL Helper] WARNING: Bluetooth sample size mismatch for {stream_key}: "
                  f"got {len(samples)}, expected {expected_channels}")
            return

        # Push sample
        stream_info['outlet'].push_sample(samples)
        stream_info['last_sample'] = samples
        stream_info['sample_count'] += 1

        # DEBUG: Periodic success confirmation (every 1 second based on nominal srate)
        nominal_srate = stream_info.get('nominal_srate', 130)
        if stream_info['sample_count'] % nominal_srate == 0:
            elapsed = time.time() - stream_info['last_fps_time']
            fps = nominal_srate / elapsed if elapsed > 0 else 0
            print(f"[LSL Helper DEBUG] Pushed sample #{stream_info['sample_count']} to {stream_key} ({fps:.1f} samples/s)")
            stream_info['last_fps_time'] = time.time()

    def close_bluetooth_stream(self, participant_id, stream_type):
        """Close a specific Bluetooth stream"""
        if not hasattr(self, 'bluetooth_streams'):
            return

        stream_key = f"{participant_id}_{stream_type}"
        if stream_key in self.bluetooth_streams:
            del self.bluetooth_streams[stream_key]
            print(f"[LSL Helper] Closed Bluetooth stream: {stream_key}")

    def close_all_bluetooth_streams(self):
        """Close all Bluetooth streams"""
        if hasattr(self, 'bluetooth_streams'):
            self.bluetooth_streams.clear()
            print("[LSL Helper] Closed all Bluetooth streams")

    # ===== End Bluetooth Support =====

    def close_all_streams(self):
        """Close all LSL streams"""
        # StreamOutlet has no close() method - just clear the dict
        # All outlet references will be cleaned up automatically
        self.streams.clear()
        self.close_all_pose_streams()
        self.close_all_metrics_streams()
        self.close_all_bluetooth_streams()

def lsl_helper_process(command_queue: MPQueue, 
                       data_queue: MPQueue,
                       correlation_buffer_name: str,
                       fps: int = 30,
                       config: dict = None):
    """
    Process function for centralized LSL management with dynamic stream creation.
    
    Args:
        command_queue: Queue for receiving commands
        data_queue: Queue for receiving participant data
        correlation_buffer_name: Shared memory buffer name for correlation output
        fps: Target frame rate
    """
    # Import at the very beginning to catch import errors
    import sys
    import os
    
    # Immediate output to verify process started
    print(f"[DEBUG] LSL process entry point reached, PID: {os.getpid()}", file=sys.stderr, flush=True)
    print(f"[DEBUG] LSL process entry point reached, PID: {os.getpid()}", flush=True)
    
    import traceback
    
    try:
        print("[DEBUG] LSL helper process started", flush=True)
        print(f"[DEBUG] LSL process PID: {os.getpid()}", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"[DEBUG] LSL: Failed at very start - {e}", flush=True)
        traceback.print_exc()
        return
    
    try:
        print("[DEBUG] LSL: Creating LSLHelper instance")
        helper = LSLHelper(fps, config)
        print("[DEBUG] LSL: LSLHelper created successfully")
        
        print("[DEBUG] LSL: Creating ChannelCorrelator")
        # Get correlator window size from config
        correlator_window_size = (config or {}).get('data_streaming', {}).get('lsl', {}).get('correlator_window_size', 60)
        correlator = ChannelCorrelator(window_size=correlator_window_size, fps=fps, config=config)
        print("[DEBUG] LSL: ChannelCorrelator created successfully")

        print("[DEBUG] LSL: Creating SingleParticipantManager for data merging")
        merger = SingleParticipantManager(config=config, enable_query_buffer=False)
        print("[DEBUG] LSL: SingleParticipantManager created successfully")

        correlator_stream_active = False
        streaming_active = False
        # Get max participants from config
        max_participants = (config or {}).get('data_streaming', {}).get('lsl', {}).get('max_participants', 6)

        # Initialize mesh_enabled_per_camera from config
        mesh_enabled_per_camera = {}  # Track mesh state per camera
        startup_mesh = (config or {}).get('startup_mode', {}).get('enable_mesh', False)
        expected_cameras = (config or {}).get('startup_mode', {}).get('camera_count', 1)
        for cam_idx in range(expected_cameras):
            mesh_enabled_per_camera[cam_idx] = startup_mesh
        print(f"[LSL Process] Initialized mesh_enabled_per_camera from config: {mesh_enabled_per_camera}")
        print(f"[LSL Process] Startup mesh setting: {startup_mesh} for {expected_cameras} cameras")
    except Exception as e:
        print(f"[DEBUG] LSL: Failed during initialization - {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return
    
    # Connect to correlation output buffer
    try:
        print(f"[DEBUG] LSL: Connecting to correlation buffer '{correlation_buffer_name}'")
        sys.stdout.flush()
        corr_buffer = shared_memory.SharedMemory(name=correlation_buffer_name)
        print("[DEBUG] LSL: Correlation buffer connected successfully")
        corr_array = np.ndarray((52,), dtype=np.float32, buffer=corr_buffer.buf)
        print("[LSL Process] Connected to correlation buffer")
        sys.stdout.flush()
    except Exception as e:
        print(f"[LSL Process] Failed to connect to correlation buffer: {e}")
        print(f"[DEBUG] LSL: Error details - {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        corr_buffer = None
        corr_array = None
    
    participant_scores = {}  # Store latest scores for correlation
    created_streams = set()  # Track which streams have been created
    running = True

    # Performance monitoring
    last_fps_report = time.time()
    # Get FPS report interval from config
    fps_report_interval = (config or {}).get('data_streaming', {}).get('lsl', {}).get('fps_report_interval', 5.0)
    data_counts = {}

    # Time-window batching state (PERSISTENT across loop iterations)
    batch_data = {}  # {participant_id: [face_data_cam0, face_data_cam1, ...]}
    cameras_reported = set()  # Which cameras have sent data in current batch
    last_batch_flush_time = time.time()  # When we last flushed
    batch_window_ms = (config or {}).get('data_streaming', {}).get('lsl', {}).get('batch_window_ms', 25)  # Configurable timeout
    expected_cameras = (config or {}).get('startup_mode', {}).get('camera_count', 1)  # Number of active cameras

    print("[LSL Process] Started")
    print(f"[LSL Process] Time-window batching: {expected_cameras} cameras, window={batch_window_ms}ms")
    print("[DEBUG] LSL: Entering main loop")
    sys.stdout.flush()

    loop_count = 0
    last_fps_report_time = time.time()
    fps_report_interval = (config or {}).get('data_streaming', {}).get('lsl', {}).get('fps_report_interval', 5.0)

    # Get verbose_debug flag from config
    verbose_debug = (config or {}).get('logging', {}).get('verbose_debug', False)

    # Statistics accumulators for periodic summaries
    stats_window_start = time.time()
    stats = {
        'participant_data_received': 0,
        'faces_processed': 0,
        'pose_data_received': 0,
        'metrics_data_received': 0,
        'batch_flushes': 0,
        'direct_pushes': 0,
        'merged_pushes': 0,
        'correlation_calculations': 0,
        'per_participant': {}  # {pid: {faces: N, quality_sum: X}}
    }

    try:
        while running:
            loop_count += 1

            # LSL samples/sec report (periodic, configurable interval)
            current_time = time.time()
            if current_time - last_fps_report_time >= fps_report_interval:
                fps_stats = helper.get_fps_stats()
                if fps_stats:
                    stats_str = ", ".join([f"{pid}: {fps:.1f} samples/s" for pid, fps in fps_stats.items()])
                    print(f"[LSL] {stats_str}")
                    sys.stdout.flush()
                last_fps_report_time = current_time
            # Check for commands
            try:
                while not command_queue.empty():
                    cmd = command_queue.get_nowait()

                    if cmd['type'] == 'streaming_started':
                        streaming_active = True
                        # Use config value if not provided in command
                        config_max_participants = (config or {}).get('data_streaming', {}).get('lsl', {}).get('max_participants', 6)
                        max_participants = cmd.get('max_participants', config_max_participants)
                        print(f"[LSL Process] ✅ STREAMING STARTED! streaming_active={streaming_active}, max_participants={max_participants}")

                    elif cmd['type'] == 'streaming_stopped':
                        streaming_active = False
                        print(f"[LSL Process] Streaming stopped (streams remain open for quick restart)")

                    elif cmd['type'] == 'create_stream':
                        # Check if mesh is enabled for any camera (use the stored state)
                        include_mesh = any(mesh_enabled_per_camera.values())
                        helper.create_face_stream(
                            cmd['participant_id'],
                            include_mesh=include_mesh
                        )
                        if cmd.get('include_pose', False):
                            helper.create_pose_stream(cmd['participant_id'], cmd.get('fps', helper.fps))
                        print(f"[LSL Process] Created stream for {cmd['participant_id']} (mesh={'enabled' if include_mesh else 'disabled'})")

                    elif cmd['type'] == 'start_comodulation':
                        # NOTE: Comodulation stream is now created DYNAMICALLY when 2+ participants are detected
                        # This prevents creating an empty stream when only 1 participant is present
                        print(f"[COMOD] Received start_comodulation command - stream will be created when 2+ participants detected")
                        print(f"[COMOD] Current state: correlator_stream_active={correlator_stream_active}")
                        # No longer pre-creating the stream here

                    elif cmd['type'] == 'stop_comodulation':
                        # Stop comodulation stream (set flag to false)
                        if correlator_stream_active:
                            print(f"[COMOD] Stopping comodulation stream")
                            correlator_stream_active = False
                            print(f"[COMOD] Comodulation stream stopped (will restart if start_comodulation is called)")
                        else:
                            print(f"[COMOD] Comodulation already stopped, no action needed")

                    elif cmd['type'] == 'close_all_streams':
                        # Destroy all LSL outlets for clean restart
                        print(f"[LSL Process] Closing all LSL streams...")

                        # Close all face landmark streams
                        for pid in list(helper.streams.keys()):
                            del helper.streams[pid]
                        print(f"[LSL Process] Closed all face landmark streams")

                        # Close all pose streams
                        helper.close_all_pose_streams()
                        print(f"[LSL Process] Closed all pose streams")

                        # Close correlator/comodulation stream
                        correlator.close()
                        correlator_stream_active = False
                        print(f"[LSL Process] Closed comodulation stream")

                        # Clear created streams tracker to force recreation
                        created_streams.clear()
                        print(f"[LSL Process] Cleared created_streams tracker")

                        print(f"[LSL Process] ✅ All streams destroyed (ready for fresh restart)")

                    elif cmd['type'] == 'config_update':
                        # Handle mesh configuration updates
                        # CRITICAL FIX: Moved from data_queue to command_queue to prevent race condition
                        camera_index = cmd['camera_index']
                        mesh_enabled_val = cmd['mesh_enabled']
                        mesh_enabled_per_camera[camera_index] = mesh_enabled_val
                        print(f"[LSL Process] Camera {camera_index} mesh {'enabled' if mesh_enabled_val else 'disabled'}")

                    elif cmd['type'] == 'stop':
                        running = False
                        streaming_active = False
                        break

                    elif cmd['type'] == 'create_pose_stream':
                        helper.create_pose_stream(cmd['participant_id'], cmd.get('fps', fps))

                    elif cmd['type'] == 'create_bluetooth_stream':
                        # Create Bluetooth device stream
                        helper.create_bluetooth_stream(
                            participant_id=cmd['participant_id'],
                            stream_type=cmd['stream_type'],
                            channel_count=cmd['channel_count'],
                            nominal_srate=cmd['nominal_srate'],
                            channel_names=cmd.get('channel_names'),
                            channel_units=cmd.get('channel_units'),
                            manufacturer=cmd.get('manufacturer', ''),
                            model=cmd.get('model', '')
                        )

                    elif cmd['type'] == 'close_bluetooth_stream':
                        # Close specific Bluetooth stream
                        helper.close_bluetooth_stream(
                            participant_id=cmd['participant_id'],
                            stream_type=cmd['stream_type']
                        )

                    elif cmd['type'] == 'bluetooth_data':
                        # Push Bluetooth data to LSL
                        # This is handled via data_queue (see below in data processing)
                        pass

                    elif cmd['type'] == 'close_stream':
                        pid = cmd['participant_id']
                        if pid in helper.streams:
                            # StreamOutlet has no close() method - just delete the reference
                            del helper.streams[pid]
                            print(f"[LSL Process] Closed stream for {pid}")
                        if pid in helper.pose_streams:
                            helper.close_pose_stream(pid)
                        if pid in created_streams:
                            created_streams.remove(pid)

                    elif cmd['type'] == 'force_recreate_streams':
                        mesh_enabled = cmd.get('mesh_enabled', False)
                        print(f"[LSL Process] Force recreating all streams with mesh={mesh_enabled}")

                        # Update all camera states (or initialize if empty)
                        if mesh_enabled_per_camera:
                            # Update existing cameras
                            for cam_idx in mesh_enabled_per_camera:
                                mesh_enabled_per_camera[cam_idx] = mesh_enabled
                        else:
                            # Initialize for all expected cameras (from config)
                            for cam_idx in range(expected_cameras):
                                mesh_enabled_per_camera[cam_idx] = mesh_enabled
                        print(f"[LSL Process] Updated mesh_enabled_per_camera: {mesh_enabled_per_camera} (expected_cameras={expected_cameras})")

                        # Recreate all active streams
                        # StreamOutlet has no close() method - just delete all references
                        for pid in list(helper.streams.keys()):
                            del helper.streams[pid]

                        # Clear created streams set to force recreation
                        created_streams.clear()

                        print(f"[LSL Process] All streams cleared, will recreate on next data")

            except:
                pass

            # Process data
            try:
                data_processed = False

                while not data_queue.empty():
                    data = data_queue.get_nowait()
                    data_processed = True

                    # Check if data has 'type' field
                    if 'type' not in data:
                        # This is regular face/tracking data, not a config update
                        pass  # Will be processed below
                    # NOTE: config_update handling moved to command_queue (line 309-315) to prevent race condition
                    elif data['type'] == 'participant_data':
                        # Handle face data from the new GPU pipeline
                        if 'faces' in data:
                            camera_idx = data.get('camera_index', 0)
                            frame_id = data.get('frame_id', -1)  # For logging only

                            # Update statistics
                            stats['participant_data_received'] += 1
                            stats['faces_processed'] += len(data['faces'])

                            if verbose_debug:
                                print(f"[LSL RECEIVE DEBUG] Received participant_data: camera={camera_idx}, frame_id={frame_id}, faces={len(data['faces'])}")
                                print(f"[LSL RECEIVE DEBUG] streaming_active={streaming_active}, created_streams={created_streams}, max_participants={max_participants}")

                            # STEP 1: Accumulate face data from this camera into shared batch
                            # Track which camera reported
                            cameras_reported.add(camera_idx)

                            for face in data['faces']:
                                pid_str = face.get('participant_id', f"Camera{camera_idx}_P{face.get('track_id', 0)}")
                                # Extract numeric ID from "P1", "P2" format
                                try:
                                    pid_num = int(pid_str.replace('P', ''))
                                except:
                                    pid_num = 1  # Fallback

                                if verbose_debug:
                                    print(f"[LSL RECEIVE DEBUG] Processing face: pid={pid_str}, has_blend_scores={'blend_scores' in face}, has_mesh_data={'mesh_data' in face}")

                                # Validate required data (only landmarks + blend_scores)
                                if 'landmarks' in face and 'blend_scores' in face:
                                    landmarks = np.array(face['landmarks'])

                                    # Get or derive bbox from landmarks
                                    if 'bbox' in face:
                                        bbox = np.array(face['bbox'])
                                    else:
                                        # Derive bbox from landmark bounds
                                        x_coords = landmarks[:, 0]
                                        y_coords = landmarks[:, 1]
                                        bbox = np.array([x_coords.min(), y_coords.min(),
                                                        x_coords.max(), y_coords.max()])
                                        print(f"[LSL] Derived bbox from landmarks for {pid_str}: {bbox}")

                                    # Get frame_shape or use default
                                    frame_shape = tuple(face['frame_shape']) if 'frame_shape' in face else (720, 1280)

                                    # Accumulate into shared batch_data (across all cameras in current window)
                                    if pid_str not in batch_data:
                                        batch_data[pid_str] = []
                                    batch_data[pid_str].append({
                                        'pid_num': pid_num,
                                        'camera_idx': camera_idx,
                                        'landmarks': landmarks,
                                        'blendshapes': np.array(face['blend_scores']),
                                        'bbox': bbox,
                                        'frame_shape': frame_shape,
                                        'mesh_data': face.get('mesh_data', None)  # Preserve mesh_data from GUI worker
                                    })
                                else:
                                    print(f"[LSL] Skipped {pid_str}: missing landmarks or blend_scores")

                        else:
                            if verbose_debug:
                                print(f"[LSL RECEIVE DEBUG] participant_data has no 'faces' key!")

                    elif data['type'] == 'pose_data':
                        pid = data['participant_id']
                        camera_idx = data.get('camera_index', -1)
                        frame_id = data.get('frame_id', -1)
                        pose_data = data.get('pose_data', [])

                        # Update statistics
                        stats['pose_data_received'] += 1

                        # ALWAYS-ON: Log first 3 pose_data receptions
                        if stats['pose_data_received'] <= 3:
                            print(f"[LSL Process] 📥 POSE DATA RECEIVED #{stats['pose_data_received']}: "
                                  f"camera={camera_idx}, frame={frame_id}, participant={pid}, "
                                  f"landmarks={len(pose_data)}, streaming_active={streaming_active}")

                        if verbose_debug:
                            print(f"[LSL RECEIVE DEBUG] Received pose_data: camera={camera_idx}, frame_id={frame_id}, "
                                  f"participant={pid}, pose_landmarks={len(pose_data)}, streaming_active={streaming_active}")

                        # LIFO FRAME-ID FILTERING: Enforce unique frame sending
                        # Track latest frame_id pushed per participant to ensure LIFO behavior
                        if not hasattr(helper, '_lsl_latest_pushed_frame_id'):
                            helper._lsl_latest_pushed_frame_id = {}

                        # Create composite key (participant + camera)
                        stream_key = f"{camera_idx}_{pid}"

                        # Check if this frame was already pushed or is older than latest
                        if stream_key in helper._lsl_latest_pushed_frame_id and frame_id <= helper._lsl_latest_pushed_frame_id[stream_key]:
                            # Skip pushing duplicate or stale frame (LIFO enforcement)
                            if verbose_debug:
                                print(f"[LSL SKIP] Frame {frame_id} skipped for {pid} (stale/duplicate, latest={helper._lsl_latest_pushed_frame_id[stream_key]})")
                            if 'pose_frames_skipped' not in stats:
                                stats['pose_frames_skipped'] = 0
                            stats['pose_frames_skipped'] += 1
                        else:
                            # Dynamically create pose stream if needed
                            if streaming_active and pid not in helper.pose_streams:
                                # Get FPS from data or use default
                                data_fps = data.get('fps', fps)
                                print(f"[LSL Process] 🎯 Dynamically creating pose stream for {pid} ({data_fps} FPS)")
                                helper.create_pose_stream(pid, data_fps)
                                print(f"[LSL Process] ✅ Pose stream created for {pid}")
                            elif not streaming_active:
                                print(f"[LSL Process] ⚠️  Cannot create pose stream: streaming_active=False")

                            if pid in helper.pose_streams:
                                helper.push_pose_data(pid, pose_data)

                                # Update latest pushed frame_id (LIFO tracking)
                                helper._lsl_latest_pushed_frame_id[stream_key] = frame_id

                                if verbose_debug:
                                    print(f"[LSL POSE PUSHED] Frame {frame_id} pushed for {pid} ({len(pose_data)} values)")

                    elif data['type'] == 'metrics_data':
                        # Handle pose metrics data
                        pid = data['participant_id']
                        camera_idx = data.get('camera_index', -1)
                        frame_id = data.get('frame_id', -1)
                        metrics = data.get('metrics', {})

                        # Update statistics
                        if 'metrics_data_received' not in stats:
                            stats['metrics_data_received'] = 0
                        stats['metrics_data_received'] += 1

                        if verbose_debug:
                            print(f"[LSL RECEIVE DEBUG] Received metrics_data: camera={camera_idx}, frame_id={frame_id}, "
                                  f"participant={pid}, metrics_count={len(metrics)}, streaming_active={streaming_active}")

                        # LIFO FRAME-ID FILTERING: Enforce unique frame sending (same as pose_data)
                        if not hasattr(helper, '_lsl_latest_pushed_metrics_frame_id'):
                            helper._lsl_latest_pushed_metrics_frame_id = {}

                        metrics_key = f"{camera_idx}_{pid}"

                        # Check if this frame was already pushed or is older than latest
                        if metrics_key in helper._lsl_latest_pushed_metrics_frame_id and frame_id <= helper._lsl_latest_pushed_metrics_frame_id[metrics_key]:
                            # Skip pushing duplicate or stale frame
                            if verbose_debug:
                                print(f"[LSL SKIP] Metrics frame {frame_id} skipped for {pid} (stale/duplicate, latest={helper._lsl_latest_pushed_metrics_frame_id[metrics_key]})")
                            if 'metrics_frames_skipped' not in stats:
                                stats['metrics_frames_skipped'] = 0
                            stats['metrics_frames_skipped'] += 1
                        else:
                            # Dynamically create metrics stream if needed
                            if streaming_active and pid not in helper.metrics_streams:
                                # Get FPS from data or use default
                                data_fps = data.get('fps', fps)
                                print(f"[LSL Process] Dynamically creating metrics stream for {pid} ({data_fps} FPS)")
                                helper.create_metrics_stream(pid, data_fps)

                            if pid in helper.metrics_streams:
                                helper.push_metrics_data(pid, metrics)

                                # Update latest pushed frame_id
                                helper._lsl_latest_pushed_metrics_frame_id[metrics_key] = frame_id

                                if verbose_debug:
                                    print(f"[LSL METRICS PUSHED] Frame {frame_id} pushed for {pid}")

                    elif data['type'] == 'bluetooth_data':
                        # Handle Bluetooth device data
                        pid = data['participant_id']
                        stream_type = data['stream_type']
                        samples = data['samples']

                        # DEBUG: ALWAYS log Bluetooth data (critical for debugging)
                        # Suppressed high-frequency debug output
                        # print(f"[LSL Process DEBUG] Bluetooth data received from queue: {pid}_{stream_type}, {len(samples)} samples")

                        # Push to Bluetooth stream
                        if streaming_active:
                            # Suppressed high-frequency debug output
                            # print(f"[LSL Process DEBUG] streaming_active=True, calling push_bluetooth_data()")
                            helper.push_bluetooth_data(pid, stream_type, samples)
                        else:
                            print(f"[LSL Process DEBUG] WARNING: Bluetooth data received but streaming_active=False!")

                # STEP 2: Time-window flush logic
                # Process batch when all cameras reported OR timeout expired
                current_time = time.time()
                time_since_last_flush = (current_time - last_batch_flush_time) * 1000  # ms
                all_cameras_reported = len(cameras_reported) >= expected_cameras
                timeout_expired = time_since_last_flush >= batch_window_ms
                has_data = len(batch_data) > 0

                should_flush = has_data and (all_cameras_reported or timeout_expired)

                if should_flush:
                    # Update statistics
                    stats['batch_flushes'] += 1

                    flush_reason = "complete" if all_cameras_reported else f"timeout ({time_since_last_flush:.1f}ms)"
                    if verbose_debug:
                        print(f"[LSL BATCH] Flushing: reason={flush_reason}, cameras={cameras_reported}, "
                              f"participants={list(batch_data.keys())}")

                    # Process accumulated batch_data (merged multi-camera data)
                    for pid_str, face_list in batch_data.items():
                        # CRITICAL FIX: Check UNIQUE cameras, not just face count
                        # (Same camera can send multiple frames in one window)
                        unique_cameras = set(fd['camera_idx'] for fd in face_list)

                        # If same camera sent multiple frames, keep only the most recent
                        if len(unique_cameras) == 1 and len(face_list) > 1:
                            # Sort by any available timestamp/order and keep last
                            face_list = [face_list[-1]]  # Most recent frame from this camera
                            if verbose_debug:
                                print(f"[LSL BATCH] {pid_str}: Same camera sent {len(batch_data[pid_str])} frames, using most recent")

                        # Dynamically create stream if it doesn't exist
                        if streaming_active and pid_str not in created_streams and len(created_streams) < max_participants:
                            # Check mesh state with fallback to config if dict is empty
                            if mesh_enabled_per_camera:
                                include_mesh = any(mesh_enabled_per_camera.values())
                            else:
                                # Fallback to config if dict not initialized yet (defensive programming)
                                include_mesh = (config or {}).get('startup_mode', {}).get('enable_mesh', False)
                                print(f"[LSL Process] WARNING: mesh_enabled_per_camera empty during stream creation, using config default: {include_mesh}")

                            # Determine if this will be merged (multi-camera) stream
                            is_merged = len(unique_cameras) > 1
                            helper.create_face_stream(pid_str, include_mesh=include_mesh, is_merged=is_merged)
                            created_streams.add(pid_str)
                            print(f"[LSL Process] Created stream for {pid_str} (unique_cameras={len(unique_cameras)}, mesh={include_mesh})")

                        if len(unique_cameras) == 1:
                            # DIRECT PATH: Single camera, calculate quality and stream
                            face_data = face_list[0]

                            # Calculate quality for this single capture
                            from core.data_processing.quality_estimator import estimate_capture_quality
                            quality_score, quality_breakdown = estimate_capture_quality(
                                bbox=face_data['bbox'],
                                landmarks=face_data['landmarks'],
                                frame_shape=face_data['frame_shape'],
                                stability_tracker=merger.stability_tracker,
                                participant_id=face_data['pid_num'],
                                camera_idx=face_data['camera_idx'],
                                weights=merger.quality_weights
                            )

                            # Prepare quality scores: [size, frontal, stability]
                            quality_scores = [
                                quality_breakdown['size'],
                                quality_breakdown['frontal'],
                                quality_breakdown['stability']
                            ]

                            # Push to LSL (direct, no merging)
                            if pid_str in helper.streams:
                                # Use mesh_data if GUI worker provided it
                                mesh_data = face_data.get('mesh_data', None)

                                # DIAGNOSTIC: Log mesh_data extraction (throttled to first 3)
                                if not hasattr(helper, '_mesh_extract_log_count'):
                                    helper._mesh_extract_log_count = 0
                                if helper._mesh_extract_log_count < 3:
                                    if mesh_data is not None:
                                        print(f"[LSL MESH DIAGNOSTIC] Extracted mesh_data from face_dict for {pid_str} "
                                              f"({len(mesh_data)} values)")
                                    else:
                                        print(f"[LSL MESH DIAGNOSTIC] No mesh_data in face_dict for {pid_str} "
                                              f"(mesh_enabled should be True in GUI worker)")
                                    helper._mesh_extract_log_count += 1

                                helper.push_face_data(
                                    pid_str,
                                    face_data['blendshapes'].tolist(),
                                    mesh_data=mesh_data,
                                    quality_scores=quality_scores
                                )

                                participant_scores[pid_str] = face_data['blendshapes']

                                # Update statistics
                                stats['direct_pushes'] += 1
                                if pid_str not in stats['per_participant']:
                                    stats['per_participant'][pid_str] = {'faces': 0, 'quality_sum': 0.0}
                                stats['per_participant'][pid_str]['faces'] += 1
                                stats['per_participant'][pid_str]['quality_sum'] += quality_score

                                if verbose_debug:
                                    print(f"[LSL DIRECT] Pushed {pid_str} from cam{face_data['camera_idx']} "
                                          f"(quality={quality_score:.3f})")

                                # Count data
                                if pid_str not in data_counts:
                                    data_counts[pid_str] = 0
                                data_counts[pid_str] += 1

                        else:
                            # MERGED PATH: Multi-camera, use merger for quality-weighted average
                            print(f"[LSL MERGE] {pid_str} has {len(unique_cameras)} unique cameras ({len(face_list)} total frames), using merger")

                            # Group faces by camera and take most recent from each
                            camera_faces = {}
                            for face_data in face_list:
                                cam_idx = face_data['camera_idx']
                                if cam_idx not in camera_faces:
                                    camera_faces[cam_idx] = []
                                camera_faces[cam_idx].append(face_data)

                            # Take most recent frame from each camera
                            frames_to_merge = []
                            for cam_idx, cam_face_list in camera_faces.items():
                                most_recent = cam_face_list[-1]  # Last frame from this camera
                                frames_to_merge.append(most_recent)
                                if len(cam_face_list) > 1:
                                    print(f"[LSL MERGE] Camera {cam_idx} sent {len(cam_face_list)} frames, using most recent")

                            # Add selected camera data to merger
                            for face_data in frames_to_merge:
                                merger.add_camera_data(
                                    participant_id=face_data['pid_num'],
                                    camera_idx=face_data['camera_idx'],
                                    landmarks=face_data['landmarks'],
                                    blendshapes=face_data['blendshapes'],
                                    bbox=face_data['bbox'],
                                    frame_shape=face_data['frame_shape']
                                )

                            # Get merged data
                            merged_landmarks, merged_blends = merger.get_merged_data(face_data['pid_num'])

                            if merged_landmarks is not None and merged_blends is not None:
                                # Calculate quality for merged result
                                # Use the quality from merger (weighted average of individual qualities)
                                cameras = merger.accumulator.get(face_data['pid_num'], [])
                                if cameras:
                                    # Average quality scores from all cameras
                                    total_quality = sum(cam.quality for cam in cameras)
                                    avg_quality_breakdown = {
                                        'size': sum(cam.quality_breakdown['size'] for cam in cameras) / len(cameras),
                                        'frontal': sum(cam.quality_breakdown['frontal'] for cam in cameras) / len(cameras),
                                        'stability': sum(cam.quality_breakdown['stability'] for cam in cameras) / len(cameras)
                                    }
                                    quality_scores = [
                                        avg_quality_breakdown['size'],
                                        avg_quality_breakdown['frontal'],
                                        avg_quality_breakdown['stability']
                                    ]
                                else:
                                    quality_scores = [0.0, 0.0, 0.0]

                                # Push merged data to LSL
                                if pid_str in helper.streams:
                                    # If any camera sent mesh_data, use merged landmarks
                                    mesh_data = None
                                    if any(fd.get('mesh_data') is not None for fd in face_list):
                                        mesh_data = merged_landmarks.flatten().tolist()

                                    helper.push_face_data(
                                        pid_str,
                                        merged_blends.tolist(),
                                        mesh_data=mesh_data,
                                        quality_scores=quality_scores
                                    )

                                    participant_scores[pid_str] = merged_blends

                                    # Update statistics
                                    stats['merged_pushes'] += 1
                                    avg_quality = sum(quality_scores) / 3
                                    if pid_str not in stats['per_participant']:
                                        stats['per_participant'][pid_str] = {'faces': 0, 'quality_sum': 0.0}
                                    stats['per_participant'][pid_str]['faces'] += 1
                                    stats['per_participant'][pid_str]['quality_sum'] += avg_quality

                                    if verbose_debug:
                                        print(f"[LSL MERGE] Pushed merged {pid_str} from {len(unique_cameras)} unique cameras "
                                              f"(avg_quality={avg_quality:.3f})")

                                    # Count data
                                    if pid_str not in data_counts:
                                        data_counts[pid_str] = 0
                                    data_counts[pid_str] += 1

                            # Clear merger for this participant
                            merger.clear_frame()

                    # Calculate correlation if we have 2+ participants CURRENTLY DETECTED (AFTER batch processing)
                    # DYNAMIC CREATION: Only create comodulation stream when 2+ participants are actually present
                    if len(participant_scores) >= 2:
                        # Create correlator stream on-demand if not already active
                        if not correlator_stream_active:
                            print(f"[CORRELATOR] 2+ participants detected ({list(participant_scores.keys())}), creating comodulation stream...")
                            correlator.setup_stream(fps=fps)
                            correlator_stream_active = True
                            print("[CORRELATOR] ✅ Comodulation stream created and active")

                        # Use the first two participants (sorted by name for consistency)
                        pids = sorted(participant_scores.keys())[:2]
                        if verbose_debug:
                            print(f"[CORRELATOR DEBUG] Calculating correlation for {pids[0]} and {pids[1]}")
                        corr = correlator.update(
                            participant_scores[pids[0]],
                            participant_scores[pids[1]]
                        )
                        if corr is not None:
                            # Update statistics
                            stats['correlation_calculations'] += 1

                            if verbose_debug:
                                print(f"[CORRELATOR DEBUG] Correlation calculated, max={np.max(corr):.3f}, mean={np.mean(corr):.3f}")
                            # Only stream to LSL (GUI worker owns the correlation buffer for display)
                            if correlator.outlet:
                                correlator.outlet.push_sample(corr.tolist())
                                if verbose_debug:
                                    print(f"[CORRELATOR DEBUG] Pushed correlation to LSL stream")
                        else:
                            if verbose_debug:
                                print(f"[CORRELATOR DEBUG] Correlation returned None (buffering)")
                    else:
                        # Less than 2 participants - no comodulation possible
                        if data_processed and len(participant_scores) > 0 and verbose_debug:
                            print(f"[CORRELATOR DEBUG] Need 2+ participants for correlation, currently have {len(participant_scores)}: {list(participant_scores.keys())}")

                        # OPTIONAL: Could close correlator stream here if desired
                        # For now, leave it open to avoid constant creation/destruction

                    # Reset batch after processing (CRITICAL!)
                    batch_data.clear()
                    cameras_reported.clear()
                    last_batch_flush_time = current_time
                    if verbose_debug:
                        print(f"[LSL BATCH] Batch reset for next window")

            except Exception as e:
                print(f"[LSL Process] Error processing data: {e}")
                import traceback
                traceback.print_exc()

            # Periodic summary report (consolidated from per-frame logs)
            current_time = time.time()
            elapsed_since_report = current_time - stats_window_start
            if elapsed_since_report >= fps_report_interval:
                # Get stream FPS stats for face and bluetooth streams
                stream_fps = helper.get_fps_stats()
                bluetooth_fps = helper.get_bluetooth_fps_stats()

                # Report consolidated statistics
                if data_counts or participant_scores or any(stats.values()) or bluetooth_fps:
                    print(f"\n[LSL Summary] {elapsed_since_report:.1f}s window:")
                    print(f"  Participants: {list(participant_scores.keys())}")
                    print(f"  Received: {stats['participant_data_received']} packets, {stats['faces_processed']} faces, "
                          f"{stats['pose_data_received']} poses, {stats['metrics_data_received']} metrics")
                    print(f"  Batches: {stats['batch_flushes']} flushes ({stats['direct_pushes']} direct, {stats['merged_pushes']} merged)")

                    # LIFO Frame Deduplication Statistics
                    pose_skipped = stats.get('pose_frames_skipped', 0)
                    metrics_skipped = stats.get('metrics_frames_skipped', 0)
                    if pose_skipped > 0 or metrics_skipped > 0:
                        pose_received = stats['pose_data_received']
                        metrics_received = stats['metrics_data_received']
                        pose_sent = pose_received - pose_skipped
                        metrics_sent = metrics_received - metrics_skipped
                        pose_efficiency = (pose_sent / pose_received * 100) if pose_received > 0 else 0
                        metrics_efficiency = (metrics_sent / metrics_received * 100) if metrics_received > 0 else 0
                        print(f"  LIFO Filtering: Pose {pose_sent}/{pose_received} sent ({pose_efficiency:.1f}% unique), "
                              f"Metrics {metrics_sent}/{metrics_received} sent ({metrics_efficiency:.1f}% unique)")

                    # Per-participant quality summary
                    for pid, pid_stats in stats['per_participant'].items():
                        avg_quality = pid_stats['quality_sum'] / pid_stats['faces'] if pid_stats['faces'] > 0 else 0.0
                        stream_fps_val = stream_fps.get(pid, 0)
                        print(f"  {pid}: {pid_stats['faces']} faces pushed, avg_quality={avg_quality:.3f}, stream_fps={stream_fps_val:.1f}")

                    # Bluetooth stream FPS summary
                    if bluetooth_fps:
                        for stream_key, fps_val in sorted(bluetooth_fps.items()):
                            print(f"  Bluetooth {stream_key}: {fps_val:.1f} samples/s")

                    if stats['correlation_calculations'] > 0:
                        print(f"  Correlation: {stats['correlation_calculations']} calculations")

                    # Reset statistics
                    stats['participant_data_received'] = 0
                    stats['faces_processed'] = 0
                    stats['pose_data_received'] = 0
                    stats['metrics_data_received'] = 0
                    stats['pose_frames_skipped'] = 0
                    stats['metrics_frames_skipped'] = 0
                    stats['batch_flushes'] = 0
                    stats['direct_pushes'] = 0
                    stats['merged_pushes'] = 0
                    stats['correlation_calculations'] = 0
                    stats['per_participant'].clear()
                    stats_window_start = current_time

                    # Reset old counters (legacy)
                    data_counts.clear()
                    last_fps_report = current_time

            # Small sleep to prevent CPU spinning
            if not data_processed:
                # Get main loop sleep from config
                main_loop_sleep = (config or {}).get('data_streaming', {}).get('lsl', {}).get('main_loop_sleep', 0.001)
                time.sleep(main_loop_sleep)
    
        # Cleanup
        helper.close_all_streams()
        helper.close_all_pose_streams()
        correlator.close()
        if corr_buffer:
            corr_buffer.close()
            
        print("[LSL Process] Stopped")
    except Exception as e:
        print(f"[DEBUG] LSL Process CRASHED in main loop: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()