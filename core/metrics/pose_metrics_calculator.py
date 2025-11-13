"""
Pose Metrics Calculator

Calculates ergonomic metrics from RTMW3D 3D whole-body pose keypoints.

Metrics include:
- Head orientation (pitch/yaw/roll)
- Neck flexion/extension angle
- Shoulder elevation (shrug detection)

Adapted from test_realtime_pose.py head tilt calculation and extended with
additional biomechanical measurements.

RTMW3D Coordinate System:
--------------------------
Keypoints are provided in camera/image coordinate space with the following conventions:
- X-axis: Horizontal (increases to the right in image)
- Y-axis: Vertical (increases DOWNWARD in image coordinates)
- Z-axis: Depth (distance from camera, positive = further away)
- Units: Meters (m) for 3D coordinates

Keypoint Format:
- Shape: (133, 4) array
- Each keypoint: [x, y, z, confidence]
- confidence: Float in range [0.0, 1.0]

Body Keypoint Indices (COCO 17-point):
- 0: nose, 1-2: eyes, 3-4: ears
- 5-6: shoulders, 7-8: elbows, 9-10: wrists
- 11-12: hips, 13-14: knees, 15-16: ankles
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple

from .metrics_dataclasses import PoseMetrics, MetricsConfig
from .angle_utils import (
    calculate_vector_angle,
    calculate_midpoint,
    validate_keypoints_batch
)

logger = logging.getLogger(__name__)


# RTMW3D Keypoint Indices (COCO body subset)
# Total: 133 keypoints (body=0-16, feet=17-22, face=23-90, hands=91-132)
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16


class PoseMetricsCalculator:
    """
    Calculate ergonomic pose metrics from RTMW3D keypoints.

    Uses 3D spatial coordinates (x, y, z in meters) from RTMW3D pose estimation
    to compute biomechanically accurate angle measurements.

    Attributes:
        config: MetricsConfig with calculation parameters
        min_confidence: Minimum keypoint confidence threshold (0.0-1.0)
    """

    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Initialize metrics calculator.

        Args:
            config: Optional MetricsConfig instance. If None, uses defaults.
        """
        self.config = config if config is not None else MetricsConfig()
        self.min_confidence = self.config.min_confidence

        logger.info(f"PoseMetricsCalculator initialized with min_confidence={self.min_confidence}")

    def calculate_metrics(self, pose_data: Dict) -> Optional[PoseMetrics]:
        """
        Calculate all enabled metrics from pose data.

        Args:
            pose_data: Dictionary from gui_processing_worker._read_pose_data()
                      Expected keys: 'keypoints' (133, 4), 'frame_id', 'timestamp_ms',
                                    'n_persons', 'is_stale', 'recovering'

        Returns:
            PoseMetrics: Calculated metrics, or None if pose_data is invalid

        Example:
            >>> calculator = PoseMetricsCalculator()
            >>> pose_data = worker._read_pose_data(cam_idx, frame_id)
            >>> metrics = calculator.calculate_metrics(pose_data)
            >>> if metrics and metrics.head_pitch is not None:
            ...     print(f"Head pitch: {metrics.head_pitch:.1f}°")
        """
        if pose_data is None:
            return None

        # Extract keypoints array
        keypoints = pose_data.get('keypoints')
        if keypoints is None:
            return None

        # Handle list of person dicts (multi-person case)
        if isinstance(keypoints, list):
            if len(keypoints) == 0:
                return None

            # For now, only process first person
            # TODO: Support multi-person metrics
            first_person = keypoints[0]

            if isinstance(first_person, dict):
                # Multi-person dict format: [{'keypoints': array, ...}, ...]
                keypoints = first_person.get('keypoints')
                if keypoints is None:
                    logger.warning("[METRICS] Multi-person dict missing 'keypoints' key")
                    return None
            else:
                # Direct array format: [array, ...]
                keypoints = first_person

        if not isinstance(keypoints, np.ndarray):
            logger.warning(f"[METRICS] Expected numpy array for keypoints, got {type(keypoints)}")
            return None

        # Validate shape: Expected (133, 4) for [x, y, z, confidence]
        if keypoints.shape != (133, 4):
            logger.warning(f"Unexpected keypoints shape: {keypoints.shape}, expected (133, 4)")
            return None

        # Extract coordinates and scores
        coords = keypoints[:, :3]  # (133, 3) - x, y, z
        scores = keypoints[:, 3]    # (133,) - confidence

        # Coordinate range validation (log warnings for suspicious values)
        coord_max = np.max(np.abs(coords))
        if coord_max > 10.0:  # 10 meters is unreasonably large
            logger.warning(f"[METRICS] Suspicious coordinate values detected: max={coord_max:.2f}m "
                          f"(expected < 10m for typical human pose)")

        # First-call logging: Log coordinate ranges to help debug coordinate system
        if not hasattr(self, '_first_call_logged'):
            self._first_call_logged = True
            # Find valid keypoints (body keypoints with high confidence)
            valid_mask = scores[:17] > 0.5
            if np.any(valid_mask):
                valid_coords = coords[:17][valid_mask]
                logger.info(f"[METRICS] First-call coordinate ranges (valid body keypoints):")
                logger.info(f"  X: [{valid_coords[:, 0].min():.3f}, {valid_coords[:, 0].max():.3f}]m")
                logger.info(f"  Y: [{valid_coords[:, 1].min():.3f}, {valid_coords[:, 1].max():.3f}]m")
                logger.info(f"  Z: [{valid_coords[:, 2].min():.3f}, {valid_coords[:, 2].max():.3f}]m")

        # Initialize metrics with metadata
        metrics = PoseMetrics(
            frame_id=pose_data.get('frame_id', -1),
            timestamp_ms=pose_data.get('timestamp_ms', 0),
            person_id=0  # TODO: Multi-person support
        )

        # Calculate overall confidence (average of body keypoints)
        body_scores = scores[:17]  # COCO 17 body keypoints
        metrics.confidence = float(np.mean(body_scores))

        # Calculate head orientation (pitch/yaw/roll)
        if self.config.enable_head_orientation:
            pitch, yaw, roll = self._calculate_head_orientation(coords, scores)
            metrics.head_pitch = pitch
            metrics.head_yaw = yaw
            metrics.head_roll = roll

        # Calculate neck angle (flexion/extension)
        if self.config.enable_neck_angle:
            metrics.neck_flexion_angle = self._calculate_neck_angle(coords, scores)
            metrics.neck_sagittal_angle = self._calculate_sagittal_neck_angle(coords, scores)
            metrics.forward_head_translation = self._calculate_forward_head_translation(coords, scores)

        # Calculate shoulder elevation (enhanced multi-component system v6.0)
        if self.config.enable_shoulder_metrics:
            (left_ear_ratio, right_ear_ratio,
             left_torso_ratio, right_torso_ratio,
             left_composite, right_composite) = self._calculate_shoulder_shrug(coords, scores)

            # Populate all six shoulder metrics
            metrics.shoulder_elevation_left = left_ear_ratio
            metrics.shoulder_elevation_right = right_ear_ratio
            metrics.torso_height_left = left_torso_ratio
            metrics.torso_height_right = right_torso_ratio
            metrics.shoulder_composite_left = left_composite
            metrics.shoulder_composite_right = right_composite

        return metrics

    def _calculate_head_orientation(self, keypoints: np.ndarray,
                                    scores: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate head orientation (pitch, yaw, roll).

        Currently implements pitch (head tilt up/down) ported from test_realtime_pose.py.
        Yaw (left/right) and roll (tilt) are placeholders for future implementation.

        Args:
            keypoints: Array of shape (133, 3) with [x, y, z] in meters
            scores: Array of shape (133,) with confidence scores

        Returns:
            Tuple[pitch, yaw, roll]: Head orientation angles in degrees, or None for each if unavailable
                - pitch: Up/down rotation (0° = neutral, - = looking down, + = looking up)
                - yaw: Left/right rotation (0° = forward, - = left, + = right)
                - roll: Tilt rotation (0° = upright, - = tilted left, + = tilted right)
        """
        # Calculate pitch using ported head tilt algorithm
        pitch = self._calculate_head_tilt_angle(keypoints, scores)

        # TODO: Implement yaw and roll calculations
        # These require additional geometric analysis of face landmarks or ear positions
        yaw = None
        roll = None

        return pitch, yaw, roll

    def _calculate_head_tilt_angle(self, keypoints: np.ndarray,
                                   scores: np.ndarray) -> Optional[float]:
        """
        Calculate head tilt (pitch) relative to neck using vector angle method.

        PORTED FROM: test_realtime_pose.py:calculate_head_tilt_angle()

        Measures the angle between:
        - Neck vector (shoulder midpoint → ear midpoint)
        - Face forward vector (ear midpoint → nose)

        Args:
            keypoints: (133, 3) array with [x, y, z] in meters (camera space)
            scores: (133,) confidence scores

        Returns:
            float: Head tilt in degrees
                   0° = neutral (face perpendicular to neck)
                   Negative = looking down
                   Positive = looking up
                   Returns None if keypoints not confident enough
        """
        # Validate required keypoints: shoulders (5,6), ears (3,4), nose (0)
        required_indices = [KP_NOSE, KP_LEFT_EAR, KP_RIGHT_EAR,
                           KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER]

        if not validate_keypoints_batch(keypoints, scores, required_indices, self.min_confidence):
            return None

        # Calculate neck base (shoulder midpoint)
        left_shoulder = keypoints[KP_LEFT_SHOULDER]
        right_shoulder = keypoints[KP_RIGHT_SHOULDER]
        neck_base = calculate_midpoint(left_shoulder, right_shoulder)

        if neck_base is None:
            return None

        # Calculate ear midpoint (top of neck)
        left_ear = keypoints[KP_LEFT_EAR]
        right_ear = keypoints[KP_RIGHT_EAR]
        ear_midpoint = calculate_midpoint(left_ear, right_ear)

        if ear_midpoint is None:
            return None

        # Apply anatomical offset correction to ear midpoint
        # The ear midpoint is anatomically higher than the nose, creating a negative bias
        # We calculate a vertical offset based on inter-ear distance to correct this
        inter_ear_distance = np.linalg.norm(right_ear - left_ear)
        vertical_offset = inter_ear_distance * self.config.ear_to_nose_drop_ratio

        # Create corrected ear midpoint (move DOWN in Z-axis toward nose level)
        # Note: Uses index [2] (Z-axis) which is displayed as vertical in visualization
        # Index [0]=X (horizontal), [1]=Y (depth), [2]=Z (VERTICAL)
        ear_midpoint_corrected = ear_midpoint.copy()
        ear_midpoint_corrected[2] -= vertical_offset

        logger.debug(
            f"Head tilt correction: inter_ear_dist={inter_ear_distance:.4f}m, "
            f"offset={vertical_offset:.4f}m (ratio={self.config.ear_to_nose_drop_ratio:.2f})"
        )

        # Neck vector (from neck base upward to ears)
        neck_vector = ear_midpoint - neck_base

        # Face forward vector (from ears forward to nose, using corrected ear position)
        nose = keypoints[KP_NOSE]
        face_vector = nose - ear_midpoint_corrected

        # Calculate angle between vectors
        angle_deg = calculate_vector_angle(neck_vector, face_vector, degrees=True)

        if angle_deg is None:
            return None

        # Convert to head tilt: 0° when perpendicular (90° angle)
        # Inverted: Negative when looking down, Positive when looking up
        head_tilt = 90.0 - angle_deg

        logger.debug(f"Head tilt angle: {head_tilt:.2f}°")

        return float(head_tilt)

    def _calculate_neck_angle(self, keypoints: np.ndarray,
                             scores: np.ndarray) -> Optional[float]:
        """
        Calculate neck flexion/extension angle (head tilt relative to torso).

        Measures angle between:
        - Torso vector (hip midpoint → shoulder midpoint) - vertical spine reference
        - Neck vector (shoulder midpoint → ear midpoint) - neck extension

        Args:
            keypoints: (133, 3) array with [x, y, z] in meters
            scores: (133,) confidence scores

        Returns:
            float: Neck flexion/extension angle in degrees
                   0° = neutral (neck perfectly aligned with torso)
                   Small angles (e.g., 10-30°) = slight forward flexion or backward extension
                   90° = neck perpendicular to torso (extreme flexion/extension)
                   Returns None if keypoints not confident enough

            Note: The sign (positive/negative) depends on the direction of deviation.
                  Typically, forward flexion shows as positive angles.
        """
        # Validate required keypoints: hips (11,12), shoulders (5,6), ears (3,4)
        required_indices = [KP_LEFT_HIP, KP_RIGHT_HIP,
                           KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
                           KP_LEFT_EAR, KP_RIGHT_EAR]

        if not validate_keypoints_batch(keypoints, scores, required_indices, self.min_confidence):
            return None

        # Calculate hip midpoint (base of torso)
        left_hip = keypoints[KP_LEFT_HIP]
        right_hip = keypoints[KP_RIGHT_HIP]
        hip_midpoint = calculate_midpoint(left_hip, right_hip)

        if hip_midpoint is None:
            return None

        # Calculate shoulder midpoint (top of torso)
        left_shoulder = keypoints[KP_LEFT_SHOULDER]
        right_shoulder = keypoints[KP_RIGHT_SHOULDER]
        shoulder_midpoint = calculate_midpoint(left_shoulder, right_shoulder)

        if shoulder_midpoint is None:
            return None

        # Calculate ear midpoint (top of neck)
        left_ear = keypoints[KP_LEFT_EAR]
        right_ear = keypoints[KP_RIGHT_EAR]
        ear_midpoint = calculate_midpoint(left_ear, right_ear)

        if ear_midpoint is None:
            return None

        # Torso vector (from hips upward to shoulders) - vertical reference
        torso_vector = shoulder_midpoint - hip_midpoint

        # Neck vector (from shoulders upward to ears)
        neck_vector = ear_midpoint - shoulder_midpoint

        # Calculate angle between torso and neck
        angle_deg = calculate_vector_angle(torso_vector, neck_vector, degrees=True)

        if angle_deg is None:
            return None

        # Convert to flexion/extension angle
        # Interpretation:
        # - angle_deg from calculate_vector_angle is 0° when vectors are aligned (parallel)
        # - angle_deg is 180° when vectors are opposite directions
        # - For ergonomic interpretation, we want 0° = neutral (aligned)
        #
        # Since both torso and neck point upward when aligned:
        # - 0° = perfectly aligned (neutral posture)
        # - Small positive angles = slight deviation from neutral
        # - For flexion (forward bend), neck tilts forward from torso alignment
        neck_angle = angle_deg

        return float(neck_angle)

    def _calculate_sagittal_neck_angle(self, keypoints: np.ndarray,
                                       scores: np.ndarray) -> Optional[float]:
        """
        Calculate neck angle in sagittal plane (Y-Z plane) for forward/backward detection.

        This method projects the neck and torso vectors onto the sagittal plane (side view)
        to isolate forward/backward head movement. Works robustly in frontal, lateral,
        and oblique camera views.

        Args:
            keypoints: (133, 3) array with [x, y, z] in meters
            scores: (133,) confidence scores

        Returns:
            float: Sagittal neck angle in degrees
                   0° = neutral (neck aligned with torso in sagittal plane)
                   Positive angles = deviation from neutral (forward or backward)
                   Returns None if keypoints not confident enough

        Note:
            Sagittal plane = Y-Z plane (vertical + depth), removes lateral (X) component
            This isolates forward head posture visible in frontal views
        """
        # Validate required keypoints: hips (11,12), shoulders (5,6), ears (3,4)
        required_indices = [KP_LEFT_HIP, KP_RIGHT_HIP,
                           KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
                           KP_LEFT_EAR, KP_RIGHT_EAR]

        if not validate_keypoints_batch(keypoints, scores, required_indices, self.min_confidence):
            return None

        # Calculate midpoints
        hip_midpoint = calculate_midpoint(keypoints[KP_LEFT_HIP], keypoints[KP_RIGHT_HIP])
        shoulder_midpoint = calculate_midpoint(keypoints[KP_LEFT_SHOULDER], keypoints[KP_RIGHT_SHOULDER])
        ear_midpoint = calculate_midpoint(keypoints[KP_LEFT_EAR], keypoints[KP_RIGHT_EAR])

        if hip_midpoint is None or shoulder_midpoint is None or ear_midpoint is None:
            return None

        # Project vectors onto sagittal (Y-Z) plane by setting X=0
        # Torso vector (hip → shoulder)
        torso_vector = shoulder_midpoint - hip_midpoint
        torso_sagittal = np.array([0, torso_vector[1], torso_vector[2]])

        # Neck vector (shoulder → ear)
        neck_vector = ear_midpoint - shoulder_midpoint
        neck_sagittal = np.array([0, neck_vector[1], neck_vector[2]])

        # Calculate angle in sagittal plane
        angle_deg = calculate_vector_angle(torso_sagittal, neck_sagittal, degrees=True)

        if angle_deg is None:
            return None

        logger.debug(f"Sagittal neck angle: {angle_deg:.2f}° (Y-Z plane projection)")

        return float(angle_deg)

    def _calculate_forward_head_translation(self, keypoints: np.ndarray,
                                            scores: np.ndarray) -> Optional[float]:
        """
        Calculate forward head translation (depth offset from shoulder plane).

        Measures how far forward the head is positioned relative to the shoulders
        using Z-axis (depth) coordinates. This metric is very sensitive to forward
        head posture in frontal camera views.

        Args:
            keypoints: (133, 3) array with [x, y, z] in meters
            scores: (133,) confidence scores

        Returns:
            float: Forward translation in meters
                   Positive = head forward of shoulders (forward head posture)
                   Negative = head backward of shoulders
                   Typical range: -0.02 to +0.02m (normal), >+0.05m (forward head posture)
                   Returns None if keypoints not confident enough

        Note:
            Uses nose position (more sensitive than ears for frontal detection)
            Z-axis: depth from camera (positive = further away)
        """
        # Validate required keypoints: shoulders (5,6), nose (0)
        required_indices = [KP_NOSE, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER]

        if not validate_keypoints_batch(keypoints, scores, required_indices, self.min_confidence):
            return None

        # Get nose and shoulder midpoint
        nose = keypoints[KP_NOSE]
        shoulder_midpoint = calculate_midpoint(keypoints[KP_LEFT_SHOULDER], keypoints[KP_RIGHT_SHOULDER])

        if shoulder_midpoint is None:
            return None

        # Calculate forward translation (positive = head closer to camera than shoulders)
        # Z increases with distance from camera, so:
        # If nose_z < shoulder_z → nose is closer → forward head posture
        forward_translation = shoulder_midpoint[2] - nose[2]

        logger.debug(f"Forward head translation: {forward_translation:.4f}m ({forward_translation*100:.1f}cm)")

        return float(forward_translation)

    def _calculate_shoulder_shrug(self, keypoints: np.ndarray,
                                  scores: np.ndarray) -> Tuple[
                                      Optional[float], Optional[float],  # ear-shoulder ratios
                                      Optional[float], Optional[float],  # torso height ratios
                                      Optional[float], Optional[float]   # composite scores
                                  ]:
        """
        Calculate enhanced shoulder elevation metrics (v6.0 - Multi-Component System).

        Measures three complementary aspects of shoulder elevation:
        1. Ear-to-shoulder distance ratio (shoulders moving toward head)
        2. Torso height ratio (shoulders moving away from hips)
        3. Composite elevation score (combines both with configurable weights)

        All metrics are normalized by shoulder width for distance-invariant measurements.

        Args:
            keypoints: (133, 3) array with [x, y, z] in meters
            scores: (133,) confidence scores

        Returns:
            Tuple of 6 values (all Optional[float], None if insufficient confidence):
                - left_ear_ratio: Left ear-to-shoulder distance / shoulder_width
                                  (lower = elevated, typical range: 1.0-2.5)
                - right_ear_ratio: Right ear-to-shoulder distance / shoulder_width
                                   (lower = elevated, typical range: 1.0-2.5)
                - left_torso_ratio: Torso height / shoulder_width (left side)
                                    (higher = elevated, typical range: 1.3-1.8)
                - right_torso_ratio: Torso height / shoulder_width (right side)
                                     (higher = elevated, typical range: 1.3-1.8)
                - left_composite: Weighted combination of deviations from neutral
                                  (positive = elevated, 0 = neutral, typical range: -0.5 to +0.5)
                - right_composite: Weighted combination of deviations from neutral
                                   (positive = elevated, 0 = neutral, typical range: -0.5 to +0.5)

        Note:
            - Ear-shoulder metric: Captures shoulders moving toward head (shrugging up)
            - Torso height metric: Captures shoulders moving away from hips (elevation from pelvis)
            - Composite score: Combines both signals with configurable neutral baselines
            - All normalized by shoulder width for distance invariance
        """
        # Validate required keypoints: ears (3,4), shoulders (5,6), hips (11,12)
        required_indices = [KP_LEFT_EAR, KP_RIGHT_EAR,
                           KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
                           KP_LEFT_HIP, KP_RIGHT_HIP]

        if not validate_keypoints_batch(keypoints, scores, required_indices, self.min_confidence):
            return None, None, None, None, None, None

        # --- Calculate midpoints ---
        left_ear = keypoints[KP_LEFT_EAR]
        right_ear = keypoints[KP_RIGHT_EAR]
        ear_midpoint = calculate_midpoint(left_ear, right_ear)

        left_shoulder = keypoints[KP_LEFT_SHOULDER]
        right_shoulder = keypoints[KP_RIGHT_SHOULDER]
        shoulder_midpoint = calculate_midpoint(left_shoulder, right_shoulder)

        left_hip = keypoints[KP_LEFT_HIP]
        right_hip = keypoints[KP_RIGHT_HIP]
        hip_midpoint = calculate_midpoint(left_hip, right_hip)

        if ear_midpoint is None or shoulder_midpoint is None or hip_midpoint is None:
            return None, None, None, None, None, None

        # --- Component 1: Ear-to-Shoulder Ratios (original metric) ---
        left_ear_distance = float(np.linalg.norm(ear_midpoint - left_shoulder))
        right_ear_distance = float(np.linalg.norm(ear_midpoint - right_shoulder))

        # Calculate shoulder width (normalization reference)
        shoulder_width = float(np.linalg.norm(right_shoulder - left_shoulder))

        # Avoid division by zero
        if shoulder_width < 1e-6:
            return None, None, None, None, None, None

        # Normalize ear-shoulder distances by shoulder width
        left_ear_ratio = left_ear_distance / shoulder_width
        right_ear_ratio = right_ear_distance / shoulder_width

        # --- Component 2: Torso Height Ratios (new metric) ---
        # Torso height = vertical distance from hips to shoulders
        # We calculate individual left/right torso heights for laterality
        left_torso_height = float(np.linalg.norm(left_shoulder - hip_midpoint))
        right_torso_height = float(np.linalg.norm(right_shoulder - hip_midpoint))

        # Normalize by shoulder width for distance invariance
        left_torso_ratio = left_torso_height / shoulder_width
        right_torso_ratio = right_torso_height / shoulder_width

        # --- Component 3: Composite Scores (new metric) ---
        # Deviation from neutral values (configured in metrics_settings)
        neutral_ear_ratio = self.config.neutral_ear_shoulder_ratio
        neutral_torso_ratio = self.config.neutral_torso_height_ratio
        ear_weight = self.config.ear_component_weight
        torso_weight = self.config.torso_component_weight

        # Calculate deviations
        # Ear-shoulder: neutral - current (positive when elevated/shrugged)
        left_ear_deviation = neutral_ear_ratio - left_ear_ratio
        right_ear_deviation = neutral_ear_ratio - right_ear_ratio

        # Torso height: current - neutral (positive when elevated)
        left_torso_deviation = left_torso_ratio - neutral_torso_ratio
        right_torso_deviation = right_torso_ratio - neutral_torso_ratio

        # Weighted combination (both components now positive when elevated)
        left_composite = (ear_weight * left_ear_deviation) + (torso_weight * left_torso_deviation)
        right_composite = (ear_weight * right_ear_deviation) + (torso_weight * right_torso_deviation)

        return (left_ear_ratio, right_ear_ratio,
                left_torso_ratio, right_torso_ratio,
                left_composite, right_composite)

    def update_config(self, config: MetricsConfig) -> None:
        """
        Update calculator configuration.

        Args:
            config: New MetricsConfig instance
        """
        self.config = config
        self.min_confidence = config.min_confidence
        logger.info(f"Metrics calculator config updated: min_confidence={self.min_confidence}")
