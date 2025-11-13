"""
Metrics Dataclasses

Defines data structures for pose metrics calculated from RTMW3D keypoints.
Follows the pattern established in core/buffer_management/layouts.py.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class PoseMetrics:
    """
    Calculated pose metrics ready for display.

    All angles are in degrees. None values indicate insufficient keypoint confidence
    or missing keypoints required for that specific metric.

    Attributes:
        # Head Orientation (Euler angles)
        head_pitch: Pitch angle (up/down rotation). 0° = neutral,
                    negative = looking down, positive = looking up
        head_yaw: Yaw angle (left/right rotation). 0° = facing forward,
                  negative = facing left, positive = facing right
        head_roll: Roll angle (tilt rotation). 0° = upright,
                   negative = tilted left, positive = tilted right

        # Neck Biomechanics
        neck_flexion_angle: Head flexion/extension relative to torso.
                           0° = neutral, negative = flexed forward,
                           positive = extended backward

        # Shoulder Biomechanics (Enhanced Multi-Component System v6.0)
        shoulder_elevation_left: Left shoulder ear-to-shoulder ratio (dimensionless).
                                 Normalized ear-to-shoulder distance divided by shoulder width.
                                 Lower values = elevated (shrugged), higher values = relaxed.
                                 Typical range: 1.0-2.5
        shoulder_elevation_right: Right shoulder ear-to-shoulder ratio (dimensionless).
                                  Normalized ear-to-shoulder distance divided by shoulder width.
                                  Lower values = elevated (shrugged), higher values = relaxed.
                                  Typical range: 1.0-2.5

        torso_height_left: Left torso height ratio (dimensionless).
                          Torso height (hip-to-shoulder) normalized by shoulder width.
                          Higher values = shoulders elevated (farther from hips), lower = relaxed.
                          Typical range: 1.3-1.8
        torso_height_right: Right torso height ratio (dimensionless).
                           Torso height (hip-to-shoulder) normalized by shoulder width.
                           Higher values = shoulders elevated (farther from hips), lower = relaxed.
                           Typical range: 1.3-1.8

        shoulder_composite_left: Left shoulder composite elevation score (dimensionless).
                                Combines ear-shoulder ratio and torso height deviations from neutral.
                                Positive = elevated (shrugging), negative = relaxed below baseline, 0 = neutral.
                                Typical range: -0.5 to +0.5
        shoulder_composite_right: Right shoulder composite elevation score (dimensionless).
                                 Combines ear-shoulder ratio and torso height deviations from neutral.
                                 Positive = elevated (shrugging), negative = relaxed below baseline, 0 = neutral.
                                 Typical range: -0.5 to +0.5

        # Future expansion slots for additional metrics
        left_elbow_angle: Left elbow flexion angle (0° = straight, 180° = fully flexed)
        right_elbow_angle: Right elbow flexion angle
        left_knee_angle: Left knee flexion angle
        right_knee_angle: Right knee flexion angle
        torso_lean_angle: Forward/backward torso lean

        # Metadata
        confidence: Overall confidence score (0.0-1.0) based on keypoint quality
        frame_id: Frame ID for which these metrics were calculated
        timestamp_ms: Timestamp in milliseconds
        person_id: Person/participant ID (for multi-person tracking)
    """

    # Head Orientation
    head_pitch: Optional[float] = None
    head_yaw: Optional[float] = None
    head_roll: Optional[float] = None

    # Neck Biomechanics
    neck_flexion_angle: Optional[float] = None
    neck_sagittal_angle: Optional[float] = None  # Y-Z plane angle (forward/backward component)
    forward_head_translation: Optional[float] = None  # Depth offset in meters (+ = forward)

    # Shoulder Biomechanics (Enhanced Multi-Component System v6.0)
    shoulder_elevation_left: Optional[float] = None  # Ear-to-shoulder ratio
    shoulder_elevation_right: Optional[float] = None  # Ear-to-shoulder ratio
    torso_height_left: Optional[float] = None  # Torso height ratio
    torso_height_right: Optional[float] = None  # Torso height ratio
    shoulder_composite_left: Optional[float] = None  # Composite elevation score
    shoulder_composite_right: Optional[float] = None  # Composite elevation score

    # Joint Angles (future expansion)
    left_elbow_angle: Optional[float] = None
    right_elbow_angle: Optional[float] = None
    left_knee_angle: Optional[float] = None
    right_knee_angle: Optional[float] = None
    torso_lean_angle: Optional[float] = None

    # Metadata
    confidence: float = 0.0
    frame_id: int = -1
    timestamp_ms: int = 0
    person_id: int = 0

    def to_dict(self):
        """
        Convert to dictionary for backward compatibility.

        Returns:
            dict: Dictionary representation of all fields
        """
        return asdict(self)

    def has_valid_metrics(self) -> bool:
        """
        Check if any metrics are available (not all None).

        Returns:
            bool: True if at least one metric is calculated
        """
        metric_fields = [
            self.head_pitch, self.head_yaw, self.head_roll,
            self.neck_flexion_angle, self.neck_sagittal_angle, self.forward_head_translation,
            self.shoulder_elevation_left, self.shoulder_elevation_right,
            self.torso_height_left, self.torso_height_right,
            self.shoulder_composite_left, self.shoulder_composite_right,
            self.left_elbow_angle, self.right_elbow_angle,
            self.left_knee_angle, self.right_knee_angle,
            self.torso_lean_angle
        ]
        return any(m is not None for m in metric_fields)

    def get_head_orientation_summary(self) -> str:
        """
        Get human-readable summary of head orientation.

        Returns:
            str: Summary text like "Pitch: +15.2° | Yaw: -5.1° | Roll: +2.3°"
        """
        parts = []
        if self.head_pitch is not None:
            parts.append(f"Pitch: {self.head_pitch:+.1f}°")
        if self.head_yaw is not None:
            parts.append(f"Yaw: {self.head_yaw:+.1f}°")
        if self.head_roll is not None:
            parts.append(f"Roll: {self.head_roll:+.1f}°")

        return " | ".join(parts) if parts else "N/A"

    def get_shoulder_summary(self) -> str:
        """
        Get human-readable summary of shoulder metrics (Enhanced v6.0).

        Returns:
            str: Summary text showing composite scores, e.g., "L: +0.12 | R: +0.08"
                 Positive = elevated (shrugging), 0 = neutral, negative = relaxed
        """
        parts = []
        # Show composite scores (most useful single metric)
        if self.shoulder_composite_left is not None:
            parts.append(f"L: {self.shoulder_composite_left:+.2f}")
        if self.shoulder_composite_right is not None:
            parts.append(f"R: {self.shoulder_composite_right:+.2f}")

        return " | ".join(parts) if parts else "N/A"


@dataclass
class MetricsConfig:
    """
    Configuration for metrics calculations.

    Attributes:
        min_confidence: Minimum keypoint confidence threshold (0.0-1.0)
        enable_head_orientation: Calculate head pitch/yaw/roll
        enable_neck_angle: Calculate neck flexion/extension
        enable_shoulder_metrics: Calculate shoulder elevation
        enable_joint_angles: Calculate elbow/knee angles
        shoulder_baseline_ratio: Baseline shoulder-hip ratio for neutral posture
        ear_to_nose_drop_ratio: Anatomical offset ratio for head tilt correction
                                (ear midpoint → nose level, as fraction of inter-ear distance)

        # Enhanced Shoulder Elevation (v6.0)
        neutral_ear_shoulder_ratio: Neutral value for ear-to-shoulder ratio
        neutral_torso_height_ratio: Neutral value for torso height ratio
        ear_component_weight: Weight for ear-shoulder component in composite score
        torso_component_weight: Weight for torso height component in composite score
    """
    min_confidence: float = 0.3
    enable_head_orientation: bool = True
    enable_neck_angle: bool = True
    enable_shoulder_metrics: bool = True
    enable_joint_angles: bool = False  # Disabled by default (future feature)
    shoulder_baseline_ratio: float = 1.0
    ear_to_nose_drop_ratio: float = 0.20  # Anatomical offset: ear level → nose level (20% of inter-ear distance)

    # Enhanced Shoulder Elevation (v6.0)
    neutral_ear_shoulder_ratio: float = 1.5  # Default neutral ear-to-shoulder ratio
    neutral_torso_height_ratio: float = 1.6  # Default neutral torso height ratio
    ear_component_weight: float = 0.5  # Equal weighting for composite
    torso_component_weight: float = 0.5  # Equal weighting for composite

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MetricsConfig':
        """
        Create MetricsConfig from dictionary.

        Handles nested 'shoulder_elevation' section by flattening into top-level fields.

        Args:
            config_dict: Dictionary with configuration values (may contain nested 'shoulder_elevation')

        Returns:
            MetricsConfig: New configuration instance
        """
        # Flatten nested shoulder_elevation config if present
        flattened = dict(config_dict)  # Copy to avoid modifying original
        if 'shoulder_elevation' in config_dict:
            shoulder_config = config_dict['shoulder_elevation']
            # Merge nested fields into top level
            flattened.update(shoulder_config)

        # Extract only valid fields
        valid_fields = {
            'min_confidence', 'enable_head_orientation', 'enable_neck_angle',
            'enable_shoulder_metrics', 'enable_joint_angles', 'shoulder_baseline_ratio',
            'ear_to_nose_drop_ratio', 'neutral_ear_shoulder_ratio',
            'neutral_torso_height_ratio', 'ear_component_weight', 'torso_component_weight'
        }
        filtered = {k: v for k, v in flattened.items() if k in valid_fields}
        return cls(**filtered)
