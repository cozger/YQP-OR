"""
Pose Metrics Module

This module provides ergonomic pose metrics calculations from RTMW3D 3D keypoints.
Metrics include head orientation, neck angles, and shoulder elevation measurements.
"""

from .metrics_dataclasses import PoseMetrics, MetricsConfig
from .pose_metrics_calculator import PoseMetricsCalculator
from .angle_utils import (
    calculate_vector_angle,
    calculate_midpoint,
    normalize_vector,
    validate_keypoint,
    validate_keypoints_batch
)

__all__ = [
    'PoseMetrics',
    'MetricsConfig',
    'PoseMetricsCalculator',
    'calculate_vector_angle',
    'calculate_midpoint',
    'normalize_vector',
    'validate_keypoint',
    'validate_keypoints_batch'
]
