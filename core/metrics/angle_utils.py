"""
Angle Calculation Utilities

Provides 3D vector and angle computation utilities for pose metrics.
All angles are returned in degrees unless otherwise specified.
"""

import numpy as np
from typing import Optional, Tuple, List


def calculate_vector_angle(v1: np.ndarray, v2: np.ndarray, degrees: bool = True) -> Optional[float]:
    """
    Calculate angle between two 3D vectors using dot product.

    Args:
        v1: First vector (3D numpy array)
        v2: Second vector (3D numpy array)
        degrees: Return angle in degrees (True) or radians (False)

    Returns:
        float: Angle between vectors, or None if vectors are invalid
               Range: [0, 180] degrees or [0, π] radians

    Example:
        >>> v1 = np.array([1, 0, 0])
        >>> v2 = np.array([0, 1, 0])
        >>> calculate_vector_angle(v1, v2)
        90.0
    """
    # Validate inputs
    if v1 is None or v2 is None:
        return None

    if not isinstance(v1, np.ndarray) or not isinstance(v2, np.ndarray):
        return None

    if v1.shape != (3,) or v2.shape != (3,):
        return None

    # Calculate magnitudes
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)

    # Check for zero vectors
    if mag1 < 1e-8 or mag2 < 1e-8:
        return None

    # Calculate angle using dot product
    dot_product = np.dot(v1, v2)
    cos_angle = dot_product / (mag1 * mag2)

    # Clamp to valid range to handle floating-point errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Calculate angle in radians
    angle_rad = np.arccos(cos_angle)

    # Convert to degrees if requested
    if degrees:
        return float(np.degrees(angle_rad))
    else:
        return float(angle_rad)


def calculate_signed_angle(v1: np.ndarray, v2: np.ndarray, normal: np.ndarray,
                           degrees: bool = True) -> Optional[float]:
    """
    Calculate signed angle between two vectors around a normal axis.

    This is useful for measuring rotation direction (e.g., flexion vs extension).

    Args:
        v1: First vector (3D numpy array) - reference/baseline
        v2: Second vector (3D numpy array) - current position
        normal: Normal vector defining the rotation plane (3D numpy array)
        degrees: Return angle in degrees (True) or radians (False)

    Returns:
        float: Signed angle (-180 to +180 degrees or -π to +π radians),
               or None if vectors are invalid
               Positive = counter-clockwise rotation from v1 to v2 (when viewed along normal)
               Negative = clockwise rotation

    Example:
        >>> # Flexion/extension around lateral axis
        >>> forward = np.array([0, 1, 0])
        >>> tilted = np.array([0.1, 1, 0])
        >>> lateral = np.array([1, 0, 0])
        >>> calculate_signed_angle(forward, tilted, lateral)
    """
    # Get unsigned angle first
    angle = calculate_vector_angle(v1, v2, degrees=False)
    if angle is None:
        return None

    # Calculate cross product to determine sign
    cross = np.cross(v1, v2)

    # Project onto normal to get sign
    sign_value = np.dot(cross, normal)

    # Apply sign
    if sign_value < 0:
        angle = -angle

    # Convert to degrees if requested
    if degrees:
        return float(np.degrees(angle))
    else:
        return float(angle)


def calculate_midpoint(p1: np.ndarray, p2: np.ndarray) -> Optional[np.ndarray]:
    """
    Calculate midpoint between two 3D points.

    Args:
        p1: First point (3D numpy array)
        p2: Second point (3D numpy array)

    Returns:
        np.ndarray: Midpoint (3D numpy array), or None if invalid

    Example:
        >>> p1 = np.array([0, 0, 0])
        >>> p2 = np.array([2, 2, 2])
        >>> calculate_midpoint(p1, p2)
        array([1., 1., 1.])
    """
    if p1 is None or p2 is None:
        return None

    if not isinstance(p1, np.ndarray) or not isinstance(p2, np.ndarray):
        return None

    if p1.shape != (3,) or p2.shape != (3,):
        return None

    return (p1 + p2) / 2.0


def normalize_vector(v: np.ndarray) -> Optional[np.ndarray]:
    """
    Normalize a 3D vector to unit length.

    Args:
        v: Vector to normalize (3D numpy array)

    Returns:
        np.ndarray: Unit vector in same direction, or None if zero vector

    Example:
        >>> v = np.array([3, 4, 0])
        >>> normalize_vector(v)
        array([0.6, 0.8, 0. ])
    """
    if v is None:
        return None

    if not isinstance(v, np.ndarray):
        return None

    if v.shape != (3,):
        return None

    magnitude = np.linalg.norm(v)

    if magnitude < 1e-8:
        return None

    return v / magnitude


def validate_keypoint(keypoint: np.ndarray, score: float,
                     min_confidence: float = 0.3) -> bool:
    """
    Check if a single keypoint is valid for calculations.

    Args:
        keypoint: Keypoint coordinates (3D or 4D array: [x, y, z] or [x, y, z, conf])
        score: Confidence score (0.0-1.0)
        min_confidence: Minimum confidence threshold

    Returns:
        bool: True if keypoint is valid and confident

    Example:
        >>> kp = np.array([1.0, 2.0, 3.0])
        >>> validate_keypoint(kp, 0.8, min_confidence=0.3)
        True
    """
    if keypoint is None or score is None:
        return False

    if not isinstance(keypoint, np.ndarray):
        return False

    # Accept both 3D [x,y,z] and 4D [x,y,z,conf]
    if keypoint.shape not in [(3,), (4,)]:
        return False

    # Check confidence
    if score < min_confidence:
        return False

    # Check for invalid coordinates (NaN, inf, or extremely large values)
    coords = keypoint[:3]  # Only check x,y,z
    if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
        return False

    # Check for unreasonably large coordinates (likely corrupted data)
    if np.any(np.abs(coords) > 100.0):  # 100 meters is clearly invalid
        return False

    return True


def validate_keypoints_batch(keypoints: np.ndarray, scores: np.ndarray,
                             indices: List[int], min_confidence: float = 0.3) -> bool:
    """
    Check if a batch of keypoints are all valid for calculations.

    Args:
        keypoints: Array of keypoints, shape (N, 3) or (N, 4)
        scores: Array of confidence scores, shape (N,)
        indices: List of keypoint indices to validate
        min_confidence: Minimum confidence threshold

    Returns:
        bool: True if ALL specified keypoints are valid

    Example:
        >>> kps = np.random.rand(133, 3)  # 133 keypoints
        >>> scores = np.random.rand(133)
        >>> validate_keypoints_batch(kps, scores, [0, 1, 2], min_confidence=0.3)
    """
    if keypoints is None or scores is None or indices is None:
        return False

    if not isinstance(keypoints, np.ndarray) or not isinstance(scores, np.ndarray):
        return False

    # Check shapes
    if keypoints.ndim != 2 or scores.ndim != 1:
        return False

    if keypoints.shape[1] not in [3, 4]:  # Accept [x,y,z] or [x,y,z,conf]
        return False

    if len(keypoints) != len(scores):
        return False

    # Validate each specified keypoint
    for idx in indices:
        if idx < 0 or idx >= len(keypoints):
            return False

        if not validate_keypoint(keypoints[idx], scores[idx], min_confidence):
            return False

    return True


def project_point_onto_plane(point: np.ndarray, plane_normal: np.ndarray,
                             plane_point: np.ndarray) -> Optional[np.ndarray]:
    """
    Project a 3D point onto a plane defined by normal and point.

    Useful for measuring angles in specific anatomical planes.

    Args:
        point: Point to project (3D numpy array)
        plane_normal: Normal vector of the plane (3D numpy array)
        plane_point: A point on the plane (3D numpy array)

    Returns:
        np.ndarray: Projected point (3D numpy array), or None if invalid

    Example:
        >>> # Project onto XY plane (z=0)
        >>> point = np.array([1, 2, 3])
        >>> normal = np.array([0, 0, 1])
        >>> plane_point = np.array([0, 0, 0])
        >>> project_point_onto_plane(point, normal, plane_point)
        array([1., 2., 0.])
    """
    if point is None or plane_normal is None or plane_point is None:
        return None

    if not all(isinstance(x, np.ndarray) for x in [point, plane_normal, plane_point]):
        return None

    if not all(x.shape == (3,) for x in [point, plane_normal, plane_point]):
        return None

    # Normalize the plane normal
    normal = normalize_vector(plane_normal)
    if normal is None:
        return None

    # Vector from plane point to target point
    v = point - plane_point

    # Distance from point to plane (signed)
    distance = np.dot(v, normal)

    # Project point onto plane
    projected = point - distance * normal

    return projected


def calculate_elevation_angle(point: np.ndarray, reference: np.ndarray,
                              vertical_axis: np.ndarray = np.array([0, 0, 1]),
                              degrees: bool = True) -> Optional[float]:
    """
    Calculate elevation angle of a point relative to a reference point.

    Useful for shoulder shrug measurements (shoulder elevation relative to neutral).

    Args:
        point: Point to measure elevation of (3D numpy array)
        reference: Reference/baseline point (3D numpy array)
        vertical_axis: Vertical axis vector (default: [0, 0, 1] = Z-axis)
        degrees: Return angle in degrees (True) or radians (False)

    Returns:
        float: Elevation angle (0° = same level, positive = elevated above),
               or None if invalid

    Example:
        >>> shoulder = np.array([0, 0, 1.5])  # 1.5m height
        >>> neutral = np.array([0, 0, 1.0])   # 1.0m height (neutral position)
        >>> calculate_elevation_angle(shoulder, neutral)
        # Returns positive angle for elevation
    """
    if point is None or reference is None:
        return None

    # Vector from reference to point
    vector = point - reference

    # Normalize vertical axis
    vertical = normalize_vector(vertical_axis)
    if vertical is None:
        return None

    # Project vector onto vertical axis
    vertical_component = np.dot(vector, vertical)

    # Get horizontal component magnitude
    horizontal_component = np.linalg.norm(vector - vertical_component * vertical)

    # Calculate elevation angle
    angle_rad = np.arctan2(vertical_component, horizontal_component)

    if degrees:
        return float(np.degrees(angle_rad))
    else:
        return float(angle_rad)
