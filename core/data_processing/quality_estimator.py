"""
Face Capture Quality Estimator

Estimates quality of face detection from MediaPipe without relying on
detection confidence scores (which MediaPipe FaceLandmarker doesn't provide).

Quality Metrics:
1. Face Size Score - Larger, clearer faces score higher
2. Frontal Score - Front-facing faces score higher (less occlusion)
3. Stability Score - Temporally stable detections score higher

Used for weighted merging of multi-camera data.
"""

import numpy as np
from collections import deque
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_face_size_score(bbox: np.ndarray, frame_width: int, frame_height: int) -> float:
    """
    Calculate quality score based on face size relative to frame.

    Args:
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        Score from 0.0 (too small/large) to 1.0 (ideal size)

    Logic:
        - Faces < 5% of frame = poor (too small, low resolution)
        - Faces 15-30% of frame = ideal (good balance)
        - Faces > 50% of frame = poor (too close, may be cropped)
    """
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    face_area = face_width * face_height
    frame_area = frame_width * frame_height

    if frame_area <= 0:
        return 0.0

    relative_area = face_area / frame_area

    # Score based on relative area
    if relative_area < 0.05:
        # Too small - linearly scale from 0 to 1
        return min(1.0, relative_area / 0.05)
    elif relative_area <= 0.30:
        # Ideal range - full score
        return 1.0
    else:
        # Too large - linearly decay
        # At 50%, score = 0.5; at 80%, score = 0
        return max(0.0, 1.0 - (relative_area - 0.30) / 0.50)


def calculate_frontal_score(landmarks: np.ndarray) -> float:
    """
    Calculate quality score based on face frontality (yaw angle).

    Args:
        landmarks: Face landmarks array (478, 3) - MediaPipe format

    Returns:
        Score from 0.0 (profile) to 1.0 (frontal)

    Logic:
        Uses eye-nose-mouth triangle symmetry to estimate yaw.
        Frontal faces have symmetric landmark positions.
    """
    if landmarks.shape[0] < 478:
        return 0.5  # Fallback for incomplete landmarks

    try:
        # Key landmark indices (MediaPipe 478-point model)
        LEFT_EYE_OUTER = 33
        RIGHT_EYE_OUTER = 263
        NOSE_TIP = 1
        LEFT_MOUTH = 61
        RIGHT_MOUTH = 291

        # Extract key points
        left_eye = landmarks[LEFT_EYE_OUTER][:2]  # x, y only
        right_eye = landmarks[RIGHT_EYE_OUTER][:2]
        nose_tip = landmarks[NOSE_TIP][:2]
        left_mouth = landmarks[LEFT_MOUTH][:2]
        right_mouth = landmarks[RIGHT_MOUTH][:2]

        # Calculate eye center
        eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
        eye_distance = abs(right_eye[0] - left_eye[0])

        # Calculate mouth center
        mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2.0

        if eye_distance < 1.0:
            return 0.5  # Degenerate case

        # Nose should be centered between eyes for frontal face
        nose_deviation = abs(nose_tip[0] - eye_center_x)
        nose_symmetry = 1.0 - min(1.0, nose_deviation / (eye_distance * 0.5))

        # Mouth should also be centered
        mouth_deviation = abs(mouth_center_x - eye_center_x)
        mouth_symmetry = 1.0 - min(1.0, mouth_deviation / (eye_distance * 0.5))

        # Combined symmetry score (quadratic to penalize asymmetry)
        symmetry_score = (nose_symmetry + mouth_symmetry) / 2.0
        frontal_score = symmetry_score ** 1.5  # Stronger penalty for non-frontal

        return float(np.clip(frontal_score, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"[Quality] Frontal score calculation failed: {e}")
        return 0.5


class StabilityTracker:
    """
    Tracks temporal stability of landmark detections.

    Stable detections (low jitter) indicate higher quality capture.
    Unstable detections may indicate motion blur, occlusion, or poor lighting.
    """

    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: Number of frames to track for variance calculation
        """
        self.window_size = window_size
        self.history = {}  # {(participant_id, camera_idx): deque of landmark arrays}

    def calculate_stability_score(self, participant_id: int, camera_idx: int,
                                  landmarks: np.ndarray) -> float:
        """
        Calculate stability score based on temporal variance of landmarks.

        Args:
            participant_id: Participant identifier
            camera_idx: Camera index
            landmarks: Current frame landmarks (478, 3)

        Returns:
            Score from 0.0 (unstable) to 1.0 (stable)
        """
        key = (participant_id, camera_idx)

        # Initialize history for this participant-camera pair
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window_size)

        # Add current landmarks to history
        self.history[key].append(landmarks.copy())

        # Need at least 2 frames to calculate variance
        if len(self.history[key]) < 2:
            return 1.0  # Perfect stability for first frame

        try:
            # Calculate variance across temporal window
            # Use only x,y coordinates (ignore z for stability)
            positions = np.array([lm[:, :2] for lm in self.history[key]])  # (frames, 478, 2)

            # Variance across time dimension
            variance = np.var(positions, axis=0)  # (478, 2)
            mean_variance = np.mean(variance)

            # Convert variance to stability score
            # Lower variance = more stable = higher score
            # Typical variance range: 0.001 (very stable) to 0.1 (very unstable)
            stability = 1.0 / (1.0 + mean_variance * 100.0)

            return float(np.clip(stability, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"[Quality] Stability calculation failed: {e}")
            return 1.0

    def clear_participant(self, participant_id: int, camera_idx: int):
        """Clear history for a specific participant-camera pair."""
        key = (participant_id, camera_idx)
        if key in self.history:
            del self.history[key]

    def clear_all(self):
        """Clear all history."""
        self.history.clear()


def estimate_capture_quality(
    bbox: np.ndarray,
    landmarks: np.ndarray,
    frame_shape: Tuple[int, int],
    stability_tracker: StabilityTracker,
    participant_id: int,
    camera_idx: int,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Estimate overall capture quality using multiple metrics.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        landmarks: Face landmarks (478, 3)
        frame_shape: (height, width) of frame
        stability_tracker: Tracker for temporal stability
        participant_id: Participant ID
        camera_idx: Camera index
        weights: Optional custom weights for metrics

    Returns:
        (quality_score, breakdown_dict)
        - quality_score: Overall quality 0.0-1.0
        - breakdown_dict: Individual metric scores
    """
    if weights is None:
        weights = {
            'size': 0.4,
            'frontal': 0.3,
            'stability': 0.3
        }

    frame_height, frame_width = frame_shape[:2]

    # Calculate individual metrics
    size_score = calculate_face_size_score(bbox, frame_width, frame_height)
    frontal_score = calculate_frontal_score(landmarks)
    stability_score = stability_tracker.calculate_stability_score(
        participant_id, camera_idx, landmarks
    )

    # Weighted combination
    quality = (
        weights['size'] * size_score +
        weights['frontal'] * frontal_score +
        weights['stability'] * stability_score
    )

    breakdown = {
        'size': float(size_score),
        'frontal': float(frontal_score),
        'stability': float(stability_score),
        'combined': float(quality)
    }

    return quality, breakdown
