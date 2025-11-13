"""
Pose Landmark Temporal Smoothing

Reduces high-frequency jitter in MediaPipe pose landmarks using
exponential moving average (EMA) with occlusion handling.
"""

import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PoseLandmarkSmoother:
    """
    Temporal smoothing filter for MediaPipe pose landmarks.

    Features:
    - Exponential moving average per landmark
    - Adaptive alpha based on landmark type (stable vs distal)
    - Freeze smoothed value on occlusions
    - Per-camera state isolation

    Architecture:
    - Inserted between pose data reading and output distribution
    - Affects both GUI rendering and LSL streaming
    - Minimal latency: 1-3 frames depending on alpha values
    """

    # Landmark groups for differential smoothing
    # MediaPipe Pose has 33 landmarks (0-32)
    STABLE_LANDMARKS = [0, 11, 12, 23, 24]  # Nose, shoulders, hips
    DISTAL_LANDMARKS = [15, 16, 19, 20, 27, 28, 31, 32]  # Wrists, ankles, feet

    def __init__(self, config: Dict):
        """
        Initialize smoother with configuration.

        Args:
            config: Full application config dict. Reads from config['pose_smoothing']:
                - alpha_stable: EMA alpha for stable landmarks (default 0.7)
                - alpha_distal: EMA alpha for distal landmarks (default 0.3)
                - alpha_default: EMA alpha for other landmarks (default 0.5)
                - visibility_threshold: Min visibility to update (default 0.5)
                - freeze_on_occlusion: Freeze smoothed value when occluded (default True)
        """
        smooth_config = config.get('pose_smoothing', {})

        # EMA alpha values (higher = less smoothing, more responsive)
        # Alpha range: 0.1 (heavy smoothing) to 0.9 (minimal smoothing)
        self.alpha_stable = smooth_config.get('alpha_stable', 0.7)
        self.alpha_distal = smooth_config.get('alpha_distal', 0.3)
        self.alpha_default = smooth_config.get('alpha_default', 0.5)

        # Occlusion handling
        self.visibility_threshold = smooth_config.get('visibility_threshold', 0.5)
        self.freeze_on_occlusion = smooth_config.get('freeze_on_occlusion', True)

        # Enabled flag
        self.enabled = smooth_config.get('enabled', True)

        # State: {camera_idx: {participant_id: smoothed_landmarks}}
        # CRITICAL FIX: Use participant_id (not pose_idx) to prevent cross-participant smoothing
        # smoothed_landmarks: np.array(33, 3) - x, y, z coordinates
        self.state = {}

        # Frame tracking for reset on stale data
        self.last_frame_id = {}

        # Track last seen frame for each participant (for cleanup)
        self.participant_last_seen = {}

        logger.info(f"[PoseSmoother] Initialized (enabled={self.enabled}, "
                   f"alpha_stable={self.alpha_stable}, alpha_distal={self.alpha_distal}, "
                   f"alpha_default={self.alpha_default})")

    def smooth(self, poses: List[Dict], camera_idx: int, frame_id: int) -> List[Dict]:
        """
        Apply temporal smoothing to pose landmarks.

        CRITICAL FIX: Uses participant_id (not array index) to prevent cross-participant smoothing.

        Args:
            poses: List of pose dicts with keys:
                - 'keypoints': List of (x, y, z) tuples (33 landmarks)
                - 'visibility': List of visibility scores (33 values)
                - 'participant_id': Participant ID (1, 2, ...) or None if unmatched
                - 'centroid': Tuple (x, y) - NOT smoothed (recalculated if needed)
                - 'pose_resolution': Tuple (w, h)
            camera_idx: Camera identifier (0-3)
            frame_id: Current frame number (for stale detection)

        Returns:
            List of pose dicts with smoothed landmarks (same structure as input)
        """
        if not self.enabled or not poses:
            return poses

        # Initialize state for this camera
        if camera_idx not in self.state:
            self.state[camera_idx] = {}
            self.last_frame_id[camera_idx] = -1
            self.participant_last_seen[camera_idx] = {}

        # Reset if frame jumped backwards or too far ahead (camera restart)
        frame_delta = frame_id - self.last_frame_id[camera_idx]
        if frame_delta < 0 or frame_delta > 100:
            logger.info(f"[PoseSmoother] Camera {camera_idx} reset (frame_delta={frame_delta})")
            self.state[camera_idx] = {}
            self.participant_last_seen[camera_idx] = {}

        self.last_frame_id[camera_idx] = frame_id

        # Cleanup stale participants (not seen for >300 frames ~10 seconds at 30fps)
        stale_participants = []
        for participant_id, last_frame in self.participant_last_seen.get(camera_idx, {}).items():
            if frame_id - last_frame > 300:
                stale_participants.append(participant_id)

        for participant_id in stale_participants:
            if participant_id in self.state[camera_idx]:
                del self.state[camera_idx][participant_id]
                logger.debug(f"[PoseSmoother] Cleaned up stale participant {participant_id} on camera {camera_idx}")
            if participant_id in self.participant_last_seen[camera_idx]:
                del self.participant_last_seen[camera_idx][participant_id]

        # Process each pose
        smoothed_poses = []
        for pose in poses:
            participant_id = pose.get('participant_id')

            # Skip smoothing for unmatched poses (no participant_id)
            if participant_id is None:
                logger.debug(f"[PoseSmoother] Skipping smoothing for unmatched pose (no participant_id)")
                smoothed_poses.append(pose)  # Return raw pose
                continue

            # Numpy slicing - keypoints is already (133, 4) array
            keypoints = pose['keypoints']  # (133, 4) [x, y, z, confidence]
            raw_landmarks = keypoints[:, :3]  # (133, 3) coordinates only
            visibility = keypoints[:, 3]      # (133,) confidence scores

            # Initialize smoothed state for this participant if first time
            if participant_id not in self.state[camera_idx]:
                # Start with raw landmarks (no smoothing on first frame)
                self.state[camera_idx][participant_id] = raw_landmarks.copy()
                smoothed_landmarks = raw_landmarks
                logger.info(f"[PoseSmoother] Initialized smoothing for participant {participant_id} on camera {camera_idx}")
            else:
                # Apply EMA smoothing using participant's own history
                prev_smoothed = self.state[camera_idx][participant_id]
                smoothed_landmarks = self._apply_ema(
                    raw_landmarks, prev_smoothed, visibility
                )
                self.state[camera_idx][participant_id] = smoothed_landmarks

            # Track when this participant was last seen
            self.participant_last_seen[camera_idx][participant_id] = frame_id

            # Reconstruct (133, 4) array with smoothed coords + original confidence
            smoothed_pose = pose.copy()
            smoothed_keypoints = keypoints.copy()
            smoothed_keypoints[:, :3] = smoothed_landmarks  # Update x,y,z with smoothed values
            smoothed_pose['keypoints'] = smoothed_keypoints

            # Note: Centroid is NOT smoothed here - it's recalculated from smoothed landmarks
            # in downstream code if needed. We preserve the original centroid for now.

            smoothed_poses.append(smoothed_pose)

        return smoothed_poses

    def _apply_ema(self, raw: np.ndarray, prev: np.ndarray,
                   visibility: np.ndarray) -> np.ndarray:
        """
        Apply exponential moving average with visibility-aware updates.

        EMA formula: smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]

        Args:
            raw: (33, 3) raw landmark coordinates
            prev: (33, 3) previous smoothed coordinates
            visibility: (33,) visibility scores (0-1)

        Returns:
            (33, 3) smoothed coordinates
        """
        smoothed = prev.copy()

        for i in range(33):
            # Select alpha based on landmark type
            if i in self.STABLE_LANDMARKS:
                alpha = self.alpha_stable
            elif i in self.DISTAL_LANDMARKS:
                alpha = self.alpha_distal
            else:
                alpha = self.alpha_default

            # Check visibility
            if visibility[i] >= self.visibility_threshold:
                # Visible: Apply EMA update
                smoothed[i] = alpha * raw[i] + (1 - alpha) * prev[i]
            else:
                # Occluded: Freeze or use raw
                if self.freeze_on_occlusion:
                    smoothed[i] = prev[i]  # Keep previous smoothed value
                else:
                    smoothed[i] = raw[i]   # Use raw (jittery but responsive)

        return smoothed

    def reset(self, camera_idx: Optional[int] = None):
        """
        Reset smoothing state (e.g., when camera restarted).

        Args:
            camera_idx: If provided, reset only this camera. Otherwise reset all.
        """
        if camera_idx is None:
            self.state = {}
            self.last_frame_id = {}
            logger.info("[PoseSmoother] Full state reset")
        elif camera_idx in self.state:
            del self.state[camera_idx]
            if camera_idx in self.last_frame_id:
                del self.last_frame_id[camera_idx]
            logger.info(f"[PoseSmoother] Reset camera {camera_idx}")
