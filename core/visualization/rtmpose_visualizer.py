"""
RTMPose3D Visualization Utilities

Provides drawing functions for RTMW3D whole-body pose estimation (133 keypoints).
Shared between test script and GUI for consistent rendering.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .skeleton_definitions import (
    BODY_CONNECTIONS,
    HAND_CONNECTIONS,
    FACE_CONTOUR_CONNECTIONS,
    KEYPOINT_GROUPS,
    get_keypoint_style,
    get_connections_for_keypoints
)


class RTMPoseVisualizer:
    """
    Visualization utilities for RTMW3D (133-keypoint whole-body pose).

    Features:
    - 2D skeleton drawing with confidence-based coloring
    - 3D pose visualization with matplotlib
    - Support for body, face, and hand keypoints
    - Single-person focus (no multi-person color coding)
    """

    @staticmethod
    def get_confidence_color(confidence):
        """
        Get BGR color based on confidence value.

        Args:
            confidence: Float between 0 and 1

        Returns:
            tuple: BGR color tuple
        """
        if confidence > 0.7:
            return (0, 255, 0)  # Green - high confidence
        elif confidence > 0.4:
            return (0, 255, 255)  # Yellow - medium confidence
        else:
            return (0, 0, 255)  # Red - low confidence

    @staticmethod
    def draw_skeleton_2d(frame, keypoints, connections=None, min_confidence=0.3,
                         draw_keypoints=True, draw_connections=True):
        """
        Draw 2D skeleton overlay on frame (single person).

        Args:
            frame: BGR image (will be modified in-place)
            keypoints: (N, 4) array with [x, y, z, confidence] OR (N, 3) with [x, y, confidence]
            connections: List of (start_idx, end_idx) tuples (auto-detected if None)
            min_confidence: Minimum confidence threshold
            draw_keypoints: Draw keypoint circles
            draw_connections: Draw skeleton lines

        Returns:
            Modified frame
        """
        h, w = frame.shape[:2]
        num_kps = len(keypoints)

        # Auto-detect connections if not provided
        if connections is None:
            connections = get_connections_for_keypoints(keypoints)

        # Draw connections (skeleton lines)
        if draw_connections:
            for start_idx, end_idx in connections:
                if start_idx >= num_kps or end_idx >= num_kps:
                    continue

                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]

                # Extract confidence (last column)
                start_conf = start_kp[-1]
                end_conf = end_kp[-1]

                # Check confidence
                if start_conf < min_confidence or end_conf < min_confidence:
                    continue

                # Validate coordinates before conversion (prevent corrupted data from causing blow-ups)
                # Check for NaN, Inf, or unreasonable values
                MAX_COORD = 100000  # Reasonable upper bound for coordinate values
                if (not np.isfinite(start_kp[0]) or not np.isfinite(start_kp[1]) or
                    not np.isfinite(end_kp[0]) or not np.isfinite(end_kp[1]) or
                    abs(start_kp[0]) > MAX_COORD or abs(start_kp[1]) > MAX_COORD or
                    abs(end_kp[0]) > MAX_COORD or abs(end_kp[1]) > MAX_COORD):
                    continue  # Skip corrupted keypoints

                # Get 2D coordinates (safe after validation)
                start_pt = (int(start_kp[0]), int(start_kp[1]))
                end_pt = (int(end_kp[0]), int(end_kp[1]))

                # Check if points are within frame
                if (0 <= start_pt[0] < w and 0 <= start_pt[1] < h and
                    0 <= end_pt[0] < w and 0 <= end_pt[1] < h):

                    # Average confidence for line color
                    avg_conf = (start_conf + end_conf) / 2.0
                    color = RTMPoseVisualizer.get_confidence_color(avg_conf)

                    # Draw line
                    cv2.line(frame, start_pt, end_pt, color, 2, cv2.LINE_AA)

        # Draw keypoints with differentiated styles for body/face/hands
        if draw_keypoints:
            # Separate into opaque and transparent groups for efficient rendering
            opaque_keypoints = []
            transparent_keypoints = []

            for idx, kp in enumerate(keypoints):
                confidence = kp[-1]
                if confidence < min_confidence:
                    continue

                # Validate coordinates before conversion (prevent corrupted data)
                MAX_COORD = 100000  # Reasonable upper bound
                if (not np.isfinite(kp[0]) or not np.isfinite(kp[1]) or
                    abs(kp[0]) > MAX_COORD or abs(kp[1]) > MAX_COORD):
                    continue  # Skip corrupted keypoint

                pt = (int(kp[0]), int(kp[1]))

                # Check if point is within frame
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    outer_radius, inner_radius, alpha = get_keypoint_style(idx)
                    color = RTMPoseVisualizer.get_confidence_color(confidence)

                    if alpha >= 1.0:
                        opaque_keypoints.append((pt, outer_radius, inner_radius, color))
                    else:
                        transparent_keypoints.append((pt, outer_radius, inner_radius, color, alpha))

            # Draw opaque keypoints directly (body + feet)
            for pt, outer_radius, inner_radius, color in opaque_keypoints:
                cv2.circle(frame, pt, outer_radius, color, -1, cv2.LINE_AA)
                if inner_radius > 0:
                    cv2.circle(frame, pt, inner_radius, (255, 255, 255), -1, cv2.LINE_AA)

            # Draw transparent keypoints with overlay (face + hands)
            if transparent_keypoints:
                overlay = frame.copy()
                for pt, outer_radius, inner_radius, color, alpha in transparent_keypoints:
                    cv2.circle(overlay, pt, outer_radius, color, -1, cv2.LINE_AA)
                    if inner_radius > 0:
                        cv2.circle(overlay, pt, inner_radius, (255, 255, 255), -1, cv2.LINE_AA)

                # Blend overlay with original frame
                cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        return frame

    @staticmethod
    def draw_skeleton_3d(keypoints, scores, connections=None, min_confidence=0.3,
                         view_angle=(20, 45), title='3D Pose Estimation'):
        """
        Create 3D skeleton visualization using matplotlib.

        Args:
            keypoints: (N, 3) array with [x, y, z] in meters (camera space)
            scores: (N,) confidence scores
            connections: List of (start_idx, end_idx) tuples (auto-detected if None)
            min_confidence: Minimum confidence threshold
            view_angle: (elevation, azimuth) for camera view
            title: Plot title

        Returns:
            RGB image of 3D plot
        """
        num_kps = len(keypoints)

        # Auto-detect connections if not provided
        if connections is None:
            connections = get_connections_for_keypoints(keypoints)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # Flip Y for correct orientation (camera Y-up → plot Z-up)
        kp_plot = keypoints.copy()
        kp_plot[:, 1] = -kp_plot[:, 1]

        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx >= num_kps or end_idx >= num_kps:
                continue
            if scores[start_idx] < min_confidence or scores[end_idx] < min_confidence:
                continue

            avg_conf = (scores[start_idx] + scores[end_idx]) / 2.0
            color_bgr = RTMPoseVisualizer.get_confidence_color(avg_conf)
            color_rgb = (color_bgr[2]/255, color_bgr[1]/255, color_bgr[0]/255)

            ax.plot([kp_plot[start_idx, 0], kp_plot[end_idx, 0]],
                   [kp_plot[start_idx, 2], kp_plot[end_idx, 2]],
                   [kp_plot[start_idx, 1], kp_plot[end_idx, 1]],
                   color=color_rgb, linewidth=2)

        # Draw keypoints
        valid_mask = scores > min_confidence
        if valid_mask.any():
            valid_kp = kp_plot[valid_mask]
            valid_scores = scores[valid_mask]
            colors = [RTMPoseVisualizer.get_confidence_color(s) for s in valid_scores]
            colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]

            ax.scatter(valid_kp[:, 0], valid_kp[:, 2], valid_kp[:, 1],
                      c=colors_rgb, s=40, alpha=0.8, depthshade=True)

        # Set labels and limits
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_zlabel('Y (m)')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([3, 7])
        ax.set_zlim([-2, 2])
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]  # Drop alpha channel

        plt.close(fig)
        return img

    @staticmethod
    def draw_bbox_with_label(frame, bbox, label, color=(0, 255, 0), thickness=2):
        """
        Draw bounding box with label.

        Args:
            frame: BGR image (modified in-place)
            bbox: [x1, y1, x2, y2] or [x1, y1, w, h]
            label: Text label
            color: Box color (BGR)
            thickness: Line thickness

        Returns:
            Modified frame
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Handle [x, y, w, h] format
        if x2 < x1 or y2 < y1:
            x2 = x1 + x2
            y2 = y1 + y2

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10),
                     (x1 + label_w + 10, y1), color, -1)

        # Draw label text
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    @staticmethod
    def draw_all_keypoints(frame, keypoints, min_confidence=0.3):
        """
        Draw all 133 keypoints: body, face, and hands (single person).

        Args:
            frame: BGR image (modified in-place)
            keypoints: (133, 4) array with [x, y, z, confidence]
            min_confidence: Minimum confidence threshold

        Returns:
            Modified frame
        """
        # Get connections for full body (133 keypoints)
        connections = get_connections_for_keypoints(keypoints)

        # Draw skeleton with all connections
        return RTMPoseVisualizer.draw_skeleton_2d(
            frame, keypoints, connections, min_confidence,
            draw_keypoints=True, draw_connections=True
        )

    @staticmethod
    def draw_body_only(frame, keypoints, min_confidence=0.3):
        """
        Draw only body skeleton (17 keypoints).

        Args:
            frame: BGR image (modified in-place)
            keypoints: (133, 4) or (17, 4) array with [x, y, z, confidence]
            min_confidence: Minimum confidence threshold

        Returns:
            Modified frame
        """
        # Extract body keypoints if full array
        if len(keypoints) == 133:
            body_kps = keypoints[:17]
        else:
            body_kps = keypoints

        return RTMPoseVisualizer.draw_skeleton_2d(
            frame, body_kps, BODY_CONNECTIONS, min_confidence,
            draw_keypoints=True, draw_connections=True
        )

    @staticmethod
    def draw_hands_only(frame, keypoints, min_confidence=0.3):
        """
        Draw only hand skeletons (42 keypoints total).

        Args:
            frame: BGR image (modified in-place)
            keypoints: (133, 4) array with [x, y, z, confidence]
            min_confidence: Minimum confidence threshold

        Returns:
            Modified frame
        """
        if len(keypoints) < 133:
            return frame  # Not enough keypoints

        # Extract hand keypoints
        left_hand_kps = keypoints[91:112]
        right_hand_kps = keypoints[112:133]

        # Draw left hand
        RTMPoseVisualizer.draw_skeleton_2d(
            frame, left_hand_kps, HAND_CONNECTIONS, min_confidence,
            draw_keypoints=True, draw_connections=True
        )

        # Draw right hand
        RTMPoseVisualizer.draw_skeleton_2d(
            frame, right_hand_kps, HAND_CONNECTIONS, min_confidence,
            draw_keypoints=True, draw_connections=True
        )

        return frame

    @staticmethod
    def draw_face_only(frame, keypoints, min_confidence=0.3):
        """
        Draw only face landmarks (68 keypoints).

        Args:
            frame: BGR image (modified in-place)
            keypoints: (133, 4) or (68, 4) array with [x, y, z, confidence]
            min_confidence: Minimum confidence threshold

        Returns:
            Modified frame
        """
        # Extract face keypoints if full array
        if len(keypoints) == 133:
            face_kps = keypoints[23:91]
        else:
            face_kps = keypoints

        return RTMPoseVisualizer.draw_skeleton_2d(
            frame, face_kps, FACE_CONTOUR_CONNECTIONS, min_confidence,
            draw_keypoints=True, draw_connections=True
        )
