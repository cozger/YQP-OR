"""
3D Skeleton Renderer for RTMW3D Pose Estimation

This module provides 3D visualization for whole-body pose estimation using matplotlib.
Optimized for GUI integration with Tkinter.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from typing import Tuple, Optional

# RTMW3D body skeleton connections (17 body keypoints COCO-style)
BODY_CONNECTIONS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Arms
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16)
]


def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
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


def calculate_head_tilt_angle(
    keypoints: np.ndarray,
    scores: np.ndarray,
    min_confidence: float = 0.3,
    ear_to_nose_drop_ratio: float = 0.20
) -> Optional[tuple]:
    """
    Calculate head tilt relative to neck using vector angle method.

    Measures the angle between:
    - Neck vector (shoulder midpoint → ear midpoint)
    - Face forward vector (ear midpoint → nose)

    Args:
        keypoints: (N, 3) array with [x, y, z] in meters (camera space)
        scores: (N,) confidence scores
        min_confidence: Minimum confidence for calculation
        ear_to_nose_drop_ratio: Anatomical offset ratio to correct negative bias
                                (ear midpoint → nose level, as fraction of inter-ear distance)

    Returns:
        tuple: (angle_deg, head_tilt) where:
               angle_deg: Raw angle between neck and face vectors in degrees
               head_tilt: Head tilt in degrees
                         0° = neutral (face perpendicular to neck)
                         Negative = looking down
                         Positive = looking up
               Returns None if keypoints not confident enough
    """
    # Check we have required keypoints
    if len(keypoints) < 7:
        return None

    # Must have shoulders, ears, and nose
    shoulders_valid = (scores[5] >= min_confidence and scores[6] >= min_confidence)
    ears_valid = (scores[3] >= min_confidence and scores[4] >= min_confidence)
    nose_valid = scores[0] >= min_confidence

    if not (shoulders_valid and ears_valid and nose_valid):
        return None

    # Calculate neck base (shoulder midpoint)
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    neck_base = (left_shoulder + right_shoulder) / 2

    # Calculate ear midpoint (top of neck)
    left_ear = keypoints[3]
    right_ear = keypoints[4]
    ear_midpoint = (left_ear + right_ear) / 2

    # Apply anatomical offset correction to ear midpoint
    # The ear midpoint is anatomically higher than the nose, creating a negative bias
    # We calculate a vertical offset based on inter-ear distance to correct this
    inter_ear_distance = np.linalg.norm(right_ear - left_ear)
    vertical_offset = inter_ear_distance * ear_to_nose_drop_ratio

    # Create corrected ear midpoint (move DOWN in Z-axis toward nose level)
    # Note: Visualization uses index [2] as vertical axis (displayed as Y on plot)
    # Index [0]=X (horizontal), [1]=Y (depth in viz), [2]=Z (VERTICAL in viz)
    ear_midpoint_corrected = ear_midpoint.copy()
    ear_midpoint_corrected[2] -= vertical_offset

    # Neck vector (from neck base upward to ears)
    neck_vector = ear_midpoint - neck_base

    # Face forward vector (from ears forward to nose, using corrected ear position)
    nose = keypoints[0]
    face_vector = nose - ear_midpoint_corrected

    # Calculate angle between vectors using dot product
    dot_product = np.dot(neck_vector, face_vector)
    neck_magnitude = np.linalg.norm(neck_vector)
    face_magnitude = np.linalg.norm(face_vector)

    if neck_magnitude < 1e-8 or face_magnitude < 1e-8:
        return None

    cos_angle = dot_product / (neck_magnitude * face_magnitude)
    angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

    # Convert to head tilt: 0° when perpendicular (90° angle)
    # Inverted: Negative when looking down, Positive when looking up
    head_tilt = 90.0 - angle_deg

    return (angle_deg, head_tilt)


def _calculate_skeleton_center(
    keypoints: np.ndarray,
    scores: np.ndarray,
    min_confidence: float = 0.3
) -> np.ndarray:
    """
    Calculate stable center point for skeleton using hierarchical fallback strategy.

    This ensures the skeleton is always centered in the visualization regardless of
    the person's absolute position in camera space. Uses anatomically stable reference
    points with graceful degradation for missing/low-confidence keypoints.

    Args:
        keypoints: (N, 3) array with [x, y, z] coordinates in meters
        scores: (N,) confidence scores for each keypoint
        min_confidence: Minimum confidence threshold for valid keypoints

    Returns:
        np.ndarray: [x_center, y_center, z_center] in meters

    Centering Strategy (priority order):
        1. Torso center: (shoulder_midpoint + hip_midpoint) / 2
           - Most anatomically stable reference
           - Represents body's center of mass
           - Requires: shoulders (5,6) and hips (11,12) valid

        2. Shoulder midpoint: (left_shoulder + right_shoulder) / 2
           - Good upper-body reference
           - Fallback when hips unavailable
           - Requires: shoulders (5,6) valid

        3. Mean of valid body keypoints: avg(keypoints[0:17])
           - Uses only body keypoints (excludes face/hands)
           - Prevents bias from numerous face/hand points
           - Requires: at least one body keypoint valid

        4. Default: [0, 0, median_z]
           - Last resort when no keypoints valid
           - Uses median Z depth if available, else 5.0m
    """
    # Strategy 1: Torso center (most stable - biomechanical root)
    if len(keypoints) > 12 and len(scores) > 12:
        # Check if shoulders are valid (indices 5=left, 6=right)
        shoulders_valid = scores[5] >= min_confidence and scores[6] >= min_confidence
        # Check if hips are valid (indices 11=left, 12=right)
        hips_valid = scores[11] >= min_confidence and scores[12] >= min_confidence

        if shoulders_valid and hips_valid:
            shoulder_mid = (keypoints[5] + keypoints[6]) / 2.0
            hip_mid = (keypoints[11] + keypoints[12]) / 2.0
            torso_center = (shoulder_mid + hip_mid) / 2.0
            return torso_center

    # Strategy 2: Shoulder midpoint (upper body reference)
    if len(keypoints) > 6 and len(scores) > 6:
        shoulders_valid = scores[5] >= min_confidence and scores[6] >= min_confidence
        if shoulders_valid:
            shoulder_mid = (keypoints[5] + keypoints[6]) / 2.0
            return shoulder_mid

    # Strategy 3: Mean of valid body keypoints only (exclude face/hands)
    # Body keypoints are indices 0-16 (COCO-17 body skeleton)
    body_keypoints = keypoints[:17] if len(keypoints) >= 17 else keypoints
    body_scores = scores[:17] if len(scores) >= 17 else scores
    valid_mask = body_scores >= min_confidence

    if valid_mask.any():
        valid_body_kpts = body_keypoints[valid_mask]
        return np.mean(valid_body_kpts, axis=0)

    # Strategy 4: Default center (fallback when no valid keypoints)
    # Use median depth if available, otherwise assume typical camera distance
    default_z = np.median(keypoints[:, 2]) if len(keypoints) > 0 else 5.0
    return np.array([0.0, 0.0, default_z])


def plot_3d_skeleton(
    keypoints: np.ndarray,
    scores: np.ndarray,
    connections: list = BODY_CONNECTIONS,
    min_confidence: float = 0.3,
    view_angle: Tuple[int, int] = (20, 45),
    figsize: Tuple[int, int] = (6, 6),
    dpi: int = 100,
    ear_to_nose_drop_ratio: float = 0.20
) -> Image.Image:
    """
    Create 3D skeleton visualization with neck angle display.

    Args:
        keypoints: (N, 3) array with [x, y, z] in meters (camera space)
        scores: (N,) confidence scores
        connections: List of (start_idx, end_idx) tuples
        min_confidence: Minimum confidence threshold
        view_angle: (elevation, azimuth) for camera view
        figsize: Figure size in inches
        dpi: Figure DPI

    Returns:
        PIL.Image: RGB image of the 3D plot
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Fix orientation: Swap axes so Z becomes vertical
    # Camera coords: [X, Y, Z] where Z varies with height
    # Plot coords: [X, Z, -Y] where Z is vertical axis

    # Only flip Y for front-back orientation
    keypoints_flipped = keypoints.copy()
    keypoints_flipped[:, 1] = -keypoints_flipped[:, 1]  # Flip Y only

    # Calculate center point for stable visualization (BEFORE flipping Y in center)
    # This ensures skeleton is always centered at origin regardless of camera position
    center_point = _calculate_skeleton_center(keypoints, scores, min_confidence)

    # Apply Y-flip to center point to match flipped keypoints coordinate system
    center_flipped = center_point.copy()
    center_flipped[1] = -center_flipped[1]

    # Apply centering transformation: translate skeleton so center is at origin
    keypoints_centered = keypoints_flipped - center_flipped

    # Draw connections (skeleton lines)
    for start_idx, end_idx in connections:
        if start_idx >= len(keypoints_centered) or end_idx >= len(keypoints_centered):
            continue
        if scores[start_idx] < min_confidence or scores[end_idx] < min_confidence:
            continue

        start_pt = keypoints_centered[start_idx]
        end_pt = keypoints_centered[end_idx]

        avg_conf = (scores[start_idx] + scores[end_idx]) / 2.0
        color = get_confidence_color(avg_conf)
        # Convert BGR to RGB and normalize
        color_rgb = (color[2]/255, color[1]/255, color[0]/255)

        # Plot with swapped axes: [X, Z, -Y]
        ax.plot([start_pt[0], end_pt[0]],    # X: left-right
                [start_pt[2], end_pt[2]],     # Z: vertical
                [start_pt[1], end_pt[1]],     # -Y: depth
                color=color_rgb, linewidth=2)

    # Draw custom neck line (shoulder midpoint to ear midpoint)
    if len(keypoints_centered) > 6:
        shoulders_valid = (scores[5] >= min_confidence and scores[6] >= min_confidence)
        ears_valid = (scores[3] >= min_confidence and scores[4] >= min_confidence)

        if shoulders_valid and ears_valid:
            # Neck base (shoulder midpoint)
            left_shoulder = keypoints_centered[5]
            right_shoulder = keypoints_centered[6]
            neck_base = (left_shoulder + right_shoulder) / 2

            # Neck top (ear midpoint)
            left_ear = keypoints_centered[3]
            right_ear = keypoints_centered[4]
            ear_midpoint = (left_ear + right_ear) / 2

            # Apply anatomical offset correction to ear midpoint (same as head tilt calculation)
            inter_ear_distance = np.linalg.norm(right_ear - left_ear)
            vertical_offset = inter_ear_distance * ear_to_nose_drop_ratio  # Use parameter from config
            ear_midpoint_corrected = ear_midpoint.copy()
            ear_midpoint_corrected[2] -= vertical_offset  # Move down in Z-axis (vertical in visualization)

            # Bright orange color for high visibility
            neck_color_rgb = (1.0, 0.5, 0.0)  # RGB: bright orange

            # Draw neck line (swapped axes) - use original ear_midpoint for neck visualization
            ax.plot([neck_base[0], ear_midpoint[0]],    # X: left-right
                    [neck_base[2], ear_midpoint[2]],    # Z: vertical
                    [neck_base[1], ear_midpoint[1]],    # -Y: depth
                    color=neck_color_rgb, linewidth=4)  # Thicker neck line in bright orange

            # Draw face direction vector (ear midpoint to nose) for visual feedback
            nose_valid = scores[0] >= min_confidence
            if nose_valid:
                nose = keypoints_centered[0]

                # Calculate face direction vector using CORRECTED ear midpoint
                face_direction = nose - ear_midpoint_corrected
                face_magnitude = np.linalg.norm(face_direction)

                if face_magnitude > 1e-8:
                    # Extend beyond nose by 80cm (0.8 meters) for high visibility
                    extension_length = 0.8
                    extended_point = nose + (face_direction / face_magnitude) * extension_length

                    # Draw extended face vector as THICK SOLID RED line
                    # Use CORRECTED ear midpoint as starting point to match metrics calculation
                    gaze_line_color = (1.0, 0.0, 0.0)  # Pure red in RGB
                    ax.plot([ear_midpoint_corrected[0], extended_point[0]],
                            [ear_midpoint_corrected[2], extended_point[2]],
                            [ear_midpoint_corrected[1], extended_point[1]],
                            color=gaze_line_color, linewidth=6, linestyle='-', alpha=1.0)

                    # Calculate and display angle values near the red line endpoint
                    angle_result = calculate_head_tilt_angle(keypoints, scores, min_confidence, ear_to_nose_drop_ratio)
                    if angle_result is not None:
                        vector_angle, head_tilt = angle_result
                        # Position text slightly offset from extended point for visibility
                        text_offset = 0.15  # 15cm offset
                        text_pos = extended_point + np.array([text_offset, 0, 0])
                        # Display with swapped axes matching the 3D plot coordinate system
                        ax.text(text_pos[0], text_pos[2], text_pos[1],
                                f'Vec: {vector_angle:.1f}° | Tilt: {head_tilt:+.1f}°',
                                fontsize=10, color='red', fontweight='normal')

    # Draw keypoints (scatter) with differentiated styles for body/face/hands
    valid_mask = scores > min_confidence
    if valid_mask.any():
        # Define keypoint groups with different visualization styles
        keypoint_groups = [
            {
                'name': 'body_feet',
                'range': (0, 23),  # Body (0-17) + feet (17-23)
                'size': 40,
                'alpha': 0.8
            },
            {
                'name': 'face',
                'range': (23, 91),  # Face keypoints (68 points)
                'size': 10,
                'alpha': 0.4
            },
            {
                'name': 'hands',
                'range': (91, 133),  # Left hand (91-112) + right hand (112-133)
                'size': 15,
                'alpha': 0.5
            }
        ]

        # Draw each group separately with its own style
        for group in keypoint_groups:
            start_idx, end_idx = group['range']
            # Clamp to actual keypoint range
            start_idx = min(start_idx, len(keypoints_centered))
            end_idx = min(end_idx, len(keypoints_centered))

            # Create mask for this group's keypoints
            group_mask = valid_mask.copy()
            group_mask[:start_idx] = False
            group_mask[end_idx:] = False

            if group_mask.any():
                group_kp = keypoints_centered[group_mask]
                group_scores = scores[group_mask]

                # Color by confidence
                colors = [get_confidence_color(s) for s in group_scores]
                colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]

                # Scatter with swapped axes and group-specific styling
                ax.scatter(group_kp[:, 0], group_kp[:, 2], group_kp[:, 1],
                           c=colors_rgb, s=group['size'], alpha=group['alpha'],
                           depthshade=True)

    # Set axis labels and limits (symmetric around origin for centered skeleton)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')      # Z is vertical axis
    ax.set_zlabel('Y (m)')      # -Y is depth
    ax.set_xlim([-1.5, 1.5])    # X: left-right (centered at 0)
    ax.set_ylim([-2, 2])         # Z: vertical (centered at 0)
    ax.set_zlim([-2, 2])         # -Y: depth (centered at 0)

    # Calculate and display head tilt angle
    angle_result = calculate_head_tilt_angle(keypoints, scores, min_confidence)
    if angle_result is not None:
        vector_angle, head_tilt = angle_result
        title_text = f'3D Pose Estimation | Head Tilt: {head_tilt:+.1f}°'
    else:
        title_text = '3D Pose Estimation | Head Tilt: N/A'

    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    ax.set_title(title_text, fontsize=12, fontweight='bold')

    # Minimize matplotlib margins to prevent overflow on canvas
    fig.tight_layout(pad=0.1)

    # Convert to PIL Image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_array = img_array[:, :, :3]  # Drop alpha channel

    # Convert to PIL Image
    img = Image.fromarray(img_array)

    plt.close(fig)
    return img
