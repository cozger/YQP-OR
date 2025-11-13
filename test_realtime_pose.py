#!/usr/bin/env python3
"""
Real-time MMPose 3D Pipeline with ZMQ Camera Integration

Complete working demo:
- ZMQ camera capture from Windows bridge
- RTMDet person detection
- RTMW3D 3D whole-body pose estimation
- 2D skeleton visualization
- 3D keypoint statistics

Usage:
    python test_realtime_pose.py
    Press 'q' to quit
"""

# CRITICAL: Set multiprocessing to spawn mode for CUDA compatibility
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import sys
import os
from pathlib import Path

# Add rtmpose3d module to Python path (required for RTMW3D config)
rtmpose3d_path = "/home/canoz/Projects/surgery/mmpose/projects/rtmpose3d"
if rtmpose3d_path not in sys.path:
    sys.path.insert(0, rtmpose3d_path)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import json
import time
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg

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

# Import camera source
from core.camera_sources.factory import CameraSourceFactory

# RTMW3D body skeleton connections (17 body keypoints COCO-style)
BODY_CONNECTIONS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Neck - drawn separately as custom midpoint line
    # Arms
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Keypoint group indices (RTMW3D: 133 keypoints total)
KEYPOINT_GROUPS = {
    'body': (0, 17),      # COCO 17 body keypoints
    'feet': (17, 23),     # 6 foot keypoints
    'face': (23, 91),     # 68 face keypoints
    'left_hand': (91, 112),   # 21 left hand keypoints
    'right_hand': (112, 133)  # 21 right hand keypoints
}


def load_config():
    """Load configuration from JSON files."""
    # Load MMPose model configuration
    mmpose_config_path = Path(__file__).parent / "mmpose_config.json"
    with open(mmpose_config_path) as f:
        mmpose_config = json.load(f)

    # Load camera/system configuration (has ZMQ settings)
    system_config_path = Path(__file__).parent / "youquantipy_config.json"
    with open(system_config_path) as f:
        system_config = json.load(f)

    # Merge configs (mmpose_config takes precedence)
    merged_config = system_config.copy()
    merged_config.update(mmpose_config)

    return merged_config


def init_models(config):
    """
    Initialize RTMDet detector and RTMW3D pose estimator.

    Returns:
        tuple: (detector, pose_estimator)
    """
    print("\n" + "=" * 70)
    print("Initializing Models")
    print("=" * 70)

    mmpose_config = config['mmpose_3d_pipeline']
    device = mmpose_config['device']

    # Initialize person detector
    print("\n[1/2] Loading RTMDet-M person detector...")
    detector = init_detector(
        mmpose_config['person_detector']['config'],
        mmpose_config['person_detector']['checkpoint'],
        device=device
    )
    # CRITICAL: Adapt MMDet pipeline to avoid registry conflicts
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    print("✓ RTMDet-M loaded")

    # Initialize pose estimator
    print("\n[2/2] Loading RTMW3D-L pose estimator...")
    pose_estimator = init_model(
        mmpose_config['pose_estimator']['config'],
        mmpose_config['pose_estimator']['checkpoint'],
        device=device
    )
    print("✓ RTMW3D-L loaded")

    print("\n✓ All models initialized successfully")

    return detector, pose_estimator


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


def get_confidence_color(confidence):
    """
    Get color based on confidence value.

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


def draw_skeleton(frame, keypoints, connections, min_confidence=0.3, debug=False):
    """
    Draw 2D skeleton overlay on frame.

    Args:
        frame: Input frame (modified in-place)
        keypoints: Array of shape (N, 4) with [x, y, z, confidence]
        connections: List of (start_idx, end_idx) tuples
        min_confidence: Minimum confidence to draw keypoint
        debug: Print debug info
    """
    h, w = frame.shape[:2]

    # DEBUG first call
    if debug:
        print(f"\nDEBUG DRAW_SKELETON: Called")
        print(f"  Frame size: {w}x{h}")
        print(f"  Keypoints shape: {keypoints.shape}")
        print(f"  Connections: {len(connections)}")
        print(f"  Min confidence: {min_confidence}")
        if len(keypoints) > 0:
            print(f"  Keypoint 0 (nose): {keypoints[0]}")
            print(f"  Keypoint range X: [{keypoints[:, 0].min():.1f}, {keypoints[:, 0].max():.1f}]")
            print(f"  Keypoint range Y: [{keypoints[:, 1].min():.1f}, {keypoints[:, 1].max():.1f}]")

    # Helper function to determine keypoint group
    def get_keypoint_style(idx):
        """Return (outer_radius, inner_radius, alpha) based on keypoint group."""
        if idx < 23:  # Body (0-17) and feet (17-23)
            return (4, 2, 1.0)  # Normal size, full opacity
        elif 23 <= idx < 91:  # Face (68 keypoints)
            return (1, 0, 0.55)  # Very small, semi-transparent
        elif 91 <= idx < 133:  # Hands (left 91-112, right 112-133)
            return (1, 0, 0.55)  # Very small, semi-transparent
        else:
            return (4, 2, 1.0)  # Default to body style

    # Draw connections (skeleton lines)
    for start_idx, end_idx in connections:
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue

        start_kp = keypoints[start_idx]
        end_kp = keypoints[end_idx]

        # Check confidence
        if start_kp[3] < min_confidence or end_kp[3] < min_confidence:
            continue

        # Get 2D coordinates
        start_pt = (int(start_kp[0]), int(start_kp[1]))
        end_pt = (int(end_kp[0]), int(end_kp[1]))

        # Check if points are within frame
        if (0 <= start_pt[0] < w and 0 <= start_pt[1] < h and
            0 <= end_pt[0] < w and 0 <= end_pt[1] < h):

            # Average confidence for line color
            avg_conf = (start_kp[3] + end_kp[3]) / 2.0
            color = get_confidence_color(avg_conf)

            # Draw line
            cv2.line(frame, start_pt, end_pt, color, 2, cv2.LINE_AA)

    # Draw keypoints with differentiated styles for body/face/hands
    # Separate into opaque and transparent groups for efficient rendering
    opaque_keypoints = []
    transparent_keypoints = []

    for idx, kp in enumerate(keypoints):
        if kp[3] < min_confidence:
            continue

        pt = (int(kp[0]), int(kp[1]))

        # Check if point is within frame
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            outer_radius, inner_radius, alpha = get_keypoint_style(idx)
            color = get_confidence_color(kp[3])

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
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)


def get_head_top_point(keypoints, scores, min_confidence=0.3):
    """
    Estimate top of head from available keypoints for better neck angle sensitivity.

    Args:
        keypoints: (N, 3) array with [x, y, z] in meters (camera space)
        scores: (N,) confidence scores
        min_confidence: Minimum confidence threshold

    Returns:
        np.array or None: [x, y, z] of estimated head top, or None if insufficient data
    """
    if len(keypoints) < 5:
        return None

    # Check if we have ears (preferred) or eyes
    ears_valid = (scores[3] >= min_confidence and scores[4] >= min_confidence)
    eyes_valid = (scores[1] >= min_confidence and scores[2] >= min_confidence)

    if ears_valid:
        # Best case: use ear midpoint + vertical offset
        # Ears are roughly mid-head level, top of head is ~12cm above
        ear_midpoint = (keypoints[3] + keypoints[4]) / 2
        head_top = ear_midpoint.copy()
        head_top[2] += 0.12  # Add 12cm in Z-axis (vertical in camera space)
        return head_top
    elif eyes_valid:
        # Fallback: use eye midpoint + vertical offset
        # Eyes are lower, so need more offset (~15cm to head top)
        eye_midpoint = (keypoints[1] + keypoints[2]) / 2
        head_top = eye_midpoint.copy()
        head_top[2] += 0.15  # Add 15cm in Z-axis
        return head_top
    else:
        return None


def calculate_head_tilt_angle(keypoints, scores, min_confidence=0.3):
    """
    Calculate head tilt relative to neck using vector angle method.

    Measures the angle between:
    - Neck vector (shoulder midpoint → ear midpoint)
    - Face forward vector (ear midpoint → nose)

    Args:
        keypoints: (N, 3) array with [x, y, z] in meters (camera space)
        scores: (N,) confidence scores
        min_confidence: Minimum confidence for calculation

    Returns:
        float: Head tilt in degrees
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

    # Neck vector (from neck base upward to ears)
    neck_vector = ear_midpoint - neck_base

    # Face forward vector (from ears forward to nose)
    nose = keypoints[0]
    face_vector = nose - ear_midpoint

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

    return head_tilt


def plot_3d_skeleton(keypoints, scores, connections, min_confidence=0.3, view_angle=(20, 45)):
    """
    Create 3D skeleton visualization with neck angle display.

    Args:
        keypoints: (N, 3) array with [x, y, z] in meters (camera space)
        scores: (N,) confidence scores
        connections: List of (start_idx, end_idx) tuples
        min_confidence: Minimum confidence threshold
        view_angle: (elevation, azimuth) for camera view

    Returns:
        numpy array: RGB image of the 3D plot
    """
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Fix orientation: Swap axes so Z becomes vertical
    # Camera coords: [X, Y, Z] where Z varies with height
    # Plot coords: [X, Z, -Y] where Z is vertical axis

    # Only flip Y for front-back orientation
    keypoints_flipped = keypoints.copy()
    keypoints_flipped[:, 1] = -keypoints_flipped[:, 1]  # Flip Y only

    # Draw connections (skeleton lines)
    for start_idx, end_idx in connections:
        if start_idx >= len(keypoints_flipped) or end_idx >= len(keypoints_flipped):
            continue
        if scores[start_idx] < min_confidence or scores[end_idx] < min_confidence:
            continue

        start_pt = keypoints_flipped[start_idx]
        end_pt = keypoints_flipped[end_idx]

        avg_conf = (scores[start_idx] + scores[end_idx]) / 2.0
        color = get_confidence_color(avg_conf)
        # Convert BGR to RGB and normalize
        color_rgb = (color[2]/255, color[1]/255, color[0]/255)

        # Plot with swapped axes: [X, -Z, Y]
        ax.plot([start_pt[0], end_pt[0]],    # X: left-right
                [start_pt[2], end_pt[2]],     # -Z: vertical (flipped)
                [start_pt[1], end_pt[1]],     # Y: depth
                color=color_rgb, linewidth=2)

    # Draw custom neck line (shoulder midpoint to ear midpoint)
    if len(keypoints_flipped) > 6:
        shoulders_valid = (scores[5] >= min_confidence and scores[6] >= min_confidence)
        ears_valid = (scores[3] >= min_confidence and scores[4] >= min_confidence)

        if shoulders_valid and ears_valid:
            # Neck base (shoulder midpoint)
            left_shoulder = keypoints_flipped[5]
            right_shoulder = keypoints_flipped[6]
            neck_base = (left_shoulder + right_shoulder) / 2

            # Neck top (ear midpoint)
            left_ear = keypoints_flipped[3]
            right_ear = keypoints_flipped[4]
            ear_midpoint = (left_ear + right_ear) / 2

            # Calculate average confidence
            avg_conf = (scores[3] + scores[4] + scores[5] + scores[6]) / 4.0
            color = get_confidence_color(avg_conf)
            color_rgb = (color[2]/255, color[1]/255, color[0]/255)

            # Draw neck line (swapped axes)
            ax.plot([neck_base[0], ear_midpoint[0]],    # X: left-right
                    [neck_base[2], ear_midpoint[2]],    # Z: vertical
                    [neck_base[1], ear_midpoint[1]],    # Y: depth
                    color=color_rgb, linewidth=4)  # Thicker neck line

            # Draw face direction vector (ear midpoint to nose) for visual feedback
            nose_valid = scores[0] >= min_confidence
            if nose_valid:
                nose = keypoints_flipped[0]

                # Calculate face direction vector
                face_direction = nose - ear_midpoint
                face_magnitude = np.linalg.norm(face_direction)

                if face_magnitude > 1e-8:
                    # Extend beyond nose by 80cm (0.8 meters) for high visibility
                    extension_length = 0.8
                    extended_point = nose + (face_direction / face_magnitude) * extension_length

                    # Draw extended face vector as THICK SOLID RED line
                    # NOTE: If it's showing blue, matplotlib might be using BGR!
                    gaze_line_color = (0.0, 0.0, 1.0)  # Pure red in BGR: B=0, G=0, R=1
                    ax.plot([ear_midpoint[0], extended_point[0]],
                            [ear_midpoint[2], extended_point[2]],
                            [ear_midpoint[1], extended_point[1]],
                            color=gaze_line_color, linewidth=6, linestyle='-', alpha=1.0)

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
            # Create mask for this group's keypoints
            group_mask = valid_mask.copy()
            group_mask[:start_idx] = False
            group_mask[end_idx:] = False

            if group_mask.any():
                group_kp = keypoints_flipped[group_mask]
                group_scores = scores[group_mask]

                # Color by confidence
                colors = [get_confidence_color(s) for s in group_scores]
                colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]

                # Scatter with swapped axes and group-specific styling
                ax.scatter(group_kp[:, 0], group_kp[:, 2], group_kp[:, 1],
                           c=colors_rgb, s=group['size'], alpha=group['alpha'],
                           depthshade=True)

    # Set axis labels and limits
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')      # Z is vertical axis
    ax.set_zlabel('Y (m)')      # -Y is depth
    ax.set_xlim([-1.5, 1.5])    # X: left-right
    ax.set_ylim([3, 7])          # Z: vertical (larger values = higher)
    ax.set_zlim([-2, 2])         # -Y: depth

    # Calculate and display head tilt angle
    head_tilt = calculate_head_tilt_angle(keypoints, scores, min_confidence)
    if head_tilt is not None:
        title_text = f'3D Pose Estimation | Head Tilt: {head_tilt:+.1f}°'
    else:
        title_text = '3D Pose Estimation | Head Tilt: N/A'

    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    ax.set_title(title_text, fontsize=12, fontweight='bold')

    # Convert to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]  # Drop alpha channel

    plt.close(fig)
    return img


def print_3d_summary(frame_id, pose_results, timings, print_interval=30):
    """
    Print 3D keypoint statistics to console.

    Args:
        frame_id: Current frame number
        pose_results: List of pose results from MMPose
        timings: Dict with 'det' and 'pose' timing in seconds
        print_interval: Print every N frames
    """
    if frame_id % print_interval != 0:
        return

    print("\n" + "=" * 70)
    print(f"Frame {frame_id} Summary:")
    print("=" * 70)

    if not pose_results:
        print("  No persons detected")
        return

    print(f"  Persons detected: {len(pose_results)}")

    for person_idx, result in enumerate(pose_results):
        # Access keypoints from PoseDataSample object
        if not hasattr(result, 'pred_instances'):
            print(f"  Person {person_idx}: No keypoint data")
            continue

        pred_instances = result.pred_instances

        # Convert to numpy if needed (could be tensor or already numpy)
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

        # Handle 2D keypoints (add z=0 column if needed)
        if keypoints.shape[1] == 2:
            keypoints = np.concatenate([keypoints, np.zeros((len(keypoints), 1))], axis=1)

        # Count valid keypoints
        valid_mask = scores > 0.3
        valid_count = np.sum(valid_mask)
        total_count = len(scores)

        print(f"\n  Person {person_idx}: {valid_count}/{total_count} valid keypoints")

        # Statistics by body part
        for group_name, (start, end) in KEYPOINT_GROUPS.items():
            group_scores = scores[start:end]
            group_keypoints = keypoints[start:end]

            group_valid = group_scores > 0.3
            if not group_valid.any():
                continue

            avg_z = np.mean(group_keypoints[group_valid, 2])
            avg_conf = np.mean(group_scores[group_valid])
            valid_in_group = np.sum(group_valid)
            total_in_group = end - start

            print(f"    {group_name:12s}: {valid_in_group:2d}/{total_in_group:2d} keypoints, "
                  f"avg Z = {avg_z:+.2f}m, conf = {avg_conf:.2f}")

    # Timing statistics
    det_ms = timings['det'] * 1000
    pose_ms = timings['pose'] * 1000
    total_ms = det_ms + pose_ms
    fps = 1000.0 / total_ms if total_ms > 0 else 0

    print(f"\n  Processing: Detection={det_ms:.1f}ms, Pose={pose_ms:.1f}ms, "
          f"Total={total_ms:.1f}ms ({fps:.1f} FPS)")


def main():
    """Main function."""
    print("\n" + "=" * 70)
    print("MMPose 3D Real-time Demo with ZMQ Camera")
    print("=" * 70)

    # Load configuration
    print("\n[Step 1/5] Loading configuration...")
    config = load_config()
    print("✓ Configuration loaded")

    # Discover cameras
    print("\n[Step 2/5] Discovering cameras...")
    available_cameras = CameraSourceFactory.discover_cameras(config, max_cameras=10)

    if not available_cameras:
        print("\n❌ ERROR: No cameras found!")
        print("\nTroubleshooting:")
        print("  1. Check if Windows camera sender is running")
        print("  2. Verify Windows host IP in youquantipy_config.json")
        print("  3. Check firewall settings (ports 5550-5560)")
        return 1

    print(f"\n✓ Found {len(available_cameras)} camera(s):")
    for cam_idx, cam_name, width, height in available_cameras:
        print(f"    Camera {cam_idx}: {cam_name} ({width}x{height})")

    # Connect to first camera
    camera_index = available_cameras[0][0]
    print(f"\n[Step 3/5] Connecting to camera {camera_index}...")

    camera = CameraSourceFactory.create(config, camera_index=camera_index)

    if not camera.open():
        print(f"\n❌ ERROR: Failed to open camera {camera_index}")
        return 1

    print(f"✓ Camera connected")
    print(f"    Name: {camera.get_camera_name()}")
    print(f"    Resolution: {camera.get_resolution()}")
    print(f"    FPS: {camera.get_fps()}")
    print(f"    Backend: {camera.get_backend_name()}")

    # Initialize models
    print("\n[Step 4/5] Initializing MMPose models...")
    try:
        detector, pose_estimator = init_models(config)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize models: {e}")
        camera.release()
        return 1

    mmpose_config = config['mmpose_3d_pipeline']
    det_conf_threshold = mmpose_config['person_detector']['confidence_threshold']
    pose_conf_threshold = mmpose_config['pose_estimator']['confidence_threshold']

    # Main processing loop
    print("\n[Step 5/5] Starting real-time processing...")
    print("\n" + "=" * 70)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("=" * 70)

    frame_count = 0
    fps_history = deque(maxlen=30)  # Track FPS over last 30 frames

    try:
        while True:
            loop_start = time.time()

            # Read frame from camera
            success, frame = camera.read()

            if not success or frame is None:
                print("WARNING: Frame read failed, skipping...")
                time.sleep(0.01)
                continue

            frame_count += 1

            # Run person detection
            det_start = time.time()

            try:
                det_results = inference_detector(detector, frame)
                det_time = time.time() - det_start

                # Verify results structure
                if not hasattr(det_results, 'pred_instances'):
                    raise RuntimeError("Detection results missing pred_instances")

                # Extract person bounding boxes
                person_bboxes = extract_person_bboxes(det_results, conf_threshold=det_conf_threshold)
            except Exception as e:
                # Fallback: use full frame as bbox if detection fails
                print(f"WARNING: Detection failed ({e}), using full frame")
                det_time = time.time() - det_start
                h, w = frame.shape[:2]
                person_bboxes = np.array([[0, 0, w, h]])

            # Run pose estimation
            pose_start = time.time()
            pose_results = []

            if len(person_bboxes) > 0:
                pose_results = inference_topdown(pose_estimator, frame, person_bboxes)

                # Debug: Print pose results structure
                if frame_count == 1:
                    print(f"\nDEBUG: Detected {len(person_bboxes)} person bbox(es)")
                    print(f"DEBUG: Got {len(pose_results)} pose result(s)")
                    if pose_results and hasattr(pose_results[0], 'pred_instances'):
                        pi = pose_results[0].pred_instances

                        # Convert to numpy if needed
                        kp = pi.keypoints
                        if hasattr(kp, 'cpu'):
                            kp = kp.cpu().numpy()
                        else:
                            kp = np.array(kp)

                        scores = pi.keypoint_scores
                        if hasattr(scores, 'cpu'):
                            scores = scores.cpu().numpy()
                        else:
                            scores = np.array(scores)

                        print(f"DEBUG: Keypoints shape: {kp.shape}")
                        print(f"DEBUG: Sample keypoints:\n{kp[:3]}")
                        print(f"DEBUG: Scores shape: {scores.shape}")
                        print(f"DEBUG: Score range: [{scores.min():.3f}, {scores.max():.3f}]")
                        print(f"DEBUG: Valid keypoints (>0.3): {(scores > 0.3).sum()}/{len(scores)}")

            pose_time = time.time() - pose_start

            # Create visualization frame
            vis_frame = frame.copy()

            # Draw bounding boxes
            for bbox in person_bboxes:
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Create 3D visualization for first person
            plot_3d = None
            if len(pose_results) > 0:
                result = pose_results[0]  # First person only
                if hasattr(result, 'pred_instances'):
                    pred_instances = result.pred_instances

                    # Extract 3D keypoints
                    kp_3d = pred_instances.keypoints
                    if hasattr(kp_3d, 'cpu'):
                        kp_3d = kp_3d.cpu().numpy()
                    else:
                        kp_3d = np.array(kp_3d)

                    scores = pred_instances.keypoint_scores
                    if hasattr(scores, 'cpu'):
                        scores = scores.cpu().numpy()
                    else:
                        scores = np.array(scores)

                    # Remove batch dimension if present: (1, 133, 3) -> (133, 3)
                    if kp_3d.ndim == 3 and kp_3d.shape[0] == 1:
                        kp_3d = kp_3d[0]
                    if scores.ndim == 2 and scores.shape[0] == 1:
                        scores = scores[0]

                    # DEBUG: Print first frame info
                    if frame_count == 1:
                        print(f"\nDEBUG 3D PLOT: Creating visualization")
                        print(f"  Keypoints shape: {kp_3d.shape}")
                        print(f"  Scores shape: {scores.shape}")
                        print(f"  Valid keypoints (>0.3): {(scores > 0.3).sum()}/{len(scores)}")
                        print(f"  X range: [{kp_3d[:, 0].min():.2f}, {kp_3d[:, 0].max():.2f}]")
                        print(f"  Y range: [{kp_3d[:, 1].min():.2f}, {kp_3d[:, 1].max():.2f}]")
                        print(f"  Z range: [{kp_3d[:, 2].min():.2f}, {kp_3d[:, 2].max():.2f}]")

                    # Generate 3D plot
                    plot_3d = plot_3d_skeleton(kp_3d, scores, BODY_CONNECTIONS,
                                                min_confidence=pose_conf_threshold)

            # Calculate FPS
            loop_time = time.time() - loop_start
            current_fps = 1.0 / loop_time if loop_time > 0 else 0
            fps_history.append(current_fps)
            avg_fps = np.mean(fps_history)

            # Draw info overlay
            info_y = 30
            cv2.putText(vis_frame, f"FPS: {avg_fps:.1f}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            info_y += 30
            cv2.putText(vis_frame, f"Persons: {len(person_bboxes)}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            info_y += 30
            total_ms = (det_time + pose_time) * 1000
            cv2.putText(vis_frame, f"Process: {total_ms:.1f}ms", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Combine video and 3D plot side-by-side
            if plot_3d is not None:
                # Resize 3D plot to match video height
                h_vid = vis_frame.shape[0]
                h_plot = plot_3d.shape[0]
                if h_plot != h_vid:
                    scale = h_vid / h_plot
                    w_plot_new = int(plot_3d.shape[1] * scale)
                    plot_3d = cv2.resize(plot_3d, (w_plot_new, h_vid))

                # Concatenate horizontally
                combined = np.hstack([vis_frame, plot_3d])
            else:
                combined = vis_frame

            # Show combined frame
            cv2.imshow("MMPose 3D Real-time Demo", combined)

            # Print 3D summary to console
            timings = {'det': det_time, 'pose': pose_time}
            print_3d_summary(frame_count, pose_results, timings, print_interval=30)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save current combined frame
                save_path = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(save_path, combined)
                print(f"\n✓ Saved frame to {save_path}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")

    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\n" + "=" * 70)
        print("Cleaning up...")
        print("=" * 70)

        camera.release()
        cv2.destroyAllWindows()

        print(f"\n✓ Processed {frame_count} frames total")
        if fps_history:
            print(f"✓ Average FPS: {np.mean(fps_history):.1f}")

        print("\nDemo completed successfully!")

    return 0


if __name__ == '__main__':
    exit(main())
