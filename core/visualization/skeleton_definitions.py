"""
Skeleton Connection Definitions for RTMW3D (133 keypoints)

RTMW3D Keypoint Layout:
- 0-16:   COCO 17 body keypoints
- 17-22:  6 foot keypoints
- 23-90:  68 face keypoints
- 91-111: 21 left hand keypoints
- 112-132: 21 right hand keypoints
"""

# COCO 17-keypoint body skeleton
# Keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
#            5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
#            9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
#            13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
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

# 21-keypoint hand skeleton (MediaPipe format)
# Keypoints: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
HAND_CONNECTIONS = [
    # Thumb (wrist to tip)
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Face outline connections (simplified for 68-point face)
# This creates a simple outline: jaw, left_eyebrow, right_eyebrow, nose, left_eye, right_eye, mouth
FACE_CONTOUR_CONNECTIONS = [
    # Jaw contour (0-16)
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    # Left eyebrow (17-21)
    (17, 18), (18, 19), (19, 20), (20, 21),
    # Right eyebrow (22-26)
    (22, 23), (23, 24), (24, 25), (25, 26),
    # Nose bridge (27-30)
    (27, 28), (28, 29), (29, 30),
    # Nose bottom (31-35)
    (31, 32), (32, 33), (33, 34), (34, 35),
    # Left eye (36-41)
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
    # Right eye (42-47)
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
    # Outer mouth (48-59)
    (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
    (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
    # Inner mouth (60-67)
    (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60),
]

# Keypoint group indices in RTMW3D output
KEYPOINT_GROUPS = {
    'body': (0, 17),       # COCO 17 body keypoints
    'feet': (17, 23),      # 6 foot keypoints
    'face': (23, 91),      # 68 face keypoints (indexed from 23)
    'left_hand': (91, 112),   # 21 left hand keypoints
    'right_hand': (112, 133)  # 21 right hand keypoints
}

# Keypoint styling by group
# Format: (outer_radius, inner_radius, alpha)
KEYPOINT_STYLES = {
    'body': (4, 2, 1.0),       # Normal size, full opacity
    'feet': (4, 2, 1.0),       # Normal size, full opacity
    'face': (1, 0, 0.55),      # Very small, semi-transparent
    'left_hand': (1, 0, 0.55), # Very small, semi-transparent
    'right_hand': (1, 0, 0.55) # Very small, semi-transparent
}


def get_keypoint_style(idx):
    """
    Get rendering style for keypoint based on its index.

    Args:
        idx: Keypoint index (0-132)

    Returns:
        tuple: (outer_radius, inner_radius, alpha)
    """
    if idx < 23:  # Body + feet
        return KEYPOINT_STYLES['body']
    elif 23 <= idx < 91:  # Face
        return KEYPOINT_STYLES['face']
    elif 91 <= idx < 133:  # Hands
        return KEYPOINT_STYLES['left_hand']  # Same style for both hands
    else:
        return KEYPOINT_STYLES['body']  # Default


def get_connections_for_keypoints(keypoints, keypoint_offset=0):
    """
    Get appropriate connections for a keypoint subset.

    Args:
        keypoints: Array of keypoints
        keypoint_offset: Offset to add to connection indices (for face/hand subsets)

    Returns:
        list: Connection tuples with adjusted indices
    """
    num_kps = len(keypoints)

    if num_kps == 17:  # Body only
        return BODY_CONNECTIONS
    elif num_kps == 21:  # Hand
        return HAND_CONNECTIONS
    elif num_kps == 68:  # Face
        return FACE_CONTOUR_CONNECTIONS
    elif num_kps == 133:  # Full body
        # Return all connections with appropriate offsets
        connections = []
        connections.extend(BODY_CONNECTIONS)
        # Add hand connections with offsets
        connections.extend([(i + 91, j + 91) for i, j in HAND_CONNECTIONS])  # Left hand
        connections.extend([(i + 112, j + 112) for i, j in HAND_CONNECTIONS])  # Right hand
        # Add face connections with offset
        connections.extend([(i + 23, j + 23) for i, j in FACE_CONTOUR_CONNECTIONS])
        return connections
    else:
        return []
