"""
Coordinate Transformation System for Face Landmark Drawing

This module provides centralized coordinate transformation utilities to convert
landmarks between different coordinate spaces in the face processing pipeline.

Coordinate Spaces:
1. **Crop Space**: 192×192 pixel coordinates (MediaPipe input crops)
2. **Bbox Space**: Face bounding box in full frame (SCRFD detection)
3. **Frame Space**: Full camera frame resolution (e.g., 1280×720)
4. **Canvas Space**: Tkinter canvas display coordinates

Transformation Pipeline:
    Crop Space → Bbox Space → Frame Space → Canvas Space
    (192×192)     (variable)    (1280×720)   (640×480 display)

Author: YouQuantiPy Team
Date: 2025-01-26
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class CoordinateSystem:
    """
    Centralized coordinate system configuration for landmark transformations.

    This class provides the single source of truth for all resolution-related
    parameters and coordinate transformations across the face processing pipeline.

    Attributes:
        crop_size: MediaPipe crop resolution (default: 192×192)
        frame_width: Full camera frame width (e.g., 1280)
        frame_height: Full camera frame height (e.g., 720)
    """
    crop_size: int = 192  # MediaPipe crop resolution (SCRFD default)
    frame_width: int = 1280  # Will be overridden with actual frame resolution
    frame_height: int = 720

    def transform_landmark_crop_to_frame(
        self,
        landmark_crop: Tuple[float, float, float],
        face_bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Transform a single landmark from crop space to full frame space.

        **Transformation Steps:**
        1. Scale from 192×192 crop to actual face bbox dimensions
        2. Translate to bbox position in full frame

        Args:
            landmark_crop: Landmark in crop space (x, y, z) where x,y ∈ [0, 192]
            face_bbox: Face bounding box in frame space (x1, y1, x2, y2)

        Returns:
            Landmark in frame space (x_frame, y_frame, z) where x,y ∈ [0, frame_width/height]

        Example:
            # Landmark at center of 192×192 crop
            landmark_crop = (96, 96, 0)

            # Face bbox at (100, 50) with size 300×300
            face_bbox = (100, 50, 400, 350)

            # Result: landmark at center of face bbox (250, 200, 0)
            result = transform_landmark_crop_to_frame(landmark_crop, face_bbox)
        """
        x1, y1, x2, y2 = face_bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # Calculate scale factors: bbox size / crop size
        # E.g., 300px bbox / 192px crop = 1.5625 scale factor
        scale_x = bbox_width / self.crop_size
        scale_y = bbox_height / self.crop_size

        # Transform: Scale from crop to bbox, then translate to frame position
        x_frame = (landmark_crop[0] * scale_x) + x1
        y_frame = (landmark_crop[1] * scale_y) + y1
        z_frame = landmark_crop[2]  # Z coordinate unchanged

        return (x_frame, y_frame, z_frame)

    def transform_landmarks_crop_to_frame(
        self,
        landmarks_crop: np.ndarray,
        face_bbox: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Transform multiple landmarks from crop space to full frame space (vectorized).

        Args:
            landmarks_crop: Nx3 array of landmarks in crop space (x, y, z)
            face_bbox: Face bounding box in frame space (x1, y1, x2, y2)

        Returns:
            Nx3 array of landmarks in frame space

        Example:
            landmarks_crop = np.array([
                [96, 96, 0],    # Landmark 0: center of crop
                [48, 48, 0],    # Landmark 1: upper-left quadrant
                [144, 144, 0]   # Landmark 2: lower-right quadrant
            ])

            face_bbox = (100, 50, 400, 350)  # 300×300 face bbox

            result = transform_landmarks_crop_to_frame(landmarks_crop, face_bbox)
            # Result:
            # [[250, 200, 0],   # Center of face
            #  [175, 125, 0],   # Upper-left of face
            #  [325, 275, 0]]   # Lower-right of face
        """
        x1, y1, x2, y2 = face_bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # Vectorized transformation
        scale_x = bbox_width / self.crop_size
        scale_y = bbox_height / self.crop_size

        # Create output array (copy to avoid modifying input)
        landmarks_frame = landmarks_crop.copy()

        # Transform x,y coordinates (leave z unchanged)
        landmarks_frame[:, 0] = (landmarks_crop[:, 0] * scale_x) + x1
        landmarks_frame[:, 1] = (landmarks_crop[:, 1] * scale_y) + y1

        return landmarks_frame

    def transform_landmark_frame_to_canvas(
        self,
        landmark_frame: Tuple[float, float, float],
        canvas_transform: dict
    ) -> Tuple[float, float, float]:
        """
        Transform landmark from frame space to canvas display space.

        Args:
            landmark_frame: Landmark in frame space (x, y, z)
            canvas_transform: Dict with keys:
                - 'video_bounds': (x_offset, y_offset, video_w, video_h)
                - 'frame_size': (frame_w, frame_h)

        Returns:
            Landmark in canvas space (x_canvas, y_canvas, z)
        """
        x_offset, y_offset, video_w, video_h = canvas_transform['video_bounds']
        frame_w, frame_h = canvas_transform['frame_size']

        # Scale from frame to canvas video area
        scale_x = video_w / frame_w
        scale_y = video_h / frame_h

        # Transform
        x_canvas = (landmark_frame[0] * scale_x) + x_offset
        y_canvas = (landmark_frame[1] * scale_y) + y_offset
        z_canvas = landmark_frame[2]

        return (x_canvas, y_canvas, z_canvas)

    def get_crop_to_frame_scale(
        self,
        face_bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float]:
        """
        Get scale factors for transforming from crop to frame space.

        Args:
            face_bbox: Face bounding box in frame space (x1, y1, x2, y2)

        Returns:
            (scale_x, scale_y) tuple
        """
        x1, y1, x2, y2 = face_bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        scale_x = bbox_width / self.crop_size
        scale_y = bbox_height / self.crop_size

        return (scale_x, scale_y)


def create_coordinate_system(config: dict, frame_width: int, frame_height: int) -> CoordinateSystem:
    """
    Factory function to create CoordinateSystem from config.

    Args:
        config: Configuration dictionary
        frame_width: Actual camera frame width
        frame_height: Actual camera frame height

    Returns:
        CoordinateSystem instance configured from config

    Example:
        config = {'scrfd_processing': {'crop_size': 192}}
        coord_sys = create_coordinate_system(config, 1280, 720)
    """
    scrfd_config = config.get('scrfd_processing', {})
    crop_size = scrfd_config.get('crop_size', 192)

    return CoordinateSystem(
        crop_size=crop_size,
        frame_width=frame_width,
        frame_height=frame_height
    )


def validate_transformation(
    landmark_crop: Tuple[float, float, float],
    landmark_frame: Tuple[float, float, float],
    face_bbox: Tuple[float, float, float, float],
    crop_size: int = 192
) -> bool:
    """
    Validate that a landmark transformation was performed correctly.

    Useful for debugging coordinate transformation issues.

    Args:
        landmark_crop: Original landmark in crop space
        landmark_frame: Transformed landmark in frame space
        face_bbox: Face bounding box used for transformation
        crop_size: Crop resolution (default: 192)

    Returns:
        True if transformation appears valid

    Example:
        landmark_crop = (96, 96, 0)  # Center of crop
        landmark_frame = (250, 200, 0)  # Expected: center of bbox
        face_bbox = (100, 50, 400, 350)  # 300×300 bbox

        valid = validate_transformation(landmark_crop, landmark_frame, face_bbox)
        # Returns True if transformation is correct
    """
    x1, y1, x2, y2 = face_bbox

    # Check if transformed landmark is within bbox bounds
    if not (x1 <= landmark_frame[0] <= x2):
        return False
    if not (y1 <= landmark_frame[1] <= y2):
        return False

    # Check if transformation preserves relative position
    # Landmark at (0, 0) in crop should map to (x1, y1) in frame
    # Landmark at (crop_size, crop_size) should map to (x2, y2)
    expected_x = ((landmark_crop[0] / crop_size) * (x2 - x1)) + x1
    expected_y = ((landmark_crop[1] / crop_size) * (y2 - y1)) + y1

    # Allow 1 pixel tolerance for floating point errors
    tolerance = 1.0
    if abs(landmark_frame[0] - expected_x) > tolerance:
        return False
    if abs(landmark_frame[1] - expected_y) > tolerance:
        return False

    return True
