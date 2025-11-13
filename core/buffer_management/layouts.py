"""
Buffer Layouts - Type-safe dataclasses for simplified MMPose 3D pipeline.

This module provides buffer layouts for the MMPose 3D pose estimation pipeline:
- Frame buffer: Camera frames
- Pose buffer: 3D whole-body pose keypoints from RTMW3D

Usage:
    from core.buffer_management.layouts import FrameBufferLayout, Pose3DBufferLayout

    # Get layout from BufferCoordinator
    layout = coordinator.get_layout('pose3d', max_persons=4)

    # Access type-safe fields
    pose_data_offset = layout.pose_data_offset
    pose_data_size = layout.pose_data_size
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any
import numpy as np


@dataclass
class FrameBufferLayout:
    """
    Layout for camera frame buffer with ring buffer structure.

    Structure:
    [write_index(8)][width(4)][height(4)][frames...][metadata_slots...]
    """
    # Configuration
    ring_buffer_size: int
    frame_width: int
    frame_height: int

    # Header
    write_index_offset: int = 0
    write_index_size: int = 8
    resolution_offset: int = 8
    header_size: int = 16

    # Calculated fields
    frame_size: int = field(init=False)
    frame_offsets: List[int] = field(init=False, default_factory=list)
    metadata_start: int = field(init=False)
    metadata_offsets: List[int] = field(init=False, default_factory=list)
    total_size: int = field(init=False)

    def __post_init__(self):
        """Calculate dependent fields after initialization."""
        self.frame_size = self.frame_width * self.frame_height * 3  # BGR

        # Frame offsets in ring buffer
        self.frame_offsets = [
            self.header_size + i * self.frame_size
            for i in range(self.ring_buffer_size)
        ]

        # Metadata offsets (64 bytes per slot)
        self.metadata_start = self.header_size + self.frame_size * self.ring_buffer_size
        self.metadata_offsets = [
            self.metadata_start + i * 64
            for i in range(self.ring_buffer_size)
        ]

        self.total_size = self.metadata_start + 64 * self.ring_buffer_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return asdict(self)


@dataclass
class Pose3DBufferLayout:
    """
    Layout for RTMW3D 3D whole-body pose buffer with ring buffer structure.

    RTMW3D outputs 133 keypoints per person:
    - 17 body keypoints
    - 6 foot keypoints
    - 68 face keypoints
    - 42 hand keypoints (21 per hand)

    Each keypoint has 4 values: (x, y, z, confidence)

    Structure:
    [write_index(8)][pose_data_slots...][metadata_slots...]
    """
    # Configuration
    max_persons: int
    ring_buffer_size: int = 4  # Number of ring buffer slots
    keypoints_per_person: int = 133  # RTMW3D whole-body keypoints
    values_per_keypoint: int = 4  # (x, y, z, confidence)

    # Header
    write_index_offset: int = 0
    write_index_size: int = 8

    # Calculated fields
    pose_slot_size: int = field(init=False)  # Size of one pose data slot
    pose_offsets: List[int] = field(init=False, default_factory=list)  # Offset for each slot
    metadata_start: int = field(init=False)
    metadata_offsets: List[int] = field(init=False, default_factory=list)
    metadata_size: int = 128  # Metadata size per slot
    total_size: int = field(init=False)

    # Metadata structure (dtype for structured access)
    metadata_dtype: np.dtype = field(init=False)

    def __post_init__(self):
        """Calculate dependent fields for ring buffer layout."""
        # Calculate size of one pose data slot
        self.pose_slot_size = (
            self.max_persons *
            self.keypoints_per_person *
            self.values_per_keypoint *
            4  # float32
        )

        # Pose data offsets in ring buffer
        self.pose_offsets = [
            self.write_index_size + i * self.pose_slot_size
            for i in range(self.ring_buffer_size)
        ]

        # Metadata offsets (after all pose data slots)
        self.metadata_start = self.write_index_size + self.pose_slot_size * self.ring_buffer_size
        self.metadata_offsets = [
            self.metadata_start + i * self.metadata_size
            for i in range(self.ring_buffer_size)
        ]

        # Metadata structure
        self.metadata_dtype = np.dtype([
            ('frame_id', 'int64'),
            ('timestamp_ms', 'int64'),
            ('n_persons', 'int32'),
            ('ready', 'int8'),
            ('processing_time_ms', 'float32'),
            ('detection_time_ms', 'float32'),
            ('pose_time_ms', 'float32'),
        ], align=True)

        self.total_size = self.metadata_start + self.metadata_size * self.ring_buffer_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            'max_persons': self.max_persons,
            'ring_buffer_size': self.ring_buffer_size,
            'keypoints_per_person': self.keypoints_per_person,
            'values_per_keypoint': self.values_per_keypoint,
            'write_index_offset': self.write_index_offset,
            'write_index_size': self.write_index_size,
            'pose_slot_size': self.pose_slot_size,
            'pose_offsets': self.pose_offsets,
            'metadata_start': self.metadata_start,
            'metadata_offsets': self.metadata_offsets,
            'metadata_size': self.metadata_size,
            'metadata_dtype': self.metadata_dtype,
            'total_size': self.total_size
        }


@dataclass
class ResultsBufferLayout:
    """
    Layout for MediaPipe results buffer (landmarks + blendshapes).

    MediaPipe FaceLandmarker outputs:
    - 468 landmarks per face × 3 coords (x, y, z) × float32
    - 52 blendshapes per face × float32

    Structure:
    [landmarks...][blendshapes...][roi_metadata...][frame_metadata]
    """
    # Configuration
    max_faces: int

    # Landmarks section
    landmarks_offset: int = 0
    landmarks_per_face: int = 468  # MediaPipe FaceLandmarker output
    values_per_landmark: int = 3  # (x, y, z)

    # Blendshapes section
    blendshapes_per_face: int = 52  # MediaPipe blendshape count

    # ROI metadata section (bbox + tracking data per face)
    roi_metadata_per_face: int = 96  # bbox + tracking info

    # Frame metadata section
    metadata_size: int = 64

    # Calculated fields
    landmarks_size: int = field(init=False)
    blendshapes_offset: int = field(init=False)
    blendshapes_size: int = field(init=False)
    roi_metadata_offset: int = field(init=False)
    roi_metadata_size: int = field(init=False)
    metadata_offset: int = field(init=False)
    total_size: int = field(init=False)

    def __post_init__(self):
        """Calculate dependent fields."""
        # Landmarks
        self.landmarks_size = (
            self.max_faces *
            self.landmarks_per_face *
            self.values_per_landmark *
            4  # float32
        )

        # Blendshapes
        self.blendshapes_offset = self.landmarks_offset + self.landmarks_size
        self.blendshapes_size = (
            self.max_faces *
            self.blendshapes_per_face *
            4  # float32
        )

        # ROI metadata
        self.roi_metadata_offset = self.blendshapes_offset + self.blendshapes_size
        self.roi_metadata_size = self.max_faces * self.roi_metadata_per_face

        # Frame metadata
        self.metadata_offset = self.roi_metadata_offset + self.roi_metadata_size

        # Total size
        self.total_size = self.metadata_offset + self.metadata_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            'landmarks_offset': self.landmarks_offset,
            'landmarks_size': self.landmarks_size,
            'blendshapes_offset': self.blendshapes_offset,
            'blendshapes_size': self.blendshapes_size,
            'roi_metadata_offset': self.roi_metadata_offset,
            'roi_metadata_size': self.roi_metadata_size,
            'metadata_offset': self.metadata_offset,
            'metadata_size': self.metadata_size,
            'total_size': self.total_size
        }


@dataclass
class ROIBufferLayout:
    """
    Layout for Region of Interest (ROI) buffer with ring buffer structure.

    Stores cropped face regions for parallel processing:
    - Preprocessing disabled: 256×256×3 uint8 per ROI
    - Preprocessing enabled: 256×256×3 float32 per ROI (4× larger)

    Structure:
    [write_index(8)][ROI data ring buffer...][metadata slots...]
    """
    # Configuration
    max_faces: int
    roi_buffer_size: int  # Ring buffer slot count (must be power of 2)
    enable_preprocessing: bool = False

    # Header
    write_index_offset: int = 0
    write_index_size: int = 8

    # ROI dimensions and format
    roi_width: int = 256
    roi_height: int = 256
    roi_channels: int = 3  # RGB

    # Calculated fields
    roi_size: int = field(init=False)  # Bytes per single ROI
    roi_data_offset: int = field(init=False)
    roi_data_size: int = field(init=False)  # Total ROI data for all faces and slots
    metadata_offset_base: int = field(init=False)
    metadata_per_slot: int = field(init=False)
    total_size: int = field(init=False)

    def __post_init__(self):
        """Calculate dependent fields."""
        # Determine ROI size based on preprocessing mode
        if self.enable_preprocessing:
            # Preprocessed: float32
            self.roi_size = (
                self.roi_width *
                self.roi_height *
                self.roi_channels *
                4  # float32
            )
        else:
            # Raw: uint8
            self.roi_size = (
                self.roi_width *
                self.roi_height *
                self.roi_channels  # uint8
            )

        # Data section
        self.roi_data_offset = self.write_index_size
        self.roi_data_size = (
            self.roi_buffer_size *
            self.max_faces *
            self.roi_size
        )

        # Metadata section (per slot)
        self.metadata_offset_base = self.roi_data_offset + self.roi_data_size
        self.metadata_per_slot = 1024  # Fixed per-slot metadata size

        # Total
        self.total_size = (
            self.write_index_size +
            self.roi_data_size +
            (self.roi_buffer_size * self.metadata_per_slot)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            'max_faces': self.max_faces,
            'roi_buffer_size': self.roi_buffer_size,
            'enable_preprocessing': self.enable_preprocessing,
            'write_index_offset': self.write_index_offset,
            'write_index_size': self.write_index_size,
            'roi_size': self.roi_size,
            'roi_data_offset': self.roi_data_offset,
            'roi_data_size': self.roi_data_size,
            'metadata_offset_base': self.metadata_offset_base,
            'metadata_per_slot': self.metadata_per_slot,
            'total_size': self.total_size
        }


@dataclass
class DetectionBufferLayout:
    """
    Layout for SCRFD face detection buffer.

    Stores detection metadata and crops for each detected face.

    Structure:
    [header(64B)][detection_metadata...][crop_buffer...]
    """
    # Configuration
    max_faces: int
    crop_width: int = 192
    crop_height: int = 192
    crop_channels: int = 3  # RGB

    # Header fields
    header_size: int = 64
    detection_metadata_size: int = 128  # Per-face detection metadata

    # Calculated fields
    crop_size: int = field(init=False)  # Bytes per crop
    detection_metadata_offset: int = field(init=False)
    crop_buffer_offset: int = field(init=False)
    total_size: int = field(init=False)

    def __post_init__(self):
        """Calculate dependent fields."""
        # Crop size (uint8 for image data)
        self.crop_size = (
            self.crop_width *
            self.crop_height *
            self.crop_channels  # uint8
        )

        # Metadata offset
        self.detection_metadata_offset = self.header_size

        # Crop buffer offset
        self.crop_buffer_offset = (
            self.header_size +
            (self.max_faces * self.detection_metadata_size)
        )

        # Total size
        self.total_size = (
            self.header_size +
            (self.max_faces * self.detection_metadata_size) +
            (self.max_faces * self.crop_size)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            'max_faces': self.max_faces,
            'header_size': self.header_size,
            'detection_metadata_size': self.detection_metadata_size,
            'crop_size': self.crop_size,
            'detection_metadata_offset': self.detection_metadata_offset,
            'crop_buffer_offset': self.crop_buffer_offset,
            'crop_width': self.crop_width,
            'crop_height': self.crop_height,
            'total_size': self.total_size
        }


@dataclass
class EmbeddingBufferLayout:
    """
    Layout for ArcFace embedding buffer for participant recognition.

    Stores 512-dimensional embeddings for each detected face.

    Structure:
    [write_index(8)][frame_id(8)][n_embeddings(4)][padding(4)][embeddings...]
    """
    # Configuration
    max_faces: int
    embedding_dim: int = 512  # ArcFace standard embedding dimension

    # Header
    header_size: int = 24  # write_index(8) + frame_id(8) + n_embeddings(4) + padding(4)

    # Calculated fields
    embedding_size: int = field(init=False)  # Total embedding data size
    total_size: int = field(init=False)

    def __post_init__(self):
        """Calculate dependent fields."""
        # Embedding section
        self.embedding_size = (
            self.max_faces *
            self.embedding_dim *
            4  # float32
        )

        # Total
        self.total_size = self.header_size + self.embedding_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            'max_faces': self.max_faces,
            'embedding_dim': self.embedding_dim,
            'header_size': self.header_size,
            'embedding_size': self.embedding_size,
            'total_size': self.total_size
        }


@dataclass
class DisplayBufferLayout:
    """
    Layout for GUI display buffer with ring buffer structure.

    Optimized for GUI rendering - stores frame references and face detection data.

    Structure:
    [write_index(8)][slot0][slot1][slot2][slot3...]
    where each slot contains: [frame_metadata][face_data...]
    """
    # Configuration
    ring_buffer_size: int
    frame_width: int
    frame_height: int
    max_faces: int

    # Header
    write_index_offset: int = 0
    write_index_size: int = 8

    # Frame metadata per slot
    frame_metadata_size: int = 64  # frame_id, timestamp, n_faces, etc.

    # Face data per face
    face_data_per_face: int = 3856  # Depends on landmarks + metadata

    # Calculated fields
    face_data_size: int = field(init=False)  # Total face data per slot
    slot_size: int = field(init=False)
    total_size: int = field(init=False)

    def __post_init__(self):
        """Calculate dependent fields."""
        # Face data section (all faces in slot)
        self.face_data_size = self.max_faces * self.face_data_per_face

        # Slot size (metadata + all face data)
        self.slot_size = self.frame_metadata_size + self.face_data_size

        # Total (write_index + all slots)
        self.total_size = (
            self.write_index_size +
            (self.ring_buffer_size * self.slot_size)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            'ring_buffer_size': self.ring_buffer_size,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'max_faces': self.max_faces,
            'write_index_offset': self.write_index_offset,
            'write_index_size': self.write_index_size,
            'frame_metadata_size': self.frame_metadata_size,
            'face_data_per_face': self.face_data_per_face,
            'face_data_size': self.face_data_size,
            'slot_size': self.slot_size,
            'total_size': self.total_size
        }


# Convenience function to create layout instances
def create_layout(buffer_type: str, **kwargs) -> Any:
    """
    Factory function to create buffer layout instances.

    Args:
        buffer_type: Type of buffer ('frame', 'pose3d', 'results', 'roi', 'detection', 'embedding', 'display')
        **kwargs: Configuration parameters for the layout

    Returns:
        Appropriate buffer layout dataclass instance

    Example:
        layout = create_layout('pose3d', max_persons=4)
        pose_offset = layout.pose_data_offset
    """
    layout_classes = {
        'frame': FrameBufferLayout,
        'pose3d': Pose3DBufferLayout,
        'pose': Pose3DBufferLayout,  # Alias
        'results': ResultsBufferLayout,
        'roi': ROIBufferLayout,
        'detection': DetectionBufferLayout,
        'embedding': EmbeddingBufferLayout,
        'display': DisplayBufferLayout,
        'gui': DisplayBufferLayout,  # Alias
    }

    if buffer_type not in layout_classes:
        raise ValueError(f"Unknown buffer type: {buffer_type}. "
                        f"Valid types: {list(layout_classes.keys())}")

    return layout_classes[buffer_type](**kwargs)
