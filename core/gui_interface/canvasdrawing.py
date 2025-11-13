"""
Canvas Drawing Module for YouQuantiPy
Handles all canvas rendering operations with optimizations for face and pose tracking.
FIXED: Bbox coordinate transformation for 720p+ resolutions
"""

import time
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from typing import Dict, List, Tuple, Optional, Any
import logging

# RTMPose visualization utilities (replaces MediaPipe)
from core.visualization import (
    RTMPoseVisualizer,
    BODY_CONNECTIONS,
    HAND_CONNECTIONS,
    FACE_CONTOUR_CONNECTIONS,
    KEYPOINT_GROUPS
)

# GPU acceleration support
from .gpu_drawing_manager import get_gpu_manager
from PIL import ImageFont, ImageDraw

logger = logging.getLogger(__name__)

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS

# Display flags bit encoding for enrollment states (from gui_processing_worker.py)
# Used to decode enrollment state from display_flags field
DISPLAY_FLAG_ENROLLED = 0x01       # Bit 0: Fully enrolled (green overlay)
DISPLAY_FLAG_COLLECTING = 0x02     # Bit 1: Collecting samples (yellow overlay)
DISPLAY_FLAG_VALIDATING = 0x04     # Bit 2: Validating (blue overlay)
DISPLAY_FLAG_FAILED = 0x08         # Bit 3: Enrollment failed (red overlay)

# PIL font cache for Unicode text rendering (performance optimization)
_pil_font_cache = {}

def _get_pil_font(font_size):
    """Load and cache PIL font for Unicode text rendering."""
    if font_size not in _pil_font_cache:
        try:
            _pil_font_cache[font_size] = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                font_size
            )
        except Exception as e:
            logger.warning(f"Failed to load DejaVu Sans font: {e}, using default")
            _pil_font_cache[font_size] = ImageFont.load_default()
    return _pil_font_cache[font_size]

def _draw_pil_text_with_outline(frame, text, position, font_size=24, color=(255, 255, 0), outline_color=(0, 0, 0)):
    """
    Draw Unicode text using PIL with outline for visibility.

    Args:
        frame: OpenCV BGR frame (numpy array)
        text: Text to render (supports Unicode)
        position: (x, y) tuple for text position
        font_size: Font size in points (default 24 ≈ cv2 scale 0.7)
        color: Text color as RGB tuple (default yellow)
        outline_color: Outline color as RGB tuple (default black)

    Returns:
        Modified frame with text rendered
    """
    # Convert BGR to RGB for PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)

    font = _get_pil_font(font_size)

    x, y = position
    # Draw outline (4-direction for visibility)
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

    # Draw main text
    draw.text((x, y), text, font=font, fill=color)

    # Convert back to BGR for OpenCV
    frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    frame[:] = frame_bgr
    return frame


class CanvasDrawingManager:
    """
    Manages efficient canvas drawing for face and pose tracking.
    Implements proper layering and coordinate transformation.
    """
    
    def __init__(self, config=None):
        # Canvas object caches
        self.canvas_objects = {}  # {canvas_idx: {face_lines: {}, face_labels: {}, ...}}
        self.transform_cache = {}  # {canvas_idx: {video_bounds: (x, y, w, h), frame_size: (w, h)}}

        # Performance tracking
        self.last_frame_time = {}  # {canvas_idx: timestamp}
        self.last_face_state = {}  # {canvas_idx: {face_id: (centroid, landmark_count)}}
        self.last_face_ids = {}  # {canvas_idx: tuple(face_ids)}

        # Load configuration
        self.config = config or {}
        canvas_config = self.config.get('gui_interface', {}).get('canvas', {})

        # GPU acceleration settings
        gpu_config = self.config.get('gpu_acceleration', {})
        self.enable_gpu = gpu_config.get('enabled', True)  # Enable by default
        self.gpu_profile = gpu_config.get('profile_performance', False)

        # Initialize GPU manager
        self.gpu_manager = get_gpu_manager(
            enable_gpu=self.enable_gpu,
            profile_performance=self.gpu_profile
        )

        logger.info(f"[CanvasDrawing] GPU acceleration: {'ENABLED' if self.gpu_manager.gpu_available else 'DISABLED (using CPU fallback)'}")
        
        # Frame rate limiting
        self.min_frame_interval = canvas_config.get('refresh_interval', 0.033)  # 30 FPS max per canvas
        
        # RTMPose connections (replaces MediaPipe)
        self.body_connections = BODY_CONNECTIONS  # 17 COCO body keypoints
        self.hand_connections = HAND_CONNECTIONS  # 21 hand keypoints
        self.face_connections = FACE_CONTOUR_CONNECTIONS  # 68 face keypoints
        self.pose_connections = BODY_CONNECTIONS  # Backward compatibility
        
        # Photo image cache to prevent garbage collection
        self._photo_images = {}
        
        # Simple landmark cache for smooth drawing when processing lags
        self.landmark_cache = {}  # {cache_key: (landmarks, timestamp)}
        self.landmark_cache_max_age = canvas_config.get('cache_timeout', 1.0)  # seconds
        
        # Draw modes from config
        draw_modes = canvas_config.get('draw_modes', {})
        self.debug_mode = draw_modes.get('debug_mode', True)  # Set to False to disable debug overlays
        self.face_draw_mode = draw_modes.get('face_draw_mode', 'full_contours')  # Options: 'full_contours', 'jaw_only', 'mesh'
        self.draw_jaw_overlay = draw_modes.get('draw_jaw_overlay', True)  # Draw jaw line as yellow overlay in debug mode

    @staticmethod
    def get_enrollment_color(enrollment_state: str) -> str:
        """
        Map enrollment state to face outline color.

        Color scheme:
        - UNKNOWN: Red - Face detected, not enrolled yet
        - COLLECTING: Yellow - Collecting enrollment samples
        - VALIDATING: Blue - Validating identity
        - ENROLLED: Green - Successfully enrolled
        - IMPROVING: Bright Green - Active and continuously learning
        - FAILED: Orange - Enrollment failed

        Args:
            enrollment_state: Enrollment state string (UNKNOWN, COLLECTING, etc.)

        Returns:
            Hex color string (e.g., '#FF4444')
        """
        color_map = {
            'UNKNOWN': '#FF4444',      # Red - needs attention
            'COLLECTING': '#FFB844',   # Yellow/Amber - processing
            'VALIDATING': '#4488FF',   # Blue - validating
            'ENROLLED': '#00FF00',     # Green - success
            'IMPROVING': '#00FF88',    # Bright Green - optimal state
            'FAILED': '#FF8844'        # Orange - error
        }
        return color_map.get(enrollment_state, '#FFFFFF')  # Default: white

    def _get_basic_face_connections(self):
        """Define basic face connections for key features when MediaPipe is not available."""
        connections = []
        
        # Jaw line (indices 0-16)
        for i in range(16):
            connections.append((i, i+1))
        
        # Left eyebrow (70, 63, 105, 66, 107)
        eyebrow_left = [70, 63, 105, 66, 107]
        for i in range(len(eyebrow_left)-1):
            connections.append((eyebrow_left[i], eyebrow_left[i+1]))
        
        # Right eyebrow (46, 53, 52, 65, 55)
        eyebrow_right = [46, 53, 52, 65, 55]
        for i in range(len(eyebrow_right)-1):
            connections.append((eyebrow_right[i], eyebrow_right[i+1]))
        
        # Left eye (outer)
        left_eye = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33]
        for i in range(len(left_eye)-1):
            connections.append((left_eye[i], left_eye[i+1]))
        
        # Right eye (outer)
        right_eye = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]
        for i in range(len(right_eye)-1):
            connections.append((right_eye[i], right_eye[i+1]))
        
        # Nose bridge
        nose_bridge = [6, 197, 196, 3, 51, 48, 115, 131, 102, 48, 64, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 321, 320, 305, 291, 330, 347, 346, 280, 425, 295, 279, 310, 392, 308, 324, 318]
        for i in range(0, len(nose_bridge)-1, 2):
            if i+1 < len(nose_bridge):
                connections.append((nose_bridge[i], nose_bridge[i+1]))
        
        # Lips (outer)
        lips_outer = [61, 84, 17, 314, 405, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 61]
        for i in range(len(lips_outer)-1):
            connections.append((lips_outer[i], lips_outer[i+1]))
        
        # Lips (inner)
        lips_inner = [62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 307, 310, 415, 308, 312, 13, 82, 81, 80, 78, 62]
        for i in range(len(lips_inner)-1):
            connections.append((lips_inner[i], lips_inner[i+1]))
        
        return connections
        
    def initialize_canvas(self, canvas: tk.Canvas, canvas_idx: int):
        """Initialize canvas with default transform and ensure proper layering"""
        if canvas_idx not in self.transform_cache:
            self.transform_cache[canvas_idx] = {}
        
        # Set default video bounds based on canvas size
        try:
            canvas.update_idletasks()
            W, H = canvas.winfo_width(), canvas.winfo_height()
            if W > 1 and H > 1:
                self.transform_cache[canvas_idx]['video_bounds'] = (0, 0, W, H)
                self.transform_cache[canvas_idx]['canvas_size'] = (W, H)
                print(f"[CanvasDrawing] Initialized canvas {canvas_idx} with size {W}x{H}")
        except:
            pass
    
    def should_skip_frame(self, canvas_idx: int) -> bool:
        """Check if we should skip this frame based on time elapsed."""
        current_time = time.time()
        last_time = self.last_frame_time.get(canvas_idx, 0)
        
        if current_time - last_time < self.min_frame_interval:
            return True
            
        self.last_frame_time[canvas_idx] = current_time
        return False
    
    def render_frame_to_canvas(self, frame_bgr: np.ndarray, canvas: tk.Canvas,
                            canvas_idx: int, original_resolution: tuple = None) -> Optional[ImageTk.PhotoImage]:
        """
        Render a frame to canvas with GPU-accelerated processing.
        Returns PhotoImage or None if canvas not ready.
        """
        try:
            # CRITICAL FIX: Ensure frame is completely independent of shared memory
            if frame_bgr is None:
                return None

            # CRITICAL FIX: Ensure canvas is properly realized before rendering
            # This fixes the black screen issue by waiting for widget to be ready
            if not canvas.winfo_exists():
                logger.warning(f"[CanvasDrawing] Canvas {canvas_idx} doesn't exist yet")
                return None

            # Force canvas to realize its geometry
            canvas.update_idletasks()

            # Check if canvas is visible (mapped to screen)
            if not canvas.winfo_viewable():
                logger.debug(f"[CanvasDrawing] Canvas {canvas_idx} not viewable yet")
                # Try to make it visible
                canvas.update()

            # Get canvas dimensions
            W, H = canvas.winfo_width(), canvas.winfo_height()
            if W <= 1 or H <= 1:
                logger.debug(f"[CanvasDrawing] Canvas {canvas_idx} has invalid size: {W}x{H}")
                return None

            # Get canvas config for display limits
            canvas_config = self.config.get('gui_interface', {}).get('canvas', {})

            # CRITICAL: Get frame dimensions BEFORE any downsampling
            frame_h, frame_w = frame_bgr.shape[:2]

            # FIX #1: Validate frame dimensions to prevent division by zero
            if frame_w <= 0 or frame_h <= 0:
                logger.warning(f"[CanvasDrawing] Invalid frame dimensions for canvas {canvas_idx}: {frame_w}x{frame_h} - skipping frame")
                return None

            # Store ORIGINAL frame dimensions (use parameter if provided, otherwise use frame shape)
            # This is crucial for correct landmark scaling
            if original_resolution:
                original_frame_w, original_frame_h = original_resolution
            else:
                # Fallback to frame dimensions if no original_resolution provided
                original_frame_w, original_frame_h = frame_w, frame_h

            # ALWAYS downsample to configured max for GUI display
            max_display_width = canvas_config.get('max_display_width', 640)
            max_display_height = canvas_config.get('max_display_height', 480)

            # Calculate if we need to downsample
            scale_factor = min(max_display_width / frame_w, max_display_height / frame_h, 1.0)
            if scale_factor < 1.0:
                display_w = int(frame_w * scale_factor)
                display_h = int(frame_h * scale_factor)
            else:
                display_w, display_h = frame_w, frame_h

            # FIX #4: Validate display dimensions before division (can be 0 if frame is tiny and gets downsampled)
            if display_w <= 0 or display_h <= 0:
                logger.error(f"[CanvasDrawing] Invalid display dimensions after downsampling: {display_w}x{display_h} (original frame: {frame_w}x{frame_h}, scale_factor: {scale_factor})")
                return None

            # Now calculate scaling for canvas
            scale = min(W / display_w, H / display_h)
            scaled_w, scaled_h = int(display_w * scale), int(display_h * scale)

            # Calculate centering offsets
            x_offset = (W - scaled_w) // 2
            y_offset = (H - scaled_h) // 2

            # Store transform for overlay calculations - CRITICAL FIX
            if canvas_idx not in self.transform_cache:
                self.transform_cache[canvas_idx] = {}

            # CRITICAL: Store ORIGINAL frame dimensions for correct landmark scaling
            # Landmarks are in original resolution space, NOT downsampled space
            self.transform_cache[canvas_idx]['video_bounds'] = (x_offset, y_offset, scaled_w, scaled_h)
            self.transform_cache[canvas_idx]['frame_size'] = (original_frame_w, original_frame_h)  # ALWAYS use original
            self.transform_cache[canvas_idx]['display_size'] = (display_w, display_h)
            self.transform_cache[canvas_idx]['canvas_size'] = (W, H)
            self.transform_cache[canvas_idx]['scale'] = scale
            self.transform_cache[canvas_idx]['downsample_scale'] = scale_factor

            # Debug logging every 100 frames
            if not hasattr(self, '_res_debug_count'):
                self._res_debug_count = 0
            self._res_debug_count += 1
            if self._res_debug_count % 100 == 0:
                print(f"[CanvasDrawing] Original: {original_frame_w}x{original_frame_h}, Display: {display_w}x{display_h}, Canvas video area: {scaled_w}x{scaled_h}")

            # =====================================================================
            # GPU-ACCELERATED PROCESSING PATH
            # Replaces: resize → color convert → memory copies
            # =====================================================================

            # Process frame on GPU: resize to canvas size + BGR→RGB conversion
            # This replaces multiple CPU operations with single GPU pipeline
            canvas_img_rgb = self.gpu_manager.process_frame_gpu(
                frame_bgr=frame_bgr,
                target_size=(scaled_w, scaled_h),
                convert_to_rgb=True
            )

            if canvas_img_rgb is None:
                logger.error(f"[GPU] Failed to process frame for canvas {canvas_idx}")
                return None

            # Create canvas-sized image with black background
            canvas_img_full = np.zeros((H, W, 3), dtype=np.uint8)

            # Place processed frame in center
            y_end = min(y_offset + scaled_h, H)
            x_end = min(x_offset + scaled_w, W)
            canvas_img_full[y_offset:y_end, x_offset:x_end] = canvas_img_rgb[:y_end-y_offset, :x_end-x_offset]

            # Create PIL image with memory-safe buffer
            pil_img = Image.fromarray(canvas_img_full, mode='RGB')
            photo = ImageTk.PhotoImage(pil_img)

            # CRITICAL FIX: Delete old image to ensure clean update
            # This prevents canvas from keeping stale references
            if hasattr(canvas, '_image_id'):
                try:
                    canvas.delete(canvas._image_id)
                except:
                    pass  # Image might already be deleted

            # Create new image with explicit anchor and tags
            canvas._image_id = canvas.create_image(W // 2, H // 2, image=photo, anchor='center', tags=('background',))
            canvas.tag_lower('background')  # Ensure background is at bottom

            # CRITICAL FIX: Store reference on canvas itself to prevent GC
            # This is more reliable than storing in separate dict
            canvas._photo_ref = photo

            # Also store in our dict for redundancy
            self._photo_images[canvas_idx] = photo

            # Force canvas to update immediately
            canvas.update_idletasks()

            # Debug: Log successful render every 100 frames
            if not hasattr(self, '_render_success_count'):
                self._render_success_count = {}
            if canvas_idx not in self._render_success_count:
                self._render_success_count[canvas_idx] = 0
            self._render_success_count[canvas_idx] += 1
            if self._render_success_count[canvas_idx] % 100 == 0:
                logger.info(f"[CanvasDrawing] Successfully rendered frame {self._render_success_count[canvas_idx]} to canvas {canvas_idx}")

            return photo

        except Exception as e:
            logger.error(f"[CanvasDrawing] Error rendering frame: {e}")
            import traceback
            traceback.print_exc()
            return None

    def convert_bbox_to_canvas(self, bbox, transform_data):
        """Convert bbox from capture frame coordinates to canvas coordinates.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2] in capture frame coordinates
            transform_data: Dictionary containing frame_size, video_bounds, etc.
        
        Returns:
            List of canvas coordinates [x1, y1, x2, y2]
        """
        # Add defensive checks for transform_data
        if not transform_data:
            print(f"[CanvasDrawing] ERROR: transform_data is None or empty in convert_bbox_to_canvas")
            return [0, 0, 100, 100]  # Return default small bbox to avoid crash
            
        # Bboxes are in original capture resolution (frame_size)
        frame_size = transform_data.get('frame_size')
        video_bounds = transform_data.get('video_bounds')
        
        if not frame_size or not video_bounds:
            print(f"[CanvasDrawing] ERROR: Missing frame_size or video_bounds in transform_data")
            return [0, 0, 100, 100]  # Return default small bbox to avoid crash
            
        frame_w, frame_h = frame_size
        x_offset, y_offset, video_w, video_h = video_bounds

        # Debug print to understand the issue
        if not hasattr(self, '_bbox_transform_debug_count'):
            self._bbox_transform_debug_count = 0
        self._bbox_transform_debug_count += 1
        
        if self._bbox_transform_debug_count % 100 == 0:
            print(f"\n[CanvasDrawing] Bbox Transform Debug:")
            print(f"  Frame size (capture res): {frame_w}x{frame_h}")
            print(f"  Display size: {transform_data.get('display_size')}")
            print(f"  Video bounds: offset=({x_offset},{y_offset}), size={video_w}x{video_h}")
            print(f"  Canvas size: {transform_data.get('canvas_size', 'unknown')}")
            print(f"  Original bbox: {bbox}")
            print(f"  Bbox is in capture frame coordinates")

        # Compute scaling from original capture resolution to video display area
        scale_x = video_w / frame_w
        scale_y = video_h / frame_h

        # Transform bbox coordinates to canvas
        x1 = bbox[0] * scale_x + x_offset
        y1 = bbox[1] * scale_y + y_offset
        x2 = bbox[2] * scale_x + x_offset
        y2 = bbox[3] * scale_y + y_offset

        if self._bbox_transform_debug_count % 100 == 0:
            print(f"  Scale: x={scale_x:.3f}, y={scale_y:.3f}")
            print(f"  Transformed bbox: ({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f})")
            print(f"  Should be within video area: x=[{x_offset},{x_offset+video_w}], y=[{y_offset},{y_offset+video_h}]")

        return [x1, y1, x2, y2]


    def draw_faces_optimized(self, canvas: tk.Canvas, faces: List[Dict],
                            canvas_idx: int, labels: Optional[Dict] = None,
                            participant_count: int = 2, participant_names: Dict = None,
                            enrollment_states: Optional[Dict[int, str]] = None) -> None:
        """
        Draw face overlays with proper coordinate transformation.

        FIXED: Simplified coordinate transformation pipeline.
        Landmarks are now consistently in frame space coordinates from landmark process.

        Args:
            canvas: Tkinter canvas to draw on
            faces: List of face dictionaries with landmarks and metadata
            canvas_idx: Canvas index for caching
            labels: Optional custom labels per face ID
            participant_count: Number of participants being tracked
            participant_names: Optional participant names mapping
            enrollment_states: Optional dict mapping participant_id -> enrollment_state string
                             (e.g., {1: 'ENROLLED', 2: 'COLLECTING'})
        """
        if not canvas.winfo_exists():
            return
        
        # Initialize cache for this canvas
        if canvas_idx not in self.canvas_objects:
            self.canvas_objects[canvas_idx] = {
                'face_lines': {},
                'face_labels': {},
                'face_bboxes': {},
                'last_seen': {}
            }
        cache = self.canvas_objects[canvas_idx]
        
        # Ensure all required keys exist (for backward compatibility)
        if 'last_seen' not in cache:
            cache['last_seen'] = {}
        if 'face_lines' not in cache:
            cache['face_lines'] = {}
        if 'face_labels' not in cache:
            cache['face_labels'] = {}
        if 'face_bboxes' not in cache:
            cache['face_bboxes'] = {}
        
        # Get video bounds and frame size for coordinate transformation
        if self.transform_cache is None:
            print(f"[CanvasDrawing] ERROR: transform_cache is None! Reinitializing...")
            self.transform_cache = {}
            
        transform_data = self.transform_cache.get(canvas_idx, {})
        video_bounds = transform_data.get('video_bounds')
        frame_size = transform_data.get('frame_size')
        
        if not video_bounds or not frame_size:
            print(f"[CanvasDrawing] No transform data for canvas {canvas_idx}")
            return
            
        x_offset, y_offset, video_w, video_h = video_bounds
        frame_w, frame_h = frame_size
        
        # Filter valid faces
        valid_faces = []
        for face in faces:
            fid = face.get('id', face.get('global_id', face.get('track_id')))
            
            # Handle temporary local IDs
            if isinstance(fid, str) and fid.startswith('local_'):
                if participant_count == 1:
                    face['id'] = 1
                    valid_faces.append(face)
                continue
            
            # Accept face IDs
            if isinstance(fid, int):
                if fid <= participant_count or fid == 0:
                    valid_faces.append(face)
        
        # Track active faces
        current_time = time.time()
        active_face_ids = set()

        # CRITICAL FIX: Scale calculation must use ORIGINAL frame resolution
        # Landmarks are in original capture resolution (e.g., 1920x1080), NOT downsampled resolution
        # frame_size from transform_cache should be the original resolution
        scale_x = video_w / frame_w
        scale_y = video_h / frame_h

        # Debug logging for first face every 100 frames
        if hasattr(self, '_scale_debug_count'):
            self._scale_debug_count += 1
        else:
            self._scale_debug_count = 0

        if self._scale_debug_count % 100 == 0 and len(valid_faces) > 0:
            print(f"[CanvasDrawing] Landmark scale: video_area={video_w}x{video_h}, frame_size={frame_w}x{frame_h}, scale=({scale_x:.3f}, {scale_y:.3f})")
        
        # Process each face
        for face in valid_faces:
            fid = face.get('id', 1)
            active_face_ids.add(fid)
            cache['last_seen'][fid] = current_time
            
            # Get display label
            if labels and fid in labels:
                display_label = labels[fid]
            elif participant_names and isinstance(fid, int):
                display_label = participant_names.get(fid - 1, f"P{fid}")
            else:
                display_label = f"P{fid}"
            
            # Draw bounding box if available
            if 'bbox' in face:
                bbox = face['bbox']
                canvas_bbox = self.convert_bbox_to_canvas(bbox, transform_data)
                x1, y1, x2, y2 = [int(coord) for coord in canvas_bbox]
                
                # Ensure coordinates stay within canvas bounds
                canvas_w, canvas_h = self.transform_cache[canvas_idx].get('canvas_size', (9999, 9999))
                x1 = max(0, min(x1, canvas_w - 1))
                y1 = max(0, min(y1, canvas_h - 1))
                x2 = max(0, min(x2, canvas_w - 1))
                y2 = max(0, min(y2, canvas_h - 1))
                
                # Create or update bbox rectangle
                bbox_key = f'bbox_{fid}'
                if bbox_key in cache.get('face_bboxes', {}):
                    bbox_id = cache['face_bboxes'][bbox_key]
                    canvas.coords(bbox_id, x1, y1, x2, y2)
                    canvas.itemconfig(bbox_id, state='normal')
                else:
                    bbox_id = canvas.create_rectangle(
                        x1, y1, x2, y2,
                        outline='#00FF00', width=2,
                        tags=('overlay', f'face_bbox_{fid}')
                    )
                    if 'face_bboxes' not in cache:
                        cache['face_bboxes'] = {}
                    cache['face_bboxes'][bbox_key] = bbox_id
                
                # Add participant label above bounding box
                label_key = f'bbox_label_{fid}'
                label_x = (x1 + x2) // 2  # Center of bbox
                label_y = y1 - 5  # Above bbox
                
                # Ensure label stays on screen
                if label_y < 10:
                    label_y = y1 + 15  # Put inside bbox if too high
                
                if label_key in cache.get('face_bboxes', {}):
                    label_id = cache['face_bboxes'][label_key]
                    canvas.coords(label_id, label_x, label_y)
                    canvas.itemconfig(label_id, text=display_label, state='normal')
                else:
                    label_id = canvas.create_text(
                        label_x, label_y,
                        text=display_label,
                        fill='#00FF00',
                        font=('Arial', 12, 'bold'),
                        anchor='s',
                        tags=('overlay', f'face_label_{fid}')
                    )
                    cache['face_bboxes'][label_key] = label_id

            # Draw face landmarks if available
            lm_xyz = face.get('landmarks', [])
            has_landmarks = (lm_xyz is not None and 
                           ((isinstance(lm_xyz, np.ndarray) and lm_xyz.size > 0) or 
                            (isinstance(lm_xyz, list) and len(lm_xyz) > 0)))
            
            if has_landmarks:
                # SIMPLIFIED: Landmarks are already in frame space from landmark process
                # Only need Frame → Canvas transformation
                canvas_landmarks = []
                
                for landmark in lm_xyz:
                    # Handle both 2D (x,y) and 3D (x,y,z) landmarks
                    if len(landmark) >= 3:
                        x, y, z = landmark[0], landmark[1], landmark[2]
                    else:
                        x, y = landmark[0], landmark[1]
                        z = 0
                    
                    # Skip (0,0) landmarks that can cause fan pattern
                    if x == 0 and y == 0:
                        canvas_landmarks.append((0, 0, z))
                        continue
                    
                    # SIMPLIFIED: Direct Frame → Canvas transformation
                    # Landmarks are already in frame pixel coordinates from landmark process
                    canvas_x = x * scale_x + x_offset
                    canvas_y = y * scale_y + y_offset
                    
                    canvas_landmarks.append((canvas_x, canvas_y, z))
                
                # Initialize face lines if needed
                if fid not in cache['face_lines']:
                    cache['face_lines'][fid] = []

                # Determine face outline color based on enrollment state
                # CRITICAL FIX: Use participant_id for enrollment lookup (not fid/track_id!)
                # enrollment_states dict is keyed by participant_id, NOT track_id
                enrollment_state = 'UNKNOWN'  # Default state

                # Extract participant_id from face dict
                participant_id = face.get('participant_id', -1)

                # PRIORITY 1: Check enrollment_states dict using PARTICIPANT_ID
                # This matches how lock button works (uses participant_id as key)
                if enrollment_states and participant_id > 0 and participant_id in enrollment_states:
                    enrollment_state = enrollment_states[participant_id]
                else:
                    # PRIORITY 2: Fallback to display_flags (worker process buffer)
                    display_flags = face.get('display_flags', 0)
                    if display_flags & DISPLAY_FLAG_ENROLLED:
                        enrollment_state = 'ENROLLED'
                    elif display_flags & DISPLAY_FLAG_COLLECTING:
                        enrollment_state = 'COLLECTING'
                    elif display_flags & DISPLAY_FLAG_VALIDATING:
                        enrollment_state = 'VALIDATING'
                    elif display_flags & DISPLAY_FLAG_FAILED:
                        enrollment_state = 'FAILED'

                # DIAGNOSTIC: Track UNKNOWN state duration and alert periodically
                if not hasattr(self, '_unknown_state_tracker'):
                    self._unknown_state_tracker = {}  # {face_id: frame_count}

                if enrollment_state == 'UNKNOWN':
                    # Increment counter for this face in UNKNOWN state
                    self._unknown_state_tracker[fid] = self._unknown_state_tracker.get(fid, 0) + 1

                    # Alert every 90 frames (3 seconds at 30 FPS)
                    if self._unknown_state_tracker[fid] % 90 == 0:
                        print(f"[GUI] 🔴 Face {fid} still UNKNOWN after {self._unknown_state_tracker[fid]} frames (~{self._unknown_state_tracker[fid]//30}s)")
                        print(f"  → Check enrollment manager logs for rejection reasons")
                        print(f"  → Common causes: low quality, insufficient samples, validation failure")
                        print(f"  → Expected: Quality ≥ 0.7, Consistency ≥ 0.85, Stability ≥ 0.8")
                else:
                    # Clear counter when face leaves UNKNOWN state
                    if fid in self._unknown_state_tracker:
                        del self._unknown_state_tracker[fid]

                face_color = self.get_enrollment_color(enrollment_state)

                # Draw or update face contours with enrollment-aware color
                self._update_face_contours(canvas, cache, fid, canvas_landmarks, display_label, [],
                                         color=face_color)
            
            # Draw centroid if available (for debugging)
            if 'centroid' in face and self.debug_mode:
                cx_frame, cy_frame = face['centroid']
                # Centroid is in frame coordinates, transform to canvas
                cx = int(cx_frame * scale_x + x_offset)
                cy = int(cy_frame * scale_y + y_offset)
                
                centroid_key = f'centroid_{fid}'
                if centroid_key in cache.get('face_bboxes', {}):
                    centroid_id = cache['face_bboxes'][centroid_key]
                    canvas.coords(centroid_id, cx-5, cy-5, cx+5, cy+5)
                    canvas.itemconfig(centroid_id, state='normal')
                else:
                    centroid_id = canvas.create_oval(
                        cx-5, cy-5, cx+5, cy+5,
                        fill='red', outline='yellow', width=2,
                        tags=('overlay', f'face_centroid_{fid}')
                    )
                    cache['face_bboxes'][centroid_key] = centroid_id
        
        # Hide inactive faces
        self._hide_inactive_faces(canvas, cache, active_face_ids)
        
        # Ensure overlays are on top
        canvas.tag_raise('overlay')
        
        # Force canvas update to ensure changes are visible
        try:
            canvas.update_idletasks()
        except Exception as e:
            print(f"[CanvasDrawing] [FATAL] Error updating canvas: {e}")
            import traceback
            traceback.print_exc()

    def _update_face_contours(self, canvas: tk.Canvas, cache: Dict, face_id: int,
                             landmarks: List[Tuple], label_text: str, face_contour: List[Tuple] = None,
                             color: str = '#00FF00') -> None:
        """
        Update or create face contour lines with specified color.

        Args:
            canvas: Tkinter canvas to draw on
            cache: Drawing cache for this canvas
            face_id: Unique face identifier
            landmarks: List of (x, y, z) landmark tuples
            label_text: Text label for this face
            face_contour: Optional face contour points
            color: Hex color string for face outline (default: green)
        """
        lines = cache['face_lines'].get(face_id, [])
        
        # Check if we have valid RTMW face connections or need fallback
        connection_list = list(self.face_connections) if self.face_connections else []

        # RTMW3D face: 68 landmarks (not 478 like MediaPipe)
        # Always use contours for better visibility
        use_fallback = self.face_draw_mode == 'points'

        # Debug output
        if not hasattr(self, '_connection_debug_shown'):
            self._connection_debug_shown = True
            print(f"[CanvasDrawing] Using {len(connection_list)} face connections (RTMW3D: 68 points)")
            print(f"[CanvasDrawing] Face draw mode: {self.face_draw_mode}")
            print(f"[CanvasDrawing] Use fallback (points): {use_fallback}")

        if use_fallback:
            # Fallback mode: Draw landmarks as points
            if 'face_points' not in cache:
                cache['face_points'] = {}

            points = cache['face_points'].get(face_id, [])

            # Create points for each landmark (RTMW3D: 68 face landmarks)
            num_points = len(landmarks)

            # Ensure we have enough point objects
            while len(points) < num_points:
                point_id = canvas.create_oval(
                    0, 0, 0, 0,
                    fill=color, outline=color, width=1,
                    tags=('overlay', f'face_{face_id}', 'face_point')
                )
                points.append(point_id)

            cache['face_points'][face_id] = points

            # Update point positions
            for i in range(num_points):
                x, y, _ = landmarks[i]
                size = 2  # Point radius for 68 landmarks
                try:
                    canvas.coords(points[i], int(x-size), int(y-size), int(x+size), int(y+size))
                    canvas.itemconfig(points[i], state='normal', fill=color)
                except:
                    pass

            # Hide excess points
            for i in range(num_points, len(points)):
                canvas.itemconfig(points[i], state='hidden')

            # Hide lines in fallback mode
            for line_id in lines:
                canvas.itemconfig(line_id, state='hidden')
        else:
            # Normal mode: Use RTMW face contour connections
            # Hide points if we were in fallback mode before
            if 'face_points' in cache and face_id in cache['face_points']:
                for point_id in cache['face_points'][face_id]:
                    canvas.itemconfig(point_id, state='hidden')

            # Ensure we have enough line objects for RTMW face connections
            while len(lines) < len(connection_list):
                line_width = 2
                line_id = canvas.create_line(
                    0, 0, 0, 0,
                    fill=color, width=line_width,
                    tags=('overlay', f'face_{face_id}', 'face_line')
                )
                lines.append(line_id)

            cache['face_lines'][face_id] = lines

            # Update line positions using RTMW face contour connections
            for i, (start_idx, end_idx) in enumerate(connection_list):
                if i >= len(lines):
                    break
                    
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    x1, y1, _ = landmarks[start_idx]
                    x2, y2, _ = landmarks[end_idx]
                    
                    # CRITICAL: Skip connections with (0,0) coordinates to prevent fan pattern
                    if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                        # Hide this line if either endpoint is at (0,0)
                        canvas.itemconfig(lines[i], state='hidden')
                        if face_id == 1 and i < 10:
                            print(f"[CONTOUR DEBUG] Skipping zero connection {i}: [{start_idx}]=({x1:.1f},{y1:.1f}) -> [{end_idx}]=({x2:.1f},{y2:.1f})")
                        continue
                    
                    # DEBUG: Print first few connections
                    if face_id == 1 and i < 3:
                        print(f"[CONTOUR DEBUG] Connection {i}: landmark[{start_idx}]=({x1:.1f},{y1:.1f}) -> landmark[{end_idx}]=({x2:.1f},{y2:.1f})")
                    
                    try:
                        canvas.coords(lines[i], int(x1), int(y1), int(x2), int(y2))
                        canvas.itemconfig(lines[i], state='normal', fill=color, width=2)
                    except:
                        pass
                else:
                    canvas.itemconfig(lines[i], state='hidden')
            
            # Hide excess lines
            for i in range(len(connection_list), len(lines)):
                canvas.itemconfig(lines[i], state='hidden')
    
    def _hide_inactive_faces(self, canvas: tk.Canvas, cache: Dict, active_ids: set) -> None:
        """Hide faces that are no longer active."""
        current_time = time.time()
        
        # Get canvas config for timeout settings
        canvas_config = self.config.get('gui_interface', {}).get('canvas', {})
        
        # Check all tracked faces
        for face_id in list(cache.get('last_seen', {}).keys()):
            if face_id not in active_ids:
                # Hide after timeout
                last_seen = cache['last_seen'].get(face_id, 0)
                time_since_seen = current_time - last_seen
                frame_timeout = canvas_config.get('coordinate_transform', {}).get('frame_timeout', 0.5)
                if time_since_seen > frame_timeout:  # Configurable timeout
                    print(f"[CANVAS HIDE] Hiding face {face_id} - not in active_ids={active_ids}, "
                          f"time_since_seen={time_since_seen:.2f}s")
                    # Hide face lines
                    for line_id in cache.get('face_lines', {}).get(face_id, []):
                        try:
                            canvas.itemconfig(line_id, state='hidden')
                        except:
                            pass
                    
                    # Hide face bboxes, labels, and debug overlays
                    for key in [f'bbox_{face_id}', f'bbox_label_{face_id}', f'centroid_{face_id}']:
                        if key in cache.get('face_bboxes', {}):
                            try:
                                canvas.itemconfig(cache['face_bboxes'][key], state='hidden')
                            except:
                                pass
    
    def draw_poses_optimized(self, canvas: tk.Canvas, poses: List[Dict],
                            canvas_idx: int, enabled: bool = True) -> None:
        """
        Draw pose overlays with proper coordinate transformation.

        RTMPose outputs NORMALIZED coordinates (0-1 range).
        This function transforms them to canvas coordinates.
        """
        if not enabled or not poses:
            # Hide all pose overlays
            if canvas_idx in self.canvas_objects:
                self._hide_all_poses(canvas, canvas_idx)
            return

        # Initialize cache
        if canvas_idx not in self.canvas_objects:
            self.canvas_objects[canvas_idx] = {}
        cache = self.canvas_objects[canvas_idx]

        # Get transform data
        transform_data = self.transform_cache.get(canvas_idx, {})
        video_bounds = transform_data.get('video_bounds')
        frame_size = transform_data.get('frame_size')

        if not video_bounds or not frame_size:
            if not hasattr(self, '_pose_transform_warning_shown'):
                print(f"[CanvasDrawing] WARNING: No transform data for pose overlay on canvas {canvas_idx}")
                self._pose_transform_warning_shown = True
            return

        x_offset, y_offset, video_w, video_h = video_bounds
        frame_w, frame_h = frame_size

        # Debug logging for first pose every 100 frames
        if not hasattr(self, '_pose_draw_count'):
            self._pose_draw_count = 0
        self._pose_draw_count += 1

        if self._pose_draw_count % 100 == 0 and len(poses) > 0:
            print(f"[CanvasDrawing] Drawing {len(poses)} pose(s) on canvas {canvas_idx}")
            print(f"  Transform: frame={frame_w}x{frame_h} → video_area={video_w}x{video_h} @ offset=({x_offset},{y_offset})")

        # Process each pose
        for pose_idx, pose in enumerate(poses):
            if not pose or 'landmarks' not in pose:
                continue

            # FIX: Use participant_id for stable cache key (prevents cross-participant line mixing)
            # When detection order changes, participant_id stays consistent
            participant_id = pose.get('participant_id', pose_idx)  # Fallback to index if no ID
            pose_key = f'pose_lines_pid_{participant_id}'
            if pose_key not in cache:
                cache[pose_key] = []

            # CRITICAL: Get pose resolution (may differ from capture resolution)
            # Pose landmarks are normalized (0-1), need pose resolution to convert to pixels
            # Then scale up to capture resolution to align with face landmarks
            pose_frame_w, pose_frame_h = pose.get('pose_resolution', (frame_w, frame_h))

            # Pre-calculate combined scale: pose pixels → canvas pixels
            # Transforms: normalized → pose_px → capture_px → canvas_px
            # Simplifies to: normalized → pose_px → canvas_px
            pose_to_canvas_scale_x = video_w / pose_frame_w
            pose_to_canvas_scale_y = video_h / pose_frame_h

            # Debug logging when resolution changes
            if not hasattr(self, '_last_pose_resolution'):
                self._last_pose_resolution = {}
            if pose_idx not in self._last_pose_resolution or self._last_pose_resolution[pose_idx] != (pose_frame_w, pose_frame_h):
                logger.info(f"[Canvas] Pose {pose_idx} resolution: {pose_frame_w}×{pose_frame_h} (capture: {frame_w}×{frame_h})")
                logger.info(f"[Canvas]   Pose→Canvas scale: x={pose_to_canvas_scale_x:.3f}, y={pose_to_canvas_scale_y:.3f}")
                self._last_pose_resolution[pose_idx] = (pose_frame_w, pose_frame_h)

            # Transform normalized coordinates (0-1) to canvas pixel coordinates
            # keypoints is numpy array (133, 4) with [x, y, z, confidence]
            pose_coords = []
            for x_norm, y_norm, z, _ in pose['keypoints']:  # Unpack all 4, ignore confidence
                # CORRECTED TRANSFORMATION:
                # Step 1: Normalized (0-1) → Pose Pixels (use pose resolution, not capture resolution!)
                x_pose_px = x_norm * pose_frame_w
                y_pose_px = y_norm * pose_frame_h

                # Step 2: Pose Pixels → Canvas Pixels (combined scale)
                canvas_x = x_pose_px * pose_to_canvas_scale_x + x_offset
                canvas_y = y_pose_px * pose_to_canvas_scale_y + y_offset

                pose_coords.append((canvas_x, canvas_y, z))

            # Debug first pose
            if self._pose_draw_count == 1 and pose_idx == 0:
                print(f"[CanvasDrawing] First pose landmark sample:")
                print(f"  Landmark 0 (nose): normalized=({pose['keypoints'][0, 0]:.3f},{pose['keypoints'][0, 1]:.3f}) → canvas=({pose_coords[0][0]:.1f},{pose_coords[0][1]:.1f})")

            # Draw or update pose
            self._draw_connections_cached(canvas, pose_coords, self.pose_connections,
                                        cache[pose_key], '#40FFFF', 3)  # Cyan, width 3

        # Periodic debug logging (every 50 frames)
        if not hasattr(self, '_pose_draw_frame_count'):
            self._pose_draw_frame_count = {}

        if canvas_idx not in self._pose_draw_frame_count:
            self._pose_draw_frame_count[canvas_idx] = 0

        self._pose_draw_frame_count[canvas_idx] += 1

        if self._pose_draw_frame_count[canvas_idx] % 50 == 0:
            logger.info(f"[Canvas] Drawing canvas {canvas_idx}: {len(poses) if poses else 0} pose(s), enabled={enabled}")

        # Hide unused pose overlays
        self._cleanup_unused_poses(canvas, cache, poses)

        # Ensure overlays are on top
        canvas.tag_raise('overlay')
    
    def draw_debug_detections(self, canvas: tk.Canvas, debug_data: Dict, 
                            canvas_idx: int) -> None:
        """Draw raw RetinaFace detections for debugging."""
        if not debug_data or not self.debug_mode:
            return
        
        # Get transform data
        transform_data = self.transform_cache.get(canvas_idx, {})
        if not transform_data.get('video_bounds') or not transform_data.get('frame_size'):
            return
        
        # Get raw detections
        raw_detections = debug_data.get('raw_detections', [])
        if not raw_detections:
            return
        
        # Clear previous debug overlays
        try:
            canvas.delete('debug_detection')
        except:
            pass
        
        # Draw each raw detection
        for i, det in enumerate(raw_detections):
            bbox = det.get('bbox', [])
            if len(bbox) != 4:
                continue
                
            confidence = det.get('confidence', 0)
            
            # Use the convert_bbox_to_canvas method
            canvas_bbox = self.convert_bbox_to_canvas(bbox, transform_data)
            x1, y1, x2, y2 = [int(coord) for coord in canvas_bbox]
            
            # Ensure coordinates stay within canvas bounds
            canvas_w, canvas_h = self.transform_cache.get(canvas_idx, {}).get('canvas_size', (9999, 9999))
            x1 = max(0, min(x1, canvas_w - 1))
            y1 = max(0, min(y1, canvas_h - 1))
            x2 = max(0, min(x2, canvas_w - 1))
            y2 = max(0, min(y2, canvas_h - 1))
            
            # Draw rectangle
            color = '#FF0000' if confidence > 0.98 else '#FFFF00'
            canvas.create_rectangle(x1, y1, x2, y2, 
                                outline=color, width=2, 
                                tags=('debug_detection', 'overlay'))
            
            # Draw confidence text
            text = f"{confidence:.3f}"
            label_y = max(5, y1 - 5)  # Ensure label stays on screen
            canvas.create_text(x1, label_y, text=text, 
                            fill=color, anchor='sw',
                            font=('Arial', 10, 'bold'),
                            tags=('debug_detection', 'overlay'))
        
        # Ensure debug overlays are on top
        canvas.tag_raise('overlay')
    
    def _draw_connections_cached(self, canvas: tk.Canvas, landmarks: List[Tuple],
                               connections: List[Tuple], line_cache: List,
                               color: str, width: int) -> None:
        """Draw connections with object pooling."""
        connection_list = list(connections)
        
        # Ensure we have enough lines
        while len(line_cache) < len(connection_list):
            line_id = canvas.create_line(0, 0, 0, 0, fill=color, width=width, 
                                       tags=('overlay',))
            line_cache.append(line_id)
        
        # Update lines
        for i, (start_idx, end_idx) in enumerate(connection_list):
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                x1, y1, _ = landmarks[start_idx]
                x2, y2, _ = landmarks[end_idx]
                
                canvas.coords(line_cache[i], int(x1), int(y1), int(x2), int(y2))
                canvas.itemconfig(line_cache[i], state='normal')
            else:
                canvas.itemconfig(line_cache[i], state='hidden')
        
        # Hide excess lines
        for i in range(len(connection_list), len(line_cache)):
            canvas.itemconfig(line_cache[i], state='hidden')
    
    def _hide_all_poses(self, canvas: tk.Canvas, canvas_idx: int) -> None:
        """Hide all pose overlays for a canvas."""
        cache = self.canvas_objects.get(canvas_idx, {})
        for key in list(cache.keys()):
            if key.startswith('pose_lines'):
                for line_id in cache[key]:
                    try:
                        canvas.itemconfig(line_id, state='hidden')
                    except:
                        pass
    
    def _cleanup_unused_poses(self, canvas: tk.Canvas, cache: Dict, active_poses: List[Dict]) -> None:
        """Hide pose overlays for inactive participants."""
        # Extract active participant IDs from current poses
        active_participant_ids = set()
        for pose in (active_poses or []):
            participant_id = pose.get('participant_id')
            if participant_id is not None:
                active_participant_ids.add(participant_id)

        # Hide cache entries for inactive participant IDs
        for key in list(cache.keys()):
            if key.startswith('pose_lines_pid_'):
                # Extract participant_id from key: "pose_lines_pid_1" -> 1
                try:
                    parts = key.split('_')
                    if len(parts) >= 4:
                        participant_id_str = parts[3]
                        # Handle both int and None participant IDs
                        if participant_id_str == 'None':
                            participant_id = None
                        else:
                            participant_id = int(participant_id_str)

                        # Hide if not in active set
                        if participant_id not in active_participant_ids:
                            for line_id in cache[key]:
                                try:
                                    canvas.itemconfig(line_id, state='hidden')
                                except:
                                    pass
                except (ValueError, IndexError):
                    # If parsing fails, skip this key
                    pass
            elif key.startswith('pose_lines_'):
                # Legacy cleanup for old-style keys (pose_lines_0, pose_lines_1)
                # These can be safely hidden as they're deprecated
                for line_id in cache[key]:
                    try:
                        canvas.itemconfig(line_id, state='hidden')
                    except:
                        pass
    
    def cleanup_canvas(self, canvas_idx: int) -> None:
        """Clean up all cached objects for a canvas."""
        if canvas_idx in self.canvas_objects:
            del self.canvas_objects[canvas_idx]
        if canvas_idx in self.transform_cache:
            del self.transform_cache[canvas_idx]
        if canvas_idx in self.last_frame_time:
            del self.last_frame_time[canvas_idx]
        if canvas_idx in self.last_face_state:
            del self.last_face_state[canvas_idx]
        if canvas_idx in self.last_face_ids:
            del self.last_face_ids[canvas_idx]
        if canvas_idx in self._photo_images:
            del self._photo_images[canvas_idx]

    def print_gpu_performance_stats(self):
        """Print GPU performance statistics."""
        if hasattr(self, 'gpu_manager') and self.gpu_manager:
            self.gpu_manager.print_performance_report()

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU performance statistics as dictionary."""
        if hasattr(self, 'gpu_manager') and self.gpu_manager:
            return self.gpu_manager.get_performance_stats()
        return {}

    def cleanup_gpu_resources(self):
        """Clean up GPU resources on shutdown."""
        if hasattr(self, 'gpu_manager') and self.gpu_manager:
            self.gpu_manager.cleanup()
            logger.info("[CanvasDrawing] GPU resources cleaned up")
    
    def get_stats(self, canvas_idx: int) -> Dict[str, Any]:
        """Get performance statistics for a canvas."""
        stats = {
            'cached_faces': 0,
            'cached_lines': 0,
            'last_update': 0,
            'transform_data': None
        }
        
        if canvas_idx in self.canvas_objects:
            cache = self.canvas_objects[canvas_idx]
            stats['cached_faces'] = len(cache.get('face_lines', {}))
            stats['cached_lines'] = sum(len(lines) for lines in cache.get('face_lines', {}).values())
        
        if canvas_idx in self.last_frame_time:
            stats['last_update'] = time.time() - self.last_frame_time[canvas_idx]
        
        if canvas_idx in self.transform_cache:
            stats['transform_data'] = self.transform_cache[canvas_idx]
        
        return stats
    
    def set_debug_mode(self, enabled: bool) -> None:
        """Toggle debug mode for all canvases."""
        self.debug_mode = enabled
        print(f"[CanvasDrawing] Debug mode set to: {enabled}")
    
    def set_face_draw_mode(self, mode: str) -> None:
        """Set face drawing mode: 'full_contours', 'jaw_only', 'mesh', 'points'."""
        if mode in ['full_contours', 'jaw_only', 'mesh', 'points']:
            self.face_draw_mode = mode
            print(f"[CanvasDrawing] Face draw mode set to: {mode}")
        else:
            print(f"[CanvasDrawing] Invalid face draw mode: {mode}. Valid modes: full_contours, jaw_only, mesh, points")


# Standalone utility function with enrollment state support
def draw_overlays_combined(frame_bgr: np.ndarray, faces: List[Dict] = None,
                          pose_landmarks: List = None, labels: Dict = None,
                          face_mesh: bool = True, face_contours: bool = True,
                          face_points: bool = True, pose_lines: bool = True,
                          enrollment_states: Optional[Dict[int, str]] = None,
                          all_poses: List[Dict] = None, config: Dict = None) -> np.ndarray:
    """
    Draw face and pose overlays on a frame.
    Used for video recording with overlays.

    Args:
        frame_bgr: Input frame in BGR format
        faces: List of face dictionaries with landmarks and IDs
        pose_landmarks: Optional pose landmarks (DEPRECATED - use all_poses)
        labels: Optional text labels per face ID
        face_mesh: Draw full face mesh (not recommended for performance)
        face_contours: Draw face contours
        face_points: Draw face landmark points
        pose_lines: Draw pose skeleton lines
        enrollment_states: Optional dict mapping face_id -> enrollment_state string
                          (e.g., {1: 'ENROLLED', 2: 'COLLECTING'})
        all_poses: List of pose dictionaries with 'landmarks' key (preferred over pose_landmarks)
        config: Optional config dict for opacity settings

    Returns:
        Frame with overlays drawn
    """
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]

    # Draw face overlays
    if faces:
        for idx, face in enumerate(faces):
            landmarks = face.get("landmarks", [])
            # FIX: Use participant_id for enrollment state lookup (not track_id)
            # enrollment_states is keyed by participant_id, not track_id
            participant_id = face.get("participant_id", -1)
            fid = participant_id if participant_id > 0 else face.get("id", idx+1)

            if landmarks is not None and len(landmarks) > 0:
                # Check if landmarks are normalized or pixel coordinates
                if landmarks[0][0] <= 1.0 and landmarks[0][1] <= 1.0:
                    # Already normalized coordinates
                    normalized_landmarks = landmarks
                else:
                    # Convert pixel coordinates to normalized
                    normalized_landmarks = [(x / w, y / h, z) for x, y, z in landmarks]

                # VALIDATION: Check for NaN/Inf BEFORE drawing
                try:
                    # Validate landmarks contain no NaN or Inf values
                    for i, (x, y, z) in enumerate(normalized_landmarks):
                        if np.isnan(x) or np.isnan(y) or np.isnan(z):
                            raise ValueError(f"[FATAL] NaN detected in normalized landmark {i}: ({x}, {y}, {z})")
                        if np.isinf(x) or np.isinf(y) or np.isinf(z):
                            raise ValueError(f"[FATAL] Inf detected in normalized landmark {i}: ({x}, {y}, {z})")

                    # Determine face contour color based on enrollment state
                    # CRITICAL FIX: Use participant_id for enrollment lookup (not fid/track_id!)
                    # enrollment_states dict is keyed by participant_id, NOT track_id
                    enrollment_state = 'UNKNOWN'  # Default

                    # Extract participant_id from face dict
                    participant_id_for_enrollment = face.get("participant_id", -1)

                    # PRIORITY 1: Check enrollment_states dict using PARTICIPANT_ID
                    # This matches how lock button works (uses participant_id as key)
                    if enrollment_states and participant_id_for_enrollment > 0 and participant_id_for_enrollment in enrollment_states:
                        enrollment_state = enrollment_states[participant_id_for_enrollment]
                    else:
                        # PRIORITY 2: Fallback to display_flags (worker process buffer)
                        display_flags = face.get('display_flags', 0)
                        if display_flags & DISPLAY_FLAG_ENROLLED:
                            enrollment_state = 'ENROLLED'
                        elif display_flags & DISPLAY_FLAG_COLLECTING:
                            enrollment_state = 'COLLECTING'
                        elif display_flags & DISPLAY_FLAG_VALIDATING:
                            enrollment_state = 'VALIDATING'
                        elif display_flags & DISPLAY_FLAG_FAILED:
                            enrollment_state = 'FAILED'

                    # Get color from enrollment state
                    hex_color = CanvasDrawingManager.get_enrollment_color(enrollment_state)
                    # Convert hex to BGR tuple (OpenCV uses BGR)
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                    bgr_color = (b, g, r)

                    # OPTIMIZED: Use batched cv2.polylines for RTMW face contours (68 keypoints)
                    # RTMPose face landmarks are 68 points, not 478 like MediaPipe
                    # Batched drawing is 10-20x faster than drawing each line individually

                    # Collect all line segments for face contours
                    contour_lines = []
                    for start_idx, end_idx in FACE_CONTOUR_CONNECTIONS:
                        if start_idx < len(landmarks) and end_idx < len(landmarks):
                            # Landmarks are in pixel coordinates
                            pt1 = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                            pt2 = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                            contour_lines.append([pt1, pt2])

                    # Draw all contour lines in a single batched call
                    if contour_lines:
                        contour_lines_np = np.array(contour_lines, dtype=np.int32)
                        cv2.polylines(frame, contour_lines_np, isClosed=False, color=bgr_color, thickness=2, lineType=cv2.LINE_AA)

                except Exception as e:
                    print(f"[CanvasDrawing] [FATAL] Face drawing crashed for face {fid}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Re-raise to fail fast and reveal the issue
                    raise
            
            # Draw bounding box if available
            if 'bbox' in face:
                bbox = face['bbox']
                # Bbox coordinates are in pixel coordinates for the frame
                x1, y1, x2, y2 = [int(v) for v in bbox]

                # DEBUG: Log bbox drawing every 30 faces
                if not hasattr(draw_overlays_combined, '_bbox_draw_count'):
                    draw_overlays_combined._bbox_draw_count = 0
                draw_overlays_combined._bbox_draw_count += 1
                if draw_overlays_combined._bbox_draw_count % 30 == 0:
                    logger.debug(f"[BBOX DRAW] Face {fid}: Drawing bbox at ({x1},{y1})-({x2},{y2}) on frame {w}x{h}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # DEBUG: Log when bbox is missing
                if not hasattr(draw_overlays_combined, '_bbox_missing_count'):
                    draw_overlays_combined._bbox_missing_count = 0
                draw_overlays_combined._bbox_missing_count += 1
                if draw_overlays_combined._bbox_missing_count % 30 == 0:
                    logger.warning(f"[BBOX MISSING] Face {fid}: 'bbox' key not in face dict, available keys: {list(face.keys())}")
            
            # Draw label
            label_text = None
            if labels and fid in labels:
                label_text = labels[fid]
            elif not labels:
                label_text = f"Face {fid}"
            
            if label_text:
                if "centroid" in face and face["centroid"] is not None:
                    # Check if centroid is normalized or pixel coordinates
                    cx_val, cy_val = face["centroid"]
                    if cx_val <= 1.0 and cy_val <= 1.0:
                        # Normalized coordinates
                        cx, cy = int(cx_val * w), int(cy_val * h) - 50
                    else:
                        # Already in pixel coordinates
                        cx, cy = int(cx_val), int(cy_val) - 50
                elif landmarks is not None and len(landmarks) > 10:
                    cx, cy = int(landmarks[10][0]), int(landmarks[10][1]) - 50
                elif 'bbox' in face:
                    bbox = face['bbox']
                    # Bbox is in pixel coordinates
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int(bbox[1] - 10)
                else:
                    cx, cy = w // 2, 50

                # Use PIL for Unicode support (enrollment symbols: ●, ◴, ◵, ○, ✗)
                # Yellow text (255, 255, 0) in RGB, black outline for visibility
                frame = _draw_pil_text_with_outline(frame, label_text, (cx, cy),
                                                    font_size=24, color=(255, 255, 0),
                                                    outline_color=(0, 0, 0))

    # Draw pose overlays with transparency (supports multiple poses)
    if pose_lines:
        # Get opacity from config (default 0.4 = 40%)
        opacity = 0.4
        if config:
            opacity = config.get('gui_interface', {}).get('canvas', {}).get('pose_skeleton_opacity', 0.4)

        # Support both new format (all_poses) and legacy format (pose_landmarks)
        poses_to_draw = []
        if all_poses:
            poses_to_draw = all_poses
        elif pose_landmarks is not None and len(pose_landmarks) > 0:
            # Legacy format: single pose as flat list
            poses_to_draw = [{'keypoints': pose_landmarks}]

        if poses_to_draw:
            # Create overlay for transparent skeleton
            overlay = frame.copy()

            for pose_idx, pose in enumerate(poses_to_draw):
                if pose and 'keypoints' in pose:
                    # keypoints is numpy (133, 4) - unpack all 4, ignore confidence
                    pose_landmarks_px = [(int(x * w), int(y * h), z) for x, y, z, _ in pose['keypoints']]

                    # DIAGNOSTIC: Log shoulder line coordinates to detect cross-participant mixing
                    if len(pose_landmarks_px) >= 13:
                        participant_id = pose.get('participant_id', 'unknown')
                        nose = pose_landmarks_px[0][:2]
                        l_shoulder = pose_landmarks_px[11][:2]
                        r_shoulder = pose_landmarks_px[12][:2]

                    # Draw all connections for this pose (RTMW body: 17 COCO keypoints)
                    for conn in BODY_CONNECTIONS:
                        i, j = conn
                        if i < len(pose_landmarks_px) and j < len(pose_landmarks_px):
                            cv2.line(overlay, pose_landmarks_px[i][:2], pose_landmarks_px[j][:2],
                                    (64,255,255), 3, cv2.LINE_AA)

            # Blend overlay with original frame using configured opacity
            cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0, frame)

    return frame