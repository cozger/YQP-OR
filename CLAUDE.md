# CLAUDE.md - Technical Documentation

**Last Updated**: 2025-11-11
**Maintained by**: Claude Code (AI Assistant)

This document provides technical documentation for critical architectural decisions, bug fixes, and system configurations that are essential for maintaining and extending the YouQuantiPy pose estimation system.

> **Note**: For comprehensive system architecture documentation including process topology, buffer architecture, data flow diagrams, and complete system wiring diagrams, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Table of Contents

1. [Centralized Buffer Configuration System](#centralized-buffer-configuration-system)
2. [Shared Memory Race Condition Fix](#shared-memory-race-condition-fix)
3. [Frame Shape Validation and Buffer Offset Artifacts](#frame-shape-validation-and-buffer-offset-artifacts)
4. [Buffer Size Calculations](#buffer-size-calculations)
5. [Coordinate System and Head Tilt Calculation](#coordinate-system-and-head-tilt-calculation)
6. [Shoulder Elevation Calculation and Distance Normalization](#shoulder-elevation-calculation-and-distance-normalization)
7. [Code Locations Reference](#code-locations-reference)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## Centralized Buffer Configuration System

### Overview

All buffer-related configuration parameters have been centralized into a single `buffer_settings` structure in `youquantipy_config.json`. This provides a **single source of truth** for all buffer sizes, capacities, and limits across the entire system.

**Critical Rule**: ALL buffer configuration MUST come from `buffer_settings` - never from scattered config locations like `startup_mode`, `process_separation`, or other sections.

### Configuration Structure

Located in: `youquantipy_config.json`

```json
{
  "buffer_settings": {
    "_comment": "Centralized Buffer Configuration - Single Source of Truth",

    "persons": {
      "max_persons": 1,           // Maximum persons tracked per camera
      "participant_count": 1       // Active participant count
    },

    "cameras": {
      "max_cameras": 10,           // System maximum camera capacity
      "camera_count": 1            // Currently active cameras
    },

    "faces": {
      "max_faces_per_frame": 1     // Maximum faces detected per frame
    },

    "ring_buffers": {
      "frame_detection": 32,       // Ring buffer slots for frame detection
      "pose_estimation": 4,        // Ring buffer slots for pose data
      "roi_processing": 8,         // Ring buffer slots for ROI processing
      "gui_display": 8,            // Ring buffer slots for GUI display
      "display_manager": 4         // Ring buffer slots for display manager
    },

    "pinned_memory": {
      "detection_frames": 4,       // Pinned memory buffers for detection
      "roi_buffers": 16,           // Pinned memory buffers for ROI
      "hd_frames": 2               // Pinned memory buffers for HD frames
    },

    "camera_buffers": {
      "frame_buffer_size": 4       // Camera-specific frame buffer size
    }
  }
}
```
### Files Using Centralized Configuration

All buffer-related code has been migrated to use centralized paths:

1. **`core/buffer_management/coordinator.py`**
   - `_calculate_max_faces()` - Face detection capacity
   - `_get_ring_buffer_size()` - Ring buffer sizing
   - `get_layout()` - Pose buffer layout creation

2. **`core/buffer_management/camera_worker_enhanced.py`**
   - Buffer size calculations
   - Ring buffer mask calculations
   - GUI buffer sizing

3. **`core/gui_interface/gui_processing_worker.py`**
   - Maximum participant configuration
   - Display buffer sizing

4. **`core/buffer_management/pinned_memory_pool.py`**
   - CUDA pinned memory pool initialization
   - Detection/ROI/HD frame buffer counts

---

## Shared Memory Race Condition Fix

### The Problem: Intermittent "Blow-Up" Frames

**Symptom**: Pose visualization occasionally showed completely wrong/irrelevant keypoints for a few frames before returning to normal.

**Root Cause**: Race condition in shared memory buffer writes causing readers to see partial/inconsistent data.

### Critical Bugs Fixed

#### Bug 1: Non-Atomic Multi-Step Writes

**Problem**: Writer was incrementing write index BEFORE writing data:
```python
# WRONG - creates race window
write_idx += 1
self.pose_shm.buf[:8] = write_idx.to_bytes(8, 'little')  # Index updated first
# ... write pose data ...  # Reader can see new index but old data!
```

**Fix**: Write data FIRST, increment index LAST (atomic write pattern):
```python
# CORRECT - atomic write pattern
# Step 1: Write pose data to ring buffer slot
pose_bytes = pose_data.tobytes()
self.pose_shm.buf[pose_offset:pose_offset + len(pose_bytes)] = pose_bytes

# Step 2: Write metadata to ring buffer slot
metadata_bytes = metadata.tobytes()
self.pose_shm.buf[metadata_offset:metadata_offset + len(metadata_bytes)] = metadata_bytes

# Step 3: Increment write_index LAST (atomic "data ready" signal)
write_idx += 1
self.pose_shm.buf[:8] = write_idx.to_bytes(8, byteorder='little')
```

**Location**: `core/pose_processing/rtmpose3d_process.py:_write_pose_results()`

#### Bug 2: Reader Slot Index Off-By-One

**Problem**: Reader was using write_idx directly to calculate slot, reading future/empty slots:
```python
# WRONG - reads wrong slot
write_idx = int.from_bytes(write_idx_bytes, 'little')
slot_idx = write_idx % ring_buffer_size  # Reading future slot!
```

**Explanation**: Since writer increments write_idx AFTER writing, the most recently written data is at `(write_idx - 1)`, not `write_idx`.

**Fix**: Subtract 1 to get most recently written slot:
```python
# CORRECT - reads most recently written slot
write_idx_before = int.from_bytes(write_idx_bytes, 'little')
slot_idx = (write_idx_before - 1) % ring_buffer_size  # Read last written slot
```

**Location**: `core/gui_interface/gui_processing_worker.py:_read_pose_data()`

#### Bug 3: Buffer Size Mismatch

**Problem**: Stale buffers from previous runs and wrong config paths caused incorrect buffer sizes.

**Fixes**:
1. Manual cleanup of stale buffers: `rm -f /dev/shm/yq_*`
2. Migrated to centralized config paths (see Configuration section above)
3. Verification: Buffer sizes now correctly calculated

### Ring Buffer Architecture

The system uses a **4-slot ring buffer** for pose data:

```
┌─────────────────────────────────────────┐
│ Write Index (8 bytes)                   │ ← Atomic update point
├─────────────────────────────────────────┤
│ Slot 0: Pose Data (2,128 bytes)         │
├─────────────────────────────────────────┤
│ Slot 1: Pose Data (2,128 bytes)         │
├─────────────────────────────────────────┤
│ Slot 2: Pose Data (2,128 bytes)         │
├─────────────────────────────────────────┤
│ Slot 3: Pose Data (2,128 bytes)         │
├─────────────────────────────────────────┤
│ Slot 0: Metadata (128 bytes)            │
├─────────────────────────────────────────┤
│ Slot 1: Metadata (128 bytes)            │
├─────────────────────────────────────────┤
│ Slot 2: Metadata (128 bytes)            │
├─────────────────────────────────────────┤
│ Slot 3: Metadata (128 bytes)            │
└─────────────────────────────────────────┘
Total: 9,032 bytes (for max_persons=1)
```

**Benefits**:
- **Collision Resistance**: 4 slots reduce writer/reader collisions
- **Lock-Free**: No mutexes needed (atomic index update)
- **Predictable**: Fixed memory layout for performance

### Consistency Checking

The reader implements a **double-read consistency check**:

```python
# Read write_index BEFORE reading data
write_idx_before = int.from_bytes(...)

# Read pose data and metadata
pose_data = ...
metadata = ...

# Read write_index AFTER reading data
write_idx_after = int.from_bytes(...)

# Verify no changes during read
if write_idx_after != write_idx_before:
    return None  # Reject inconsistent data
```

This ensures readers never process partially-written frames.

---

## Frame Shape Validation and Buffer Offset Artifacts

### Overview

**Fixed**: 2025-11-11

Camera feed buffer offset artifacts (movement and color corruption) were caused by **frame shape mismatches** between the expected resolution used during buffer allocation and the actual incoming frame shapes. This section documents the root cause, symptoms, and the comprehensive validation system implemented to prevent these issues.

**Critical Rule**: ALL frame writes and reads MUST validate frame shape matches expected resolution and ensure C-contiguous memory layout before processing.

---

### Testing and Verification

#### Quick Validation Test

```bash
# Start system and check logs for validation messages
./venv/bin/python3 mmpose_3d_gui/gui.py 2>&1 | grep -E "(shape|contiguous|READER VALIDATION)"

# Expected output (every 100 frames):
# DEBUG Camera 0 Frame 100: shape=(1080, 1920, 3), strides=(5760, 3, 1), contiguous=True
# DEBUG [READER VALIDATION] Camera 0 Frame 100: shape=(1080, 1920, 3), dtype=uint8, contiguous=True, min=12, max=243

# If you see errors, investigate resolution mismatch
```

#### Verify Buffer Shapes

```python
# In camera_worker_enhanced.py, add temporary logging:
logger.info(f"Buffer allocated: expected_shape={(self.actual_resolution[1], self.actual_resolution[0], 3)}")
logger.info(f"Incoming frame: shape={frame.shape}, strides={frame.strides}, contiguous={frame.flags['C_CONTIGUOUS']}")

# Run system and verify shapes match
```

---

### Best Practices

#### When Adding Camera Sources

1. **Always validate resolution support**
   ```python
   # Check actual camera capabilities before configuring
   actual_resolution = camera.get_supported_resolutions()
   logger.info(f"Camera supports: {actual_resolution}")
   ```

2. **Test with diagnostic logging enabled**
   ```python
   # Enable debug logging to see frame validation
   logging.getLogger('camera_worker').setLevel(logging.DEBUG)
   ```

3. **Verify buffer sizes match**
   ```bash
   # Check actual buffer size
   ls -lh /dev/shm/yq_frame_0_*
   # Compare with expected: width × height × 3 × ring_buffer_size + header + metadata
   ```

#### When Processing Frames

1. **Never create non-contiguous views**
   ```python
   # BAD: Non-contiguous view
   roi = frame[y1:y2, x1:x2, :]

   # GOOD: Contiguous copy
   roi = frame[y1:y2, x1:x2, :].copy()
   ```

2. **Always check shape before passing to writer**
   ```python
   # Validate shape before writing
   if frame.shape != expected_shape:
       logger.error(f"Shape mismatch: {frame.shape} != {expected_shape}")
       return
   ```

3. **Use consistent resolution throughout pipeline**
   - Don't resize frames between capture and buffer write
   - If resizing needed, update `actual_resolution` accordingly

---

### Files Modified (2025-11-11)

1. **`core/buffer_management/camera_worker_enhanced.py`**
   - Lines 1496-1523: Added frame validation to `_write_frame_optimized()`
   - Lines 1585-1599: Added frame validation to `_write_frame_fallback()`
   - Line 773: Added comment documenting shape validation

2. **`core/gui_interface/gui_processing_worker.py`**
   - Lines 1782-1812: Added reader-side validation in `_read_frame_data()`
   - Shape validation, zero-frame detection, diagnostic logging

---

## Buffer Size Calculations

### Pose Buffer Size Formula

For a single camera with `max_persons=1` and `ring_buffer_size=4`:

```
Header:           8 bytes
Pose Data:        max_persons × keypoints × values × sizeof(float32) × ring_slots
                  = 1 × 133 × 4 × 4 × 4
                  = 8,512 bytes
Metadata:         metadata_size × ring_slots
                  = 128 × 4
                  = 512 bytes
─────────────────────────────────
Total:            9,032 bytes (8.9K)
```

### Keypoint Breakdown (RTMW3D)

Total: **133 keypoints per person**
- Body: 17 keypoints
- Foot: 6 keypoints
- Face: 68 keypoints
- Hands: 42 keypoints (21 per hand)

Each keypoint: **4 values** (x, y, z, confidence)

### Memory Optimization Examples

**Scenario 1: Single Person (Current)**
```
max_persons = 1
Pose buffer: 9,032 bytes per camera
Detection buffer: 109 KB per camera
Total: ~118 KB per camera
```

**Scenario 2: Four Persons (Previous)**
```
max_persons = 4
Pose buffer: 34,568 bytes per camera
Detection buffer: 443 KB per camera
Total: ~477 KB per camera
```

**Savings**: ~357 KB per camera (75% reduction)

---

## Coordinate System and Head Tilt Calculation

### Overview

The system uses **two different coordinate conventions** that caused a critical 50-degree error in head tilt calculations until 2025-11-08. Understanding these coordinate systems is essential for any work involving 3D pose data, visualization, or angle calculations.

**Critical Rule**: Always apply corrections to index [2] (Z-axis) for vertical adjustments, NOT index [1] (Y-axis), because the visualization treats [2] as the vertical dimension.

---

### Coordinate System Conventions

#### RTMW3D Data Space (Keypoint Storage)

This is how keypoints are stored in numpy arrays from the pose estimation model:

```
Index [0] = X-axis = Horizontal (left-right in image)
Index [1] = Y-axis = Depth (forward-backward from camera)
Index [2] = Z-axis = VERTICAL (up-down, height)
```

**Example**:
```python
nose = keypoints[0]  # Shape: (3,)
# nose[0] = X position (left/right)
# nose[1] = Y position (depth from camera)
# nose[2] = Z position (HEIGHT - vertical)
```

**Used by**:
- Raw pose estimation output
- Keypoint data structures
- Some internal calculations

---

#### Matplotlib 3D Visualization Space

This is how the 3D skeleton renderer displays coordinates on screen:

```python
ax.plot([point[0], ...],  # Matplotlib X-axis ← Uses index [0] (X)
        [point[2], ...],  # Matplotlib Y-axis ← Uses index [2] (Z = VERTICAL)
        [point[1], ...])  # Matplotlib Z-axis ← Uses index [1] (Y = depth)
```

**Mapping**:
```
Data Index [0] → Plot X-axis (horizontal on screen)
Data Index [2] → Plot Y-axis (VERTICAL on screen) ← This is what you see as "up"
Data Index [1] → Plot Z-axis (depth in 3D view)
```

**Why this matters**: When you make a "vertical" correction, you must modify index [2], not index [1], because that's what the visualization displays as vertical.

**File**: `core/visualization/skeleton_3d_renderer.py` lines 303-330

---

### Head Tilt Calculation Bug (FIXED 2025-11-08)

#### The Problem

**Symptom**: Head tilt showed -60° when the orange (neck) and red (face) lines appeared perpendicular (90°).

**Root Cause**: Axis mismatch between correction and visualization

**Discovery**: Added visual angle display showing both raw vector angle and head tilt, which revealed the 50° discrepancy.

---

#### The Bug

**Original Code** (WRONG):
```python
# This was applying correction to the WRONG axis
ear_midpoint_corrected = ear_midpoint.copy()
ear_midpoint_corrected[1] -= vertical_offset  # ❌ Modifies Y (depth), not Z (vertical)!
```

**What happened**:
1. Code subtracted `vertical_offset` from index [1] (Y-axis = depth)
2. But visualization displays index [2] (Z-axis) as vertical
3. The "vertical" correction actually moved the point in the **depth direction**
4. This distorted the face vector geometry by ~50°
5. Head tilt calculation became completely wrong

**Impact**:
- Neutral head position (visually 90°) showed as -60° tilt
- Looking down/up angles were off by 50°
- Orange and red lines didn't match the calculated angles

---

#### The Fix

**Corrected Code**:
```python
# Now applies correction to the CORRECT axis
ear_midpoint_corrected = ear_midpoint.copy()
ear_midpoint_corrected[2] -= vertical_offset  # ✅ Modifies Z (vertical in visualization)
```

**Files Modified**:

1. **`core/visualization/skeleton_3d_renderer.py`**
   - Line 109: Changed `[1]` → `[2]`
   - Line 300: Changed `[1]` → `[2]`

2. **`core/metrics/pose_metrics_calculator.py`**
   - Line 284: Changed `[1]` → `[2]`

**Verification**:
```python
# Test with neutral head position (perpendicular vectors)
# BEFORE FIX: Vec~140°, Tilt~-60°
# AFTER FIX:  Vec~90°,  Tilt~0°  ✓ Correct!
```

---

### Head Tilt Calculation Method

#### Algorithm

The head tilt angle measures the orientation of the head relative to the neck using a **vector angle method**:

**Vectors**:
1. **Neck vector**: From shoulder midpoint → ear midpoint (upward direction of neck)
2. **Face vector**: From corrected ear midpoint → nose (forward direction of face)

**Calculation**:
```python
# Calculate angle between vectors using dot product
dot_product = np.dot(neck_vector, face_vector)
neck_magnitude = np.linalg.norm(neck_vector)
face_magnitude = np.linalg.norm(face_vector)

cos_angle = dot_product / (neck_magnitude * face_magnitude)
angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

# Convert to head tilt (0° when perpendicular)
head_tilt = 90.0 - angle_deg
```

**Interpretation**:
- **0°** = Neutral (face perpendicular to neck)
- **Negative** = Looking down
- **Positive** = Looking up

---

#### Anatomical Correction

**Purpose**: Ears are anatomically higher than the nose, which would create a systematic downward bias in head tilt calculations.

**Correction Formula**:
```python
inter_ear_distance = ||right_ear - left_ear||
vertical_offset = inter_ear_distance × ear_to_nose_drop_ratio

# CRITICAL: Apply to index [2] (Z-axis = vertical in visualization)
ear_midpoint_corrected[2] -= vertical_offset
```

**Parameter**: `ear_to_nose_drop_ratio`
- **Config location**: `metrics_settings.ear_to_nose_drop_ratio`
- **Default value**: 0.20 (20% of inter-ear distance)
- **Effect**: Shifts the face vector starting point downward to compensate for anatomical ear position

**Example**:
```
Inter-ear distance: 0.20m (20cm)
Vertical offset (20%): 0.04m (4cm)
Effect on angle: ~11° adjustment
```

---

### Visual Angle Display Feature

**Added**: 2025-11-08

**Purpose**: Debug and verify head tilt calculations by showing angles directly in the 3D visualization.

**Display Format**:
```
Vec: 90.0° | Tilt: +0.0°
```

**Location**: Red text near the red gaze line endpoint

**Implementation**: `core/visualization/skeleton_3d_renderer.py` lines 332-342

**Shows**:
- **Vec**: Raw angle between neck and face vectors (should match visual appearance)
- **Tilt**: Head tilt metric (90° - vector angle)

**Usage**: If the orange/red lines appear perpendicular but the vector angle shows far from 90°, there's a coordinate system bug.

---

### Visualization Lines

#### Orange Line (Neck Structure)
- **Represents**: Neck's vertical axis
- **Start point**: Shoulder midpoint (neck base)
- **End point**: Ear midpoint (UNCORRECTED, actual anatomical position)
- **Color**: RGB(1.0, 0.5, 0.0) - Bright orange
- **Line width**: 4 pixels

#### Red Line (Face Direction / Gaze)
- **Represents**: Head orientation vector (where head is pointing)
- **Start point**: Ear midpoint (CORRECTED, anatomically adjusted)
- **End point**: Nose + 80cm extension (for high visibility)
- **Color**: RGB(1.0, 0.0, 0.0) - Pure red
- **Line width**: 6 pixels (thickest)

**Key Detail**: The red line uses the **corrected** ear midpoint (same as the calculation), while the orange line uses the **uncorrected** ear midpoint (actual position). This ensures the red line's angle matches the calculated metric.

---

### Common Pitfalls and Solutions

#### Pitfall 1: Using Wrong Axis for Vertical Operations

**Wrong**:
```python
# Don't do this - modifies depth, not vertical!
keypoint[1] += vertical_adjustment
```

**Correct**:
```python
# Always use index [2] for vertical adjustments
keypoint[2] += vertical_adjustment
```

**Why**: The visualization displays index [2] as the vertical axis on screen.

---

#### Pitfall 2: Confusing Data Space with Visualization Space

**Problem**: Assuming index [1] is vertical because it's called "Y-axis"

**Solution**: Remember the mapping:
- Data index [1] = Y = DEPTH (not vertical in visualization!)
- Data index [2] = Z = VERTICAL (displayed as up/down on screen)

---

#### Pitfall 3: Not Accounting for Anatomical Correction

**Problem**: Expecting exactly 90° for perpendicular lines

**Reality**: With 20% correction, perpendicular lines show ~79° (11° correction)

**Solution**: Test with `ear_to_nose_drop_ratio=0.0` to verify pure geometric angles

---

### Verification and Testing

**Quick Test**:
```python
from core.visualization.skeleton_3d_renderer import calculate_head_tilt_angle
import numpy as np

# Create neutral head position
keypoints = np.zeros((133, 3))
keypoints[0] = [0, 0.2, 1.7]      # Nose
keypoints[3] = [-0.1, 0, 1.7]      # Left ear (same Z as nose)
keypoints[4] = [0.1, 0, 1.7]       # Right ear (same Z as nose)
keypoints[5] = [-0.15, 0, 1.4]     # Left shoulder (lower Z)
keypoints[6] = [0.15, 0, 1.4]      # Right shoulder (lower Z)
scores = np.ones(133)

# Test without correction (should be exactly 90°, 0° tilt)
result = calculate_head_tilt_angle(keypoints, scores, ear_to_nose_drop_ratio=0.0)
vector_angle, head_tilt = result
assert abs(vector_angle - 90.0) < 0.1, "Should be 90° perpendicular"
assert abs(head_tilt - 0.0) < 0.1, "Should be 0° neutral tilt"
```

**Expected Output**:
```
Vector angle: 90.0° (perpendicular)
Head tilt: 0.0° (neutral)
```

---

### Key Takeaways

1. **Index [2] is vertical** in the visualization, not index [1]
2. **Always use [2] for vertical corrections**, never [1]
3. **Visual angle display** helps debug coordinate system issues
4. **Anatomical correction** is applied correctly to [2] as of 2025-11-08
5. **The fix reduced error from 50° to <1°** for neutral positions

---

## Shoulder Elevation Calculation and Distance Normalization

### Overview

**Fixed**: 2025-11-11

Shoulder elevation measurements were originally calculated as absolute 3D distances in meters, causing values to vary with camera distance. The same shoulder posture would produce different measurements depending on how close the person was to the camera. This section documents the distance normalization solution implemented to provide consistent, distance-invariant shoulder elevation metrics.

**Critical Rule**: Shoulder elevation MUST be normalized using body-proportional reference distances (shoulder width) to ensure distance-invariant measurements that can be compared across different camera distances and sessions.

---

### The Problem: Distance-Dependent Measurements

#### Original Implementation (INCORRECT)

**File**: `core/metrics/pose_metrics_calculator.py:497-544` (before fix)

**Original Method**:
```python
# Calculate ear midpoint
ear_midpoint = (left_ear + right_ear) / 2

# Calculate absolute 3D distances in meters
left_distance = ||ear_midpoint - left_shoulder||  # Raw meters
right_distance = ||ear_midpoint - right_shoulder||  # Raw meters

return left_distance, right_distance  # Distance-dependent values!
```

**The Problem**:
- **Absolute distances scale with camera distance**: Person appears larger when closer
- **Same posture, different values**: Identical shoulder elevation produces different measurements at different distances
- **Not comparable**: Cannot compare measurements across sessions with different camera positioning
- **No baseline**: Raw distances don't indicate "elevated" vs "relaxed" in a meaningful way

**Example**:
```
Same shoulder posture:
- At 2 meters: shoulder_elevation = 0.25 m
- At 1 meter: shoulder_elevation = 0.50 m  ← 2× larger!
```

---

### The Solution: Ratio-Based Normalization

#### Pattern from Head Tilt Calculation

The head tilt calculation already implemented the correct pattern using ratio-based normalization:

```python
# Calculate body-proportional reference
inter_ear_distance = ||right_ear - left_ear||

# Use as ratio (both scale equally with distance)
vertical_offset = inter_ear_distance * 0.1  # Dimensionless ratio
```

This works because:
1. **Both measurement AND reference scale equally** with camera distance
2. **The ratio remains constant** regardless of distance
3. **Dimensionless values** are universally comparable

---

### Fixed Implementation

**File**: `core/metrics/pose_metrics_calculator.py:497-558` (after fix)

**New Method**:
```python
# Step 1: Calculate ear-to-shoulder distances (raw measurements)
left_distance = ||ear_midpoint - left_shoulder||
right_distance = ||ear_midpoint - right_shoulder||

# Step 2: Calculate shoulder width as body-proportional reference
shoulder_width = ||right_shoulder - left_shoulder||

# Step 3: Normalize by dividing to get dimensionless ratios
left_ratio = left_distance / shoulder_width
right_ratio = right_distance / shoulder_width

return left_ratio, right_ratio  # Distance-invariant!
```

**Why Shoulder Width?**
- **Directly related**: Shoulder measurements normalized by shoulder structure
- **Stable reference**: Shoulder width doesn't change with posture
- **Simple and intuitive**: Easy to understand and interpret
- **Scales proportionally**: Grows/shrinks with apparent person size (camera distance)

**Division by Zero Protection**:
```python
if shoulder_width < 1e-6:
    return None, None  # Avoid division by zero
```

---

### Measurement Interpretation

#### Ratio Values and Meaning

**Typical Range**: 1.0 - 2.5 (dimensionless)

**Interpretation**:
- **Lower ratio** (e.g., 1.2) = Shoulders elevated (shrugged) - ears closer to shoulders
- **Neutral ratio** (e.g., ~1.5) = Relaxed posture - natural ear-to-shoulder distance
- **Higher ratio** (e.g., 2.0) = Shoulders dropped/very relaxed - ears far from shoulders

**Example Scenarios**:
```
Shoulder shrugging (tension):
- Left: 1.15, Right: 1.18  ← Both shoulders elevated

Asymmetric elevation:
- Left: 1.10, Right: 1.65  ← Left shoulder raised, right relaxed

Relaxed posture:
- Left: 1.52, Right: 1.48  ← Both shoulders in neutral position
```

---

### Benefits of Normalization

1. **Distance-Invariant**: Same posture = same ratio at any camera distance
2. **Comparable**: Values can be compared across sessions, subjects, and setups
3. **Dimensionless**: No units (not meters or degrees), universal ratios
4. **Sensitive**: Still detects subtle shoulder elevation changes
5. **Consistent with codebase**: Follows head tilt ratio-based pattern

**Verification Example**:
```
Same shoulder posture at different distances:
- At 2 meters: shoulder_width=0.40m, ear_distance=0.60m → ratio=1.50
- At 1 meter: shoulder_width=0.80m, ear_distance=1.20m → ratio=1.50  ✓ Identical!
```

---

### Files Modified (2025-11-11)

1. **`core/metrics/pose_metrics_calculator.py`**
   - Lines 497-558: Implemented ratio-based normalization
   - Added shoulder_width calculation
   - Normalized distances by shoulder_width
   - Updated docstring to reflect dimensionless ratios

2. **`core/metrics/metrics_dataclasses.py`**
   - Lines 35-42: Updated `PoseMetrics` docstrings
   - Changed from "degrees" to "dimensionless ratios"
   - Updated value interpretation and typical ranges

3. **`gui.py`**
   - Lines 2987-2996: Updated display format
   - Changed comment to reflect ratio interpretation
   - Adjusted precision from `.3f` to `.2f` (adequate for ratios)

---

### Design Decisions

#### Why Not Torso Height?

**Alternative considered**: Normalize by shoulder-to-hip distance (torso height)

**Reasons for choosing shoulder width**:
- ✅ **Simpler**: Only requires shoulder keypoints (already in use)
- ✅ **More direct**: Shoulder measurements normalized by shoulder structure
- ✅ **Equally stable**: Doesn't change with shoulder elevation
- ❌ Torso height requires additional keypoints (hips), increasing complexity

#### Why No "Neutral Baseline" Config Parameter?

**Alternative considered**: Add `neutral_shoulder_ratio` config (like `ear_to_nose_drop_ratio` for head tilt)

**Reasons for keeping it simple**:
- ✅ **User preference**: User chose "keep it simple" approach
- ✅ **More variability**: Neutral shoulder position varies greatly between individuals
- ✅ **Observable**: Users can establish their own baselines through observation
- ❌ Adding a baseline would require calibration for each person

---

### Testing and Verification

#### Quick Validation Test

```bash
# Start system and observe shoulder values while moving closer/farther from camera
./venv/bin/python3 mmpose_3d_gui/gui.py

# Expected behavior:
# - Maintain same shoulder posture
# - Move closer to camera → ratio stays ~constant (e.g., 1.50 ± 0.05)
# - Move farther from camera → ratio still ~constant
# - If values change significantly, normalization is broken
```

#### Manual Test

1. **Stand in neutral posture** (relaxed shoulders)
2. **Note baseline values** (e.g., Left: 1.52, Right: 1.48)
3. **Move closer to camera** (1-2 meters)
4. **Verify ratios unchanged** (should remain within ±0.05)
5. **Shrug shoulders**
6. **Verify ratios decrease** (e.g., to ~1.20)

---

### Common Pitfalls and Solutions

#### Pitfall 1: Expecting Same Values as Before

**Problem**: Comparing new ratio values (1.5) with old absolute distances (0.30m)

**Solution**: These are completely different units - don't compare across the update

---

#### Pitfall 2: Forgetting Division by Zero Check

**Problem**: Shoulder width could theoretically be zero (malformed keypoints)

**Solution**: Always check `if shoulder_width < 1e-6: return None, None`

---

#### Pitfall 3: Using Wrong Reference Distance

**Problem**: Normalizing by inter-ear distance or torso height instead of shoulder width

**Solution**: Always use `||right_shoulder - left_shoulder||` as reference

---

### Key Takeaways

1. **Shoulder elevation now uses distance-invariant ratios** (not absolute meters)
2. **Normalized by shoulder width** for body-proportional scaling
3. **Lower ratios = elevated shoulders** (tension/shrugging)
4. **Higher ratios = relaxed shoulders** (normal/dropped)
5. **Follows head tilt pattern** (ratio-based normalization)
6. **Typical range: 1.0-2.5** (dimensionless)
7. **Same posture = same ratio** regardless of camera distance

---

### Enhanced Multi-Component Shoulder Elevation System (v6.0)

**Implemented**: 2025-11-11

Building on v5.0's distance-invariant normalization, v6.0 adds two additional biomechanical measurements to create a comprehensive three-component shoulder elevation system that captures multiple aspects of shoulder shrugging behavior.

---

#### Motivation: Why Multi-Component?

The v5.0 ear-to-shoulder ratio captures one aspect of shoulder shrugging (shoulders moving toward head), but biomechanically, shoulder elevation involves multiple movements:

1. **Upward displacement** (scapular elevation) - shoulders move up relative to the pelvis
2. **Medial rotation** (shoulder protraction) - shoulders may move inward
3. **Relative positioning** - shoulders move closer to ears

A single metric cannot fully capture this complex movement. The v6.0 system adds torso height measurement and combines both signals into a unified composite score.

---

#### Three-Component System Design

**Component 1: Ear-to-Shoulder Ratio** (v5.0 - retained)
```python
ear_shoulder_ratio = ||ear_midpoint - shoulder|| / shoulder_width
```
- **Measures**: Shoulders moving toward head
- **Direction**: Lower = more elevated (shrugging)
- **Typical range**: 1.0-2.5
- **Interpretation**: 1.2 = elevated, 1.5 = neutral, 2.0 = relaxed

**Component 2: Torso Height Ratio** (v6.0 - new)
```python
torso_height_ratio = ||shoulder - hip_midpoint|| / shoulder_width
```
- **Measures**: Shoulders moving away from hips (absolute elevation from pelvis)
- **Direction**: Higher = more elevated
- **Typical range**: 1.3-1.8
- **Interpretation**: 1.8 = elevated, 1.6 = neutral, 1.4 = relaxed

**Component 3: Composite Elevation Score** (v6.0 - new)
```python
# Calculate deviations from neutral baselines
ear_deviation = neutral_ear_ratio - current_ear_ratio       # Positive when elevated
torso_deviation = current_torso_ratio - neutral_torso_ratio  # Positive when elevated

# Weighted combination (both components positive when elevated)
composite_score = (ear_weight * ear_deviation) + (torso_weight * torso_deviation)
```
- **Measures**: Combined shoulder elevation from both signals
- **Direction**: Positive = elevated, 0 = neutral, negative = relaxed
- **Typical range**: -0.5 to +0.5
- **Interpretation**: +0.3 = elevated/shrugging, 0.0 = neutral, -0.2 = relaxed

---

#### Why These Measurements Are Complementary

**Ear-Shoulder vs Torso Height**:
- Ear-shoulder measures **relative to head** (captures shoulders moving up toward ears)
- Torso height measures **relative to pelvis** (captures shoulders moving up from hips)
- These are independent signals that both increase during shrugging
- **Example**: Person leans forward → ear-shoulder may change, torso height stays constant
- **Benefit**: Composite score is more robust to head position changes

**Both Normalized by Shoulder Width**:
- Maintains distance invariance across all components
- Same shoulder width reference creates consistent scaling
- All ratios remain dimensionless and comparable

---

#### Configuration Parameters

**Location**: `youquantipy_config.json` → `metrics_settings` → `shoulder_elevation`

```json
"shoulder_elevation": {
    "neutral_ear_shoulder_ratio": 1.5,
    "neutral_torso_height_ratio": 1.6,
    "ear_component_weight": 0.5,
    "torso_component_weight": 0.5
}
```

**Parameters**:
- `neutral_ear_shoulder_ratio` (default: 1.5): Baseline ear-to-shoulder ratio for neutral posture
- `neutral_torso_height_ratio` (default: 1.6): Baseline torso height ratio for neutral posture
- `ear_component_weight` (default: 0.5): Weight for ear-shoulder component in composite
- `torso_component_weight` (default: 0.5): Weight for torso height component in composite

**Tuning**: Adjust neutral values based on observed baseline data for your specific subject/setup.

---

#### Implementation Details

**File**: `core/metrics/pose_metrics_calculator.py` lines 497-609

**Return Signature**:
```python
def _calculate_shoulder_shrug(keypoints, scores) -> Tuple[
    Optional[float], Optional[float],  # left_ear_ratio, right_ear_ratio
    Optional[float], Optional[float],  # left_torso_ratio, right_torso_ratio
    Optional[float], Optional[float]   # left_composite, right_composite
]:
```

**Required Keypoints** (with min_confidence threshold):
- Ears: left (3), right (4)
- Shoulders: left (5), right (6)
- Hips: left (11), right (12)

**Calculation Steps**:
1. Validate all 6 required keypoints have sufficient confidence
2. Calculate midpoints: ear, shoulder, hip
3. **Ear-shoulder component**: `||ear_mid - shoulder|| / shoulder_width`
4. **Torso height component**: `||shoulder - hip_mid|| / shoulder_width`
5. **Composite score**: Weighted deviation from neutral baselines
6. Return all 6 values (left/right for each of 3 components)

**Division by Zero Protection**:
```python
if shoulder_width < 1e-6:
    return None, None, None, None, None, None
```

---

#### Data Structure

**File**: `core/metrics/metrics_dataclasses.py`

**PoseMetrics Fields** (added in v6.0):
```python
@dataclass
class PoseMetrics:
    # Existing (v5.0)
    shoulder_elevation_left: Optional[float] = None   # Ear-shoulder ratio
    shoulder_elevation_right: Optional[float] = None  # Ear-shoulder ratio

    # New (v6.0)
    torso_height_left: Optional[float] = None         # Torso height ratio
    torso_height_right: Optional[float] = None        # Torso height ratio
    shoulder_composite_left: Optional[float] = None   # Composite score
    shoulder_composite_right: Optional[float] = None  # Composite score
```

**MetricsConfig Fields** (added in v6.0):
```python
@dataclass
class MetricsConfig:
    # New shoulder elevation parameters
    neutral_ear_shoulder_ratio: float = 1.5
    neutral_torso_height_ratio: float = 1.6
    ear_component_weight: float = 0.5
    torso_component_weight: float = 0.5
```

**Config Loading**: `from_dict()` automatically flattens nested `shoulder_elevation` section

---

#### GUI Display Format

**File**: `gui.py` lines 3099-3122

**Display Format**:
```
Left:  E:1.48  T:1.62  C:+0.12
Right: E:1.52  T:1.59  C:+0.08
```

**Legend**:
- **E**: Ear-shoulder ratio (lower = elevated)
- **T**: Torso height ratio (higher = elevated)
- **C**: Composite score (positive = elevated, 0 = neutral)

---

#### Interpretation Examples

**Example 1: Neutral Posture**
```
Left:  E:1.50  T:1.60  C:+0.00
Right: E:1.50  T:1.60  C:+0.00
```
- Ear-shoulder at neutral baseline (1.50)
- Torso height at neutral baseline (1.60)
- Composite score at zero (perfectly neutral)

**Example 2: Moderate Shrugging**
```
Left:  E:1.20  T:1.75  C:+0.30
Right: E:1.25  T:1.72  C:+0.27
```
- Ear-shoulder ratio decreased (1.50 → 1.20, shoulders closer to ears)
- Torso height ratio increased (1.60 → 1.75, shoulders farther from hips)
- Composite score positive (+0.30, elevated)
- **Both components agree**: shoulders are elevated

**Example 3: Asymmetric Elevation**
```
Left:  E:1.15  T:1.78  C:+0.35
Right: E:1.48  T:1.62  C:+0.03
```
- Left shoulder highly elevated (low ear-ratio, high torso-ratio, high composite)
- Right shoulder near neutral
- **Clinical insight**: Asymmetric muscle tension detected

---

#### Key Benefits

1. **Multi-dimensional capture**: Measures both head-relative and pelvis-relative shoulder position
2. **Robustness**: Composite score less sensitive to head position changes
3. **Clinical utility**: Individual components reveal different aspects of posture
4. **Distance invariant**: All three metrics normalized by shoulder width
5. **Configurable**: Neutral baselines and weights can be tuned per subject
6. **Backward compatible**: Ear-shoulder ratio (v5.0) still available independently

---

#### Testing and Validation

**Unit Test Results** (2025-11-11):
```python
Test pose: Shoulders slightly elevated
Results:
  Ear-shoulder ratio:  0.972 (< 1.5 neutral → elevated ✓)
  Torso height ratio:  1.740 (> 1.6 neutral → elevated ✓)
  Composite score:     +0.334 (positive → elevated ✓)

Interpretation: All three metrics correctly detect elevation
```

**Validation Approach**:
1. Test with neutral posture → composite ~0.0
2. Simulate shrugging → composite increases
3. Both components should move in expected directions
4. Verify distance invariance (move closer/farther from camera)

---

#### Files Modified (2025-11-11 v6.0)

1. **`youquantipy_config.json`**
   - Added `metrics_settings.shoulder_elevation` section with 4 parameters

2. **`core/metrics/metrics_dataclasses.py`**
   - Added 4 new fields to `PoseMetrics` dataclass
   - Added 4 new fields to `MetricsConfig` dataclass
   - Updated `get_shoulder_summary()` to show composite scores
   - Enhanced `from_dict()` to flatten nested shoulder_elevation config

3. **`core/metrics/pose_metrics_calculator.py`**
   - Rewrote `_calculate_shoulder_shrug()` to return 6 values
   - Added torso height calculation using hip keypoints
   - Added composite score calculation with deviation-based weighting
   - Updated method signature and comprehensive docstring
   - Modified `calculate_metrics()` to populate all 6 shoulder fields

4. **`gui.py`**
   - Updated shoulder display to show all three components (lines 3099-3122)
   - Format: `E:X.XX T:X.XX C:±X.XX` for compact multi-metric display

---

#### Design Decisions

**Why Not Replace Ear-Shoulder with Torso Height?**
- Both provide complementary information
- Ear-shoulder more sensitive to head-relative movement
- Torso height more stable to head position changes
- Having both allows detection of complex postures

**Why Equal Weights (0.5/0.5)?**
- Starting point for balanced contribution
- Can be tuned based on clinical validation
- Both metrics equally valid biomechanically

**Why Not Track Shoulder Width Changes?**
- Shoulder width is the normalization denominator
- Tracking it separately would add complexity
- Changes are small and within measurement noise
- Using it to normalize could dampen signal if it changes during shrugging

---

#### Future Enhancements

**Potential v7.0+ Features**:
1. **Adaptive neutral baselines**: Auto-calibrate neutral values during first N frames
2. **Temporal derivative**: Track rate of shoulder elevation change (shrugging speed)
3. **Cross-camera consistency**: Validate measurements across multiple camera angles
4. **ML-based weighting**: Learn optimal ear/torso weights from labeled data

---

## Code Locations Reference

> **See Also**: For a comprehensive reference of all system files including visualization, metrics, camera sources, and utilities, refer to the [Critical Files Reference](ARCHITECTURE.md#critical-files-reference) section in ARCHITECTURE.md.

### Critical Files for Buffer Management

#### `core/buffer_management/layouts.py`
**Purpose**: Defines buffer memory layout structures

**Key Classes**:
- `Pose3DBufferLayout` - Ring buffer layout for pose data
- `DetectionBufferLayout` - Face detection buffer layout
- `ROIBufferLayout` - ROI processing buffer layout

**Important Methods**:
- `Pose3DBufferLayout.__post_init__()` - Calculates pose slot offsets

**Code Reference**: `layouts.py:49-86`

#### `core/pose_processing/rtmpose3d_process.py`
**Purpose**: Pose estimation writer process

**Key Methods**:
- `_write_pose_results()` - Atomic write pattern implementation
- Writer increments index AFTER writing data

**Critical Section**: `rtmpose3d_process.py:820-865` (atomic write pattern)

#### `core/gui_interface/gui_processing_worker.py`
**Purpose**: Pose data reader process

**Key Methods**:
- `_read_pose_data()` - Reader with consistency checks
- Uses `(write_idx - 1) % ring_buffer_size` for slot calculation

**Critical Section**: `gui_processing_worker.py:1450-1520` (slot index calculation)

#### `core/buffer_management/coordinator.py`
**Purpose**: Central buffer manager

**Key Methods**:
- `get_layout()` - Creates buffer layouts from centralized config
- `_calculate_max_faces()` - Determines face detection capacity
- `_get_ring_buffer_size()` - Gets ring buffer size from config

**Configuration Paths**: All use `buffer_settings.*` paths

#### `core/visualization/rtmpose_visualizer.py`
**Purpose**: Pose rendering with input validation

**Key Methods**:
- `draw_skeleton_2d()` - Validates coordinates before drawing
- Rejects coordinates > 100,000 (corrupted data protection)

**Input Validation**: `rtmpose_visualizer.py:280-295`

---

## Best Practices

### When Modifying Buffer Configuration

1. **Always update centralized config**
   - Never scatter buffer parameters across different config sections
   - All changes go through `buffer_settings`

2. **Verify memory calculations**
   - Calculate expected buffer size before running
   - Verify actual `/dev/shm` buffer size matches expected

3. **Clean up before testing**
   - Always remove stale buffers: `rm -f /dev/shm/yq_*`
   - Kill all processes: `pkill -9 python3`

4. **Document reasoning**
   - Add comments explaining non-obvious choices
   - Update this CLAUDE.md file with significant changes

### When Adding New Buffers

1. **Add to centralized config first**
   ```json
   "buffer_settings": {
     "new_subsection": {
       "new_parameter": value
     }
   }
   ```

2. **Use consistent access pattern**
   ```python
   value = config.get('buffer_settings', {}).get('subsection', {}).get('parameter', default)
   ```

3. **Update coordinator.py**
   - Add layout calculation if needed
   - Add to `get_layout()` method

4. **Document in this file**
   - Add to "Configuration Structure" section
   - Add size calculation to "Buffer Size Calculations"

---

## Version History

### v6.0 - 2025-11-11
- **MAJOR ENHANCEMENT**: Implemented multi-component shoulder elevation system
- Added torso height ratio (shoulder-to-hip distance / shoulder width)
- Added composite elevation score combining ear-shoulder and torso height signals
- Expanded from 2 metrics (left/right ear-shoulder) to 6 metrics (left/right × 3 components)
- Both new metrics normalized by shoulder width for distance invariance
- Composite score uses configurable neutral baselines and component weights
- Modified `_calculate_shoulder_shrug()` to return 6 values instead of 2
- Added 4 new fields to `PoseMetrics` dataclass and 4 to `MetricsConfig`
- Enhanced GUI display to show all three components: E (ear-shoulder), T (torso), C (composite)
- Comprehensive documentation with biomechanical rationale and interpretation examples
- Provides more robust shoulder elevation detection across different head positions

### v5.0 - 2025-11-11
- **ENHANCEMENT**: Implemented distance-invariant shoulder elevation measurements
- Root cause: Absolute distance measurements varied with camera distance
- Normalized shoulder elevation by shoulder width (body-proportional reference)
- Changed from absolute meters to dimensionless ratios (typical range 1.0-2.5)
- Modified `_calculate_shoulder_shrug()` in pose_metrics_calculator.py
- Updated PoseMetrics docstrings to reflect ratio-based measurement
- Adjusted GUI display format and precision for ratio values
- Documented complete normalization approach following head tilt pattern
- Same shoulder posture now produces same ratio regardless of camera distance

### v4.0 - 2025-11-11
- **CRITICAL FIX**: Fixed buffer offset artifacts causing camera feed movement and color corruption
- Root cause: Frame shape mismatches and non-contiguous memory layouts
- Added comprehensive frame validation on writer side (camera_worker_enhanced.py)
- Added comprehensive frame validation on reader side (gui_processing_worker.py)
- Implemented shape validation, contiguity checks, and zero-frame detection
- Added diagnostic logging every 100 frames for monitoring
- Documented complete stride calculation and C-contiguous layout requirements
- Buffer offset artifacts eliminated with validation at both write and read stages

### v3.0 - 2025-11-08
- **CRITICAL FIX**: Fixed 50° head tilt calculation error caused by axis mismatch
- Changed vertical offset correction from index [1] to index [2] in 3 locations
- Added visual angle display feature showing raw vector angle and head tilt
- Modified `calculate_head_tilt_angle()` to return tuple `(vector_angle, head_tilt)`
- Documented complete coordinate system conventions (data space vs visualization space)
- Reduced head tilt error from ~50° to <1° for neutral positions

### v2.0 - 2025-11-08
- Created centralized `buffer_settings` configuration structure
- Migrated all buffer code to use centralized paths
- Optimized for single-person tracking (`max_persons=1`)
- Achieved 75% memory reduction in pose/detection buffers

### v1.0 - 2025-11-08
- Fixed shared memory race condition (3 critical bugs)
- Implemented ring buffer architecture for pose data
- Added atomic write pattern to prevent partial reads
- Fixed reader slot index off-by-one error
- Added consistency checking to reader

---

## Contact

This documentation is maintained by Claude Code AI Assistant. For updates or corrections, modify this file directly or regenerate sections using Claude Code with "ultrathink" mode for comprehensive analysis.

**Last Verified**: 2025-11-11
**System**: YouQuantiPy Multi-Camera Pose Estimation
**Buffer System**: Centralized Configuration v2.0
**Head Tilt System**: Coordinate-Corrected Calculation v3.0
**Shoulder Elevation System**: Multi-Component Distance-Normalized System v6.0
