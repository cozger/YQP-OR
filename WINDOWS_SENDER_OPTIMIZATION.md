# Windows H.264 Sender Optimization Guide

## Problem Statement

**Current Performance**: 20.9 FPS (should be 30 FPS)

**Root Cause**: Sending individual NAL units creates excessive network overhead.

### Current Behavior
```
Frame 1: [SPS][PPS][IDR slice][SEI][P-slice][SEI][P-slice][SEI]...
         ↓    ↓     ↓         ↓    ↓        ↓    ↓        ↓
         8 separate ZMQ messages per frame!
```

**Analysis from debug logs**:
- 1063 NAL units in 15 seconds = 70.8 messages/second
- 315 frames decoded = 20.9 FPS
- **3.37 NAL units per frame** (excessive message overhead)

### Network Overhead Per Message
Each ZMQ `send_multipart()` call includes:
- TCP packet overhead (~40 bytes)
- ZMQ framing (~20 bytes)
- Serialization overhead
- **Total**: ~60-80 bytes overhead per 12-byte SEI packet!

## Solution: NAL Unit Aggregation

### Aggregate Before Sending
```
Frame 1: [SPS+PPS+IDR+SEI+P-slice+SEI+P-slice+SEI] (single message)
         ↓
         1 ZMQ message per frame
```

**Expected improvement**:
- 70.8 messages/sec → **30 messages/sec** (57% reduction)
- **20.9 FPS → 30 FPS** (43% increase)

---

## Implementation Guide

### Step 1: Understanding H.264 NAL Unit Types

NAL units start with a 4-byte start code `00 00 00 01` followed by a NAL header byte.

**NAL Header Format**: `0b0XXXXYYY`
- `XXXXX` = NAL unit type (bits 0-4)
- `YYY` = unused (bits 5-7)

**Common NAL types**:
```python
NAL_TYPE_SPS = 0x07      # Sequence Parameter Set
NAL_TYPE_PPS = 0x08      # Picture Parameter Set
NAL_TYPE_IDR = 0x05      # IDR (keyframe) slice
NAL_TYPE_SLICE = 0x01    # Non-IDR slice (P/B-frame)
NAL_TYPE_SEI = 0x06      # Supplemental info (metadata)
```

### Step 2: Frame Boundary Detection

**A complete frame includes**:
1. **Optional**: SPS/PPS (only sent on keyframes or parameter changes)
2. **Required**: IDR slice (0x05) OR Non-IDR slice (0x01)
3. **Optional**: SEI metadata (0x06)

**Detection Strategy**:
```python
def is_slice_nal(nal_data):
    """Check if NAL unit is a slice (marks frame end)."""
    if len(nal_data) < 5:
        return False

    # Skip start code (4 bytes: 00 00 00 01)
    nal_header = nal_data[4]
    nal_type = nal_header & 0x1F  # Extract type (bits 0-4)

    # Slice types indicate frame completion
    return nal_type in [0x01, 0x05, 0x21]  # Non-IDR, IDR, Coded Slice
```

### Step 3: Buffering and Aggregation

#### Pseudo-Code
```python
# Windows sender script

import struct

class FrameAggregator:
    def __init__(self):
        self.nal_buffer = []  # Buffer NAL units for current frame

    def add_nal_unit(self, nal_data):
        """Add NAL unit to buffer and check if frame is complete."""
        self.nal_buffer.append(nal_data)

        # Check if this NAL marks end of frame
        if self.is_slice_nal(nal_data):
            # Frame complete - send aggregated data
            complete_frame = b''.join(self.nal_buffer)
            self.send_frame(complete_frame)

            # Reset buffer for next frame
            self.nal_buffer = []

    def is_slice_nal(self, nal_data):
        """Detect frame boundary (slice NAL)."""
        if len(nal_data) < 5:
            return False

        nal_header = nal_data[4]
        nal_type = nal_header & 0x1F

        # Slice NAL types indicate frame end
        return nal_type in [0x01, 0x05, 0x21]

    def send_frame(self, frame_data):
        """Send complete frame via ZMQ."""
        topic = b"cam0"
        msg_type = b"H264_NAL"
        length = struct.pack("!I", len(frame_data))

        # Single ZMQ message for entire frame
        zmq_socket.send_multipart([topic, msg_type, length, frame_data])
```

### Step 4: Integration with Existing Sender

#### Current Code Pattern (Typical)
```python
# Current: Sending each NAL individually
while capturing:
    ret, frame = cap.read()

    # Encode frame to H.264
    nal_units = encoder.encode(frame)  # Returns list of NAL units

    for nal in nal_units:
        # ❌ INEFFICIENT: Send each NAL separately
        send_nal_unit(nal)
```

#### Optimized Code Pattern
```python
# Optimized: Aggregate NALs before sending
aggregator = FrameAggregator()

while capturing:
    ret, frame = cap.read()

    # Encode frame to H.264
    nal_units = encoder.encode(frame)

    for nal in nal_units:
        # ✅ EFFICIENT: Buffer and aggregate
        aggregator.add_nal_unit(nal)  # Sends automatically when frame complete
```

---

## Implementation Checklist

### Windows Sender Modifications

- [ ] **Add `FrameAggregator` class** (see pseudo-code above)
- [ ] **Implement `is_slice_nal()` detector**
- [ ] **Replace individual sends with aggregation**
- [ ] **Test with single camera first**
- [ ] **Extend to multi-camera setup**

### Testing Steps

1. **Baseline measurement** (current):
   ```bash
   # WSL2
   python3 test_h264_zmq_simple.py --ip 172.17.112.1 --duration 15
   ```
   Expected: ~20.9 FPS, ~70 messages/sec

2. **After aggregation** (target):
   ```bash
   # WSL2 (same command)
   python3 test_h264_zmq_simple.py --ip 172.17.112.1 --duration 15
   ```
   Expected: **30 FPS**, ~30 messages/sec

3. **Validate frame quality**:
   - Check for decoding artifacts
   - Verify frame synchronization
   - Confirm no data corruption

---

## Expected Performance Gains

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **FPS** | 20.9 | **30** | **+43%** |
| **Messages/sec** | 71 | 30 | -57% |
| **Frame time** | 47.8ms | **33.3ms** | -30% |
| **Bandwidth** | Higher | Lower | -40% |
| **CPU (network)** | Higher | Lower | -57% |

---

## Troubleshooting

### Issue: No frames decoded after changes
**Cause**: Aggregation logic may be buffering incomplete frames
**Solution**: Verify `is_slice_nal()` correctly detects frame boundaries

### Issue: Decoding errors or artifacts
**Cause**: NAL units may be incorrectly concatenated
**Solution**: Ensure start codes (`00 00 00 01`) are preserved between NAL units

### Issue: Still only ~21 FPS
**Cause**: Aggregation not working, still sending individual NALs
**Solution**: Add debug logging to confirm aggregation:
```python
print(f"Sending aggregated frame: {len(self.nal_buffer)} NAL units, {len(frame_data)} bytes")
```

---

## Alternative: Simple Buffering Approach

If the full aggregation is complex, try this simpler approach:

### Buffer by Time (30ms)
```python
import time

nal_buffer = []
last_send_time = time.time()

while capturing:
    # ... encode frame ...

    for nal in nal_units:
        nal_buffer.append(nal)

    # Send buffered NALs every 30ms (30 FPS)
    if time.time() - last_send_time >= 0.030:
        if nal_buffer:
            aggregated = b''.join(nal_buffer)
            send_frame(aggregated)
            nal_buffer = []
        last_send_time = time.time()
```

**Pros**: Simple, no NAL parsing needed
**Cons**: May split frames incorrectly, less precise

---

## Summary

**Key Takeaway**: The Windows sender must aggregate NAL units into complete frames before sending to achieve 30 FPS.

**Implementation Priority**:
1. ✅ **WSL2 receiver optimized** (CUDA working, minimal gain)
2. ⏳ **Windows sender aggregation** (CRITICAL - will achieve 30 FPS)
3. 📋 **Optional**: Time-based buffering as fallback

**Files to modify on Windows**:
- Windows sender script (exact path: `D:\Projects\youquantipy-redo\windows_bridge\win_cam_sender.py` or equivalent)

**Questions?** Review H.264 NAL structure at: https://en.wikipedia.org/wiki/Network_Abstraction_Layer
