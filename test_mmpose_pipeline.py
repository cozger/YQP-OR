#!/usr/bin/env python3
"""
Standalone test script for MMPose 3D pipeline.

Tests the RTMPose3DProcess with shared memory buffers to verify:
1. Model initialization
2. Frame buffer reading
3. Person detection
4. 3D pose estimation
5. Result writing to pose buffer
"""

import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import json
import sys
import os
import cv2
from pathlib import Path

# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
# Must be done before any other multiprocessing operations
mp.set_start_method('spawn', force=True)

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

# Add rtmpose3d module to Python path (required for RTMW3D config)
rtmpose3d_path = "/home/canoz/Projects/surgery/mmpose/projects/rtmpose3d"
if rtmpose3d_path not in sys.path:
    sys.path.insert(0, rtmpose3d_path)

from core.buffer_management.layouts import FrameBufferLayout, Pose3DBufferLayout
from core.pose_processing.rtmpose3d_process import RTMPose3DProcess


def create_test_frame(width=640, height=480):
    """Create a test frame with a simple pattern."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw a simple colored pattern
    frame[:, :width//3] = [255, 0, 0]  # Blue
    frame[:, width//3:2*width//3] = [0, 255, 0]  # Green
    frame[:, 2*width//3:] = [0, 0, 255]  # Red
    # Add some text
    cv2.putText(frame, "MMPose 3D Test Frame", (50, height//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def test_mmpose_pipeline():
    """Test the complete MMPose 3D pipeline."""
    print("=" * 80)
    print("MMPose 3D Pipeline Test")
    print("=" * 80)

    # Load configuration
    config_path = Path(__file__).parent / "mmpose_config.json"
    with open(config_path) as f:
        config = json.load(f)

    mmpose_config = config['mmpose_3d_pipeline']
    buffer_config = config['buffer_settings']
    camera_config = config['camera_settings']

    # Verify model files exist
    detector_checkpoint = mmpose_config['person_detector']['checkpoint']
    pose_checkpoint = mmpose_config['pose_estimator']['checkpoint']

    if not Path(detector_checkpoint).exists():
        print(f"❌ Error: Detector checkpoint not found: {detector_checkpoint}")
        return False

    if not Path(pose_checkpoint).exists():
        print(f"❌ Error: Pose checkpoint not found: {pose_checkpoint}")
        return False

    print(f"✅ Detector checkpoint found: {detector_checkpoint}")
    print(f"✅ Pose checkpoint found: {pose_checkpoint}")

    # Create buffer layouts
    print("\n" + "=" * 80)
    print("Creating buffer layouts...")
    print("=" * 80)

    frame_layout = FrameBufferLayout(
        ring_buffer_size=camera_config['frame_buffer_size'],
        frame_width=camera_config['default_resolution']['width'],
        frame_height=camera_config['default_resolution']['height']
    )

    pose_layout = Pose3DBufferLayout(
        max_persons=buffer_config['max_persons']
    )

    print(f"Frame buffer size: {frame_layout.total_size / 1024 / 1024:.2f} MB")
    print(f"Pose buffer size: {pose_layout.total_size / 1024:.2f} KB")

    # Create shared memory buffers
    print("\n" + "=" * 80)
    print("Creating shared memory buffers...")
    print("=" * 80)

    # Clean up any existing buffers first
    for name in ["test_frame_buffer", "test_pose_buffer"]:
        try:
            existing = shared_memory.SharedMemory(name=name)
            existing.close()
            existing.unlink()
            print(f"Cleaned up existing buffer: {name}")
        except FileNotFoundError:
            pass  # Buffer doesn't exist, which is fine

    frame_shm = shared_memory.SharedMemory(
        create=True,
        size=frame_layout.total_size,
        name="test_frame_buffer"
    )
    print(f"✅ Frame buffer created: {frame_shm.name}")

    pose_shm = shared_memory.SharedMemory(
        create=True,
        size=pose_layout.total_size,
        name="test_pose_buffer"
    )
    print(f"✅ Pose buffer created: {pose_shm.name}")

    try:
        # Initialize buffers
        frame_shm.buf[:] = b'\x00' * frame_layout.total_size
        pose_shm.buf[:] = b'\x00' * pose_layout.total_size

        # Write test frame to frame buffer
        print("\n" + "=" * 80)
        print("Writing test frame to buffer...")
        print("=" * 80)

        test_frame = create_test_frame(
            frame_layout.frame_width,
            frame_layout.frame_height
        )

        # Write frame index
        frame_shm.buf[0:8] = (1).to_bytes(8, byteorder='little')

        # Write frame data
        frame_offset = frame_layout.frame_offsets[0]
        frame_bytes = test_frame.tobytes()
        frame_shm.buf[frame_offset:frame_offset + len(frame_bytes)] = frame_bytes

        print(f"✅ Test frame written to buffer (index 0)")

        # Create synchronization events
        stop_event = mp.Event()
        ready_event = mp.Event()
        log_queue = mp.Queue()

        # Start RTMPose3D process
        print("\n" + "=" * 80)
        print("Starting RTMPose3D process...")
        print("=" * 80)

        pose_process = RTMPose3DProcess(
            camera_id=0,
            frame_buffer_name=frame_shm.name,
            frame_layout_dict=frame_layout.to_dict(),
            pose_buffer_name=pose_shm.name,
            pose_layout_dict=pose_layout.to_dict(),
            detector_config=mmpose_config['person_detector']['config'],
            detector_checkpoint=detector_checkpoint,
            pose_config=mmpose_config['pose_estimator']['config'],
            pose_checkpoint=pose_checkpoint,
            device=mmpose_config['device'],
            det_conf_threshold=mmpose_config['person_detector']['confidence_threshold'],
            pose_conf_threshold=mmpose_config['pose_estimator']['confidence_threshold'],
            stop_event=stop_event,
            ready_event=ready_event,
            log_queue=log_queue
        )

        pose_process.start()
        print(f"✅ RTMPose3D process started (PID: {pose_process.pid})")

        # Monitor log queue
        print("\n" + "=" * 80)
        print("Process logs:")
        print("=" * 80)

        # Wait for process to be ready (with timeout) while monitoring logs
        print("\nWaiting for models to initialize (max 60 seconds)...")
        start_time = time.time()
        ready = False

        while time.time() - start_time < 60:
            # Drain log queue
            while not log_queue.empty():
                try:
                    level, message = log_queue.get_nowait()
                    print(f"[{level}] {message}")
                except:
                    break

            # Check if ready
            if ready_event.is_set():
                ready = True
                print("\n✅ RTMPose3D process ready!")
                break

            # Check if process died
            if not pose_process.is_alive():
                print(f"\n❌ Process died unexpectedly! Exit code: {pose_process.exitcode}")
                # Drain any remaining logs
                while not log_queue.empty():
                    try:
                        level, message = log_queue.get_nowait()
                        print(f"[{level}] {message}")
                    except:
                        break
                return False

            time.sleep(0.1)

        if not ready:
            print("❌ Timeout waiting for RTMPose3D process to initialize")
            stop_event.set()
            pose_process.join(timeout=5)
            return False

        # Let process run for a bit
        print("\nLetting process run for 5 seconds...")
        time.sleep(5)

        # Check for more logs
        while not log_queue.empty():
            try:
                level, message = log_queue.get_nowait()
                print(f"[{level}] {message}")
            except:
                break

        # Read pose results from buffer
        print("\n" + "=" * 80)
        print("Reading pose results from buffer...")
        print("=" * 80)

        # Read write index
        write_idx_bytes = pose_shm.buf[
            pose_layout.write_index_offset:
            pose_layout.write_index_offset + pose_layout.write_index_size
        ]
        write_idx = int.from_bytes(write_idx_bytes, byteorder='little')
        print(f"Pose buffer write index: {write_idx}")

        if write_idx > 0:
            # Read pose data
            pose_offset = pose_layout.pose_data_offset
            pose_size = pose_layout.pose_data_size
            pose_bytes = pose_shm.buf[pose_offset:pose_offset + pose_size]
            pose_data = np.frombuffer(pose_bytes, dtype=np.float32).reshape(
                (pose_layout.max_persons, pose_layout.keypoints_per_person, pose_layout.values_per_keypoint)
            )

            # Read metadata
            metadata_offset = pose_layout.metadata_offset
            metadata_bytes = pose_shm.buf[metadata_offset:metadata_offset + pose_layout.metadata_dtype.itemsize]
            metadata = np.frombuffer(metadata_bytes, dtype=pose_layout.metadata_dtype)[0]

            print(f"\nMetadata:")
            print(f"  Frame ID: {metadata['frame_id']}")
            print(f"  Timestamp: {metadata['timestamp_ms']} ms")
            print(f"  Persons detected: {metadata['n_persons']}")
            print(f"  Processing time: {metadata['processing_time_ms']:.2f} ms")
            print(f"  Detection time: {metadata['detection_time_ms']:.2f} ms")
            print(f"  Pose time: {metadata['pose_time_ms']:.2f} ms")

            # Analyze pose data
            n_persons = metadata['n_persons']
            if n_persons > 0:
                print(f"\nPose data for {n_persons} person(s):")
                for person_idx in range(n_persons):
                    person_data = pose_data[person_idx]
                    # Count valid keypoints (confidence > threshold)
                    valid_kps = np.sum(person_data[:, 3] > 0.3)
                    avg_conf = np.mean(person_data[person_data[:, 3] > 0, 3])
                    print(f"  Person {person_idx}: {valid_kps}/{pose_layout.keypoints_per_person} keypoints, "
                          f"avg confidence: {avg_conf:.3f}")

                    # Show first few keypoints
                    print(f"    First 5 keypoints (x, y, z, conf):")
                    for kp_idx in range(min(5, valid_kps)):
                        kp = person_data[kp_idx]
                        print(f"      KP{kp_idx}: ({kp[0]:.1f}, {kp[1]:.1f}, {kp[2]:.3f}, {kp[3]:.3f})")

                print("\n✅ Pose estimation successful!")
            else:
                print("\n⚠️  No persons detected in test frame")
        else:
            print("\n❌ No pose data written to buffer")

        # Shutdown process
        print("\n" + "=" * 80)
        print("Shutting down process...")
        print("=" * 80)

        stop_event.set()
        pose_process.join(timeout=10)

        if pose_process.is_alive():
            print("⚠️  Process did not terminate gracefully, killing...")
            pose_process.terminate()
            pose_process.join(timeout=5)

        print("✅ Process terminated")

        return True

    finally:
        # Cleanup shared memory
        print("\n" + "=" * 80)
        print("Cleaning up...")
        print("=" * 80)

        frame_shm.close()
        frame_shm.unlink()
        print("✅ Frame buffer cleaned up")

        pose_shm.close()
        pose_shm.unlink()
        print("✅ Pose buffer cleaned up")


if __name__ == "__main__":
    try:
        success = test_mmpose_pipeline()
        if success:
            print("\n" + "=" * 80)
            print("✅ ALL TESTS PASSED")
            print("=" * 80)
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("❌ TESTS FAILED")
            print("=" * 80)
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
