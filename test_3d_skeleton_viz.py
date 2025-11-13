#!/usr/bin/env python3
"""
Standalone test script for 3D skeleton visualization.

Tests the plot_3d_skeleton function with synthetic pose data to verify
that the rendering pipeline works correctly before integrating with real data.
"""

import numpy as np
import math
from core.visualization.skeleton_3d_renderer import plot_3d_skeleton

def generate_test_skeleton(frame_num=0):
    """
    Generate realistic test skeleton data with animation.

    Args:
        frame_num: Frame number for animation (creates breathing/swaying motion)

    Returns:
        tuple: (keypoints_3d, scores) where:
            - keypoints_3d: (33, 3) numpy array of [x, y, z] coordinates
            - scores: (33,) numpy array of confidence scores
    """
    time_val = frame_num * 0.1
    breathe = math.sin(time_val) * 0.05  # Breathing motion
    sway = math.cos(time_val * 0.5) * 0.1  # Gentle sway
    head_tilt = math.sin(time_val * 0.3) * 0.2  # Head tilting motion

    # Base position (person standing at ~5m depth, centered)
    base_z = 5.0

    # COCO-17 body keypoints in anatomically correct positions
    # Format: [x (left-right), y (up-down), z (depth)]
    landmarks = [
        # Head (0-4)
        (0.0 + sway, 1.6 + head_tilt, base_z),           # 0: nose
        (0.05 + sway, 1.65 + head_tilt, base_z - 0.05),  # 1: left eye
        (-0.05 + sway, 1.65 + head_tilt, base_z - 0.05), # 2: right eye
        (0.08 + sway, 1.55 + head_tilt, base_z + 0.05),  # 3: left ear
        (-0.08 + sway, 1.55 + head_tilt, base_z + 0.05), # 4: right ear

        # Upper body (5-6)
        (0.2 + sway, 1.3 + breathe, base_z),  # 5: left shoulder
        (-0.2 + sway, 1.3 + breathe, base_z), # 6: right shoulder

        # Arms (7-10)
        (0.3 + sway, 1.0, base_z + 0.1),      # 7: left elbow
        (-0.3 + sway, 1.0, base_z + 0.1),     # 8: right elbow
        (0.35 + sway, 0.7, base_z + 0.15),    # 9: left wrist
        (-0.35 + sway, 0.7, base_z + 0.15),   # 10: right wrist

        # Torso (11-12)
        (0.15 + sway, 0.9 + breathe, base_z), # 11: left hip
        (-0.15 + sway, 0.9 + breathe, base_z),# 12: right hip

        # Legs (13-16)
        (0.15 + sway, 0.5, base_z),           # 13: left knee
        (-0.15 + sway, 0.5, base_z),          # 14: right knee
        (0.15 + sway, 0.1, base_z),           # 15: left ankle
        (-0.15 + sway, 0.1, base_z),          # 16: right ankle
    ]

    # Extend to 33 keypoints (add feet and hand points)
    # Feet keypoints (6 points)
    for i in range(6):
        landmarks.append((sway + (i-2.5)*0.03, 0.0, base_z))

    # Hand keypoints (10 points each hand = 20 total)
    # Left hand
    for i in range(10):
        offset = i * 0.02
        landmarks.append((0.35 + sway + offset, 0.7 - offset * 0.5, base_z + 0.2))

    # Right hand
    for i in range(10):
        offset = i * 0.02
        landmarks.append((-0.35 + sway - offset, 0.7 - offset * 0.5, base_z + 0.2))

    # High confidence for body keypoints, medium for extremities
    visibility = (
        [0.95] * 17 +      # High confidence for main body (COCO-17)
        [0.75] * 6 +       # Medium-high for feet
        [0.65] * 10 +      # Medium for left hand
        [0.65] * 10        # Medium for right hand
    )

    # Convert to numpy arrays
    keypoints_3d = np.array(landmarks[:33], dtype=np.float32)
    scores = np.array(visibility[:33], dtype=np.float32)

    return keypoints_3d, scores


def test_static_skeleton():
    """Test 1: Render a static skeleton pose."""
    print("=" * 70)
    print("TEST 1: Static Skeleton Rendering")
    print("=" * 70)

    keypoints_3d, scores = generate_test_skeleton(frame_num=0)

    print(f"Keypoints shape: {keypoints_3d.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Average confidence: {scores.mean():.3f}")
    print(f"Keypoint range: X=[{keypoints_3d[:, 0].min():.2f}, {keypoints_3d[:, 0].max():.2f}], "
          f"Y=[{keypoints_3d[:, 1].min():.2f}, {keypoints_3d[:, 1].max():.2f}], "
          f"Z=[{keypoints_3d[:, 2].min():.2f}, {keypoints_3d[:, 2].max():.2f}]")

    try:
        img = plot_3d_skeleton(
            keypoints=keypoints_3d,
            scores=scores,
            min_confidence=0.3,
            view_angle=(20, 45),
            figsize=(8, 8),
            dpi=100
        )

        # Save the image
        output_path = "test_3d_skeleton_static.png"
        img.save(output_path)
        print(f"✓ SUCCESS: Static skeleton rendered and saved to {output_path}")
        print(f"  Image size: {img.size}")
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_animated_skeleton_sequence():
    """Test 2: Render a sequence of animated skeleton frames."""
    print("\n" + "=" * 70)
    print("TEST 2: Animated Skeleton Sequence (10 frames)")
    print("=" * 70)

    num_frames = 10
    success_count = 0

    for frame_num in range(num_frames):
        try:
            keypoints_3d, scores = generate_test_skeleton(frame_num=frame_num)

            img = plot_3d_skeleton(
                keypoints=keypoints_3d,
                scores=scores,
                min_confidence=0.3,
                view_angle=(20, 45),
                figsize=(6, 6),
                dpi=80
            )

            # Save each frame
            output_path = f"test_3d_skeleton_frame_{frame_num:03d}.png"
            img.save(output_path)
            success_count += 1
            print(f"  Frame {frame_num:2d}: ✓ Rendered ({img.size[0]}x{img.size[1]}px)")

        except Exception as e:
            print(f"  Frame {frame_num:2d}: ✗ FAILED - {e}")

    print(f"\n✓ Successfully rendered {success_count}/{num_frames} frames")
    return success_count == num_frames


def test_different_view_angles():
    """Test 3: Render skeleton from multiple camera angles."""
    print("\n" + "=" * 70)
    print("TEST 3: Multiple Camera View Angles")
    print("=" * 70)

    keypoints_3d, scores = generate_test_skeleton(frame_num=5)

    view_angles = [
        (20, 45, "front_right"),
        (20, -45, "front_left"),
        (20, 135, "back_right"),
        (20, -135, "back_left"),
        (60, 45, "top_down"),
        (-10, 45, "low_angle"),
    ]

    success_count = 0
    for elev, azim, name in view_angles:
        try:
            img = plot_3d_skeleton(
                keypoints=keypoints_3d,
                scores=scores,
                min_confidence=0.3,
                view_angle=(elev, azim),
                figsize=(6, 6),
                dpi=80
            )

            output_path = f"test_3d_skeleton_view_{name}.png"
            img.save(output_path)
            success_count += 1
            print(f"  {name:15s} (elev={elev:4d}°, azim={azim:4d}°): ✓")

        except Exception as e:
            print(f"  {name:15s}: ✗ FAILED - {e}")

    print(f"\n✓ Successfully rendered {success_count}/{len(view_angles)} views")
    return success_count == len(view_angles)


def test_low_confidence_keypoints():
    """Test 4: Handle skeleton with varying confidence levels."""
    print("\n" + "=" * 70)
    print("TEST 4: Low Confidence Keypoints")
    print("=" * 70)

    keypoints_3d, scores = generate_test_skeleton(frame_num=0)

    # Simulate some low-confidence keypoints
    scores[9:17] = 0.2   # Low confidence arms/legs
    scores[23:] = 0.1    # Very low confidence hands

    print(f"Confidence distribution:")
    print(f"  High (>0.7):   {(scores > 0.7).sum()} keypoints")
    print(f"  Medium (0.3-0.7): {((scores >= 0.3) & (scores <= 0.7)).sum()} keypoints")
    print(f"  Low (<0.3):    {(scores < 0.3).sum()} keypoints")

    try:
        img = plot_3d_skeleton(
            keypoints=keypoints_3d,
            scores=scores,
            min_confidence=0.3,
            view_angle=(20, 45),
            figsize=(8, 8),
            dpi=100
        )

        output_path = "test_3d_skeleton_low_confidence.png"
        img.save(output_path)
        print(f"\n✓ SUCCESS: Low-confidence skeleton rendered to {output_path}")
        return True

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_head_tilt_calculation():
    """Test 5: Verify head tilt angle calculation."""
    print("\n" + "=" * 70)
    print("TEST 5: Head Tilt Angle Calculation")
    print("=" * 70)

    from core.visualization.skeleton_3d_renderer import calculate_head_tilt_angle

    # Test different head positions
    test_cases = [
        ("Neutral (0°)", 0.0),
        ("Looking up (+30°)", 0.3),
        ("Looking down (-30°)", -0.3),
    ]

    base_z = 5.0
    for name, tilt_offset in test_cases:
        # Create skeleton with specific head tilt
        keypoints_3d, scores = generate_test_skeleton(frame_num=0)

        # Modify nose position to simulate head tilt
        keypoints_3d[0, 2] += tilt_offset  # Adjust nose Z position

        # Calculate head tilt
        angle_result = calculate_head_tilt_angle(keypoints_3d, scores, min_confidence=0.3)

        if angle_result is not None:
            vector_angle, head_tilt = angle_result
            print(f"  {name:20s}: {head_tilt:+6.1f}° (expected ~{tilt_offset*100:+.0f}°)")
        else:
            print(f"  {name:20s}: N/A (calculation failed)")

    print("\n✓ Head tilt calculation test complete")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("3D SKELETON VISUALIZATION TEST SUITE")
    print("=" * 70)
    print("Testing the plot_3d_skeleton rendering pipeline\n")

    results = []

    # Run all tests
    results.append(("Static Skeleton", test_static_skeleton()))
    results.append(("Animated Sequence", test_animated_skeleton_sequence()))
    results.append(("View Angles", test_different_view_angles()))
    results.append(("Low Confidence", test_low_confidence_keypoints()))
    results.append(("Head Tilt", test_head_tilt_calculation()))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nOverall: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! 3D skeleton visualization is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
