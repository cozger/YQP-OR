#!/usr/bin/env python3
"""Simple test to verify MMPose models can be loaded."""

import sys
import os
import json
from pathlib import Path
import torch

# Add rtmpose3d module to Python path
rtmpose3d_path = "/home/canoz/Projects/surgery/mmpose/projects/rtmpose3d"
if rtmpose3d_path not in sys.path:
    sys.path.insert(0, rtmpose3d_path)
    os.environ['PYTHONPATH'] = rtmpose3d_path + ':' + os.environ.get('PYTHONPATH', '')

print("=" * 80)
print("Testing MMPose Model Loading")
print("=" * 80)

# Check CUDA
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load config
config_path = Path(__file__).parent / "mmpose_config.json"
with open(config_path) as f:
    config = json.load(f)

mmpose_config = config['mmpose_3d_pipeline']

detector_config = mmpose_config['person_detector']['config']
detector_checkpoint = mmpose_config['person_detector']['checkpoint']
pose_config = mmpose_config['pose_estimator']['config']
pose_checkpoint = mmpose_config['pose_estimator']['checkpoint']

print(f"\nDetector config: {detector_config}")
print(f"Detector checkpoint: {detector_checkpoint}")
print(f"Pose config: {pose_config}")
print(f"Pose checkpoint: {pose_checkpoint}")

# Try importing MMPose
print("\n" + "=" * 80)
print("Importing MMPose modules...")
print("=" * 80)

try:
    from mmpose.apis import init_model, inference_topdown
    from mmdet.apis import init_detector, inference_detector
    print("✅ MMPose modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import MMPose: {e}")
    sys.exit(1)

# Load detector
print("\n" + "=" * 80)
print("Loading RTMDet person detector...")
print("=" * 80)

try:
    detector = init_detector(
        detector_config,
        detector_checkpoint,
        device='cuda:0'
    )
    print("✅ RTMDet detector loaded successfully")
except Exception as e:
    print(f"❌ Failed to load detector: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load pose estimator
print("\n" + "=" * 80)
print("Loading RTMW3D pose estimator...")
print("=" * 80)

try:
    pose_estimator = init_model(
        pose_config,
        pose_checkpoint,
        device='cuda:0'
    )
    print("✅ RTMW3D pose estimator loaded successfully")
except Exception as e:
    print(f"❌ Failed to load pose estimator: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL MODELS LOADED SUCCESSFULLY")
print("=" * 80)

# Check GPU memory usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"\nGPU Memory Usage:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
