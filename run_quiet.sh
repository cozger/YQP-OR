#!/usr/bin/env bash
# Run YouQuantiPy with suppressed MediaPipe verbose logging
# This ensures environment variables are set BEFORE Python starts

# Suppress MediaPipe/GLOG verbose output
export GLOG_minloglevel=3  # Only show FATAL errors
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow
export GRPC_VERBOSITY=ERROR  # Suppress gRPC if used

# Optional: Force specific camera backend (uncomment one)
# export OPENCV_VIDEOIO_PRIORITY_V4L2=9999  # Force V4L2 on Linux
# export OPENCV_VIDEOIO_PRIORITY_MSMF=9999  # Force MSMF on Windows
# export OPENCV_VIDEOIO_PRIORITY_DSHOW=9999  # Force DirectShow on Windows

# Activate virtual environment (go up one directory to find venv)
cd "$(dirname "$0")/.." || exit 1
source venv/bin/activate

# Run the main application (now we're in the project root, so cd to main)
echo "Starting YouQuantiPy with suppressed logging..."
echo "To see which camera backend is being used, check the startup logs"
echo "================================================================"
cd main || exit 1
python gui.py "$@"