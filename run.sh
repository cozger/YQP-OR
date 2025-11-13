#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# YouQuantiPy MediaPipe - Unified Run Script (stable venv + robust detection)
# ============================================================================

# ----------------------------------------------------------------------------
# Command-line options
# ----------------------------------------------------------------------------
QUIET_MODE=true       # default: suppress noisy logs
SKIP_CAMERAS=false
FORCE_RUN=false       # bypass display check (not recommended)
USE_ZMQ_BRIDGE=false  # use ZMQ camera bridge instead of V4L2
VERIFY_CUDA=false     # run CUDA verification before launch

for arg in "$@"; do
  case $arg in
    -v|--verbose) QUIET_MODE=false; shift ;;
    --skip-cameras) SKIP_CAMERAS=true; shift ;;
    --force) FORCE_RUN=true; shift ;;
    --zmq|--zmq-bridge) USE_ZMQ_BRIDGE=true; SKIP_CAMERAS=true; shift ;;
    --verify-cuda) VERIFY_CUDA=true; shift ;;
    -h|--help)
      cat <<'EOF'
YouQuantiPy MediaPipe - Unified Run Script

Usage:
  ./run.sh [OPTIONS]

Options:
  -v, --verbose       Enable verbose MediaPipe/TensorFlow logging (default: quiet)
  --skip-cameras      Skip USB camera attachment prompt
  --zmq-bridge        Use ZMQ camera bridge from Windows (skips USB setup)
  --verify-cuda       Run CUDA verification before launching
  --force             Bypass display server check (not recommended)
  -h, --help          Show this help message
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Use -h or --help for usage information"
      exit 1
      ;;
  esac
done

# ----------------------------------------------------------------------------
# Project root & venv discovery/activation
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"

find_project_root() {
  local d="$1"
  while [ "$d" != "/" ]; do
    if [ -f "$d/venv/bin/activate" ]; then
      echo "$d"
      return 0
    fi
    d="$(dirname "$d")"
  done
  return 1
}

# Optional override: export VENV_PATH=/abs/path/to/venv/bin
if [[ -n "${VENV_PATH:-}" ]]; then
  if [ -f "$VENV_PATH/activate" ]; then
    PROJECT_ROOT="$(cd "$(dirname "$VENV_PATH")/.." && pwd -P)"
  else
    echo "❌ VENV_PATH is set but $VENV_PATH/activate doesn't exist."
    exit 1
  fi
else
  PROJECT_ROOT="$(find_project_root "$SCRIPT_DIR")" || {
    echo "❌ Couldn't find project root with venv/ above $SCRIPT_DIR"
    echo "   Expected: venv/bin/activate under /home/canoz/Projects/surgery"
    exit 1
  }
fi

# shellcheck source=/dev/null
source "$PROJECT_ROOT/venv/bin/activate"
echo "✅ Activated venv at: $PROJECT_ROOT/venv"
echo ""

# ----------------------------------------------------------------------------
# WSLg warm-up helpers (X server & window manager)
# ----------------------------------------------------------------------------
wslg_warmup_x11() {
  local max_attempts=3 attempt=1
  echo "  Initializing WSLg X server (cold start detection)..."
  while [ $attempt -le $max_attempts ]; do
    if timeout 2s xset q >/dev/null 2>&1; then
      echo "  ✅ X server ready (attempt $attempt/$max_attempts)"; return 0
    fi
    if timeout 2s xdpyinfo >/dev/null 2>&1; then
      echo "  ✅ X server ready via xdpyinfo (attempt $attempt/$max_attempts)"; return 0
    fi
    if [ $attempt -lt $max_attempts ]; then
      local delay=$((2 ** (attempt - 1)))
      echo "  ⏳ X server initializing, retry in ${delay}s... (attempt $attempt/$max_attempts)"
      sleep "$delay"
    fi
    ((attempt++))
  done
  echo "  ⚠️  WARNING: X server warm-up failed after $max_attempts attempts"
  return 1
}

wslg_warmup_window_manager() {
  local max_attempts=5 attempt=1
  echo "  Waiting for window manager (decoration system)..."
  while [ $attempt -le $max_attempts ]; do
    if xprop -root _NET_SUPPORTING_WM_CHECK >/dev/null 2>&1; then
      echo "  ✅ Window manager detected (attempt $attempt/$max_attempts)"
      sleep 1
      echo "  ✅ Window manager decorations ready"
      return 0
    fi
    if command -v wmctrl >/dev/null 2>&1 && wmctrl -m >/dev/null 2>&1; then
      echo "  ✅ Window manager detected via wmctrl (attempt $attempt/$max_attempts)"
      sleep 1
      echo "  ✅ Window manager decorations ready"
      return 0
    fi
    if [ $attempt -lt $max_attempts ]; then
      # Exponential backoff: 0.5, 1, 2, 4, 8
      local delay
      delay=$(awk "BEGIN {print 0.5 * (2 ^ ($attempt - 1))}")
      echo "  ⏳ WM initializing, retry in ${delay}s... (attempt $attempt/$max_attempts)"
      sleep "$delay"
    fi
    ((attempt++))
  done
  echo "  ⚠️  WARNING: Window manager warm-up incomplete after $max_attempts attempts"
  return 1
}

# ----------------------------------------------------------------------------
# Platform detection (ALWAYS exit 0 and print assignments)
# ----------------------------------------------------------------------------
echo "=== Platform Detection ==="
PLATFORM_MODE=$(
python3 - "$PROJECT_ROOT" <<'PYCODE'
import json, sys, os
project_root = sys.argv[1] if len(sys.argv) > 1 else ""
# Defaults
mode="auto"; cam="v4l2_native"; gpu="opengl"; disp="wslg"
try:
    if project_root:
        # Legacy config filename (youquantipy_config.json) still in use
        cfg_path = os.path.join(project_root, 'mmpose_3d_gui', 'youquantipy_config.json')
        try:
            with open(cfg_path, 'r') as f:
                config = json.load(f)
        except Exception:
            config = {}
        sys.path.insert(0, project_root)
        try:
            from core.platform_detection import PlatformContext  # type: ignore
            ctx = PlatformContext.detect(config)
            mode = ctx.get('mode', mode)
            cam  = ctx.get('camera_discovery', cam)
            gpu  = ctx.get('gpu_backend', gpu)
            disp = ctx.get('display_mode', disp)
        except Exception:
            pass
except Exception:
    pass

print(f"MODE={mode}")
print(f"CAMERA_DISCOVERY={cam}")
print(f"GPU_BACKEND={gpu}")
print(f"DISPLAY_MODE={disp}")
PYCODE
)
# Make sure eval never crashes on empty
if [[ -z "${PLATFORM_MODE// }" ]]; then
  PLATFORM_MODE=$'MODE=auto\nCAMERA_DISCOVERY=v4l2_native\nGPU_BACKEND=opengl\nDISPLAY_MODE=wslg'
fi
eval "$PLATFORM_MODE"

: "${MODE:=auto}"
: "${CAMERA_DISCOVERY:=v4l2_native}"
: "${GPU_BACKEND:=opengl}"
: "${DISPLAY_MODE:=wslg}"

echo "  Platform Mode: ${MODE}"
echo "  Camera Discovery: ${CAMERA_DISCOVERY}"
echo "  GPU Backend: ${GPU_BACKEND}"
echo "  Display Mode: ${DISPLAY_MODE}"
echo ""

# ----------------------------------------------------------------------------
# Display validation (platform-aware)
# ----------------------------------------------------------------------------
if [ "$FORCE_RUN" = false ]; then
  case $DISPLAY_MODE in
    wslg)
      echo "=== Display Validation: WSLg Mode ==="
      if [[ -z "${WAYLAND_DISPLAY:-}" && -z "${DISPLAY:-}" ]]; then
        echo "❌ ERROR: No display server detected (WSLg or X11)."
        echo "   Launch from Windows Terminal → Ubuntu profile, or use --force."
        exit 1
      elif [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
        echo "✅ WSLg Wayland session detected (WAYLAND_DISPLAY=${WAYLAND_DISPLAY})"
        if [[ -z "${DISPLAY:-}" ]]; then
          echo "⚠️  DISPLAY not set - Tkinter requires X11 backend"
          export DISPLAY=:0
          if [[ -S "/tmp/.X11-unix/X0" || -S "/mnt/wslg/.X11-unix/X0" ]]; then
            echo "✅ X11 socket found (XWayland ready)"
          else
            echo "❌ WARNING: X11 socket not found (XWayland)!"
          fi
        else
          echo "✅ X11 display configured (DISPLAY=${DISPLAY})"
        fi
        echo ""
        wslg_warmup_x11 || true
        wslg_warmup_window_manager || true
        echo ""
      fi
      ;;
    x11)
      echo "=== Display Validation: X11 Mode ==="
      if [[ -z "${DISPLAY:-}" ]]; then
        echo "❌ ERROR: DISPLAY not set (X11)"
        echo "   export DISPLAY=:0   # or your X server display"
        exit 1
      else
        echo "✅ X11 display detected (DISPLAY=${DISPLAY})"
      fi
      ;;
    wayland)
      echo "=== Display Validation: Wayland Mode ==="
      if [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
        echo "❌ ERROR: WAYLAND_DISPLAY not set"
        exit 1
      else
        echo "✅ Wayland display detected (WAYLAND_DISPLAY=${WAYLAND_DISPLAY})"
      fi
      ;;
    *)
      echo "⚠️  Unknown display mode: $DISPLAY_MODE, skipping validation"
      ;;
  esac
else
  echo "⚠️  WARNING: Display check bypassed (--force)"
fi

# ----------------------------------------------------------------------------
# Camera setup (platform-aware)
# ----------------------------------------------------------------------------
if [ "$SKIP_CAMERAS" = false ]; then
  echo ""
  echo "======================================================================"
  echo "=== Camera Setup: $CAMERA_DISCOVERY Mode ==="
  echo "======================================================================"
  echo ""
  case $CAMERA_DISCOVERY in
    zmq)
      echo "ZMQ Camera Bridge enabled - cameras will stream over network"
      echo "ZMQ cameras will be auto-discovered during GUI initialization."
      echo ""
      ;;
    usbipd)
      echo "WSL2 USB passthrough mode - attaching cameras via usbipd"
      echo ""
      python "$PROJECT_ROOT/main/cam_sweep.py" --attach-only
      ;;
    v4l2_native)
      echo "Native Linux V4L2 mode - cameras available at /dev/video*"
      echo "Available camera devices:"
      if command -v v4l2-ctl >/dev/null 2>&1; then
        v4l2-ctl --list-devices 2>/dev/null || ls -la /dev/video* 2>/dev/null || echo "  (No cameras found)"
      else
        ls -la /dev/video* 2>/dev/null || echo "  (No cameras found)"
      fi
      echo ""
      echo "Cameras will be auto-detected during GUI initialization."
      echo ""
      ;;
    *)
      echo "⚠️  Unknown camera discovery mode: $CAMERA_DISCOVERY"
      echo "   Cameras may not be available."
      echo ""
      ;;
  esac
else
  echo ""
  echo "=== Skipping Camera Setup (--skip-cameras) ==="
  echo ""
fi

# ----------------------------------------------------------------------------
# GPU backend env (platform-aware)
# ----------------------------------------------------------------------------
case $GPU_BACKEND in
  d3d12)
    echo "=== GPU Backend: D3D12 (WSL2 Translation Layer) ==="
    export LIBGL_DRIVERS_PATH=/usr/lib/wsl/lib
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
    export MESA_LOADER_DRIVER_OVERRIDE=d3d12
    export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    unset LIBGL_ALWAYS_SOFTWARE 2>/dev/null || true
    echo "  Mesa D3D12 backend configured for WSL2"
    echo ""
    ;;
  opengl)
    echo "=== GPU Backend: Native OpenGL ==="
    unset MESA_LOADER_DRIVER_OVERRIDE 2>/dev/null || true
    unset LIBGL_ALWAYS_SOFTWARE 2>/dev/null || true
    export LIBGL_ALWAYS_INDIRECT=0
    echo "  Native OpenGL configured for direct GPU access"
    echo ""
    ;;
  *)
    echo "⚠️  Unknown GPU backend: $GPU_BACKEND"
    echo "   GPU acceleration may not work correctly"
    echo ""
    ;;
esac

# ----------------------------------------------------------------------------
# CUDA/cuDNN library paths (PyTorch bundled + system fallback)
# ----------------------------------------------------------------------------
echo "=== CUDA/cuDNN Library Setup ==="

# PRIORITY 1: PyTorch bundled CUDA libraries (CUDA 12.1)
# These must come FIRST to avoid conflicts with system CUDA 12.0
PYTORCH_CUDA_LIBS=$(python -c "import site; import os; sp = site.getsitepackages()[0]; nvidia_libs = [os.path.join(sp, 'nvidia', d, 'lib') for d in ['nvjitlink', 'cusparse', 'cublas', 'cudnn', 'cuda_runtime', 'cuda_nvrtc', 'cusolver', 'cufft', 'curand'] if os.path.exists(os.path.join(sp, 'nvidia', d, 'lib'))]; print(':'.join(nvidia_libs))" 2>/dev/null || echo "")

if [[ -n "$PYTORCH_CUDA_LIBS" ]]; then
  export LD_LIBRARY_PATH="$PYTORCH_CUDA_LIBS:${LD_LIBRARY_PATH:-}"
  echo "  ✅ Added PyTorch bundled CUDA 12.1 libraries (takes precedence)"
else
  echo "  ⚠️  PyTorch bundled CUDA libraries not found"
fi

# PRIORITY 2: System CUDA libraries (fallback for non-PyTorch components)
CUDA_PATHS="/usr/lib/x86_64-linux-gnu"
for cuda_path in $CUDA_PATHS; do
  if [[ ":${LD_LIBRARY_PATH:-}:" != *":$cuda_path:"* ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$cuda_path"
    echo "  Added $cuda_path to LD_LIBRARY_PATH (system fallback)"
  fi
done

# PRIORITY 3: WSL2 NVIDIA driver libraries
if [[ "${MODE}" == "wsl2" && ":${LD_LIBRARY_PATH:-}:" != *":/usr/lib/wsl/lib:"* ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib/wsl/lib"
  echo "  Added /usr/lib/wsl/lib for WSL2 NVIDIA drivers"
fi

echo "  Final LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<empty>}"
echo ""

# ----------------------------------------------------------------------------
# Optional CUDA verification
# ----------------------------------------------------------------------------
if [ "$VERIFY_CUDA" = true ]; then
  echo ""
  echo "======================================================================"
  echo "=== CUDA Verification ==="
  echo "======================================================================"
  echo ""
  if [ -f "$SCRIPT_DIR/verify_cuda_runtime.sh" ]; then
    "$SCRIPT_DIR/verify_cuda_runtime.sh" || true
  elif [ -f "$PROJECT_ROOT/verify_cuda_runtime.sh" ]; then
    "$PROJECT_ROOT/verify_cuda_runtime.sh" || true
  else
    echo "⚠️  CUDA verification script not found"
  fi
  echo ""
fi

# ----------------------------------------------------------------------------
# Logging configuration (quiet vs verbose)
# ----------------------------------------------------------------------------
if [ "$QUIET_MODE" = true ]; then
  echo ""
  echo "=== Running in QUIET mode (MediaPipe logs suppressed) ==="
  export GLOG_minloglevel=3
  export TF_CPP_MIN_LOG_LEVEL=3
  export GRPC_VERBOSITY=ERROR
else
  echo ""
  echo "=== Running in NORMAL mode (verbose MediaPipe logs) ==="
  export GLOG_logtostderr=1
  export TF_CPP_MIN_LOG_LEVEL=0
fi

# ----------------------------------------------------------------------------
# Launch application
# ----------------------------------------------------------------------------
echo ""
echo "Starting MMPose 3D GUI (Single-Camera Mode)..."
echo "================================================================"
cd "$SCRIPT_DIR" || exit 1  # Stay in mmpose_3d_gui/
python gui.py
