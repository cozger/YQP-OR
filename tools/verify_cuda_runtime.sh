#!/bin/bash
#
# CUDA Runtime Verification Script for YouQuantiPy
# Checks if CUDA is properly configured before application launch
#

set -euo pipefail

echo "======================================================================"
echo "CUDA Runtime Verification for YouQuantiPy"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
CUDA_STATUS="OK"

# Function to check command availability
check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✅${NC} $2"
        return 0
    else
        echo -e "${RED}❌${NC} $2"
        return 1
    fi
}

# Function to check library
check_library() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅${NC} $2"
        return 0
    else
        echo -e "${RED}❌${NC} $2"
        return 1
    fi
}

echo "1. Checking System Environment"
echo "------------------------------"

# Check if in WSL2
if [ -f "/proc/sys/fs/binfmt_misc/WSLInterop" ]; then
    echo -e "${GREEN}✅${NC} Running in WSL2 environment"
    IS_WSL=true
else
    echo -e "${GREEN}✅${NC} Running in native Linux environment"
    IS_WSL=false
fi

echo ""
echo "2. Checking NVIDIA Driver"
echo "-------------------------"

# Check nvidia-smi
if check_command "nvidia-smi" "nvidia-smi is available"; then
    # Get GPU info
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")

    echo "   GPU: $GPU_NAME"
    echo "   Memory: $GPU_MEMORY"
    echo "   Driver: $DRIVER_VERSION"
else
    echo -e "${RED}❌${NC} nvidia-smi not found - GPU acceleration will not work"
    CUDA_STATUS="FAIL"
fi

echo ""
echo "3. Checking CUDA Libraries"
echo "--------------------------"

# Check essential CUDA libraries
CUDA_LIBS_OK=true

# Check CUDA runtime
if ls /usr/lib/x86_64-linux-gnu/libcudart.so* &>/dev/null; then
    echo -e "${GREEN}✅${NC} CUDA runtime libraries found"
    CUDA_VERSION=$(ls /usr/lib/x86_64-linux-gnu/libcudart.so.* 2>/dev/null | grep -oP 'libcudart\.so\.\K[0-9]+' | head -1 || echo "Unknown")
    echo "   CUDA version: $CUDA_VERSION"
else
    echo -e "${RED}❌${NC} CUDA runtime libraries not found"
    CUDA_LIBS_OK=false
    CUDA_STATUS="FAIL"
fi

# Check cuDNN
if ls /usr/lib/x86_64-linux-gnu/libcudnn.so* &>/dev/null; then
    echo -e "${GREEN}✅${NC} cuDNN libraries found"
    CUDNN_VERSION=$(ls /usr/lib/x86_64-linux-gnu/libcudnn.so.* 2>/dev/null | grep -oP 'libcudnn\.so\.\K[0-9]+' | head -1 || echo "Unknown")
    echo "   cuDNN version: $CUDNN_VERSION"
else
    echo -e "${YELLOW}⚠️${NC} cuDNN libraries not found (face recognition will use CPU)"
    CUDA_LIBS_OK=false
fi

# Check WSL2 CUDA driver
if [ "$IS_WSL" = true ]; then
    if check_library "/usr/lib/wsl/lib/libcuda.so" "WSL2 CUDA driver found"; then
        :
    else
        echo -e "${RED}❌${NC} WSL2 CUDA driver not found"
        CUDA_STATUS="FAIL"
    fi
fi

echo ""
echo "4. Checking Environment Variables"
echo "----------------------------------"

# Check LD_LIBRARY_PATH
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    echo -e "${GREEN}✅${NC} LD_LIBRARY_PATH is set"

    # Check if required paths are in LD_LIBRARY_PATH
    if echo "$LD_LIBRARY_PATH" | grep -q "/usr/lib/x86_64-linux-gnu"; then
        echo -e "${GREEN}✅${NC} CUDA library path in LD_LIBRARY_PATH"
    else
        echo -e "${YELLOW}⚠️${NC} /usr/lib/x86_64-linux-gnu not in LD_LIBRARY_PATH"
        echo "   Current: ${LD_LIBRARY_PATH:0:100}..."
    fi

    if [ "$IS_WSL" = true ]; then
        if echo "$LD_LIBRARY_PATH" | grep -q "/usr/lib/wsl/lib"; then
            echo -e "${GREEN}✅${NC} WSL2 driver path in LD_LIBRARY_PATH"
        else
            echo -e "${YELLOW}⚠️${NC} /usr/lib/wsl/lib not in LD_LIBRARY_PATH"
        fi
    fi
else
    echo -e "${YELLOW}⚠️${NC} LD_LIBRARY_PATH not set"
    echo "   Will be set by run.sh script"
fi

echo ""
echo "5. Checking Python CUDA Support"
echo "--------------------------------"

# Activate virtual environment
cd "$(dirname "$0")/.." || exit 1
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✅${NC} Virtual environment activated"
else
    echo -e "${RED}❌${NC} Virtual environment not found"
    CUDA_STATUS="FAIL"
fi

# Test ONNX Runtime CUDA
echo ""
echo "Testing ONNX Runtime CUDA..."
python3 -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up environment if not already set
if '/usr/lib/x86_64-linux-gnu' not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        print('   ✅ ONNX CUDA provider available')
        print('   Face recognition will use GPU (~8ms latency)')
        exit(0)
    else:
        print('   ❌ ONNX CUDA provider NOT available')
        print('   Face recognition will use CPU (~200ms latency)')
        print(f'   Available providers: {providers[:3]}')
        exit(1)
except ImportError:
    print('   ❌ onnxruntime not installed')
    exit(1)
except Exception as e:
    print(f'   ❌ Error: {e}')
    exit(1)
" || CUDA_STATUS="PARTIAL"

# Test OpenCV CUDA
echo ""
echo "Testing OpenCV CUDA..."
python3 -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import cv2
    if hasattr(cv2, 'cuda'):
        devices = cv2.cuda.getCudaEnabledDeviceCount()
        if devices > 0:
            print(f'   ✅ OpenCV CUDA: {devices} device(s) detected')
            print('   GUI operations will be GPU-accelerated')
        else:
            print('   ⚠️  OpenCV CUDA: No devices detected')
            print('   Note: This is expected in spawned processes')
    else:
        print('   ⚠️  OpenCV not built with CUDA support')
except Exception as e:
    print(f'   ❌ Error: {e}')
" || true  # Don't fail on OpenCV issues

echo ""
echo "======================================================================"
echo "VERIFICATION SUMMARY"
echo "======================================================================"

if [ "$CUDA_STATUS" = "OK" ]; then
    echo -e "${GREEN}✅ CUDA is properly configured for YouQuantiPy${NC}"
    echo ""
    echo "Expected performance:"
    echo "  • Face Recognition: ~8ms per face (GPU)"
    echo "  • Target FPS: 30+ with multiple cameras"
    echo ""
    echo "Ready to run: cd main && ./run.sh"
elif [ "$CUDA_STATUS" = "PARTIAL" ]; then
    echo -e "${YELLOW}⚠️  CUDA partially configured${NC}"
    echo ""
    echo "Some CUDA features may not work. Check the errors above."
    echo "The application will still run but with reduced performance."
else
    echo -e "${RED}❌ CUDA is not properly configured${NC}"
    echo ""
    echo "Please address the issues above before running the application."
    echo "The application will fall back to CPU (slow performance)."
fi

echo "======================================================================"

# Exit with appropriate code
if [ "$CUDA_STATUS" = "OK" ]; then
    exit 0
elif [ "$CUDA_STATUS" = "PARTIAL" ]; then
    exit 1
else
    exit 2
fi