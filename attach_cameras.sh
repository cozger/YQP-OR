#!/bin/bash
# Helper script to attach USB cameras from WSL2

echo "=== USB Camera Attachment Helper ==="
echo ""

# Check if usbipd.exe is available
if ! command -v usbipd.exe >/dev/null 2>&1; then
    echo "ERROR: usbipd.exe not found. Please install usbipd-win on Windows:"
    echo "https://github.com/dorssel/usbipd-win/releases"
    exit 1
fi

# Show current status
echo "Current USB device status:"
usbipd.exe list
echo ""

# Check for unattached cameras
UNATTACHED_CAMERAS=$(usbipd.exe list | grep -E "Camera|Webcam|Video|Imaging" | grep "Not shared")

if [ -n "$UNATTACHED_CAMERAS" ]; then
    echo "Found unattached cameras that need to be attached:"
    echo "$UNATTACHED_CAMERAS"
    echo ""
    echo "To attach these cameras, you need to run commands with Administrator privileges."
    echo ""
    echo "Option 1: Run the PowerShell script (recommended):"
    echo "  1. Open PowerShell as Administrator on Windows"
    echo "  2. Navigate to: $(wslpath -w $(pwd))"
    echo "  3. Run: .\\attach_cameras.ps1"
    echo ""
    echo "Option 2: Run these commands manually in elevated PowerShell:"
    echo "$UNATTACHED_CAMERAS" | while IFS= read -r line; do
        BUSID=$(echo "$line" | awk '{print $1}')
        echo "  usbipd bind --busid $BUSID"
        echo "  usbipd attach --wsl --busid $BUSID"
    done
    echo ""
    echo "Option 3: Open PowerShell as Admin from here:"
    echo "  Press Y to open PowerShell (requires Windows Terminal)"
    read -n 1 -p "  Open PowerShell as Administrator? (y/N): " response
    echo ""
    if [[ "$response" =~ ^[Yy]$ ]]; then
        # Convert current path to Windows path
        WIN_PATH=$(wslpath -w "$(pwd)/attach_cameras.ps1")
        # Try to launch PowerShell as admin
        powershell.exe -Command "Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -File \"$WIN_PATH\"' -Verb RunAs" 2>/dev/null || {
            echo "Failed to launch PowerShell. Please run manually as described above."
        }
    fi
else
    echo "All cameras appear to be attached or no cameras found."
    echo ""
    echo "Attached cameras:"
    usbipd.exe list | grep -E "Camera|Webcam|Video|Imaging" | grep "Attached"
fi

echo ""
echo "After attaching cameras, run: ./run_gui.sh"