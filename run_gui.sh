#!/usr/bin/env bash
set -euo pipefail

# --- REQUIRE a WSLg GUI session (prevents headless llvmpipe) ---
if [[ -z "${WAYLAND_DISPLAY:-}" || "${XDG_SESSION_TYPE:-}" != "wayland" ]]; then
  echo "ERROR: No WSLg Wayland session detected (WAYLAND_DISPLAY/XDG_SESSION_TYPE)."
  echo "Launch Ubuntu from Windows Terminal/Start menu (not wsl.exe -e) and try again."
  exit 1
fi

# --- Fresh Camera Discovery for USB devices ---
echo "=== USB Camera Setup for WSL2 ==="

# Check for usbipd.exe on Windows side
if command -v usbipd.exe >/dev/null 2>&1; then
  echo "Checking USB cameras from Windows..."

  # List all USB devices
  echo "Current USB devices:"
  usbipd.exe list

  # Auto-attach any USB cameras that are not already attached
  echo ""
  echo "Checking for unattached cameras..."

  # Get list of unattached camera devices (including "Not shared" and "Shared" but not attached)
  UNATTACHED_CAMERAS=$(usbipd.exe list | grep -E "Camera|Webcam|Video|Imaging" | grep -v "Attached")

  if [ -n "$UNATTACHED_CAMERAS" ]; then
    echo "❌ Found cameras that need to be attached:"
    echo "$UNATTACHED_CAMERAS"
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "IMPORTANT: These cameras must be attached before starting the GUI!"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""

    # Extract BUSIDs for processing
    BUSIDS_TO_ATTACH=""
    echo "$UNATTACHED_CAMERAS" | while IFS= read -r line; do
      BUSID=$(echo "$line" | awk '{print $1}')
      STATE=$(echo "$line" | grep -o "Not shared\|Shared" || echo "Unknown")
      BUSIDS_TO_ATTACH="$BUSIDS_TO_ATTACH $BUSID"
      echo "  Camera BUSID: $BUSID (Status: $STATE)"
    done
    echo ""

    # Create a batch file that's easier to execute from Windows
    TEMP_BAT="/tmp/attach_cameras_$$.bat"
    cat > "$TEMP_BAT" << 'EOF'
@echo off
echo === Attaching USB Cameras to WSL2 ===
echo.
EOF

    # Add commands for each camera
    echo "$UNATTACHED_CAMERAS" | while IFS= read -r line; do
      BUSID=$(echo "$line" | awk '{print $1}')
      STATE=$(echo "$line" | grep -o "Not shared\|Shared" || echo "Unknown")

      echo "echo Attaching camera $BUSID..." >> "$TEMP_BAT"
      if [[ "$STATE" == "Not shared" ]]; then
        echo "usbipd bind --busid $BUSID" >> "$TEMP_BAT"
        echo "timeout /t 1 /nobreak >nul" >> "$TEMP_BAT"
      fi
      echo "usbipd attach --wsl --busid $BUSID" >> "$TEMP_BAT"
      echo "if %ERRORLEVEL% EQU 0 (echo   [OK] Camera $BUSID attached) else (echo   [FAIL] Could not attach $BUSID)" >> "$TEMP_BAT"
      echo "echo." >> "$TEMP_BAT"
    done

    cat >> "$TEMP_BAT" << 'EOF'
echo === Attachment Complete ===
pause
EOF

    # Convert to Windows path
    WIN_BAT=$(wslpath -w "$TEMP_BAT")

    echo "Choose how to attach cameras:"
    echo ""
    echo "  [A] Auto-attach (opens Admin PowerShell) - RECOMMENDED"
    echo "  [M] Manual commands (show commands to copy/paste)"
    echo "  [S] Skip (continue without new cameras)"
    echo ""
    read -n 1 -p "Your choice [A/M/S]: " choice
    echo ""
    echo ""

    case "$choice" in
      [Aa])
        echo "Opening elevated command prompt..."
        echo "Please approve the Administrator request when it appears!"
        echo ""

        # Use cmd.exe to launch elevated prompt with the batch file
        cmd.exe /c "powershell -Command \"Start-Process cmd -ArgumentList '/c', '\\\"$WIN_BAT\\\"' -Verb RunAs\"" 2>/dev/null

        echo "Waiting for camera attachment..."
        echo "After the cameras are attached in the Admin window, press Enter here to continue"
        read -p "Press Enter when done..."

        # Verify attachment
        echo ""
        echo "Verifying camera attachment..."
        STILL_UNATTACHED=$(usbipd.exe list | grep -E "Camera|Webcam|Video|Imaging" | grep -v "Attached")
        if [ -z "$STILL_UNATTACHED" ]; then
          echo "✅ All cameras successfully attached!"
        else
          echo "⚠️ Some cameras may still be unattached:"
          echo "$STILL_UNATTACHED"
          echo ""
          read -n 1 -p "Continue anyway? [Y/N]: " cont
          echo ""
          if [[ ! "$cont" =~ ^[Yy]$ ]]; then
            echo "Exiting. Please attach cameras and try again."
            rm -f "$TEMP_BAT" 2>/dev/null
            exit 1
          fi
        fi
        ;;

      [Mm])
        echo "Run these commands in an elevated PowerShell or CMD:"
        echo ""
        echo "$UNATTACHED_CAMERAS" | while IFS= read -r line; do
          BUSID=$(echo "$line" | awk '{print $1}')
          STATE=$(echo "$line" | grep -o "Not shared\|Shared" || echo "Unknown")
          if [[ "$STATE" == "Not shared" ]]; then
            echo "usbipd bind --busid $BUSID"
          fi
          echo "usbipd attach --wsl --busid $BUSID"
        done
        echo ""
        read -p "Press Enter after you've attached the cameras..."
        ;;

      [Ss])
        echo "Skipping camera attachment. Some cameras won't be available."
        ;;

      *)
        echo "Invalid choice. Continuing without new cameras."
        ;;
    esac

    # Clean up temp file
    rm -f "$TEMP_BAT" 2>/dev/null
  else
    echo "✅ All camera devices are already attached!"
  fi
else
  echo "Warning: usbipd.exe not found. USB passthrough may not be configured."
  echo "Install from: https://github.com/dorssel/usbipd-win/releases"
fi

echo ""
echo "Triggering device discovery in WSL2..."

# Trigger udev to re-enumerate video devices (requires sudo)
if command -v udevadm >/dev/null 2>&1; then
  # Try without sudo first, fall back to sudo if needed
  udevadm trigger --subsystem-match=video4linux 2>/dev/null || \
    sudo udevadm trigger --subsystem-match=video4linux 2>/dev/null || \
    echo "Warning: Could not trigger udev (may need sudo)"

  # Wait for udev events to settle
  udevadm settle --timeout=2 2>/dev/null || true
fi

# List available video devices for debugging
if command -v v4l2-ctl >/dev/null 2>&1; then
  echo "Available video devices:"
  v4l2-ctl --list-devices 2>/dev/null || echo "  (v4l2-ctl not available or no devices found)"
else
  # Fallback to listing /dev/video* if v4l2-ctl not available
  echo "Video device nodes:"
  ls -la /dev/video* 2>/dev/null || echo "  (No video devices found)"
fi

# Small delay to ensure devices are fully initialized
echo "Waiting for device initialization..."
sleep 1

# --- Make sure Mesa picks the D3D12 driver path and the NVIDIA adapter ---
export LIBGL_DRIVERS_PATH=/usr/lib/wsl/lib
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
export MESA_LOADER_DRIVER_OVERRIDE=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
unset LIBGL_ALWAYS_SOFTWARE 2>/dev/null || true

# --- (Optional) Loud logs so you can SEE GPU delegation in the console ---
export GLOG_logtostderr=1
export TF_CPP_MIN_LOG_LEVEL=0

# --- Activate your venv and run your app ---
# adjust this path to your venv
source ./venv/bin/activate
python gui.py
