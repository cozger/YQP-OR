#!/usr/bin/env bash
# ============================================================================
# YouQuantiPy MediaPipe - Resource Verification & Cleanup Script
# ============================================================================
# Checks for orphaned resources and optionally cleans them up
#
# Usage:
#   ./verify_resources.sh           - Check for orphaned resources (dry-run)
#   ./verify_resources.sh --fix     - Check and clean up orphaned resources
# ============================================================================

set -euo pipefail

# Parse arguments
FIX_MODE=false
if [[ "${1:-}" == "--fix" ]]; then
    FIX_MODE=true
fi

echo "========================================================================"
echo "YouQuantiPy MediaPipe - Resource Verification"
echo "========================================================================"
echo ""

# Track if we found any issues
ISSUES_FOUND=false

# ============================================================================
# 1. Check for orphaned Python processes (MediaPipe/face_processing)
# ============================================================================
echo "[1] Checking for orphaned Python processes..."
ORPHANED_PROCESSES=$(ps aux | grep -i 'python.*\(mediapipe\|face_processing\|camera_worker\)' | grep -v grep | grep -v "$$" || true)

if [[ -n "$ORPHANED_PROCESSES" ]]; then
    echo "⚠️  Found orphaned Python processes:"
    echo "$ORPHANED_PROCESSES" | awk '{print "   PID " $2 ": " $11 " " $12 " " $13 " " $14}'
    ISSUES_FOUND=true

    if [[ "$FIX_MODE" == true ]]; then
        echo "   🔧 Killing orphaned processes..."
        echo "$ORPHANED_PROCESSES" | awk '{print $2}' | while read pid; do
            # Don't kill ourselves
            if [[ "$pid" != "$$" ]] && [[ "$pid" != "$PPID" ]]; then
                kill -9 "$pid" 2>/dev/null && echo "      Killed PID $pid" || echo "      Failed to kill PID $pid"
            fi
        done
    fi
else
    echo "✅ No orphaned Python processes found"
fi
echo ""

# ============================================================================
# 2. Check for stale shared memory objects in /dev/shm
# ============================================================================
echo "[2] Checking for stale shared memory objects..."
STALE_SHM=$(find /dev/shm -user "$USER" \( -name "psm_*" -o -name "wnsm_*" -o -name "youquantipy_*" \) 2>/dev/null || true)

if [[ -n "$STALE_SHM" ]]; then
    SHM_COUNT=$(echo "$STALE_SHM" | wc -l)
    echo "⚠️  Found $SHM_COUNT stale shared memory object(s):"
    echo "$STALE_SHM" | awk -F/ '{print "   /dev/shm/" $NF}'
    ISSUES_FOUND=true

    if [[ "$FIX_MODE" == true ]]; then
        echo "   🔧 Removing stale shared memory objects..."
        echo "$STALE_SHM" | while read shm_file; do
            rm -f "$shm_file" && echo "      Removed $(basename "$shm_file")" || echo "      Failed to remove $(basename "$shm_file")"
        done
    fi
else
    echo "✅ No stale shared memory objects found"
fi
echo ""

# ============================================================================
# 3. Check for locked camera devices
# ============================================================================
echo "[3] Checking for locked camera devices..."
LOCKED_CAMERAS=false

# Find all video devices
for video_dev in /dev/video*; do
    if [[ ! -e "$video_dev" ]]; then
        continue
    fi

    # Check if any process is using this device
    if command -v fuser &> /dev/null; then
        PROCS=$(fuser "$video_dev" 2>/dev/null || true)
        if [[ -n "$PROCS" ]]; then
            echo "⚠️  Camera device $video_dev is locked by process(es): $PROCS"
            LOCKED_CAMERAS=true
            ISSUES_FOUND=true

            if [[ "$FIX_MODE" == true ]]; then
                echo "   🔧 Killing processes locking $video_dev..."
                for pid in $PROCS; do
                    # Don't kill ourselves
                    if [[ "$pid" != "$$" ]] && [[ "$pid" != "$PPID" ]]; then
                        kill -9 "$pid" 2>/dev/null && echo "      Killed PID $pid" || echo "      Failed to kill PID $pid"
                    fi
                done
            fi
        fi
    else
        echo "⚠️  'fuser' command not found (install psmisc package)"
        echo "   Cannot check for locked camera devices"
        break
    fi
done

if [[ "$LOCKED_CAMERAS" == false ]]; then
    echo "✅ No locked camera devices found"
fi
echo ""

# ============================================================================
# 4. Summary
# ============================================================================
echo "========================================================================"
if [[ "$ISSUES_FOUND" == false ]]; then
    echo "✅ All clear! No orphaned resources found."
else
    if [[ "$FIX_MODE" == true ]]; then
        echo "🔧 Cleanup completed. Please verify by running:"
        echo "   ./verify_resources.sh"
    else
        echo "⚠️  Orphaned resources detected!"
        echo ""
        echo "To clean up these resources, run:"
        echo "   ./verify_resources.sh --fix"
    fi
fi
echo "========================================================================"
echo ""

# Exit with appropriate code
if [[ "$ISSUES_FOUND" == true ]] && [[ "$FIX_MODE" == false ]]; then
    exit 1
else
    exit 0
fi
