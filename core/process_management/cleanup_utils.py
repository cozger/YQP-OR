"""
Defensive cleanup utilities for handling orphaned processes and resources.

This module provides functions to clean up resources that may be left behind
after abrupt program termination (crash, kill -9, terminal force-close).
"""

import os
import logging
import subprocess
import glob
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger('CleanupUtils')


def kill_orphaned_processes() -> int:
    """
    Find and kill orphaned Python processes running MediaPipe or face detection.

    Returns:
        Number of processes killed
    """
    killed_count = 0

    # Pattern to match: python processes with 'mediapipe' or 'face_processing' in command line
    # Exclude the current process and grep itself
    current_pid = os.getpid()

    try:
        # Find processes matching our patterns
        cmd = [
            'ps', 'aux'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

        if result.returncode != 0:
            logger.warning(f"Failed to list processes: {result.stderr}")
            return 0

        # Parse ps output
        lines = result.stdout.strip().split('\n')
        for line in lines[1:]:  # Skip header
            # Look for python processes with mediapipe/face_processing patterns
            if 'python' in line.lower() and any(pattern in line.lower() for pattern in ['mediapipe', 'face_processing', 'camera_worker']):
                parts = line.split()
                if len(parts) < 2:
                    continue

                try:
                    pid = int(parts[1])

                    # Don't kill ourselves or parent GUI process
                    if pid == current_pid or pid == os.getppid():
                        continue

                    # Check if this is actually an orphaned subprocess
                    # (parent process no longer exists or is init/systemd)
                    try:
                        parent_check = subprocess.run(
                            ['ps', '-o', 'ppid=', '-p', str(pid)],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if parent_check.returncode == 0:
                            ppid = int(parent_check.stdout.strip())
                            # If parent is 1 (init/systemd), it's orphaned
                            if ppid != 1:
                                continue  # Has valid parent, skip
                    except (ValueError, subprocess.TimeoutExpired):
                        pass  # Can't determine parent, assume orphaned

                    # Kill the orphaned process
                    logger.info(f"Killing orphaned process {pid}: {line[80:]}")
                    try:
                        os.kill(pid, 9)  # SIGKILL
                        killed_count += 1
                    except ProcessLookupError:
                        pass  # Process already gone
                    except PermissionError:
                        logger.warning(f"Permission denied to kill process {pid}")

                except (ValueError, IndexError):
                    continue

    except subprocess.TimeoutExpired:
        logger.error("Timeout while listing processes")
    except Exception as e:
        logger.error(f"Error killing orphaned processes: {e}")

    if killed_count > 0:
        logger.info(f"Killed {killed_count} orphaned process(es)")

    return killed_count


def cleanup_shared_memory() -> int:
    """
    Remove stale shared memory objects in /dev/shm.

    Removes objects matching our naming patterns:
    - psm_* (POSIX shared memory)
    - wnsm_* (Windows named shared memory compatibility)
    - youquantipy_* (application-specific)

    Returns:
        Number of objects removed
    """
    removed_count = 0
    shm_dir = Path('/dev/shm')

    if not shm_dir.exists():
        logger.warning("/dev/shm does not exist")
        return 0

    # Patterns to match our shared memory objects
    patterns = ['psm_*', 'wnsm_*', 'youquantipy_*']

    try:
        for pattern in patterns:
            for shm_file in shm_dir.glob(pattern):
                try:
                    # Check if file is actually accessible and owned by us
                    stat = shm_file.stat()

                    # Only remove if we own it
                    if stat.st_uid == os.getuid():
                        logger.info(f"Removing stale shared memory: {shm_file.name}")
                        shm_file.unlink()
                        removed_count += 1
                    else:
                        logger.debug(f"Skipping shared memory not owned by us: {shm_file.name}")

                except FileNotFoundError:
                    pass  # Already removed
                except PermissionError:
                    logger.warning(f"Permission denied to remove {shm_file.name}")
                except Exception as e:
                    logger.warning(f"Error removing {shm_file.name}: {e}")

    except Exception as e:
        logger.error(f"Error cleaning shared memory: {e}")

    if removed_count > 0:
        logger.info(f"Removed {removed_count} stale shared memory object(s)")

    return removed_count


def cleanup_camera_devices(device_paths: List[str] = None) -> int:
    """
    Release locked camera devices by killing processes holding them.

    Args:
        device_paths: List of device paths (e.g., ['/dev/video0', '/dev/video2'])
                     If None, will attempt all /dev/video* devices

    Returns:
        Number of processes killed
    """
    killed_count = 0

    # If no specific devices provided, find all video devices
    if device_paths is None:
        device_paths = glob.glob('/dev/video*')

    for device_path in device_paths:
        if not os.path.exists(device_path):
            continue

        try:
            # Use fuser to find processes using this device
            result = subprocess.run(
                ['fuser', device_path],
                capture_output=True,
                text=True,
                timeout=2
            )

            # fuser returns 0 if processes found, 1 if none
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split()
                logger.info(f"Found {len(pids)} process(es) holding {device_path}")

                for pid in pids:
                    try:
                        # Don't kill ourselves
                        pid_int = int(pid)
                        if pid_int == os.getpid():
                            continue

                        logger.info(f"Killing process {pid} holding {device_path}")
                        subprocess.run(['kill', '-9', pid], timeout=1, check=False)
                        killed_count += 1
                    except (ValueError, subprocess.TimeoutExpired):
                        pass

        except FileNotFoundError:
            logger.debug(f"fuser command not found (install psmisc package)")
            break  # Don't try other devices if fuser not available
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout checking {device_path}")
        except Exception as e:
            logger.warning(f"Error cleaning up {device_path}: {e}")

    if killed_count > 0:
        logger.info(f"Killed {killed_count} process(es) holding camera devices")

    return killed_count


def defensive_cleanup_all(cleanup_cameras: bool = True) -> Tuple[int, int, int]:
    """
    Perform comprehensive defensive cleanup of all orphaned resources.

    This should be called on application startup to clean up resources
    left behind by previous crashed instances.

    Args:
        cleanup_cameras: Whether to clean up camera device locks (default: True)

    Returns:
        Tuple of (processes_killed, shm_removed, camera_processes_killed)
    """
    logger.info("=" * 60)
    logger.info("Starting defensive cleanup of orphaned resources...")
    logger.info("=" * 60)

    # Step 1: Kill orphaned processes
    processes_killed = kill_orphaned_processes()

    # Step 2: Clean up shared memory
    shm_removed = cleanup_shared_memory()

    # Step 3: Clean up camera devices (optional)
    camera_processes_killed = 0
    if cleanup_cameras:
        camera_processes_killed = cleanup_camera_devices()

    # Summary
    logger.info("=" * 60)
    logger.info("Defensive cleanup complete:")
    logger.info(f"  - Orphaned processes killed: {processes_killed}")
    logger.info(f"  - Shared memory objects removed: {shm_removed}")
    logger.info(f"  - Camera device processes killed: {camera_processes_killed}")
    logger.info("=" * 60)

    return (processes_killed, shm_removed, camera_processes_killed)


if __name__ == "__main__":
    # Test/manual cleanup mode
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(name)s: %(message)s'
    )

    print("\n🧹 YouQuantiPy Resource Cleanup Utility")
    print("This will clean up orphaned processes and resources.\n")

    response = input("Proceed with cleanup? (y/N): ")
    if response.lower() == 'y':
        defensive_cleanup_all()
    else:
        print("Cleanup cancelled.")
