"""
ZeroMQ Camera Source Implementation

Receives camera frames over network from Windows host using ZeroMQ.
Designed for Windows→WSL2 camera bridging to bypass USB passthrough issues.

Supports both MJPEG (FRAM) and H.264 (H264_NAL) encoding formats.
"""

import cv2
import numpy as np
import time
import logging
import struct
from typing import Tuple, Optional, Dict
from .base import CameraSource
from .ffmpeg_h264_decoder import FFmpegH264Decoder

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    zmq = None

logger = logging.getLogger('ZMQCameraSource')
logger.setLevel(logging.INFO)  # Disable DEBUG logs to reduce console spam


def _check_discovery_service_health(windows_ip: str, discovery_port: int) -> bool:
    """
    Check if Windows discovery service is reachable.

    Performs a quick connectivity test to verify the discovery service is running
    before attempting full manifest reception.

    Args:
        windows_ip: Windows host IP address
        discovery_port: Discovery service port (default: 5550)

    Returns:
        bool: True if service appears healthy, False otherwise
    """
    import socket

    logger.info(f"[Health Check] Testing discovery service at {windows_ip}:{discovery_port}...")

    try:
        # Try to connect to the discovery port
        # Use a very short timeout (1 second) for health check
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)

        result = sock.connect_ex((windows_ip, discovery_port))
        sock.close()

        if result == 0:
            logger.info(f"[Health Check] ✓ Discovery service port {discovery_port} is open")
            return True
        else:
            logger.warning(f"[Health Check] ✗ Discovery service port {discovery_port} is not reachable (error {result})")
            logger.warning(f"[Health Check]   This usually means:")
            logger.warning(f"[Health Check]     - Windows camera manager is not running")
            logger.warning(f"[Health Check]     - Discovery service failed to start")
            logger.warning(f"[Health Check]     - Firewall blocking port {discovery_port}")
            logger.warning(f"[Health Check]   Run on Windows: start_all_cameras.bat")
            return False

    except socket.timeout:
        logger.warning(f"[Health Check] ✗ Timeout connecting to {windows_ip}:{discovery_port}")
        logger.warning(f"[Health Check]   Windows host may be unreachable or firewall blocking connection")
        return False
    except Exception as e:
        logger.warning(f"[Health Check] ✗ Error checking discovery service: {e}")
        return False


def _try_receive_manifest(windows_ip: str, discovery_port: int, timeout_ms: int) -> Optional[Dict]:
    """
    Try to receive camera manifest from Windows discovery service.

    Args:
        windows_ip: Windows host IP address
        discovery_port: Discovery service port (default: 5550)
        timeout_ms: Timeout in milliseconds

    Returns:
        Optional[Dict]: Manifest dictionary if received, None otherwise
    """
    if not ZMQ_AVAILABLE:
        return None

    logger.info(f"[Discovery] Attempting to receive manifest from {windows_ip}:{discovery_port}...")

    context = None
    socket = None

    try:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b'')  # Subscribe to all messages
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)

        endpoint = f"tcp://{windows_ip}:{discovery_port}"
        socket.connect(endpoint)

        logger.info(f"[Discovery] Connected to {endpoint}, waiting for manifest...")

        # Allow subscription to establish (slow joiner problem)
        import time as time_module
        time_module.sleep(0.3)

        # Try to receive manifest
        try:
            import json
            manifest_bytes = socket.recv()
            manifest = json.loads(manifest_bytes.decode('utf-8'))

            if manifest.get('type') == 'camera_manifest':
                num_cameras = len(manifest.get('cameras', []))
                logger.info(f"[Discovery] ✓ Received manifest: {num_cameras} camera(s)")
                for cam in manifest.get('cameras', []):
                    logger.info(f"[Discovery]   - Camera {cam['logical_index']}: {cam['name']} (port {cam['port']})")
                return manifest
            else:
                logger.warning(f"[Discovery] Received unknown message type: {manifest.get('type')}")
                return None

        except zmq.Again:
            logger.info(f"[Discovery] No manifest received (timeout after {timeout_ms}ms)")
            return None

    except Exception as e:
        logger.info(f"[Discovery] Error receiving manifest: {e}")
        return None

    finally:
        # Cleanup
        if socket:
            try:
                socket.close(linger=0)
            except:
                pass
        if context:
            try:
                context.term()
            except:
                pass


def _probe_with_manifest(manifest: Dict, config: Dict, windows_ip: str) -> list:
    """
    Probe cameras using manifest from discovery service.

    Waits for ALL cameras from manifest with progress reporting.

    Args:
        manifest: Camera manifest from Windows discovery service
        config: Configuration dictionary
        windows_ip: Windows host IP address

    Returns:
        list: List of tuples (logical_camera_index, camera_name) for discovered cameras
    """
    cameras_from_manifest = manifest.get('cameras', [])
    if not cameras_from_manifest:
        logger.warning("[Manifest] Manifest contains no cameras")
        return []

    expected_cameras = {cam['logical_index']: cam for cam in cameras_from_manifest}
    found_cameras = {}

    zmq_settings = config.get('zmq_camera_bridge', {})
    camera_connection_timeout_ms = zmq_settings.get('camera_connection_timeout_ms', 60000)

    logger.info(f"[Manifest] Waiting for {len(expected_cameras)} camera(s) (timeout: {camera_connection_timeout_ms}ms)...")

    context = zmq.Context()
    created_sockets = []

    import time as time_module
    import json
    start_time = time_module.time()
    timeout_seconds = camera_connection_timeout_ms / 1000.0

    try:
        # Create sockets for all cameras
        sockets_by_index = {}
        for cam_info in cameras_from_manifest:
            logical_idx = cam_info['logical_index']
            port = cam_info['port']
            topic = cam_info['topic']

            try:
                endpoint = f"tcp://{windows_ip}:{port}"
                sock = context.socket(zmq.SUB)
                created_sockets.append(sock)

                sock.setsockopt(zmq.SUBSCRIBE, topic.encode('utf-8'))
                sock.setsockopt(zmq.RCVTIMEO, 500)  # 500ms per recv
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(endpoint)

                sockets_by_index[logical_idx] = sock
                logger.debug(f"[Manifest]   Socket created for camera {logical_idx} ({endpoint})")

            except Exception as e:
                logger.warning(f"[Manifest]   Failed to create socket for camera {logical_idx}: {e}")

        # Allow subscriptions to establish
        # Increased from 0.3s to 1.5s to catch initial META message (sent every 5s)
        # This avoids the ZMQ "slow joiner" problem where early messages are lost
        time_module.sleep(1.5)

        # Poll all sockets until all cameras found or timeout
        while len(found_cameras) < len(expected_cameras):
            elapsed = time_module.time() - start_time
            if elapsed >= timeout_seconds:
                logger.warning(f"[Manifest] Timeout after {elapsed:.1f}s - found {len(found_cameras)}/{len(expected_cameras)} cameras")
                break

            # Try each camera that hasn't been found yet
            for logical_idx, cam_info in expected_cameras.items():
                if logical_idx in found_cameras:
                    continue  # Already found

                sock = sockets_by_index.get(logical_idx)
                if not sock:
                    continue

                try:
                    # Try to receive message
                    parts = sock.recv_multipart()

                    # Check for metadata message: [topic][msg_type][metadata_json]
                    if len(parts) == 3:
                        recv_topic, msg_type, payload = parts

                        if msg_type == b"META":
                            # Found camera metadata!
                            try:
                                metadata = json.loads(payload.decode('utf-8'))
                                if metadata.get('type') == 'camera_info':
                                    camera_name = metadata.get('camera_name', cam_info['name'])
                                    # Extract resolution from metadata (defaults to 720p if missing)
                                    width = metadata.get('width', 1280)
                                    height = metadata.get('height', 720)
                                    # Store as tuple: (name, width, height)
                                    found_cameras[logical_idx] = (camera_name, width, height)
                                    logger.info(f"[Manifest] ✓ Found camera {logical_idx}: {camera_name} ({width}x{height}) ({len(found_cameras)}/{len(expected_cameras)})")
                            except Exception as e:
                                logger.debug(f"[Manifest]   Error parsing metadata for camera {logical_idx}: {e}")

                        elif msg_type == b"FRAM":
                            # Camera is alive and sending frames! Use friendly name from manifest
                            camera_name = cam_info.get('name', f"Camera {logical_idx}")
                            # No metadata received, use default resolution (720p)
                            found_cameras[logical_idx] = (camera_name, 1280, 720)
                            logger.info(f"[Manifest] ✓ Found camera {logical_idx}: {camera_name} (1280x720 default) (detected via frame) ({len(found_cameras)}/{len(expected_cameras)})")

                    # Also handle 4-part frame format: [topic][msg_type][length][payload]
                    elif len(parts) == 4:
                        recv_topic, msg_type, length_bytes, payload = parts

                        if msg_type == b"FRAM":
                            # Camera is alive and sending frames! Use friendly name from manifest
                            camera_name = cam_info.get('name', f"Camera {logical_idx}")
                            # No metadata received, use default resolution (720p)
                            found_cameras[logical_idx] = (camera_name, 1280, 720)
                            logger.info(f"[Manifest] ✓ Found camera {logical_idx}: {camera_name} (1280x720 default) (detected via 4-part frame) ({len(found_cameras)}/{len(expected_cameras)})")

                except zmq.Again:
                    # Timeout on individual recv - continue to next camera
                    pass
                except Exception as e:
                    logger.debug(f"[Manifest]   Error receiving from camera {logical_idx}: {e}")

            # Brief sleep to avoid tight loop
            time_module.sleep(0.05)

        # Report results
        if found_cameras:
            logger.info(f"[Manifest] ✓ Discovery complete: {len(found_cameras)}/{len(expected_cameras)} cameras found")

            # Report missing cameras
            missing = set(expected_cameras.keys()) - set(found_cameras.keys())
            if missing:
                logger.warning(f"[Manifest] Missing cameras: {sorted(missing)}")
                for idx in sorted(missing):
                    cam_info = expected_cameras[idx]
                    logger.warning(f"[Manifest]   - Camera {idx}: {cam_info['name']} (port {cam_info['port']})")

            # Build result list: [(index, name, width, height), ...]
            result = []
            for idx, camera_data in sorted(found_cameras.items()):
                if isinstance(camera_data, tuple) and len(camera_data) == 3:
                    # (name, width, height) from metadata
                    name, width, height = camera_data
                    result.append((idx, name, width, height))
                else:
                    # Fallback (shouldn't happen with current code)
                    name = camera_data if isinstance(camera_data, str) else f"Camera {idx}"
                    result.append((idx, name, 1280, 720))

            # AUTO-UPDATE: Save discovered camera names to config file
            # This ensures GUI labels always match actual camera feeds (plug-and-play)
            try:
                import os
                # Build discovered_camera_names dict from found cameras
                discovered_names = {}
                for idx, camera_data in found_cameras.items():
                    if isinstance(camera_data, tuple):
                        name, width, height = camera_data
                        discovered_names[str(idx)] = name
                    else:
                        discovered_names[str(idx)] = camera_data

                # Handle both ConfigHandler and plain dict
                if hasattr(config, 'config'):
                    # ConfigHandler wrapper - access underlying dict
                    config_dict = config.config
                else:
                    # Plain dict
                    config_dict = config

                # Update config in memory
                if 'zmq_camera_bridge' not in config_dict:
                    config_dict['zmq_camera_bridge'] = {}
                config_dict['zmq_camera_bridge']['discovered_camera_names'] = discovered_names

                # Save to file (so child processes and future restarts get updated names)
                config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'youquantipy_config.json')
                config_path = os.path.abspath(config_path)

                import json
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=4)

                logger.info(f"[Manifest] ✓ Auto-updated discovered_camera_names in config: {discovered_names}")

            except Exception as e:
                logger.warning(f"[Manifest] Failed to auto-update config with camera names: {e}")
                # Non-fatal - continue with discovery even if config update fails

            return result
        else:
            logger.warning("[Manifest] No cameras found")
            return []

    finally:
        # Cleanup all sockets
        for sock in list(created_sockets):
            try:
                sock.close(linger=0)
            except:
                pass
        created_sockets.clear()

        # Terminate context
        try:
            context.term()
        except:
            pass


def probe_zmq_cameras(config: Dict, max_cameras: int = 6) -> list:
    """
    Probe ZMQ endpoints to discover available cameras with their friendly names.

    This function tests connectivity to ZMQ camera senders and returns
    a list of available cameras with their metadata.

    NEW: Uses manifest-based discovery for reliable camera detection.
    Fallback: Uses legacy port-by-port probing if manifest not available.

    CRITICAL: Camera indices returned are LOGICAL INDICES (0, 1, 2, ...), which are
    guaranteed sequential and match port/topic numbers. This ensures correct
    port/topic mapping in the factory (port = base_port + logical_index).

    Args:
        config: Configuration dictionary with ZMQ settings
        max_cameras: Maximum number of ports to probe (default: 6, checks ports 5551-5556)

    Returns:
        list: List of tuples (logical_camera_index, camera_name, width, height) for available cameras
              - logical_camera_index: Sequential logical index (0, 1, 2, ...) from metadata
              - camera_name: Friendly name from Windows WMI (or "Camera N" if WMI unavailable)
              - width: Frame width in pixels (from ZMQ metadata, defaults to 1280 if unavailable)
              - height: Frame height in pixels (from ZMQ metadata, defaults to 720 if unavailable)

    Example:
        >>> config = {'zmq_camera_bridge': {'enabled': True}}
        >>> available = probe_zmq_cameras(config, max_cameras=5)
        >>> print(available)  # [(0, "Integrated IR Camera", 1920, 1080), (1, "HD Pro Webcam", 1280, 720)]
        # Note: Indices 0, 1 are logical indices (sequential), NOT Windows device indices
        # Resolution is automatically extracted from Windows camera metadata
    """
    if not ZMQ_AVAILABLE:
        logger.warning("pyzmq not available, cannot probe ZMQ cameras")
        return []

    zmq_settings = config.get('zmq_camera_bridge', {})

    # Get Windows host IP (explicit config takes priority)
    windows_ip = zmq_settings.get('windows_host_ip')

    if not windows_ip:
        # Fall back to auto-detection
        windows_ip = _get_windows_host_ip_static()

    if not windows_ip:
        logger.warning("Could not detect Windows host IP, cannot probe ZMQ cameras")
        return []

    # Check if manifest-based discovery is enabled (default: true)
    enable_manifest_discovery = zmq_settings.get('enable_manifest_discovery', True)

    # Phase 1: Try manifest-based discovery (NEW, RELIABLE METHOD)
    if enable_manifest_discovery:
        logger.info("=" * 60)
        logger.info("Phase 1: Manifest-Based Discovery")
        logger.info("=" * 60)

        discovery_port = zmq_settings.get('discovery_port', 5550)
        discovery_timeout_ms = zmq_settings.get('discovery_timeout_ms', 20000)

        # Health check: Verify discovery service is running before attempting manifest reception
        service_healthy = _check_discovery_service_health(windows_ip, discovery_port)

        if not service_healthy:
            logger.warning("[Discovery] Skipping manifest discovery - service not reachable")
            logger.info("[Discovery] Falling back to legacy port-by-port probing...")
            # Fall through to legacy probing
        else:
            manifest = _try_receive_manifest(windows_ip, discovery_port, discovery_timeout_ms)

            if manifest:
                # Manifest received! Use it to discover cameras
                logger.info("[Discovery] Using manifest-based discovery (reliable)")
                result = _probe_with_manifest(manifest, config, windows_ip)

                if result:
                    logger.info(f"[Discovery] ✓ Manifest discovery succeeded: {len(result)} camera(s) found")
                    return result
                else:
                    logger.warning("[Discovery] Manifest discovery found no cameras")
                    # Fall through to legacy probing

            else:
                logger.info("[Discovery] No manifest received - falling back to legacy probing")

    # Phase 2: Legacy port-by-port probing (FALLBACK)
    logger.info("=" * 60)
    logger.info("Phase 2: Legacy Port-by-Port Probing (Fallback)")
    logger.info("=" * 60)

    # Get port range
    base_port = zmq_settings.get('port_range', [5551, 5560])[0]
    max_port = zmq_settings.get('port_range', [5551, 5560])[1]
    # OPTIMIZED: Reduced timeout from 15s to 5s for faster discovery
    # 5s is sufficient to catch periodic metadata broadcasts (sent every 2-5s)
    # This reduces worst-case discovery time from 150s to 50s (10 cameras)
    probe_timeout_ms = zmq_settings.get('connection_timeout_ms', 5000)

    logger.info(f"Probing ZMQ cameras at {windows_ip}:{base_port}-{max_port}...")

    available_cameras = []

    # Early termination: only stop if we've probed many cameras and found ZERO
    # This prevents premature termination with staggered camera startup
    consecutive_misses = 0
    max_consecutive_misses = 999  # Effectively disabled - probe all cameras in range

    # Alternative safety: only stop if probed 5+ cameras and found NONE
    # This prevents infinite loops while allowing discovery of all cameras

    # Create DEDICATED ZMQ context for probing (will be terminated after probing)
    # Using dedicated context ensures clean separation from camera instances
    context = zmq.Context()

    # Track all created sockets for cleanup (failsafe)
    created_sockets = []

    try:
        for cam_idx in range(max_cameras):
            port = base_port + cam_idx

            if port > max_port:
                break

            endpoint = f"tcp://{windows_ip}:{port}"
            topic = f"cam{cam_idx}".encode('utf-8')

            sock = None  # Initialize for finally block
            try:
                # Create SUB socket
                sock = context.socket(zmq.SUB)
                created_sockets.append(sock)  # Track for cleanup
                # Short timeout per recv (500ms) - allows loop to receive multiple messages
                # Total probe time controlled by while loop (typically 6 seconds)
                sock.setsockopt(zmq.RCVTIMEO, 500)
                sock.setsockopt(zmq.SUBSCRIBE, topic)
                sock.setsockopt(zmq.LINGER, 0)

                # Connect
                sock.connect(endpoint)

                # Wait for metadata (skip video frames)
                # Metadata is broadcast every 2-5 seconds, so we wait up to 6 seconds
                camera_name = None
                actual_camera_index = None  # Will be populated from metadata (Windows device index)
                import json
                import time as time_module

                # CRITICAL: Allow ZMQ subscription to establish before receiving
                # PUB/SUB has a "slow joiner" problem - subscription takes 30-200ms
                # Messages sent before subscription is ready are permanently lost
                time_module.sleep(0.2)  # 200ms should cover worst case
                logger.info(f"  ⏳ Waiting 200ms for subscription to establish...")

                probe_start = time_module.time()
                timeout_seconds = probe_timeout_ms / 1000.0
                any_message_received = False  # Track if camera is actually responding

                try:
                    while (time_module.time() - probe_start) < timeout_seconds:
                        try:
                            parts = sock.recv_multipart()

                            # Log every message received with timestamp for debugging
                            elapsed = time_module.time() - probe_start
                            logger.debug(f"  [{elapsed:.1f}s] Camera {cam_idx}: Received {len(parts)}-part message")

                            # Check for metadata message: [topic][msg_type][metadata_json]
                            if len(parts) == 3:
                                recv_topic, msg_type, payload = parts

                                if recv_topic != topic:
                                    logger.debug(f"  ✗ Camera {cam_idx}: Topic mismatch")
                                    continue

                                if msg_type == b"META":
                                    # Found metadata! Parse it
                                    try:
                                        metadata = json.loads(payload.decode('utf-8'))
                                        logger.info(f"  📦 Camera {cam_idx}: Metadata = {metadata}")
                                        if metadata.get('type') == 'camera_info':
                                            # CRITICAL FIX: Use logical index (guaranteed sequential 0,1,2...)
                                            # This ensures port/topic math works correctly
                                            actual_camera_index = metadata.get('camera_index', cam_idx)
                                            windows_device_index = metadata.get('windows_device_index', actual_camera_index)
                                            camera_name = metadata.get('camera_name', f"Camera {actual_camera_index}")
                                            logger.debug(f"  Parsed: logical={actual_camera_index}, Windows device={windows_device_index}, name={camera_name}")

                                            # Use fallback name if empty (show all cameras)
                                            if camera_name is None or camera_name == "":
                                                camera_name = f"Camera {actual_camera_index}"
                                                logger.debug(f"  Using fallback name: {camera_name}")

                                            # Add camera (uses logical index for consistent port/topic mapping)
                                            # Logical index ensures port = base_port + logical_index works correctly
                                            any_message_received = True  # Camera is responding
                                            logger.info(f"  ✓ Port {port} → Logical Camera {actual_camera_index} (Windows device {windows_device_index}, {camera_name}) available")
                                            break  # Got what we need, exit loop
                                        else:
                                            logger.debug(f"  ? Camera {cam_idx}: Unknown metadata type")
                                    except Exception as e:
                                        logger.debug(f"  ? Camera {cam_idx}: Error parsing metadata: {e}")

                                elif msg_type == b"FRAM":
                                    # Video frame - camera is responding, keep waiting for metadata
                                    any_message_received = True  # Camera is sending frames
                                    logger.debug(f"  → Camera {cam_idx}: Received frame, waiting for metadata...")
                                    continue

                                else:
                                    logger.debug(f"  ? Camera {cam_idx}: Unknown message type: {msg_type}")
                                    continue

                            elif len(parts) == 4:
                                # New 4-part frame format: [topic][msg_type][length][payload]
                                recv_topic, msg_type, _, _ = parts

                                if recv_topic != topic:
                                    continue

                                if msg_type == b"FRAM":
                                    # Video frame - camera is responding
                                    any_message_received = True  # Camera is sending frames
                                    continue
                                elif msg_type == b"META":
                                    # Should be 3-part, but handle it
                                    logger.debug(f"  ? Camera {cam_idx}: Unexpected 4-part META message")
                                    continue

                            else:
                                logger.debug(f"  ? Camera {cam_idx}: Unexpected message format ({len(parts)} parts)")

                        except zmq.Again:
                            # Timeout on individual recv - continue waiting (don't break)
                            # This allows the probe to wait the full timeout_seconds for metadata
                            elapsed = time_module.time() - probe_start
                            logger.debug(f"  [{elapsed:.1f}s] Camera {cam_idx}: ⏳ No message in 500ms, continuing to wait...")
                            continue  # Don't break - keep waiting for metadata re-broadcast

                    # Add camera if it responded with metadata
                    if any_message_received and actual_camera_index is not None:
                        # Camera is available - use logical index (guaranteed sequential 0,1,2...)
                        # This ensures port/topic math works correctly: port = base_port + logical_index
                        available_cameras.append((actual_camera_index, camera_name))
                        logger.info(f"  + Added logical camera {actual_camera_index} (port {port}) to available list")
                        consecutive_misses = 0  # Reset counter on success
                    else:
                        # No messages received - sender not running
                        logger.debug(f"  ✗ Camera {cam_idx}: No messages received - sender not running")
                        consecutive_misses += 1

                        # Safety: only stop if probed 5+ ports and found ZERO cameras
                        # This prevents infinite loops while allowing discovery of all cameras
                        if cam_idx >= 5 and len(available_cameras) == 0:
                            logger.info(f"  🛑 Stopping probe: probed {cam_idx + 1} ports but found no cameras")
                            break  # Exit for loop early

                except Exception as e:
                    logger.debug(f"  ✗ Camera {cam_idx}: Error during probe: {e}")

            except Exception as e:
                logger.debug(f"  ✗ Camera {cam_idx}: Error probing: {e}")

            finally:
                # CRITICAL: Close socket even if break/exception occurs
                # This ensures resources are freed before context.term()
                if sock is not None:
                    try:
                        sock.close(linger=0)
                        if sock in created_sockets:
                            created_sockets.remove(sock)
                    except Exception as e:
                        logger.debug(f"  ⚠️  Error closing socket for camera {cam_idx}: {e}")

    finally:
        # CRITICAL: Cleanup any remaining sockets before terminating context
        # This is a failsafe in case socket cleanup in the loop failed
        for sock in list(created_sockets):  # Create copy to avoid modification during iteration
            try:
                sock.close(linger=0)
                logger.debug(f"Cleaned up remaining socket in finally block")
            except Exception as e:
                logger.debug(f"Error closing socket in finally: {e}")
        created_sockets.clear()

        # CRITICAL: Terminate context after all sockets are closed
        try:
            context.term()
            logger.debug("Probe context terminated")
        except Exception as e:
            logger.warning(f"Error terminating probe context: {e}")

    # CRITICAL FIX: Sort cameras by logical index to ensure correct GUI dropdown order
    # Without sorting, cameras may appear out of order if probe receives responses non-sequentially
    # This ensures dropdown index matches logical camera index for correct port/topic mapping
    available_cameras.sort(key=lambda x: x[0])  # Sort by logical index (first element of tuple)

    # Discovery Summary
    logger.info("=" * 60)
    logger.info("Camera Discovery Summary")
    logger.info("=" * 60)

    if available_cameras:
        logger.info(f"✓ Success: Found {len(available_cameras)} camera(s)")
        for idx, name in available_cameras:
            port = base_port + idx
            topic = f"cam{idx}"
            logger.info(f"  - Camera {idx}: {name}")
            logger.info(f"      Endpoint: tcp://{windows_ip}:{port}")
            logger.info(f"      Topic: {topic}")
    else:
        logger.warning("✗ No cameras discovered!")
        logger.warning("  Troubleshooting steps:")
        logger.warning("    1. Verify Windows camera sender is running:")
        logger.warning(f"       Run on Windows: start_all_cameras.bat")
        logger.warning("    2. Check if cameras are actually sending:")
        logger.warning(f"       Run on Windows: python list_camera_processes.py")
        logger.warning("    3. Test network connectivity:")
        logger.warning(f"       ping {windows_ip}")
        logger.warning("    4. Check firewall settings (ports 5550-5560)")
        logger.warning("    5. Review Windows bridge logs:")
        logger.warning("       Check logs/ directory on Windows")

    logger.info("=" * 60)

    return available_cameras


def _get_windows_host_ip_static() -> Optional[str]:
    """
    Auto-detect Windows host IP from /etc/resolv.conf.

    In WSL2, the Windows host IP appears as the nameserver.

    Returns:
        Optional[str]: Windows host IP or None if not found
    """
    try:
        with open('/etc/resolv.conf', 'r') as f:
            for line in f:
                if line.startswith('nameserver'):
                    ip = line.split()[1].strip()
                    return ip
    except Exception as e:
        logger.warning(f"Could not read /etc/resolv.conf: {e}")

    return None


class ZMQCameraSource(CameraSource):
    """
    ZeroMQ-based camera source for network streaming.

    Features:
    - Receives MJPEG frames from Windows host via ZeroMQ
    - Auto-detects Windows host IP from /etc/resolv.conf
    - Automatic reconnection on network failures
    - Frame validation and timestamping
    - Low-latency configuration (HWM=10)
    """

    def __init__(self, config: Dict, camera_index: int, camera_name: Optional[str] = None):
        """
        Initialize ZeroMQ camera source.

        Args:
            config: Camera configuration dictionary
            camera_index: Camera device index
            camera_name: Optional camera friendly name from manifest discovery.
                        If provided, skips metadata handshake during open()
        """
        if not ZMQ_AVAILABLE:
            raise ImportError("pyzmq is required for ZMQ camera source. Install with: pip install pyzmq")

        self.config = config
        self.camera_index = camera_index
        self.backend_name = "ZMQ"
        self.actual_resolution = None
        self.actual_fps = None
        self._opened = False

        # Parse configuration
        camera_key = f"camera_{camera_index}"
        camera_settings = config.get('camera_settings', {}).get(camera_key, {})
        zmq_settings = config.get('zmq_camera_bridge', {})

        # ZMQ connection settings
        self.endpoint = camera_settings.get('zmq_endpoint')
        self.topic = camera_settings.get('zmq_topic', f'cam{camera_index}').encode('utf-8')
        self.reconnect_retries = int(camera_settings.get('reconnect_retries', 5))
        self.connection_timeout_ms = int(zmq_settings.get('connection_timeout_ms', 5000))
        self.frame_timeout_ms = int(zmq_settings.get('frame_timeout_ms', 100))
        self.reconnect_interval_ms = int(zmq_settings.get('reconnect_interval_ms', 1000))

        # Expected resolution (may differ from actual)
        self.target_width = int(camera_settings.get('width', 1280))
        self.target_height = int(camera_settings.get('height', 720))
        self.target_fps = float(camera_settings.get('fps', 30))

        # Auto-detect Windows host IP if endpoint not specified
        if not self.endpoint:
            # Check for explicit IP first
            windows_ip = zmq_settings.get('windows_host_ip')

            if not windows_ip:
                # Fall back to auto-detection
                auto_detect = zmq_settings.get('auto_detect_windows_ip', True)
                if auto_detect:
                    windows_ip = self._get_windows_host_ip()

            if windows_ip:
                # Use default port 5551 + camera_index
                base_port = zmq_settings.get('port_range', [5551, 5560])[0]
                port = base_port + camera_index
                self.endpoint = f"tcp://{windows_ip}:{port}"
                logger.info(f"Using Windows host: {self.endpoint}")
            else:
                raise ValueError("Could not determine Windows host IP. Please specify 'windows_host_ip' or 'zmq_endpoint' in config.")

        # ZMQ objects - CRITICAL: Create dedicated context per camera (not singleton)
        # This ensures proper cleanup and prevents socket leakage across camera restarts
        self.context = None
        self.socket = None

        # Camera metadata (may come from manifest discovery or Windows sender)
        self.camera_name = camera_name  # If provided from manifest, skip metadata handshake

        # H.264 decoder (initialized on first H264_NAL message)
        self.h264_decoder = None
        self.encoding_format = None  # Will be 'MJPEG' or 'H264' based on first message

        # Performance tracking
        self.frame_count = 0
        self.last_frame_time = 0
        self.fps_start_time = 0
        self.fps_frame_count = 0

        # Debug logging throttle (only log first 3 frames + every 30th frame)
        self._debug_frame_counter = 0

    def open(self) -> bool:
        """
        Open ZeroMQ connection to camera sender.

        Returns:
            bool: True if connection successful, False otherwise
        """
        logger.info(f"[DIAGNOSTIC] Camera {self.camera_index} ZMQ open() starting")
        logger.info(f"[DIAGNOSTIC]   Endpoint: {self.endpoint}")
        logger.info(f"[DIAGNOSTIC]   Topic: {self.topic.decode()}")
        logger.info(f"[DIAGNOSTIC]   Connection timeout: {self.connection_timeout_ms}ms")
        logger.info(f"[DIAGNOSTIC]   Frame timeout: {self.frame_timeout_ms}ms")
        logger.info(f"[DIAGNOSTIC]   Reconnect retries: {self.reconnect_retries}")
        logger.info(f"Connecting to ZMQ camera source: {self.endpoint} (topic: {self.topic.decode()})")

        for attempt in range(self.reconnect_retries):
            if attempt > 0:
                logger.info(f"Retry attempt {attempt + 1}/{self.reconnect_retries}")
                time.sleep(self.reconnect_interval_ms / 1000.0)

            try:
                # Create DEDICATED ZMQ context for this camera (not singleton)
                # CRITICAL: Using dedicated context prevents socket leakage and allows
                # proper cleanup when camera is stopped/restarted
                logger.info(f"[DIAGNOSTIC] Creating ZMQ context...")
                self.context = zmq.Context()
                logger.info(f"[DIAGNOSTIC] ✓ ZMQ context created")

                logger.info(f"[DIAGNOSTIC] Creating ZMQ SUB socket...")
                self.socket = self.context.socket(zmq.SUB)
                logger.info(f"[DIAGNOSTIC] ✓ ZMQ socket created")

                # Low-latency settings
                logger.info(f"[DIAGNOSTIC] Setting socket options...")
                self.socket.setsockopt(zmq.RCVHWM, 1)  # Minimal receive buffer (1 frame) + manual drain for lowest latency
                self.socket.setsockopt(zmq.SUBSCRIBE, self.topic)
                self.socket.setsockopt(zmq.RCVTIMEO, self.frame_timeout_ms)
                self.socket.setsockopt(zmq.LINGER, 0)  # Don't linger on close
                logger.info(f"[DIAGNOSTIC] ✓ Socket options set (HWM=1, RCVTIMEO={self.frame_timeout_ms}ms)")

                # Connect to endpoint
                logger.info(f"[DIAGNOSTIC] Connecting to {self.endpoint}...")
                self.socket.connect(self.endpoint)
                logger.info(f"[DIAGNOSTIC] ✓ Socket connected to {self.endpoint}")

                logger.info(f"  Connected to {self.endpoint}")

                # METADATA HANDSHAKE: Receive camera info from Windows sender
                # Skip if camera name already provided from manifest discovery
                if not self.camera_name:
                    # No camera name provided - try to receive from sender (legacy fallback)
                    logger.info(f"[DIAGNOSTIC] No camera name from manifest - attempting metadata handshake...")
                    logger.debug(f"  Waiting for camera metadata (timeout: {self.connection_timeout_ms}ms)...")
                    metadata_received = self._receive_metadata()

                    if not metadata_received:
                        logger.warning(f"[DIAGNOSTIC] ⚠️  No metadata received - using fallback camera name")
                        logger.warning(f"  No metadata received - using fallback camera name")
                        self.camera_name = f"Camera {self.camera_index}"
                    else:
                        logger.info(f"[DIAGNOSTIC] ✓ Metadata received: {self.camera_name}")
                else:
                    # Camera name already known from manifest - skip metadata waiting
                    logger.info(f"[DIAGNOSTIC] ✓ Using camera name from manifest: {self.camera_name}")
                    logger.debug(f"  Skipping metadata handshake (name from discovery)")

                # PRE-FLIGHT CHECK: Wait for first frame to verify connection and sender health
                # For H.264, may need multiple NAL units (SPS, PPS, I-frame) before first frame is decoded
                logger.info(f"[DIAGNOSTIC] Waiting for first frame (pre-flight check)...")
                logger.debug(f"  Waiting for first frame...")

                max_attempts = 50  # 50 * 100ms = 5 seconds total for H.264 accumulation
                frame = None
                for attempt in range(max_attempts):
                    success, frame = self._receive_frame(drain_queue=False)

                    if success and frame is not None:
                        # Got a complete decoded frame!
                        logger.info(f"[DIAGNOSTIC] ✓ First frame received successfully (attempt {attempt+1}/{max_attempts})")
                        break

                    # For H.264, we may need multiple NAL units before getting a frame
                    # Continue looping to feed more NAL units
                    if self.h264_decoder is not None:
                        logger.debug(f"  H.264: Waiting for complete frame (attempt {attempt+1}/{max_attempts})...")
                        continue
                    else:
                        # MJPEG should return frame immediately - if failed, break
                        logger.error(f"[DIAGNOSTIC] ❌ MJPEG frame receive failed")
                        break

                if frame is not None:
                    logger.info(f"[DIAGNOSTIC] ✓ First frame received successfully")
                    # Update actual resolution from first frame
                    h, w = frame.shape[:2]
                    self.actual_resolution = (w, h)

                    # DIAGNOSTIC: Compare config resolution vs actual resolution
                    if (w, h) != (self.target_width, self.target_height):
                        logger.error(f"[ZMQ RESOLUTION MISMATCH]")
                        logger.error(f"  Config resolution: {self.target_width}x{self.target_height}")
                        logger.error(f"  Actual stream resolution: {w}x{h}")
                        logger.error(f"  ⚠️  THIS WILL CAUSE ISSUES!")
                        logger.error(f"  ⚠️  H.264 decoder was initialized for {self.target_width}x{self.target_height}")
                        logger.error(f"  ⚠️  But stream is sending {w}x{h}")
                        logger.error(f"  ")
                        logger.error(f"  FIX: Check Windows sender resolution:")
                        logger.error(f"       D:\\Projects\\youquantipy-redo\\windows_bridge\\start_cameras_silent.bat")
                        logger.error(f"       Look for --width and --height parameters")
                    else:
                        logger.info(f"[ZMQ RESOLUTION] ✓ Config matches actual stream: {w}x{h}")

                    logger.info(f"ZMQ camera {self.camera_index} opened successfully:")
                    logger.info(f"  Camera Name: {self.camera_name or 'Unknown'}")
                    logger.info(f"  Endpoint: {self.endpoint}")
                    logger.info(f"  Topic: {self.topic.decode()}")
                    logger.info(f"  Resolution: {w}x{h}")
                    logger.info(f"  Format: {self.encoding_format or 'MJPEG'}")
                    logger.info(f"  Backend: ZMQ")

                    self._opened = True
                    self.fps_start_time = time.time()
                    logger.info(f"[DIAGNOSTIC] ✅ Camera {self.camera_index} opened successfully")
                    return True
                else:
                    logger.error(f"[DIAGNOSTIC] ❌ Pre-flight check FAILED: No frames received from {self.endpoint}")
                    logger.warning(f"  Pre-flight check failed: No frames received from {self.endpoint}")
                    logger.warning(f"  This could indicate:")
                    logger.warning(f"    - Windows sender not running")
                    logger.warning(f"    - Orphaned sender (camera locked)")
                    logger.warning(f"    - Network connectivity issue")
                    logger.info(f"[DIAGNOSTIC]   Cleaning up socket and context...")
                    self._close_socket()
                    self._terminate_context()

            except zmq.ZMQError as e:
                logger.error(f"[DIAGNOSTIC] ❌ ZMQ error during connection: {e}")
                logger.error(f"  ZMQ error: {e}")
                self._close_socket()
                self._terminate_context()
            except Exception as e:
                logger.error(f"[DIAGNOSTIC] ❌ Exception during ZMQ source opening: {e}")
                logger.error(f"  Error opening ZMQ source: {e}")
                self._close_socket()
                self._terminate_context()

        logger.error(f"Failed to open ZMQ camera {self.camera_index} after {self.reconnect_retries} attempts")
        logger.error(f"  Check Windows sender status: python check_camera_status.py")
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the ZMQ camera source.

        Automatically retries if metadata messages are received,
        only failing after true timeout with no messages.

        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - success: True if frame read successfully
                - frame: BGR numpy array (H, W, 3) or None on failure
        """
        if not self._opened or not self.socket:
            return False, None

        # Retry loop: skip metadata messages, only fail on persistent timeout
        # With 100ms timeout per recv, 10 retries = 1 second total wait
        max_retries = 10

        for attempt in range(max_retries):
            success, frame = self._receive_frame(drain_queue=True)  # Enable drain for low latency

            if success:  # Got a real frame
                # Update performance tracking
                self.frame_count += 1
                self.fps_frame_count += 1
                self.last_frame_time = time.time()

                # Report FPS every 5 seconds
                elapsed = self.last_frame_time - self.fps_start_time
                if elapsed >= 5.0:
                    fps = self.fps_frame_count / elapsed
                    self.actual_fps = fps
                    logger.debug(f"ZMQ camera {self.camera_index}: {fps:.1f} fps")
                    self.fps_frame_count = 0
                    self.fps_start_time = self.last_frame_time

                return True, frame

            # Failed (metadata or timeout) - retry automatically
            # This allows skipping metadata messages without reconnecting

        # All retries exhausted - true capture failure
        logger.debug(f"ZMQ camera {self.camera_index}: No frames after {max_retries} attempts")
        return False, None

    def release(self):
        """Release the ZMQ connection and terminate context."""
        # Stop H.264 decoder if active
        if self.h264_decoder is not None:
            logger.info(f"Stopping H.264 decoder for camera {self.camera_index}")
            self.h264_decoder.stop()
            self.h264_decoder = None

        self._close_socket()
        self._terminate_context()
        self._opened = False
        logger.info(f"ZMQ camera {self.camera_index} released")

    def get_resolution(self) -> Tuple[int, int]:
        """Get the actual camera resolution."""
        if self.actual_resolution:
            return self.actual_resolution
        return (self.target_width, self.target_height)

    def get_fps(self) -> float:
        """Get the actual camera FPS."""
        if self.actual_fps:
            return self.actual_fps
        return self.target_fps

    def is_opened(self) -> bool:
        """Check if the ZMQ connection is active."""
        return self._opened and self.socket is not None

    def get_backend_name(self) -> str:
        """Get the backend name."""
        return self.backend_name

    def get_camera_name(self) -> str:
        """
        Get the camera friendly name.

        Returns:
            str: Camera friendly name (e.g., "HD Webcam") or fallback name
        """
        if self.camera_name:
            return self.camera_name
        return f"Camera {self.camera_index}"

    def _receive_metadata(self) -> bool:
        """
        Receive camera metadata from Windows sender.

        The metadata message format is: [topic][message_type][metadata_json]
        where message_type = b"META"

        Returns:
            bool: True if metadata received successfully, False otherwise
        """
        try:
            # Set temporary longer timeout for metadata handshake
            original_timeout = self.socket.getsockopt(zmq.RCVTIMEO)
            self.socket.setsockopt(zmq.RCVTIMEO, self.connection_timeout_ms)

            try:
                # Receive multipart message: [topic][message_type][metadata_json]
                parts = self.socket.recv_multipart()

                if len(parts) != 3:
                    logger.debug(f"Unexpected message format during metadata handshake: {len(parts)} parts")
                    return False

                topic, msg_type, payload = parts

                # Validate topic
                if topic != self.topic:
                    logger.debug(f"Topic mismatch in metadata: expected {self.topic}, got {topic}")
                    return False

                # Check if this is a metadata message
                if msg_type != b"META":
                    logger.debug(f"Not a metadata message: type={msg_type}")
                    return False

                # Parse metadata JSON
                import json
                metadata = json.loads(payload.decode('utf-8'))

                if metadata.get('type') == 'camera_info':
                    self.camera_name = metadata.get('camera_name', f"Camera {self.camera_index}")

                    # NEW: Extract resolution from metadata if Windows sender provides it
                    if 'width' in metadata and 'height' in metadata:
                        metadata_width = int(metadata['width'])
                        metadata_height = int(metadata['height'])

                        # Override config resolution with actual stream resolution
                        if (metadata_width, metadata_height) != (self.target_width, self.target_height):
                            logger.warning(f"  [METADATA] Overriding config resolution:")
                            logger.warning(f"    Config: {self.target_width}x{self.target_height}")
                            logger.warning(f"    Metadata (actual stream): {metadata_width}x{metadata_height}")
                            self.target_width = metadata_width
                            self.target_height = metadata_height
                            logger.info(f"  ✓ Resolution updated from metadata: {self.target_width}x{self.target_height}")
                        else:
                            logger.info(f"  ✓ Metadata resolution matches config: {self.target_width}x{self.target_height}")
                    else:
                        logger.warning(f"  [METADATA] No resolution in metadata - using config values: {self.target_width}x{self.target_height}")
                        logger.warning(f"    Consider updating Windows sender to broadcast resolution")

                    logger.info(f"  Received camera metadata: {self.camera_name}")
                    return True
                else:
                    logger.debug(f"Unknown metadata type: {metadata.get('type')}")
                    return False

            finally:
                # Restore original timeout
                self.socket.setsockopt(zmq.RCVTIMEO, original_timeout)

        except zmq.Again:
            # Timeout - no metadata received
            logger.debug("Timeout waiting for metadata")
            return False
        except Exception as e:
            logger.warning(f"Error receiving metadata: {e}")
            return False

    def _receive_frame(self, drain_queue: bool = True) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Receive and decode a single frame from ZMQ.

        Implements manual frame conflation by draining the receive queue
        to always get the latest frame (prevents delay with buffered streams).

        Format: [topic][message_type][length][payload]
        where message_type = b"FRAM" for video frames, b"META" for metadata

        Args:
            drain_queue: If True, drain all pending frames and return only the latest

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        latest_frame = None
        frames_received = 0

        try:
            # Drain queue: read all available frames (non-blocking)
            while drain_queue:
                try:
                    # Non-blocking receive
                    parts = self.socket.recv_multipart(zmq.NOBLOCK)

                    # Parse 4-part format: [topic][msg_type][length][payload]
                    if len(parts) == 4:
                        topic, msg_type, length_bytes, payload = parts

                        # Handle H.264 NAL units
                        if msg_type == b"H264_NAL":
                            # Initialize decoder on first H264 message
                            if self.h264_decoder is None:
                                logger.warning(f"[ZMQ DIAGNOSTIC] Initializing H.264 decoder with dimensions from CONFIG:")
                                logger.warning(f"[ZMQ DIAGNOSTIC]   Config resolution: {self.target_width}x{self.target_height}")
                                logger.warning(f"[ZMQ DIAGNOSTIC]   If Windows sender uses different resolution, decoder may fail!")
                                logger.warning(f"[ZMQ DIAGNOSTIC]   Check Windows sender script for --width/--height parameters")
                                self.h264_decoder = FFmpegH264Decoder(self.target_width, self.target_height, self.target_fps)
                                self.encoding_format = 'H264'
                                logger.info(f"[ZMQ] H.264 decoder initialized for {self.target_width}x{self.target_height} @ {self.target_fps} FPS")

                            # DEBUG: Log received NAL unit details
                            if frames_received < 10 or frames_received % 100 == 0:
                                length_val = struct.unpack("!I", length_bytes)[0]
                                header = payload[:min(8, len(payload))].hex()
                                logger.info(f"[ZMQ DEBUG] Received H264_NAL: size={len(payload)} bytes (length={length_val}), header={header}")

                            # Feed NAL unit to decoder
                            self.h264_decoder.feed_nal_unit(payload)

                            # Try to get decoded frame (allow up to 50ms for CUDA decode)
                            # Increased from 0.001s to 0.05s to prevent dropped frames
                            frame = self.h264_decoder.get_frame(timeout=0.05)
                            if frame is not None:
                                latest_frame = frame
                                frames_received += 1
                            continue

                        # Skip non-frame messages
                        if msg_type != b"FRAM":
                            continue

                    # Parse 3-part format (legacy/metadata)
                    elif len(parts) == 3:
                        topic, second_part, payload = parts
                        # Skip metadata messages
                        if second_part == b"META":
                            continue
                        length_bytes = second_part
                    else:
                        continue

                    # Validate topic
                    if topic != self.topic:
                        continue

                    # Validate length
                    (expected_len,) = struct.unpack("!I", length_bytes)
                    if expected_len != len(payload):
                        logger.warning(f"Length mismatch: expected {expected_len}, got {len(payload)}")
                        continue

                    # Decode JPEG
                    arr = np.frombuffer(payload, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        latest_frame = frame
                        frames_received += 1
                    else:
                        logger.error(f"cv2.imdecode returned None (invalid JPEG?)")

                except zmq.Again:
                    # No more frames available - exit drain loop
                    break

            # If no frames were drained, try one blocking receive
            if latest_frame is None:
                parts = self.socket.recv_multipart()  # Blocking receive

                # Parse frame (same logic as above)
                if len(parts) == 4:
                    topic, msg_type, length_bytes, payload = parts

                    # Handle H.264 NAL units in blocking path
                    if msg_type == b"H264_NAL":
                        # Initialize decoder on first H264 message
                        if self.h264_decoder is None:
                            logger.warning(f"[ZMQ DIAGNOSTIC] Initializing H.264 decoder (blocking path) with CONFIG dimensions:")
                            logger.warning(f"[ZMQ DIAGNOSTIC]   Config resolution: {self.target_width}x{self.target_height}")
                            self.h264_decoder = FFmpegH264Decoder(self.target_width, self.target_height, self.target_fps)
                            self.encoding_format = 'H264'

                        # Feed NAL unit to decoder
                        self.h264_decoder.feed_nal_unit(payload)

                        # Wait for decoded frame (blocking with timeout)
                        latest_frame = self.h264_decoder.get_frame(timeout=self.frame_timeout_ms / 1000.0)

                        if latest_frame is None:
                            # Continue loop to get more NAL units
                            return False, None

                    elif msg_type == b"FRAM":
                        # MJPEG frame - decode below
                        pass

                    else:
                        return False, None

                elif len(parts) == 3:
                    topic, second_part, payload = parts
                    if second_part == b"META":
                        return False, None
                    length_bytes = second_part
                    msg_type = b"FRAM"  # 3-part format is legacy MJPEG

                else:
                    logger.error(f"Invalid part count in blocking recv: {len(parts)}")
                    return False, None

                # Only decode MJPEG if we don't have a frame from H.264
                if latest_frame is None and msg_type == b"FRAM":
                    # Validate topic
                    if topic != self.topic:
                        logger.warning(f"Topic mismatch: expected {self.topic}, got {topic}")
                        return False, None

                    # Validate length
                    (expected_len,) = struct.unpack("!I", length_bytes)
                    if expected_len != len(payload):
                        logger.warning(f"Length mismatch: expected {expected_len}, got {len(payload)}")
                        return False, None

                    # Decode JPEG
                    arr = np.frombuffer(payload, dtype=np.uint8)
                    latest_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    if latest_frame is None:
                        logger.error(f"Blocking cv2.imdecode returned None")

            if latest_frame is None:
                return False, None

            # Log if we drained multiple frames (indicates buffering)
            if frames_received > 1:
                logger.debug(f"Drained {frames_received} frames, using latest")

            return True, latest_frame

        except zmq.Again:
            # Timeout - no frame available
            return False, None
        except Exception as e:
            logger.error(f"Exception in _receive_frame: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False, None

    def _close_socket(self):
        """Close ZMQ socket and clean up."""
        if self.socket:
            try:
                self.socket.close(linger=0)
                logger.debug(f"ZMQ socket closed for camera {self.camera_index}")
            except Exception as e:
                logger.warning(f"Error closing socket for camera {self.camera_index}: {e}")
            self.socket = None

    def _terminate_context(self):
        """Terminate ZMQ context to ensure complete cleanup."""
        if self.context:
            try:
                # Terminate context with a short timeout
                # This ensures all sockets are properly closed before process exit
                self.context.term()
                logger.debug(f"ZMQ context terminated for camera {self.camera_index}")
            except Exception as e:
                logger.warning(f"Error terminating context for camera {self.camera_index}: {e}")
            self.context = None

    def _get_windows_host_ip(self) -> Optional[str]:
        """
        Auto-detect Windows host IP from /etc/resolv.conf.

        In WSL2, the Windows host IP appears as the nameserver.

        Returns:
            Optional[str]: Windows host IP or None if not found
        """
        try:
            with open('/etc/resolv.conf', 'r') as f:
                for line in f:
                    if line.startswith('nameserver'):
                        ip = line.split()[1].strip()
                        logger.info(f"Detected Windows host IP: {ip}")
                        return ip
        except Exception as e:
            logger.warning(f"Could not read /etc/resolv.conf: {e}")

        return None
