"""
Camera Source Factory

Creates appropriate camera source instance based on configuration.
Supports auto-discovery of ZMQ cameras when bridge mode is enabled.
"""

import logging
from typing import Dict, List, Optional
from .base import CameraSource
from .v4l2_source import V4L2CameraSource
from .zmq_source import ZMQCameraSource, probe_zmq_cameras

logger = logging.getLogger('CameraSourceFactory')


class CameraSourceFactory:
    """
    Factory class for creating camera sources.

    Supports:
    - V4L2 (local Linux camera devices)
    - ZMQ (network camera bridge from Windows)

    Features:
    - Auto-discovery of ZMQ cameras when bridge mode enabled
    - Dynamic camera enumeration
    - Backward-compatible with explicit per-camera configuration
    """

    @staticmethod
    def discover_cameras(config: Dict, max_cameras: int = 10) -> List[int]:
        """
        Discover available cameras based on configuration.

        If ZMQ bridge is enabled, probes ZMQ endpoints.
        Otherwise, returns list of camera indices from config.

        Args:
            config: Full configuration dictionary
            max_cameras: Maximum number of cameras to discover (ignored for ZMQ - uses zmq_max_cameras)

        Returns:
            List[int]: List of available camera indices
        """
        zmq_settings = config.get('zmq_camera_bridge', {})

        if zmq_settings.get('enabled', False):
            # ZMQ bridge mode - probe for available cameras
            # Use system-wide max_cameras setting (single source of truth)
            max_cameras = config.get('system', {}).get('max_cameras', 10)
            logger.info(f"ZMQ bridge enabled - discovering cameras via network (max: {max_cameras})...")
            available = probe_zmq_cameras(config, max_cameras=max_cameras)

            if available:
                logger.info(f"ZMQ camera discovery: found cameras {available}")
                # Camera names will be injected into config by gui.py for subprocess access
            else:
                logger.warning("ZMQ camera discovery: no cameras found")

            return available
        else:
            # Traditional mode - use camera_settings from config
            logger.info("ZMQ bridge disabled - using configured cameras")
            camera_settings = config.get('camera_settings', {})

            available = []
            for i in range(max_cameras):
                camera_key = f"camera_{i}"
                if camera_key in camera_settings:
                    cam_config = camera_settings[camera_key]
                    if cam_config.get('enabled', False):
                        available.append(i)

            logger.info(f"Configured cameras: {available}")
            return available

    @staticmethod
    def create(config: Dict, camera_index: int) -> CameraSource:
        """
        Create a camera source instance based on configuration.

        When ZMQ bridge is enabled, automatically creates ZMQ sources.
        Otherwise, uses explicit source_type from camera_settings.

        Args:
            config: Full configuration dictionary
            camera_index: Camera device index

        Returns:
            CameraSource: Instance of appropriate camera source

        Raises:
            ValueError: If source_type is invalid or required config is missing
        """
        zmq_settings = config.get('zmq_camera_bridge', {})
        camera_key = f"camera_{camera_index}"

        # Check if ZMQ bridge mode is enabled
        if zmq_settings.get('enabled', False):
            logger.info(f"[DIAGNOSTIC] ZMQ bridge enabled for camera {camera_index}")
            logger.info(f"Creating ZMQ camera source for camera {camera_index} (auto-enumeration mode)")

            # In ZMQ bridge mode, all cameras are ZMQ sources
            # Create a temporary config entry if it doesn't exist
            camera_settings = config.get('camera_settings', {})

            if camera_key not in camera_settings:
                logger.info(f"[DIAGNOSTIC] Camera {camera_index} has no explicit config - auto-generating")
                # Auto-generate camera config for ZMQ
                logger.info(f"Auto-generating ZMQ config for camera {camera_index}")

                # Get Windows IP (explicit config takes priority)
                windows_ip = zmq_settings.get('windows_host_ip')
                logger.info(f"[DIAGNOSTIC] Windows IP from config: {windows_ip}")

                if not windows_ip:
                    # Fall back to auto-detection
                    logger.info(f"[DIAGNOSTIC] No explicit Windows IP - attempting auto-detection")
                    from .zmq_source import _get_windows_host_ip_static
                    windows_ip = _get_windows_host_ip_static()
                    logger.info(f"[DIAGNOSTIC] Auto-detected Windows IP: {windows_ip}")

                if not windows_ip:
                    raise ValueError(f"Cannot auto-configure camera {camera_index}: Windows host IP not found")

                # Create temporary config
                base_port = zmq_settings.get('port_range', [5551, 5560])[0]
                port = base_port + camera_index
                topic = f"cam{camera_index}"
                endpoint = f"tcp://{windows_ip}:{port}"

                default_res = zmq_settings.get('default_resolution', [1280, 720])
                default_fps = zmq_settings.get('default_fps', 30)

                logger.info(f"[DIAGNOSTIC] Auto-generated config for camera {camera_index}:")
                logger.info(f"[DIAGNOSTIC]   Endpoint: {endpoint}")
                logger.info(f"[DIAGNOSTIC]   Topic: {topic}")
                logger.info(f"[DIAGNOSTIC]   Resolution: {default_res[0]}x{default_res[1]}")
                logger.info(f"[DIAGNOSTIC]   FPS: {default_fps}")

                auto_config = config.copy()
                if 'camera_settings' not in auto_config:
                    auto_config['camera_settings'] = {}

                auto_config['camera_settings'][camera_key] = {
                    'enabled': True,
                    'source_type': 'zmq',
                    'zmq_endpoint': endpoint,
                    'zmq_topic': topic,
                    'width': default_res[0],
                    'height': default_res[1],
                    'fps': default_fps
                }

                # Lookup camera name from config (works across process boundaries)
                # Module-level cache doesn't persist to subprocesses, so we read from config
                zmq_settings_local = auto_config.get('zmq_camera_bridge', {})
                discovered_names = zmq_settings_local.get('discovered_camera_names', {})
                # JSON serialization converts int keys to strings, must match when looking up
                camera_name = discovered_names.get(str(camera_index))
                if camera_name:
                    logger.info(f"[DIAGNOSTIC] Using camera name from config: {camera_name}")
                else:
                    logger.debug(f"[DIAGNOSTIC] No camera name in config for camera {camera_index}")

                logger.info(f"[DIAGNOSTIC] Creating ZMQCameraSource instance for camera {camera_index}")
                return ZMQCameraSource(config=auto_config, camera_index=camera_index, camera_name=camera_name)

            # Existing config - just use it
            logger.info(f"[DIAGNOSTIC] Camera {camera_index} has explicit config - using existing settings")
            existing_cam_config = camera_settings.get(camera_key, {})
            logger.info(f"[DIAGNOSTIC]   Endpoint: {existing_cam_config.get('zmq_endpoint', 'N/A')}")
            logger.info(f"[DIAGNOSTIC]   Topic: {existing_cam_config.get('zmq_topic', 'N/A')}")

            # Lookup camera name from config (works across process boundaries)
            zmq_settings_local = config.get('zmq_camera_bridge', {})
            discovered_names = zmq_settings_local.get('discovered_camera_names', {})
            # JSON serialization converts int keys to strings, must match when looking up
            camera_name = discovered_names.get(str(camera_index))
            if camera_name:
                logger.info(f"[DIAGNOSTIC] Using camera name from config: {camera_name}")

            return ZMQCameraSource(config=config, camera_index=camera_index, camera_name=camera_name)

        # Traditional mode - use explicit source_type
        camera_settings = config.get('camera_settings', {}).get(camera_key, {})

        if not camera_settings:
            raise ValueError(f"No configuration found for {camera_key}")

        # Get source type (default to V4L2 for backward compatibility)
        source_type = camera_settings.get('source_type', 'v4l2').lower()

        logger.info(f"Creating camera source for camera {camera_index}: type={source_type}")

        if source_type == 'v4l2':
            return V4L2CameraSource(config=config, camera_index=camera_index)

        elif source_type == 'zmq':
            return ZMQCameraSource(config=config, camera_index=camera_index)

        else:
            raise ValueError(f"Unknown camera source type: {source_type}. "
                           f"Supported types: 'v4l2', 'zmq'")

    @staticmethod
    def get_supported_types() -> list:
        """
        Get list of supported camera source types.

        Returns:
            list: List of source type strings
        """
        return ['v4l2', 'zmq']
