"""
Platform Detection Module

Centralized platform detection for WSL2 vs Native Linux.
Controls all platform-specific behavior via single config setting.

Usage:
    from core.platform_detection import PlatformContext

    platform = PlatformContext.detect(config)
    if platform['mode'] == 'wsl2':
        # WSL2-specific logic
    elif platform['mode'] == 'native_linux':
        # Native Linux-specific logic
"""

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger('PlatformDetection')


class PlatformContext:
    """
    Platform detection and configuration for WSL2 vs Native Linux.

    Provides single source of truth for platform-specific behavior:
    - Camera discovery method (usbipd, v4l2_native, zmq)
    - GPU backend (d3d12, opengl)
    - Display mode (wslg, x11, wayland)
    """

    # Valid platform modes
    MODES = ['auto', 'wsl2', 'native_linux']

    # Camera discovery modes
    CAMERA_DISCOVERY_MODES = ['usbipd', 'v4l2_native', 'zmq']

    # GPU backends
    GPU_BACKENDS = ['d3d12', 'opengl']

    # Display modes
    DISPLAY_MODES = ['wslg', 'x11', 'wayland']

    @staticmethod
    def _detect_wsl2() -> bool:
        """
        Detect if running in WSL2 environment.

        Detection methods:
        1. Check for WSLInterop file (most reliable)
        2. Check /proc/version for Microsoft kernel
        3. Check /etc/resolv.conf for Windows nameserver

        Returns:
            bool: True if WSL2 detected, False if native Linux
        """
        # Method 1: WSLInterop file (most reliable)
        if os.path.exists('/proc/sys/fs/binfmt_misc/WSLInterop'):
            logger.debug("WSL2 detected: /proc/sys/fs/binfmt_misc/WSLInterop exists")
            return True

        # Method 2: Check /proc/version for Microsoft kernel
        try:
            with open('/proc/version', 'r') as f:
                version_str = f.read().lower()
                if 'microsoft' in version_str or 'wsl' in version_str:
                    logger.debug(f"WSL2 detected: /proc/version contains Microsoft/WSL: {version_str[:100]}")
                    return True
        except:
            pass

        # Method 3: Check for Windows nameserver in /etc/resolv.conf
        try:
            with open('/etc/resolv.conf', 'r') as f:
                for line in f:
                    if line.startswith('nameserver'):
                        # WSL2 typically uses 172.x.x.x range for Windows host
                        ip = line.split()[1].strip()
                        if ip.startswith('172.') or ip.startswith('192.168.'):
                            logger.debug(f"Possible WSL2: Windows-like nameserver {ip} in /etc/resolv.conf")
                            # This is less reliable, so don't return True immediately
                            # Only use as fallback confirmation
        except:
            pass

        logger.debug("Native Linux detected: no WSL2 indicators found")
        return False

    @staticmethod
    def _get_windows_host_ip() -> Optional[str]:
        """
        Get Windows host IP from /etc/resolv.conf (WSL2 only).

        Returns:
            Optional[str]: Windows host IP or None if not found/not WSL2
        """
        try:
            with open('/etc/resolv.conf', 'r') as f:
                for line in f:
                    if line.startswith('nameserver'):
                        ip = line.split()[1].strip()
                        return ip
        except Exception as e:
            logger.debug(f"Could not read /etc/resolv.conf: {e}")

        return None

    @staticmethod
    def detect(config: Optional[Dict] = None) -> Dict:
        """
        Detect platform mode and return context.

        Args:
            config: Configuration dictionary (optional)
                    If provided, respects config['platform']['mode'] override

        Returns:
            Dict with keys:
                - mode: 'wsl2' or 'native_linux'
                - detected: bool (True if auto-detected, False if manual override)
                - windows_host_ip: str (WSL2 only, None for native)
                - camera_discovery: str (usbipd/v4l2_native/zmq)
                - gpu_backend: str (d3d12/opengl)
                - display_mode: str (wslg/x11/wayland)
        """
        # Get platform config section
        platform_config = {}
        if config:
            platform_config = config.get('platform', {})

        configured_mode = platform_config.get('mode', 'auto')

        # Auto-detect or use configured mode
        if configured_mode == 'auto':
            is_wsl2 = PlatformContext._detect_wsl2()
            detected_mode = 'wsl2' if is_wsl2 else 'native_linux'
            detected = True
            logger.info(f"Platform auto-detected: {detected_mode}")
        elif configured_mode in ['wsl2', 'native_linux']:
            detected_mode = configured_mode
            detected = False
            logger.info(f"Platform mode manually configured: {detected_mode}")
        else:
            logger.warning(f"Invalid platform mode '{configured_mode}', falling back to auto-detection")
            is_wsl2 = PlatformContext._detect_wsl2()
            detected_mode = 'wsl2' if is_wsl2 else 'native_linux'
            detected = True

        # Build platform context
        context = {
            'mode': detected_mode,
            'detected': detected,
            'windows_host_ip': None
        }

        # WSL2-specific: Get Windows host IP
        if detected_mode == 'wsl2':
            # Check config first, then fall back to auto-detection
            if config:
                zmq_settings = config.get('zmq_camera_bridge', {})
                config_ip = zmq_settings.get('windows_host_ip')
                if config_ip:
                    context['windows_host_ip'] = config_ip
                    logger.info(f"Using Windows host IP from config: {config_ip}")
                else:
                    context['windows_host_ip'] = PlatformContext._get_windows_host_ip()
                    logger.info(f"Auto-detected Windows host IP: {context['windows_host_ip']}")
            else:
                context['windows_host_ip'] = PlatformContext._get_windows_host_ip()
                logger.info(f"Auto-detected Windows host IP: {context['windows_host_ip']}")

        # Determine camera discovery mode
        context['camera_discovery'] = PlatformContext.get_camera_discovery_mode(config, detected_mode)

        # Determine GPU backend
        context['gpu_backend'] = PlatformContext.get_gpu_backend(config, detected_mode)

        # Determine display mode
        context['display_mode'] = PlatformContext.get_display_mode(config, detected_mode)

        logger.info(f"Platform context: {context}")
        return context

    @staticmethod
    def get_camera_discovery_mode(config: Optional[Dict], platform_mode: str) -> str:
        """
        Determine camera discovery mode based on platform and config.

        Args:
            config: Configuration dictionary
            platform_mode: 'wsl2' or 'native_linux'

        Returns:
            str: 'usbipd', 'v4l2_native', or 'zmq'
        """
        # Check for manual override
        if config:
            platform_config = config.get('platform', {})
            override = platform_config.get('force_camera_discovery')
            if override in PlatformContext.CAMERA_DISCOVERY_MODES:
                logger.info(f"Camera discovery mode overridden: {override}")
                return override

        # Check if ZMQ bridge is enabled (takes precedence)
        if config:
            zmq_settings = config.get('zmq_camera_bridge', {})
            if zmq_settings.get('enabled', False):
                logger.info("Camera discovery: ZMQ bridge enabled in config")
                return 'zmq'

        # Platform-specific defaults
        if platform_mode == 'wsl2':
            return 'usbipd'  # WSL2 default: USB passthrough via usbipd
        else:
            return 'v4l2_native'  # Native Linux: direct V4L2 access

    @staticmethod
    def get_gpu_backend(config: Optional[Dict], platform_mode: str) -> str:
        """
        Determine GPU backend based on platform and config.

        Args:
            config: Configuration dictionary
            platform_mode: 'wsl2' or 'native_linux'

        Returns:
            str: 'd3d12' or 'opengl'
        """
        # Check for manual override
        if config:
            platform_config = config.get('platform', {})
            override = platform_config.get('force_gpu_backend')
            if override in PlatformContext.GPU_BACKENDS:
                logger.info(f"GPU backend overridden: {override}")
                return override

        # Platform-specific defaults
        if platform_mode == 'wsl2':
            return 'd3d12'  # WSL2: OpenGL ES → D3D12 translation
        else:
            return 'opengl'  # Native Linux: direct OpenGL/OpenGL ES

    @staticmethod
    def get_display_mode(config: Optional[Dict], platform_mode: str) -> str:
        """
        Determine display mode based on platform and config.

        Args:
            config: Configuration dictionary
            platform_mode: 'wsl2' or 'native_linux'

        Returns:
            str: 'wslg', 'x11', or 'wayland'
        """
        # Check for manual override
        if config:
            platform_config = config.get('platform', {})
            override = platform_config.get('force_display_mode')
            if override in PlatformContext.DISPLAY_MODES:
                logger.info(f"Display mode overridden: {override}")
                return override

        # Platform-specific defaults
        if platform_mode == 'wsl2':
            return 'wslg'  # WSL2: WSLg (Wayland + XWayland)
        else:
            # Native Linux: Auto-detect X11 vs Wayland
            if os.environ.get('WAYLAND_DISPLAY'):
                return 'wayland'
            else:
                return 'x11'

    @staticmethod
    def is_wsl2(config: Optional[Dict] = None) -> bool:
        """
        Convenience method to check if running in WSL2.

        Args:
            config: Configuration dictionary (optional)

        Returns:
            bool: True if WSL2, False if native Linux
        """
        context = PlatformContext.detect(config)
        return context['mode'] == 'wsl2'

    @staticmethod
    def is_native_linux(config: Optional[Dict] = None) -> bool:
        """
        Convenience method to check if running on native Linux.

        Args:
            config: Configuration dictionary (optional)

        Returns:
            bool: True if native Linux, False if WSL2
        """
        context = PlatformContext.detect(config)
        return context['mode'] == 'native_linux'


# Convenience function for backward compatibility
def detect_platform(config: Optional[Dict] = None) -> Dict:
    """
    Convenience wrapper for PlatformContext.detect().

    Args:
        config: Configuration dictionary (optional)

    Returns:
        Dict: Platform context
    """
    return PlatformContext.detect(config)
