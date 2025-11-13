"""
ZMQ Bluetooth Source Backend

Receives Bluetooth device data from Windows bridge via ZeroMQ.
Used in WSL2 where direct Bluetooth access is not available.

The Windows bridge (win_bluetooth_bridge.py) handles actual Bluetooth
communication and sends data to WSL2 over TCP/ZMQ.
"""

import zmq
import time
import json
from typing import List, Optional, Callable, Dict, Any
import socket

from .base import BluetoothDeviceBackend, DeviceInfo, StreamInfo, ConnectionState


def _detect_windows_host_ip() -> str:
    """
    Auto-detect Windows host IP from WSL2.

    Uses default gateway (more reliable than resolv.conf).

    Returns:
        Windows host IP address
    """
    try:
        import subprocess
        # Use default gateway (Windows host) instead of resolv.conf
        result = subprocess.run(['ip', 'route', 'show'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'default' in line:
                parts = line.split()
                if len(parts) >= 3 and parts[0] == 'default':
                    ip = parts[2]
                    print(f"[ZMQBluetoothDiscover] Detected Windows IP via default gateway: {ip}")
                    return ip
    except Exception as e:
        print(f"[ZMQBluetoothDiscover] Failed to detect Windows IP via ip route: {e}")

    # Fallback to resolv.conf
    try:
        with open('/etc/resolv.conf', 'r') as f:
            for line in f:
                if line.startswith('nameserver'):
                    ip = line.split()[1]
                    print(f"[ZMQBluetoothDiscover] Using Windows IP from resolv.conf: {ip}")
                    return ip
    except Exception as e:
        print(f"[ZMQBluetoothDiscover] Failed to read resolv.conf: {e}")

    # Final fallback
    print(f"[ZMQBluetoothDiscover] WARNING: Using localhost fallback")
    return "127.0.0.1"


def _check_discovery_health(windows_ip: str, port: int) -> bool:
    """
    TCP socket health check for discovery service.

    Args:
        windows_ip: Windows host IP
        port: Discovery port (5650)

    Returns:
        True if port is reachable
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        result = sock.connect_ex((windows_ip, port))
        return result == 0
    finally:
        sock.close()


def _try_receive_manifest(windows_ip: str, port: int, timeout_ms: int) -> Optional[Dict[str, Any]]:
    """
    Receive manifest via ZMQ SUB socket.

    Args:
        windows_ip: Windows host IP
        port: Discovery port (5650)
        timeout_ms: Receive timeout in milliseconds

    Returns:
        Manifest dict if received, None otherwise
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b'')
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)

    try:
        sock.connect(f"tcp://{windows_ip}:{port}")

        # Sleep to solve ZeroMQ slow joiner problem
        time.sleep(0.3)

        # Receive manifest
        manifest_bytes = sock.recv()
        manifest = json.loads(manifest_bytes.decode('utf-8'))

        if manifest.get('type') == 'bluetooth_manifest':
            return manifest

    except zmq.Again:
        print(f"[ZMQBluetoothDiscover] Timeout waiting for manifest")
        return None
    except Exception as e:
        print(f"[ZMQBluetoothDiscover] Manifest receive error: {e}")
        return None
    finally:
        sock.close()
        ctx.term()

    return None


def _save_discovered_devices(config: Dict[str, Any], devices: List[DeviceInfo]):
    """
    Save discovered devices to config (indexed by MAC address).

    Args:
        config: Configuration dict
        devices: List of discovered devices
    """
    discovered = {
        dev.mac_address: {
            'name': dev.name,
            'port': dev.metadata['port'],
            'topic': dev.metadata['topic'],
            'device_index': dev.metadata['device_index']
        }
        for dev in devices
    }

    # Update config (assumes config object has a set method)
    if hasattr(config, 'set'):
        config.set('bluetooth_bridge.discovered_devices', discovered)
    else:
        # Fallback: direct dict update
        if 'bluetooth_bridge' not in config:
            config['bluetooth_bridge'] = {}
        config['bluetooth_bridge']['discovered_devices'] = discovered


def discover_zmq_bluetooth_devices(config: Dict[str, Any], timeout_ms: int = 20000) -> List[DeviceInfo]:
    """
    Discover Bluetooth devices via ZMQ manifest (port 5650).

    This function connects to the Windows bridge discovery service
    and receives a manifest of available Bluetooth devices.

    Args:
        config: Configuration dict with bluetooth_bridge settings
        timeout_ms: Discovery timeout in milliseconds

    Returns:
        List[DeviceInfo] with device details (NO participant assignment)
    """
    print("[ZMQBluetoothDiscover] Starting discovery via Windows bridge manifest...")

    # Get configuration
    bridge_config = config.get('bluetooth_bridge', {})
    windows_ip = bridge_config.get('windows_host_ip', 'auto')

    if windows_ip == 'auto':
        windows_ip = _detect_windows_host_ip()

    discovery_port = bridge_config.get('discovery_port', 5650)

    print(f"[ZMQBluetoothDiscover] Connecting to {windows_ip}:{discovery_port}")

    # Phase 1: Health check
    print("[ZMQBluetoothDiscover] Phase 1: Health check...")
    if not _check_discovery_health(windows_ip, discovery_port):
        print(f"[ZMQBluetoothDiscover] Discovery service not reachable at {windows_ip}:{discovery_port}")
        print("[ZMQBluetoothDiscover] Make sure Windows bridge manager is running:")
        print("[ZMQBluetoothDiscover]   python zmq_bluetooth_manager.py --silent")
        return []

    print("[ZMQBluetoothDiscover] ✓ Discovery service is reachable")

    # Phase 2: Receive manifest
    print("[ZMQBluetoothDiscover] Phase 2: Receiving manifest...")
    manifest = _try_receive_manifest(windows_ip, discovery_port, timeout_ms)

    if not manifest:
        print("[ZMQBluetoothDiscover] No manifest received (timeout)")
        return []

    print(f"[ZMQBluetoothDiscover] ✓ Manifest received with {len(manifest.get('devices', []))} device(s)")

    # Check adapter status from manifest
    adapter_status = manifest.get('adapter_status', 'unknown')
    if adapter_status != 'ready':
        print(f"[ZMQBluetoothDiscover] ⚠️  Windows Bluetooth adapter status: {adapter_status}")
        print(f"[ZMQBluetoothDiscover] Check Windows bridge console for troubleshooting guidance")
        if adapter_status in ['adapter_not_ready', 'adapter not ready']:
            print(f"[ZMQBluetoothDiscover] Suggestion: Wait a few seconds, then restart Windows bridge")
        elif adapter_status in ['adapter_not_found', 'adapter not found']:
            print(f"[ZMQBluetoothDiscover] Suggestion: Ensure Bluetooth adapter is installed and enabled on Windows")
    else:
        print(f"[ZMQBluetoothDiscover] ✓ Windows Bluetooth adapter is ready")

    # Phase 3: Build device list from manifest
    print("[ZMQBluetoothDiscover] Phase 3: Building device list...")
    devices = []

    for dev_info in manifest.get('devices', []):
        device = DeviceInfo(
            mac_address=dev_info['mac_address'],
            name=dev_info['name'],
            device_type='zmq_bluetooth',  # Backend identifier for registry
            rssi=None,  # N/A for ZMQ
            metadata={
                'port': dev_info['port'],  # 5651, 5652, etc.
                'topic': dev_info['topic'],  # bt_device0, bt_device1, etc.
                'device_index': dev_info['device_index']  # 0, 1, 2, ...
            }
        )
        devices.append(device)
        print(f"[ZMQBluetoothDiscover]   - {device.name} ({device.mac_address}) → port {dev_info['port']}")

    # Save to config for future use
    _save_discovered_devices(config, devices)
    print(f"[ZMQBluetoothDiscover] ✓ Saved {len(devices)} device(s) to config")

    return devices


class ZMQBluetoothSource(BluetoothDeviceBackend):
    """
    Bluetooth backend that receives data from Windows bridge via ZMQ.

    This backend does NOT connect to Bluetooth directly. Instead, it subscribes
    to a ZMQ stream from the Windows bridge script running on the host.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ZMQ Bluetooth source.

        Args:
            config: Configuration dict with bluetooth_bridge settings
        """
        super().__init__()
        self.config = config or {}
        self.windows_host_ip = self._get_windows_ip()
        self.context = None
        self.socket = None
        self.port: Optional[int] = None
        self.topic: Optional[str] = None
        self.device_name: Optional[str] = None
        self._stream_info: Optional[StreamInfo] = None
        self._last_sample_time = 0.0
        self._last_timeout_log = 0.0  # For periodic timeout logging

    def _get_windows_ip(self) -> str:
        """Get Windows host IP from config or auto-detect"""
        bridge_config = self.config.get('bluetooth_bridge', {})
        windows_ip = bridge_config.get('windows_host_ip', 'auto')

        if windows_ip == 'auto':
            return _detect_windows_host_ip()

        return windows_ip

    @classmethod
    def get_backend_identifier(cls) -> str:
        return "zmq_bluetooth"

    @classmethod
    def matches_device(cls, device_name: str) -> bool:
        """ZMQ sources don't match by name - they're assigned explicitly"""
        return False

    async def discover(self, timeout: float = 10.0) -> List[DeviceInfo]:
        """
        Discover devices via Windows bridge manifest.

        Args:
            timeout: Discovery timeout in seconds

        Returns:
            List of devices discovered by Windows bridge
        """
        timeout_ms = int(timeout * 1000)
        devices = discover_zmq_bluetooth_devices(self.config, timeout_ms)
        return devices

    async def connect(self, mac_address: str) -> bool:
        """
        Connect to ZMQ stream for device with given MAC address.

        Looks up port/topic from discovered_devices config (indexed by MAC).

        Args:
            mac_address: Device MAC address (e.g., "AA:BB:CC:DD:EE:FF")

        Returns:
            True if ZMQ connection successful
        """
        try:
            self.state = ConnectionState.CONNECTING
            self.mac_address = mac_address

            # Lookup device info from discovered_devices (indexed by MAC)
            bridge_config = self.config.get('bluetooth_bridge', {})
            discovered = bridge_config.get('discovered_devices', {})

            if mac_address in discovered:
                # Use port/topic from discovery
                dev_info = discovered[mac_address]
                self.port = dev_info['port']
                self.topic = dev_info['topic']
                self.device_name = dev_info['name']
                print(f"[ZMQBluetoothSource] Using discovered device: {self.device_name}")
                print(f"[ZMQBluetoothSource]   MAC: {mac_address}")
                print(f"[ZMQBluetoothSource]   Port: {self.port}")
                print(f"[ZMQBluetoothSource]   Topic: {self.topic}")
            else:
                # Fallback: device not in manifest
                print(f"[ZMQBluetoothSource] ERROR: Device {mac_address} not found in discovered_devices")
                print(f"[ZMQBluetoothSource] Available devices:")
                for mac, info in discovered.items():
                    print(f"[ZMQBluetoothSource]   - {info['name']} ({mac}) → port {info['port']}")
                if not discovered:
                    print(f"[ZMQBluetoothSource]   (No devices discovered)")
                    print(f"[ZMQBluetoothSource] Make sure Windows bridge manager is running:")
                    print(f"[ZMQBluetoothSource]   python zmq_bluetooth_manager.py --silent")
                self.state = ConnectionState.ERROR
                return False

            # Create ZMQ context and socket
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)

            # Connect to Windows bridge
            endpoint = f"tcp://{self.windows_host_ip}:{self.port}"
            self.socket.connect(endpoint)
            self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)  # Subscribe to specific topic

            # Set receive timeout
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout

            print(f"[ZMQBluetoothSource] Connected to {endpoint} (topic: {self.topic})")

            # Test connection by trying to receive one message
            try:
                message_str = self.socket.recv_string()
                # Parse: "bt_device0 {json_data}"
                topic, data_str = message_str.split(' ', 1)
                data = json.loads(data_str)
                print(f"[ZMQBluetoothSource] ✓ Received test message from bridge")
                self.state = ConnectionState.CONNECTED
                return True
            except zmq.Again:
                print(f"[ZMQBluetoothSource] No data from bridge (timeout)")
                print(f"[ZMQBluetoothSource] Make sure Windows bridge is running for this device")
                self.state = ConnectionState.ERROR
                return False

        except Exception as e:
            print(f"[ZMQBluetoothSource] Connection error: {e}")
            import traceback
            traceback.print_exc()
            self.state = ConnectionState.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect from ZMQ stream"""
        try:
            if self.socket:
                self.socket.close()
            if self.context:
                self.context.term()

            self.state = ConnectionState.DISCONNECTED
            print(f"[ZMQBluetoothSource] Disconnected from {self.windows_host_ip}:{self.port}")

        except Exception as e:
            print(f"[ZMQBluetoothSource] Disconnect error: {e}")

    async def start_streaming(self, data_callback: Callable[[List[float]], None]) -> bool:
        """
        Start receiving data from ZMQ stream.

        Args:
            data_callback: Function to call with received samples

        Returns:
            True if streaming started
        """
        if not self.is_connected():
            print("[ZMQBluetoothSource] Not connected, cannot start streaming")
            return False

        self._data_callback = data_callback
        self.state = ConnectionState.STREAMING
        print("[ZMQBluetoothSource] Streaming started")
        return True

    async def stop_streaming(self) -> None:
        """Stop receiving data"""
        self.state = ConnectionState.CONNECTED
        print("[ZMQBluetoothSource] Streaming stopped")

    def get_stream_info(self) -> StreamInfo:
        """
        Get LSL stream metadata.

        Returns:
            StreamInfo for ECG data (Polar H10 default)
        """
        # Default to Polar H10 ECG
        # In production, this should be detected from bridge metadata
        if self._stream_info:
            return self._stream_info

        return StreamInfo(
            stream_type="ecg",
            channel_count=1,
            nominal_srate=130.0,
            channel_format="float32",
            channel_names=["ECG"],
            channel_units=["microvolts"],
            manufacturer="Polar",
            model="H10 (via ZMQ Bridge)"
        )

    def parse_data(self, data: bytearray) -> List[float]:
        """
        Not used for ZMQ source (data comes as JSON).

        Args:
            data: Raw bytearray (not used)

        Returns:
            Empty list
        """
        return []

    def receive_samples(self, timeout_ms: int = 100) -> Optional[List[float]]:
        """
        Receive samples from ZMQ stream (blocking with timeout).

        This should be called in a loop when streaming is active.

        Args:
            timeout_ms: Receive timeout in milliseconds

        Returns:
            List of samples if received, None if timeout/error
        """
        if not self.is_streaming():
            return None

        try:
            # Set socket timeout
            self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

            # Receive message with topic prefix: "bt_device0 {json_data}"
            message_str = self.socket.recv_string()

            # Parse topic and data
            topic, data_str = message_str.split(' ', 1)
            message = json.loads(data_str)

            # Extract samples
            samples = message.get('samples', [])

            # DEBUG: Log message reception
            print(f"[ZMQ DEBUG] Received message: topic={topic}, samples={len(samples)}, "
                  f"battery={message.get('battery', 'N/A')}")

            # Update battery level
            battery = message.get('battery')
            if battery is not None:
                self._battery_level = battery

            # Send to callback
            if self._data_callback and samples:
                print(f"[ZMQ DEBUG] Calling data_callback with {len(samples)} samples")
                self._data_callback(samples)
            elif not self._data_callback:
                print(f"[ZMQ DEBUG] WARNING: No data_callback registered, samples not routed!")
            elif not samples:
                print(f"[ZMQ DEBUG] WARNING: Received message with no samples!")

            self._last_sample_time = time.time()
            return samples

        except zmq.Again:
            # Timeout - this is normal if no data available
            # Log periodically (every 5 seconds) to detect if Windows bridge is not sending
            current_time = time.time()
            if current_time - self._last_timeout_log > 5.0:
                print(f"[ZMQ DEBUG] No data received for 5+ seconds (ZMQ timeout)")
                print(f"[ZMQ DEBUG] Check if Windows bridge is running and sending to port {self.port}")
                self._last_timeout_log = current_time
            return None
        except Exception as e:
            print(f"[ZMQBluetoothSource] Receive error: {e}")
            return None

    async def get_battery_level(self) -> Optional[int]:
        """
        Get battery level from last received message.

        Returns:
            Battery percentage or None
        """
        return getattr(self, '_battery_level', None)
