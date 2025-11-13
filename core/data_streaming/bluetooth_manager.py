"""
Bluetooth Device Manager

This module manages all Bluetooth device connections, streaming, and LSL integration.
Handles multiple devices across multiple participants with automatic reconnection.
"""

import asyncio
import threading
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
import time

from .bluetooth_backends import (
    BluetoothDeviceBackend,
    DeviceInfo,
    StreamInfo,
    ConnectionState,
    get_registry
)


@dataclass
class DeviceAssignment:
    """Represents a Bluetooth device assigned to a participant"""
    participant_id: str  # e.g., "P1", "P2"
    mac_address: str
    device_type: str  # Backend identifier
    enabled: bool = True


@dataclass
class ActiveDevice:
    """Runtime state for an active Bluetooth device"""
    assignment: DeviceAssignment
    backend: BluetoothDeviceBackend
    thread: Optional[threading.Thread] = None
    event_loop: Optional[asyncio.AbstractEventLoop] = None
    reconnect_attempts: int = 0
    last_sample_time: float = 0.0
    sample_buffer: deque = field(default_factory=lambda: deque(maxlen=100))  # For live preview
    battery_level: Optional[int] = None


class BluetoothDeviceManager:
    """
    Central manager for all Bluetooth device operations.

    Responsibilities:
    - Device discovery and assignment
    - Connection management and reconnection
    - Data streaming and LSL integration
    - Status monitoring and battery tracking
    """

    def __init__(self, data_callback: Optional[Callable] = None, max_reconnect_attempts: int = 5, config: Optional[Dict] = None):
        """
        Initialize Bluetooth device manager.

        Args:
            data_callback: Function to call with device data
                          Signature: callback(participant_id, stream_type, samples)
            max_reconnect_attempts: Maximum reconnection attempts per device
            config: Configuration dict (used by backends like ZMQBluetoothSource)
        """
        self.registry = get_registry()
        self.data_callback = data_callback
        self.max_reconnect_attempts = max_reconnect_attempts
        self.config = config or {}

        # Active devices indexed by (participant_id, mac_address)
        self.active_devices: Dict[Tuple[str, str], ActiveDevice] = {}

        # Thread safety
        self._lock = threading.Lock()
        self._running = False

    def get_available_backends(self) -> List[str]:
        """
        Get list of available backend types.

        Returns:
            List of backend identifiers
        """
        return self.registry.get_all_identifiers()

    async def discover_devices(self, backend_type: Optional[str] = None, timeout: float = 10.0) -> List[DeviceInfo]:
        """
        Discover available Bluetooth devices.

        Args:
            backend_type: Specific backend to use, or None for all backends
            timeout: Discovery timeout in seconds

        Returns:
            List of discovered devices
        """
        devices = []

        if backend_type:
            # Discover from specific backend
            backend = self.registry.create_backend(backend_type)
            if backend:
                # Pass config to backend if it supports it
                if hasattr(backend, 'config'):
                    backend.config = self.config
                devices = await backend.discover(timeout)
        else:
            # Discover from all backends
            for identifier in self.registry.get_all_identifiers():
                backend = self.registry.create_backend(identifier)
                if backend:
                    # Pass config to backend if it supports it
                    if hasattr(backend, 'config'):
                        backend.config = self.config
                    backend_devices = await backend.discover(timeout)
                    devices.extend(backend_devices)

        return devices

    def discover_devices_sync(self, backend_type: Optional[str] = None, timeout: float = 10.0) -> List[DeviceInfo]:
        """
        Synchronous wrapper for device discovery (for GUI).

        Args:
            backend_type: Specific backend to use, or None for all backends
            timeout: Discovery timeout in seconds

        Returns:
            List of discovered devices
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            devices = loop.run_until_complete(
                self.discover_devices(backend_type, timeout)
            )
            return devices
        finally:
            loop.close()

    def add_device_assignment(self, participant_id: str, mac_address: str, device_type: str, enabled: bool = True):
        """
        Assign a Bluetooth device to a participant.

        Args:
            participant_id: Participant ID (e.g., "P1")
            mac_address: Device MAC address
            device_type: Backend identifier
            enabled: Whether device should be enabled
        """
        with self._lock:
            assignment = DeviceAssignment(
                participant_id=participant_id,
                mac_address=mac_address,
                device_type=device_type,
                enabled=enabled
            )

            key = (participant_id, mac_address)

            # If device was previously active, stop it
            if key in self.active_devices:
                self._stop_device(key)

            # Create backend instance
            backend = self.registry.create_backend(device_type)
            if not backend:
                print(f"[BluetoothManager] Unknown device type: {device_type}")
                return

            # Pass config to backend if it supports it
            if hasattr(backend, 'config'):
                backend.config = self.config

            # Create active device entry
            self.active_devices[key] = ActiveDevice(
                assignment=assignment,
                backend=backend
            )

            print(f"[BluetoothManager] Assigned {mac_address} ({device_type}) to {participant_id}")

    def remove_device_assignment(self, participant_id: str, mac_address: str):
        """
        Remove a device assignment.

        Args:
            participant_id: Participant ID
            mac_address: Device MAC address
        """
        with self._lock:
            key = (participant_id, mac_address)
            if key in self.active_devices:
                self._stop_device(key)
                del self.active_devices[key]
                print(f"[BluetoothManager] Removed device {mac_address} from {participant_id}")

    def get_participant_devices(self, participant_id: str) -> List[ActiveDevice]:
        """
        Get all devices assigned to a participant.

        Args:
            participant_id: Participant ID

        Returns:
            List of active devices for this participant
        """
        with self._lock:
            return [
                device for (pid, _), device in self.active_devices.items()
                if pid == participant_id
            ]

    def get_all_assignments(self) -> List[DeviceAssignment]:
        """
        Get all device assignments.

        Returns:
            List of all device assignments
        """
        with self._lock:
            return [device.assignment for device in self.active_devices.values()]

    def start_all(self):
        """
        Start streaming from all enabled devices.
        """
        self._running = True

        with self._lock:
            for key, active_device in self.active_devices.items():
                if active_device.assignment.enabled:
                    self._start_device(key)

    def stop_all(self):
        """
        Stop streaming from all devices.
        """
        self._running = False

        with self._lock:
            for key in list(self.active_devices.keys()):
                self._stop_device(key)

    def _start_device(self, key: Tuple[str, str]):
        """
        Start a specific device (internal, assumes lock held).

        Args:
            key: (participant_id, mac_address) tuple
        """
        active_device = self.active_devices.get(key)
        if not active_device:
            return

        # Create thread with asyncio event loop
        def device_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            active_device.event_loop = loop

            try:
                loop.run_until_complete(self._device_lifecycle(active_device))
            except Exception as e:
                print(f"[BluetoothManager] Device thread error: {e}")
            finally:
                loop.close()

        active_device.thread = threading.Thread(target=device_thread, daemon=True)
        active_device.thread.start()

    def _stop_device(self, key: Tuple[str, str]):
        """
        Stop a specific device (internal, assumes lock held).

        Args:
            key: (participant_id, mac_address) tuple
        """
        active_device = self.active_devices.get(key)
        if not active_device:
            return

        # Stop event loop
        if active_device.event_loop:
            active_device.event_loop.call_soon_threadsafe(active_device.event_loop.stop)

        # Wait for thread to finish
        if active_device.thread and active_device.thread.is_alive():
            active_device.thread.join(timeout=2.0)

        active_device.thread = None
        active_device.event_loop = None

    async def _device_lifecycle(self, active_device: ActiveDevice):
        """
        Manage device lifecycle: connect, stream, reconnect on failure.

        Args:
            active_device: Active device instance
        """
        backend = active_device.backend
        assignment = active_device.assignment

        while self._running and active_device.reconnect_attempts < self.max_reconnect_attempts:
            try:
                # Connect to device
                print(f"[BluetoothManager] Connecting to {assignment.mac_address}...")
                connected = await backend.connect(assignment.mac_address)

                if not connected:
                    raise ConnectionError("Failed to connect")

                # Read battery level
                battery = await backend.get_battery_level()
                active_device.battery_level = battery
                if battery is not None:
                    print(f"[BluetoothManager] Battery: {battery}%")

                # Start streaming
                print(f"[BluetoothManager] Starting stream for {assignment.participant_id}...")

                def data_handler(samples: List[float]):
                    """Handle incoming data samples"""
                    # DEBUG: Log data handler invocation
                    # Suppressed high-frequency debug output
                    # print(f"[BT Manager DEBUG] data_handler called: {len(samples)} samples for {assignment.participant_id}")

                    # Update sample buffer for live preview
                    active_device.sample_buffer.extend(samples)
                    active_device.last_sample_time = time.time()

                    # Send to LSL via callback
                    if self.data_callback:
                        stream_info = backend.get_stream_info()
                        # Suppressed high-frequency debug output
                        # print(f"[BT Manager DEBUG] Calling LSL callback: {assignment.participant_id}_{stream_info.stream_type}")
                        self.data_callback(
                            assignment.participant_id,
                            stream_info.stream_type,
                            samples
                        )
                    else:
                        print(f"[BT Manager DEBUG] WARNING: No data_callback registered! Samples not sent to LSL")

                success = await backend.start_streaming(data_handler)
                if not success:
                    raise RuntimeError("Failed to start streaming")

                # Reset reconnect counter on successful connection
                active_device.reconnect_attempts = 0

                # Keep connection alive until stopped or disconnected
                while self._running and backend.is_streaming():
                    # Check if backend uses pull-based data (ZMQ, mock devices)
                    if hasattr(backend, 'receive_samples'):
                        # Poll for data with short timeout
                        # Note: receive_samples() internally calls data_callback
                        # if samples are received (see zmq_bluetooth_source.py line 470)
                        backend.receive_samples(timeout_ms=100)
                    else:
                        # Push-based backend (Polar H10) - just wait
                        await asyncio.sleep(0.1)

                    # Periodically update battery
                    if time.time() % 30 < 1:  # Every ~30 seconds
                        battery = await backend.get_battery_level()
                        active_device.battery_level = battery

            except Exception as e:
                print(f"[BluetoothManager] Device error: {e}")
                active_device.reconnect_attempts += 1

                # Exponential backoff for reconnection
                backoff = min(2 ** active_device.reconnect_attempts, 16)
                print(f"[BluetoothManager] Reconnecting in {backoff}s... (attempt {active_device.reconnect_attempts}/{self.max_reconnect_attempts})")
                await asyncio.sleep(backoff)

            finally:
                # Ensure clean disconnect
                try:
                    await backend.disconnect()
                except:
                    pass

        if active_device.reconnect_attempts >= self.max_reconnect_attempts:
            print(f"[BluetoothManager] Max reconnect attempts reached for {assignment.mac_address}")

    def get_device_status(self, participant_id: str, mac_address: str) -> Dict:
        """
        Get status of a specific device.

        Args:
            participant_id: Participant ID
            mac_address: Device MAC address

        Returns:
            Status dictionary
        """
        key = (participant_id, mac_address)
        active_device = self.active_devices.get(key)

        if not active_device:
            return {
                'connected': False,
                'streaming': False,
                'battery': None,
                'reconnect_attempts': 0
            }

        return {
            'connected': active_device.backend.is_connected(),
            'streaming': active_device.backend.is_streaming(),
            'battery': active_device.battery_level,
            'reconnect_attempts': active_device.reconnect_attempts,
            'state': active_device.backend.state.value
        }

    def get_sample_buffer(self, participant_id: str, mac_address: str) -> List[float]:
        """
        Get recent samples for live preview.

        Args:
            participant_id: Participant ID
            mac_address: Device MAC address

        Returns:
            List of recent samples (up to 100)
        """
        key = (participant_id, mac_address)
        active_device = self.active_devices.get(key)

        if active_device:
            return list(active_device.sample_buffer)
        return []
