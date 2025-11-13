"""
Abstract base class for Bluetooth device backends.

This module defines the interface that all Bluetooth device backends must implement.
Each device type (Polar H10, Shimmer, etc.) should have its own backend class that
inherits from BluetoothDeviceBackend.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any
from enum import Enum
import asyncio


class ConnectionState(Enum):
    """Connection state for Bluetooth devices"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


@dataclass
class DeviceInfo:
    """Information about a discovered Bluetooth device"""
    mac_address: str
    name: str
    rssi: Optional[int] = None  # Signal strength
    battery_level: Optional[int] = None  # 0-100 percentage
    device_type: Optional[str] = None  # Backend identifier (e.g., "polar_h10")
    metadata: Optional[Dict[str, Any]] = None  # Backend-specific metadata (e.g., ZMQ port/topic)


@dataclass
class StreamInfo:
    """LSL stream metadata for a Bluetooth device"""
    stream_type: str  # e.g., "heartrate", "ecg", "accel", "gsr"
    channel_count: int
    nominal_srate: float  # Sampling rate in Hz
    channel_format: str  # "float32", "int32", etc.
    channel_names: List[str]  # Names of each channel
    channel_units: List[str]  # Units for each channel (e.g., "microvolts", "bpm")
    manufacturer: str
    model: str


class BluetoothDeviceBackend(ABC):
    """
    Abstract base class for Bluetooth device backends.

    Each device type should implement this interface to provide:
    - Device discovery
    - Connection management
    - Data streaming
    - LSL stream metadata
    """

    def __init__(self):
        self.state = ConnectionState.DISCONNECTED
        self.mac_address: Optional[str] = None
        self.client = None  # BleakClient instance
        self._data_callback: Optional[Callable] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5

    @abstractmethod
    async def discover(self, timeout: float = 10.0) -> List[DeviceInfo]:
        """
        Scan for available devices of this type.

        Args:
            timeout: Scan timeout in seconds

        Returns:
            List of discovered devices matching this backend type
        """
        pass

    @abstractmethod
    async def connect(self, mac_address: str) -> bool:
        """
        Connect to a specific device by MAC address.

        Args:
            mac_address: Bluetooth MAC address (e.g., "AA:BB:CC:DD:EE:FF")

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the device cleanly.
        """
        pass

    @abstractmethod
    async def start_streaming(self, data_callback: Callable[[List[float]], None]) -> bool:
        """
        Start streaming data from the device.

        Args:
            data_callback: Function to call with parsed data samples
                          Signature: callback(samples: List[float])

        Returns:
            True if streaming started successfully, False otherwise
        """
        pass

    @abstractmethod
    async def stop_streaming(self) -> None:
        """
        Stop streaming data from the device.
        """
        pass

    @abstractmethod
    def get_stream_info(self) -> StreamInfo:
        """
        Get LSL stream metadata for this device.

        Returns:
            StreamInfo with channel count, sampling rate, etc.
        """
        pass

    @abstractmethod
    def parse_data(self, data: bytearray) -> List[float]:
        """
        Parse raw Bluetooth data into numeric samples.

        Args:
            data: Raw bytearray received from device

        Returns:
            List of float values representing samples
        """
        pass

    async def get_battery_level(self) -> Optional[int]:
        """
        Get battery level if device supports it.

        Returns:
            Battery percentage (0-100) or None if not supported
        """
        return None  # Default: not supported

    def is_connected(self) -> bool:
        """Check if device is connected"""
        return self.state in [ConnectionState.CONNECTED, ConnectionState.STREAMING]

    def is_streaming(self) -> bool:
        """Check if device is actively streaming"""
        return self.state == ConnectionState.STREAMING

    @classmethod
    @abstractmethod
    def get_backend_identifier(cls) -> str:
        """
        Get unique identifier for this backend.

        Returns:
            Identifier string (e.g., "polar_h10", "shimmer_gsr")
        """
        pass

    @classmethod
    @abstractmethod
    def matches_device(cls, device_name: str) -> bool:
        """
        Check if a device name matches this backend.

        Args:
            device_name: Bluetooth device name from discovery

        Returns:
            True if this backend can handle the device
        """
        pass
