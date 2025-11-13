"""
Bluetooth Device Backend Registry

This module provides a registry system for Bluetooth device backends.
New device types can be added by implementing the BluetoothDeviceBackend
interface and registering them here.
"""

from typing import Dict, Type, List, Optional
from .base import BluetoothDeviceBackend, DeviceInfo
from .polar_h10 import PolarH10Backend
from .mock_device import MockBluetoothBackend
from .zmq_bluetooth_source import ZMQBluetoothSource


class BackendRegistry:
    """
    Registry for Bluetooth device backends.

    Provides:
    - Backend registration and lookup
    - Automatic device-to-backend matching
    - Backend instantiation
    """

    def __init__(self):
        self._backends: Dict[str, Type[BluetoothDeviceBackend]] = {}
        self._register_default_backends()

    def _register_default_backends(self):
        """Register built-in backends"""
        self.register(PolarH10Backend)
        self.register(MockBluetoothBackend)
        self.register(ZMQBluetoothSource)

    def register(self, backend_class: Type[BluetoothDeviceBackend]):
        """
        Register a new backend.

        Args:
            backend_class: Backend class to register
        """
        identifier = backend_class.get_backend_identifier()
        self._backends[identifier] = backend_class
        print(f"[BackendRegistry] Registered backend: {identifier}")

    def get_backend(self, identifier: str) -> Optional[Type[BluetoothDeviceBackend]]:
        """
        Get backend class by identifier.

        Args:
            identifier: Backend identifier (e.g., "polar_h10")

        Returns:
            Backend class or None if not found
        """
        return self._backends.get(identifier)

    def create_backend(self, identifier: str) -> Optional[BluetoothDeviceBackend]:
        """
        Create a new backend instance.

        Args:
            identifier: Backend identifier

        Returns:
            Backend instance or None if identifier not found
        """
        backend_class = self.get_backend(identifier)
        if backend_class:
            return backend_class()
        return None

    def get_all_identifiers(self) -> List[str]:
        """
        Get all registered backend identifiers.

        Returns:
            List of backend identifiers
        """
        return list(self._backends.keys())

    def match_device(self, device_name: str) -> Optional[str]:
        """
        Find backend that matches a device name.

        Args:
            device_name: Bluetooth device name from discovery

        Returns:
            Backend identifier or None if no match
        """
        for identifier, backend_class in self._backends.items():
            if backend_class.matches_device(device_name):
                return identifier
        return None

    def auto_assign_backend(self, device_info: DeviceInfo) -> Optional[str]:
        """
        Automatically assign backend type to a discovered device.

        Args:
            device_info: Device information from discovery

        Returns:
            Backend identifier or None if no match
        """
        # If device already has a type, validate it
        if device_info.device_type:
            if device_info.device_type in self._backends:
                return device_info.device_type

        # Otherwise, try to match by name
        return self.match_device(device_info.name)


# Global registry instance
_registry = BackendRegistry()


def get_registry() -> BackendRegistry:
    """
    Get the global backend registry.

    Returns:
        Global BackendRegistry instance
    """
    return _registry


# Import all classes to make them available
from .base import BluetoothDeviceBackend, DeviceInfo, StreamInfo, ConnectionState

# Convenience exports
__all__ = [
    'BluetoothDeviceBackend',
    'DeviceInfo',
    'StreamInfo',
    'ConnectionState',
    'PolarH10Backend',
    'MockBluetoothBackend',
    'ZMQBluetoothSource',
    'BackendRegistry',
    'get_registry'
]
