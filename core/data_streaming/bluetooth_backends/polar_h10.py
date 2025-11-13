"""
Polar H10 Heart Rate Monitor Backend

This module implements the BluetoothDeviceBackend interface for the Polar H10
heart rate monitor with ECG streaming capability.

Based on: https://github.com/markspan/PolarBand2lsl/
GATT UUIDs courtesy of N. Pareek
"""

import asyncio
from typing import List, Optional, Callable
import bleak
from bleak import BleakClient, BleakScanner

from .base import BluetoothDeviceBackend, DeviceInfo, StreamInfo, ConnectionState


# Polar H10 GATT UUIDs
PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_DATA = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"
BATTERY_LEVEL_UUID = "00002a19-0000-1000-8000-00805f9b34fb"  # Standard battery UUID

# ECG streaming configuration
ECG_WRITE = bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00])
ECG_SAMPLING_FREQ = 130  # Hz


class PolarH10Backend(BluetoothDeviceBackend):
    """
    Backend for Polar H10 heart rate monitor with ECG streaming.

    Provides:
    - ECG data at 130 Hz
    - Battery level reading
    - Automatic reconnection on disconnection
    """

    def __init__(self):
        super().__init__()
        self._battery_level: Optional[int] = None
        self._stop_event = asyncio.Event()

    @classmethod
    def get_backend_identifier(cls) -> str:
        return "polar_h10"

    @classmethod
    def matches_device(cls, device_name: str) -> bool:
        """Check if device is a Polar H10"""
        return "Polar H10" in device_name or "Polar H9" in device_name

    async def discover(self, timeout: float = 10.0) -> List[DeviceInfo]:
        """
        Scan for Polar H10 devices.

        Args:
            timeout: Scan timeout in seconds

        Returns:
            List of discovered Polar H10 devices
        """
        devices = []

        try:
            # Scan for BLE devices
            discovered = await BleakScanner.discover(
                timeout=timeout,
                return_adv=True,
                cb=dict(use_bdaddr=False),
                scanning_mode='active'
            )

            # Filter for Polar devices
            for device, adv_data in discovered.values():
                if self.matches_device(str(device.name)):
                    devices.append(DeviceInfo(
                        mac_address=device.address,
                        name=device.name,
                        rssi=adv_data.rssi if hasattr(adv_data, 'rssi') else None,
                        device_type=self.get_backend_identifier()
                    ))

        except Exception as e:
            print(f"[PolarH10Backend] Discovery error: {e}")

        return devices

    async def connect(self, mac_address: str) -> bool:
        """
        Connect to a Polar H10 device.

        Args:
            mac_address: Bluetooth MAC address

        Returns:
            True if connected successfully
        """
        try:
            self.state = ConnectionState.CONNECTING
            self.mac_address = mac_address

            # Create BleakClient
            self.client = BleakClient(mac_address)
            await self.client.connect()

            if self.client.is_connected:
                self.state = ConnectionState.CONNECTED
                print(f"[PolarH10Backend] Connected to {mac_address}")

                # Read battery level
                try:
                    battery_bytes = await self.client.read_gatt_char(BATTERY_LEVEL_UUID)
                    self._battery_level = int.from_bytes(battery_bytes, byteorder='little')
                    print(f"[PolarH10Backend] Battery level: {self._battery_level}%")
                except Exception as e:
                    print(f"[PolarH10Backend] Could not read battery: {e}")

                return True
            else:
                self.state = ConnectionState.DISCONNECTED
                return False

        except Exception as e:
            print(f"[PolarH10Backend] Connection error: {e}")
            self.state = ConnectionState.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect from the device"""
        try:
            if self.client and self.client.is_connected:
                # Stop streaming if active
                if self.is_streaming():
                    await self.stop_streaming()

                await self.client.disconnect()
                print(f"[PolarH10Backend] Disconnected from {self.mac_address}")

        except Exception as e:
            print(f"[PolarH10Backend] Disconnect error: {e}")

        finally:
            self.state = ConnectionState.DISCONNECTED
            self.client = None

    async def start_streaming(self, data_callback: Callable[[List[float]], None]) -> bool:
        """
        Start ECG streaming from Polar H10.

        Args:
            data_callback: Function called with each ECG sample

        Returns:
            True if streaming started successfully
        """
        if not self.is_connected():
            print("[PolarH10Backend] Not connected, cannot start streaming")
            return False

        try:
            self._data_callback = data_callback
            self._stop_event.clear()

            # Read PMD control characteristic
            await self.client.read_gatt_char(PMD_CONTROL)

            # Write ECG configuration
            await self.client.write_gatt_char(PMD_CONTROL, ECG_WRITE)

            # Start notifications
            await self.client.start_notify(PMD_DATA, self._notification_handler)

            self.state = ConnectionState.STREAMING
            print("[PolarH10Backend] ECG streaming started")
            return True

        except Exception as e:
            print(f"[PolarH10Backend] Failed to start streaming: {e}")
            self.state = ConnectionState.ERROR
            return False

    async def stop_streaming(self) -> None:
        """Stop ECG streaming"""
        try:
            if self.client and self.client.is_connected:
                await self.client.stop_notify(PMD_DATA)
                self._stop_event.set()

            self.state = ConnectionState.CONNECTED
            print("[PolarH10Backend] ECG streaming stopped")

        except Exception as e:
            print(f"[PolarH10Backend] Error stopping streaming: {e}")

    def _notification_handler(self, sender, data: bytearray):
        """
        Handle incoming ECG data from Polar H10.

        Data format:
        - Byte 0: Data type (0x00 for ECG)
        - Bytes 1-9: Timestamp and metadata
        - Bytes 10+: ECG samples (3 bytes per sample, signed int)
        """
        if data[0] == 0x00 and self._data_callback:
            # Parse ECG samples
            samples = self.parse_data(data)

            # Send each sample to callback
            for sample in samples:
                self._data_callback([sample])

    def parse_data(self, data: bytearray) -> List[float]:
        """
        Parse Polar H10 ECG data.

        Args:
            data: Raw bytearray from device

        Returns:
            List of ECG samples (microvolts)
        """
        samples = []

        if data[0] == 0x00:  # ECG data packet
            step = 3  # 3 bytes per sample
            raw_samples = data[10:]  # Skip header
            offset = 0

            while offset < len(raw_samples):
                # Convert 3-byte signed integer to float
                ecg = int.from_bytes(
                    bytearray(raw_samples[offset: offset + step]),
                    byteorder="little",
                    signed=True
                )
                samples.append(float(ecg))
                offset += step

        return samples

    def get_stream_info(self) -> StreamInfo:
        """
        Get LSL stream metadata for Polar H10 ECG.

        Returns:
            StreamInfo configured for ECG data
        """
        return StreamInfo(
            stream_type="ecg",
            channel_count=1,
            nominal_srate=float(ECG_SAMPLING_FREQ),
            channel_format="float32",
            channel_names=["ECG"],
            channel_units=["microvolts"],
            manufacturer="Polar",
            model="H10"
        )

    async def get_battery_level(self) -> Optional[int]:
        """
        Get battery level from Polar H10.

        Returns:
            Battery percentage (0-100) or None if unavailable
        """
        if not self.is_connected():
            return self._battery_level  # Return cached value

        try:
            battery_bytes = await self.client.read_gatt_char(BATTERY_LEVEL_UUID)
            self._battery_level = int.from_bytes(battery_bytes, byteorder='little')
            return self._battery_level

        except Exception as e:
            print(f"[PolarH10Backend] Battery read error: {e}")
            return self._battery_level  # Return cached value
