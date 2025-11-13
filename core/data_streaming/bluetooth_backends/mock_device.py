"""
Mock Bluetooth Device Backend

This module provides a simulated Bluetooth device for testing the Bluetooth
integration without physical hardware. Generates synthetic heart rate data
with realistic variability.
"""

import asyncio
import time
import math
import random
from typing import List, Optional, Callable

from .base import BluetoothDeviceBackend, DeviceInfo, StreamInfo, ConnectionState


class MockBluetoothBackend(BluetoothDeviceBackend):
    """
    Mock Bluetooth device for testing.

    Simulates a heart rate monitor with:
    - Realistic ECG waveform (sine wave + noise)
    - Variable battery level
    - Configurable sampling rate
    - Simulated connection delays
    """

    def __init__(self, device_id: int = 0, base_hr: int = 70):
        """
        Initialize mock device.

        Args:
            device_id: Unique ID for this mock device (0, 1, 2...)
            base_hr: Base heart rate in BPM (default: 70)
        """
        super().__init__()
        self.device_id = device_id
        self.base_hr = base_hr
        self._battery_level = 85  # Start at 85%
        self._streaming_task: Optional[asyncio.Task] = None
        self._sample_count = 0
        self._sampling_rate = 130  # Hz

    @classmethod
    def get_backend_identifier(cls) -> str:
        return "mock_device"

    @classmethod
    def matches_device(cls, device_name: str) -> bool:
        """Check if device is a mock device"""
        return "Mock" in device_name

    async def discover(self, timeout: float = 10.0) -> List[DeviceInfo]:
        """
        Simulate device discovery.

        Returns:
            List of 3 mock devices
        """
        await asyncio.sleep(1.0)  # Simulate scan delay

        devices = []
        for i in range(3):
            devices.append(DeviceInfo(
                mac_address=f"00:11:22:33:44:{i:02X}",
                name=f"Mock Device {i+1}",
                rssi=-50 - (i * 10),  # Simulated signal strength
                battery_level=85 - (i * 5),
                device_type=self.get_backend_identifier()
            ))

        return devices

    async def connect(self, mac_address: str) -> bool:
        """
        Simulate connection to mock device.

        Args:
            mac_address: MAC address from discovery

        Returns:
            True (always succeeds for mock)
        """
        try:
            self.state = ConnectionState.CONNECTING
            self.mac_address = mac_address

            # Simulate connection delay
            await asyncio.sleep(0.5)

            # Extract device ID from MAC address
            device_id = int(mac_address.split(':')[-1], 16)
            self.device_id = device_id
            self._battery_level = 85 - (device_id * 5)

            self.state = ConnectionState.CONNECTED
            print(f"[MockBluetoothBackend] Connected to {mac_address}")
            return True

        except Exception as e:
            print(f"[MockBluetoothBackend] Connection error: {e}")
            self.state = ConnectionState.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect from mock device"""
        if self.is_streaming():
            await self.stop_streaming()

        self.state = ConnectionState.DISCONNECTED
        print(f"[MockBluetoothBackend] Disconnected from {self.mac_address}")

    async def start_streaming(self, data_callback: Callable[[List[float]], None]) -> bool:
        """
        Start simulated data streaming.

        Args:
            data_callback: Function to call with generated samples

        Returns:
            True if streaming started
        """
        if not self.is_connected():
            print("[MockBluetoothBackend] Not connected, cannot start streaming")
            return False

        try:
            self._data_callback = data_callback
            self._sample_count = 0

            # Start background streaming task
            self._streaming_task = asyncio.create_task(self._stream_data())

            self.state = ConnectionState.STREAMING
            print("[MockBluetoothBackend] Streaming started")
            return True

        except Exception as e:
            print(f"[MockBluetoothBackend] Failed to start streaming: {e}")
            self.state = ConnectionState.ERROR
            return False

    async def stop_streaming(self) -> None:
        """Stop simulated data streaming"""
        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass

        self.state = ConnectionState.CONNECTED
        print("[MockBluetoothBackend] Streaming stopped")

    async def _stream_data(self):
        """
        Background task that generates and sends synthetic ECG data.
        """
        interval = 1.0 / self._sampling_rate  # Time between samples

        while True:
            try:
                # Generate synthetic ECG sample
                sample = self._generate_ecg_sample()

                # Send to callback
                if self._data_callback:
                    self._data_callback([sample])

                self._sample_count += 1

                # Simulate battery drain (1% per 10000 samples)
                if self._sample_count % 10000 == 0:
                    self._battery_level = max(0, self._battery_level - 1)

                # Wait for next sample
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[MockBluetoothBackend] Streaming error: {e}")
                break

    def _generate_ecg_sample(self) -> float:
        """
        Generate a synthetic ECG sample.

        Returns:
            Float representing ECG value in microvolts
        """
        # Time in seconds
        t = self._sample_count / self._sampling_rate

        # Convert base heart rate to frequency (Hz)
        hr_freq = self.base_hr / 60.0

        # Generate synthetic ECG waveform
        # Combination of sine waves at fundamental and harmonics
        ecg = 0.0
        ecg += 500 * math.sin(2 * math.pi * hr_freq * t)  # Fundamental (R-wave)
        ecg += 100 * math.sin(4 * math.pi * hr_freq * t)  # 2nd harmonic
        ecg += 50 * math.sin(6 * math.pi * hr_freq * t)   # 3rd harmonic

        # Add realistic noise
        noise = random.gauss(0, 20)  # Gaussian noise
        ecg += noise

        # Add baseline wander (slow drift)
        baseline = 50 * math.sin(2 * math.pi * 0.1 * t)
        ecg += baseline

        # Add device-specific offset (so different mock devices have different baselines)
        ecg += self.device_id * 50

        return ecg

    def parse_data(self, data: bytearray) -> List[float]:
        """
        Mock implementation (not used since we generate data directly).

        Args:
            data: Raw bytearray

        Returns:
            Empty list (not applicable for mock)
        """
        return []

    def get_stream_info(self) -> StreamInfo:
        """
        Get LSL stream metadata for mock device.

        Returns:
            StreamInfo configured for mock ECG data
        """
        return StreamInfo(
            stream_type="ecg",
            channel_count=1,
            nominal_srate=float(self._sampling_rate),
            channel_format="float32",
            channel_names=["ECG"],
            channel_units=["microvolts"],
            manufacturer="Mock",
            model=f"MockDevice{self.device_id}"
        )

    async def get_battery_level(self) -> Optional[int]:
        """
        Get simulated battery level.

        Returns:
            Battery percentage (0-100)
        """
        return self._battery_level
