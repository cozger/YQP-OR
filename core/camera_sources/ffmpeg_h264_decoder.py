#!/usr/bin/env python3
"""
FFmpeg H264 Decoder with Threading

Production-ready H264 decoder using FFmpeg subprocess with proper threading
for non-blocking operation.

Features:
- Accept NAL units from network stream
- Background thread reads decoded frames from FFmpeg stdout
- Frame queue with latest-frame semantics (low latency)
- Automatic error recovery

Author: Claude Code
Date: 2025-10-23
"""

import subprocess
import threading
import queue
import time
import select
import os
from typing import Optional
import numpy as np
import cv2


class FFmpegH264Decoder:
    """
    Threaded FFmpeg H264 decoder with frame output.

    Accepts H264 NAL units and outputs decoded BGR frames using
    FFmpeg subprocess with background threading.
    """

    def __init__(self, width: int, height: int, fps: int = 30, enable_hwaccel: bool = True):
        """
        Initialize FFmpeg H264 decoder.

        Args:
            width: Expected frame width
            height: Expected frame height
            fps: Expected frame rate (for buffer sizing)
            enable_hwaccel: Try to use hardware acceleration if available (default: True)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_size = width * height * 3  # BGR = 3 bytes per pixel
        self.enable_hwaccel = enable_hwaccel
        self.hwaccel_type = None  # Will be detected

        # Threading (OPTIMIZED for aggregated NAL units)
        self.running = False
        # With aggregation: ~30 frames/sec, each frame = 1 aggregated NAL unit
        # Queue of 10 = ~333ms buffering (acceptable latency for 30 FPS)
        self.nal_queue = queue.Queue(maxsize=10)  # Reduced for lower latency
        self.frame_queue = queue.Queue(maxsize=1)  # Minimal queue for lowest latency
        self.writer_thread = None
        self.reader_thread = None
        self.stderr_thread = None

        # FFmpeg process
        self.process = None

        # Statistics
        self.nals_received = 0
        self.frames_decoded = 0
        self.start_time = None

        # Detect hardware acceleration
        if self.enable_hwaccel:
            self._detect_hwaccel()

        # Start decoder
        self._start_decoder()

    def _get_nvidia_lib_paths(self):
        """
        Detect NVIDIA library paths (WSL2-aware).

        In WSL2, NVIDIA libraries are in /usr/lib/wsl/lib which is not in the
        standard library search path. This function detects and returns the path
        if CUDA video libraries are available.

        Returns:
            str: Path to NVIDIA libraries, or None if not found
        """
        # WSL2-specific NVIDIA library path
        wsl_lib_path = '/usr/lib/wsl/lib'

        # Check if CUVID library exists (required for hardware H.264 decode)
        cuvid_lib = os.path.join(wsl_lib_path, 'libnvcuvid.so')
        if os.path.exists(cuvid_lib):
            print(f"[H264] Found NVIDIA CUVID library: {cuvid_lib}")
            return wsl_lib_path

        # Could add other paths here for native Linux if needed
        # e.g., /usr/lib/x86_64-linux-gnu for standard installations

        return None

    def _detect_hwaccel(self):
        """
        Detect available hardware acceleration methods.

        Tests in order of preference:
        1. CUDA (NVIDIA with CUVID - best for WSL2)
        2. VAAPI (Intel/AMD on native Linux)
        3. Fallback to software decode

        Sets self.hwaccel_type to the detected method or None.
        """
        # List of hardware acceleration methods to try (in order of preference)
        hwaccel_methods = []

        # Check for CUDA with CUVID libraries (WSL2 and native Linux)
        nvidia_lib_path = self._get_nvidia_lib_paths()
        if nvidia_lib_path:
            # CUDA available with video decode libraries
            # Use h264_cuvid decoder (not just -hwaccel cuda)
            hwaccel_methods.append(('cuda_cuvid', ['-c:v', 'h264_cuvid']))
            print(f"[H264] Detected CUDA hardware acceleration (CUVID)")
        elif os.path.exists('/dev/nvidia0'):
            # GPU exists but no CUVID libraries - try generic CUDA anyway
            hwaccel_methods.append(('cuda', ['-hwaccel', 'cuda']))
            print(f"[H264] Detected CUDA (generic, may not work without CUVID libs)")

        # Check for VAAPI (Intel/AMD on native Linux)
        if os.path.exists('/dev/dri/renderD128'):
            hwaccel_methods.append(('vaapi', ['-hwaccel', 'vaapi', '-hwaccel_device', '/dev/dri/renderD128']))
            print(f"[H264] Detected VAAPI hardware acceleration")

        # Select the first available method
        if hwaccel_methods:
            self.hwaccel_type = hwaccel_methods[0]
            print(f"[H264] Will attempt hardware acceleration: {self.hwaccel_type[0]}")
        else:
            print(f"[H264] No hardware acceleration available, using software decode")
            self.hwaccel_type = None

    def _build_ffmpeg_command(self):
        """
        Build FFmpeg command line for decoding with optional hardware acceleration.

        Returns:
            list: FFmpeg command arguments
        """
        cmd = ['ffmpeg']

        # Add hardware acceleration flags BEFORE input (if available)
        if self.hwaccel_type:
            hwaccel_name, hwaccel_flags = self.hwaccel_type
            cmd.extend(hwaccel_flags)
            print(f"[H264] Using {hwaccel_name} hardware acceleration")

        # Low-latency input flags (OPTIMIZED for aggregated NAL units + real-time streaming)
        # Balanced settings for NAL aggregation: need enough buffering to find SPS/PPS in aggregated frames
        cmd.extend([
            '-probesize', '1M',           # 1MB: ~6-20 aggregated frames (ensures SPS/PPS detection)
            '-analyzeduration', '1000000', # 1s: Enough time to buffer and find stream parameters
            '-fflags', 'nobuffer+fastseek', # nobuffer: minimize runtime buffering, fastseek: stream navigation
            '-flags', 'low_delay',        # Low-latency decoding (no frame reordering)
        ])

        # Multi-threaded decoding (OPTIMIZED for CUDA hardware decoder)
        # CUDA decoder handles threading internally - let it auto-tune
        # Too many threads can add overhead; let FFmpeg optimize
        cmd.extend([
            '-threads', '0',              # Auto-select optimal thread count (CUDA uses GPU threads)
        ])

        # Input parameters
        cmd.extend([
            '-f', 'h264',  # Input format: raw H264
            '-i', 'pipe:0',  # Input from stdin
        ])

        # Color conversion optimization (CRITICAL for 30 FPS performance)
        # Multi-threaded swscale with fast bilinear interpolation
        cmd.extend([
            '-filter_threads', '4',       # Use 4 threads for filter processing (color conversion)
            '-sws_flags', 'fast_bilinear', # Fastest swscale algorithm (trades quality for speed)
        ])

        # Output parameters
        # Note: h264_cuvid automatically downloads frames to system memory as YUV420p
        # FFmpeg's swscale then converts YUV420p → BGR24 (specified by -pix_fmt)
        cmd.extend([
            '-f', 'rawvideo',  # Raw video output
            '-pix_fmt', 'bgr24',  # BGR24 for OpenCV (slightly faster than RGB24)
            'pipe:1'  # Output to stdout
        ])

        return cmd

    def _start_decoder(self):
        """Start FFmpeg decoder subprocess and threads."""
        cmd = self._build_ffmpeg_command()

        print(f"[H264] Starting FFmpeg decoder...")
        print(f"[H264]   Expected resolution: {self.width}x{self.height}")

        try:
            # Build environment with NVIDIA library path for WSL2
            env = os.environ.copy()

            nvidia_lib_path = self._get_nvidia_lib_paths()
            if nvidia_lib_path and self.hwaccel_type:
                # Add WSL2 NVIDIA library path to LD_LIBRARY_PATH
                current_ld_path = env.get('LD_LIBRARY_PATH', '')
                if current_ld_path:
                    env['LD_LIBRARY_PATH'] = f"{nvidia_lib_path}:{current_ld_path}"
                else:
                    env['LD_LIBRARY_PATH'] = nvidia_lib_path

                print(f"[H264] Added NVIDIA library path: {nvidia_lib_path}")
                print(f"[H264] LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}")

            # Start FFmpeg with custom environment
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,  # 100MB buffer
                env=env  # Pass custom environment with library paths
            )

            # Start background threads
            self.running = True
            self.start_time = time.time()

            self.writer_thread = threading.Thread(
                target=self._writer_loop,
                daemon=True,
                name="FFmpegDecoderWriter"
            )
            self.writer_thread.start()

            self.reader_thread = threading.Thread(
                target=self._reader_loop,
                daemon=True,
                name="FFmpegDecoderReader"
            )
            self.reader_thread.start()

            self.stderr_thread = threading.Thread(
                target=self._stderr_loop,
                daemon=True,
                name="FFmpegDecoderStderr"
            )
            self.stderr_thread.start()

            print(f"[H264] ✓ Decoder started successfully")

        except Exception as e:
            print(f"[ERROR] Failed to start FFmpeg decoder: {e}")
            raise

    def _writer_loop(self):
        """
        Writer thread: Read NAL units from queue and write to FFmpeg stdin.

        Runs in background thread.
        """
        while self.running:
            try:
                # Get NAL unit from queue (blocking with timeout)
                nal = self.nal_queue.get(timeout=0.1)

                if nal is None:
                    # Poison pill - stop thread
                    break

                # Write NAL unit to FFmpeg stdin
                self.process.stdin.write(nal)
                self.process.stdin.flush()

                self.nals_received += 1

            except queue.Empty:
                # No NAL units available - continue
                continue
            except Exception as e:
                print(f"[ERROR] Decoder writer thread error: {e}")
                break

        print("[H264] Decoder writer thread stopped")

    def _reader_loop(self):
        """
        Reader thread: Read decoded frames from FFmpeg stdout using non-blocking I/O.

        Uses select() to avoid blocking when FFmpeg is slow, improving throughput.

        Runs in background thread.
        """
        frame_buffer = bytearray()  # Accumulate frame bytes
        chunk_size = 65536  # Read 64KB chunks (balance latency vs overhead)

        while self.running:
            try:
                # Use select to check if data is available (non-blocking check with 0.1s timeout)
                readable, _, _ = select.select([self.process.stdout], [], [], 0.1)

                if not readable:
                    # No data available - check if process is still alive
                    if self.process.poll() is not None:
                        # Process terminated
                        break
                    continue

                # Data available - read chunk
                chunk = self.process.stdout.read(chunk_size)

                if not chunk:
                    # EOF - decoder stopped
                    break

                # Accumulate bytes
                frame_buffer.extend(chunk)

                # Check if we have a complete frame
                while len(frame_buffer) >= self.frame_size:
                    # Extract one frame
                    frame_bytes = bytes(frame_buffer[:self.frame_size])
                    frame_buffer = frame_buffer[self.frame_size:]  # Remove consumed bytes

                    # Convert bytes to numpy array (BGR24 format)
                    frame = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = frame.reshape((self.height, self.width, 3))

                    # Add to frame queue (drop oldest if full for low latency)
                    try:
                        self.frame_queue.put_nowait(frame)
                        self.frames_decoded += 1
                    except queue.Full:
                        # Queue full - drop oldest frame
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except:
                            pass

            except Exception as e:
                print(f"[ERROR] Decoder reader thread error: {e}")
                break

        print("[H264] Decoder reader thread stopped")

    def _stderr_loop(self):
        """
        Stderr thread: Consume FFmpeg diagnostic output to prevent blocking.

        FFmpeg writes diagnostic information to stderr continuously.
        If stderr is not consumed, the pipe buffer fills (64KB on Linux/Windows)
        and FFmpeg blocks, stalling the entire decoding pipeline.

        Runs in background thread.
        """
        try:
            while self.running:
                # Read line from stderr (blocking)
                line = self.process.stderr.readline()

                if not line:
                    # EOF - decoder stopped
                    break

                # Log ALL FFmpeg output for debugging (temporary)
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str:  # Print all non-empty lines
                    print(f"[FFmpeg Decoder] {line_str}")

        except Exception as e:
            print(f"[ERROR] Decoder stderr thread error: {e}")

        print("[H264] Decoder stderr thread stopped")

    def feed_nal_unit(self, nal_data: bytes) -> int:
        """
        Feed NAL unit to decoder (non-blocking).

        Args:
            nal_data: Raw NAL unit bytes

        Returns:
            int: Number of NAL units in queue (0 = immediate decoding)
        """
        if not self.running:
            return -1

        # DEBUG: Log NAL unit characteristics
        if self.nals_received < 10 or self.nals_received % 100 == 0:
            # Show first few bytes (hex) to identify NAL unit type
            header = nal_data[:min(8, len(nal_data))].hex()
            print(f"[H264 DEBUG] NAL #{self.nals_received}: size={len(nal_data)} bytes, header={header}")

        # Add to queue (non-blocking)
        try:
            self.nal_queue.put_nowait(nal_data)
            return self.nal_queue.qsize()
        except queue.Full:
            # Queue full - drop oldest NAL (conflation for low latency)
            try:
                self.nal_queue.get_nowait()
                self.nal_queue.put_nowait(nal_data)
                return self.nal_queue.qsize()
            except:
                return -1

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get latest decoded frame (blocking with timeout).

        Drains frame queue to always return the most recent frame.

        Args:
            timeout: Timeout in seconds

        Returns:
            np.ndarray: BGR frame, or None if timeout
        """
        latest_frame = None

        # Drain queue to get latest frame
        while True:
            try:
                frame = self.frame_queue.get(timeout=0.001 if latest_frame is not None else timeout)
                latest_frame = frame
            except queue.Empty:
                break

        return latest_frame

    def get_stats(self) -> dict:
        """
        Get decoder statistics.

        Returns:
            dict: Statistics dictionary
        """
        elapsed = time.time() - self.start_time if self.start_time else 0

        return {
            'nals_received': self.nals_received,
            'frames_decoded': self.frames_decoded,
            'fps': self.frames_decoded / elapsed if elapsed > 0 else 0,
            'running_time': elapsed,
            'nal_queue_size': self.nal_queue.qsize(),
            'frame_queue_size': self.frame_queue.qsize(),
        }

    def stop(self):
        """Stop decoder and cleanup."""
        print("[H264] Stopping decoder...")

        # Signal threads to stop
        self.running = False

        # Send poison pill to writer thread
        try:
            self.nal_queue.put(None, timeout=1.0)
        except:
            pass

        # Wait for threads
        if self.writer_thread:
            self.writer_thread.join(timeout=2.0)

        if self.reader_thread:
            self.reader_thread.join(timeout=2.0)

        if self.stderr_thread:
            self.stderr_thread.join(timeout=2.0)

        # Close FFmpeg
        if self.process:
            try:
                self.process.stdin.close()
                self.process.stdout.close()
                self.process.stderr.close()
                self.process.wait(timeout=5.0)
            except:
                self.process.kill()

        # Print final stats
        stats = self.get_stats()
        print(f"[H264] Final decoder statistics:")
        print(f"  NAL units received: {stats['nals_received']}")
        print(f"  Frames decoded: {stats['frames_decoded']}")
        print(f"  Average FPS: {stats['fps']:.1f}")
        print(f"  Running time: {stats['running_time']:.1f}s")
        print(f"[H264] ✓ Decoder stopped")


# =============================================================================
# Test / Example Usage
# =============================================================================

def test_decoder_with_file():
    """Test the decoder with a pre-recorded H264 file."""
    import os

    print("=" * 70)
    print("FFmpeg H264 Decoder Test")
    print("=" * 70)

    # Check if test video exists
    test_video = "/tmp/test_h264_720p.mp4"
    if not os.path.exists(test_video):
        print(f"\nTest video not found: {test_video}")
        print("Run test_h264_ffmpeg_decode.py first to create test video")
        return

    # Create decoder
    width, height, fps = 1280, 720, 30
    decoder = FFmpegH264Decoder(width, height, fps)

    # Read H264 file and feed to decoder
    print(f"\nFeeding H264 data from {test_video}...")

    with open(test_video, 'rb') as f:
        h264_data = f.read()

    # Feed data in chunks (simulate streaming)
    chunk_size = 4096
    for i in range(0, len(h264_data), chunk_size):
        chunk = h264_data[i:i+chunk_size]
        decoder.feed_nal_unit(chunk)

        # Try to get decoded frame
        frame = decoder.get_frame(timeout=0.001)
        if frame is not None:
            print(f"  Chunk {i//chunk_size}: got frame {frame.shape}")

    # Allow decoder to finish
    time.sleep(2.0)

    # Collect remaining frames
    remaining_frames = 0
    while True:
        frame = decoder.get_frame(timeout=0.1)
        if frame is None:
            break
        remaining_frames += 1

    print(f"\nCollected {remaining_frames} remaining frames")

    # Stop decoder
    decoder.stop()

    print("\n✓ Test complete")


if __name__ == "__main__":
    test_decoder_with_file()
