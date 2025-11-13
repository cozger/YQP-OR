import cv2
import time
import threading
import queue
import numpy as np
from datetime import datetime
from pathlib import Path
from pylsl import StreamInfo, StreamOutlet
from multiprocessing import Process, Queue as MPQueue, Value, Array
import ctypes
from tkinter import ttk


class VideoRecorder:
    """
    Records video from shared frames and outputs frame numbers via LSL.
    Designed to work alongside the existing YouQuantiPy system.
    """
    def __init__(self, output_dir="recordings", codec=None, fps=None, config=None):
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get settings from config with fallbacks
        video_config = self.config.get('video_recording', {})
        self.codec = codec if codec is not None else video_config.get('codec', 'MJPG')
        self.target_fps = fps if fps is not None else video_config.get('fps', 30)
        
        self.recording = False
        self.writer = None
        # Get queue size from config
        queue_size = video_config.get('queue_size', 30)
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.record_thread = None
        self.frame_number = 0
        self.start_time = None
        
        # LSL outlet for frame numbers
        self.lsl_outlet = None
        
    def setup_lsl_stream(self, participant_id=""):
        """Setup LSL stream for frame numbers with proper metadata"""
        stream_name = f"{participant_id}_video_frames" if participant_id else "video_frames"
        info = StreamInfo(
            name=stream_name,
            type="VideoSync",
            channel_count=3,  # frame_number, lsl_timestamp, video_timestamp
            nominal_srate=0,  # Irregular rate
            channel_format="double64",
            source_id=f"{stream_name}_{participant_id}_uid"
        )
        
        # Add metadata
        desc = info.desc()
        desc.append_child_value("participant", participant_id)
        desc.append_child_value("fps", str(self.target_fps))
        
        # Channel descriptions
        channels = desc.append_child("channels")
        for label, unit in [("frame_number", "count"), ("lsl_timestamp", "seconds"), ("video_timestamp", "seconds")]:
            chan = channels.append_child("channel")
            chan.append_child_value("label", label)
            chan.append_child_value("unit", unit)
        
        # Get LSL settings from config
        max_buffered = self.config.get('video_recording', {}).get('max_buffered_lsl', 360)
        self.lsl_outlet = StreamOutlet(info, chunk_size=1, max_buffered=max_buffered)
        print(f"[Recorder] LSL stream created: {stream_name}")
        
    def start_recording(self, width, height, participant_id=""):
        """Start recording video"""
        if self.recording:
            return
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{participant_id}_{timestamp}.avi" if participant_id else f"recording_{timestamp}.avi"
        filepath = self.output_dir / filename
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(filepath), fourcc, self.target_fps, (width, height)
        )
        
        if not self.writer.isOpened():
            print(f"[Recorder] Failed to open video writer for {filepath}")
            return False
            
        # Setup LSL if not already done
        if self.lsl_outlet is None:
            self.setup_lsl_stream(participant_id)
            
        self.recording = True
        self.frame_number = 0
        self.start_time = time.time()
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()
        
        print(f"[Recorder] Started recording to {filepath}")
        return True
        
    def stop_recording(self):
        """Stop recording and close video file"""
        if not self.recording:
            return
            
        self.recording = False
        
        # Wait for thread to finish
        if self.record_thread:
            timeout = self.config.get('video_recording', {}).get('timeout_seconds', 2.0)
            self.record_thread.join(timeout=timeout)
            
        # Close video writer
        if self.writer:
            self.writer.release()
            self.writer = None
            
        print(f"[Recorder] Stopped recording. Total frames: {self.frame_number}")
        
    def add_frame(self, frame):
        """Add a frame to the recording queue"""
        if not self.recording:
            return False
            
        try:
            self.frame_queue.put_nowait(frame.copy())
            return True
        except queue.Full:
            # Drop frame if queue is full
            return False
            
    def _record_loop(self):
        """Recording thread loop"""
        last_lsl_push = 0
        # Get push interval from config
        push_interval_divisor = self.config.get('video_recording', {}).get('push_interval_divisor', 1.0)
        push_interval = push_interval_divisor / self.target_fps
        
        while self.recording:
            try:
                timeout = self.config.get('video_recording', {}).get('timeout_seconds', 0.1)
                frame = self.frame_queue.get(timeout=timeout)
                
                # Write frame
                if self.writer and self.writer.isOpened():
                    self.writer.write(frame)
                    self.frame_number += 1
                    
                    # Send frame number via LSL with throttling
                    current_time = time.time()
                    if self.lsl_outlet and (current_time - last_lsl_push >= push_interval):
                        video_time = (self.frame_number - 1) / self.target_fps
                        sample = [float(self.frame_number), current_time, video_time]
                        self.lsl_outlet.push_sample(sample, current_time)
                        last_lsl_push = current_time
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Recorder] Error in record loop: {e}")
                
    def close(self):
        """Clean up resources"""
        self.stop_recording()
        if self.lsl_outlet:
            try:
                self.lsl_outlet.__del__()
            except:
                pass
            self.lsl_outlet = None

class VideoRecorderProcess:
    """
    Multiprocess version that receives frames from shared memory.
    More efficient for high-performance recording.
    """
    def __init__(self, output_dir="recordings", codec=None, fps=None, camera_index=None, config=None):
        self.config = config or {}
        self.output_dir = output_dir
        
        # Get settings from config with fallbacks
        video_config = self.config.get('video_recording', {})
        self.codec = codec if codec is not None else video_config.get('codec', 'MJPG')
        self.fps = fps if fps is not None else video_config.get('fps', 30)
        
        self.camera_index = camera_index  # Store camera index
        self.process = None
        # Get queue size from config
        queue_size = video_config.get('queue_size', 30)
        self.frame_queue = MPQueue(maxsize=queue_size)
        self.stop_flag = Value(ctypes.c_bool, False)
        self.frame_count = Value(ctypes.c_int, 0)
        
    def start_recording(self, participant_id="", filename=None, annotate_frames=False):
        """Start recording process - dimensions auto-detected from first frame"""
        if self.process and self.process.is_alive():
            return False

        self.stop_flag.value = False
        self.frame_count.value = 0

        self.process = Process(
            target=_recorder_worker,
            args=(self.frame_queue, self.stop_flag, self.frame_count,
                  self.output_dir, self.codec, self.fps, participant_id, filename,
                  self.camera_index, self.config, annotate_frames),
            daemon=True
        )
        self.process.start()
        return True
        
    def add_frame(self, frame, frame_id=None):
        """
        Add frame to recording queue with optional global frame ID.

        Args:
            frame: Video frame (numpy array)
            frame_id: Optional global frame ID for synchronized recording across cameras.
                     If None, internal counter will be used.

        Returns:
            bool: True if frame was queued successfully, False otherwise
        """
        if not self.process or not self.process.is_alive():
            return False

        try:
            # Package frame with metadata
            frame_data = {
                'frame': frame,
                'frame_id': frame_id
            }
            self.frame_queue.put_nowait(frame_data)
            return True
        except:
            return False
                    
    def stop_recording(self):
        """Stop recording process"""
        if self.process and self.process.is_alive():
            print(f"[Recorder] Stopping recording, processing remaining frames...")
            self.stop_flag.value = True
            
            # DON'T clear the queue - let worker process remaining frames
            # Just wait for the process to finish
            self.process.join(timeout=5.0)
            
            if self.process.is_alive():
                print("[Recorder] Process didn't stop cleanly, terminating...")
                self.process.terminate()
                self.process.join(timeout=1.0)
                
        print(f"[Recorder] Total frames recorded: {self.frame_count.value}")

    def get_frame_count(self):
        """Get current frame count"""
        return self.frame_count.value


def _recorder_worker(frame_queue, stop_flag, frame_count,
                    output_dir, codec, fps, participant_id, filename=None, camera_index=None, config=None, annotate_frames=False):
    """Worker process for video recording - auto-detects dimensions and optionally annotates frames"""
    from pylsl import StreamInfo, StreamOutlet, local_clock
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{participant_id}_{timestamp}.avi" if participant_id else f"recording_{timestamp}.avi"
    
    filepath = Path(output_dir) / filename
    filepath.parent.mkdir(exist_ok=True)

    # Setup LSL with camera index in the name
    stream_name = f"camera_{camera_index}_video_frames" if camera_index is not None else "video_frames"
    
    info = StreamInfo(
        name=stream_name,
        type="VideoSync",
        channel_count=4,  # frame_number, unix_timestamp, lsl_timestamp, video_timestamp
        nominal_srate=0,  # Irregular rate
        channel_format="double64",
        source_id=f"{stream_name}_cam{camera_index}_uid"
    )
    
    # Add metadata
    desc = info.desc()
    desc.append_child_value("camera_index", str(camera_index) if camera_index is not None else "unknown")
    desc.append_child_value("participant", str(participant_id))
    desc.append_child_value("video_file", str(filename))
    desc.append_child_value("fps", str(fps))
    
    # Channel descriptions
    channels = desc.append_child("channels")
    for label, unit in [
        ("frame_number", "count"),
        ("unix_timestamp", "seconds"), 
        ("lsl_timestamp", "seconds"),
        ("video_timestamp", "seconds")
    ]:
        chan = channels.append_child("channel")
        chan.append_child_value("label", label)
        chan.append_child_value("unit", unit)
    
    # Get LSL settings from config
    max_buffered = (config or {}).get('video_recording', {}).get('max_buffered_lsl', 360)
    outlet = StreamOutlet(info, chunk_size=1, max_buffered=max_buffered)
    print(f"[Recorder] Created LSL stream '{stream_name}' for camera {camera_index}")

    writer = None
    frames_written = 0
    lsl_samples_pushed = 0
    expected_dimensions = None
    start_time = None
    last_push_time = 0
    # Get push interval from config
    push_interval_divisor = (config or {}).get('video_recording', {}).get('push_interval_divisor', 1.0)
    push_interval = push_interval_divisor / fps
    
    print(f"[Recorder] Waiting for first frame...")
    
    # Get overlay settings from config
    overlay_config = (config or {}).get('video_recording', {}).get('overlay_settings', {})
    overlay_enabled = overlay_config.get('enabled', True) and annotate_frames
    show_frame_id = overlay_config.get('show_frame_id', True)
    show_unix_time = overlay_config.get('show_unix_time', True)
    show_human_time = overlay_config.get('show_human_time', True)
    font_scale = overlay_config.get('font_scale', 0.8)
    overlay_color = tuple(overlay_config.get('color', [0, 255, 0]))
    overlay_thickness = overlay_config.get('thickness', 2)
    line_spacing = overlay_config.get('line_spacing', 30)

    # Recording loop
    while not stop_flag.value or not frame_queue.empty():
        try:
            timeout = (config or {}).get('video_recording', {}).get('timeout_seconds', 0.1)
            frame_data = frame_queue.get(timeout=timeout)

            # Support both old (raw frame) and new (dict) formats for backward compatibility
            if isinstance(frame_data, dict):
                frame = frame_data['frame']
                provided_frame_id = frame_data.get('frame_id')
            else:
                frame = frame_data
                provided_frame_id = None
            
            # Initialize on first frame
            if writer is None:
                height, width = frame.shape[:2]
                channels = frame.shape[2] if len(frame.shape) > 2 else 1
                
                print(f"[Recorder] Auto-detected dimensions: {width}x{height} with {channels} channels")
                expected_dimensions = (height, width, channels) if channels == 3 else (height, width)
                
                # Create writer
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
                
                if not writer.isOpened():
                    # Try fallback
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
                
                print(f"[Recorder] Video writer initialized")
                start_time = time.time()
            
            # Ensure proper dimensions
            current_shape = frame.shape[:2] if len(frame.shape) == 2 else frame.shape
            if expected_dimensions and current_shape != expected_dimensions:
                height, width = expected_dimensions[:2]
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # Ensure proper format
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            # Determine frame ID to use: provided (global) or internal counter
            current_frame_id = provided_frame_id if provided_frame_id is not None else (frames_written + 1)

            # Enhanced timestamp overlay (three-line format for comprehensive synchronization)
            if overlay_enabled:
                # Create a copy to avoid modifying original frame
                frame = frame.copy()

                # Capture current timestamps
                unix_timestamp = time.time()
                human_readable_time = datetime.fromtimestamp(unix_timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                # Build overlay text lines
                y_offset = 30
                if show_frame_id:
                    frame_text = f"Frame: {current_frame_id}"
                    cv2.putText(frame, frame_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, overlay_color,
                               overlay_thickness, cv2.LINE_AA)
                    y_offset += line_spacing

                if show_unix_time:
                    unix_text = f"Unix: {unix_timestamp:.3f}"
                    cv2.putText(frame, unix_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, overlay_color,
                               overlay_thickness, cv2.LINE_AA)
                    y_offset += line_spacing

                if show_human_time:
                    time_text = f"Time: {human_readable_time}"
                    cv2.putText(frame, time_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, overlay_color,
                               overlay_thickness, cv2.LINE_AA)

            # Write frame (ignore return value since we know it works)
            if writer:
                writer.write(frame)

                # Periodic flush monitoring (every 30 frames = ~1 second at 30 FPS)
                if (frames_written + 1) % 30 == 0:
                    try:
                        # Note: cv2.VideoWriter with MJPG codec writes frames progressively to disk.
                        # OpenCV doesn't expose explicit flush(), but MJPG writes complete frames immediately.
                        # We verify this by checking file size on disk periodically.
                        if filepath.exists():
                            file_size_mb = filepath.stat().st_size / 1024 / 1024
                            if file_size_mb < (frames_written + 1) * 0.001:  # Sanity check: at least ~1KB per frame
                                print(f"[Recorder Cam {camera_index}] WARNING: File size ({file_size_mb:.1f} MB) "
                                      f"seems too small for {frames_written + 1} frames")
                    except Exception as e:
                        print(f"[Recorder] Flush monitoring error: {e}")

                # Enhanced status logging (every 300 frames = ~10 seconds at 30 FPS)
                if (frames_written + 1) % 300 == 0:
                    try:
                        file_size_mb = filepath.stat().st_size / 1024 / 1024
                        elapsed = time.time() - start_time if start_time else 0
                        avg_fps = (frames_written + 1) / elapsed if elapsed > 0 else 0
                        print(f"[Recorder Cam {camera_index}] Status: {frames_written + 1} frames, "
                              f"{file_size_mb:.1f} MB on disk, {avg_fps:.1f} FPS avg, "
                              f"{lsl_samples_pushed} LSL samples")
                    except Exception as e:
                        print(f"[Recorder] Status logging error: {e}")

            frames_written += 1
            frame_count.value = frames_written

            # LSL push logic - use current_frame_id for synchronization
            unix_time = time.time()
            video_time = (frames_written - 1) / fps

            # Throttle LSL pushes
            if unix_time - last_push_time >= push_interval:
                try:
                    lsl_time = local_clock()
                    sample = [
                        float(current_frame_id),  # Use provided/global frame ID for synchronization
                        unix_time,
                        lsl_time,
                        video_time
                    ]
                    outlet.push_sample(sample, lsl_time)
                    last_push_time = unix_time
                    lsl_samples_pushed += 1
                    
                    if lsl_samples_pushed % 30 == 1:  # Log periodically
                        print(f"[Recorder] Frame {frames_written}, LSL samples: {lsl_samples_pushed}")
                        
                except Exception as e:
                    print(f"[Recorder] Error pushing to LSL: {e}")
                    
        except queue.Empty:
            if stop_flag.value:
                break
            continue
        except Exception as e:
            print(f"[Recorder] Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    print(f"[Recorder] Finalizing - {frames_written} frames, {lsl_samples_pushed} LSL samples")
    
    if writer:
        writer.release()
    
    # Final LSL sample
    if outlet and frames_written > 0:
        try:
            sample = [float(frames_written), time.time(), local_clock(), frames_written/fps]
            outlet.push_sample(sample)
        except:
            pass
    
    # Wait before closing outlet
    time.sleep(0.5)
    
    # Verify file
    if filepath.exists():
        file_size = filepath.stat().st_size
        print(f"[Recorder] Saved {filepath.name} ({file_size/1024/1024:.1f} MB)")
    
    print(f"[Recorder] Done.")