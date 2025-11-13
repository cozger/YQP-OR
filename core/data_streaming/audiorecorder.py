import pyaudio
import wave
import threading
import queue
import numpy as np
from datetime import datetime
from pathlib import Path
from pylsl import StreamInfo, StreamOutlet
import sounddevice as sd
import time


class AudioDeviceManager:
    """Manages audio device enumeration and selection"""
    
    @staticmethod
    def list_audio_devices():
        """List all available audio input devices"""
        devices = []
        p = pyaudio.PyAudio()
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:  # Input device
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate']),
                    'host_api': info['hostApi']
                })
        
        p.terminate()
        return devices
    
    @staticmethod
    def get_default_input_device():
        """Get the default input device index"""
        p = pyaudio.PyAudio()
        try:
            info = p.get_default_input_device_info()
            device_index = info['index']
        except:
            device_index = None
        p.terminate()
        return device_index


class AudioRecorder:
    """Standalone audio recorder with LSL support"""
    
    def __init__(self, device_index=None, sample_rate=None, channels=None, 
                 chunk_size=None, output_dir="recordings", config=None):
        self.config = config or {}
        self.device_index = device_index
        
        # Get settings from config with fallbacks
        audio_config = self.config.get('audio_recording', {})
        self.sample_rate = sample_rate if sample_rate is not None else audio_config.get('sample_rate', 44100)
        self.channels = channels if channels is not None else audio_config.get('channels', 1)
        self.chunk_size = chunk_size if chunk_size is not None else audio_config.get('chunk_size', 1024)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.recording = False
        self.audio = None
        self.stream = None
        self.wave_file = None
        self.record_thread = None
        self.audio_queue = queue.Queue()
        
        # LSL outlet for audio timestamps
        self.lsl_outlet = None
        self.sample_count = 0
        
    def setup_lsl_stream(self, participant_id=""):
        """Setup LSL stream for audio timestamps"""
        stream_name = f"{participant_id}_audio_timestamps" if participant_id else "audio_timestamps"
        info = StreamInfo(
            name=stream_name,
            type="AudioSync",
            channel_count=2,  # sample_number, timestamp
            nominal_srate=self.sample_rate / self.chunk_size,  # Chunk rate
            channel_format="float32",
            source_id=f"{stream_name}_uid"
        )
        self.lsl_outlet = StreamOutlet(info)
        print(f"[AudioRecorder] LSL stream created: {stream_name}")
        
    def start_recording(self, participant_id="", filename=None):
        """Start audio recording"""
        if self.recording:
            return False
            
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{participant_id}_audio_{timestamp}.wav" if participant_id else f"audio_{timestamp}.wav"
        
        filepath = self.output_dir / filename
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Open wave file for writing
        self.wave_file = wave.open(str(filepath), 'wb')
        self.wave_file.setnchannels(self.channels)
        self.wave_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        self.wave_file.setframerate(self.sample_rate)
        
        # Setup LSL if not already done
        if self.lsl_outlet is None:
            self.setup_lsl_stream(participant_id)
            
        # Open audio stream
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.recording = True
            self.sample_count = 0
            
            # Start processing thread
            self.record_thread = threading.Thread(target=self._process_audio, daemon=True)
            self.record_thread.start()
            
            # Start stream
            self.stream.start_stream()
            
            print(f"[AudioRecorder] Started recording to {filepath}")
            return True
            
        except Exception as e:
            print(f"[AudioRecorder] Failed to start recording: {e}")
            if self.wave_file:
                self.wave_file.close()
            return False
            
    def stop_recording(self):
        """Stop audio recording"""
        if not self.recording:
            return
            
        self.recording = False
        
        # Stop and close stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        # Wait for processing thread
        if self.record_thread:
            timeout = self.config.get('audio_recording', {}).get('queue_timeout_seconds', 2.0)
            self.record_thread.join(timeout=timeout)
            
        # Close wave file
        if self.wave_file:
            self.wave_file.close()
            
        # Terminate PyAudio
        if self.audio:
            self.audio.terminate()
            
        # Close LSL outlet
        if self.lsl_outlet:
            try:
                self.lsl_outlet = None  # LSL outlets are cleaned up automatically
            except:
                pass
                
        print(f"[AudioRecorder] Stopped recording. Total samples: {self.sample_count}")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if self.recording:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
        
    def _process_audio(self):
        """Process audio data from queue"""
        while self.recording or not self.audio_queue.empty():
            try:
                timeout = self.config.get('audio_recording', {}).get('timeout_seconds', 0.1)
                audio_data = self.audio_queue.get(timeout=timeout)
                
                # Write to file
                self.wave_file.writeframes(audio_data)
                
                # Update sample count
                samples = len(audio_data) // (2 * self.channels)  # 16-bit = 2 bytes
                self.sample_count += samples
                
                # Send LSL timestamp
                if self.lsl_outlet:
                    timestamp = time.time()
                    self.lsl_outlet.push_sample([self.sample_count, timestamp])
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AudioRecorder] Error processing audio: {e}")


class VideoAudioRecorder:
    """Enhanced video recorder with optional audio support"""
    
    def __init__(self, output_dir="recordings", video_codec=None, fps=None,
                 audio_device=None, audio_rate=None, audio_channels=None, config=None):
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get settings from config with fallbacks
        video_config = self.config.get('video_recording', {})
        audio_config = self.config.get('audio_recording', {})
        
        self.video_codec = video_codec if video_codec is not None else video_config.get('codec', 'MJPG')
        self.fps = fps if fps is not None else video_config.get('fps', 30)
        
        # Audio settings
        self.audio_device = audio_device
        self.audio_rate = audio_rate if audio_rate is not None else audio_config.get('sample_rate', 44100)
        self.audio_channels = audio_channels if audio_channels is not None else audio_config.get('channels', 1)
        self.audio_enabled = audio_device is not None
        
        # Recording state
        self.recording = False
        self.video_writer = None
        self.audio_recorder = None
        # Get queue size from config
        queue_size = self.config.get('video_recording', {}).get('queue_size', 60)
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.record_thread = None
        self.frame_number = 0
        
        # LSL outlets
        self.video_lsl_outlet = None
        self.audio_lsl_outlet = None
        
    def start_recording(self, width, height, participant_id="", with_audio=True):
        """Start recording video and optionally audio"""
        if self.recording:
            return False
            
        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{participant_id}_{timestamp}.avi" if participant_id else f"recording_{timestamp}.avi"
        audio_filename = f"{participant_id}_{timestamp}.wav" if participant_id else f"recording_{timestamp}.wav"
        
        video_filepath = self.output_dir / video_filename
        audio_filepath = self.output_dir / audio_filename
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        self.video_writer = cv2.VideoWriter(
            str(video_filepath), fourcc, self.fps, (width, height)
        )
        
        if not self.video_writer.isOpened():
            print(f"[VideoAudioRecorder] Failed to open video writer")
            return False
            
        # Setup audio recording if enabled
        if with_audio and self.audio_enabled:
            self.audio_recorder = AudioRecorder(
                device_index=self.audio_device,
                sample_rate=self.audio_rate,
                channels=self.audio_channels,
                output_dir=str(self.output_dir),
                config=self.config
            )
            if not self.audio_recorder.start_recording(participant_id, audio_filename):
                print(f"[VideoAudioRecorder] Failed to start audio recording")
                self.video_writer.release()
                return False
                
        # Setup LSL streams
        self._setup_lsl_streams(participant_id)
        
        self.recording = True
        self.frame_number = 0
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()
        
        print(f"[VideoAudioRecorder] Started recording video to {video_filepath}")
        if self.audio_recorder:
            print(f"[VideoAudioRecorder] Recording audio to {audio_filepath}")
            
        return True
        
    def stop_recording(self):
        """Stop video and audio recording"""
        if not self.recording:
            return
            
        self.recording = False
        
        # Wait for recording thread
        if self.record_thread:
            timeout = self.config.get('audio_recording', {}).get('queue_timeout_seconds', 2.0)
            self.record_thread.join(timeout=timeout)
            
        # Stop audio recording
        if self.audio_recorder:
            self.audio_recorder.stop_recording()
            self.audio_recorder = None
            
        # Close video writer
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        print(f"[VideoAudioRecorder] Stopped recording. Total frames: {self.frame_number}")
        
    def add_frame(self, frame):
        """Add a frame to the recording queue"""
        if not self.recording:
            return False
            
        try:
            self.frame_queue.put_nowait(frame.copy())
            return True
        except queue.Full:
            return False
            
    def _setup_lsl_streams(self, participant_id):
        """Setup LSL streams for synchronization"""
        # Video frame numbers
        video_stream_name = f"{participant_id}_video_frames" if participant_id else "video_frames"
        info = StreamInfo(
            name=video_stream_name,
            type="VideoSync",
            channel_count=2,
            nominal_srate=self.fps,
            channel_format="float32",
            source_id=f"{video_stream_name}_uid"
        )
        self.video_lsl_outlet = StreamOutlet(info)
        
    def _record_loop(self):
        """Recording thread loop"""
        while self.recording or not self.frame_queue.empty():
            try:
                timeout = self.config.get('video_recording', {}).get('timeout_seconds', 0.1)
                frame = self.frame_queue.get(timeout=timeout)
                
                # Write frame
                if self.video_writer and self.video_writer.isOpened():
                    self.video_writer.write(frame)
                    self.frame_number += 1
                    
                    # Send frame number via LSL
                    if self.video_lsl_outlet:
                        timestamp = time.time()
                        self.video_lsl_outlet.push_sample([self.frame_number, timestamp])
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VideoAudioRecorder] Error in record loop: {e}")


# Import this fix for cv2
import cv2