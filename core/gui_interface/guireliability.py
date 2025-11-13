import threading
import time
import gc
import psutil
import os
from collections import deque
import queue
from pathlib import Path
import json
import signal
import atexit
from datetime import datetime

"""
GUI Reliability Monitor Module

Provides comprehensive monitoring and protection for long-running GUI applications.
Prevents memory leaks, queue overflows, and GUI freezes during extended operation.
"""

class GUIReliabilityMonitor:
    """
    Comprehensive reliability monitoring for GUI applications.
    
    Features:
    - Memory usage monitoring and automatic cleanup
    - Queue health monitoring with overflow prevention
    - GUI responsiveness watchdog
    - Performance statistics tracking
    - Emergency recovery capabilities
    """
    
    def __init__(self, gui_instance, config=None):
        self.gui = gui_instance
        self.config = config or {}
        
        # Load reliability monitor config
        reliability_config = self.config.get('gui_interface', {}).get('reliability_monitor', {})
        self.monitor_config = {**self._default_config(), **reliability_config}
        
        # Monitoring threads
        self.resource_monitor_thread = None
        self.queue_health_thread = None
        self.gui_watchdog_thread = None
        
        # Monitoring states
        self.resource_monitor_active = False
        self.queue_health_active = False
        self.gui_watchdog_active = False
        
        # Tracking variables
        self.last_memory_usage = 0
        self.last_gui_update = time.time()
        
        # Performance statistics
        self.performance_stats = {
            'preview_updates': 0,
            'dropped_frames': 0,
            'queue_overflows': 0,
            'memory_warnings': 0,
            'gui_freeze_warnings': 0,
            'emergency_cleanups': 0,
            'start_time': time.time()
        }
        
        # Don't setup emergency handlers immediately - wait for start_monitoring()
        self._signal_handlers_installed = False
        
        print("[Reliability] Monitor initialized")
    
    def _default_config(self):
        """Default configuration settings"""
        return {
            'memory_growth_threshold': 500,    # MB growth before cleanup
            'max_queue_size': 10,              # Max queue items before drain
            'gui_freeze_threshold': 60.0,      # Seconds before freeze warning
            'resource_check_interval': 30,     # Seconds between resource checks
            'queue_check_interval': 5,         # Seconds between queue checks
            'gui_check_interval': 1,           # Seconds between GUI checks
            'stats_report_interval': 300,      # Seconds between stats reports (5 min)
            'canvas_cache_limit': 10,          # Max cached objects per canvas
            'startup_grace_period': 30.0       # Grace period before watchdog actions
        }
    
    def _setup_emergency_handlers(self):
        """Setup emergency signal handlers"""
        try:
            # REMOVED: Signal handlers already registered in gui.py __main__ before mainloop()
            # Overwriting them after mainloop() starts breaks Tkinter event loop on WSL2/Linux
            # The main signal handlers in gui.py call _shutdown_all_processes() which triggers
            # reliability monitor cleanup, so no functionality is lost
            # signal.signal(signal.SIGINT, self._emergency_signal_handler)
            # signal.signal(signal.SIGTERM, self._emergency_signal_handler)

            # Keep atexit handler for process exit cleanup
            atexit.register(self._cleanup_on_exit)
        except Exception as e:
            print(f"[Reliability] Could not setup signal handlers: {e}")
    
    def _emergency_signal_handler(self, signum, frame):
        """Handle emergency shutdown signals"""
        print(f"[Reliability] Emergency signal {signum} received")
        
        # CRITICAL: Set shutdown flag first to stop callbacks
        if hasattr(self.gui, '_shutdown_in_progress'):
            self.gui._shutdown_in_progress = True
        
        # Cancel all pending callbacks before cleanup
        if hasattr(self.gui, '_cancel_all_pending_callbacks'):
            self.gui._cancel_all_pending_callbacks()
        
        # Mark cleanup as completed to prevent double execution
        if hasattr(self.gui, '_emergency_cleanup_completed'):
            if self.gui._emergency_cleanup_completed:
                print("[Reliability] Emergency cleanup already completed, skipping")
                return
            self.gui._emergency_cleanup_completed = True
        
        self.emergency_cleanup()
        
        # Only call shutdown if not already in progress
        if hasattr(self.gui, '_shutdown_all_processes') and not getattr(self.gui, '_shutdown_started', False):
            self.gui._shutdown_started = True
            self.gui._shutdown_all_processes()
    
    def _cleanup_on_exit(self):
        """Cleanup function called on program exit"""
        print("[Reliability] Program exit cleanup")
        self.stop_monitoring()
    
    def start_monitoring(self):
        """Start all monitoring threads"""
        if self.resource_monitor_active:
            print("[Reliability] Monitoring already active")
            return
        
        # Setup emergency handlers only when monitoring starts
        if not self._signal_handlers_installed:
            self._setup_emergency_handlers()
            self._signal_handlers_installed = True
        
        print("[Reliability] Starting monitoring threads...")
        
        self.resource_monitor_active = True
        self.queue_health_active = True
        self.gui_watchdog_active = True
        
        # Start monitoring threads
        self.resource_monitor_thread = threading.Thread(
            target=self._resource_monitor_loop, daemon=True, name="ResourceMonitor")
        self.resource_monitor_thread.start()
        
        self.queue_health_thread = threading.Thread(
            target=self._queue_health_loop, daemon=True, name="QueueMonitor")
        self.queue_health_thread.start()
        
        self.gui_watchdog_thread = threading.Thread(
            target=self._gui_watchdog_loop, daemon=True, name="GUIWatchdog")
        self.gui_watchdog_thread.start()
        
        print("[Reliability] All monitoring threads started")
    
    def stop_monitoring(self):
        """Stop all monitoring threads"""
        print("[Reliability] Stopping monitoring...")
        
        self.resource_monitor_active = False
        self.queue_health_active = False
        self.gui_watchdog_active = False
        
        # Wait for threads to finish
        for thread in [self.resource_monitor_thread, 
                      self.queue_health_thread, 
                      self.gui_watchdog_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=1.0)
        
        print("[Reliability] Monitoring stopped")
    
    def _resource_monitor_loop(self):
        """Monitor memory usage and force cleanup if needed"""
        last_stats_report = time.time()
        
        while self.resource_monitor_active:
            try:
                # Get current memory usage
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                # Check for memory growth
                if self.last_memory_usage > 0:
                    growth = memory_mb - self.last_memory_usage
                    if growth > self.monitor_config['memory_growth_threshold']:
                        print(f"[Reliability] WARNING: Memory grew by {growth:.1f}MB (now {memory_mb:.1f}MB)")
                        self.performance_stats['memory_warnings'] += 1
                        
                        # Trigger cleanup
                        self._perform_memory_cleanup()
                
                self.last_memory_usage = memory_mb
                
                # Periodic stats reporting
                current_time = time.time()
                if current_time - last_stats_report >= self.monitor_config['stats_report_interval']:
                    self._report_stats(memory_mb)
                    last_stats_report = current_time
                
            except Exception as e:
                print(f"[Reliability] Resource monitor error: {e}")
            
            time.sleep(self.monitor_config['resource_check_interval'])
    
    def _perform_memory_cleanup(self):
        """Perform memory cleanup operations"""
        print("[Reliability] Performing memory cleanup...")
        
        try:
            # Force garbage collection
            collected = gc.collect()
            print(f"[Reliability] Garbage collected {collected} objects")
            
            # Clear photo image cache if it exists
            if hasattr(self.gui, '_photo_images'):
                cache_size = len(self.gui._photo_images)
                self.gui._photo_images.clear()
                print(f"[Reliability] Cleared photo cache ({cache_size} images)")
            
            # Clean canvas cache
            self._cleanup_canvas_cache()
            
        except Exception as e:
            print(f"[Reliability] Error during memory cleanup: {e}")
    
    def _cleanup_canvas_cache(self):
        """Clean up canvas object cache to prevent memory leaks"""
        try:
            if not hasattr(self.gui, 'canvas_objects'):
                return
                
            cleaned_objects = 0
            
            for canvas_idx in list(self.gui.canvas_objects.keys()):
                cache = self.gui.canvas_objects[canvas_idx]
                
                # Clean up face lines
                if 'face_lines' in cache:
                    face_ids = list(cache['face_lines'].keys())
                    if len(face_ids) > self.monitor_config['canvas_cache_limit']:
                        # Remove oldest entries
                        to_remove = face_ids[:-self.monitor_config['canvas_cache_limit']]
                        for face_id in to_remove:
                            lines = cache['face_lines'].pop(face_id, [])
                            self._delete_canvas_objects(canvas_idx, lines)
                            cleaned_objects += len(lines)
                
                # Clean up face labels
                if 'face_labels' in cache:
                    label_ids = list(cache['face_labels'].keys())
                    if len(label_ids) > self.monitor_config['canvas_cache_limit']:
                        to_remove = label_ids[:-self.monitor_config['canvas_cache_limit']]
                        for face_id in to_remove:
                            label_id = cache['face_labels'].pop(face_id, None)
                            if label_id:
                                self._delete_canvas_objects(canvas_idx, [label_id])
                                cleaned_objects += 1
            
            if cleaned_objects > 0:
                print(f"[Reliability] Canvas cache cleaned - removed {cleaned_objects} objects")
                
        except Exception as e:
            print(f"[Reliability] Error cleaning canvas cache: {e}")
    
    def _delete_canvas_objects(self, canvas_idx, object_ids):
        """Safely delete canvas objects"""
        try:
            if canvas_idx < len(self.gui.frames):
                canvas = self.gui.frames[canvas_idx]['canvas']
                for obj_id in object_ids:
                    try:
                        canvas.delete(obj_id)
                    except:
                        pass
        except:
            pass
    
    def _queue_health_loop(self):
        """Monitor queue sizes and prevent overflow"""
        while self.queue_health_active:
            try:
                self._check_preview_queues()
                self._check_recording_queues()
                self._check_system_queues()
                
            except Exception as e:
                print(f"[Reliability] Queue health monitor error: {e}")
            
            time.sleep(self.monitor_config['queue_check_interval'])
    
    def _check_preview_queues(self):
        """Check preview queue health"""
        if not hasattr(self.gui, 'preview_queues'):
            return
            
        for idx, q in enumerate(self.gui.preview_queues):
            if q and q.qsize() > self.monitor_config['max_queue_size']:
                print(f"[Reliability] WARNING: Preview queue {idx} overflow ({q.qsize()} items)")
                self.performance_stats['queue_overflows'] += 1
                
                # Emergency drain
                drained = self._drain_queue(q, keep_latest=2)
                print(f"[Reliability] Drained {drained} items from preview queue {idx}")
    
    def _check_recording_queues(self):
        """Check recording queue health"""
        if not hasattr(self.gui, 'recording_queues'):
            return
        
        queues = self.gui.recording_queues
        if isinstance(queues, dict):
            queue_items = queues.items()
        elif isinstance(queues, list):
            queue_items = enumerate(queues)
        else:
            return
            
        for idx, q in queue_items:
            if q and q.qsize() > self.monitor_config['max_queue_size'] * 2:  # Recording queues can be larger
                print(f"[Reliability] WARNING: Recording queue {idx} overflow ({q.qsize()} items)")
                # Don't drain recording queues as aggressively
    
    def _check_system_queues(self):
        """Check system queue health"""
        critical_queues = [
            ('participant_update', getattr(self.gui, 'participant_update_queue', None)),
            ('correlation', getattr(self.gui, 'correlation_queue', None)),
            ('lsl_command', getattr(self.gui, 'lsl_command_queue', None)),
            ('lsl_data', getattr(self.gui, 'lsl_data_queue', None))
        ]
        
        for name, q in critical_queues:
            if q and q.qsize() > 50:  # These should stay small
                print(f"[Reliability] WARNING: {name} queue overflow ({q.qsize()} items)")
                # Emergency drain for non-essential queues
                if name in ['lsl_data']:
                    drained = self._drain_queue(q, keep_latest=10)
                    print(f"[Reliability] Emergency drained {drained} items from {name} queue")
    
    def _drain_queue(self, queue_obj, keep_latest=2):
        """Safely drain a queue, keeping only the latest items"""
        drained = 0
        try:
            while queue_obj.qsize() > keep_latest:
                try:
                    queue_obj.get_nowait()
                    drained += 1
                except:
                    break
        except:
            pass
        return drained
    
    def _gui_watchdog_loop(self):
        """Monitor GUI responsiveness"""
        # Add startup grace period to prevent false positives
        startup_grace_period = self.monitor_config.get('startup_grace_period', 30.0)
        watchdog_start_time = time.time()
        
        while self.gui_watchdog_active:
            try:
                current_time = time.time()
                time_since_startup = current_time - watchdog_start_time
                time_since_update = current_time - self.last_gui_update
                
                # Skip watchdog actions during grace period
                if time_since_startup < startup_grace_period:
                    pass  # Do nothing during grace period
                elif time_since_update > self.monitor_config['gui_freeze_threshold']:
                    print(f"[Reliability] WARNING: GUI may be frozen ({time_since_update:.1f}s since last update)")
                    self.performance_stats['gui_freeze_warnings'] += 1
                    
                    # Emergency actions for severe freezes
                    if time_since_update > 60.0:
                        print("[Reliability] CRITICAL: GUI freeze detected, taking emergency actions")
                        self._emergency_gui_recovery()
                
            except Exception as e:
                print(f"[Reliability] GUI watchdog error: {e}")
            
            time.sleep(self.monitor_config['gui_check_interval'])
    
    def _emergency_gui_recovery(self):
        """Emergency GUI recovery actions"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Try to force a GUI update
            try:
                self.gui.after_idle(lambda: setattr(self, 'last_gui_update', time.time()))
            except:
                pass
            
            # Trigger emergency cleanup
            self.emergency_cleanup()
            
        except Exception as e:
            print(f"[Reliability] Error in emergency GUI recovery: {e}")
    
    def update_gui_timestamp(self):
        """Call this from the GUI's main update loop to track responsiveness"""
        current_time = time.time()
        self.last_gui_update = current_time
    
    def track_preview_update(self):
        """Call this when a preview update completes"""
        self.performance_stats['preview_updates'] += 1
    
    def track_dropped_frames(self, count=1):
        """Call this when frames are dropped"""
        self.performance_stats['dropped_frames'] += count
    
    def emergency_cleanup(self):
        """Emergency cleanup function to recover from potential issues"""
        print("[Reliability] EMERGENCY CLEANUP INITIATED")
        self.performance_stats['emergency_cleanups'] += 1
        
        try:
            # Force garbage collection
            collected = gc.collect()
            print(f"[Reliability] Emergency GC collected {collected} objects")
            
            # Clear all caches
            if hasattr(self.gui, 'canvas_objects'):
                cache_size = len(self.gui.canvas_objects)
                self.gui.canvas_objects.clear()
                print(f"[Reliability] Cleared canvas cache ({cache_size} entries)")
            
            if hasattr(self.gui, 'transform_cache'):
                self.gui.transform_cache.clear()
                print("[Reliability] Cleared transform cache")
            
            if hasattr(self.gui, 'last_frame_time'):
                self.gui.last_frame_time.clear()
                print("[Reliability] Cleared frame time cache")
            
            if hasattr(self.gui, '_photo_images'):
                self.gui._photo_images.clear()
                print("[Reliability] Cleared photo image cache")
            
            # Force canvas updates
            for info in getattr(self.gui, 'frames', []):
                try:
                    canvas = info.get('canvas')
                    if canvas:
                        canvas.delete('overlay')  # Clear all overlays
                        canvas.update_idletasks()
                except:
                    pass
            
            print("[Reliability] Emergency cleanup completed")
            
        except Exception as e:
            print(f"[Reliability] Error in emergency cleanup: {e}")
    
    def _report_stats(self, memory_mb):
        """Report performance statistics"""
        uptime = time.time() - self.performance_stats['start_time']
        
        print(f"\n[Reliability] === PERFORMANCE REPORT ===")
        print(f"Uptime: {uptime/3600:.1f} hours")
        print(f"Memory: {memory_mb:.1f} MB")
        print(f"Preview updates: {self.performance_stats['preview_updates']}")
        print(f"Dropped frames: {self.performance_stats['dropped_frames']}")
        print(f"Queue overflows: {self.performance_stats['queue_overflows']}")
        print(f"Memory warnings: {self.performance_stats['memory_warnings']}")
        print(f"GUI freeze warnings: {self.performance_stats['gui_freeze_warnings']}")
        print(f"Emergency cleanups: {self.performance_stats['emergency_cleanups']}")
        
        # Calculate rates
        if uptime > 0:
            update_rate = self.performance_stats['preview_updates'] / uptime
            print(f"Average GUI update rate: {update_rate:.1f} updates/sec")
        
        print("=" * 40)
    
    def get_stats(self):
        """Get current performance statistics"""
        uptime = time.time() - self.performance_stats['start_time']
        stats = self.performance_stats.copy()
        stats['uptime_hours'] = uptime / 3600
        stats['memory_mb'] = self.last_memory_usage
        return stats


class VideoRecordingProtection:
    """
    Protection system for video recordings to prevent data loss.
    """
    
    def __init__(self, gui_instance):
        self.gui = gui_instance
        self.recording_state_file = Path("recording_state.json")
        self.emergency_stop_flag = False
        
        # Register emergency handlers - DISABLED FOR DEBUGGING
        print("[DEBUG] DISABLED: Recording protection signal handlers disabled for debugging")
        # signal.signal(signal.SIGINT, self._emergency_stop_handler)
        # signal.signal(signal.SIGTERM, self._emergency_stop_handler)
        # atexit.register(self._cleanup_on_exit)
        
        # Check for recovery on startup
        self._check_recovery_state()
    
    def _emergency_stop_handler(self, signum, frame):
        """Handle emergency shutdown signals"""
        print(f"[Recording Protection] Emergency signal {signum} received")
        self.emergency_stop_flag = True
        self._save_emergency_state()
        self._emergency_stop_all_recordings()
    
    def _cleanup_on_exit(self):
        """Cleanup function called on program exit"""
        if hasattr(self.gui, 'recording_active') and self.gui.recording_active:
            print("[Recording Protection] Emergency cleanup on exit")
            self._emergency_stop_all_recordings()
    
    def _save_emergency_state(self):
        """Save current recording state for recovery"""
        try:
            state = {
                'timestamp': time.time(),
                'recording_active': getattr(self.gui, 'recording_active', False),
                'active_cameras': [],
                'save_directory': getattr(self.gui, 'save_dir', {}).get() if hasattr(self.gui, 'save_dir') else '',
                'filename_template': getattr(self.gui, 'filename_template', {}).get() if hasattr(self.gui, 'filename_template') else ''
            }
            
            # Save info about active recordings
            if hasattr(self.gui, 'video_recorders'):
                for idx, rec_info in self.gui.video_recorders.items():
                    if 'recorder' in rec_info:
                        recorder = rec_info['recorder']
                        state['active_cameras'].append({
                            'index': idx,
                            'participant': getattr(recorder, 'participant', f'P{idx+1}'),
                            'frame_count': recorder.get_frame_count() if hasattr(recorder, 'get_frame_count') else 0
                        })
            
            with open(self.recording_state_file, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"[Recording Protection] Emergency state saved")
            
        except Exception as e:
            print(f"[Recording Protection] Failed to save emergency state: {e}")
    
    def _check_recovery_state(self):
        """Check if there's a previous recording to recover"""
        if not self.recording_state_file.exists():
            return
            
        try:
            with open(self.recording_state_file, 'r') as f:
                state = json.load(f)
            
            # Check if the state is recent (within last hour)
            if time.time() - state['timestamp'] < 3600:
                self._handle_recovery(state)
            
            # Clean up state file
            self.recording_state_file.unlink()
            
        except Exception as e:
            print(f"[Recording Protection] Error checking recovery state: {e}")
    
    def _handle_recovery(self, state):
        """Handle recovery from previous session"""
        if not state.get('active_cameras'):
            return
            
        print("[Recording Protection] Found recent recording state, checking for recovery...")
        
        save_dir = Path(state['save_directory'])
        if not save_dir.exists():
            return
            
        recovered_files = []
        
        for camera_info in state['active_cameras']:
            participant = camera_info['participant']
            pattern = f"{participant}_*.avi"
            recent_files = list(save_dir.glob(pattern))
            
            if recent_files:
                latest_file = max(recent_files, key=lambda p: p.stat().st_mtime)
                file_age = time.time() - latest_file.stat().st_mtime
                
                if file_age < 3600:  # Within last hour
                    recovered_files.append({
                        'file': latest_file,
                        'participant': participant,
                        'size_mb': latest_file.stat().st_size / (1024*1024)
                    })
        
        if recovered_files:
            self._show_recovery_dialog(recovered_files)
    
    def _show_recovery_dialog(self, recovered_files):
        """Show recovery dialog to user"""
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            file_list = "\n".join([f"• {rf['file'].name} ({rf['size_mb']:.1f} MB)" 
                                  for rf in recovered_files])
            
            message = (f"Recording Recovery Detected!\n\n"
                      f"Found {len(recovered_files)} recent recording(s):\n{file_list}\n\n"
                      f"These files appear to be from a previous session that may not have "
                      f"closed properly. Please verify they contain valid data.")
            
            messagebox.showinfo("Recording Recovery", message)
            
        except Exception as e:
            print(f"[Recording Protection] Could not show recovery dialog: {e}")
    
    def _emergency_stop_all_recordings(self):
        """Emergency stop all recordings"""
        try:
            print("[Recording Protection] Emergency stop initiated...")
            
            if hasattr(self.gui, 'recording_active'):
                self.gui.recording_active = False
            
            time.sleep(0.5)  # Let threads see the flag
            
            # Force stop all recorders
            if hasattr(self.gui, 'video_recorders'):
                for idx, rec_info in list(self.gui.video_recorders.items()):
                    try:
                        print(f"[Recording Protection] Emergency stopping recorder {idx}")
                        
                        if 'stop_flag' in rec_info and rec_info['stop_flag']:
                            rec_info['stop_flag'].set()
                        
                        if 'recorder' in rec_info:
                            recorder = rec_info['recorder']
                            if hasattr(recorder, 'get_frame_count'):
                                frame_count = recorder.get_frame_count()
                                print(f"[Recording Protection] Recorder {idx} had {frame_count} frames")
                            
                            recorder.stop_recording()
                            time.sleep(0.2)  # Give it time to finish
                        
                    except Exception as e:
                        print(f"[Recording Protection] Error stopping recorder {idx}: {e}")
                
                self.gui.video_recorders.clear()
            
            print("[Recording Protection] Emergency stop completed")
            
        except Exception as e:
            print(f"[Recording Protection] Critical error in emergency stop: {e}")


# Convenience function for easy integration
def setup_reliability_monitoring(gui_instance, config=None):
    """
    Convenience function to set up reliability monitoring for a GUI instance.
    
    Args:
        gui_instance: The main GUI instance
        config: Optional configuration dict
        
    Returns:
        tuple: (reliability_monitor, recording_protection)
    """
    reliability_monitor = GUIReliabilityMonitor(gui_instance, config)
    recording_protection = VideoRecordingProtection(gui_instance)
    
    # Add convenience methods to GUI instance
    gui_instance.reliability_monitor = reliability_monitor
    gui_instance.recording_protection = recording_protection
    gui_instance.emergency_cleanup = reliability_monitor.emergency_cleanup
    gui_instance.get_performance_stats = reliability_monitor.get_stats
    
    return reliability_monitor, recording_protection