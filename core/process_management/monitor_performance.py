"""
Real-time performance monitoring for YouQuantiPy
Tracks landmark retrieval rate and ring buffer efficiency
"""

import time
import re
import sys
import os
from collections import deque
from datetime import datetime
import threading
import subprocess
from .confighandler import ConfigHandler

class PerformanceMonitor:
    def __init__(self, log_file="test_output.log", config=None):
        # Load configuration
        self.config = config if config else ConfigHandler()
        
        self.log_file = log_file
        
        # Get configuration values with defaults
        metrics_history_size = self.config.get('process_management.performance.metrics_history_size', 10)
        timestamps_history_size = self.config.get('process_management.performance.timestamps_history_size', 100)
        self.target_rate = self.config.get('process_management.performance.target_retrieval_rate', 2.0)
        self.update_interval = self.config.get('process_management.performance.update_interval_seconds', 1.0)
        self.progress_bar_length = self.config.get('process_management.performance.progress_bar_length', 40)
        self.cleanup_frequency = self.config.get('process_management.performance.cleanup_frequency', 10)
        
        self.metrics = {
            'landmark_retrievals': 0,
            'frames_processed': 0,
            'wait_cycles': 0,
            'buffer_recoveries': 0,
            'detection_fps': deque(maxlen=metrics_history_size),
            'landmark_fps': deque(maxlen=metrics_history_size),
            'capture_fps': deque(maxlen=metrics_history_size),
            'retrieval_timestamps': deque(maxlen=timestamps_history_size)
        }
        self.start_time = time.time()
        self.monitoring = True
        
    def parse_log_line(self, line):
        """Extract metrics from log line."""
        # Landmark retrieval
        if "Landmark extraction complete" in line or "landmarks extracted" in line:
            self.metrics['landmark_retrievals'] += 1
            self.metrics['retrieval_timestamps'].append(time.time())
            
        # Frame processing
        elif re.search(r'Processing frame (\d+)', line):
            self.metrics['frames_processed'] += 1
            
        # Wait cycles
        elif "FRAME WAIT" in line:
            self.metrics['wait_cycles'] += 1
            
        # Buffer recovery
        elif "RECOVERY" in line and "jumping to" in line:
            self.metrics['buffer_recoveries'] += 1
            
        # FPS metrics
        elif match := re.search(r'Detection FPS: ([\d.]+)', line):
            self.metrics['detection_fps'].append(float(match.group(1)))
        elif match := re.search(r'Landmark batch FPS: ([\d.]+)', line):
            self.metrics['landmark_fps'].append(float(match.group(1)))
        elif match := re.search(r'capture FPS: ([\d.]+)', line):
            self.metrics['capture_fps'].append(float(match.group(1)))
    
    def calculate_retrieval_rate(self):
        """Calculate current retrieval rate."""
        if len(self.metrics['retrieval_timestamps']) < 2:
            return 0.0
        
        # Calculate rate over last 10 seconds
        now = time.time()
        recent_retrievals = [ts for ts in self.metrics['retrieval_timestamps'] 
                           if now - ts < 10.0]
        
        if len(recent_retrievals) < 2:
            return 0.0
        
        time_span = now - recent_retrievals[0]
        if time_span > 0:
            return len(recent_retrievals) / time_span
        return 0.0
    
    def get_average_fps(self, fps_list):
        """Get average FPS from deque."""
        if not fps_list:
            return 0.0
        return sum(fps_list) / len(fps_list)
    
    def print_dashboard(self):
        """Print performance dashboard."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        elapsed = time.time() - self.start_time
        retrieval_rate = self.calculate_retrieval_rate()
        target_achievement = (retrieval_rate / self.target_rate) * 100
        
        print("=" * 60)
        print("YouQuantiPy Performance Monitor".center(60))
        print("=" * 60)
        print(f"Elapsed Time: {elapsed:.1f}s")
        print()
        
        # Retrieval Rate (Main Metric)
        print("🎯 LANDMARK RETRIEVAL RATE")
        print(f"  Current Rate: {retrieval_rate:.2f}/sec")
        print(f"  Target Rate:  {self.target_rate:.2f}/sec")
        print(f"  Achievement:  {target_achievement:.1f}%")
        
        # Progress bar
        filled = int(self.progress_bar_length * min(target_achievement / 100, 1.0))
        bar = "█" * filled + "░" * (self.progress_bar_length - filled)
        print(f"  Progress: [{bar}]")
        print()
        
        # FPS Metrics
        print("📊 FPS METRICS")
        print(f"  Capture FPS:   {self.get_average_fps(self.metrics['capture_fps']):.1f}")
        print(f"  Detection FPS: {self.get_average_fps(self.metrics['detection_fps']):.1f}")
        print(f"  Landmark FPS:  {self.get_average_fps(self.metrics['landmark_fps']):.1f}")
        print()
        
        # Processing Metrics
        print("⚙️  PROCESSING METRICS")
        print(f"  Total Retrievals:    {self.metrics['landmark_retrievals']}")
        print(f"  Frames Processed:    {self.metrics['frames_processed']}")
        print(f"  Wait Cycles:         {self.metrics['wait_cycles']}")
        print(f"  Buffer Recoveries:   {self.metrics['buffer_recoveries']}")
        print()
        
        # Efficiency Metrics
        if self.metrics['frames_processed'] > 0:
            efficiency = (self.metrics['landmark_retrievals'] / self.metrics['frames_processed']) * 100
            wait_ratio = (self.metrics['wait_cycles'] / self.metrics['frames_processed']) * 100
            
            print("📈 EFFICIENCY METRICS")
            print(f"  Retrieval Efficiency: {efficiency:.1f}%")
            print(f"  Wait Ratio:          {wait_ratio:.1f}%")
            print()
        
        # Status
        if target_achievement >= 100:
            print("✅ STATUS: TARGET ACHIEVED!")
        elif target_achievement >= 80:
            print("🟡 STATUS: Near target")
        elif target_achievement >= 50:
            print("🟠 STATUS: Moderate performance")
        else:
            print("🔴 STATUS: Performance issues")
        
        print("=" * 60)
        print("Press Ctrl+C to exit")
    
    def tail_log_file(self):
        """Tail the log file and parse new lines."""
        try:
            # Use subprocess to tail the file
            cmd = f"tail -f {self.log_file}"
            if os.name == 'nt':  # Windows
                cmd = f"powershell Get-Content {self.log_file} -Wait -Tail 0"
            
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            for line in process.stdout:
                if not self.monitoring:
                    break
                self.parse_log_line(line.strip())
                
        except Exception as e:
            print(f"Error tailing log file: {e}")
    
    def run(self):
        """Run the performance monitor."""
        # Start log tailing in background thread
        tail_thread = threading.Thread(target=self.tail_log_file)
        tail_thread.daemon = True
        tail_thread.start()
        
        try:
            while self.monitoring:
                self.print_dashboard()
                time.sleep(self.update_interval)  # Update dashboard based on config
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            self.monitoring = False

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor YouQuantiPy performance")
    parser.add_argument("--log", default="test_output.log", 
                       help="Log file to monitor (default: test_output.log)")
    parser.add_argument("--target", type=float, default=None,
                       help="Target retrieval rate per second (uses config if not specified)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log):
        print(f"Error: Log file '{args.log}' not found!")
        print("Make sure YouQuantiPy is running and outputting to the log file.")
        return
    
    monitor = PerformanceMonitor(args.log)
    if args.target is not None:
        monitor.target_rate = args.target
    monitor.run()

if __name__ == "__main__":
    main()