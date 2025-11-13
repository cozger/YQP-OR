"""
Process Health Monitoring and Recovery System
Provides comprehensive health monitoring and automatic recovery for GPU processes.
"""

import time
import logging
import threading
import psutil
import multiprocessing as mp
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
from queue import Queue, Empty
import traceback
from .confighandler import ConfigHandler

logger = logging.getLogger('ProcessHealthMonitor')

class ProcessState(Enum):
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    TERMINATED = "terminated"

@dataclass
class ProcessHealthInfo:
    """Health information for a monitored process."""
    process_id: int
    process_name: str
    state: ProcessState
    last_heartbeat: float
    error_count: int = 0
    restart_count: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_error_message: str = ""
    startup_time: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.startup_time
    
    def is_responsive(self, heartbeat_timeout: float = 10.0) -> bool:
        return time.time() - self.last_heartbeat < heartbeat_timeout

class ProcessHealthMonitor:
    """
    Comprehensive process health monitoring with automatic recovery.
    
    Features:
    - Real-time health monitoring
    - Automatic process restart on failure
    - Resource usage tracking
    - Detailed error logging and diagnostics
    - GUI status indicators
    """
    
    def __init__(self, max_restart_attempts: int = None, heartbeat_timeout: float = None, config=None):
        # Load configuration
        self.config = config if config else ConfigHandler()
        
        # Get configuration values with defaults
        self.max_restart_attempts = (
            max_restart_attempts if max_restart_attempts is not None 
            else self.config.get('process_management.health_monitor.max_restart_attempts', 3)
        )
        self.heartbeat_timeout = (
            heartbeat_timeout if heartbeat_timeout is not None 
            else self.config.get('process_management.health_monitor.heartbeat_timeout_seconds', 10.0)
        )
        self.check_interval = self.config.get('process_management.health_monitor.check_interval_seconds', 1.0)
        self.thread_timeout = self.config.get('process_management.health_monitor.thread_timeout_seconds', 2.0)
        self.history_limit = self.config.get('process_management.health_monitor.history_limit', 100)
        
        # Process tracking
        self.monitored_processes: Dict[int, ProcessHealthInfo] = {}
        self.process_factories: Dict[int, Callable] = {}  # Functions to recreate processes
        
        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Status callbacks
        self.status_callbacks: List[Callable[[Dict[int, ProcessHealthInfo]], None]] = []
        
        # Health check history
        self.health_history: Dict[int, List[Tuple[float, ProcessState]]] = {}
        
        logger.info("Process health monitor initialized")
    
    def register_process(self, process: mp.Process, process_name: str, 
                        factory_function: Callable = None) -> int:
        """
        Register a process for health monitoring.
        
        Args:
            process: The multiprocessing.Process to monitor
            process_name: Human-readable name for the process
            factory_function: Function to recreate the process if it fails
            
        Returns:
            int: Process ID for tracking
        """
        process_id = process.pid if process.pid else id(process)
        
        health_info = ProcessHealthInfo(
            process_id=process_id,
            process_name=process_name,
            state=ProcessState.INITIALIZING,
            last_heartbeat=time.time()
        )
        
        self.monitored_processes[process_id] = health_info
        
        if factory_function:
            self.process_factories[process_id] = factory_function
        
        # Initialize health history
        self.health_history[process_id] = [(time.time(), ProcessState.INITIALIZING)]
        
        logger.info(f"Registered process for monitoring: {process_name} (PID: {process_id})")
        
        # Start monitoring if this is the first process
        if not self.monitoring_active:
            self.start_monitoring()
        
        return process_id
    
    def update_process_heartbeat(self, process_id: int, additional_data: Dict[str, Any] = None):
        """Update heartbeat for a process."""
        if process_id in self.monitored_processes:
            health_info = self.monitored_processes[process_id]
            health_info.last_heartbeat = time.time()
            
            # Update state to healthy if it was initializing
            if health_info.state == ProcessState.INITIALIZING:
                health_info.state = ProcessState.HEALTHY
                self._add_health_event(process_id, ProcessState.HEALTHY)
                logger.info(f"Process {health_info.process_name} transitioned to HEALTHY state")
            
            # Update additional data if provided
            if additional_data:
                if 'memory_mb' in additional_data:
                    health_info.memory_usage_mb = additional_data['memory_mb']
                if 'cpu_percent' in additional_data:
                    health_info.cpu_usage_percent = additional_data['cpu_percent']
    
    def report_process_error(self, process_id: int, error_message: str, is_fatal: bool = False):
        """Report an error for a process."""
        if process_id in self.monitored_processes:
            health_info = self.monitored_processes[process_id]
            health_info.error_count += 1
            health_info.consecutive_failures += 1
            health_info.last_error_message = error_message
            
            if is_fatal:
                health_info.state = ProcessState.FAILED
                self._add_health_event(process_id, ProcessState.FAILED)
                logger.error(f"FATAL ERROR in {health_info.process_name}: {error_message}")
                
                # Trigger automatic restart if enabled
                self._attempt_process_restart(process_id)
            else:
                health_info.state = ProcessState.DEGRADED
                self._add_health_event(process_id, ProcessState.DEGRADED)
                logger.warning(f"ERROR in {health_info.process_name}: {error_message}")
    
    def start_monitoring(self):
        """Start the health monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop the health monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=self.thread_timeout)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_all_processes()
                time.sleep(self.check_interval)  # Check based on config
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                traceback.print_exc()
    
    def _check_all_processes(self):
        """Check health of all monitored processes."""
        current_time = time.time()
        
        for process_id, health_info in self.monitored_processes.items():
            try:
                # Check if process is still alive using psutil
                if psutil.pid_exists(process_id):
                    process = psutil.Process(process_id)
                    
                    # Update resource usage
                    try:
                        health_info.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                        health_info.cpu_usage_percent = process.cpu_percent()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Process may have died between checks
                        pass
                    
                    # Check responsiveness based on heartbeat
                    if current_time - health_info.last_heartbeat > self.heartbeat_timeout:
                        if health_info.state != ProcessState.DEGRADED:
                            health_info.state = ProcessState.DEGRADED
                            self._add_health_event(process_id, ProcessState.DEGRADED)
                            logger.warning(f"Process {health_info.process_name} not responding (no heartbeat for {current_time - health_info.last_heartbeat:.1f}s)")
                    
                else:
                    # Process is dead
                    if health_info.state != ProcessState.TERMINATED:
                        health_info.state = ProcessState.TERMINATED
                        self._add_health_event(process_id, ProcessState.TERMINATED)
                        logger.error(f"Process {health_info.process_name} has terminated unexpectedly")
                        
                        # Attempt restart
                        self._attempt_process_restart(process_id)
                        
            except Exception as e:
                logger.error(f"Error checking process {process_id}: {e}")
        
        # Notify status callbacks
        self._notify_status_callbacks()
    
    def _attempt_process_restart(self, process_id: int):
        """Attempt to restart a failed process."""
        health_info = self.monitored_processes[process_id]
        
        if health_info.restart_count >= self.max_restart_attempts:
            logger.error(f"Max restart attempts ({self.max_restart_attempts}) reached for {health_info.process_name}")
            return False
        
        if process_id not in self.process_factories:
            logger.error(f"No factory function available for restarting {health_info.process_name}")
            return False
        
        try:
            logger.info(f"Attempting to restart process {health_info.process_name} (attempt {health_info.restart_count + 1})")
            
            # Call factory function to create new process
            factory_function = self.process_factories[process_id]
            new_process = factory_function()
            
            if new_process and hasattr(new_process, 'start'):
                new_process.start()
                
                # Update tracking info
                health_info.restart_count += 1
                health_info.consecutive_failures = 0
                health_info.state = ProcessState.INITIALIZING
                health_info.startup_time = time.time()
                health_info.last_heartbeat = time.time()
                
                # Update process ID if it changed
                if new_process.pid:
                    new_process_id = new_process.pid
                    if new_process_id != process_id:
                        # Move health info to new process ID
                        self.monitored_processes[new_process_id] = health_info
                        self.process_factories[new_process_id] = self.process_factories[process_id]
                        self.health_history[new_process_id] = self.health_history[process_id]
                        
                        # Remove old tracking
                        del self.monitored_processes[process_id]
                        del self.process_factories[process_id]
                        del self.health_history[process_id]
                        
                        health_info.process_id = new_process_id
                
                self._add_health_event(health_info.process_id, ProcessState.INITIALIZING)
                logger.info(f"Process {health_info.process_name} restart initiated successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to restart process {health_info.process_name}: {e}")
            traceback.print_exc()
        
        return False
    
    def _add_health_event(self, process_id: int, state: ProcessState):
        """Add a health event to the history."""
        if process_id not in self.health_history:
            self.health_history[process_id] = []
        
        self.health_history[process_id].append((time.time(), state))
        
        # Keep only recent history based on config
        if len(self.health_history[process_id]) > self.history_limit:
            self.health_history[process_id] = self.health_history[process_id][-self.history_limit:]
    
    def _notify_status_callbacks(self):
        """Notify all registered status callbacks."""
        for callback in self.status_callbacks:
            try:
                callback(self.monitored_processes.copy())
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    def add_status_callback(self, callback: Callable[[Dict[int, ProcessHealthInfo]], None]):
        """Add a callback to receive status updates."""
        self.status_callbacks.append(callback)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of all process health."""
        summary = {
            'total_processes': len(self.monitored_processes),
            'healthy_processes': 0,
            'degraded_processes': 0,
            'failed_processes': 0,
            'total_restarts': 0,
            'processes': {}
        }
        
        for process_id, health_info in self.monitored_processes.items():
            if health_info.state == ProcessState.HEALTHY:
                summary['healthy_processes'] += 1
            elif health_info.state == ProcessState.DEGRADED:
                summary['degraded_processes'] += 1
            elif health_info.state in [ProcessState.FAILED, ProcessState.TERMINATED]:
                summary['failed_processes'] += 1
            
            summary['total_restarts'] += health_info.restart_count
            
            summary['processes'][process_id] = {
                'name': health_info.process_name,
                'state': health_info.state.value,
                'uptime': health_info.uptime_seconds,
                'memory_mb': health_info.memory_usage_mb,
                'cpu_percent': health_info.cpu_usage_percent,
                'error_count': health_info.error_count,
                'restart_count': health_info.restart_count,
                'last_error': health_info.last_error_message,
                'responsive': health_info.is_responsive(self.heartbeat_timeout)
            }
        
        return summary
    
    def get_process_diagnostics(self, process_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed diagnostics for a specific process."""
        if process_id not in self.monitored_processes:
            return None
        
        health_info = self.monitored_processes[process_id]
        history = self.health_history.get(process_id, [])
        
        # Calculate stability metrics
        state_changes = len(history)
        recent_failures = sum(1 for _, state in history[-10:] if state in [ProcessState.FAILED, ProcessState.DEGRADED])
        
        diagnostics = {
            'process_info': {
                'id': process_id,
                'name': health_info.process_name,
                'current_state': health_info.state.value,
                'uptime_seconds': health_info.uptime_seconds,
                'responsive': health_info.is_responsive(self.heartbeat_timeout)
            },
            'performance': {
                'memory_usage_mb': health_info.memory_usage_mb,
                'cpu_usage_percent': health_info.cpu_usage_percent,
                'last_heartbeat_age': time.time() - health_info.last_heartbeat
            },
            'reliability': {
                'error_count': health_info.error_count,
                'restart_count': health_info.restart_count,
                'consecutive_failures': health_info.consecutive_failures,
                'state_changes': state_changes,
                'recent_failures': recent_failures,
                'last_error': health_info.last_error_message
            },
            'history': [
                {'timestamp': ts, 'state': state.value}
                for ts, state in history[-20:]  # Last 20 events
            ]
        }
        
        return diagnostics

# Example usage for process health monitoring integration
class ProcessHealthGUI:
    """Simple GUI component for displaying process health."""
    
    def __init__(self, health_monitor: ProcessHealthMonitor):
        self.health_monitor = health_monitor
        self.health_monitor.add_status_callback(self.on_health_update)
    
    def on_health_update(self, processes: Dict[int, ProcessHealthInfo]):
        """Handle health updates from the monitor."""
        # This would update GUI indicators
        for process_id, health_info in processes.items():
            status_text = f"{health_info.process_name}: {health_info.state.value}"
            if health_info.state == ProcessState.HEALTHY:
                status_color = "green"
            elif health_info.state == ProcessState.DEGRADED:
                status_color = "yellow"  
            else:
                status_color = "red"
            
            # Update GUI here (would be actual GUI elements in real implementation)
            print(f"[GUI] {status_text} ({status_color})")

# Test and example usage
if __name__ == "__main__":
    # Create health monitor
    monitor = ProcessHealthMonitor()
    
    # Example: Mock process for testing
    import multiprocessing as mp
    
    def mock_process_function():
        """Mock process that runs for testing."""
        import time
        while True:
            print("Mock process running...")
            time.sleep(1)
    
    def create_mock_process():
        """Factory function to create mock process."""
        return mp.Process(target=mock_process_function)
    
    # Create and register a test process
    test_process = create_mock_process()
    test_process.start()
    
    process_id = monitor.register_process(
        test_process, 
        "Test Process", 
        create_mock_process
    )
    
    # Simulate some activity
    time.sleep(2)
    monitor.update_process_heartbeat(process_id)
    
    # Get health summary
    summary = monitor.get_health_summary()
    print(f"Health Summary: {summary}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    test_process.terminate()