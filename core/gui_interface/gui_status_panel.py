"""
GUI Status Panel for Process Health Monitoring
Provides visual indicators and controls for process health and recovery.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import queue

from core.process_management.process_health_monitor import ProcessHealthMonitor, ProcessHealthInfo, ProcessState
# from tools.startup_diagnostics import StartupDiagnostics  # Module not found

class StatusIndicatorColor(Enum):
    HEALTHY = "#4CAF50"      # Green
    DEGRADED = "#FF9800"     # Orange
    FAILED = "#F44336"       # Red
    INITIALIZING = "#2196F3" # Blue
    TERMINATED = "#757575"   # Gray

class ProcessStatusWidget(tk.Frame):
    """Widget to display status of a single process."""
    
    def __init__(self, parent, process_name: str, process_id: int):
        super().__init__(parent, relief=tk.RAISED, bd=1)
        
        self.process_name = process_name
        self.process_id = process_id
        self.current_state = ProcessState.INITIALIZING
        
        # Layout
        self.grid_columnconfigure(1, weight=1)
        
        # Status indicator (colored circle)
        self.status_canvas = tk.Canvas(self, width=16, height=16, highlightthickness=0)
        self.status_canvas.grid(row=0, column=0, padx=(5, 10), pady=5)
        self.status_circle = self.status_canvas.create_oval(2, 2, 14, 14, 
                                                           fill=StatusIndicatorColor.INITIALIZING.value,
                                                           outline="")
        
        # Process info
        info_frame = tk.Frame(self)
        info_frame.grid(row=0, column=1, sticky="w", padx=5)
        
        self.name_label = tk.Label(info_frame, text=process_name, font=("Arial", 9, "bold"))
        self.name_label.pack(anchor="w")
        
        self.status_label = tk.Label(info_frame, text="Initializing...", font=("Arial", 8))
        self.status_label.pack(anchor="w")
        
        # Action buttons
        button_frame = tk.Frame(self)
        button_frame.grid(row=0, column=2, padx=5)
        
        self.restart_button = tk.Button(button_frame, text="Restart", font=("Arial", 8),
                                       command=self._on_restart_clicked, state=tk.DISABLED)
        self.restart_button.pack(side="left", padx=2)
        
        self.details_button = tk.Button(button_frame, text="Details", font=("Arial", 8),
                                       command=self._on_details_clicked)
        self.details_button.pack(side="left", padx=2)
        
        # Callbacks
        self.restart_callback = None
        self.details_callback = None
    
    def update_status(self, health_info: ProcessHealthInfo):
        """Update the status display based on health info."""
        self.current_state = health_info.state
        
        # Update status indicator color
        color_map = {
            ProcessState.HEALTHY: StatusIndicatorColor.HEALTHY,
            ProcessState.DEGRADED: StatusIndicatorColor.DEGRADED,
            ProcessState.FAILED: StatusIndicatorColor.FAILED,
            ProcessState.INITIALIZING: StatusIndicatorColor.INITIALIZING,
            ProcessState.TERMINATED: StatusIndicatorColor.TERMINATED
        }
        
        color = color_map.get(health_info.state, StatusIndicatorColor.TERMINATED)
        self.status_canvas.itemconfig(self.status_circle, fill=color.value)
        
        # Update status text
        status_text = health_info.state.value.title()
        if health_info.state == ProcessState.HEALTHY:
            status_text += f" (↑{health_info.uptime_seconds:.0f}s)"
        elif health_info.state in [ProcessState.FAILED, ProcessState.TERMINATED]:
            status_text += f" (↻{health_info.restart_count})"
        elif health_info.state == ProcessState.DEGRADED:
            status_text += f" ({health_info.error_count} errors)"
        
        self.status_label.config(text=status_text)
        
        # Update button states
        can_restart = health_info.state in [ProcessState.FAILED, ProcessState.TERMINATED]
        self.restart_button.config(state=tk.NORMAL if can_restart else tk.DISABLED)
    
    def set_restart_callback(self, callback: Callable[[int], None]):
        """Set callback for restart button."""
        self.restart_callback = callback
    
    def set_details_callback(self, callback: Callable[[int], None]):
        """Set callback for details button."""
        self.details_callback = callback
    
    def _on_restart_clicked(self):
        """Handle restart button click."""
        if self.restart_callback:
            self.restart_callback(self.process_id)
    
    def _on_details_clicked(self):
        """Handle details button click."""
        if self.details_callback:
            self.details_callback(self.process_id)

class ProcessDetailsDialog(tk.Toplevel):
    """Dialog showing detailed process information."""
    
    def __init__(self, parent, health_monitor: ProcessHealthMonitor, process_id: int):
        super().__init__(parent)
        
        self.health_monitor = health_monitor
        self.process_id = process_id
        
        self.title(f"Process Details - ID {process_id}")
        self.geometry("600x500")
        self.resizable(True, True)
        
        # Create UI
        self._create_widgets()
        self._update_display()
        
        # Make modal
        self.transient(parent)
        self.grab_set()
    
    def _create_widgets(self):
        """Create the dialog widgets."""
        # Main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Details text area
        self.details_text = scrolledtext.ScrolledText(main_frame, height=25, width=70)
        self.details_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Refresh", command=self._update_display).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Close", command=self.destroy).pack(side=tk.RIGHT)
    
    def _update_display(self):
        """Update the details display."""
        diagnostics = self.health_monitor.get_process_diagnostics(self.process_id)
        
        self.details_text.delete(1.0, tk.END)
        
        if diagnostics:
            # Format the diagnostics information
            details = []
            
            details.append("=" * 60)
            details.append("PROCESS DIAGNOSTICS REPORT")
            details.append("=" * 60)
            details.append("")
            
            # Basic info
            proc_info = diagnostics['process_info']
            details.append("Process Information:")
            details.append(f"  Name: {proc_info['name']}")
            details.append(f"  ID: {proc_info['id']}")
            details.append(f"  State: {proc_info['current_state'].upper()}")
            details.append(f"  Uptime: {proc_info['uptime_seconds']:.1f} seconds")
            details.append(f"  Responsive: {'Yes' if proc_info['responsive'] else 'No'}")
            details.append("")
            
            # Performance metrics
            perf = diagnostics['performance']
            details.append("Performance Metrics:")
            details.append(f"  Memory Usage: {perf['memory_usage_mb']:.1f} MB")
            details.append(f"  CPU Usage: {perf['cpu_usage_percent']:.1f}%")
            details.append(f"  Last Heartbeat: {perf['last_heartbeat_age']:.1f} seconds ago")
            details.append("")
            
            # Reliability metrics
            reliability = diagnostics['reliability']
            details.append("Reliability Metrics:")
            details.append(f"  Total Errors: {reliability['error_count']}")
            details.append(f"  Restart Count: {reliability['restart_count']}")
            details.append(f"  Consecutive Failures: {reliability['consecutive_failures']}")
            details.append(f"  State Changes: {reliability['state_changes']}")
            details.append(f"  Recent Failures: {reliability['recent_failures']}")
            
            if reliability['last_error']:
                details.append(f"  Last Error: {reliability['last_error']}")
            details.append("")
            
            # History
            history = diagnostics['history']
            if history:
                details.append("Recent State History:")
                for event in history[-10:]:  # Last 10 events
                    timestamp = time.strftime('%H:%M:%S', time.localtime(event['timestamp']))
                    details.append(f"  {timestamp}: {event['state'].upper()}")
                details.append("")
            
            details.append("=" * 60)
            
            self.details_text.insert(tk.END, "\n".join(details))
        else:
            self.details_text.insert(tk.END, "No diagnostics available for this process.")

class SystemStatusPanel(tk.Frame):
    """Main system status panel for the GUI."""
    
    def __init__(self, parent, health_monitor: ProcessHealthMonitor, config=None):
        super().__init__(parent, relief=tk.SUNKEN, bd=1)
        
        self.health_monitor = health_monitor
        self.process_widgets: Dict[int, ProcessStatusWidget] = {}
        self.startup_diagnostics = StartupDiagnostics()
        
        # Load configuration
        self.config = config or {}
        self.panel_config = self.config.get('gui_interface', {}).get('status_panel', {})
        
        # Add status callback to health monitor
        self.health_monitor.add_status_callback(self._on_health_update)
        
        self._create_widgets()
        self._start_update_thread()
    
    def _create_widgets(self):
        """Create the status panel widgets."""
        # Header
        header_frame = tk.Frame(self, bg="#f0f0f0")
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        title_label = tk.Label(header_frame, text="System Status", 
                              font=("Arial", 12, "bold"), bg="#f0f0f0")
        title_label.pack(side=tk.LEFT)
        
        # System actions
        self.diagnose_button = tk.Button(header_frame, text="Run Diagnostics", 
                                        command=self._run_diagnostics)
        self.diagnose_button.pack(side=tk.RIGHT, padx=5)
        
        self.restart_all_button = tk.Button(header_frame, text="Restart All", 
                                           command=self._restart_all_processes)
        self.restart_all_button.pack(side=tk.RIGHT, padx=5)
        
        # Status summary
        self.summary_label = tk.Label(self, text="No processes monitored", 
                                     font=("Arial", 9), fg="#666")
        self.summary_label.pack(pady=5)
        
        # Scrollable process list
        panel_height = self.panel_config.get('panel_height', 200)
        self.canvas = tk.Canvas(self, height=panel_height)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.scrollbar.pack(side="right", fill="y", pady=5)
    
    def _on_health_update(self, processes: Dict[int, ProcessHealthInfo]):
        """Handle health updates from the monitor."""
        # Update existing widgets
        for process_id, health_info in processes.items():
            if process_id in self.process_widgets:
                self.process_widgets[process_id].update_status(health_info)
            else:
                # Create new widget for this process
                self._add_process_widget(process_id, health_info)
        
        # Remove widgets for processes that no longer exist
        to_remove = []
        for process_id in self.process_widgets:
            if process_id not in processes:
                to_remove.append(process_id)
        
        for process_id in to_remove:
            self._remove_process_widget(process_id)
        
        # Update summary
        self._update_summary(processes)
    
    def _add_process_widget(self, process_id: int, health_info: ProcessHealthInfo):
        """Add a new process status widget."""
        widget = ProcessStatusWidget(self.scrollable_frame, health_info.process_name, process_id)
        widget.pack(fill=tk.X, padx=5, pady=2)
        
        widget.set_restart_callback(self._restart_process)
        widget.set_details_callback(self._show_process_details)
        
        widget.update_status(health_info)
        self.process_widgets[process_id] = widget
    
    def _remove_process_widget(self, process_id: int):
        """Remove a process status widget."""
        if process_id in self.process_widgets:
            widget = self.process_widgets[process_id]
            widget.destroy()
            del self.process_widgets[process_id]
    
    def _update_summary(self, processes: Dict[int, ProcessHealthInfo]):
        """Update the status summary."""
        if not processes:
            self.summary_label.config(text="No processes monitored", fg="#666")
            return
        
        healthy = sum(1 for p in processes.values() if p.state == ProcessState.HEALTHY)
        degraded = sum(1 for p in processes.values() if p.state == ProcessState.DEGRADED)
        failed = sum(1 for p in processes.values() if p.state in [ProcessState.FAILED, ProcessState.TERMINATED])
        total = len(processes)
        
        if failed > 0:
            color = "#F44336"  # Red
            status = f"⚠ {failed} processes failed"
        elif degraded > 0:
            color = "#FF9800"  # Orange
            status = f"⚠ {degraded} processes degraded"
        else:
            color = "#4CAF50"  # Green
            status = "✓ All processes healthy"
        
        summary_text = f"{status} | {healthy}/{total} healthy"
        self.summary_label.config(text=summary_text, fg=color)
    
    def _restart_process(self, process_id: int):
        """Restart a specific process."""
        # This would be implemented to restart the specific process
        # For now, just show a message
        messagebox.showinfo("Restart Process", 
                           f"Restart requested for process {process_id}\n"
                           f"This feature would restart the specific process.")
    
    def _restart_all_processes(self):
        """Restart all failed processes."""
        messagebox.showinfo("Restart All", 
                           "This would restart all failed processes.\n"
                           "Feature to be implemented.")
    
    def _show_process_details(self, process_id: int):
        """Show detailed process information."""
        ProcessDetailsDialog(self, self.health_monitor, process_id)
    
    def _run_diagnostics(self):
        """Run comprehensive system diagnostics."""
        DiagnosticsDialog(self, self.startup_diagnostics)
    
    def _start_update_thread(self):
        """Start the UI update thread."""
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def _update_loop(self):
        """Main update loop for the status panel."""
        update_interval = self.panel_config.get('update_interval', 1.0)
        while True:
            try:
                # The health monitor callbacks handle most updates,
                # this thread can handle periodic tasks if needed
                time.sleep(update_interval)
            except Exception as e:
                print(f"Error in status panel update loop: {e}")

class DiagnosticsDialog(tk.Toplevel):
    """Dialog for running and displaying system diagnostics."""
    
    def __init__(self, parent, startup_diagnostics: StartupDiagnostics):
        super().__init__(parent)
        
        self.startup_diagnostics = startup_diagnostics
        
        self.title("System Diagnostics")
        self.geometry("800x600")
        self.resizable(True, True)
        
        self._create_widgets()
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        # Run diagnostics automatically
        self.after(100, self._run_diagnostics)
    
    def _create_widgets(self):
        """Create the dialog widgets."""
        # Main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Running diagnostics...", 
                                    font=("Arial", 10))
        self.status_label.pack(pady=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 10))
        self.progress.start()
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(main_frame, height=30, width=90)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        self.run_again_button = ttk.Button(button_frame, text="Run Again", 
                                          command=self._run_diagnostics, state=tk.DISABLED)
        self.run_again_button.pack(side=tk.LEFT)
        
        ttk.Button(button_frame, text="Close", command=self.destroy).pack(side=tk.RIGHT)
    
    def _run_diagnostics(self):
        """Run the diagnostic analysis."""
        def run_in_thread():
            try:
                # Update status
                self.after(0, lambda: self.status_label.config(text="Running diagnostics..."))
                self.after(0, lambda: self.run_again_button.config(state=tk.DISABLED))
                
                # Import configuration (would normally get from main app)
                from confighandler import ConfigHandler
                config_handler = ConfigHandler()
                config = config_handler.config
                
                # Run diagnostics for camera 0 as example
                diagnosis = self.startup_diagnostics.diagnose_camera_startup_failure(
                    camera_index=0,
                    config=config,
                    error_message=None,
                    timeout_occurred=False
                )
                
                # Create report
                report = self.startup_diagnostics.create_diagnostic_report(diagnosis)
                
                # Update UI in main thread
                self.after(0, lambda: self._display_results(report))
                
            except Exception as e:
                error_msg = f"Error running diagnostics: {e}\n\n"
                error_msg += "This may indicate a configuration or system issue."
                self.after(0, lambda: self._display_results(error_msg))
        
        # Run in separate thread to avoid blocking UI
        threading.Thread(target=run_in_thread, daemon=True).start()
    
    def _display_results(self, report: str):
        """Display the diagnostic results."""
        self.progress.stop()
        self.status_label.config(text="Diagnostics complete")
        self.run_again_button.config(state=tk.NORMAL)
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, report)

# Example usage and testing
if __name__ == "__main__":
    # Create test GUI
    root = tk.Tk()
    root.title("Process Status Panel Test")
    root.geometry("800x600")
    
    # Create health monitor
    health_monitor = ProcessHealthMonitor()
    
    # Create status panel
    status_panel = SystemStatusPanel(root, health_monitor)
    status_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Add some test processes
    import multiprocessing as mp
    
    def mock_process():
        import time
        while True:
            time.sleep(1)
    
    # Create and register mock processes
    for i in range(3):
        process = mp.Process(target=mock_process)
        process.start()
        health_monitor.register_process(process, f"Test Process {i+1}")
        
        # Simulate some health updates
        if i == 1:
            health_monitor.report_process_error(process.pid, "Test error", is_fatal=False)
    
    root.mainloop()