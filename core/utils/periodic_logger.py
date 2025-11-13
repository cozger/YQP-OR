#!/usr/bin/env python3
"""
Periodic Logger - Consolidates per-frame verbose logs into periodic summaries.

Instead of logging every frame (creating thousands of log lines), this utility:
1. Counts events/metrics between periodic checkpoints
2. Aggregates statistics (min/max/avg)
3. Logs single-line summaries every N events

This reduces log output by ~95% while maintaining visibility into system health.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PeriodicStats:
    """Statistics aggregated between periodic log outputs."""
    count: int = 0  # Number of events
    total_time: float = 0.0  # Total elapsed time (ms)
    min_time: float = float('inf')  # Minimum time (ms)
    max_time: float = 0.0  # Maximum time (ms)
    errors: int = 0  # Error count
    custom_metrics: Dict[str, Any] = field(default_factory=dict)  # Custom metrics

    def avg_time(self) -> float:
        """Average time in ms."""
        return self.total_time / self.count if self.count > 0 else 0

    def reset(self):
        """Reset statistics."""
        self.count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.errors = 0
        self.custom_metrics.clear()


class PeriodicLogger:
    """
    Logs aggregated metrics at periodic intervals.

    Usage:
        logger = PeriodicLogger('MediaPipeProcess', period_frames=30)
        for frame in frames:
            start = time.time()
            # ... processing ...
            elapsed_ms = (time.time() - start) * 1000
            logger.record_frame(elapsed_ms)
            logger.log_if_periodic(f"Processed {frame.shape}")
    """

    def __init__(self, component_name: str, period_frames: int = 30, logger_obj: Optional[logging.Logger] = None):
        """
        Initialize periodic logger.

        Args:
            component_name: Name of component (e.g., "MediaPipeProcess", "ArcFace")
            period_frames: Number of frames between periodic logs
            logger_obj: Logger object (default: module logger)
        """
        self.component_name = component_name
        self.period_frames = period_frames
        self.logger = logger_obj or logging.getLogger(component_name)

        # Statistics tracking
        self.stats = PeriodicStats()
        self.frame_counter = 0
        self.last_log_time = time.time()

    def record_frame(self, elapsed_ms: float = 0.0, **kwargs):
        """
        Record a processed frame with timing.

        Args:
            elapsed_ms: Processing time in milliseconds
            **kwargs: Additional metrics to track
        """
        self.frame_counter += 1
        self.stats.count += 1

        if elapsed_ms > 0:
            self.stats.total_time += elapsed_ms
            self.stats.min_time = min(self.stats.min_time, elapsed_ms)
            self.stats.max_time = max(self.stats.max_time, elapsed_ms)

        # Track custom metrics
        for key, value in kwargs.items():
            if key not in self.stats.custom_metrics:
                self.stats.custom_metrics[key] = []
            self.stats.custom_metrics[key].append(value)

    def record_error(self):
        """Record an error event."""
        self.stats.errors += 1

    def should_log(self) -> bool:
        """Check if it's time to log."""
        return self.frame_counter % self.period_frames == 0 and self.frame_counter > 0

    def get_summary(self) -> str:
        """Get formatted summary string."""
        avg_ms = self.stats.avg_time()
        summary = (
            f"[{self.component_name}] "
            f"Processed {self.stats.count} frames | "
            f"Time: {avg_ms:.2f}ms (min={self.stats.min_time:.2f}, max={self.stats.max_time:.2f}) ms"
        )

        if self.stats.errors > 0:
            summary += f" | Errors: {self.stats.errors}"

        # Add custom metrics
        for key, values in self.stats.custom_metrics.items():
            if isinstance(values[0], (int, float)):
                avg_val = sum(values) / len(values)
                summary += f" | {key}: {avg_val:.2f}"
            else:
                summary += f" | {key}: {values[-1]}"

        return summary

    def log_if_periodic(self, extra_info: str = ""):
        """Log summary if period reached."""
        if self.should_log():
            summary = self.get_summary()
            if extra_info:
                summary += f" | {extra_info}"
            self.logger.info(summary)
            self.stats.reset()

    def force_log(self, extra_info: str = ""):
        """Force immediate log regardless of period."""
        summary = self.get_summary()
        if extra_info:
            summary += f" | {extra_info}"
        self.logger.info(summary)
        self.stats.reset()


class BatchPeriodicLogger:
    """
    Tracks multiple metrics separately and logs when period reached.

    Usage:
        logger = BatchPeriodicLogger('GUIWorker', period_frames=300)
        logger.track('pose_matches', 1)  # Increment counter
        logger.track('landmark_reads', 1)
        logger.log_if_periodic()
    """

    def __init__(self, component_name: str, period_frames: int = 300, logger_obj: Optional[logging.Logger] = None):
        """Initialize batch periodic logger."""
        self.component_name = component_name
        self.period_frames = period_frames
        self.logger = logger_obj or logging.getLogger(component_name)

        self.frame_counter = 0
        self.metrics: Dict[str, int] = defaultdict(int)
        self.time_metrics: Dict[str, List[float]] = defaultdict(list)

    def track(self, metric_name: str, value: int = 1):
        """Track a metric (counter)."""
        self.metrics[metric_name] += value
        self.frame_counter += 1

    def track_time(self, metric_name: str, elapsed_ms: float):
        """Track a timing metric."""
        self.time_metrics[metric_name].append(elapsed_ms)

    def should_log(self) -> bool:
        """Check if it's time to log."""
        return self.frame_counter >= self.period_frames and self.frame_counter > 0

    def get_summary(self) -> str:
        """Get formatted summary of all metrics."""
        parts = [f"[{self.component_name}]"]

        # Add counter metrics
        for name, count in sorted(self.metrics.items()):
            parts.append(f"{name}={count}")

        # Add timing metrics
        for name, times in sorted(self.time_metrics.items()):
            if times:
                avg = sum(times) / len(times)
                parts.append(f"{name}_ms={avg:.2f}")

        return " | ".join(parts)

    def log_if_periodic(self):
        """Log summary if period reached."""
        if self.should_log():
            self.logger.info(self.get_summary())
            self.reset()

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.time_metrics.clear()
        self.frame_counter = 0


class DebugPeriodicLogger:
    """
    Periodic logger that only logs when explicitly enabled (via config flag).

    Useful for components that have heavy diagnostic logging that should be
    disabled in production but available for debugging.
    """

    def __init__(self, component_name: str, enable: bool = False, period_frames: int = 30):
        """Initialize debug logger (disabled by default)."""
        self.enabled = enable
        self.component_name = component_name
        self.logger = PeriodicLogger(component_name, period_frames)

    def record_frame(self, elapsed_ms: float = 0.0, **kwargs):
        """Record frame only if enabled."""
        if self.enabled:
            self.logger.record_frame(elapsed_ms, **kwargs)

    def log_if_periodic(self, extra_info: str = ""):
        """Log only if enabled and period reached."""
        if self.enabled:
            self.logger.log_if_periodic(extra_info)
