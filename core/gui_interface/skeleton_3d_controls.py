"""
3D Skeleton View Control Panel

A compact, reusable Tkinter component for interactive 3D skeleton view manipulation.
Provides arrow buttons with press-and-hold functionality, preset views, and reset controls.

Author: Claude Code
Date: 2025-11-08
"""

import tkinter as tk
from tkinter import ttk
import logging

logger = logging.getLogger(__name__)


class Skeleton3DControlPanel(ttk.Frame):
    """
    Compact toolbar for 3D skeleton view manipulation.

    Features:
    - 4 directional arrow buttons (up/down for elevation, left/right for azimuth)
    - Press-and-hold for continuous rotation (30°/sec)
    - Single click for precise 5° steps
    - Preset view buttons (Front, Side, Top, Isometric)
    - Reset button to restore default view

    Usage:
        control_panel = Skeleton3DControlPanel(parent_frame, config)
        control_panel.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)

        # In render loop:
        elevation, azimuth = control_panel.get_view_angle()
    """

    # Rotation behavior constants
    SINGLE_CLICK_STEP = 5  # Degrees per single click
    CONTINUOUS_ROTATION_SPEED = 30  # Degrees per second when holding
    HOLD_THRESHOLD_MS = 300  # Milliseconds before continuous rotation starts
    ROTATION_UPDATE_MS = 50  # Update interval for continuous rotation (20 FPS)

    # Preset view angles (elevation, azimuth)
    PRESET_VIEWS = {
        'front': (0, 0),
        'side': (0, 90),
        'top': (90, 45),
        'iso': (20, 45)  # Default isometric view
    }

    def __init__(self, parent, config):
        """
        Initialize the 3D skeleton control panel.

        Args:
            parent: Parent Tkinter widget (typically skeleton_3d_frame)
            config: YouQuantiPy configuration dict
        """
        super().__init__(parent, style='Card.TFrame')

        self.config = config

        # Extract default angles from config
        default_elevation = config.get('visualization', {}).get('skeleton_3d', {}).get(
            'view_angle', {}
        ).get('elevation', 20)
        default_azimuth = config.get('visualization', {}).get('skeleton_3d', {}).get(
            'view_angle', {}
        ).get('azimuth', 45)

        # State variables for current view angles
        self.elevation = tk.IntVar(value=default_elevation)
        self.azimuth = tk.IntVar(value=default_azimuth)

        # State for press-and-hold functionality
        self.rotation_timer = None  # Timer ID for continuous rotation
        self.rotation_direction = None  # Current rotation direction
        self.hold_timer = None  # Timer ID for detecting long press

        # Build the UI components
        self._create_widgets()
        self._create_layout()

        logger.info(f"[3D CONTROLS] Initialized with elevation={default_elevation}°, azimuth={default_azimuth}°")

    def _create_widgets(self):
        """Create all UI widgets for the control panel."""

        # --- Arrow Buttons ---
        self.arrow_frame = ttk.Frame(self)

        # Up arrow (increase elevation)
        self.btn_up = ttk.Button(
            self.arrow_frame,
            text="⬆",
            width=3,
            command=lambda: self._single_step('up')
        )
        self.btn_up.bind('<ButtonPress-1>', lambda e: self._on_arrow_press('up'))
        self.btn_up.bind('<ButtonRelease-1>', lambda e: self._on_arrow_release())

        # Down arrow (decrease elevation)
        self.btn_down = ttk.Button(
            self.arrow_frame,
            text="⬇",
            width=3,
            command=lambda: self._single_step('down')
        )
        self.btn_down.bind('<ButtonPress-1>', lambda e: self._on_arrow_press('down'))
        self.btn_down.bind('<ButtonRelease-1>', lambda e: self._on_arrow_release())

        # Left arrow (decrease azimuth - rotate left)
        self.btn_left = ttk.Button(
            self.arrow_frame,
            text="⬅",
            width=3,
            command=lambda: self._single_step('left')
        )
        self.btn_left.bind('<ButtonPress-1>', lambda e: self._on_arrow_press('left'))
        self.btn_left.bind('<ButtonRelease-1>', lambda e: self._on_arrow_release())

        # Right arrow (increase azimuth - rotate right)
        self.btn_right = ttk.Button(
            self.arrow_frame,
            text="➡",
            width=3,
            command=lambda: self._single_step('right')
        )
        self.btn_right.bind('<ButtonPress-1>', lambda e: self._on_arrow_press('right'))
        self.btn_right.bind('<ButtonRelease-1>', lambda e: self._on_arrow_release())

        # --- Preset View Buttons ---
        self.preset_frame = ttk.Frame(self)

        self.btn_front = ttk.Button(
            self.preset_frame,
            text="Front",
            width=6,
            command=lambda: self._set_preset_view('front')
        )

        self.btn_side = ttk.Button(
            self.preset_frame,
            text="Side",
            width=6,
            command=lambda: self._set_preset_view('side')
        )

        self.btn_top = ttk.Button(
            self.preset_frame,
            text="Top",
            width=6,
            command=lambda: self._set_preset_view('top')
        )

        self.btn_iso = ttk.Button(
            self.preset_frame,
            text="Iso",
            width=6,
            command=lambda: self._set_preset_view('iso')
        )

        # --- Reset Button ---
        self.btn_reset = ttk.Button(
            self,
            text="🔄 Reset",
            command=self._reset_view
        )

    def _create_layout(self):
        """Arrange widgets in a compact layout."""

        # Arrow buttons in cross/diamond pattern (3x3 grid)
        self.arrow_frame.grid(row=0, column=0, padx=5, pady=5)

        # Row 0: Up arrow (center column)
        self.btn_up.grid(row=0, column=1, padx=2, pady=2)

        # Row 1: Left and Right arrows
        self.btn_left.grid(row=1, column=0, padx=2, pady=2)
        self.btn_right.grid(row=1, column=2, padx=2, pady=2)

        # Row 2: Down arrow (center column)
        self.btn_down.grid(row=2, column=1, padx=2, pady=2)

        # Preset buttons in horizontal row
        self.preset_frame.grid(row=1, column=0, padx=5, pady=2)
        self.btn_front.grid(row=0, column=0, padx=1)
        self.btn_side.grid(row=0, column=1, padx=1)
        self.btn_top.grid(row=0, column=2, padx=1)
        self.btn_iso.grid(row=0, column=3, padx=1)

        # Reset button at bottom
        self.btn_reset.grid(row=2, column=0, padx=5, pady=(2, 5))

    # --- Public Interface ---

    def get_view_angle(self):
        """
        Get the current 3D view angle.

        Returns:
            tuple: (elevation, azimuth) in degrees
        """
        return (self.elevation.get(), self.azimuth.get())

    # --- Button Event Handlers ---

    def _single_step(self, direction):
        """
        Execute a single rotation step.

        This is called by the button command (single click without hold).

        Args:
            direction: One of 'up', 'down', 'left', 'right'
        """
        # Note: This is redundant with the command binding, but kept for clarity
        # The actual single-step happens when the user releases quickly
        pass

    def _on_arrow_press(self, direction):
        """
        Handle arrow button press (mouse down).

        Initiates a single step immediately, then starts a timer to detect
        if this is a long press (press-and-hold).

        Args:
            direction: One of 'up', 'down', 'left', 'right'
        """
        # Cancel any existing timers
        self._cancel_timers()

        # Execute immediate single step
        self._rotate_step(direction, self.SINGLE_CLICK_STEP)

        # Store the direction for continuous rotation
        self.rotation_direction = direction

        # Start timer to detect long press
        self.hold_timer = self.after(
            self.HOLD_THRESHOLD_MS,
            self._start_continuous_rotation
        )

    def _on_arrow_release(self):
        """
        Handle arrow button release (mouse up).

        Stops any ongoing continuous rotation.
        """
        self._cancel_timers()
        self.rotation_direction = None

    def _start_continuous_rotation(self):
        """
        Start continuous rotation after hold threshold is reached.

        This is called by the hold_timer after HOLD_THRESHOLD_MS.
        """
        logger.debug(f"[3D CONTROLS] Starting continuous rotation: {self.rotation_direction}")
        self._continuous_rotate()

    def _continuous_rotate(self):
        """
        Continuous rotation timer callback.

        Rotates the view smoothly at CONTINUOUS_ROTATION_SPEED degrees per second.
        Reschedules itself until the button is released.
        """
        if self.rotation_direction is None:
            return

        # Calculate step size based on rotation speed and update interval
        step = (self.CONTINUOUS_ROTATION_SPEED * self.ROTATION_UPDATE_MS) / 1000.0

        # Apply rotation
        self._rotate_step(self.rotation_direction, step)

        # Schedule next rotation update
        self.rotation_timer = self.after(
            self.ROTATION_UPDATE_MS,
            self._continuous_rotate
        )

    def _cancel_timers(self):
        """Cancel all active timers."""
        if self.hold_timer is not None:
            self.after_cancel(self.hold_timer)
            self.hold_timer = None

        if self.rotation_timer is not None:
            self.after_cancel(self.rotation_timer)
            self.rotation_timer = None

    # --- Angle Adjustment Methods ---

    def _rotate_step(self, direction, step_size):
        """
        Apply a rotation step in the specified direction.

        Args:
            direction: One of 'up', 'down', 'left', 'right'
            step_size: Rotation amount in degrees
        """
        if direction == 'up':
            self._adjust_elevation(step_size)
        elif direction == 'down':
            self._adjust_elevation(-step_size)
        elif direction == 'left':
            self._adjust_azimuth(-step_size)
        elif direction == 'right':
            self._adjust_azimuth(step_size)

    def _adjust_elevation(self, delta):
        """
        Adjust elevation angle with clamping.

        Elevation is clamped to [-90°, 90°] to prevent gimbal lock
        and maintain meaningful views.

        Args:
            delta: Change in elevation (degrees)
        """
        current = self.elevation.get()
        new_value = max(-90, min(90, current + delta))
        self.elevation.set(int(new_value))

        logger.debug(f"[3D CONTROLS] Elevation: {int(new_value)}°")

    def _adjust_azimuth(self, delta):
        """
        Adjust azimuth angle with wrapping.

        Azimuth wraps around [0°, 360°] for seamless rotation.

        Args:
            delta: Change in azimuth (degrees)
        """
        current = self.azimuth.get()
        new_value = (current + delta) % 360
        self.azimuth.set(int(new_value))

        logger.debug(f"[3D CONTROLS] Azimuth: {int(new_value)}°")

    def _set_preset_view(self, preset_name):
        """
        Jump to a preset view angle.

        Args:
            preset_name: One of 'front', 'side', 'top', 'iso'
        """
        if preset_name not in self.PRESET_VIEWS:
            logger.warning(f"[3D CONTROLS] Unknown preset view: {preset_name}")
            return

        elev, azim = self.PRESET_VIEWS[preset_name]
        self.elevation.set(elev)
        self.azimuth.set(azim)

        logger.info(f"[3D CONTROLS] Preset view '{preset_name}': elevation={elev}°, azimuth={azim}°")

    def _reset_view(self):
        """Reset view to default angles from config."""
        default_elev = self.config.get('visualization', {}).get('skeleton_3d', {}).get(
            'view_angle', {}
        ).get('elevation', 20)
        default_azim = self.config.get('visualization', {}).get('skeleton_3d', {}).get(
            'view_angle', {}
        ).get('azimuth', 45)

        self.elevation.set(default_elev)
        self.azimuth.set(default_azim)

        logger.info(f"[3D CONTROLS] Reset view: elevation={default_elev}°, azimuth={default_azim}°")

    def cleanup(self):
        """Clean up resources (call when destroying the widget)."""
        self._cancel_timers()
        logger.debug("[3D CONTROLS] Cleanup complete")
