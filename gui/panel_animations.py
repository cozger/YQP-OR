"""
Panel animation system for collapsible GUI panels.

Provides smooth width animations for Tkinter grid-based panels with easing,
state management, and customizable collapse/expand behavior.
"""

import tkinter as tk
from typing import Optional, Callable, Dict, Any
import time


def easing_cubic_in_out(t: float) -> float:
    """
    Cubic ease-in-out easing function for smooth animations.

    Args:
        t: Progress value from 0.0 to 1.0

    Returns:
        Eased value from 0.0 to 1.0
    """
    if t < 0.5:
        return 4 * t * t * t
    else:
        p = 2 * t - 2
        return 1 + 0.5 * p * p * p


class PanelAnimationController:
    """
    Manages smooth width animations for Tkinter grid panels.

    Supports concurrent animations, easing functions, and automatic
    cleanup of completed animations.
    """

    def __init__(
        self,
        root_widget: tk.Widget,
        left_panel: Optional[tk.Widget] = None,
        right_panel: Optional[tk.Widget] = None,
        center_container: Optional[tk.Widget] = None
    ):
        """
        Initialize animation controller.

        Args:
            root_widget: Root Tk widget that contains the grid layout
            left_panel: Optional left panel widget for width tracking
            right_panel: Optional right panel widget for width tracking
            center_container: Optional center container to explicitly resize
        """
        self.root = root_widget
        self.left_panel = left_panel
        self.right_panel = right_panel
        self.center_container = center_container
        self.active_animations: Dict[str, Dict[str, Any]] = {}

    def animate_width(
        self,
        widget: tk.Widget,
        column_idx: int,
        target_width: int,
        duration_ms: int = 250,
        on_complete: Optional[Callable[[], None]] = None,
        animation_id: Optional[str] = None
    ) -> str:
        """
        Animate a grid column's width from current to target value.

        Args:
            widget: The widget whose grid column to animate
            column_idx: Grid column index to animate
            target_width: Target width in pixels (0 = auto-size)
            duration_ms: Animation duration in milliseconds
            on_complete: Optional callback when animation completes
            animation_id: Optional unique ID for this animation (auto-generated if None)

        Returns:
            Animation ID string
        """
        if animation_id is None:
            animation_id = f"col_{column_idx}_{int(time.time() * 1000)}"

        # Cancel existing animation for this ID
        if animation_id in self.active_animations:
            self.cancel_animation(animation_id)

        # Get current width - BUGFIX: Query actual widget width, not minsize
        # Problem: grid_columnconfigure()['minsize'] returns 0 when no constraint set,
        # but widget is actually much wider due to content auto-sizing
        self.root.update_idletasks()
        widget.update_idletasks()
        current_width = widget.winfo_width()  # Get actual rendered width

        # For collapsing animations, lock current width as starting minsize
        # This prevents content from expanding back during animation
        if target_width > 0 and target_width < current_width:
            self.root.grid_columnconfigure(column_idx, minsize=current_width)
            self.root.update_idletasks()

        # Calculate animation parameters
        fps = 60
        frame_duration_ms = 1000 // fps
        total_frames = max(1, duration_ms // frame_duration_ms)
        width_delta = target_width - current_width

        # Store animation state
        self.active_animations[animation_id] = {
            'widget': widget,
            'column_idx': column_idx,
            'start_width': current_width,
            'target_width': target_width,
            'width_delta': width_delta,
            'current_frame': 0,
            'total_frames': total_frames,
            'frame_duration_ms': frame_duration_ms,
            'on_complete': on_complete,
            'after_id': None
        }

        # Start animation
        self._animate_frame(animation_id)

        return animation_id

    def _animate_frame(self, animation_id: str) -> None:
        """
        Execute one frame of animation.

        Args:
            animation_id: ID of animation to advance
        """
        if animation_id not in self.active_animations:
            return

        anim = self.active_animations[animation_id]

        # Calculate progress (0.0 to 1.0)
        progress = anim['current_frame'] / anim['total_frames']

        # Apply easing
        eased_progress = easing_cubic_in_out(progress)

        # Calculate current width
        current_width = anim['start_width'] + (anim['width_delta'] * eased_progress)

        # Update grid column
        if current_width <= 0:
            # Auto-size (remove minsize constraint)
            self.root.grid_columnconfigure(anim['column_idx'], minsize=0)
            # Remove widget width constraint to allow auto-sizing
            try:
                anim['widget'].config(width=0)
            except:
                pass  # Not all widgets support width config
        else:
            self.root.grid_columnconfigure(anim['column_idx'], minsize=int(current_width))
            # CRITICAL FIX: Force widget width to match minsize (overrides content size requests)
            try:
                anim['widget'].config(width=int(current_width))
            except:
                pass  # Not all widgets support width config
            self.root.update_idletasks()
            w, h = self.root.winfo_width(), self.root.winfo_height()
            self.root.geometry(f'{w}x{h}')  # Force grid recalculation

        # Advance frame
        anim['current_frame'] += 1

        # Check if animation complete
        if anim['current_frame'] > anim['total_frames']:
            # Final position
            if anim['target_width'] <= 0:
                self.root.grid_columnconfigure(anim['column_idx'], minsize=0)
                # CRITICAL FIX: Remove widget width constraint for auto-sizing
                try:
                    anim['widget'].config(width=0)
                except:
                    pass  # Not all widgets support width config
            else:
                self.root.grid_columnconfigure(anim['column_idx'], minsize=anim['target_width'])
                # CRITICAL FIX: Force widget width to match target (overrides content size requests)
                try:
                    anim['widget'].config(width=anim['target_width'])
                except:
                    pass  # Not all widgets support width config

            self.root.update_idletasks()
            w, h = self.root.winfo_width(), self.root.winfo_height()
            self.root.geometry(f'{w}x{h}')  # Force grid recalculation

            # Call completion callback
            on_complete = anim['on_complete']

            # Clean up
            del self.active_animations[animation_id]

            # Execute callback after cleanup
            if on_complete:
                on_complete()
        else:
            # Schedule next frame
            after_id = self.root.after(anim['frame_duration_ms'], lambda: self._animate_frame(animation_id))
            anim['after_id'] = after_id

    def cancel_animation(self, animation_id: str) -> None:
        """
        Cancel an ongoing animation.

        Args:
            animation_id: ID of animation to cancel
        """
        if animation_id not in self.active_animations:
            return

        anim = self.active_animations[animation_id]

        # Cancel pending after callback
        if anim['after_id'] is not None:
            self.root.after_cancel(anim['after_id'])

        # Remove from active animations
        del self.active_animations[animation_id]

    def is_animating(self, animation_id: str) -> bool:
        """
        Check if an animation is currently active.

        Args:
            animation_id: ID to check

        Returns:
            True if animation is in progress
        """
        return animation_id in self.active_animations

    def cancel_all(self) -> None:
        """Cancel all active animations."""
        for animation_id in list(self.active_animations.keys()):
            self.cancel_animation(animation_id)


def create_collapse_button(
    parent: tk.Widget,
    direction: str = 'left',
    command: Optional[Callable[[], None]] = None,
    theme: str = 'light'
) -> tk.Button:
    """
    Create a collapse/expand toggle button with arrow icon.

    Args:
        parent: Parent widget to attach button to
        direction: 'left' or 'right' (determines initial arrow orientation)
        command: Callback function when button is clicked
        theme: 'light' or 'dark' for color scheme

    Returns:
        Configured tk.Button widget
    """
    # Determine arrow icon based on direction
    if direction == 'left':
        arrow_icon = '◀'  # Point left (collapse)
    else:  # right
        arrow_icon = '▶'  # Point right (collapse)

    # Theme colors
    if theme == 'dark':
        bg_color = '#2b2b2b'
        fg_color = '#ffffff'
        active_bg = '#3c3c3c'
    else:  # light
        bg_color = '#f0f0f0'
        fg_color = '#000000'
        active_bg = '#e0e0e0'

    # Create button
    button = tk.Button(
        parent,
        text=arrow_icon,
        command=command,
        font=('Arial', 10, 'bold'),
        bg=bg_color,
        fg=fg_color,
        activebackground=active_bg,
        activeforeground=fg_color,
        relief='raised',
        borderwidth=1,
        width=2,
        height=1,
        cursor='hand2'
    )

    return button
