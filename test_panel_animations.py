#!/usr/bin/env python3
"""
Test script for panel animation system.

Tests the PanelAnimationController and collapse button independently
before full GUI integration.
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Add gui module to path
sys.path.insert(0, os.path.dirname(__file__))

from gui.panel_animations import PanelAnimationController, create_collapse_button


class TestPanelAnimationGUI(tk.Tk):
    """Minimal test GUI for panel animations."""

    def __init__(self):
        super().__init__()
        self.title("Panel Animation Test")
        self.geometry("800x400")

        # Configure grid
        self.grid_columnconfigure(0, weight=0)  # Left panel (collapsible)
        self.grid_columnconfigure(1, weight=1)  # Center (expands)
        self.grid_columnconfigure(2, weight=0, minsize=200)  # Right panel (collapsible)
        self.grid_rowconfigure(0, weight=1)

        self.left_collapsed = False
        self.right_collapsed = False

        # Create left panel
        self.left_panel = ttk.Frame(self, relief='solid', borderwidth=1)
        self.left_panel.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        # Add collapse button to left panel
        self.left_btn = create_collapse_button(
            self.left_panel,
            direction='left',
            command=self._toggle_left,
            theme='light'
        )
        self.left_btn.place(relx=1.0, rely=0.0, anchor='ne', x=-5, y=5)

        # Add some content to left panel
        ttk.Label(self.left_panel, text="Left Panel", font=('Arial', 14, 'bold')).pack(pady=10)
        for i in range(5):
            ttk.Label(self.left_panel, text=f"Control {i+1}").pack(pady=5)

        # Create center panel
        self.center_panel = ttk.Frame(self, relief='solid', borderwidth=1)
        self.center_panel.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        ttk.Label(
            self.center_panel,
            text="Center Panel\n(Expands to fill space)",
            font=('Arial', 14, 'bold')
        ).pack(expand=True)

        # Create right panel
        self.right_panel = ttk.Frame(self, relief='solid', borderwidth=1)
        self.right_panel.grid(row=0, column=2, sticky='nsew', padx=5, pady=5)

        # Add collapse button to right panel
        self.right_btn = create_collapse_button(
            self.right_panel,
            direction='right',
            command=self._toggle_right,
            theme='light'
        )
        self.right_btn.place(relx=0.0, rely=0.0, anchor='nw', x=5, y=5)

        # Add some content to right panel
        ttk.Label(self.right_panel, text="Right Panel", font=('Arial', 14, 'bold')).pack(pady=10)
        for i in range(5):
            ttk.Label(self.right_panel, text=f"Info {i+1}").pack(pady=5)

        # Initialize animation controller (after all panels are created)
        self.panel_animator = PanelAnimationController(
            self,
            left_panel=self.left_panel,
            right_panel=self.right_panel,
            center_container=self.center_panel
        )

        # Add status label
        self.status_label = ttk.Label(
            self,
            text="Click arrow buttons to test collapse/expand animation",
            font=('Arial', 10)
        )
        self.status_label.grid(row=1, column=0, columnspan=3, pady=5)

    def _toggle_left(self):
        """Toggle left panel."""
        if self.panel_animator.is_animating('left_panel'):
            return

        target_width = 30 if not self.left_collapsed else 0
        self.left_collapsed = not self.left_collapsed

        if self.left_collapsed:
            self._hide_children(self.left_panel, except_widget=self.left_btn)

        self.panel_animator.animate_width(
            self.left_panel,
            column_idx=0,
            target_width=target_width,
            duration_ms=250,
            on_complete=self._on_left_complete,
            animation_id='left_panel'
        )

        self._update_status("Left panel animating...")

    def _toggle_right(self):
        """Toggle right panel."""
        if self.panel_animator.is_animating('right_panel'):
            return

        target_width = 30 if not self.right_collapsed else 200
        self.right_collapsed = not self.right_collapsed

        if self.right_collapsed:
            self._hide_children(self.right_panel, except_widget=self.right_btn)

        self.panel_animator.animate_width(
            self.right_panel,
            column_idx=2,
            target_width=target_width,
            duration_ms=250,
            on_complete=self._on_right_complete,
            animation_id='right_panel'
        )

        self._update_status("Right panel animating...")

    def _hide_children(self, panel, except_widget=None):
        """Hide all children except specified widget."""
        for child in panel.winfo_children():
            if child != except_widget:
                child.pack_forget()

    def _show_children(self, panel):
        """Restore all children."""
        for child in panel.winfo_children():
            child.pack(pady=5)

    def _on_left_complete(self):
        """Callback after left panel animation."""
        if not self.left_collapsed:
            self._show_children(self.left_panel)

        # Update button icon
        self.left_btn.config(text='▶' if self.left_collapsed else '◀')
        self._update_status(f"Left panel {'collapsed' if self.left_collapsed else 'expanded'}")

    def _on_right_complete(self):
        """Callback after right panel animation."""
        if not self.right_collapsed:
            self._show_children(self.right_panel)

        # Update button icon
        self.right_btn.config(text='◀' if self.right_collapsed else '▶')
        self._update_status(f"Right panel {'collapsed' if self.right_collapsed else 'expanded'}")

    def _update_status(self, message):
        """Update status label."""
        self.status_label.config(text=message)


if __name__ == '__main__':
    print("Starting panel animation test...")
    app = TestPanelAnimationGUI()
    print("Test GUI created. Click arrow buttons to test animations.")
    app.mainloop()
    print("Test complete.")
