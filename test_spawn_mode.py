#!/usr/bin/env python3
"""Test if gui.py correctly sets spawn mode"""
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.dirname(__file__))

# Import gui module (will execute module-level code including set_start_method)
import gui

# Check start method
import multiprocessing
print(f"Multiprocessing start method: {multiprocessing.get_start_method()}")

# Test semaphore creation
try:
    s = multiprocessing.Semaphore(0)
    print(f"✅ Semaphore created successfully: {type(s).__name__}")
except Exception as e:
    print(f"❌ Semaphore creation failed: {e}")

print("Test complete")
