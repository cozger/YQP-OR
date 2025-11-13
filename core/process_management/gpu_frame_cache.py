"""
GPU Frame Cache for reducing redundant CPU-GPU transfers.

This module provides a thread-safe cache for GPU frames that can be shared
across different processing stages within the same process.
"""

import time
import threading
from typing import Dict, Optional, Tuple, Any
import numpy as np
from .confighandler import ConfigHandler

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


class GPUFrameCache:
    """
    Thread-safe cache for GPU frames with TTL and size management.
    
    Features:
    - LRU eviction when size limit reached
    - TTL-based expiration
    - Thread-safe operations
    - Memory usage tracking
    """
    
    def __init__(self, max_size_mb: int = None, ttl_seconds: float = None, config=None):
        """
        Initialize GPU frame cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes (uses config if None)
            ttl_seconds: Time-to-live for cached frames in seconds (uses config if None)
            config: Configuration handler instance
        """
        # Load configuration
        self.config = config if config else ConfigHandler()
        
        # Get configuration values with defaults
        max_size_mb = (
            max_size_mb if max_size_mb is not None 
            else self.config.get('process_management.gpu_frame_cache.max_size_mb', 500)
        )
        self.ttl_seconds = (
            ttl_seconds if ttl_seconds is not None 
            else self.config.get('process_management.gpu_frame_cache.ttl_seconds', 0.5)
        )
        self.cleanup_frequency = self.config.get('process_management.gpu_frame_cache.cleanup_frequency', 10)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        # Cache storage: frame_id -> (gpu_frame, timestamp, size_bytes)
        self.cache: Dict[int, Tuple[Any, float, int]] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.current_size_bytes = 0
        
        # Access tracking for LRU
        self.access_times: Dict[int, float] = {}
        
        self.enabled = HAS_CUPY
        if not self.enabled:
            print("[GPUFrameCache] CuPy not available, cache disabled")
    
    def put(self, frame_id: int, gpu_frame: Any) -> None:
        """
        Add a frame to the cache.
        
        Args:
            frame_id: Unique identifier for the frame
            gpu_frame: CuPy array on GPU
        """
        if not self.enabled or gpu_frame is None:
            return
            
        with self.lock:
            # Calculate frame size
            frame_size = gpu_frame.nbytes if hasattr(gpu_frame, 'nbytes') else 0
            
            # Remove old entry if exists
            if frame_id in self.cache:
                self._remove_entry(frame_id)
            
            # Check if we need to evict entries
            while (self.current_size_bytes + frame_size > self.max_size_bytes and 
                   len(self.cache) > 0):
                self._evict_lru()
            
            # Add new entry
            self.cache[frame_id] = (gpu_frame, time.time(), frame_size)
            self.access_times[frame_id] = time.time()
            self.current_size_bytes += frame_size
            
            # Clean expired entries periodically
            if len(self.cache) % self.cleanup_frequency == 0:
                self._clean_expired()
    
    def get(self, frame_id: int) -> Optional[Any]:
        """
        Retrieve a frame from cache.
        
        Args:
            frame_id: Frame identifier
            
        Returns:
            CuPy array if found and valid, None otherwise
        """
        if not self.enabled:
            return None
            
        with self.lock:
            if frame_id not in self.cache:
                self.misses += 1
                return None
            
            gpu_frame, timestamp, size_bytes = self.cache[frame_id]
            
            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                self._remove_entry(frame_id)
                self.misses += 1
                return None
            
            # Update access time
            self.access_times[frame_id] = time.time()
            self.hits += 1
            
            return gpu_frame
    
    def _remove_entry(self, frame_id: int) -> None:
        """Remove an entry from cache (internal use)."""
        if frame_id in self.cache:
            _, _, size_bytes = self.cache[frame_id]
            del self.cache[frame_id]
            del self.access_times[frame_id]
            self.current_size_bytes -= size_bytes
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.access_times:
            return
            
        # Find LRU entry
        lru_frame_id = min(self.access_times.items(), key=lambda x: x[1])[0]
        self._remove_entry(lru_frame_id)
        self.evictions += 1
    
    def _clean_expired(self) -> None:
        """Remove all expired entries."""
        current_time = time.time()
        expired_ids = []
        
        for frame_id, (_, timestamp, _) in self.cache.items():
            if current_time - timestamp > self.ttl_seconds:
                expired_ids.append(frame_id)
        
        for frame_id in expired_ids:
            self._remove_entry(frame_id)
    
    def clear(self) -> None:
        """Clear all cached frames."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.current_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate,
                'size_mb': self.current_size_bytes / (1024 * 1024),
                'num_entries': len(self.cache),
                'enabled': self.enabled
            }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self.lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0


# Global cache instance (per process)
_global_cache = None


def get_gpu_frame_cache(config=None) -> GPUFrameCache:
    """Get or create the global GPU frame cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = GPUFrameCache(config=config)
    return _global_cache