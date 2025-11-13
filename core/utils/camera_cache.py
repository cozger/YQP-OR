"""
Camera Discovery Cache Module

Provides caching for camera discovery results to speed up application startup.
Cache is stored in ~/.youquantipy_cache.json with configurable TTL.
"""

import json
import logging
import os
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class CameraCache:
    """Thread-safe camera discovery cache with TTL."""

    DEFAULT_CACHE_PATH = Path.home() / ".youquantipy_camera_cache.json"
    DEFAULT_TTL_SECONDS = 300  # 5 minutes

    def __init__(self, cache_path: Optional[Path] = None, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        """
        Initialize camera cache.

        Args:
            cache_path: Path to cache file (default: ~/.youquantipy_camera_cache.json)
            ttl_seconds: Time-to-live in seconds (default: 300 = 5 minutes)
        """
        self.cache_path = cache_path or self.DEFAULT_CACHE_PATH
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        logger.info(f"Camera cache initialized: path={self.cache_path}, ttl={ttl_seconds}s")

    def save(self, cameras: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save camera list to cache.

        Args:
            cameras: List of camera dictionaries (from list_video_devices)
            metadata: Optional metadata to save alongside cameras

        Returns:
            True if save successful, False otherwise
        """
        with self._lock:
            try:
                cache_data = {
                    "version": "1.0",
                    "timestamp": time.time(),
                    "ttl_seconds": self.ttl_seconds,
                    "cameras": cameras,
                    "metadata": metadata or {}
                }

                # Ensure parent directory exists
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)

                # Write atomically (write to temp file, then rename)
                temp_path = self.cache_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(cache_data, f, indent=2)

                temp_path.replace(self.cache_path)

                logger.info(f"Saved {len(cameras)} camera(s) to cache: {self.cache_path}")
                return True

            except Exception as e:
                logger.error(f"Failed to save camera cache: {e}", exc_info=True)
                return False

    def load(self, max_age_seconds: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Load camera list from cache if valid.

        Args:
            max_age_seconds: Override TTL for this load (None = use instance TTL)

        Returns:
            Dict with 'cameras' and 'metadata' if cache is valid, None otherwise
        """
        with self._lock:
            try:
                if not self.cache_path.exists():
                    logger.debug("Cache file does not exist")
                    return None

                with open(self.cache_path, 'r') as f:
                    cache_data = json.load(f)

                # Validate cache structure
                if not isinstance(cache_data, dict):
                    logger.warning("Invalid cache format (not a dict)")
                    return None

                required_keys = ["timestamp", "cameras"]
                if not all(key in cache_data for key in required_keys):
                    logger.warning(f"Cache missing required keys: {required_keys}")
                    return None

                # Check TTL
                cache_age = time.time() - cache_data["timestamp"]
                max_age = max_age_seconds if max_age_seconds is not None else self.ttl_seconds

                if cache_age > max_age:
                    logger.info(f"Cache expired (age={cache_age:.1f}s, max={max_age}s)")
                    return None

                cameras = cache_data["cameras"]
                metadata = cache_data.get("metadata", {})

                logger.info(f"Loaded {len(cameras)} camera(s) from cache (age={cache_age:.1f}s)")
                return {
                    "cameras": cameras,
                    "metadata": metadata,
                    "cache_age": cache_age
                }

            except json.JSONDecodeError as e:
                logger.error(f"Cache file corrupted (invalid JSON): {e}")
                logger.warning(f"Invalidating corrupted cache: {self.cache_path}")
                try:
                    self.invalidate()  # FIX #8: Delete corrupted file
                except:
                    pass
                return None
            except Exception as e:
                logger.error(f"Failed to load camera cache: {e}", exc_info=True)
                return None

    def invalidate(self) -> bool:
        """
        Invalidate (delete) the cache file.

        Returns:
            True if cache was deleted, False if it didn't exist or deletion failed
        """
        with self._lock:
            try:
                if self.cache_path.exists():
                    self.cache_path.unlink()
                    logger.info(f"Cache invalidated: {self.cache_path}")
                    return True
                else:
                    logger.debug("Cache file does not exist (already invalid)")
                    return False
            except Exception as e:
                logger.error(f"Failed to invalidate cache: {e}", exc_info=True)
                return False

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache without loading it.

        Returns:
            Dict with cache metadata (exists, size, age, etc.)
        """
        with self._lock:
            info = {
                "exists": self.cache_path.exists(),
                "path": str(self.cache_path),
                "ttl_seconds": self.ttl_seconds
            }

            if info["exists"]:
                try:
                    stat = self.cache_path.stat()
                    info["size_bytes"] = stat.st_size
                    info["modified_time"] = stat.st_mtime
                    info["age_seconds"] = time.time() - stat.st_mtime
                    info["is_expired"] = info["age_seconds"] > self.ttl_seconds

                    # Try to read camera count without full validation
                    try:
                        with open(self.cache_path, 'r') as f:
                            data = json.load(f)
                        info["camera_count"] = len(data.get("cameras", []))
                    except:
                        info["camera_count"] = None
                        info["error"] = "Could not read cache file"

                except Exception as e:
                    info["error"] = str(e)

            return info


# Global cache instance (lazy initialization)
_global_cache: Optional[CameraCache] = None
_cache_lock = threading.Lock()


def get_camera_cache(ttl_seconds: int = CameraCache.DEFAULT_TTL_SECONDS) -> CameraCache:
    """
    Get or create the global camera cache instance.

    Args:
        ttl_seconds: TTL for cache entries (only used on first call)

    Returns:
        CameraCache instance
    """
    global _global_cache

    if _global_cache is None:
        with _cache_lock:
            # Double-check locking
            if _global_cache is None:
                _global_cache = CameraCache(ttl_seconds=ttl_seconds)

    return _global_cache


# Convenience functions for backward compatibility
def save_camera_cache(cameras: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Save cameras to global cache."""
    return get_camera_cache().save(cameras, metadata)


def load_camera_cache(max_age_seconds: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Load cameras from global cache."""
    return get_camera_cache().load(max_age_seconds)


def invalidate_camera_cache() -> bool:
    """Invalidate global cache."""
    return get_camera_cache().invalidate()


def get_cache_info() -> Dict[str, Any]:
    """Get global cache info."""
    return get_camera_cache().get_cache_info()
