import json
import os
from pathlib import Path


class ConfigHandler:
    """Handles loading and saving configuration for YouQuantiPy"""
    
    DEFAULT_CONFIG = {
        "video_recording": {
            "enabled": False,
            "save_directory": "./recordings",
            "filename_template": "{participant}_{timestamp}",
            "codec": "MJPG"
        },
        "camera_settings": {
            "target_fps": 30,
            "resolution": "720p"
        },
        "startup_mode": {
            "multi_face": False,
            "participant_count": 2,
            "camera_count": 2,
            "enable_mesh": False
        },
        "platform": {
            "mode": "auto",
            "force_camera_discovery": None,
            "force_gpu_backend": None,
            "force_display_mode": None
        },
        "advanced_detection": {
            "retinaface_model": "D:/Projects/youquantipy/retinaface.onnx",
            "retinaface_trt_path": "D:/Projects/youquantipy/retinaface.trt",
            "landmark_trt_path": "D:/Projects/youquantipy/landmark.trt", 
            "arcface_model": "D:/Projects/youquantipy/arcface.onnx",
            "tile_size": 640,
            "tile_overlap": 0.2,
            "detection_confidence": 0.3,
            "nms_threshold": 0.4,
            "max_detection_workers": 4,
            "landmark_worker_count": 4,
            "max_batch_size": 8,
            "max_track_id_reuse": 1000,
            "tracker_settings": {
                "max_age": 30,
                "min_hits": 3,
                "iou_threshold": 0.3,
                "max_drift": 50.0,
                "drift_correction_rate": 0.1,
                "detection_interval": 3,
                "tracking_confidence": 0.3,
                "stable_track_threshold": 5
            },
            "roi_settings": {
                "target_size": [256, 256],
                "padding_ratio": 0.3,
                "min_quality_score": 0.5,
                "max_roi_workers": 4,
                "quality_scoring": {
                    "aspect_ratio_penalty": 0.5,
                    "position_penalty": 0.5,
                    "boundary_margin_pixels": 50.0,
                    "score_weights": {
                        "size": 0.4,
                        "aspect": 0.2,
                        "position": 0.2,
                        "boundary": 0.2
                    }
                }
            },
            "face_size_limits": {
                "min_pixels": 20,
                "max_size_ratio": 0.9,
                "aspect_ratio_range": [0.5, 2.0]
            },
            "gpu_settings": {
                "gpu_device_id": 0,
                "workspace_size_mb": 1024,
                "batch_timeout_ms": 50,
                "use_gpu_roi_processing": True,
                "enable_gpu_ipc": False,
                "enable_optical_flow": False,
                "use_managed_memory": True,
                "memory_pool_limit_mb": 2048,
                "roi_batch_timeout_ms": 10,
                "landmark_poll_interval_ms": 0.1,
                "error_recovery_interval_ms": 100,
                "fps_report_interval_s": 5.0,
                "enable_fp16": True,
                "max_batch_size": 8
            },
            "recognition_settings": {
                "embedding_dim": 512,
                "max_embeddings_per_person": 50,
                "similarity_threshold": 0.5,
                "update_threshold": 0.7
            },
            "enrollment_settings": {
                "min_samples": 10,
                "min_quality": 0.7,
                "min_consistency": 0.85,
                "min_stability": 0.8,
                "collection_timeout": 30.0,
                "improvement_window": 20
            }
        },
        "audio_recording": {
            "enabled": False,
            "standalone_audio": False,
            "audio_with_video": False,
            "sample_rate": 44100,
            "channels": 1,
            "chunk_size": 1024,
            "timeout_seconds": 0.1,
            "queue_timeout_seconds": 2.0
        },
        "occlusion_recovery": {
            "enabled": True,
            "recovery_window_seconds": 30,
            "recovery_window_frames": 300,
            "zombie_track_min_hits": 3,
            "zombie_track_min_confidence": 0.3,
            "recovery_thresholds": {
                "embedding": 0.75,
                "shape": 0.85,
                "position": 0.7,
                "combined": 0.7
            },
            "recovery_weights": {
                "embedding": 0.5,
                "shape": 0.3,
                "position": 0.2
            },
            "enable_biometric_recovery": True,
            "recovery_query_timeout_ms": 50,
            "recovery_track_id_offset": 10000
        },
        "audio_devices": {
            # Will be populated with device assignments like "cam0": device_index
        },
        "camera_resolutions": {
            "480p": [640, 480],
            "720p": [1280, 720],
            "1080p": [1920, 1080],
            "4K": [3840, 2160]
        },
        "process_management": {
            "performance": {
                "target_retrieval_rate": 2.0,
                "metrics_history_size": 10,
                "timestamps_history_size": 100,
                "update_interval_seconds": 1.0,
                "progress_bar_length": 40,
                "cleanup_frequency": 10
            },
            "health_monitor": {
                "heartbeat_timeout_seconds": 10.0,
                "max_restart_attempts": 3,
                "check_interval_seconds": 1.0,
                "thread_timeout_seconds": 2.0,
                "history_limit": 100
            },
            "gpu_memory": {
                "pool_size": 100,
                "pool_limit_gb": 2.0,
                "device_id": 0
            },
            "gpu_frame_cache": {
                "max_size_mb": 500,
                "ttl_seconds": 0.5,
                "cleanup_frequency": 10
            }
        },
        "process_separation": {
            "ring_buffer_size": 16,
            "roi_buffer_size": 8,
            "gui_buffer_size": 8,
            "max_faces_per_frame": 8,
            "detection_interval": 3,
            "enable_pinned_memory": True,
            "continuous_roi_generation": True,
            "enable_temporal_smoothing": False,
            "smoothing_factor": 0.7,
            "enable_adaptive_batch_size": True,
            "use_zero_copy_transfers": True,
            "enable_preprocessing_process": True
        },
        "data_streaming": {
            "lsl": {
                "correlator_window_size": 60,
                "max_participants": 6,
                "fps_report_interval": 5.0,
                "main_loop_sleep": 0.001,
                "chunk_size": 1,
                "max_buffered": 360
            },
            "correlator": {
                "fast_window_size": 50,
                "slow_window_multiplier": 5,
                "delta_threshold": 0.005,
                "similarity_threshold": 0.6,
                "derivative_threshold": 0.03,
                "derivative_smooth": 0.3,
                "transient_decay": 0.8,
                "brief_decay": 0.9,
                "neutral_threshold_multiplier": 0.5
            },
            "performance": {
                "debug_interval_seconds": 2.0,
                "lsl_push_throttle": True,
                "queue_timeout_seconds": 0.1
            }
        },
        "gui_interface": {
            "canvas": {
                "refresh_interval": 0.033,
                "max_display_width": 640,
                "max_display_height": 480,
                "cache_timeout": 1.0,
                "draw_modes": {
                    "face_draw_mode": "full_contours",
                    "draw_jaw_overlay": True,
                    "debug_mode": True
                },
                "coordinate_transform": {
                    "frame_timeout": 0.5
                }
            },
            "status_panel": {
                "panel_height": 200,
                "update_interval": 1.0,
                "summary_format": {
                    "show_uptime": True,
                    "show_memory": True,
                    "show_fps": True
                },
                "diagnostic_timeout": 10.0
            },
            "reliability_monitor": {
                "memory_growth_threshold": 500,
                "max_queue_size": 10,
                "gui_freeze_threshold": 60.0,
                "resource_check_interval": 30,
                "queue_check_interval": 5,
                "gui_check_interval": 1,
                "stats_report_interval": 300,
                "canvas_cache_limit": 10,
                "startup_grace_period": 30.0
            }
        },
        "participant_management": {
            "enrollment": {
                "min_samples": 10,
                "min_quality_score": 0.7,
                "min_consistency_score": 0.85,
                "min_stability_score": 0.8,
                "collection_timeout": 30.0,
                "improvement_window": 20,
                "validation_criteria": {
                    "consistency_threshold": 0.8,
                    "stability_threshold": 0.7
                }
            },
            "face_recognition": {
                "embedding_dim": 512,
                "max_embeddings_per_person": 50,
                "similarity_threshold": 0.5,
                "update_threshold": 0.7,
                "queue_sizes": {
                    "input_queue": 100,
                    "output_queue": 100,
                    "command_queue": 50
                },
                "processing": {
                    "timeout": 0.001,
                    "batch_size": 1,
                    "model_init_timeout": 10.0
                }
            },
            "participant_tracking": {
                "face_threshold": 0.1,
                "pose_threshold": 0.15,
                "max_missed_frames": 60,
                "recently_lost_frames": 300,
                "procrustes_threshold": 0.05,
                "weights": {
                    "shape_weight": 0.7,
                    "position_weight": 0.3
                },
                "stable_landmarks": [33, 133, 362, 263, 168, 6, 10, 234, 454, 152],
                "assignment": {
                    "cost_threshold": 1.0,
                    "penalty_lost": 0.1,
                    "penalty_global": 0.2,
                    "penalty_missed_frames": 0.01
                }
            },
            "global_manager": {
                "distance_threshold": 0.2,
                "procrustes_threshold": 0.02,
                "recognition_threshold": 0.7,
                "weights": {
                    "shape_weight": 0.5,
                    "position_weight": 0.2,
                    "recognition_weight": 0.3
                },
                "cache_settings": {
                    "recently_lost_timeout": 10.0,
                    "max_embeddings_per_participant": 10,
                    "embedding_similarity_threshold": 0.65
                }
            }
        },
        "buffer_management": {
            "timeouts": {
                "recovery_polling_interval_ms": 5,
                "recovery_response_timeout_ms": 100,
                "command_retry_delay_ms": 10,
                "no_faces_sleep_ms": 10
            },
            "camera": {
                "buffer_size": 1
            },
            "commands": {
                "retry_count": 3,
                "timeout_seconds": 2.0
            },
            "pinned_memory": {
                "detection_frame_count": 4,
                "roi_buffer_count": 16,
                "hd_frame_count": 2
            }
        }
    }
    
    def __init__(self, config_file="./youquantipy_config.json"):
        # If relative path, make it relative to the project root (where gui.py is)
        config_path = Path(config_file)
        if not config_path.is_absolute():
            # Get the project root directory (two levels up from this file)
            this_dir = Path(__file__).parent
            project_root = this_dir.parent.parent  # Go up from core/process_management to main
            self.config_file = project_root / config_path
        else:
            self.config_file = config_path
        print(f"[ConfigHandler] Initializing with config file: {self.config_file.absolute()}")
        print(f"[ConfigHandler] Config file exists: {self.config_file.exists()}")
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                merged = self._merge_configs(self.DEFAULT_CONFIG, loaded_config)
                return merged
            except Exception as e:
                print(f"[Config] Error loading config: {e}")
                return self.DEFAULT_CONFIG.copy()
        else:
            # Create default config file
            self.save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default, loaded):
        """Recursively merge loaded config with defaults, preserving loaded values"""
        # Start with loaded config to preserve all user values
        result = loaded.copy()
        
        # Add missing keys from defaults
        for key, value in default.items():
            if key not in result:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result[key], dict):
                # Recursively merge nested dicts
                result[key] = self._merge_configs(value, result[key])
        
        return result
    
    def save_config(self, config=None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"[Config] Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"[Config] Error saving config: {e}")
    
    def get(self, key_path, default=None):
        """Get config value using dot notation (e.g., 'video_recording.save_directory')"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path, value):
        """Set config value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self.save_config()