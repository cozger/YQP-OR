"""
Configuration Validation Module for YouQuantiPy
Provides comprehensive validation and detailed error reporting for configuration issues.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger('ConfigValidator')

class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    def __init__(self, message: str, missing_keys: List[str] = None, invalid_values: List[str] = None):
        super().__init__(message)
        self.missing_keys = missing_keys or []
        self.invalid_values = invalid_values or []

class ConfigValidator:
    """
    Comprehensive configuration validator with detailed error reporting.
    
    Provides specific error messages instead of vague "'gpu_settings'" errors.
    """
    
    # Define expected configuration structure
    REQUIRED_STRUCTURE = {
        'advanced_detection': {
            'required_keys': [
                'retinaface_trt_path',
                'landmark_trt_path', 
                'detection_confidence',
                'gpu_settings'
            ],
            'gpu_settings': {
                'required_keys': [
                    'enable_fp16',
                    'max_batch_size'
                ],
                'optional_keys': [
                    'enable_gpu_tracking',
                    'memory_pool_size_mb'
                ]
            }
        },
        'startup_mode': {
            'required_keys': [
                'participant_count',
                'camera_count'
            ]
        },
        'process_separation': {
            'required_keys': [
                'ring_buffer_size',
                'roi_buffer_size',
                'detection_interval'
            ],
            'optional_keys': [
                'enable_pinned_memory',
                'continuous_roi_generation',
                'enable_temporal_smoothing',
                'smoothing_factor'
            ]
        }
    }
    
    # Model file validation
    REQUIRED_MODEL_FILES = {
        'advanced_detection.retinaface_trt_path': 'RetinaFace TensorRT engine',
        'advanced_detection.landmark_trt_path': 'MediaPipe Landmark TensorRT engine'
    }
    
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Comprehensive configuration validation.
        
        Returns:
            Tuple[bool, List[str], List[str]]: (is_valid, errors, warnings)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        logger.info("Starting comprehensive configuration validation...")
        
        # 1. Validate structure and required keys
        self._validate_structure(config)
        
        # 2. Validate GPU settings specifically
        self._validate_gpu_settings(config)
        
        # 3. Validate model file paths
        self._validate_model_files(config)
        
        # 4. Validate participant/camera settings
        self._validate_participant_settings(config)
        
        # 5. Validate process separation settings
        self._validate_process_separation(config)
        
        # 6. Check for deprecated or moved settings
        self._check_deprecated_settings(config)
        
        is_valid = len(self.validation_errors) == 0
        
        if is_valid:
            logger.info("✅ Configuration validation passed")
        else:
            logger.error(f"❌ Configuration validation failed with {len(self.validation_errors)} errors")
            
        return is_valid, self.validation_errors, self.validation_warnings
    
    def _validate_structure(self, config: Dict[str, Any]):
        """Validate basic configuration structure."""
        logger.info("Validating configuration structure...")
        
        for section, requirements in self.REQUIRED_STRUCTURE.items():
            if section not in config:
                self.validation_errors.append(
                    f"Missing required configuration section: '{section}'"
                )
                continue
            
            section_config = config[section]
            if not isinstance(section_config, dict):
                self.validation_errors.append(
                    f"Configuration section '{section}' must be a dictionary, got {type(section_config).__name__}"
                )
                continue
            
            # Check required keys in this section
            if 'required_keys' in requirements:
                for key in requirements['required_keys']:
                    if key not in section_config:
                        self.validation_errors.append(
                            f"Missing required key '{key}' in section '{section}'"
                        )
    
    def _validate_gpu_settings(self, config: Dict[str, Any]):
        """Validate GPU settings with detailed error messages."""
        logger.info("Validating GPU settings...")
        
        # Check for common configuration mistakes
        
        # 1. Check if gpu_settings is at wrong level
        if 'gpu_settings' in config and 'advanced_detection' in config:
            if 'gpu_settings' not in config['advanced_detection']:
                self.validation_warnings.append(
                    "Found 'gpu_settings' at root level but not under 'advanced_detection'. "
                    "Process separation expects gpu_settings under advanced_detection."
                )
                
                # Auto-fix: Move gpu_settings to correct location
                config['advanced_detection']['gpu_settings'] = config['gpu_settings']
                logger.info("Auto-fixed: Moved gpu_settings under advanced_detection")
        
        # 2. Validate gpu_settings content
        gpu_settings_path = self._get_nested_value(config, 'advanced_detection.gpu_settings')
        if gpu_settings_path is None:
            self.validation_errors.append(
                "Missing 'gpu_settings' under 'advanced_detection'. Required for GPU pipeline operation."
            )
            return
        
        gpu_settings = gpu_settings_path
        
        # Validate specific GPU settings
        if 'enable_fp16' not in gpu_settings:
            self.validation_warnings.append(
                "Missing 'enable_fp16' in gpu_settings. Defaulting to True."
            )
            gpu_settings['enable_fp16'] = True
        
        if 'max_batch_size' not in gpu_settings:
            self.validation_warnings.append(
                "Missing 'max_batch_size' in gpu_settings. Defaulting to 8."
            )
            gpu_settings['max_batch_size'] = 8
        
        # Validate values
        max_batch_size = gpu_settings.get('max_batch_size')
        if not isinstance(max_batch_size, int) or max_batch_size < 1 or max_batch_size > 32:
            self.validation_errors.append(
                f"Invalid 'max_batch_size' in gpu_settings: {max_batch_size}. Must be integer between 1 and 32."
            )
        
        if not isinstance(gpu_settings.get('enable_fp16'), bool):
            self.validation_errors.append(
                f"Invalid 'enable_fp16' in gpu_settings: {gpu_settings.get('enable_fp16')}. Must be boolean."
            )
    
    def _validate_model_files(self, config: Dict[str, Any]):
        """Validate that required model files exist."""
        logger.info("Validating model file paths...")
        
        for path_key, description in self.REQUIRED_MODEL_FILES.items():
            model_path = self._get_nested_value(config, path_key)
            
            if model_path is None:
                self.validation_errors.append(
                    f"Missing model path configuration: '{path_key}' ({description})"
                )
                continue
            
            if not isinstance(model_path, str):
                self.validation_errors.append(
                    f"Model path '{path_key}' must be a string, got {type(model_path).__name__}"
                )
                continue
            
            # Check if file exists
            model_file = Path(model_path)
            if not model_file.exists():
                self.validation_errors.append(
                    f"Model file not found: {model_path} ({description})"
                )
                continue
            
            # Check file size (TensorRT engines should be substantial)
            file_size = model_file.stat().st_size
            if file_size < 1024:  # Less than 1KB is suspicious
                self.validation_warnings.append(
                    f"Model file suspiciously small: {model_path} ({file_size} bytes)"
                )
            
            logger.info(f"✅ Model file validated: {description} ({file_size:,} bytes)")
    
    def _validate_participant_settings(self, config: Dict[str, Any]):
        """Validate participant and camera settings."""
        logger.info("Validating participant settings...")
        
        startup_mode = config.get('startup_mode', {})
        
        participant_count = startup_mode.get('participant_count')
        camera_count = startup_mode.get('camera_count')
        
        if participant_count is None:
            self.validation_errors.append(
                "Missing 'participant_count' in startup_mode"
            )
        elif not isinstance(participant_count, int) or participant_count < 1:
            self.validation_errors.append(
                f"Invalid participant_count: {participant_count}. Must be positive integer."
            )
        
        if camera_count is None:
            self.validation_errors.append(
                "Missing 'camera_count' in startup_mode"
            )
        elif not isinstance(camera_count, int) or camera_count < 1:
            self.validation_errors.append(
                f"Invalid camera_count: {camera_count}. Must be positive integer."
            )
        
        # Warn about resource usage
        if participant_count and camera_count:
            total_load = participant_count * camera_count
            if total_load > 8:
                self.validation_warnings.append(
                    f"High resource usage: {participant_count} participants × {camera_count} cameras = {total_load} processing streams"
                )
    
    def _validate_process_separation(self, config: Dict[str, Any]):
        """Validate process separation settings."""
        logger.info("Validating process separation settings...")
        
        # Check if process_separation section exists, if not create defaults
        if 'process_separation' not in config:
            self.validation_warnings.append(
                "Missing 'process_separation' section. Creating with defaults."
            )
            config['process_separation'] = {
                'ring_buffer_size': 32,
                'roi_buffer_size': 16,
                'detection_interval': 3,
                'enable_pinned_memory': True,
                'continuous_roi_generation': True,
                'enable_temporal_smoothing': True,
                'smoothing_factor': 0.7
            }
            return
        
        process_sep = config['process_separation']
        
        # Validate buffer sizes (must be powers of 2 for efficiency)
        ring_buffer_size = process_sep.get('ring_buffer_size', 32)
        if not self._is_power_of_2(ring_buffer_size):
            self.validation_warnings.append(
                f"ring_buffer_size ({ring_buffer_size}) should be power of 2 for optimal performance"
            )
        
        detection_interval = process_sep.get('detection_interval')
        if detection_interval is None:
            self.validation_warnings.append(
                "Missing 'detection_interval' in process_separation. Defaulting to 3."
            )
            process_sep['detection_interval'] = 3
        elif not isinstance(detection_interval, int) or detection_interval < 1:
            self.validation_errors.append(
                f"Invalid detection_interval: {detection_interval}. Must be positive integer."
            )
    
    def _check_deprecated_settings(self, config: Dict[str, Any]):
        """Check for deprecated or moved configuration settings."""
        logger.info("Checking for deprecated settings...")
        
        # Common migration issues
        deprecated_mappings = {
            'gpu_settings': 'advanced_detection.gpu_settings',
            'detection_interval': 'process_separation.detection_interval'
        }
        
        for old_key, new_key in deprecated_mappings.items():
            if old_key in config and self._get_nested_value(config, new_key) is None:
                self.validation_warnings.append(
                    f"Configuration key '{old_key}' should be moved to '{new_key}' for new architecture"
                )
    
    def _get_nested_value(self, config: Dict[str, Any], key_path: str) -> Any:
        """Get nested configuration value using dot notation."""
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _is_power_of_2(self, n: int) -> bool:
        """Check if number is power of 2."""
        return n > 0 and (n & (n - 1)) == 0
    
    def create_diagnostic_report(self, config: Dict[str, Any]) -> str:
        """Create a comprehensive diagnostic report."""
        report = []
        report.append("=" * 60)
        report.append("YouQuantiPy Configuration Diagnostic Report")
        report.append("=" * 60)
        report.append("")
        
        # System information
        report.append("System Information:")
        report.append(f"  Configuration file: {getattr(self, 'config_file', 'Unknown')}")
        report.append(f"  Process separation: {'Enabled' if 'process_separation' in config else 'Disabled'}")
        report.append(f"  GPU settings location: {'advanced_detection.gpu_settings' if self._get_nested_value(config, 'advanced_detection.gpu_settings') else 'root.gpu_settings' if 'gpu_settings' in config else 'Missing'}")
        report.append("")
        
        # Validation results
        if self.validation_errors:
            report.append("❌ CRITICAL ERRORS:")
            for i, error in enumerate(self.validation_errors, 1):
                report.append(f"  {i}. {error}")
            report.append("")
        
        if self.validation_warnings:
            report.append("⚠️  WARNINGS:")
            for i, warning in enumerate(self.validation_warnings, 1):
                report.append(f"  {i}. {warning}")
            report.append("")
        
        # Configuration summary
        report.append("Configuration Summary:")
        startup_mode = config.get('startup_mode', {})
        report.append(f"  Participants: {startup_mode.get('participant_count', 'Unknown')}")
        report.append(f"  Cameras: {startup_mode.get('camera_count', 'Unknown')}")
        
        gpu_settings = self._get_nested_value(config, 'advanced_detection.gpu_settings') or config.get('gpu_settings', {})
        report.append(f"  GPU Batch Size: {gpu_settings.get('max_batch_size', 'Unknown')}")
        report.append(f"  FP16 Enabled: {gpu_settings.get('enable_fp16', 'Unknown')}")
        
        process_sep = config.get('process_separation', {})
        report.append(f"  Detection Interval: {process_sep.get('detection_interval', 'Unknown')}")
        report.append("")
        
        # Recommendations
        report.append("Recommendations:")
        if self.validation_errors:
            report.append("  1. Fix all critical errors before starting the application")
        if 'gpu_settings' in config and 'advanced_detection' in config:
            if 'gpu_settings' not in config['advanced_detection']:
                report.append("  2. Move gpu_settings under advanced_detection for new architecture")
        report.append("  3. Ensure all TensorRT model files are present and valid")
        report.append("  4. Test with minimal configuration first (1 participant, 1 camera)")
        
        return "\n".join(report)

def validate_config_and_report(config: Dict[str, Any], config_file: str = None) -> Tuple[bool, str]:
    """
    Convenience function to validate configuration and generate report.
    
    Returns:
        Tuple[bool, str]: (is_valid, diagnostic_report)
    """
    validator = ConfigValidator()
    if config_file:
        validator.config_file = config_file
    
    is_valid, errors, warnings = validator.validate_configuration(config)
    report = validator.create_diagnostic_report(config)
    
    return is_valid, report

# Example usage and testing
if __name__ == "__main__":
    # Test with a sample configuration
    test_config = {
        "startup_mode": {
            "participant_count": 2,
            "camera_count": 1
        },
        "gpu_settings": {  # Wrong location - should be under advanced_detection
            "max_batch_size": 8,
            "enable_fp16": True
        },
        "advanced_detection": {
            "retinaface_trt_path": "../retinaface.trt",
            "landmark_trt_path": "../landmark.trt",
            "detection_confidence": 0.7
            # Missing gpu_settings here
        }
    }
    
    is_valid, report = validate_config_and_report(test_config, "test_config.json")
    print(report)
    print(f"\nValidation result: {'PASSED' if is_valid else 'FAILED'}")