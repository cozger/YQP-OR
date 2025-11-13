"""
Command Protocol for YouQuantiPy IPC Communication

Standardized command formats and validation for the CommandBuffer system.
Replaces ad-hoc dictionary formats with structured, validated commands.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import json

logger = logging.getLogger('CommandProtocol')


def get_nested_config(config: Dict, path: str, default=None):
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., 'buffer_management.timeouts.command_retry_delay_ms')
        default: Default value if path not found
        
    Returns:
        Configuration value or default
    """
    if not config:
        return default
        
    keys = path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


class CommandProtocol:
    """
    Standardized command format and validation for YouQuantiPy IPC.
    
    Provides consistent command structure, validation, and serialization
    across all process communication channels.
    """
    
    # Command definitions with validation rules
    COMMANDS = {
        'register_track_participant': {
            'description': 'Register track ID with participant ID for occlusion recovery',
            'required_fields': ['track_id', 'participant_id'],
            'optional_fields': ['camera_index', 'confidence', 'embedding', 'bbox'],
            'field_types': {
                'track_id': int,
                'participant_id': int,
                'camera_index': int,
                'confidence': float,
                'embedding': list,  # List of floats
                'bbox': list       # [x, y, w, h]
            },
            'timeout': 2.0,
            'retry_count': 3,
            'priority': 'high'
        },
        'update_participant_embedding': {
            'description': 'Update face embedding for a participant',
            'required_fields': ['participant_id', 'embedding'],
            'optional_fields': ['camera_index', 'confidence', 'bbox'],
            'field_types': {
                'participant_id': int,
                'embedding': list,
                'camera_index': int,
                'confidence': float,
                'bbox': list
            },
            'timeout': 1.0,
            'retry_count': 2,
            'priority': 'medium'
        },
        'system_pause': {
            'description': 'Pause system processing',
            'required_fields': [],
            'optional_fields': ['duration_seconds', 'reason'],
            'field_types': {
                'duration_seconds': float,
                'reason': str
            },
            'timeout': 0.5,
            'retry_count': 1,
            'priority': 'high'
        },
        'system_resume': {
            'description': 'Resume system processing',
            'required_fields': [],
            'optional_fields': ['reason'],
            'field_types': {
                'reason': str
            },
            'timeout': 0.5,
            'retry_count': 1,
            'priority': 'high'
        },
        'update_detection_params': {
            'description': 'Update detection parameters at runtime',
            'required_fields': ['param_name', 'param_value'],
            'optional_fields': ['camera_index'],
            'field_types': {
                'param_name': str,
                'param_value': Union[int, float, str, bool],
                'camera_index': int
            },
            'timeout': 1.0,
            'retry_count': 2,
            'priority': 'medium'
        },
        'request_status': {
            'description': 'Request status information from process',
            'required_fields': ['status_type'],
            'optional_fields': ['detail_level'],
            'field_types': {
                'status_type': str,  # 'health', 'performance', 'config', 'buffers'
                'detail_level': str  # 'basic', 'detailed', 'debug'
            },
            'timeout': 1.0,
            'retry_count': 1,
            'priority': 'low'
        },
        'shutdown': {
            'description': 'Graceful shutdown request',
            'required_fields': [],
            'optional_fields': ['reason', 'timeout_seconds'],
            'field_types': {
                'reason': str,
                'timeout_seconds': float
            },
            'timeout': 2.0,
            'retry_count': 1,
            'priority': 'critical'
        },
        'handshake_ack': {
            'description': 'Acknowledgment of camera worker handshake',
            'required_fields': [],
            'optional_fields': ['camera_index', 'req_id'],
            'field_types': {
                'camera_index': int,
                'req_id': str
            },
            'timeout': 0.5,
            'retry_count': 1,
            'priority': 'high'
        },
        # Test commands for validation testing
        'test_command': {
            'description': 'Test command for validation and throughput testing',
            'required_fields': [],
            'optional_fields': ['index', 'data', 'payload'],
            'field_types': {
                'index': int,
                'data': str,
                'payload': dict
            },
            'timeout': 1.0,
            'retry_count': 1,
            'priority': 'low'
        },
        'fail_command': {
            'description': 'Test command that should intentionally fail',
            'required_fields': [],
            'optional_fields': ['data', 'reason'],
            'field_types': {
                'data': str,
                'reason': str
            },
            'timeout': 1.0,
            'retry_count': 1,
            'priority': 'low'
        }
    }
    
    # Priority levels for command processing
    PRIORITY_LEVELS = {
        'critical': 0,
        'high': 1,
        'medium': 2,
        'low': 3
    }
    
    @staticmethod
    def validate_command(command: Dict) -> Tuple[bool, str]:
        """
        Validate command structure and requirements.
        
        Args:
            command: Command dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check basic structure
            if not isinstance(command, dict):
                return False, "Command must be a dictionary"
            
            if 'type' not in command:
                return False, "Command must have 'type' field"
            
            command_type = command['type']
            if command_type not in CommandProtocol.COMMANDS:
                return False, f"Unknown command type: {command_type}"
            
            spec = CommandProtocol.COMMANDS[command_type]
            payload = command.get('payload', {})
            
            # Check required fields
            for field in spec['required_fields']:
                if field not in payload:
                    return False, f"Missing required field: {field}"
            
            # Check field types
            for field, expected_type in spec.get('field_types', {}).items():
                if field in payload:
                    value = payload[field]
                    
                    # Handle Union types (like Union[int, float, str, bool])
                    if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
                        # Check if value matches any of the union types
                        if not any(isinstance(value, t) for t in expected_type.__args__):
                            return False, f"Field '{field}' has invalid type. Expected one of {expected_type.__args__}, got {type(value)}"
                    else:
                        if not isinstance(value, expected_type):
                            return False, f"Field '{field}' has invalid type. Expected {expected_type.__name__}, got {type(value).__name__}"
            
            # Check for unknown fields (warning, not error)
            allowed_fields = set(spec['required_fields'] + spec.get('optional_fields', []))
            unknown_fields = set(payload.keys()) - allowed_fields
            if unknown_fields:
                logger.warning(f"Command {command_type} has unknown fields: {unknown_fields}")
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    @staticmethod
    def create_command(cmd_type: str, sender_id: str = None, **kwargs) -> Dict:
        """
        Create properly formatted command.
        
        Args:
            cmd_type: Type of command to create
            sender_id: Identifier of the sending process
            **kwargs: Command payload fields
            
        Returns:
            Formatted command dictionary
        """
        if cmd_type not in CommandProtocol.COMMANDS:
            raise ValueError(f"Unknown command type: {cmd_type}")
        
        # Create base command structure
        command = {
            'type': cmd_type,
            'payload': kwargs,
            'timestamp': int(time.time() * 1000000),  # microseconds
            'sender_id': sender_id or f"pid_{os.getpid()}",
            'version': '1.0'
        }
        
        # Add command metadata
        spec = CommandProtocol.COMMANDS[cmd_type]
        command['timeout'] = spec.get('timeout', 1.0)
        command['retry_count'] = spec.get('retry_count', 1)
        command['priority'] = spec.get('priority', 'medium')
        
        # Validate the created command
        is_valid, error = CommandProtocol.validate_command(command)
        if not is_valid:
            raise ValueError(f"Created invalid command: {error}")
        
        return command
    
    @staticmethod
    def create_response(original_command: Dict, success: bool, 
                       result: Any = None, error: str = None) -> Dict:
        """
        Create response to a command.
        
        Args:
            original_command: The command being responded to
            success: Whether the command was processed successfully
            result: Result data (if success is True)
            error: Error message (if success is False)
            
        Returns:
            Formatted response dictionary
        """
        response = {
            'type': 'response',
            'original_command_id': original_command.get('id'),
            'original_command_type': original_command.get('type'),
            'success': success,
            'timestamp': int(time.time() * 1000000),
            'responder_id': f"pid_{os.getpid()}"
        }
        
        if success and result is not None:
            response['result'] = result
        
        if not success and error:
            response['error'] = error
        
        return response
    
    @staticmethod
    def serialize_command(command: Dict, max_size: int = 1024) -> bytes:
        """
        Serialize command to bytes for transmission.
        
        Args:
            command: Command dictionary to serialize
            max_size: Maximum size in bytes
            
        Returns:
            Serialized command bytes
        """
        try:
            # Use compact JSON serialization
            json_str = json.dumps(command, separators=(',', ':'), ensure_ascii=False)
            json_bytes = json_str.encode('utf-8')
            
            if len(json_bytes) > max_size:
                raise ValueError(f"Command too large: {len(json_bytes)} > {max_size} bytes")
            
            return json_bytes
            
        except Exception as e:
            raise ValueError(f"Serialization failed: {e}")
    
    @staticmethod
    def deserialize_command(data: bytes) -> Dict:
        """
        Deserialize command from bytes.
        
        Args:
            data: Serialized command bytes
            
        Returns:
            Command dictionary
        """
        try:
            json_str = data.decode('utf-8')
            command = json.loads(json_str)
            
            # Validate deserialized command
            is_valid, error = CommandProtocol.validate_command(command)
            if not is_valid:
                raise ValueError(f"Deserialized invalid command: {error}")
            
            return command
            
        except Exception as e:
            raise ValueError(f"Deserialization failed: {e}")
    
    @staticmethod
    def get_command_info(cmd_type: str) -> Dict:
        """
        Get information about a command type.
        
        Args:
            cmd_type: Command type to get info for
            
        Returns:
            Command specification dictionary
        """
        if cmd_type not in CommandProtocol.COMMANDS:
            return {}
        
        return CommandProtocol.COMMANDS[cmd_type].copy()
    
    @staticmethod
    def list_commands() -> List[str]:
        """
        Get list of all supported command types.
        
        Returns:
            List of command type strings
        """
        return list(CommandProtocol.COMMANDS.keys())
    
    @staticmethod
    def get_priority(cmd_type: str) -> int:
        """
        Get numeric priority for a command type (lower = higher priority).
        
        Args:
            cmd_type: Command type
            
        Returns:
            Priority level (0 = critical, 3 = low)
        """
        if cmd_type not in CommandProtocol.COMMANDS:
            return 3  # Default to low priority
        
        priority_name = CommandProtocol.COMMANDS[cmd_type].get('priority', 'medium')
        return CommandProtocol.PRIORITY_LEVELS.get(priority_name, 2)


class CommandProcessor:
    """
    Separate thread for command processing to avoid blocking main loops.
    Handles command reception, validation, processing, and acknowledgment.
    """
    
    def __init__(self, command_buffer, handler_func, processor_id: str = None, config: Dict = None):
        """
        Initialize command processor.
        
        Args:
            command_buffer: CommandBuffer instance for receiving commands
            handler_func: Function to handle received commands
            processor_id: Identifier for this processor
            config: Configuration dictionary for timeouts
        """
        from sharedbuffer import CommandBuffer
        import threading
        
        if not isinstance(command_buffer, CommandBuffer):
            raise ValueError("command_buffer must be a CommandBuffer instance")
        
        self.command_buffer = command_buffer
        self.handler_func = handler_func
        self.processor_id = processor_id or f"processor_{os.getpid()}"
        self.config = config or {}
        
        # Threading control
        self.running = threading.Event()
        self.thread = None
        self.stats = {
            'commands_processed': 0,
            'commands_failed': 0,
            'start_time': None,
            'last_command_time': None
        }
        
        logger.info(f"CommandProcessor '{self.processor_id}' initialized")
    
    def start(self):
        """Start command processing thread."""
        if self.thread and self.thread.is_alive():
            logger.warning(f"CommandProcessor '{self.processor_id}' already running")
            return
        
        self.running.set()
        self.stats['start_time'] = time.time()
        
        import threading
        self.thread = threading.Thread(
            target=self._process_commands, 
            daemon=True,
            name=f"CommandProcessor-{self.processor_id}"
        )
        self.thread.start()
        
        logger.info(f"CommandProcessor '{self.processor_id}' started")
    
    def stop(self, timeout: float = 2.0):
        """
        Stop command processing thread.
        
        Args:
            timeout: Maximum time to wait for thread to stop
        """
        if not self.thread or not self.thread.is_alive():
            return
        
        logger.info(f"Stopping CommandProcessor '{self.processor_id}'...")
        self.running.clear()
        
        if self.thread:
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                logger.warning(f"CommandProcessor '{self.processor_id}' did not stop within timeout")
            else:
                logger.info(f"CommandProcessor '{self.processor_id}' stopped")
    
    def _process_commands(self):
        """Main command processing loop (runs in separate thread)."""
        logger.info(f"CommandProcessor '{self.processor_id}' processing loop started")
        
        while self.running.is_set():
            try:
                # Get next command with short timeout
                command = self.command_buffer.get_command(timeout=0.1)
                
                if command:
                    self.stats['last_command_time'] = time.time()
                    success, error = self._handle_command(command)
                    
                    # Send acknowledgment
                    self.command_buffer.send_acknowledgment(
                        command.get('id', -1), success, error
                    )
                    
                    if success:
                        self.stats['commands_processed'] += 1
                    else:
                        self.stats['commands_failed'] += 1
                        
                else:
                    # No command available, short sleep
                    time.sleep(0.001)  # 1ms
                    
            except Exception as e:
                logger.error(f"CommandProcessor '{self.processor_id}' processing error: {e}")
                error_sleep = get_nested_config(self.config, 'buffer_management.timeouts.command_retry_delay_ms', 10) / 1000.0
                time.sleep(error_sleep)  # Configurable delay on error
        
        logger.info(f"CommandProcessor '{self.processor_id}' processing loop ended")
    
    def _handle_command(self, command: Dict) -> Tuple[bool, Optional[str]]:
        """
        Handle a single command.
        
        Args:
            command: Command dictionary to process
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Validate command
            is_valid, error = CommandProtocol.validate_command(command)
            if not is_valid:
                logger.error(f"Invalid command received: {error}")
                return False, f"Invalid command: {error}"
            
            # Log command processing
            cmd_type = command.get('type', 'unknown')
            cmd_id = command.get('id', -1)
            logger.debug(f"Processing command {cmd_id} ({cmd_type})")
            
            # Call handler function
            result = self.handler_func(command)
            
            # Handler can return bool, tuple, or None
            if isinstance(result, bool):
                return result, None if result else "Handler returned False"
            elif isinstance(result, tuple) and len(result) == 2:
                return result[0], result[1]
            elif result is None:
                return True, None  # Assume success if None returned
            else:
                return True, None  # Assume success for other return types
                
        except Exception as e:
            error_msg = f"Command handler error: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        stats['is_running'] = self.running.is_set()
        stats['thread_alive'] = self.thread.is_alive() if self.thread else False
        
        if stats['start_time']:
            stats['uptime_seconds'] = time.time() - stats['start_time']
        
        return stats
    
    def __del__(self):
        """Destructor - ensure thread is stopped."""
        if hasattr(self, 'running'):
            self.stop(timeout=0.5)


# Convenience functions for common command operations
def create_track_registration_command(track_id: int, participant_id: int, 
                                     camera_index: int = None, **kwargs) -> Dict:
    """Create a track-participant registration command."""
    payload = {'track_id': track_id, 'participant_id': participant_id}
    if camera_index is not None:
        payload['camera_index'] = camera_index
    payload.update(kwargs)
    
    return CommandProtocol.create_command('register_track_participant', **payload)


def create_embedding_update_command(participant_id: int, embedding: List[float], **kwargs) -> Dict:
    """Create a participant embedding update command."""
    payload = {'participant_id': participant_id, 'embedding': embedding}
    payload.update(kwargs)
    
    return CommandProtocol.create_command('update_participant_embedding', **payload)


def create_system_pause_command(duration_seconds: float = None, reason: str = None) -> Dict:
    """Create a system pause command."""
    payload = {}
    if duration_seconds is not None:
        payload['duration_seconds'] = duration_seconds
    if reason:
        payload['reason'] = reason
    
    return CommandProtocol.create_command('system_pause', **payload)


def create_status_request_command(status_type: str, detail_level: str = 'basic') -> Dict:
    """Create a status request command."""
    return CommandProtocol.create_command('request_status', 
                                        status_type=status_type, 
                                        detail_level=detail_level)


# Import os at module level for use in CommandProtocol
import os