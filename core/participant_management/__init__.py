"""
Unified Participant Management Module

Single-participant tracking system with:
- Multi-camera data fusion
- IPC query buffer communication
- Enrollment state management
"""

from .single_participant_manager import SingleParticipantManager, EnrollmentState

# Backward compatibility aliases
# Old code can still import these names, they point to the unified manager
GlobalParticipantManager = SingleParticipantManager
GridParticipantManager = SingleParticipantManager
EnrollmentManager = SingleParticipantManager

__all__ = [
    'SingleParticipantManager',
    'EnrollmentState',
    'GlobalParticipantManager',  # Compatibility
    'GridParticipantManager',     # Compatibility
    'EnrollmentManager',          # Compatibility
]
