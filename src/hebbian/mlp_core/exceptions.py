"""
Custom exceptions for MLP construction and training.

This module defines a hierarchy of exceptions used throughout the MLP codebase
to provide clear, actionable error messages and enable appropriate error handling.

Design Principle: Never return None to indicate failure. Always raise exceptions
with comprehensive context about what went wrong and how to fix it.
"""

from typing import List, Tuple, Optional


class MLPError(Exception):
    """Base exception for all MLP-related errors."""
    pass


class MLPConstructionError(MLPError):
    """
    Raised when MLP construction fails after all retry attempts.

    This exception includes comprehensive information about all failed attempts
    and provides actionable suggestions for resolving the issue.
    """

    def __init__(
        self,
        message: str,
        attempts: Optional[List[Tuple[int, str, str]]] = None,
        suggestions: Optional[List[str]] = None,
        construction_config: Optional[dict] = None,
    ):
        """
        Initialize MLPConstructionError with detailed context.

        Args:
            message: Primary error message
            attempts: List of (attempt_num, exception_type, exception_message) tuples
            suggestions: List of actionable suggestions to fix the problem
            construction_config: Dict of configuration details for debugging
        """
        super().__init__(message)
        self.attempts = attempts or []
        self.suggestions = suggestions or []
        self.construction_config = construction_config or {}

    def __str__(self):
        """Format error message with full context."""
        msg = super().__str__()

        # Add configuration details
        if self.construction_config:
            msg += "\n\nConfiguration:"
            for key, value in self.construction_config.items():
                msg += f"\n  {key}: {value}"

        # Add attempt history (show first few and last few)
        if self.attempts:
            msg += f"\n\nAttempt History ({len(self.attempts)} total attempts):"

            # Show first 3 attempts
            for attempt_num, exc_type, exc_msg in self.attempts[:3]:
                # Truncate long messages
                exc_msg_short = exc_msg[:100] + "..." if len(exc_msg) > 100 else exc_msg
                msg += f"\n  Attempt {attempt_num+1}: {exc_type}: {exc_msg_short}"

            # Show ellipsis if there are many attempts
            if len(self.attempts) > 6:
                msg += f"\n  ... ({len(self.attempts)-6} more attempts) ..."

            # Show last 3 attempts (if different from first 3)
            if len(self.attempts) > 6:
                for attempt_num, exc_type, exc_msg in self.attempts[-3:]:
                    exc_msg_short = exc_msg[:100] + "..." if len(exc_msg) > 100 else exc_msg
                    msg += f"\n  Attempt {attempt_num+1}: {exc_type}: {exc_msg_short}"

        # Add suggestions
        if self.suggestions:
            msg += "\n\nSuggestions to fix this issue:"
            for suggestion in self.suggestions:
                msg += f"\n  - {suggestion}"

        return msg


class EncodingError(MLPConstructionError):
    """Raised when encoding step fails during MLP construction."""
    pass


class DecodingError(MLPConstructionError):
    """Raised when decoding step fails during MLP construction."""
    pass


class InvalidMappingError(MLPError, ValueError):
    """Raised when a mapping is invalid for the given configuration."""
    pass


class InvalidEmbeddingError(MLPError, ValueError):
    """Raised when embeddings are invalid (wrong shape, dtype, etc.)."""
    pass


class InvalidConfigurationError(MLPError, ValueError):
    """Raised when a configuration is invalid or inconsistent."""
    pass


class TrainingError(MLPError):
    """Raised when training fails."""
    pass


class ConvergenceError(TrainingError):
    """Raised when training fails to converge within specified constraints."""
    pass


class InvalidTaskError(MLPError, ValueError):
    """Raised when an MLPTask is invalid."""
    pass


class BinningError(EncodingError):
    """Raised when binning operation fails."""
    pass


class DeviceError(MLPError, RuntimeError):
    """Raised when device-related operations fail."""
    pass
