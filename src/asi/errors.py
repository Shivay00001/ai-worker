"""
ASI v8 Exception Hierarchy
All custom exceptions for the ASI v8 runtime.
"""


class ASIError(Exception):
    """Base exception for all ASI errors."""


class ASIValidationError(ASIError):
    """Raised when input or output fails schema validation."""


class ASISecurityError(ASIError):
    """Raised when a security policy is violated."""


class ASIStateError(ASIError):
    """Raised on invalid state machine transitions."""


class ASIModelError(ASIError):
    """Raised when an LLM call fails."""


class ASIAllProvidersFailedError(ASIModelError):
    """Raised when all LLM providers are unavailable (circuit open)."""


class ASIToolError(ASIError):
    """Raised when a tool execution fails."""


class ASIToolTimeoutError(ASIToolError):
    """Raised when a tool exceeds its timeout."""


class ASIPermissionError(ASIToolError):
    """Raised when a tool permission check fails."""


class ASIMemoryError(ASIError):
    """Raised when a database operation fails."""
