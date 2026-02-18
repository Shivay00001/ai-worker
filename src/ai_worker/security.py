"""
ASI v8 Security Module
InputValidator, safe_path, SecretRedactionFilter
"""

import logging
import re
from pathlib import Path
from typing import List

from .errors import ASISecurityError, ASIValidationError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT_INJECTION_PATTERNS: List[str] = [
    r"ignore\s+previous\s+instructions",
    r"system\s+prompt",
    r"act\s+as\b",
    r"<\|im_start\|>",
    r"\[INST\]",
    r"\bjailbreak\b",
    r"disregard\s+(all\s+)?(prior|previous)\s+instructions",
    r"forget\s+(all\s+)?previous",
    r"new\s+instructions?:",
]

SECRET_REDACTION_PATTERNS: List[tuple] = [
    # Bearer tokens
    (re.compile(r"Bearer\s+sk-[A-Za-z0-9\-_]+", re.IGNORECASE), "Bearer sk-[REDACTED]"),
    (re.compile(r"Bearer\s+gsk_[A-Za-z0-9\-_]+", re.IGNORECASE), "Bearer gsk_[REDACTED]"),
    (re.compile(r"Bearer\s+[A-Za-z0-9\-_\.]{20,}", re.IGNORECASE), "Bearer [REDACTED]"),
    # x-api-key header
    (re.compile(r"x-api-key:\s*[A-Za-z0-9\-_\.]{16,}", re.IGNORECASE), "x-api-key: [REDACTED]"),
    # Generic API key patterns
    (re.compile(r"(api[_\-]?key|apikey)\s*[=:]\s*['\"]?[A-Za-z0-9\-_\.]{16,}['\"]?", re.IGNORECASE),
     r"\1=[REDACTED]"),
]


# ---------------------------------------------------------------------------
# Input Validator
# ---------------------------------------------------------------------------

class InputValidator:
    """Validates and sanitises user input before processing."""

    def __init__(self, max_query_len: int = 8000):
        self.max_query_len = max_query_len
        self._injection_patterns = [re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS]

    def validate(self, raw_input: str) -> str:
        """
        Validate raw user input.
        Returns sanitised input string, or raises ASIValidationError / ASISecurityError.
        """
        # Must be a string
        if not isinstance(raw_input, str):
            raise ASIValidationError("Input must be a UTF-8 string.")

        # Must be valid UTF-8 (Python str is always unicode; catch bad bytes if bytes passed)
        try:
            raw_input.encode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError) as exc:
            raise ASIValidationError(f"Input is not valid UTF-8: {exc}") from exc

        # Length check
        if len(raw_input) > self.max_query_len:
            raise ASIValidationError(
                f"Input exceeds maximum length of {self.max_query_len} characters "
                f"(got {len(raw_input)})."
            )

        # Prompt injection detection
        for pattern in self._injection_patterns:
            if pattern.search(raw_input):
                raise ASISecurityError(
                    f"Prompt injection pattern detected: '{pattern.pattern}'"
                )

        return raw_input

    def wrap_for_prompt(self, validated_input: str) -> str:
        """Wrap validated input with explicit delimiters for LLM prompts."""
        return (
            "<<<USER_INPUT_START>>>\n"
            f"{validated_input}\n"
            "<<<USER_INPUT_END>>>"
        )


# ---------------------------------------------------------------------------
# Safe Path
# ---------------------------------------------------------------------------

def safe_path(requested_path: str, allowed_dirs: List[Path]) -> Path:
    """
    Resolve *requested_path* and confirm it resides under one of *allowed_dirs*.
    Raises ASISecurityError on path traversal attempts.
    """
    resolved = Path(requested_path).resolve()
    for allowed in allowed_dirs:
        try:
            resolved.relative_to(allowed.resolve())
            return resolved
        except ValueError:
            continue
    raise ASISecurityError(
        f"Path '{requested_path}' resolves to '{resolved}' which is outside "
        f"allowed directories: {[str(d) for d in allowed_dirs]}"
    )


# ---------------------------------------------------------------------------
# Secret Redaction Log Filter
# ---------------------------------------------------------------------------

class SecretRedactionFilter(logging.Filter):
    """Logging filter that strips API keys and secrets from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._redact(str(record.msg))
        record.args = self._redact_args(record.args)
        return True

    def _redact(self, text: str) -> str:
        for pattern, replacement in SECRET_REDACTION_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def _redact_args(self, args):
        if args is None:
            return args
        if isinstance(args, tuple):
            return tuple(self._redact(str(a)) if isinstance(a, str) else a for a in args)
        if isinstance(args, dict):
            return {k: self._redact(str(v)) if isinstance(v, str) else v for k, v in args.items()}
        return args
