"""
ASI v8 Tool Registry
Schema validation, permission checks, timeouts, audit logging.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from ..audit import AuditLogger
from ..errors import (
    ASIPermissionError,
    ASIToolError,
    ASIToolTimeoutError,
    ASIValidationError,
)
from ..security import SecretRedactionFilter

logger = logging.getLogger(__name__)
logger.addFilter(SecretRedactionFilter())


# ---------------------------------------------------------------------------
# Permission enum
# ---------------------------------------------------------------------------

class Permission(str, Enum):
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    WEB_FETCH = "web_fetch"
    CALC = "calc"


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: Any
    error_message: Optional[str] = None
    duration_ms: int


# ---------------------------------------------------------------------------
# Tool base class
# ---------------------------------------------------------------------------

class ToolDefinition(ABC):
    name: str
    description: str
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]
    required_permissions: List[Permission]
    timeout_seconds: float

    @abstractmethod
    async def _execute(self, validated_input: BaseModel) -> BaseModel:
        """Execute the tool with already-validated input. Return validated output."""


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Execution flow (in this exact order):
      a) Check tool exists
      b) Validate input pydantic schema
      c) Check permissions
      d) Execute with asyncio.wait_for(timeout)
      e) Validate output pydantic schema
      f) Log to audit
      g) Return ToolResult
    """

    def __init__(
        self,
        granted_permissions: List[Permission],
        audit_logger: Optional[AuditLogger] = None,
    ):
        self._tools: Dict[str, ToolDefinition] = {}
        self._granted_permissions = set(granted_permissions)
        self._audit = audit_logger

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    async def execute(
        self,
        tool_name: str,
        raw_input: Dict[str, Any],
        session_id: str = "",
        execution_id: str = "",
        trace_id: str = "",
    ) -> ToolResult:
        # a) Check tool exists
        if tool_name not in self._tools:
            raise ASIToolError(f"Unknown tool: '{tool_name}'")

        tool = self._tools[tool_name]

        # b) Validate input schema
        try:
            validated_input = tool.input_schema(**raw_input)
        except ValidationError as exc:
            raise ASIValidationError(
                f"Tool '{tool_name}' input validation failed: {exc}"
            ) from exc

        # c) Check permissions
        missing = set(tool.required_permissions) - self._granted_permissions
        if missing:
            raise ASIPermissionError(
                f"Tool '{tool_name}' requires permissions: {[p.value for p in missing]}"
            )

        # d) Execute with timeout
        t_start = time.monotonic()
        try:
            raw_output = await asyncio.wait_for(
                tool._execute(validated_input),
                timeout=tool.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            duration_ms = int((time.monotonic() - t_start) * 1000)
            self._audit_event(
                tool_name, raw_input, None, False,
                f"Timeout after {tool.timeout_seconds}s", duration_ms,
                session_id, execution_id, trace_id,
            )
            raise ASIToolTimeoutError(
                f"Tool '{tool_name}' exceeded timeout of {tool.timeout_seconds}s"
            ) from exc
        except Exception as exc:
            duration_ms = int((time.monotonic() - t_start) * 1000)
            self._audit_event(
                tool_name, raw_input, None, False, str(exc), duration_ms,
                session_id, execution_id, trace_id,
            )
            raise ASIToolError(f"Tool '{tool_name}' execution error: {exc}") from exc

        duration_ms = int((time.monotonic() - t_start) * 1000)

        # e) Validate output schema
        try:
            validated_output = tool.output_schema.model_validate(
                raw_output.model_dump() if hasattr(raw_output, "model_dump") else dict(raw_output)
            )
        except ValidationError as exc:
            raise ASIValidationError(
                f"Tool '{tool_name}' output validation failed: {exc}"
            ) from exc

        # f) Log to audit
        self._audit_event(
            tool_name,
            validated_input.model_dump(),
            validated_output.model_dump(),
            True,
            None,
            duration_ms,
            session_id,
            execution_id,
            trace_id,
        )

        # g) Return ToolResult
        return ToolResult(
            tool_name=tool_name,
            success=True,
            output=validated_output.model_dump(),
            duration_ms=duration_ms,
        )

    def _audit_event(
        self,
        tool_name: str,
        input_data: Any,
        output_data: Any,
        success: bool,
        error: Optional[str],
        duration_ms: int,
        session_id: str,
        execution_id: str,
        trace_id: str,
    ) -> None:
        if self._audit is None:
            return
        try:
            self._audit.log(
                event="tool_execution",
                data={
                    "tool_name": tool_name,
                    "input": input_data,
                    "output": output_data,
                    "success": success,
                    "error": error,
                    "duration_ms": duration_ms,
                    "session_id": session_id,
                },
                execution_id=execution_id,
                trace_id=trace_id,
            )
        except Exception as exc:
            logger.error("Failed to write tool audit event: %s", exc)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())
