"""
ASI v8 File Tool
file_read with strict path validation — no traversal possible.
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from ..errors import ASISecurityError, ASIToolError
from ..security import safe_path
from .registry import Permission, ToolDefinition


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class FileReadInput(BaseModel):
    path: str = Field(..., description="Path to file to read")
    max_bytes: int = Field(default=102400, ge=1, le=1048576, description="Max bytes to read (default 100KB)")


class FileReadOutput(BaseModel):
    path: str
    content: str
    size_bytes: int


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class FileReadTool(ToolDefinition):
    name = "file_read"
    description = "Read a file from the workspace. Restricted to allowed directories."
    input_schema = FileReadInput
    output_schema = FileReadOutput
    required_permissions = [Permission.FILE_READ]
    timeout_seconds = 10.0

    def __init__(self, allowed_dirs: List[Path]):
        self._allowed_dirs = [Path(d).resolve() for d in allowed_dirs]

    async def _execute(self, validated_input: FileReadInput) -> FileReadOutput:
        try:
            resolved = safe_path(validated_input.path, self._allowed_dirs)
        except ASISecurityError:
            raise

        if not resolved.exists():
            raise ASIToolError(f"File not found: {validated_input.path}")

        if not resolved.is_file():
            raise ASIToolError(f"Path is not a file: {validated_input.path}")

        try:
            with resolved.open("r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(validated_input.max_bytes)
        except OSError as exc:
            raise ASIToolError(f"Cannot read file: {exc}") from exc

        return FileReadOutput(
            path=str(resolved),
            content=content,
            size_bytes=len(content.encode("utf-8")),
        )
