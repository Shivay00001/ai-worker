"""
ASI v8 Web Fetch Tool
HTTP GET only, URL whitelist, no localhost, max 100KB response.
"""

import re
from typing import List, Optional
from urllib.parse import urlparse

import aiohttp
from pydantic import BaseModel, Field, HttpUrl

from ..errors import ASISecurityError, ASIToolError
from .registry import Permission, ToolDefinition

_BLOCKED_HOSTS = re.compile(
    r"^(localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|::1|0:0:0:0:0:0:0:1)$",
    re.IGNORECASE,
)

MAX_RESPONSE_BYTES = 102400  # 100 KB


class WebFetchInput(BaseModel):
    url: str = Field(..., description="URL to fetch (must be in ALLOWED_URLS)")


class WebFetchOutput(BaseModel):
    url: str
    status_code: int
    content: str
    content_length: int


class WebFetchTool(ToolDefinition):
    name = "web_fetch"
    description = "Fetch a URL via HTTP GET. Only whitelisted URLs are allowed."
    input_schema = WebFetchInput
    output_schema = WebFetchOutput
    required_permissions = [Permission.WEB_FETCH]
    timeout_seconds = 30.0

    def __init__(self, allowed_url_prefixes: List[str]):
        self._allowed_prefixes = allowed_url_prefixes

    async def _execute(self, validated_input: WebFetchInput) -> WebFetchOutput:
        url = validated_input.url

        # Parse and check scheme
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ASISecurityError(f"Only http/https allowed, got: {parsed.scheme}")

        # Block localhost / internal IPs
        host = parsed.hostname or ""
        if _BLOCKED_HOSTS.match(host):
            raise ASISecurityError(f"Blocked host: {host}")

        # Whitelist check
        if not any(url.startswith(prefix) for prefix in self._allowed_prefixes):
            raise ASISecurityError(
                f"URL not in whitelist. Allowed prefixes: {self._allowed_prefixes}"
            )

        client_timeout = aiohttp.ClientTimeout(total=self.timeout_seconds, connect=5)
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.get(url) as resp:
                    # Read up to MAX_RESPONSE_BYTES
                    raw = await resp.content.read(MAX_RESPONSE_BYTES + 1)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise ASIToolError(
                            f"Response exceeds maximum size of {MAX_RESPONSE_BYTES} bytes"
                        )
                    content = raw.decode("utf-8", errors="replace")
                    return WebFetchOutput(
                        url=url,
                        status_code=resp.status,
                        content=content,
                        content_length=len(raw),
                    )
        except aiohttp.ClientError as exc:
            raise ASIToolError(f"HTTP request failed: {exc}") from exc
