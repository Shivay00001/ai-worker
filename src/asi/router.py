"""
ASI v8 Model Router
CircuitBreaker per provider, hard timeouts, fallback, ModelResponse with hashes.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import aiohttp

from .config import Config
from .errors import ASIAllProvidersFailedError, ASIModelError
from .security import SecretRedactionFilter

logger = logging.getLogger(__name__)
logger.addFilter(SecretRedactionFilter())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# ModelResponse
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse:
    content: str
    provider: str
    model_name: str
    prompt_hash: str
    response_hash: str
    duration_ms: int


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    Per-provider circuit breaker.
    CLOSED → (3 failures in 60s) → OPEN → (120s) → HALF_OPEN → (success) → CLOSED
    """

    FAILURE_THRESHOLD = 3
    FAILURE_WINDOW_S = 60
    OPEN_DURATION_S = 120

    def __init__(self, provider: str):
        self.provider = provider
        self._state = CircuitState.CLOSED
        self._failure_times: List[float] = []
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> CircuitState:
        self._maybe_transition()
        return self._state

    def is_available(self) -> bool:
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        prev = self._state
        self._state = CircuitState.CLOSED
        self._failure_times.clear()
        self._opened_at = None
        if prev != CircuitState.CLOSED:
            logger.warning(
                "CircuitBreaker[%s]: %s → CLOSED (success)", self.provider, prev.value
            )

    def record_failure(self) -> None:
        now = time.monotonic()
        self._failure_times.append(now)
        # Prune old failures outside window
        self._failure_times = [
            t for t in self._failure_times if now - t <= self.FAILURE_WINDOW_S
        ]
        if (
            self._state == CircuitState.CLOSED
            and len(self._failure_times) >= self.FAILURE_THRESHOLD
        ):
            self._state = CircuitState.OPEN
            self._opened_at = now
            logger.warning(
                "CircuitBreaker[%s]: CLOSED → OPEN (%d failures in %ds)",
                self.provider,
                len(self._failure_times),
                self.FAILURE_WINDOW_S,
            )
        elif self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = now
            logger.warning(
                "CircuitBreaker[%s]: HALF_OPEN → OPEN (test call failed)", self.provider
            )

    def _maybe_transition(self) -> None:
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.OPEN_DURATION_S:
                self._state = CircuitState.HALF_OPEN
                logger.warning(
                    "CircuitBreaker[%s]: OPEN → HALF_OPEN (probe allowed)", self.provider
                )


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

async def _call_groq(
    prompt: str, api_key: str, model: str, timeout: int
) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
    }
    client_timeout = aiohttp.ClientTimeout(total=timeout, connect=5)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


async def _call_openai(
    prompt: str, api_key: str, model: str, timeout: int
) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
    }
    client_timeout = aiohttp.ClientTimeout(total=timeout, connect=5)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


async def _call_anthropic(
    prompt: str, api_key: str, model: str, timeout: int
) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
    }
    client_timeout = aiohttp.ClientTimeout(total=timeout, connect=5)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["content"][0]["text"]


async def _call_ollama(
    prompt: str, base_url: str, model: str, timeout: int
) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    client_timeout = aiohttp.ClientTimeout(total=timeout, connect=5)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["response"]


# ---------------------------------------------------------------------------
# Model Router
# ---------------------------------------------------------------------------

class ModelRouter:
    """
    Routes LLM calls to available providers with circuit breaking and fallback.
    Primary provider is tried first; on failure exactly one fallback is attempted.
    If all available providers have open circuits, raises ASIAllProvidersFailedError.
    """

    def __init__(self, config: Config):
        self.config = config
        self._breakers: Dict[str, CircuitBreaker] = {
            p: CircuitBreaker(p) for p in ["groq", "openai", "anthropic", "ollama"]
        }

    async def call(self, prompt: str) -> ModelResponse:
        """
        Call the preferred provider; fall back to one alternative on failure.
        Returns a ModelResponse. Raises ASIAllProvidersFailedError if unavailable.
        """
        ordered = self._provider_order()
        available = [p for p in ordered if self._breakers[p].is_available()]

        if not available:
            raise ASIAllProvidersFailedError(
                "All LLM providers have open circuit breakers."
            )

        prompt_hash = _sha256(prompt)
        last_error: Optional[Exception] = None

        for i, provider in enumerate(available[:2]):  # try primary + ONE fallback
            if i > 0:
                logger.warning(
                    "ModelRouter: switching to fallback provider '%s' (primary failed: %s)",
                    provider,
                    last_error,
                )
            try:
                t_start = time.monotonic()
                content = await asyncio.wait_for(
                    self._dispatch(provider, prompt),
                    timeout=self.config.model_timeout,
                )
                duration_ms = int((time.monotonic() - t_start) * 1000)
                self._breakers[provider].record_success()
                return ModelResponse(
                    content=content,
                    provider=provider,
                    model_name=self._model_name(provider),
                    prompt_hash=prompt_hash,
                    response_hash=_sha256(content),
                    duration_ms=duration_ms,
                )
            except asyncio.TimeoutError as exc:
                self._breakers[provider].record_failure()
                last_error = exc
                logger.warning(
                    "ModelRouter: provider '%s' timed out after %ds",
                    provider,
                    self.config.model_timeout,
                )
            except Exception as exc:
                self._breakers[provider].record_failure()
                last_error = exc
                logger.warning("ModelRouter: provider '%s' failed: %s", provider, exc)

        raise ASIModelError(
            f"All attempted providers failed. Last error: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provider_order(self) -> List[str]:
        preferred = self.config.preferred_provider
        others = [p for p in self.config.available_providers() if p != preferred]
        return [preferred] + others

    def _model_name(self, provider: str) -> str:
        return {
            "groq": self.config.groq_model,
            "openai": self.config.openai_model,
            "anthropic": self.config.anthropic_model,
            "ollama": self.config.ollama_model,
        }.get(provider, "unknown")

    async def _dispatch(self, provider: str, prompt: str) -> str:
        cfg = self.config
        if provider == "groq":
            if not cfg.groq_api_key:
                raise ASIModelError("GROQ_API_KEY not configured")
            return await _call_groq(prompt, cfg.groq_api_key, cfg.groq_model, cfg.model_timeout)
        if provider == "openai":
            if not cfg.openai_api_key:
                raise ASIModelError("OPENAI_API_KEY not configured")
            return await _call_openai(prompt, cfg.openai_api_key, cfg.openai_model, cfg.model_timeout)
        if provider == "anthropic":
            if not cfg.anthropic_api_key:
                raise ASIModelError("ANTHROPIC_API_KEY not configured")
            return await _call_anthropic(prompt, cfg.anthropic_api_key, cfg.anthropic_model, cfg.model_timeout)
        if provider == "ollama":
            return await _call_ollama(prompt, cfg.ollama_url, cfg.ollama_model, cfg.model_timeout)
        raise ASIModelError(f"Unknown provider: {provider}")

    def circuit_status(self) -> Dict[str, str]:
        return {p: cb.state.value for p, cb in self._breakers.items()}
