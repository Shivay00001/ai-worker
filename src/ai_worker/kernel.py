"""
ASI v8 Execution Kernel

SELF_REVIEW:
1. Weakest architectural assumption: the PLAN state simply passes the query to the model with memory context; real planning with structured tool-use decisions is absent.
2. Highest security risk remaining: the web_fetch and file_read allowed lists are configured at runtime — a misconfigured caller could open them too broadly.
3. Most likely failure point in production: circuit breakers all open simultaneously if the network hiccups during startup, requiring a 120s wait before recovery.
4. What was removed that might be needed in V2: multi-agent collaboration, parallel reasoning streams, streaming responses, sandboxed code execution.
5. Where could this be simpler: the VALIDATE_OUTPUT state is currently a stub that just checks non-empty content; richer output validation would need a schema.
"""

import asyncio
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

from .audit import AuditLogger
from .config import Config
from .errors import (
    ASIAllProvidersFailedError,
    ASIError,
    ASIMemoryError,
    ASIModelError,
    ASISecurityError,
    ASIStateError,
    ASIToolError,
    ASIValidationError,
)
from .memory import MemoryLayer
from .router import ModelResponse, ModelRouter
from .security import InputValidator, SecretRedactionFilter
from .tools.registry import Permission, ToolRegistry

logger = logging.getLogger(__name__)
logger.addFilter(SecretRedactionFilter())


# ---------------------------------------------------------------------------
# State machine definition
# ---------------------------------------------------------------------------

class KernelState(str, Enum):
    INIT = "INIT"
    CLASSIFY_INTENT = "CLASSIFY_INTENT"
    FETCH_CONTEXT = "FETCH_CONTEXT"
    PLAN = "PLAN"
    MODEL_CALL = "MODEL_CALL"
    TOOL_CALL = "TOOL_CALL"
    VALIDATE_OUTPUT = "VALIDATE_OUTPUT"
    STORE_MEMORY = "STORE_MEMORY"
    FINALIZE = "FINALIZE"
    ERROR = "ERROR"


# Valid transitions — no hidden paths
_VALID_TRANSITIONS: Dict[KernelState, List[KernelState]] = {
    KernelState.INIT:            [KernelState.CLASSIFY_INTENT, KernelState.ERROR],
    KernelState.CLASSIFY_INTENT: [KernelState.FETCH_CONTEXT, KernelState.ERROR],
    KernelState.FETCH_CONTEXT:   [KernelState.PLAN, KernelState.ERROR],
    KernelState.PLAN:            [KernelState.MODEL_CALL, KernelState.ERROR],
    KernelState.MODEL_CALL:      [KernelState.TOOL_CALL, KernelState.VALIDATE_OUTPUT, KernelState.ERROR],
    KernelState.TOOL_CALL:       [KernelState.MODEL_CALL, KernelState.VALIDATE_OUTPUT, KernelState.ERROR],
    KernelState.VALIDATE_OUTPUT: [KernelState.STORE_MEMORY, KernelState.ERROR],
    KernelState.STORE_MEMORY:    [KernelState.FINALIZE, KernelState.ERROR],
    KernelState.FINALIZE:        [],  # terminal
    KernelState.ERROR:           [],  # terminal
}


# ---------------------------------------------------------------------------
# State I/O dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InitInput:
    raw_query: str
    session_id: str
    execution_id: str
    trace_id: str

@dataclass
class InitOutput:
    validated_query: str
    session_id: str
    execution_id: str
    trace_id: str

@dataclass
class ClassifyIntentOutput:
    intent: str          # e.g. "question", "task", "tool_request"
    requires_tools: bool
    validated_query: str

@dataclass
class FetchContextOutput:
    memories: List[Dict[str, Any]]
    validated_query: str
    intent: str
    requires_tools: bool

@dataclass
class PlanOutput:
    plan_prompt: str
    validated_query: str
    intent: str
    requires_tools: bool
    memories: List[Dict[str, Any]]

@dataclass
class ModelCallOutput:
    model_response: ModelResponse
    plan_prompt: str
    validated_query: str
    requires_tools: bool

@dataclass
class ToolCallOutput:
    tool_results: List[Dict[str, Any]]
    model_response: ModelResponse
    plan_prompt: str
    validated_query: str

@dataclass
class ValidateOutputOutput:
    final_content: str
    model_response: ModelResponse
    tool_results: List[Dict[str, Any]]

@dataclass
class StoreMemoryOutput:
    memory_id: str
    final_content: str
    model_response: ModelResponse

@dataclass
class FinalizeOutput:
    response: str
    session_id: str
    execution_id: str
    provider: str
    model_name: str
    prompt_hash: str
    response_hash: str
    duration_ms: int

@dataclass
class ErrorOutput:
    error_type: str
    error_message: str
    failed_state: str
    session_id: str
    execution_id: str


# ---------------------------------------------------------------------------
# Execution context (carries state across transitions)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionContext:
    raw_query: str = ""
    validated_query: str = ""
    session_id: str = ""
    execution_id: str = ""
    trace_id: str = ""
    intent: str = "unknown"
    requires_tools: bool = False
    memories: List[Dict[str, Any]] = field(default_factory=list)
    plan_prompt: str = ""
    model_response: Optional[ModelResponse] = None
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    final_content: str = ""
    memory_id: str = ""
    error: Optional[Exception] = None
    failed_state: str = ""
    start_time: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ExecutionKernel:
    """
    Single-process, single-asyncio-event-loop execution kernel.
    Drives a strict state machine from INIT to FINALIZE (or ERROR).
    """

    def __init__(
        self,
        config: Config,
        router: ModelRouter,
        memory: MemoryLayer,
        audit: AuditLogger,
        tool_registry: ToolRegistry,
    ):
        self._config = config
        self._router = router
        self._memory = memory
        self._audit = audit
        self._tools = tool_registry
        self._validator = InputValidator(max_query_len=config.max_query_len)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, raw_query: str) -> FinalizeOutput:
        """
        Execute a full query through the state machine.
        Returns FinalizeOutput on success.
        On error, logs to audit and raises the original exception.
        """
        session_id = secrets.token_hex(16)
        execution_id = secrets.token_hex(16)
        trace_id = secrets.token_hex(16)

        ctx = ExecutionContext(
            raw_query=raw_query,
            session_id=session_id,
            execution_id=execution_id,
            trace_id=trace_id,
        )

        current_state = KernelState.INIT

        try:
            ctx = await self._state_init(ctx)
            current_state = KernelState.CLASSIFY_INTENT
            self._transition(KernelState.INIT, current_state)

            ctx = await self._state_classify_intent(ctx)
            current_state = KernelState.FETCH_CONTEXT
            self._transition(KernelState.CLASSIFY_INTENT, current_state)

            ctx = await self._state_fetch_context(ctx)
            current_state = KernelState.PLAN
            self._transition(KernelState.FETCH_CONTEXT, current_state)

            ctx = await self._state_plan(ctx)
            current_state = KernelState.MODEL_CALL
            self._transition(KernelState.PLAN, current_state)

            ctx = await self._state_model_call(ctx)
            if ctx.requires_tools and ctx.tool_results == []:
                current_state = KernelState.TOOL_CALL
                self._transition(KernelState.MODEL_CALL, current_state)
                ctx = await self._state_tool_call(ctx)
                current_state = KernelState.VALIDATE_OUTPUT
                self._transition(KernelState.TOOL_CALL, current_state)
            else:
                current_state = KernelState.VALIDATE_OUTPUT
                self._transition(KernelState.MODEL_CALL, current_state)

            ctx = await self._state_validate_output(ctx)
            current_state = KernelState.STORE_MEMORY
            self._transition(KernelState.VALIDATE_OUTPUT, current_state)

            ctx = await self._state_store_memory(ctx)
            current_state = KernelState.FINALIZE
            self._transition(KernelState.STORE_MEMORY, current_state)

            result = await self._state_finalize(ctx)
            return result

        except ASIError as exc:
            ctx.error = exc
            ctx.failed_state = current_state.value
            await self._state_error(ctx)
            raise
        except Exception as exc:
            ctx.error = exc
            ctx.failed_state = current_state.value
            await self._state_error(ctx)
            raise

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    async def _state_init(self, ctx: ExecutionContext) -> ExecutionContext:
        t = time.monotonic()
        self._audit.log(
            "state_enter",
            {"state": "INIT", "session_id": ctx.session_id, "query_len": len(ctx.raw_query)},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )

        try:
            validated = self._validator.validate(ctx.raw_query)
        except (ASIValidationError, ASISecurityError):
            raise

        ctx.validated_query = validated
        query_hash = _sha256(validated)

        try:
            self._memory.create_session(
                session_id=ctx.session_id,
                execution_id=ctx.execution_id,
                trace_id=ctx.trace_id,
                query_hash=query_hash,
            )
        except ASIMemoryError:
            raise

        duration_ms = int((time.monotonic() - t) * 1000)
        self._memory.record_step(
            session_id=ctx.session_id,
            state_from="",
            state_to="INIT",
            input_data={"raw_query": ctx.raw_query},
            output_data={"validated_query": validated},
            duration_ms=duration_ms,
        )
        self._audit.log(
            "state_exit",
            {"state": "INIT", "session_id": ctx.session_id, "duration_ms": duration_ms},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )
        return ctx

    async def _state_classify_intent(self, ctx: ExecutionContext) -> ExecutionContext:
        t = time.monotonic()
        self._audit.log(
            "state_enter",
            {"state": "CLASSIFY_INTENT", "session_id": ctx.session_id},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )

        # Simple heuristic classification (no extra LLM call in v1)
        query_lower = ctx.validated_query.lower()
        # Broader keywords to catch natural language variations
        tool_keywords = ["read", "file", "fetch", "url", "calculate", "compute", "calc", "http", "https"]
        requires_tools = any(kw in query_lower for kw in tool_keywords)

        if "?" in ctx.validated_query:
            intent = "question"
        elif requires_tools:
            intent = "tool_request"
        else:
            intent = "task"

        ctx.intent = intent
        ctx.requires_tools = requires_tools

        duration_ms = int((time.monotonic() - t) * 1000)
        self._memory.record_step(
            session_id=ctx.session_id,
            state_from="INIT",
            state_to="CLASSIFY_INTENT",
            input_data={"query": ctx.validated_query},
            output_data={"intent": intent, "requires_tools": requires_tools},
            duration_ms=duration_ms,
        )
        self._audit.log(
            "state_exit",
            {"state": "CLASSIFY_INTENT", "intent": intent, "requires_tools": requires_tools, "duration_ms": duration_ms},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )
        return ctx

    async def _state_fetch_context(self, ctx: ExecutionContext) -> ExecutionContext:
        t = time.monotonic()
        self._audit.log(
            "state_enter",
            {"state": "FETCH_CONTEXT", "session_id": ctx.session_id},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )

        try:
            memories = self._memory.retrieve_memories(ctx.validated_query, limit=5)
        except ASIMemoryError:
            # Non-critical — continue with empty memories
            memories = []
            logger.warning("Memory retrieval failed; continuing without context")

        ctx.memories = memories

        duration_ms = int((time.monotonic() - t) * 1000)
        self._memory.record_step(
            session_id=ctx.session_id,
            state_from="CLASSIFY_INTENT",
            state_to="FETCH_CONTEXT",
            input_data={"query": ctx.validated_query},
            output_data={"memory_count": len(memories)},
            duration_ms=duration_ms,
        )
        self._audit.log(
            "state_exit",
            {"state": "FETCH_CONTEXT", "memory_count": len(memories), "duration_ms": duration_ms},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )
        return ctx

    async def _state_plan(self, ctx: ExecutionContext) -> ExecutionContext:
        t = time.monotonic()
        self._audit.log(
            "state_enter",
            {"state": "PLAN", "session_id": ctx.session_id},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )

        memory_snippets = ""
        if ctx.memories:
            snippets = []
            for m in ctx.memories[:3]:
                snippets.append(f"- {m.get('content', '')[:200]}")
            memory_snippets = "\n".join(snippets)

        available_tools = self._tools.list_tools()
        tool_list = ", ".join(available_tools) if available_tools else "none"

        ctx.plan_prompt = (
            f"You are a helpful AI assistant.\n\n"
            f"Available tools: {tool_list}\n"
            f"To use a tool, output ONLY the command on a separate line:\n"
            f"- FILE_READ: <absolute_path>\n"
            f"- WEB_FETCH: <url>\n"
            f"- CALC: <expression>\n\n"
            f"EXAMPLES:\n"
            f"User: Read the file C:\\test.txt\n"
            f"Assistant:\nFILE_READ: C:\\test.txt\n\n"
            f"User: Calculate 5 + 5\n"
            f"Assistant:\nCALC: 5 + 5\n\n"
            f"Relevant context from memory:\n{memory_snippets}\n\n"
            f"Intent: {ctx.intent}\n\n"
            f"{self._validator.wrap_for_prompt(ctx.validated_query)}\n\n"
            f"Provide a clear, accurate response. If you need to read a file, USE THE TOOL."
        )

        duration_ms = int((time.monotonic() - t) * 1000)
        self._memory.record_step(
            session_id=ctx.session_id,
            state_from="FETCH_CONTEXT",
            state_to="PLAN",
            input_data={"intent": ctx.intent, "memory_count": len(ctx.memories)},
            output_data={"plan_prompt_len": len(ctx.plan_prompt)},
            duration_ms=duration_ms,
        )
        self._audit.log(
            "state_exit",
            {"state": "PLAN", "plan_prompt_len": len(ctx.plan_prompt), "duration_ms": duration_ms},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )
        return ctx

    async def _state_model_call(self, ctx: ExecutionContext) -> ExecutionContext:
        t = time.monotonic()
        self._audit.log(
            "state_enter",
            {"state": "MODEL_CALL", "session_id": ctx.session_id},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )

        try:
            model_response = await self._router.call(ctx.plan_prompt)
        except (ASIAllProvidersFailedError, ASIModelError):
            raise

        ctx.model_response = model_response

        self._memory.record_model_call(
            session_id=ctx.session_id,
            provider=model_response.provider,
            model_name=model_response.model_name,
            prompt_text=ctx.plan_prompt,
            response_text=model_response.content,
            duration_ms=model_response.duration_ms,
        )

        duration_ms = int((time.monotonic() - t) * 1000)
        self._memory.record_step(
            session_id=ctx.session_id,
            state_from="PLAN",
            state_to="MODEL_CALL",
            input_data={"prompt_hash": model_response.prompt_hash},
            output_data={"response_hash": model_response.response_hash, "provider": model_response.provider},
            duration_ms=duration_ms,
        )
        self._audit.log(
            "state_exit",
            {
                "state": "MODEL_CALL",
                "provider": model_response.provider,
                "model": model_response.model_name,
                "duration_ms": model_response.duration_ms,
            },
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )
        return ctx

    async def _state_tool_call(self, ctx: ExecutionContext) -> ExecutionContext:
        """
        In v1, tool calls are parsed from model response text via simple keyword detection.
        Full tool-use JSON protocol is a v2 feature.
        """
        t = time.monotonic()
        self._audit.log(
            "state_enter",
            {"state": "TOOL_CALL", "session_id": ctx.session_id},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )

        results: List[Dict[str, Any]] = []

        # ------------------------------------------------------------------
        # Tool Dispatch: Regex based (Robust)
        # ------------------------------------------------------------------
        import re
        response_text = ctx.model_response.content if ctx.model_response else ""
        available = self._tools.list_tools()
        
        # Regex patterns
        # Matches: FILE_READ: <path>  (case insensitive prefix, capture path)
        file_pat = re.compile(r"FILE_READ:\s*(.+)", re.IGNORECASE)
        # Matches: WEB_FETCH: <url>
        web_pat = re.compile(r"WEB_FETCH:\s*(.+)", re.IGNORECASE)
        # Matches: CALC: <expr>
        calc_pat = re.compile(r"CALC:\s*(.+)", re.IGNORECASE)

        lines = response_text.splitlines()
        for line in lines:
            line = line.strip()
            
            # FILE_READ
            if "file_read" in available:
                m = file_pat.search(line)
                if m:
                    path = m.group(1).strip()
                    try:
                        tool_result = await self._tools.execute(
                            "file_read",
                            {"path": path},
                            session_id=ctx.session_id,
                            execution_id=ctx.execution_id,
                            trace_id=ctx.trace_id,
                        )
                        results.append(tool_result.model_dump())
                        self._memory.record_tool_call(
                            session_id=ctx.session_id,
                            tool_name="file_read",
                            input_data={"path": path},
                            output_data={"size": tool_result.output.get("size_bytes", 0)},
                            success=True,
                            error_message=None,
                            duration_ms=tool_result.duration_ms,
                        )
                    except ASIToolError as exc:
                        logger.warning("File tool call failed: %s", exc)
                        self._memory.record_tool_call(
                            session_id=ctx.session_id,
                            tool_name="file_read",
                            input_data={"path": path},
                            output_data={},
                            success=False,
                            error_message=str(exc),
                            duration_ms=0,
                        )
                    continue # One tool per line preferred

            # WEB_FETCH
            if "web_fetch" in available:
                m = web_pat.search(line)
                if m:
                    url = m.group(1).strip()
                    try:
                        tool_result = await self._tools.execute(
                            "web_fetch",
                            {"url": url},
                            session_id=ctx.session_id,
                            execution_id=ctx.execution_id,
                            trace_id=ctx.trace_id,
                        )
                        results.append(tool_result.model_dump())
                        self._memory.record_tool_call(
                            session_id=ctx.session_id,
                            tool_name="web_fetch",
                            input_data={"url": url},
                            output_data={"status": tool_result.output.get("status", 0)},
                            success=True,
                            error_message=None,
                            duration_ms=tool_result.duration_ms,
                        )
                    except ASIToolError as exc:
                        logger.warning("Web tool call failed: %s", exc)
                        self._memory.record_tool_call(
                            session_id=ctx.session_id,
                            tool_name="web_fetch",
                            input_data={"url": url},
                            output_data={},
                            success=False,
                            error_message=str(exc),
                            duration_ms=0,
                        )
                    continue

            # CALC
            if "calc" in available:
                m = calc_pat.search(line)
                expr = m.group(1).strip() if m else None
                
                # Fallback: legacy query extraction if no explicit command
                if not expr and any(kw in ctx.validated_query.lower() for kw in ["calculate", "compute", "+", "*"]):
                     expr_match = re.search(r"[\d\s\+\-\*\/\(\)\.]+", ctx.validated_query)
                     if expr_match and len(expr_match.group(0).strip()) > 3:
                         expr = expr_match.group(0).strip()
                
                if expr:
                    try:
                        tool_result = await self._tools.execute(
                            "calc",
                            {"expression": expr},
                            session_id=ctx.session_id,
                            execution_id=ctx.execution_id,
                            trace_id=ctx.trace_id,
                        )
                        results.append(tool_result.model_dump())
                        self._memory.record_tool_call(
                            session_id=ctx.session_id,
                            tool_name="calc",
                            input_data={"expression": expr},
                            output_data=tool_result.output,
                            success=True,
                            error_message=None,
                            duration_ms=tool_result.duration_ms,
                        )
                    except ASIToolError as exc:
                        logger.warning("Calc tool call failed: %s", exc)



        ctx.tool_results = results

        duration_ms = int((time.monotonic() - t) * 1000)
        self._memory.record_step(
            session_id=ctx.session_id,
            state_from="MODEL_CALL",
            state_to="TOOL_CALL",
            input_data={"tool_count": len(available)},
            output_data={"results_count": len(results)},
            duration_ms=duration_ms,
        )
        self._audit.log(
            "state_exit",
            {"state": "TOOL_CALL", "results_count": len(results), "duration_ms": duration_ms},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )
        return ctx

    async def _state_validate_output(self, ctx: ExecutionContext) -> ExecutionContext:
        t = time.monotonic()
        self._audit.log(
            "state_enter",
            {"state": "VALIDATE_OUTPUT", "session_id": ctx.session_id},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )

        content = ctx.model_response.content if ctx.model_response else ""

        if not content or not content.strip():
            raise ASIValidationError("Model returned empty response")

        # Augment with tool results if available
        if ctx.tool_results:
            tool_summary = "\n\nTool results:\n" + json.dumps(ctx.tool_results, indent=2)
            content = content + tool_summary

        ctx.final_content = content

        duration_ms = int((time.monotonic() - t) * 1000)
        self._memory.record_step(
            session_id=ctx.session_id,
            state_from="TOOL_CALL" if ctx.tool_results else "MODEL_CALL",
            state_to="VALIDATE_OUTPUT",
            input_data={"content_len": len(content)},
            output_data={"valid": True},
            duration_ms=duration_ms,
        )
        self._audit.log(
            "state_exit",
            {"state": "VALIDATE_OUTPUT", "content_len": len(content), "duration_ms": duration_ms},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )
        return ctx

    async def _state_store_memory(self, ctx: ExecutionContext) -> ExecutionContext:
        t = time.monotonic()
        self._audit.log(
            "state_enter",
            {"state": "STORE_MEMORY", "session_id": ctx.session_id},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )

        memory_content = f"Q: {ctx.validated_query}\nA: {ctx.final_content[:500]}"
        try:
            memory_id = self._memory.store_memory(
                session_id=ctx.session_id,
                content=memory_content,
                importance=0.7,
            )
        except ASIMemoryError:
            raise

        ctx.memory_id = memory_id

        duration_ms = int((time.monotonic() - t) * 1000)
        self._memory.record_step(
            session_id=ctx.session_id,
            state_from="VALIDATE_OUTPUT",
            state_to="STORE_MEMORY",
            input_data={"content_len": len(memory_content)},
            output_data={"memory_id": memory_id},
            duration_ms=duration_ms,
        )
        self._audit.log(
            "state_exit",
            {"state": "STORE_MEMORY", "memory_id": memory_id, "duration_ms": duration_ms},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )
        return ctx

    async def _state_finalize(self, ctx: ExecutionContext) -> FinalizeOutput:
        t = time.monotonic()
        self._audit.log(
            "state_enter",
            {"state": "FINALIZE", "session_id": ctx.session_id},
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )

        total_ms = int((time.monotonic() - ctx.start_time) * 1000)
        final_hash = _sha256(ctx.final_content)

        self._memory.finalize_session(
            session_id=ctx.session_id,
            status="success",
            final_hash=final_hash,
        )

        result = FinalizeOutput(
            response=ctx.final_content,
            session_id=ctx.session_id,
            execution_id=ctx.execution_id,
            provider=ctx.model_response.provider if ctx.model_response else "none",
            model_name=ctx.model_response.model_name if ctx.model_response else "none",
            prompt_hash=ctx.model_response.prompt_hash if ctx.model_response else "",
            response_hash=ctx.model_response.response_hash if ctx.model_response else "",
            duration_ms=total_ms,
        )

        self._audit.log(
            "state_exit",
            {
                "state": "FINALIZE",
                "session_id": ctx.session_id,
                "total_ms": total_ms,
                "provider": result.provider,
            },
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )
        return result

    async def _state_error(self, ctx: ExecutionContext) -> ErrorOutput:
        error_type = type(ctx.error).__name__ if ctx.error else "UnknownError"
        error_msg = str(ctx.error) if ctx.error else "Unknown error"

        # Log to audit (full detail) — never print stack trace to user
        self._audit.log(
            "state_error",
            {
                "state": "ERROR",
                "failed_state": ctx.failed_state,
                "error_type": error_type,
                "error_message": error_msg,
                "session_id": ctx.session_id,
            },
            execution_id=ctx.execution_id,
            trace_id=ctx.trace_id,
        )

        logger.error("Kernel error in state %s: %s: %s", ctx.failed_state, error_type, error_msg)

        try:
            self._memory.finalize_session(
                session_id=ctx.session_id,
                status="error",
                error_message=f"{error_type}: {error_msg}",
            )
        except ASIMemoryError:
            pass  # best-effort

        return ErrorOutput(
            error_type=error_type,
            error_message=error_msg,
            failed_state=ctx.failed_state,
            session_id=ctx.session_id,
            execution_id=ctx.execution_id,
        )

    # ------------------------------------------------------------------
    # State machine guard
    # ------------------------------------------------------------------

    def _transition(self, from_state: KernelState, to_state: KernelState) -> None:
        allowed = _VALID_TRANSITIONS.get(from_state, [])
        if to_state not in allowed:
            raise ASIStateError(
                f"Invalid state transition: {from_state.value} → {to_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Return a health summary of all kernel components."""
        import subprocess

        # Python version check (only safe subprocess use)
        try:
            proc = subprocess.run(
                ["python", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            python_version = proc.stdout.strip() or proc.stderr.strip()
        except Exception:
            python_version = "unknown"

        return {
            "status": "ok",
            "python": python_version,
            "circuit_breakers": self._router.circuit_status(),
            "db_path": self._config.db_path,
            "preferred_provider": self._config.preferred_provider,
            "available_providers": self._config.available_providers(),
            "registered_tools": self._tools.list_tools(),
        }
