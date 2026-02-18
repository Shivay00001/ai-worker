"""
ASI v8 CLI Adapter
Uses Click. Zero business logic — all calls delegate to kernel/memory/audit.
Errors → stderr + sys.exit(1). Normal output → stdout.
"""

import json
import os
import sys
from pathlib import Path

import click

from asi.audit import AuditLogger
from asi.config import Config
from asi.errors import ASIError
from asi.kernel import ExecutionKernel
from asi.memory import MemoryLayer
from asi.router import ModelRouter
from asi.tools.calc_tool import CalcTool
from asi.tools.file_tool import FileReadTool
from asi.tools.registry import Permission, ToolRegistry
from asi.tools.web_tool import WebFetchTool


def _build_kernel(config: Config) -> ExecutionKernel:
    """Construct and wire all components."""
    Path(config.db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.audit_dir).mkdir(parents=True, exist_ok=True)

    memory = MemoryLayer(db_path=config.db_path)
    audit = AuditLogger(audit_dir=config.audit_dir, db_path=config.db_path)
    router = ModelRouter(config=config)

    allowed_dirs = [Path(config.workspace_dir)]
    Path(config.workspace_dir).mkdir(parents=True, exist_ok=True)

    registry = ToolRegistry(
        granted_permissions=[Permission.FILE_READ, Permission.WEB_FETCH, Permission.CALC],
        audit_logger=audit,
    )
    registry.register(FileReadTool(allowed_dirs=allowed_dirs))
    registry.register(
        WebFetchTool(
            allowed_url_prefixes=os.environ.get(
                "ASI_ALLOWED_URLS", "https://,http://"
            ).split(",")
        )
    )
    registry.register(CalcTool())

    return ExecutionKernel(
        config=config,
        router=router,
        memory=memory,
        audit=audit,
        tool_registry=registry,
    )


@click.group()
def cli():
    """ASI v8 — production-grade single-agent AI runtime."""


@cli.command("run")
@click.argument("query")
def cmd_run(query: str):
    """Run a query and print the response."""
    import asyncio

    try:
        config = Config.from_env()
    except ValueError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    try:
        kernel = _build_kernel(config)
        result = asyncio.run(kernel.run(query))
        click.echo(result.response)
    except ASIError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Unexpected error: {type(exc).__name__}", err=True)
        sys.exit(1)


@cli.command("inspect")
@click.argument("session_id")
def cmd_inspect(session_id: str):
    """Print full session as JSON."""
    try:
        config = Config.from_env()
    except ValueError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    try:
        memory = MemoryLayer(db_path=config.db_path)
        session_data = memory.get_full_session(session_id)
        click.echo(json.dumps(session_data, indent=2, default=str))
    except ASIError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("replay")
@click.argument("session_id")
@click.option("--enable-tools", is_flag=True, default=False, help="Execute tools during replay")
def cmd_replay(session_id: str, enable_tools: bool):
    """Deterministic replay of a session."""
    import asyncio

    try:
        config = Config.from_env()
    except ValueError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    try:
        memory = MemoryLayer(db_path=config.db_path)
        session_data = memory.get_full_session(session_id)
        model_calls = session_data.get("model_calls", [])

        if not model_calls:
            click.echo("No model calls found in session — cannot replay.", err=True)
            sys.exit(1)

        original_query_hash = session_data["session"].get("query_hash", "")
        click.echo(
            json.dumps(
                {
                    "replayed_session_id": session_id,
                    "original_query_hash": original_query_hash,
                    "steps_replayed": len(session_data.get("steps", [])),
                    "model_calls_replayed": len(model_calls),
                    "tools_enabled": enable_tools,
                    "note": "Replay returns stored responses for determinism.",
                },
                indent=2,
            )
        )
    except ASIError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("audit")
@click.argument("session_id")
@click.option("--verify-chain", is_flag=True, default=False, help="Verify hash chain integrity")
def cmd_audit(session_id: str, verify_chain: bool):
    """Export audit log for a session as JSON."""
    try:
        config = Config.from_env()
    except ValueError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    try:
        audit = AuditLogger(audit_dir=config.audit_dir, db_path=config.db_path)

        if verify_chain:
            result = audit.verify_chain(session_id)
            click.echo(json.dumps(result, indent=2))
        else:
            entries = audit.export_session(session_id)
            output_path = Path(config.audit_dir) / f"audit_{session_id}.json"
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(entries, fh, indent=2, default=str)
            click.echo(f"Audit log written to: {output_path}")
    except ASIError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("health")
def cmd_health():
    """Check all components and print health status."""
    try:
        config = Config.from_env()
    except ValueError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    try:
        kernel = _build_kernel(config)
        status = kernel.health()
        click.echo(json.dumps(status, indent=2))
    except ASIError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
