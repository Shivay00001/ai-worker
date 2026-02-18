"""
ASI v8 Audit Logger
Tamper-evident hash chain, JSON log entries, secret redaction, SQLite persistence.
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import ASIMemoryError
from .security import SecretRedactionFilter

logger = logging.getLogger(__name__)
logger.addFilter(SecretRedactionFilter())


class AuditLogger:
    """
    Writes structured JSON audit events to both a file and SQLite.
    Each entry includes a sha256 hash of itself and the previous entry's hash,
    forming a tamper-evident chain.
    """

    _SECRET_PATTERNS = [
        re.compile(r"Bearer\s+sk-[A-Za-z0-9\-_]+", re.IGNORECASE),
        re.compile(r"Bearer\s+gsk_[A-Za-z0-9\-_]+", re.IGNORECASE),
        re.compile(r"Bearer\s+[A-Za-z0-9\-_\.]{20,}", re.IGNORECASE),
        re.compile(r"x-api-key:\s*[A-Za-z0-9\-_\.]{16,}", re.IGNORECASE),
    ]
    _SECRET_REPLACEMENTS = [
        "Bearer sk-[REDACTED]",
        "Bearer gsk_[REDACTED]",
        "Bearer [REDACTED]",
        "x-api-key: [REDACTED]",
    ]

    def __init__(self, audit_dir: str, db_path: str):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._log_file = self.audit_dir / "audit.jsonl"
        self._prev_hash: str = "0" * 64  # genesis hash
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_log (
                        entry_id   TEXT PRIMARY KEY,
                        timestamp  TEXT NOT NULL,
                        execution_id TEXT,
                        trace_id   TEXT,
                        event      TEXT NOT NULL,
                        data_json  TEXT,
                        prev_hash  TEXT,
                        entry_hash TEXT
                    )
                """)
                conn.commit()

                # Restore last hash from DB for chain continuity
                row = conn.execute(
                    "SELECT entry_hash FROM audit_log ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
                if row:
                    self._prev_hash = row[0]
        except sqlite3.Error as exc:
            raise ASIMemoryError(f"Audit DB init failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        event: str,
        data: Dict[str, Any],
        execution_id: str = "",
        trace_id: str = "",
    ) -> str:
        """
        Write an audit event. Returns the entry_hash of this event.
        Raises ASIMemoryError on DB failure.
        """
        data = self._redact_dict(data)

        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_id": execution_id,
            "trace_id": trace_id,
            "event": event,
            "data": data,
            "prev_hash": self._prev_hash,
        }

        entry_hash = hashlib.sha256(
            json.dumps(entry, sort_keys=True).encode("utf-8")
        ).hexdigest()
        entry["entry_hash"] = entry_hash

        # Write to JSONL file
        try:
            with self._log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            # File write failure is non-fatal but logged
            logger.error("Audit file write error: %s", exc)

        # Write to SQLite
        import secrets as _secrets
        entry_id = _secrets.token_hex(16)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO audit_log
                        (entry_id, timestamp, execution_id, trace_id, event, data_json, prev_hash, entry_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry_id,
                        entry["timestamp"],
                        execution_id,
                        trace_id,
                        event,
                        json.dumps(data),
                        self._prev_hash,
                        entry_hash,
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise ASIMemoryError(f"Audit DB write failed: {exc}") from exc

        self._prev_hash = entry_hash
        return entry_hash

    def export_session(self, session_id: str) -> list:
        """Export all audit entries associated with a session_id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT timestamp, execution_id, trace_id, event, data_json, prev_hash, entry_hash
                    FROM audit_log
                    WHERE execution_id = ? OR data_json LIKE ?
                    ORDER BY rowid ASC
                    """,
                    (session_id, f'%"{session_id}"%'),
                ).fetchall()
        except sqlite3.Error as exc:
            raise ASIMemoryError(f"Audit DB read failed: {exc}") from exc

        return [
            {
                "timestamp": r[0],
                "execution_id": r[1],
                "trace_id": r[2],
                "event": r[3],
                "data": json.loads(r[4]) if r[4] else {},
                "prev_hash": r[5],
                "entry_hash": r[6],
            }
            for r in rows
        ]

    def verify_chain(self, session_id: str) -> Dict[str, Any]:
        """
        Verify hash chain integrity for entries belonging to a session.
        Returns dict with keys: valid (bool), broken_at (int|None), total (int).
        """
        entries = self.export_session(session_id)
        for i, entry in enumerate(entries):
            expected_hash = hashlib.sha256(
                json.dumps(
                    {k: v for k, v in entry.items() if k != "entry_hash"},
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if expected_hash != entry["entry_hash"]:
                return {"valid": False, "broken_at": i, "total": len(entries)}
        return {"valid": True, "broken_at": None, "total": len(entries)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _redact(self, text: str) -> str:
        for pattern, replacement in zip(self._SECRET_PATTERNS, self._SECRET_REPLACEMENTS):
            text = pattern.sub(replacement, text)
        return text

    def _redact_dict(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._redact_dict(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._redact_dict(v) for v in data]
        if isinstance(data, str):
            return self._redact(data)
        return data
