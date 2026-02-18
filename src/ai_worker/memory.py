"""
ASI v8 Memory Layer
Full SQLite schema, explicit transactions, TF-IDF keyword retrieval.
"""

import hashlib
import json
import logging
import math
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import ASIMemoryError
from .security import SecretRedactionFilter

logger = logging.getLogger(__name__)
logger.addFilter(SecretRedactionFilter())

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    execution_id  TEXT,
    trace_id      TEXT,
    created_at    TEXT,
    status        TEXT,
    query_hash    TEXT,
    final_hash    TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    step_id      TEXT PRIMARY KEY,
    session_id   TEXT REFERENCES sessions(session_id),
    state_from   TEXT,
    state_to     TEXT,
    timestamp    TEXT,
    input_hash   TEXT,
    output_hash  TEXT,
    duration_ms  INTEGER
);

CREATE TABLE IF NOT EXISTS model_calls (
    call_id       TEXT PRIMARY KEY,
    session_id    TEXT REFERENCES sessions(session_id),
    provider      TEXT,
    model_name    TEXT,
    prompt_text   TEXT,
    response_text TEXT,
    prompt_hash   TEXT,
    response_hash TEXT,
    duration_ms   INTEGER,
    timestamp     TEXT
);

CREATE TABLE IF NOT EXISTS tool_calls (
    tool_call_id  TEXT PRIMARY KEY,
    session_id    TEXT REFERENCES sessions(session_id),
    tool_name     TEXT,
    input_json    TEXT,
    output_json   TEXT,
    success       INTEGER,
    error_message TEXT,
    duration_ms   INTEGER,
    timestamp     TEXT
);

CREATE TABLE IF NOT EXISTS episodic_memory (
    memory_id     TEXT PRIMARY KEY,
    session_id    TEXT REFERENCES sessions(session_id),
    content       TEXT,
    keywords      TEXT,
    importance    REAL,
    created_at    TEXT,
    access_count  INTEGER DEFAULT 0,
    last_accessed TEXT
);
"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> List[str]:
    """Simple tokeniser: lowercase, split on non-alphanumeric."""
    return [t for t in re.split(r"\W+", text.lower()) if len(t) > 1]


class MemoryLayer:
    """
    Persistent storage for ASI v8.
    All writes use explicit BEGIN IMMEDIATE / COMMIT / ROLLBACK transactions.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(_DDL)
                conn.commit()
        except sqlite3.Error as exc:
            raise ASIMemoryError(f"DB init failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_id: str,
        execution_id: str,
        trace_id: str,
        query_hash: str,
    ) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO sessions
                        (session_id, execution_id, trace_id, created_at, status, query_hash)
                    VALUES (?, ?, ?, ?, 'running', ?)
                    """,
                    (session_id, execution_id, trace_id, _now_iso(), query_hash),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise ASIMemoryError(f"create_session failed: {exc}") from exc

    def finalize_session(
        self,
        session_id: str,
        status: str,
        final_hash: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    UPDATE sessions
                    SET status=?, final_hash=?, error_message=?
                    WHERE session_id=?
                    """,
                    (status, final_hash, error_message, session_id),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise ASIMemoryError(f"finalize_session failed: {exc}") from exc

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM sessions WHERE session_id=?", (session_id,)
                ).fetchone()
                return dict(row) if row else None
        except sqlite3.Error as exc:
            raise ASIMemoryError(f"get_session failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def record_step(
        self,
        session_id: str,
        state_from: str,
        state_to: str,
        input_data: Any,
        output_data: Any,
        duration_ms: int,
    ) -> str:
        step_id = secrets.token_hex(16)
        input_hash = _sha256(json.dumps(input_data, sort_keys=True, default=str))
        output_hash = _sha256(json.dumps(output_data, sort_keys=True, default=str))
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO steps
                        (step_id, session_id, state_from, state_to, timestamp, input_hash, output_hash, duration_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (step_id, session_id, state_from, state_to, _now_iso(), input_hash, output_hash, duration_ms),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise ASIMemoryError(f"record_step failed: {exc}") from exc
        return step_id

    def get_steps(self, session_id: str) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM steps WHERE session_id=? ORDER BY rowid ASC",
                    (session_id,),
                ).fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            raise ASIMemoryError(f"get_steps failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Model calls
    # ------------------------------------------------------------------

    def record_model_call(
        self,
        session_id: str,
        provider: str,
        model_name: str,
        prompt_text: str,
        response_text: str,
        duration_ms: int,
    ) -> str:
        call_id = secrets.token_hex(16)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO model_calls
                        (call_id, session_id, provider, model_name, prompt_text, response_text,
                         prompt_hash, response_hash, duration_ms, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        call_id,
                        session_id,
                        provider,
                        model_name,
                        prompt_text,
                        response_text,
                        _sha256(prompt_text),
                        _sha256(response_text),
                        duration_ms,
                        _now_iso(),
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise ASIMemoryError(f"record_model_call failed: {exc}") from exc
        return call_id

    def get_model_calls(self, session_id: str) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM model_calls WHERE session_id=? ORDER BY rowid ASC",
                    (session_id,),
                ).fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            raise ASIMemoryError(f"get_model_calls failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------

    def record_tool_call(
        self,
        session_id: str,
        tool_name: str,
        input_data: Any,
        output_data: Any,
        success: bool,
        error_message: Optional[str],
        duration_ms: int,
    ) -> str:
        tool_call_id = secrets.token_hex(16)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO tool_calls
                        (tool_call_id, session_id, tool_name, input_json, output_json,
                         success, error_message, duration_ms, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tool_call_id,
                        session_id,
                        tool_name,
                        json.dumps(input_data, default=str),
                        json.dumps(output_data, default=str),
                        1 if success else 0,
                        error_message,
                        duration_ms,
                        _now_iso(),
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise ASIMemoryError(f"record_tool_call failed: {exc}") from exc
        return tool_call_id

    def get_tool_calls(self, session_id: str) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM tool_calls WHERE session_id=? ORDER BY rowid ASC",
                    (session_id,),
                ).fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            raise ASIMemoryError(f"get_tool_calls failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Episodic memory with TF-IDF retrieval
    # ------------------------------------------------------------------

    def store_memory(
        self,
        session_id: str,
        content: str,
        importance: float = 0.5,
    ) -> str:
        memory_id = secrets.token_hex(16)
        keywords = " ".join(set(_tokenize(content)))
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO episodic_memory
                        (memory_id, session_id, content, keywords, importance, created_at,
                         access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (memory_id, session_id, content, keywords, importance, _now_iso(), _now_iso()),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise ASIMemoryError(f"store_memory failed: {exc}") from exc
        return memory_id

    def retrieve_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """TF-IDF keyword matching for relevant episodic memories."""
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM episodic_memory ORDER BY created_at DESC LIMIT 200"
                ).fetchall()
        except sqlite3.Error as exc:
            raise ASIMemoryError(f"retrieve_memories failed: {exc}") from exc

        if not rows:
            return []

        # Build corpus for IDF
        corpus = [set(_tokenize(r["content"])) for r in rows]
        n_docs = len(corpus)

        def idf(term: str) -> float:
            df = sum(1 for doc in corpus if term in doc)
            return math.log((n_docs + 1) / (df + 1)) + 1.0

        scored = []
        for i, row in enumerate(rows):
            doc_tokens = corpus[i]
            doc_len = len(_tokenize(row["content"])) or 1
            score = 0.0
            for token in query_tokens:
                if token in doc_tokens:
                    tf = doc_tokens.count(token) / doc_len if hasattr(doc_tokens, "count") else 1.0 / doc_len
                    score += tf * idf(token)
            score *= (0.5 + 0.5 * row["importance"])
            if score > 0:
                scored.append((score, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in scored[:limit]]

        # Update access stats
        if results:
            ids = [r["memory_id"] for r in results]
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    for mid in ids:
                        conn.execute(
                            """
                            UPDATE episodic_memory
                            SET access_count = access_count + 1, last_accessed = ?
                            WHERE memory_id = ?
                            """,
                            (_now_iso(), mid),
                        )
                    conn.commit()
            except sqlite3.Error:
                # Non-critical access stat update — don't raise
                pass

        return results

    # ------------------------------------------------------------------
    # Full session dump (for inspect / replay)
    # ------------------------------------------------------------------

    def get_full_session(self, session_id: str) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if not session:
            raise ASIMemoryError(f"Session not found: {session_id}")
        return {
            "session": session,
            "steps": self.get_steps(session_id),
            "model_calls": self.get_model_calls(session_id),
            "tool_calls": self.get_tool_calls(session_id),
        }
