"""
error_logger.py -- Centralised DB error logging
================================================
Call log_error(...) from any module to persist a failed API call, tool call,
LLM error, or any system-level fault to the error_logs PostgreSQL table.

Design goals:
  - Fire-and-forget: never raises; a logging failure must not crash the caller.
  - DB-safe: if error_type starts with "db_" the write is skipped (prevents
    infinite recursion when the DB itself is the problem).
  - PII-safe: request_payload is sanitised before storage.
  - Retention: rows older than cfg.ERROR_LOG_RETENTION_DAYS are purged at
    startup -- set to -1 to keep records forever.

Standardised error_type values
  tool_http_error           Non-200 HTTP response from a tool call
  tool_connection_error     httpx.ConnectError -- server unreachable
  tool_timeout_error        httpx.TimeoutException or async job timeout
  tool_not_found            Tool ID missing from registry
  async_tool_error          Async job background failure
  llm_generation_error      model.generate() crash / OOM
  llm_queue_full            LLM request queue size exceeded
  llm_request_timeout       Request waited too long in LLM queue
  llm_transition_error      Transition evaluation returned unexpected result
  db_connection_error       Cannot connect to PostgreSQL (console-only)
  db_query_error            SQL query failed (console-only)
  session_not_found         session_id missing from cache + DB
  agent_not_found           agent_id missing
  cbs_customer_not_found    CBS lookup returned no customer
  call_log_write_error      _write_call_log() failure
  validation_error          FastAPI 422 / Pydantic validation failure
  variable_extraction_failure  LLM failed to extract required tool variables
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, Optional

from config import cfg
from logger import get_logger

logger = get_logger(__name__)


# -- Table DDL ----------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS error_logs (
    id              BIGSERIAL PRIMARY KEY,
    error_id        TEXT        NOT NULL UNIQUE,
    error_type      TEXT        NOT NULL,
    severity        TEXT        NOT NULL DEFAULT 'ERROR',
    session_id      TEXT,
    agent_id        TEXT,
    tool_id         TEXT,
    tool_name       TEXT,
    endpoint        TEXT,
    http_method     TEXT,
    http_status     INT,
    error_message   TEXT        NOT NULL,
    error_detail    JSONB,
    request_payload JSONB,
    source_module   TEXT,
    source_function TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_error_logs_created_at  ON error_logs (created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_error_logs_session_id  ON error_logs (session_id)",
    "CREATE INDEX IF NOT EXISTS idx_error_logs_error_type  ON error_logs (error_type)",
    "CREATE INDEX IF NOT EXISTS idx_error_logs_tool_id     ON error_logs (tool_id)",
]


def init_error_logs_table() -> None:
    """
    Create the error_logs table (and indexes) if they do not exist yet.
    Called once from database.init_db() at startup.
    Also purges rows older than cfg.ERROR_LOG_RETENTION_DAYS (unless -1).
    """
    from database import execute_write  # late import -- avoids circular at module load

    try:
        execute_write(CREATE_TABLE_SQL)
        for idx_sql in CREATE_INDEXES_SQL:
            execute_write(idx_sql)
        logger.info("[ErrorLogger] error_logs table ready")
    except Exception as e:
        logger.error(f"[ErrorLogger] Failed to create error_logs table: {e}")
        return

    # -- Retention purge ------------------------------------------------------
    days = cfg.ERROR_LOG_RETENTION_DAYS
    if days != -1:
        try:
            execute_write(
                "DELETE FROM error_logs WHERE created_at < NOW() - (%s || ' days')::INTERVAL",
                (str(days),),
            )
            logger.info(f"[ErrorLogger] Purged error_logs rows older than {days} days")
        except Exception as e:
            logger.warning(f"[ErrorLogger] Retention purge failed: {e}")


# -- PII Sanitiser ------------------------------------------------------------

def _sanitise_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Recursively sanitise a request payload before storing it:
      - Mobile / phone numbers (10-digit) -> masked to last 4 digits (e.g. ******7890)
      - Long base64 strings (>200 chars, no spaces) -> truncated with [base64 redacted] marker
      - Sensitive key names (password, token, secret, auth, api_key, otp) -> [REDACTED]
    """
    if payload is None:
        return None

    _SENSITIVE_KEYS = re.compile(r"^(password|token|secret|auth|api.?key|otp)$", re.I)
    _MOBILE_RE = re.compile(r"\b([6-9]\d{5})(\d{4})\b")

    def _clean(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {k: _clean(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [_clean(v) for v in value]
        if isinstance(value, str):
            if _SENSITIVE_KEYS.match(key):
                return "[REDACTED]"
            value = _MOBILE_RE.sub(lambda m: "******" + m.group(2), value)
            if len(value) > 200 and " " not in value and "\n" not in value:
                return value[:20] + "...[base64 redacted]..." + value[-10:]
        return value

    return _clean(payload)


# -- Core Logger --------------------------------------------------------------

def log_error(
    error_type: str,
    error_message: str,
    *,
    severity: str = "ERROR",
    session_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    tool_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    http_method: Optional[str] = None,
    http_status: Optional[int] = None,
    error_detail: Optional[Dict[str, Any]] = None,
    request_payload: Optional[Dict[str, Any]] = None,
    source_module: Optional[str] = None,
    source_function: Optional[str] = None,
) -> None:
    """
    Persist an error record to the error_logs table.

    Never raises -- a DB failure here falls back to console-only logging.
    DB errors (error_type starting with "db_") are always console-only to
    prevent infinite recursion.

    Args:
        error_type:       Standardised string (see module docstring).
        error_message:    Human-readable description of what went wrong.
        severity:         DEBUG | INFO | WARNING | ERROR | CRITICAL
        session_id:       Active session (if applicable).
        agent_id:         Agent involved (if applicable).
        tool_id:          Tool that failed (if applicable).
        tool_name:        Display name of the tool.
        endpoint:         URL that was called (HTTP errors).
        http_method:      HTTP verb used (GET / POST / ...).
        http_status:      HTTP status code returned (0 for connection errors).
        error_detail:     Dict with extra context (stack trace, raw response, ...).
        request_payload:  Request body sent -- will be PII-sanitised before storage.
        source_module:    Python module name (e.g. "tool_registry").
        source_function:  Function name where the error occurred.
    """
    # Always log to console regardless of DB outcome
    _log_level = severity if severity in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL") else "ERROR"
    logger.log(
        _log_level,
        f"[ErrorLog] type={error_type} | {error_message}"
        + (f" | session={session_id}" if session_id else "")
        + (f" | tool={tool_id}" if tool_id else "")
        + (f" | status={http_status}" if http_status is not None else ""),
    )

    # Skip DB write for DB-layer errors to prevent infinite recursion
    if error_type.startswith("db_"):
        return

    try:
        from database import execute_write  # late import

        error_id = f"err_{uuid.uuid4().hex[:8]}"
        clean_payload = _sanitise_payload(request_payload)

        execute_write(
            """INSERT INTO error_logs (
                   error_id, error_type, severity,
                   session_id, agent_id,
                   tool_id, tool_name,
                   endpoint, http_method, http_status,
                   error_message, error_detail, request_payload,
                   source_module, source_function
               ) VALUES (
                   %s, %s, %s,
                   %s, %s,
                   %s, %s,
                   %s, %s, %s,
                   %s, %s, %s,
                   %s, %s
               )""",
            (
                error_id,
                error_type,
                severity,
                session_id,
                agent_id,
                tool_id,
                tool_name,
                endpoint,
                http_method,
                http_status,
                error_message,
                json.dumps(error_detail) if error_detail else None,
                json.dumps(clean_payload) if clean_payload else None,
                source_module,
                source_function,
            ),
        )
    except Exception as exc:
        # Absolute last resort -- console only, never re-raise
        logger.error(f"[ErrorLogger] Failed to write to error_logs: {exc}")