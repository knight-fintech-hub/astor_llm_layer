"""
tool_registry.py — Tool/API Registration & Execution
======================================================
Users register external APIs/tools from UI. At runtime, the orchestrator
calls tools as needed by the current node.

Tool Types:
  - api_call: HTTP call to internal/external API
  - webhook: Fire-and-forget HTTP POST
  - internal: Built-in function (e.g., set variable)

Endpoints:
  POST   /api/tools         — Register a tool
  GET    /api/tools         — List all tools
  GET    /api/tools/{id}    — Get tool by ID
  PUT    /api/tools/{id}    — Update tool
  DELETE /api/tools/{id}    — Delete tool
  POST   /api/tools/{id}/execute — Execute a tool with given variables
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from config import cfg
from database import execute_query, execute_one, execute_write
from error_logger import log_error
from logger import get_logger

logger = get_logger(__name__)

tool_router = APIRouter(prefix="/api/tools", tags=["Tool Registry"])


# ── Models ───────────────────────────────────────────────────────────────────

class ToolCreate(BaseModel):
    name: str = Field(..., min_length=1, description="Tool display name")
    description: Optional[str] = Field(None, description="What this tool does")
    type: str = Field("api_call", description="Tool type: api_call, webhook, internal")
    method: str = Field("GET", description="HTTP method: GET, POST, PUT, DELETE")
    endpoint: str = Field(..., description="API endpoint — full URL (https://...) or local path (/api/...)")
    request_body: Dict[str, Any] = Field(default_factory=dict, description="Request body with {Variable} placeholders")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers (e.g., Authorization, Content-Type)")
    response_body: Dict[str, Any] = Field(default_factory=dict, description="Response body template with {Variable} placeholders")
    # Orchestration metadata — never sent to the LLM (only tool_id/name/description
    # are ever placed in a prompt, see decide_transition_and_tool()/auto_tool_call()
    # in orchestrator.py). Routes execution through dispatch_tool() in this module.
    x_execution_mode: Literal["sync", "async"] = Field(
    
        "sync", description="sync: orchestrator awaits the real result. async: dispatched as a background task; a stub is returned immediately."
    )
    x_timeout_ms: int = Field(30000, ge=1000, description="Async only — max wait before the background job is marked failed")
    x_on_complete: Literal["conversation", "next_tool_call"] = Field(
        "conversation", description="Async only — what the agent should do once the real result arrives"
    )
    x_meanwhile: Literal["progress_the_journey", "small_talk"] = Field(
        "progress_the_journey", description="Async only — what the agent does while the job runs in the background"
    )
    next_tool: Optional[str] = Field(None, description="ID of the next tool to call if x_on_complete is next_tool_call")


class ToolUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    method: Optional[str] = None
    endpoint: Optional[str] = None
    
    request_body: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    response_body: Optional[Dict[str, Any]] = None
    x_execution_mode: Optional[Literal["sync", "async"]] = None
    x_timeout_ms: Optional[int] = Field(None, ge=1000)
    x_on_complete: Optional[Literal["conversation", "next_tool_call"]] = None
    x_meanwhile: Optional[Literal["progress_the_journey", "small_talk"]] = None
    next_tool: Optional[str] = None


class ToolExecuteRequest(BaseModel):
    variables: Dict[str, str] = Field(default_factory=dict, description="Current session variables")


# ── Tool Store (PostgreSQL + local in-memory fallback) ───────────────────────

class ToolStore:
    """PostgreSQL-backed tool store with an in-memory local fallback cache.
    
    The local cache is used when USE_DB=false (testing mode): tools loaded from
    agent_config.json are registered via register_local_tool() and are returned
    by get() when no DB row is found.
    """

    def __init__(self):
        # In-memory fallback for local/testing mode (populated by register_local_tool)
        self._local_cache: Dict[str, Dict] = {}

    def register_local_tool(self, tool: Dict) -> None:
        """Register a tool into the in-memory local cache (used when USE_DB=false)."""
        tid = tool.get("tool_id")
        if tid:
            self._local_cache[tid] = tool
            logger.debug(f"[ToolStore] Registered local tool: {tid} ({tool.get('name')})")


    def create(self, **kwargs) -> Dict:
        tool_id = f"tool_{uuid.uuid4().hex[:6]}"
        execute_write(
            """INSERT INTO tools (tool_id, name, description, type, method, endpoint, params, headers, response_mapping,
               x_execution_mode, x_timeout_ms, x_on_complete, x_meanwhile, next_tool)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                tool_id,
                kwargs["name"],
                kwargs.get("description"),
                kwargs.get("type", "api_call"),
                kwargs.get("method", "GET"),
                kwargs["endpoint"],
                json.dumps(kwargs.get("request_body", {})),
                json.dumps(kwargs.get("headers", {})),
                json.dumps(kwargs.get("response_body", {})),
                kwargs.get("x_execution_mode", "sync"),
                kwargs.get("x_timeout_ms", 30000),
                kwargs.get("x_on_complete", "conversation"),
                kwargs.get("x_meanwhile", "progress_the_journey"),
                kwargs.get("next_tool"),
            ),
        )
        logger.info(f"[ToolStore] Created: {tool_id} ({kwargs['name']}) | mode={kwargs.get('x_execution_mode', 'sync')}")
        return self.get(tool_id)

    def get(self, tool_id: str) -> Optional[Dict]:
        # Try DB first (may raise if no DB connection in local mode)
        try:
            row = execute_one("SELECT * FROM tools WHERE tool_id = %s", (tool_id,))
            if row:
                return self._format(row)
        except Exception:
            pass  # DB unavailable in local mode — fall through to cache
        # Fall back to in-memory local cache
        return self._local_cache.get(tool_id)

    def list_all(self) -> List[Dict]:
        try:
            rows = execute_query("SELECT * FROM tools ORDER BY created_at")
            db_tools = [self._format(r) for r in rows]
        except Exception:
            db_tools = []  # DB unavailable in local mode
        # Merge: DB tools take priority; add any local-cache-only tools not in DB
        db_ids = {t["tool_id"] for t in db_tools}
        local_only = [t for tid, t in self._local_cache.items() if tid not in db_ids]
        return db_tools + local_only


    def update(self, tool_id: str, **kwargs) -> Optional[Dict]:
        existing = self.get(tool_id)
        if not existing:
            return None

        updates = []
        values = []
        for key in (
            "name", "description", "type", "method", "endpoint",
            "x_execution_mode", "x_timeout_ms", "x_on_complete", "x_meanwhile", "next_tool"
        ):
            if kwargs.get(key) is not None:
                updates.append(f"{key} = %s")
                values.append(kwargs[key])

        for key, db_key in [("request_body", "params"), ("headers", "headers"), ("response_body", "response_mapping")]:
            if kwargs.get(key) is not None:
                updates.append(f"{db_key} = %s")
                values.append(json.dumps(kwargs[key]))

        if not updates:
            return existing

        values.append(tool_id)
        execute_write(f"UPDATE tools SET {', '.join(updates)} WHERE tool_id = %s", tuple(values))
        logger.info(f"[ToolStore] Updated: {tool_id}")
        return self.get(tool_id)

    def delete(self, tool_id: str) -> bool:
        existing = self.get(tool_id)
        if not existing:
            return False
        execute_write("DELETE FROM tools WHERE tool_id = %s", (tool_id,))
        logger.info(f"[ToolStore] Deleted: {tool_id}")
        return True

    def _format(self, row: Dict) -> Dict:
        if not row:
            return None

        def _parse_json_col(value, default):
            """Parse a JSON column that psycopg2 may return as a raw string."""
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return default
            return value if value is not None else default

        return {
            "tool_id": row["tool_id"],
            "name": row["name"],
            "description": row.get("description"),
            "type": row.get("type", "api_call"),
            "method": row.get("method", "GET"),
            "endpoint": row["endpoint"],
            "request_body": _parse_json_col(row.get("params"), {}),
            "headers": _parse_json_col(row.get("headers"), {}),
            "response_body": _parse_json_col(row.get("response_mapping"), {}),
            "x_execution_mode": row.get("x_execution_mode", "sync"),
            "x_timeout_ms": row.get("x_timeout_ms", 30000),
            "x_on_complete": row.get("x_on_complete", "conversation"),
            "x_meanwhile": row.get("x_meanwhile", "progress_the_journey"),
            "next_tool": row.get("next_tool"),
            "created_at": str(row.get("created_at", "")),
        }


# Initialize store
tool_store = ToolStore()



# ── OTP Helpers ─────────────────────────────────────────────────────────────

def _generate_otp(length: int = 6) -> str:
    """Generate a numeric OTP of given length."""
    return "".join(str(random.randint(0, 9)) for _ in range(length))


# ── Tool Execution Engine ───────────────────────────────
def _resolve_variables(template: str, variables: Dict[str, str]) -> str:
    """Replace {VariableName} and {{VariableName}} placeholders with actual values."""
    def replacer(match):
        var_name = match.group(1)
        is_opt = match.group(2)
        if var_name in variables:
            return str(variables[var_name])
        if is_opt:
            return "" # Optional variable missing
        return match.group(0)

    # Double-brace (Retell-style {{Var}}) first, then single-brace {Var}
    template = re.sub(r"\{\{(\w+)(\??)\}\}", replacer, template)
    template = re.sub(r"\{(\w+)(\??)\}", replacer, template)
    return template


def resolve_request_body(body: Any, variables: Dict[str, str]) -> Any:
    """Recursively resolve {Variable} and {Variable?} placeholders in a dictionary or list."""  
    
    if isinstance(body, dict):
        res = {}
        for k, v in body.items():
            if isinstance(v, str):
                match = re.fullmatch(r"\{\{?(\w+)\?\}?\}", v.strip())
                if match:
                    var_name = match.group(1)
                    if var_name not in variables:
                        continue
            res[k] = resolve_request_body(v, variables)
        return res
    elif isinstance(body, list):
        return [resolve_request_body(v, variables) for v in body]
    elif isinstance(body, str):
        # Find variables, handling both {Var} and {{Var}}
        matches = re.findall(r"\{\{?(\w+)(\??)\}?\}", body)
        if not matches:
            return body
        
        # Full replacement, keep type if possible
        stripped = body.strip()
        if len(matches) == 1:
            var_name, is_opt = matches[0]
            if is_opt:
                if stripped in ("{" + var_name + "?}", "{{" + var_name + "?}}"):
                    if var_name in variables:
                        return variables.get(var_name)
                    return None
            else:
                if stripped in ("{" + var_name + "}", "{{" + var_name + "}}"):
                    return variables.get(var_name, body)
        
        # String replacement for partial matches
        res = body
        for var_name, is_opt in matches:
            if is_opt:
                val = str(variables.get(var_name, ""))
                res = res.replace("{{" + var_name + "?}}", val)
                res = res.replace("{" + var_name + "?}", val)
            else:
                val = str(variables.get(var_name, "{" + var_name + "}"))
                res = res.replace("{{" + var_name + "}}", val)
                res = res.replace("{" + var_name + "}", val)
        return res
    return body


def extract_response_vars(template: Any, actual_response: Any, new_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Recursively extract values from actual_response based on the {Variable} placeholders in template."""
    if new_vars is None:
        new_vars = {}
        
    if isinstance(template, dict) and isinstance(actual_response, dict):
        for k, v in template.items():
            if k in actual_response:
                extract_response_vars(v, actual_response[k], new_vars)
    elif isinstance(template, list) and isinstance(actual_response, list):
        for t, a in zip(template, actual_response):
            extract_response_vars(t, a, new_vars)
    elif isinstance(template, str):
        matches = re.findall(r"\{(\w+)\}", template)
        if matches and template.strip() == "{" + matches[0] + "}":
            new_vars[matches[0]] = actual_response
            
    return new_vars

def _uses_input_image(tool: Dict) -> bool:
    """True if this tool's endpoint or request_body references {InputImage}."""
    def _scan(value) -> bool:
        if isinstance(value, dict):
            return any(_scan(v) for v in value.values())
        if isinstance(value, list):
            return any(_scan(v) for v in value)
        if isinstance(value, str):
            return bool(re.search(r"\{\{?Input(?:Image|File)\}?\}", value))
        return False
    return _scan(tool.get("endpoint", "")) or _scan(tool.get("request_body", {}))


# Tools that consume {InputImage}/{InputFile} but whose receiving API wants
# the base64 string delivered as a plain JSON field, not decoded into raw
# bytes and sent as a multipart file part. _uses_input_image() would otherwise
# force multipart for these exactly like every other file-consuming tool
# (PAN/Aadhaar OCR genuinely need multipart — only opt out tools here whose
# API contract has actually been confirmed to accept base64-in-JSON).
BASE64_JSON_FILE_TOOL_IDS: set = {
    "tool_08b022",  # BSA Api — switched to base64 JSON input 2026-08-18
    "tool_d12b96",  # PAN Image verification (PANOCR) — same base64-in-JSON contract, 2026-08-18
    "tool_d0bf92",  # Aadhaar Image verification (AadhaarOCR) — same base64-in-JSON contract, 2026-08-18
}


def _is_file_value(value: Any) -> bool:
    """Heuristic: long, space-free strings are likely base64-encoded file bytes.

    We use this to decide whether a resolved request_body field should be sent
    as a multipart file part (files=) vs. a plain form field (data=).
    A threshold of 200 chars avoids misidentifying short tokens/IDs.
    """
    if not isinstance(value, str):
        return False
    return len(value) > 200 and " " not in value and "\n" not in value


def _decode_if_base64(value: str) -> bytes:
    """Decode a base64 string to raw bytes for multipart upload.
    Falls back to UTF-8 encoding if the string is not valid base64.
    """
    try:
        return base64.b64decode(value)
    except Exception:
        return value.encode("utf-8")


async def execute_tool(tool_id: str, variables: Dict[str, str]) -> Dict[str, str]:
    """
    Execute a tool with given variables.
    Supports both local (/api/...) and external (https://...) endpoints.
    Returns a dict of new variable mappings from the response.
    """
    tool = tool_store.get(tool_id)
    if not tool:    
        logger.warning(f"[ToolExec] Tool not found: {tool_id}")
        log_error(
            "tool_not_found",
            f"Tool '{tool_id}' not found in registry",
            severity="WARNING",
            tool_id=tool_id,
            source_module="tool_registry",
            source_function="execute_tool",
        )
        return {}

    # Auto-generate OTP if the tool's request body needs {GeneratedOTP} or {OTP}
    # and neither is already in session variables. Data-driven: reads the tool
    # config rather than relying on a hardcoded set of tool IDs.
    import json as _json
    body_str = _json.dumps(tool.get("request_body", {}))
    needs_otp = re.search(r"\{GeneratedOTP\}|\{OTP\}", body_str)
    if needs_otp and not variables.get("GeneratedOTP"):
        otp = _generate_otp()
        variables = {**variables, "GeneratedOTP": otp, "OTP": otp}
        logger.info(f"[ToolExec] Auto-generated OTP for {tool_id}")

    method = tool.get("method", "GET").upper()
    endpoint = _resolve_variables(tool["endpoint"], variables)
    processed_body = resolve_request_body(tool.get("request_body", {}), variables)
    headers = {k: _resolve_variables(v, variables) for k, v in tool.get("headers", {}).items()}

    # Guard: _resolve_variables()/resolve_request_body() fall back to the raw
    # "{Var}" template text when a variable is missing from `variables` (e.g.
    # the LLM picked this tool before an image was actually uploaded). Firing
    # the request anyway ships literal placeholder text to the external API
    # instead of real data — refuse instead, so callers get {} and can react
    # (retry once the data is available) rather than a confusing 4xx from a
    # third-party API that received "{InputFile}" as a string.
    def _unresolved_placeholders(value: Any) -> List[str]:
        if isinstance(value, dict):
            return [v for x in value.values() for v in _unresolved_placeholders(x)]
        if isinstance(value, list):
            return [v for x in value for v in _unresolved_placeholders(x)]
        if isinstance(value, str):
            return re.findall(r"\{\{?(\w+)\}?\}", value)
        return []

    # _unresolved = _unresolved_placeholders(endpoint) + _unresolved_placeholders(processed_body)
    # if _unresolved:
    #     logger.warning(
    #         f"[ToolExec] Skipped {tool['name']} ({tool_id}): unresolved variable(s) "
    #         f"{_unresolved} — required data not yet available, not calling the API"
    #     )
    #     return {}

    # Track whether this tool consumed {InputImage} so callers can explicitly
    # clear it from the session after the call — we do NOT mutate the input
    # dict here (that was a hidden side-effect; cleanup now lives in the
    # orchestrator call sites via the _clear_input_image sentinel).
    _consumed_image = ("InputImage" in variables or "InputFile" in variables) and _uses_input_image(tool)

    # Determine URL — external (https://) or local (/api/...)
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        url = endpoint
    else:
        base_url = f"http://localhost:{cfg.PORT}"
        url = f"{base_url}{endpoint}"

    # Detect content type — must happen before we mutate headers below.
    # Tools that consume {InputImage}/{InputFile} are auto-treated as
    # multipart regardless of the tool's registered Content-Type: the chat
    # API only ever hands us base64 in a JSON-shaped session variable, but
    # the receiving API (e.g. a BSA/OCR endpoint declaring `file: UploadFile`)
    # needs a genuine multipart file part, not a base64 string inside JSON.
    # Requiring the user to remember to set Content-Type: multipart/form-data
    # by hand is exactly the class of misconfiguration that silently degrades
    # to the JSON path and produces "Expected UploadFile, received <class 'str'>".
    _ct = (headers.get("Content-Type") or headers.get("content-type") or "").lower()
    _is_multipart = tool_id not in BASE64_JSON_FILE_TOOL_IDS and (
        "multipart/form-data" in _ct or _uses_input_image(tool)
    )

    # For JSON POST/PUT: ensure Content-Type is set if caller omitted it
    if method in ("POST", "PUT") and not _is_multipart:
        if "Content-Type" not in headers and "content-type" not in headers:
            headers["Content-Type"] = "application/json"

    logger.info(f"[ToolExec] Executing: {tool['name']} | {method} {url} | multipart={_is_multipart}")
    
    def _redact(val: Any) -> Any:
        if isinstance(val, dict): return {k: _redact(v) for k, v in val.items()}
        if isinstance(val, list): return [_redact(v) for v in val]
        if isinstance(val, str) and len(val) > 200: return val[:20] + "...[base64 redacted]..." + val[-20:]
        return val
        
    logger.info(f"[ToolExec] Resolved body: {_redact(processed_body)}")

    try:
        timeout_seconds = tool.get("x_timeout_ms", 30000) / 1000
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            if method == "GET":
                resp = await client.get(url, params=processed_body, headers=headers)
            elif method in ("POST", "PUT") and _is_multipart:
                # Strip Content-Type so httpx can set it with the correct boundary
                _headers_no_ct = {
                    k: v for k, v in headers.items() if k.lower() != "content-type"
                }
                _files: Dict[str, Any] = {}
                _data: Dict[str, str] = {}
                for _field, _val in (processed_body if isinstance(processed_body, dict) else {}).items():
                    if _is_file_value(_val):
                        _raw = _decode_if_base64(_val)
                        _files[_field] = (_field, _raw, "application/octet-stream")
                    else:
                        _data[_field] = str(_val) if _val is not None else ""
                logger.info(f"[ToolExec] multipart: file_fields={list(_files.keys())} data_fields={list(_data.keys())}")
                if method == "POST":
                    resp = await client.post(url, files=_files, data=_data, headers=_headers_no_ct)
                else:
                    resp = await client.put(url, files=_files, data=_data, headers=_headers_no_ct)
            elif method == "POST":
                resp = await client.post(url, json=processed_body, headers=headers)
            elif method == "PUT":
                resp = await client.put(url, json=processed_body, headers=headers)
            elif method == "DELETE":
                resp = await client.delete(url, headers=headers)
            else:
                logger.warning(f"[ToolExec] Unknown method: {method}")
                return {}

        if resp.status_code not in (200, 201):
            logger.warning(f"[ToolExec] Failed: {resp.status_code} - {resp.text[:200]}")
            log_error(
                "tool_http_error",
                f"Tool '{tool['name']}' returned HTTP {resp.status_code}",
                severity="WARNING",
                tool_id=tool_id,
                tool_name=tool["name"],
                endpoint=url,
                http_method=method,
                http_status=resp.status_code,
                error_detail={"response_body": resp.text[:500]},
                request_payload=processed_body if isinstance(processed_body, dict) else None,
                source_module="tool_registry",
                source_function="execute_tool",
            )
            return {}

        # Try to parse JSON response
        try:
            response_data = resp.json()
        except Exception:
            logger.info(f"[ToolExec] Non-JSON response from {url}, wrapping in dict")
            response_data = {"text": resp.text, "raw": resp.text}

        logger.info(f"[ToolExec] API response: {str(response_data)[:500]}")

        # Map response fields to variables using the response_body template
        new_variables = {}
        extract_response_vars(tool.get("response_body", {}), response_data, new_variables)
        
        # Ensure all extracted values are strings (session vars convention)
        new_variables = {k: str(v) if not isinstance(v, str) else v for k, v in new_variables.items()}

        # If OTP was auto-generated for this call, carry it back into the session
        # so the LLM can reference it (e.g. to tell the customer what OTP was sent).
        if needs_otp and variables.get("GeneratedOTP"):
            new_variables.setdefault("GeneratedOTP", variables["GeneratedOTP"])

        # Signal to the caller that InputImage was consumed by this tool run
        # so it can perform explicit cleanup (pop InputImage, set sentinel flags).
        # We use a private underscore key — excluded from LLM prompts and logs.
        if _consumed_image:
            new_variables["_clear_input_image"] = "true"

        logger.info(f"[ToolExec] Success: {tool['name']} | Mapped {len(new_variables)} variables")
        logger.info(f"variables -> {new_variables}")
        
        # Chaining logic
        if tool.get("x_on_complete") == "next_tool_call" and tool.get("next_tool"):
            next_tool_id = tool.get("next_tool")
            logger.info(f"[ToolExec] Chaining to next tool: {next_tool_id}")
            combined_vars = {**variables, **new_variables}
            next_tool_results = await execute_tool(next_tool_id, combined_vars)
            new_variables.update(next_tool_results)

        return new_variables

    except httpx.ConnectError:
        logger.error(f"[ToolExec] Connection failed for {url}")
        log_error(
            "tool_connection_error",
            f"Tool '{tool['name']}' — connection refused or server unreachable: {url}",
            tool_id=tool_id,
            tool_name=tool["name"],
            endpoint=url,
            http_method=method,
            http_status=0,
            request_payload=processed_body if isinstance(processed_body, dict) else None,
            source_module="tool_registry",
            source_function="execute_tool",
        )
        return {}
    except httpx.TimeoutException:
        logger.error(f"[ToolExec] Timeout for {url}")
        log_error(
            "tool_timeout_error",
            f"Tool '{tool['name']}' — request timed out after {tool.get('x_timeout_ms', 30000)}ms: {url}",
            tool_id=tool_id,
            tool_name=tool["name"],
            endpoint=url,
            http_method=method,
            error_detail={"timeout_ms": tool.get("x_timeout_ms", 30000)},
            request_payload=processed_body if isinstance(processed_body, dict) else None,
            source_module="tool_registry",
            source_function="execute_tool",
        )
        return {}
    except Exception as e:
        logger.error(f"[ToolExec] Error executing {tool['name']}: {e}")
        log_error(
            "tool_http_error",
            f"Tool '{tool['name']}' — unexpected error: {e}",
            severity="CRITICAL",
            tool_id=tool_id,
            tool_name=tool["name"],
            endpoint=url,
            http_method=method,
            error_detail={"exception": str(e)},
            request_payload=processed_body if isinstance(processed_body, dict) else None,
            source_module="tool_registry",
            source_function="execute_tool",
        )
        return {}


# ── Async Tool Dispatch (fire-and-forget) ───────────────────────────────────

# Tools flagged x_execution_mode="async" don't block the turn: dispatch_tool()
# kicks execute_tool() off as a background asyncio.Task and returns a stub
# immediately. The real result is written to async_tool_jobs and picked up
# by orchestrator.collect_background_results() on a later turn (poll-and-inject).

# Callers MUST route every tool call through dispatch_tool() instead of
# execute_tool() directly — it's the single choke point that makes async
# tools transparent to the rest of the orchestrator.

_ASYNC_STUB_KEY = "_async_dispatched"


class AsyncJobStore:
    """PostgreSQL-backed job store (async_tool_jobs table) — correlates a
    background dispatch to its eventual result via job_id."""

    def create(self, tool_id: str, conversation_id: str) -> str:
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        execute_write(
            """INSERT INTO async_tool_jobs (job_id, tool_id, conversation_id, status, delivered)
               VALUES (%s, %s, %s, 'pending', FALSE)""",
            (job_id, tool_id, conversation_id),
        )
        return job_id

    def mark_done(self, job_id: str, result: Dict[str, Any]):
        execute_write(
            "UPDATE async_tool_jobs SET status='done', result=%s, completed_at=NOW() WHERE job_id=%s",
            (json.dumps(result), job_id),
        )

    def mark_failed(self, job_id: str, reason: str):
        execute_write(
            "UPDATE async_tool_jobs SET status='failed', result=%s, completed_at=NOW() WHERE job_id=%s",
            (json.dumps({"error": reason}), job_id),
        )

    def get_undelivered_done(self, conversation_id: str) -> List[Dict]:
        """Jobs that finished (done or failed) since the last poll and haven't
        been shown to the conversation yet."""
        return execute_query(
            """SELECT * FROM async_tool_jobs
               WHERE conversation_id = %s AND delivered = FALSE AND status IN ('done', 'failed')
               ORDER BY completed_at""",
            (conversation_id,),
        )

    def mark_delivered(self, job_ids: List[str]):
        if not job_ids:
            return
        execute_write(
            "UPDATE async_tool_jobs SET delivered = TRUE WHERE job_id = ANY(%s)",
            (job_ids,),
        )


job_store = AsyncJobStore()


async def _run_async_tool_job(job_id: str, tool_id: str, variables: Dict[str, str], timeout_ms: int):
    """Background worker: runs the real tool call and writes the outcome to job_store.
    Never raises — failures (including timeout) are recorded as status='failed'."""
    try:
        result = await asyncio.wait_for(execute_tool(tool_id, variables), timeout=timeout_ms / 1000)
        job_store.mark_done(job_id, result)
        logger.info(f"[AsyncTool] Job {job_id} ({tool_id}) completed | vars={list(result.keys())}")
    except asyncio.TimeoutError:
        job_store.mark_failed(job_id, f"Timed out after {timeout_ms}ms")
        logger.warning(f"[AsyncTool] Job {job_id} ({tool_id}) timed out after {timeout_ms}ms")
        log_error(
            "async_tool_error",
            f"Async job {job_id} timed out after {timeout_ms}ms",
            severity="WARNING",
            tool_id=tool_id,
            error_detail={"job_id": job_id, "timeout_ms": timeout_ms},
            source_module="tool_registry",
            source_function="_run_async_tool_job",
        )
    except Exception as e:
        job_store.mark_failed(job_id, str(e))
        logger.error(f"[AsyncTool] Job {job_id} ({tool_id}) failed: {e}")
        log_error(
            "async_tool_error",
            f"Async job {job_id} failed: {e}",
            tool_id=tool_id,
            error_detail={"job_id": job_id, "exception": str(e)},
            source_module="tool_registry",
            source_function="_run_async_tool_job",
        )


async def dispatch_tool(tool_id: str, variables: Dict[str, str], conversation_id: str) -> Dict[str, Any]:
    """
    Async-aware replacement for calling execute_tool() directly.

    sync tools (default): awaits execute_tool() and returns the real result,
    identical to calling execute_tool() directly.

    async tools: creates a job_store row, fires execute_tool() as a background
    asyncio.Task (NOT awaited), and returns a stub dict immediately so the
    caller can keep the turn moving:
        {"_async_dispatched": "<tool name>", "_async_job_id": "...", "_async_meanwhile": "..."}
    Callers must NOT merge this stub into session variables (it's routing
    metadata, not tool data) — check for _async_dispatched before persisting.
    Use format_tool_result() to turn it into model-facing text.
    """
    tool = tool_store.get(tool_id)
    if not tool or tool.get("x_execution_mode", "sync") != "async":
        try:
            return await execute_tool(tool_id, variables)
        except Exception as e:
            log_error(
                "tool_http_error",
                f"dispatch_tool: unexpected error executing tool '{tool_id}': {e}",
                severity="CRITICAL",
                tool_id=tool_id,
                tool_name=tool["name"] if tool else tool_id,
                error_detail={"exception": str(e), "exception_type": type(e).__name__},
                source_module="tool_registry",
                source_function="dispatch_tool",
            )
            return {}

    job_id = job_store.create(tool_id, conversation_id)
    timeout_ms = tool.get("x_timeout_ms", 30000)
    # Snapshot variables now — session state may keep changing before the
    # background task actually runs.
    asyncio.create_task(_run_async_tool_job(job_id, tool_id, dict(variables), timeout_ms))

    logger.info(f"[AsyncTool] Dispatched: {tool['name']} ({tool_id}) | job={job_id} | timeout={timeout_ms}ms")
    stub = {
        _ASYNC_STUB_KEY: tool["name"],
        "_async_job_id": job_id,
        "_async_meanwhile": tool.get("x_meanwhile", "progress_the_journey"),
    }
    
    # If this async tool consumed an image, signal the caller to clean it up
    # immediately (synchronously) so it doesn't re-trigger on the next turn.
    if ("InputImage" in variables or "InputFile" in variables) and _uses_input_image(tool):
        stub["_clear_input_image"] = "true"
        
    return stub


def is_async_stub(new_vars: Optional[Dict[str, Any]]) -> bool:
    """True if `new_vars` is a dispatch stub from dispatch_tool() rather than real tool data."""
    return bool(new_vars) and _ASYNC_STUB_KEY in new_vars


def format_tool_result(tool_id: str, new_vars: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Build the '[Tool Result: ...]' text block for a tool_id + new_vars pair,
    for injection into the LLM's dynamic context. Recognizes the async-dispatch
    stub shape and renders a 'running in background' notice instead of
    pretending real data came back — this is what satisfies section 6 of the
    async design (the model must be told, not left to guess).
    """
    if not new_vars:
        return None

    if is_async_stub(new_vars):
        tool_name = new_vars[_ASYNC_STUB_KEY]
        meanwhile = new_vars.get("_async_meanwhile", "progress_the_journey")
        hint = (
            "There is nothing to wait on here — briefly acknowledge and continue naturally."
            if meanwhile == "small_talk"
            else "Move on to the next useful step in the conversation now instead of waiting on this."
        )
        return (
            f"[Tool Dispatched: {tool_name}] Running in the background — the real result will "
            f"be delivered to you automatically on a later turn. Do NOT say you are \"processing\" "
            f"and do NOT call this tool again to check on it. {hint}"
        )

    tool = tool_store.get(tool_id)
    tool_name = tool["name"] if tool else tool_id
    lines = [f"[Tool Result: {tool_name}]"]
    for k, v in new_vars.items():
        if not str(k).startswith("_"):  # exclude private sentinel keys
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def format_background_result(job_row: Dict[str, Any]) -> str:
    """Render a completed/failed background job (from poll_completed_jobs) as
    model-facing text for the turn it's delivered into."""
    tool = tool_store.get(job_row["tool_id"])
    tool_name = tool["name"] if tool else job_row["tool_id"]
    result = job_row.get("result") or {}
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            result = {}

    if job_row.get("status") == "failed":
        reason = result.get("error", "unknown error")
        return (
            f"[Background Job Failed: {tool_name}] {reason}. "
            f"Tell the customer honestly that it didn't come back in time — offer to retry only if they ask."
        )

    lines = [f"[Background Result: {tool_name}]"]
    for k, v in result.items():
        if not str(k).startswith("_"):
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def poll_completed_jobs(conversation_id: str) -> List[Dict[str, Any]]:
    """
    Pattern B (poll-and-inject): fetch background jobs that finished since the
    last turn and haven't been shown to this conversation yet, then mark them
    delivered so they aren't shown twice. Batches everything completed since
    the last turn into one call so multiple in-flight jobs surface together.
    """
    rows = job_store.get_undelivered_done(conversation_id)
    if not rows:
        return []
    job_store.mark_delivered([r["job_id"] for r in rows])
    return rows


# ── API Endpoints ────────────────────────────────────────────────────────────

@tool_router.post("")
def create_tool(req: ToolCreate):
    """Register a new tool."""
    tool = tool_store.create(
        name=req.name,
        description=req.description,
        type=req.type,
        method=req.method,
        endpoint=req.endpoint,
        request_body=req.request_body,
        headers=req.headers,
        response_body=req.response_body,
        x_execution_mode=req.x_execution_mode,
        x_timeout_ms=req.x_timeout_ms,
        x_on_complete=req.x_on_complete,
        x_meanwhile=req.x_meanwhile,
        next_tool=req.next_tool,
    )
    return tool


@tool_router.get("")
def list_tools():
    """List all registered tools."""
    tools = tool_store.list_all()
    return {"total": len(tools), "tools": tools}


@tool_router.get("/{tool_id}")
def get_tool(tool_id: str):
    """Get a specific tool."""
    tool = tool_store.get(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_id}")
    return tool


@tool_router.put("/{tool_id}")
def update_tool(tool_id: str, req: ToolUpdate):
    """Update a tool."""
    tool = tool_store.update(
        tool_id,
        name=req.name,
        description=req.description,
        type=req.type,
        method=req.method,
        endpoint=req.endpoint,
        request_body=req.request_body,
        headers=req.headers,
        response_body=req.response_body,
        x_execution_mode=req.x_execution_mode,
        x_timeout_ms=req.x_timeout_ms,
        x_on_complete=req.x_on_complete,
        x_meanwhile=req.x_meanwhile,
        next_tool=req.next_tool,
    )
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_id}")
    return tool


@tool_router.delete("/{tool_id}")
def delete_tool(tool_id: str):
    """Delete a tool."""
    success = tool_store.delete(tool_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_id}")
    return {"detail": "Tool deleted", "tool_id": tool_id}


@tool_router.post("/{tool_id}/execute")
async def execute_tool_endpoint(tool_id: str, req: ToolExecuteRequest):
    """Execute a tool with given variables (for testing)."""
    tool = tool_store.get(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_id}")

    result = await execute_tool(tool_id, req.variables)
    return {"tool_id": tool_id, "result_variables": result}


# ── Image Upload Endpoint ────────────────────────────────────────────────────

image_router = APIRouter(prefix="/api/image", tags=["Image Upload"])


@image_router.post("/upload")
async def upload_image(
    file: UploadFile = File(..., description="Image or PDF file (Aadhaar, PAN, etc.)"),
    session_id: Optional[str] = Form(None, description="Active chat session ID — if provided, InputImage is stored in session variables"),
):
    """
    Convert an uploaded image/PDF to a base64 string and optionally store it
    in the active session as the `InputImage` variable so the orchestrator
    can inject it into the Aadhaar OCR tool's request body.

    - Accepts any image (JPEG, PNG, WebP) or PDF.
    - If `session_id` is provided and the session exists, sets
      `session.variables["InputImage"]` to the base64 string.
    - Returns the base64 string regardless.
    """
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    b64_string = base64.b64encode(file_bytes).decode("utf-8")
    logger.info(f"[ImageUpload] file={file.filename} size={len(file_bytes)} bytes → base64 len={len(b64_string)}")

    stored = False
    if session_id:
        # Import here to avoid circular imports at module load time
        try:
            from memory_store import memory_store
            session = memory_store.get_session(session_id)
            if session:
                session.set_variable("InputImage", b64_string)
                session.set_variable("InputFile", b64_string)
                session.set_variable("InputImageFileName", file.filename)
                # Mark that a new image is pending OCR and reset the
                # already-fired sentinel so re-uploads work correctly.
                session.set_variable("_image_pending", "true")
                session.variables.pop("_image_tool_fired", None)
                logger.info(f"[ImageUpload] Stored InputImage in session={session_id} | pending=true, fired sentinel cleared")
                stored = True
            else:
                logger.warning(f"[ImageUpload] Session not found: {session_id}")
        except Exception as e:
            logger.error(f"[ImageUpload] Failed to store in session: {e}")

    return {
        "success": True,
        "filename": file.filename,
        "size_bytes": len(file_bytes),
        "base64": b64_string,
        "session_id": session_id,
        "stored_in_session": stored,
        "message": (
            f"InputImage stored in session {session_id}. Aadhaar OCR will be triggered on next message."
            if stored else
            "Base64 conversion successful. Provide a session_id to auto-store for OCR."
        ),
    }
