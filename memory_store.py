"""
memory_store.py — Session Memory & Message Storage (PostgreSQL)
================================================================
All sessions and messages persist to PostgreSQL.
  - sessions table: session state, variables, node tracking
  - messages table: every message in conversation
  - call_logs table: summary when call ends

Also keeps in-memory cache for fast access during active session.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


from database import execute_write, execute_query, execute_one
from error_logger import log_error
from logger import get_logger

logger = get_logger(__name__)


# ── Session Model (in-memory during active call) ─────────────────────────────

@dataclass
class Session:
    """Tracks a single conversation session."""
    session_id: str
    agent_id: str
    current_node: str
    previous_node: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)
    node_turns: int = 0
    node_tools_executed: bool = False
    swap_history: List[Dict] = field(default_factory=list)
    status: str = "active"
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: Optional[str] = None
    # In-memory node cache populated from agent_config_json at session start.
    # Key: node_id, Value: node dict (internal format).
    # Eliminates per-turn DB lookups for node data during the conversation.
    node_cache: Dict[str, Any] = field(default_factory=dict)
    # Wall-clock time of the last get_session() access — used by TTL eviction.
    last_accessed: float = field(default_factory=lambda: __import__("time").time())


    def set_variable(self, key: str, value: Any):
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def set_variables(self, new_vars: Dict[str, Any]):
        self.variables.update(new_vars)

    def move_to_node(self, node_id: str):
        self.previous_node = self.current_node
        self.current_node = node_id
        self.node_turns = 0
        self.node_tools_executed = False
        # Update DB
        execute_write(
            "UPDATE sessions SET current_node=%s, previous_node=%s, node_turns=0, node_tools_executed=FALSE WHERE session_id=%s",
            (node_id, self.previous_node, self.session_id),
        )
        logger.info(f"[Session:{self.session_id}] Moved: {self.previous_node} → {node_id}")

    def increment_turns(self):
        self.node_turns += 1
        execute_write(
            "UPDATE sessions SET node_turns=%s WHERE session_id=%s",
            (self.node_turns, self.session_id),
        )

    def add_message(self, role: str, content: str):
        """Add message to history + save to DB."""
        self.history.append({
            "role": role,
            "content": content,
            "node": self.current_node,
            "timestamp": datetime.now().isoformat(),
        })
        # Save to messages table
        execute_write(
            "INSERT INTO messages (session_id, role, content, node) VALUES (%s, %s, %s, %s)",
            (self.session_id, role, content, self.current_node),
        )

    def get_clean_history(self) -> List[Dict[str, str]]:
        return [
            {"role": h["role"], "content": h["content"]}
            for h in self.history
            if h["role"] in ("user", "assistant")
        ]

    def record_swap(self, from_agent: str, to_agent: str, reason: str):
        self.swap_history.append({
            "from_agent": from_agent,
            "to_agent": to_agent,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })
        execute_write(
            "UPDATE sessions SET swap_history=%s WHERE session_id=%s",
            (json.dumps(self.swap_history), self.session_id),
        )

    def end(self):
        """End session — update DB + write call_log."""
        self.status = "completed"
        self.ended_at = datetime.now().isoformat()
        execute_write(
            "UPDATE sessions SET status='completed', ended_at=%s, variables=%s WHERE session_id=%s",
            (self.ended_at, json.dumps(self.variables), self.session_id),
        )
        # Write call_log summary
        self._write_call_log()
        logger.info(f"[Session:{self.session_id}] Ended")

    def _write_call_log(self):
        """Write summary to call_logs table."""
        try:
            execute_write(
                """INSERT INTO call_logs 
                   (session_id, agent_id, customer_mobile, customer_name, disposition,
                    total_messages, ptp_created, ptp_date, ptp_amount, started_at, ended_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    self.session_id,
                    self.agent_id,
                    self.variables.get("mobile_number") or self.variables.get("MobileNumber", ""),
                    self.variables.get("CustomerName", ""),
                    "PTP" if self.variables.get("PTPStatus") else "Completed",
                    len(self.history),
                    bool(self.variables.get("PTPId")),
                    self.variables.get("PTPDate"),
                    float(self.variables.get("PTPAmount", 0)) if self.variables.get("PTPAmount") else None,
                    self.started_at,
                    self.ended_at,
                ),
            )
            logger.info(f"[CallLog] Written for session: {self.session_id}")
        except Exception as e:
            logger.error(f"[CallLog] Failed to write: {e}")
            log_error(
                "call_log_write_error",
                f"Failed to write call_log for session '{self.session_id}': {e}",
                session_id=self.session_id,
                agent_id=self.agent_id,
                error_detail={"exception": str(e)},
                source_module="memory_store",
                source_function="_write_call_log",
            )

    def save_state(self):
        """Persist current state to DB (called periodically or on important changes)."""
        execute_write(
            """UPDATE sessions SET 
               current_node=%s, previous_node=%s, variables=%s, 
               node_turns=%s, node_tools_executed=%s, status=%s
               WHERE session_id=%s""",
            (
                self.current_node,
                self.previous_node,
                json.dumps(self.variables),
                self.node_turns,
                self.node_tools_executed,
                self.status,
                self.session_id,
            ),
        )

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "current_node": self.current_node,
            "previous_node": self.previous_node,
            "variables": self.variables,
            "history_length": len(self.history),
            "node_turns": self.node_turns,
            "swap_history": self.swap_history,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

    def get_memory_context(self) -> str:
        lines = [
            "=== SESSION MEMORY ===",
            f"Customer: {self.get_variable('CustomerName', 'Unknown')}",
            f"Mobile: {self.get_variable('mobile_number') or self.get_variable('MobileNumber', 'Unknown')}",
        ]
        skip_keys = {"CustomerName", "MobileNumber", "mobile_number", "LenderName"}
        for k, v in self.variables.items():
            if k not in skip_keys and not k.startswith("_"):
                lines.append(f"{k}: {v}")
        lines.append("\n--- Recent Conversation ---")
        recent = self.history[-10:]
        for msg in recent:
            role = "Agent" if msg["role"] == "assistant" else "Customer"
            lines.append(f"  {role}: {msg['content'][:100]}")
        lines.append("=== END ===")
        return "\n".join(lines)


# ── Memory Store (PostgreSQL-backed) ─────────────────────────────────────────

class MemoryStore:
    """PostgreSQL-backed session store with in-memory cache for active sessions."""

    def __init__(self):
        self._cache: Dict[str, Session] = {}

    def create_session(self, agent_id: str, start_node: str) -> Session:
        """Create a new session — saved to DB immediately."""
        session_id = f"sess_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        # Insert into DB
        execute_write(
            """INSERT INTO sessions 
               (session_id, agent_id, current_node, variables, node_turns, status, started_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (session_id, agent_id, start_node, json.dumps({}), 0, "active", now),
        )

        # Create in-memory session
        session = Session(
            session_id=session_id,
            agent_id=agent_id,
            current_node=start_node,
            started_at=now,
        )
        self._cache[session_id] = session
        logger.info(f"[MemoryStore] Created session: {session_id} | agent={agent_id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session — from cache or reload from DB. Refreshes last_accessed."""
        # Check cache first
        if session_id in self._cache:
            self._cache[session_id].last_accessed = time.time()
            return self._cache[session_id]

        # Try to reload from DB
        row = execute_one("SELECT * FROM sessions WHERE session_id = %s", (session_id,))
        if not row:
            return None

        # Rebuild session from DB
        session = Session(
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            current_node=row["current_node"],
            previous_node=row.get("previous_node"),
            variables=row.get("variables", {}),
            node_turns=row.get("node_turns", 0),
            node_tools_executed=row.get("node_tools_executed", False),
            status=row.get("status", "active"),
            swap_history=row.get("swap_history", []),
            started_at=str(row.get("started_at", "")),
            ended_at=str(row.get("ended_at", "")) if row.get("ended_at") else None,
        )

        # Reload messages from DB
        messages = execute_query(
            "SELECT role, content, node, created_at FROM messages WHERE session_id = %s ORDER BY id",
            (session_id,),
        )
        session.history = [
            {"role": m["role"], "content": m["content"], "node": m.get("node", ""), "timestamp": str(m.get("created_at", ""))}
            for m in messages
        ]

        session.last_accessed = time.time()
        self._cache[session_id] = session
        logger.info(f"[MemoryStore] Reloaded session from DB: {session_id}")
        return session


    def list_active(self) -> List[Dict]:
        """List all active sessions from DB."""
        rows = execute_query("SELECT * FROM sessions WHERE status = 'active' ORDER BY started_at DESC")
        return [
            {
                "session_id": r["session_id"],
                "agent_id": r["agent_id"],
                "current_node": r["current_node"],
                "status": r["status"],
                "started_at": str(r.get("started_at", "")),
            }
            for r in rows
        ]

    def end_session(self, session_id: str):
        """End a session."""
        session = self.get_session(session_id)
        if session:
            session.end()
            # Remove from cache
            self._cache.pop(session_id, None)

    def swap_agent(self, session_id: str, new_agent_id: str, new_start_node: str, reason: str) -> Optional[Session]:
        """Swap agent for a session, preserving memory."""
        session = self.get_session(session_id)
        if not session:
            return None

        old_agent = session.agent_id
        session.record_swap(old_agent, new_agent_id, reason)
        session.agent_id = new_agent_id
        session.move_to_node(new_start_node)

        # Update DB
        execute_write(
            "UPDATE sessions SET agent_id=%s WHERE session_id=%s",
            (new_agent_id, session_id),
        )

        logger.info(f"[MemoryStore] Agent swap: {old_agent} → {new_agent_id}")
        return session

    def evict_stale(self, max_idle_sec: Optional[int] = None) -> int:
        """
        Remove idle/completed sessions from the in-memory cache.

        Only cache entries whose `last_accessed` is older than `max_idle_sec`
        AND whose status is not 'active' are evicted. Active sessions (mid-
        conversation) are never evicted regardless of age — their data is
        always needed instantly without a DB round-trip.

        Sessions remain fully persisted in PostgreSQL; only the cache entry
        is dropped. The next get_session() call for an evicted session
        transparently reloads it from the DB.

        Args:
            max_idle_sec: Override the config TTL for this call. Defaults to
                          cfg.SESSION_CACHE_TTL_SEC (typically 3600 s / 1 h).

        Returns:
            Number of sessions evicted from cache.
        """
        from config import cfg as _cfg
        ttl = max_idle_sec if max_idle_sec is not None else _cfg.SESSION_CACHE_TTL_SEC
        now = time.time()
        stale_ids = [
            sid for sid, s in self._cache.items()
            if s.status != "active" and (now - s.last_accessed) > ttl
        ]
        for sid in stale_ids:
            del self._cache[sid]
        if stale_ids:
            logger.info(f"[MemoryStore] Evicted {len(stale_ids)} stale session(s) from cache")
        return len(stale_ids)

    @property
    def active_session_count(self) -> int:
        """Number of sessions currently held in the in-memory cache."""
        return len(self._cache)

    @property
    def cache_summary(self) -> dict:
        """Return a snapshot of the cache for monitoring (/health endpoint)."""
        active = sum(1 for s in self._cache.values() if s.status == "active")
        completed = sum(1 for s in self._cache.values() if s.status != "active")
        return {
            "cached_sessions": len(self._cache),
            "active": active,
            "completed_in_cache": completed,
        }


# ── Singleton ────────────────────────────────────────────────────────────────

memory_store = MemoryStore()
