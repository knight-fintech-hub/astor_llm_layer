"""
node_manager.py — Node Data Layer (PostgreSQL)
======================================================
Nodes stored in PostgreSQL `nodes` table.

Nodes are no longer exposed as a standalone API — they're managed inline as
part of an agent's workflow (see agent_manager.py, which owns the
POST/GET/PUT/DELETE /api/nodes endpoints). This module just provides the
`node_store` data layer that agent_manager.py and orchestrator.py read from.
"""

from __future__ import annotations

import json
import uuid
from typing import Dict, List, Optional

from database import execute_query, execute_one, execute_write
from logger import get_logger

logger = get_logger(__name__)


# ── Node Store (PostgreSQL) ──────────────────────────────────────────────────

class NodeStore:
    """PostgreSQL-backed node store."""

    def create(self, **kwargs) -> Dict:
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        execute_write(
            """INSERT INTO nodes (node_id, name, type, prompt, static_message, 
               variables_needed, tools, is_global, max_turns, transitions)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                node_id,
                kwargs["name"],
                kwargs.get("type", "conversation"),
                kwargs["prompt"],
                kwargs.get("static_message"),
                json.dumps(kwargs.get("variables_needed", [])),
                json.dumps(kwargs.get("tools", [])),
                kwargs.get("is_global", False),
                kwargs.get("max_turns", 3),
                json.dumps(kwargs.get("transitions", [])),
            ),
        )
        logger.info(f"[NodeStore] Created: {node_id} ({kwargs['name']})")
        return self.get(node_id)

    def get(self, node_id: str) -> Optional[Dict]:
        row = execute_one("SELECT * FROM nodes WHERE node_id = %s", (node_id,))
        return self._format(row) if row else None

    def list_all(self) -> List[Dict]:
        rows = execute_query("SELECT * FROM nodes ORDER BY created_at")
        return [self._format(r) for r in rows]

    def update(self, node_id: str, **kwargs) -> Optional[Dict]:
        existing = self.get(node_id)
        if not existing:
            return None

        updates = []
        values = []
        for key in ("name", "type", "prompt", "static_message", "is_global", "max_turns"):
            if kwargs.get(key) is not None:
                updates.append(f"{key} = %s")
                values.append(kwargs[key])

        for key in ("variables_needed", "tools", "transitions"):
            if kwargs.get(key) is not None:
                updates.append(f"{key} = %s")
                values.append(json.dumps(kwargs[key]))

        if not updates:
            return existing

        values.append(node_id)
        execute_write(f"UPDATE nodes SET {', '.join(updates)} WHERE node_id = %s", tuple(values))
        logger.info(f"[NodeStore] Updated: {node_id}")
        return self.get(node_id)

    def delete(self, node_id: str) -> bool:
        existing = self.get(node_id)
        if not existing:
            return False
        execute_write("DELETE FROM nodes WHERE node_id = %s", (node_id,))
        logger.info(f"[NodeStore] Deleted: {node_id}")
        return True

    def get_global_nodes(self) -> List[Dict]:
        rows = execute_query("SELECT * FROM nodes WHERE is_global = TRUE")
        return [self._format(r) for r in rows]

    def _format(self, row: Dict) -> Dict:
        """Format DB row to match expected dict structure."""
        if not row:
            return None
        return {
            "node_id": row["node_id"],
            "name": row["name"],
            "type": row["type"],
            "prompt": row["prompt"],
            "static_message": row.get("static_message"),
            "variables_needed": row.get("variables_needed", []),
            "tools": row.get("tools", []),
            "is_global": row.get("is_global", False),
            "max_turns": row.get("max_turns", 3),
            "transitions": row.get("transitions", []),
            "created_at": str(row.get("created_at", "")),
        }


# Initialize store
node_store = NodeStore()
