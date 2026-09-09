"""
agent_manager.py — Dynamic Agent Configuration API (PostgreSQL)
========================================================
Agents stored in PostgreSQL `agents` table. Node workflow management lives
here too — nodes are configured inline as part of building an agent's flow,
so this module owns both the agent endpoints and the node endpoints.

Endpoints:
  POST   /api/agents         — Create an agent
  GET    /api/agents         — List all agents
  GET    /api/agents/{id}    — Get agent by ID
  PUT    /api/agents/{id}    — Update agent
  DELETE /api/agents/{id}    — Delete agent

  POST   /api/nodes          — Create a node (attach to an agent's workflow)
  GET    /api/nodes          — List all nodes
  GET    /api/nodes/{id}     — Get node by ID
  PUT    /api/nodes/{id}     — Update node
  DELETE /api/nodes/{id}     — Delete node
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from database import execute_query, execute_one, execute_write
from node_manager import node_store
from logger import get_logger

logger = get_logger(__name__)


# ── Consolidated JSON Builder ────────────────────────────────────────────────

def build_consolidated_json(agent_id: str, agent: Dict) -> Dict:
    """
    Build a single consolidated agent config JSON in Retell-style format,
    extended with internal fields (tools, variables_needed, keywords, etc.)
    not present in standard Retell schema.

    This is stored in `agents.agent_config_json` and consumed by the
    orchestrator at session-start — eliminating per-turn DB node lookups.
    """
    from tool_registry import tool_store as _tool_store

    # ── Collect all node IDs (regular + global) ──────────────────────────────
    all_node_ids: List[str] = list(agent.get("nodes", []))
    global_node_ids: List[str] = list(agent.get("global_nodes", []))
    combined_ids = list(dict.fromkeys(all_node_ids + global_node_ids))  # preserve order, dedup

    # ── Build node list in Retell-style ─────────────────────────────────────
    nodes_out = []
    for nid in combined_ids:
        node = node_store.get(nid)
        if not node:
            logger.warning(f"[build_consolidated_json] Node not found: {nid}")
            continue

        # Convert internal `transitions` → Retell-style `edges`
        edges = []
        for i, t in enumerate(node.get("transitions", [])):
            edge_id = f"edge-{nid}-{i}"
            edges.append({
                "id": edge_id,
                "condition": t.get("condition", ""),
                "transition_condition": {
                    "type": "prompt",
                    "prompt": t.get("condition", ""),
                },
                "keywords": t.get("keywords", []),          # extended (not in Retell)
                "destination_node_id": t.get("to_node"),
                "on_transition_tool": t.get("on_transition_tool"),  # extended
            })

        nodes_out.append({
            "id": node["node_id"],
            "name": node["name"],
            "type": node["type"],
            "start_speaker": "agent",
            "instruction": {
                "type": "prompt",
                "text": node.get("prompt", ""),
            },
            "static_message": node.get("static_message"),  # extended
            "variables_needed": node.get("variables_needed", []),  # extended
            "is_global": node.get("is_global", False),          # extended
            "max_turns": node.get("max_turns", 3),             # extended
            "node_tools": node.get("tools", []),               # extended: node-level tool IDs
            "edges": edges,
        })

    # ── Collect tool definitions ─────────────────────────────────────────────
    tool_ids: List[str] = list(agent.get("tools", []))
    tools_out = []
    for tid in tool_ids:
        tool = _tool_store.get(tid)
        if not tool:
            logger.warning(f"[build_consolidated_json] Tool not found: {tid}")
            continue
        tools_out.append({
            "tool_id": tool["tool_id"],
            "name": tool["name"],
            "description": tool.get("description"),
            "type": tool.get("type", "api_call"),
            "method": tool.get("method", "GET"),
            "endpoint": tool.get("endpoint", ""),
            "request_body": tool.get("request_body", {}),
            "headers": tool.get("headers", {}),
            "response_body": tool.get("response_body", {}),
        })

    # ── Determine start_node_id ──────────────────────────────────────────────
    start_node_id = agent.get("start_node") or (all_node_ids[0] if all_node_ids else None)

    # ── Assemble final JSON ──────────────────────────────────────────────────
    import time as _time
    consolidated = {
        "agent_id": agent_id,
        "agent_name": agent.get("name", ""),
        "language": agent.get("language", "english"),
        "last_modification_timestamp": int(_time.time() * 1000),
        "response_engine": {
            "type": "conversation-flow",
            "version": 0,
        },
        # Extended fields (not in standard Retell)
        "mode": agent.get("mode", "flow"),
        "system_prompt": agent.get("system_prompt", ""),
        "personality": agent.get("personality"),
        "swap_rules": agent.get("swap_rules", {}),
        # Retell-style conversation flow
        "conversationFlow": {
            "start_node_id": start_node_id,
            "start_speaker": "agent",
            "global_prompt": agent.get("system_prompt", ""),
            "knowledge_base_ids": [],
            "nodes": nodes_out,
            "global_node_ids": global_node_ids,  # extended: list of global node IDs for fast lookup
        },
        # Extended: full tool definitions
        "tools": tools_out,
        "built_at": datetime.now().isoformat(),
    }

    logger.info(
        f"[build_consolidated_json] Built config for {agent_id}: "
        f"{len(nodes_out)} nodes, {len(tools_out)} tools"
    )
    return consolidated

agent_router = APIRouter(prefix="/api/agents", tags=["Agent Manager"])
node_router = APIRouter(prefix="/api/nodes", tags=["Agent Manager"])


# ── Models ───────────────────────────────────────────────────────────────────

class AgentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    mode: str = Field("flow")
    system_prompt: str = Field(..., min_length=1)
    start_node: Optional[str] = None
    nodes: List[str] = Field(default_factory=list)
    global_nodes: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    language: str = Field("english")
    personality: Optional[str] = None
    swap_rules: Dict[str, Optional[str]] = Field(default_factory=dict)


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    mode: Optional[str] = None
    system_prompt: Optional[str] = None
    start_node: Optional[str] = None
    nodes: Optional[List[str]] = None
    global_nodes: Optional[List[str]] = None
    tools: Optional[List[str]] = None
    language: Optional[str] = None
    personality: Optional[str] = None
    swap_rules: Optional[Dict[str, Optional[str]]] = None


class NodeCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    type: str = Field("conversation")
    prompt: str = Field(..., min_length=1)
    static_message: Optional[str] = None
    variables_needed: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    is_global: bool = Field(False)
    max_turns: int = Field(3)
    transitions: List[Dict] = Field(
        default_factory=list,
        description=(
            "[{to_node, condition, on_transition_tool?}] — on_transition_tool is "
            "an optional tool_id that fires automatically the moment this specific "
            "transition is taken (declarative trigger, e.g. send a payment link "
            "when moving from Node 1 to Node 2)."
        ),
    )


class NodeUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    prompt: Optional[str] = None
    static_message: Optional[str] = None
    variables_needed: Optional[List[str]] = None
    tools: Optional[List[str]] = None
    is_global: Optional[bool] = None
    max_turns: Optional[int] = None
    transitions: Optional[List[Dict]] = None


# ── Agent Store (PostgreSQL) ─────────────────────────────────────────────────

class AgentStore:
    """PostgreSQL-backed agent store."""

    def create(self, **kwargs) -> Dict:
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        execute_write(
            """INSERT INTO agents (agent_id, name, mode, system_prompt, start_node,
               nodes, global_nodes, tools, language, personality, swap_rules)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                agent_id,
                kwargs["name"],
                kwargs.get("mode", "flow"),
                kwargs["system_prompt"],
                kwargs.get("start_node"),
                json.dumps(kwargs.get("nodes", [])),
                json.dumps(kwargs.get("global_nodes", [])),
                json.dumps(kwargs.get("tools", [])),
                kwargs.get("language", "english"),
                kwargs.get("personality"),
                json.dumps(kwargs.get("swap_rules", {})),
            ),
        )
        logger.info(f"[AgentStore] Created: {agent_id} ({kwargs['name']})")

        # Build and store the consolidated agent config JSON
        agent = self.get(agent_id)
        if agent:
            config_json = build_consolidated_json(agent_id, agent)
            execute_write(
                "UPDATE agents SET agent_config_json = %s WHERE agent_id = %s",
                (json.dumps(config_json), agent_id),
            )
            logger.info(f"[AgentStore] Consolidated JSON built and stored for {agent_id}")

        return self.get(agent_id)

    def get(self, agent_id: str) -> Optional[Dict]:
        row = execute_one("SELECT * FROM agents WHERE agent_id = %s", (agent_id,))
        return self._format(row) if row else None

    def list_all(self) -> List[Dict]:
        rows = execute_query("SELECT * FROM agents ORDER BY created_at")
        return [self._format(r) for r in rows]

    def update(self, agent_id: str, **kwargs) -> Optional[Dict]:
        existing = self.get(agent_id)
        if not existing:
            return None

        updates = []
        values = []
        for key in ("name", "mode", "system_prompt", "start_node", "language", "personality"):
            if kwargs.get(key) is not None:
                updates.append(f"{key} = %s")
                values.append(kwargs[key])

        for key in ("nodes", "global_nodes", "tools", "swap_rules"):
            if kwargs.get(key) is not None:
                updates.append(f"{key} = %s")
                values.append(json.dumps(kwargs[key]))

        updates.append("updated_at = %s")
        values.append(datetime.now())

        if not updates:
            return existing

        values.append(agent_id)
        execute_write(f"UPDATE agents SET {', '.join(updates)} WHERE agent_id = %s", tuple(values))
        logger.info(f"[AgentStore] Updated: {agent_id}")

        # Rebuild consolidated JSON after update
        updated = self.get(agent_id)
        if updated:
            config_json = build_consolidated_json(agent_id, updated)
            execute_write(
                "UPDATE agents SET agent_config_json = %s WHERE agent_id = %s",
                (json.dumps(config_json), agent_id),
            )
            logger.info(f"[AgentStore] Consolidated JSON rebuilt for {agent_id}")

        return self.get(agent_id)

    def delete(self, agent_id: str) -> bool:
        existing = self.get(agent_id)
        if not existing:
            return False
        execute_write("DELETE FROM agents WHERE agent_id = %s", (agent_id,))
        logger.info(f"[AgentStore] Deleted: {agent_id}")
        return True

    def _format(self, row: Dict) -> Dict:
        if not row:
            return None

        # agent_config_json is stored as JSONB — psycopg2 returns it as a dict
        # already; guard against string form just in case.
        raw_config = row.get("agent_config_json")
        if isinstance(raw_config, str):
            try:
                raw_config = json.loads(raw_config)
            except Exception:
                raw_config = None

        return {
            "agent_id": row["agent_id"],
            "name": row["name"],
            "mode": row.get("mode", "flow"),
            "system_prompt": row["system_prompt"],
            "start_node": row.get("start_node"),
            "nodes": row.get("nodes", []),
            "global_nodes": row.get("global_nodes", []),
            "tools": row.get("tools", []),
            "language": row.get("language", "english"),
            "personality": row.get("personality"),
            "swap_rules": row.get("swap_rules", {}),
            "agent_config_json": raw_config,  # consolidated agent config JSON
            "created_at": str(row.get("created_at", "")),
            "updated_at": str(row.get("updated_at", "")),
        }


# Initialize store
agent_store = AgentStore()


# ── Node-change helpers ──────────────────────────────────────────────────────

def _get_agents_referencing_node(node_id: str) -> List[str]:
    """Return agent_ids for all agents whose nodes or global_nodes include node_id."""
    all_agents = agent_store.list_all()
    return [
        a["agent_id"]
        for a in all_agents
        if node_id in a.get("nodes", []) or node_id in a.get("global_nodes", [])
    ]


def _rebuild_agents_for_node(node_id: str) -> None:
    """Rebuild agent_config_json for every agent that references node_id."""
    for agent_id in _get_agents_referencing_node(node_id):
        agent = agent_store.get(agent_id)
        if agent:
            config_json = build_consolidated_json(agent_id, agent)
            execute_write(
                "UPDATE agents SET agent_config_json = %s WHERE agent_id = %s",
                (json.dumps(config_json), agent_id),
            )
            logger.info(
                f"[_rebuild_agents_for_node] Rebuilt agent_config_json for "
                f"{agent_id} after node {node_id} updated"
            )


# ── API Endpoints ────────────────────────────────────────────────────────────

@agent_router.post("")
def create_agent(req: AgentCreate):
    agent = agent_store.create(
        name=req.name, mode=req.mode, system_prompt=req.system_prompt,
        start_node=req.start_node, nodes=req.nodes, global_nodes=req.global_nodes,
        tools=req.tools, language=req.language, personality=req.personality,
        swap_rules=req.swap_rules,
    )
    return agent


@agent_router.get("")
def list_agents():
    agents = agent_store.list_all()
    return {"total": len(agents), "agents": agents}


@agent_router.get("/{agent_id}")
def get_agent(agent_id: str):
    agent = agent_store.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    return agent


@agent_router.put("/{agent_id}")
def update_agent(agent_id: str, req: AgentUpdate):
    agent = agent_store.update(
        agent_id, name=req.name, mode=req.mode, system_prompt=req.system_prompt,
        start_node=req.start_node, nodes=req.nodes, global_nodes=req.global_nodes,
        tools=req.tools, language=req.language, personality=req.personality,
        swap_rules=req.swap_rules,
    )
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    return agent


@agent_router.delete("/{agent_id}")
def delete_agent(agent_id: str):
    success = agent_store.delete(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    return {"detail": "Agent deleted", "agent_id": agent_id}


# ── Node Endpoints (agent workflow building blocks) ─────────────────────────

@node_router.post("")
def create_node(req: NodeCreate):
    node = node_store.create(
        name=req.name, type=req.type, prompt=req.prompt,
        static_message=req.static_message, variables_needed=req.variables_needed,
        tools=req.tools, is_global=req.is_global, max_turns=req.max_turns,
        transitions=req.transitions,
    )
    return node


@node_router.get("")
def list_nodes():
    nodes = node_store.list_all()
    return {"total": len(nodes), "nodes": nodes}


@node_router.get("/{node_id}")
def get_node(node_id: str):
    node = node_store.get(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    return node


@node_router.put("/{node_id}")
def update_node(node_id: str, req: NodeUpdate):
    node = node_store.update(
        node_id, name=req.name, type=req.type, prompt=req.prompt,
        static_message=req.static_message, variables_needed=req.variables_needed,
        tools=req.tools, is_global=req.is_global, max_turns=req.max_turns,
        transitions=req.transitions,
    )
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    # Rebuild agent_config_json for every agent that references this node
    _rebuild_agents_for_node(node_id)

    return node


@node_router.delete("/{node_id}")
def delete_node(node_id: str):
    # Collect affected agents BEFORE deleting the node so we can rebuild after
    affected = _get_agents_referencing_node(node_id)

    success = node_store.delete(node_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    # Rebuild agent_config_json for affected agents (node will be absent now)
    for agent_id in affected:
        agent = agent_store.get(agent_id)
        if agent:
            config_json = build_consolidated_json(agent_id, agent)
            execute_write(
                "UPDATE agents SET agent_config_json = %s WHERE agent_id = %s",
                (json.dumps(config_json), agent_id),
            )
            logger.info(f"[delete_node] Rebuilt agent_config_json for {agent_id} after node {node_id} deleted")

    return {"detail": "Node deleted", "node_id": node_id}
