from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from config import cfg
from logger import get_logger

logger = get_logger(__name__)

_loaded: bool = False
_agents: Dict[str, Dict] = {}
_raw_config: Optional[Dict] = None


def _load_once() -> None:
    global _loaded, _agents, _raw_config
    if _loaded:
        return
    path = cfg.LOCAL_AGENT_CONFIG_PATH
    if not os.path.isfile(path):
        logger.error(f"[LocalAgentLoader] File not found: {path}")
        _loaded = True
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            _raw_config = json.load(fh)
    except Exception as exc:
        logger.error(f"[LocalAgentLoader] Failed to parse {path}: {exc}")
        _loaded = True
        return
    agent_id = _raw_config.get("agent_id") or "local_agent"
    flow = _raw_config.get("conversationFlow", {})
    nodes_list = flow.get("nodes", [])
    global_node_ids = flow.get("global_node_ids", [])
    all_node_ids = [n["id"] for n in nodes_list]
    tools_list = _raw_config.get("tools", [])
    tool_ids = [t["tool_id"] for t in tools_list if "tool_id" in t]
    agent_dict = {
        "agent_id": agent_id,
        "name": _raw_config.get("agent_name", "Local Agent"),
        "mode": _raw_config.get("mode", "flow"),
        "system_prompt": (_raw_config.get("system_prompt") or flow.get("global_prompt", "")),
        "start_node": flow.get("start_node_id", all_node_ids[0] if all_node_ids else ""),
        "nodes": all_node_ids,
        "global_nodes": global_node_ids,
        "tools": tool_ids,
        "language": _raw_config.get("language", "english"),
        "personality": _raw_config.get("personality"),
        "swap_rules": _raw_config.get("swap_rules", {}),
        "agent_config_json": _raw_config,
        "created_at": "",
        "updated_at": "",
    }
    _agents[agent_id] = agent_dict
    try:
        from tool_registry import tool_store as _tool_store
        for tool in tools_list:
            tid = tool.get("tool_id")
            if not tid:
                continue
            _tool_store.register_local_tool(tool)
        logger.info(f"[LocalAgentLoader] Registered {len(tools_list)} tools from local config")
    except Exception as exc:
        logger.warning(f"[LocalAgentLoader] Could not register tools: {exc}")
    _loaded = True
    logger.info(f"[LocalAgentLoader] Loaded agent '{agent_dict['name']}' (id={agent_id}) from {path}")


class LocalAgentStore:
    def get(self, agent_id: str) -> Optional[Dict]:
        _load_once()
        if agent_id in _agents:
            return _agents[agent_id]
        if _agents:
            first = next(iter(_agents.values()))
            return first
        return None

    def list_all(self) -> List[Dict]:
        _load_once()
        return list(_agents.values())

    def create(self, **kwargs) -> Optional[Dict]:
        logger.warning("[LocalAgentLoader] create() called in local mode -- no-op")
        return None

    def update(self, agent_id: str, **kwargs) -> Optional[Dict]:
        logger.warning("[LocalAgentLoader] update() called in local mode -- no-op")
        return self.get(agent_id)

    def delete(self, agent_id: str) -> bool:
        logger.warning("[LocalAgentLoader] delete() called in local mode -- no-op")
        return False


local_agent_store = LocalAgentStore()
