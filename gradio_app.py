"""
gradio_app.py — Knight Flow Generic UI
========================================
3-tab interface:
  Tab 1: Tool Registry — Register APIs and tools
  Tab 2: Agent Builder — Configure agent + build its node workflow inline
  Tab 3: Chat          — Dynamic conversation with flow-based agents
"""

import json\
    
import httpx
import gradio as gr
26
API_BASE = "http://192.168.0.155:1230"


# ── Helper ────────────────────────────────────────────────────────────────────

def api(method, endpoint, json_data=None):
    try:
        url = f"{API_BASE}{endpoint}"
        timeout = 120 if method == "POST" and "/chat" in endpoint else 5
        if method == "GET":
            r = httpx.get(url, timeout=timeout)
        elif method == "POST":
            r = httpx.post(url, json=json_data, timeout=timeout)
        elif method == "PUT":
            r = httpx.put(url, json=json_data, timeout=timeout)
        elif method == "DELETE":
            r = httpx.delete(url, timeout=timeout)
        else:
            return None, "Bad method"
        if r.status_code == 200:
            return r.json(), None
        return None, f"Error {r.status_code}: {r.text[:200]}"
    except httpx.ConnectError:
        return None, "Server not running (port 8101)"
    except httpx.TimeoutException:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: TOOL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def refresh_tools():
    data, err = api("GET", "/api/tools")
    if err:
        return f"Error: {err}"
    tools = data.get("tools", [])
    if not tools:
        return "No tools registered."
    lines = []
    for t in tools:
        mode_badge = " ⚡ `async`" if t.get("x_execution_mode") == "async" else ""
        lines.append(f"**{t['name']}** (`{t['tool_id']}`) — {t['method']} {t['endpoint']}{mode_badge}")
        if t.get("description"):
            lines.append(f"  _{t['description']}_")
        if t.get("x_execution_mode") == "async":
            lines.append(
                f"  Async: timeout={t.get('x_timeout_ms', 30000)}ms | "
                f"on_complete={t.get('x_on_complete', 'conversation')} | "
                f"meanwhile={t.get('x_meanwhile', 'progress_the_journey')}"
            )
        if t.get("response_body"):
            lines.append(f"  Maps: {t['response_body']}")
        lines.append("")
    return "\n".join(lines)


def toggle_async_fields(execution_mode):
    """Show the async-only fields (timeout, on_complete, meanwhile) only when
    Execution Mode is set to async — mirrors toggle_mode_panels() below."""
    return gr.update(visible=(execution_mode == "async"))


def apply_content_type(content_type_choice, current_headers):
    """Auto-inject or replace the Content-Type line in the headers textarea
    when the user picks a quick-select content type.
    """
    if not content_type_choice or content_type_choice == "(none / keep as-is)":
        return current_headers or ""

    ct_value = {
        "application/json": "application/json",
        "multipart/form-data": "multipart/form-data",
        "application/x-www-form-urlencoded": "application/x-www-form-urlencoded",
    }.get(content_type_choice, content_type_choice)

    lines = [l for l in (current_headers or "").strip().splitlines()
             if not l.strip().lower().startswith("content-type")]
    lines.insert(0, f"Content-Type={ct_value}")
    return "\n".join(lines)


def create_tool(
    name, description, method, endpoint, request_body, headers, response_body,
    execution_mode, timeout_ms, on_complete, meanwhile,
):
    if not name or not endpoint:
        return "Name and Endpoint required."

    request_body_dict = {}
    if request_body:
        try:
            request_body_dict = json.loads(request_body)
        except json.JSONDecodeError as e:
            return f"Error: Request Body is not valid JSON. {e}"

    headers_dict = {}
    if headers:
        for line in headers.strip().split("\n"):
            line = line.strip()
            if line.startswith("#") or not line:   # skip comment / blank lines
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                headers_dict[k.strip()] = v.strip()

    response_body_dict = {}
    if response_body:
        try:
            response_body_dict = json.loads(response_body)
        except json.JSONDecodeError as e:
            return f"Error: Response Body Template is not valid JSON. {e}"

    payload = {
        "name": name.strip(),
        "description": description.strip() if description else None,
        "type": "api_call",
        "method": method,
        "endpoint": endpoint.strip(),
        "request_body": request_body_dict,
        "headers": headers_dict,
        "response_body": response_body_dict,
        "x_execution_mode": execution_mode,
        "x_timeout_ms": int(timeout_ms) if timeout_ms else 30000,
        "x_on_complete": on_complete,
        "x_meanwhile": meanwhile,
    }
    data, err = api("POST", "/api/tools", payload)
    if err:
        return f"Error: {err}"
    mode_note = " (⚡ async)" if data.get("x_execution_mode") == "async" else ""
    return f"✅ Created: **{data['name']}** (`{data['tool_id']}`){mode_note}"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: AGENT BUILDER — with inline Node Workflow Builder
# ══════════════════════════════════════════════════════════════════════════════

def _parse_transitions(transitions_text):
    """Parse transitions from text: to_node_id | condition | kw1, kw2"""
    transitions = []
    if transitions_text and transitions_text.strip():
        for line in transitions_text.strip().split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                to_node = parts[0].strip()
                condition = parts[1].strip()
                keywords = [k.strip() for k in parts[2].split(",") if k.strip()] if len(parts) > 2 else []
                transitions.append({"to_node": to_node, "condition": condition, "keywords": keywords})
    return transitions


def render_draft_nodes(draft_state):
    """Render the draft node list as a markdown summary."""
    if not draft_state:
        return "_(No nodes added yet. Use the form below to add nodes to this agent.)_"
    lines = []
    for i, n in enumerate(draft_state):
        star = " ⭐ **START**" if n.get("_is_start") else ""
        globe = " 🌐 GLOBAL" if n.get("is_global") else ""
        trans_count = len(n.get("transitions", []))
        lines.append(
            f"**{i+1}. {n['name']}** (`{n['node_id']}`){star}{globe}"
            f" — {n['type']} | Turns: {n.get('max_turns', 3)} | Transitions: {trans_count}"
        )
        lines.append(f"   _Prompt: {n['prompt'][:70]}..._")
        lines.append("")
    return "\n".join(lines)


def get_draft_node_choices(draft_state):
    """Return dropdown choices for start-node selector from draft."""
    if not draft_state:
        return gr.update(choices=[], value=None)
    choices = [(f"{n['name']} ({n['node_id']})", n["node_id"]) for n in draft_state]
    return gr.update(choices=choices)


def add_node_to_draft(n_name, n_type, n_prompt, n_static, n_variables, n_tools,
                      n_global, n_max_turns, n_transitions, draft_state):
    """Create a node via API and add it to the draft state."""
    if not n_name or not n_prompt:
        return draft_state, "⚠️ Node Name and Prompt are required.", render_draft_nodes(draft_state), get_draft_node_choices(draft_state)

    transitions = _parse_transitions(n_transitions)

    payload = {
        "name": n_name.strip(),
        "type": n_type,
        "prompt": n_prompt.strip(),
        "static_message": n_static.strip() if n_static else None,
        "variables_needed": [v.strip() for v in n_variables.split(",") if v.strip()] if n_variables else [],
        "tools": [t.strip() for t in n_tools.split(",") if t.strip()] if n_tools else [],
        "is_global": n_global,
        "max_turns": int(n_max_turns) if n_max_turns else 3,
        "transitions": transitions,
    }

    data, err = api("POST", "/api/nodes", payload)
    if err:
        return draft_state, f"❌ Error creating node: {err}", render_draft_nodes(draft_state), get_draft_node_choices(draft_state)

    # Attach draft metadata
    data["_is_start"] = False
    new_draft = list(draft_state) + [data]

    msg = f"✅ Node **{data['name']}** (`{data['node_id']}`) added."
    return new_draft, msg, render_draft_nodes(new_draft), get_draft_node_choices(new_draft)


def remove_node_from_draft(node_id_to_remove, draft_state):
    """Remove a node from the draft (and delete from backend)."""
    if not node_id_to_remove:
        return draft_state, "⚠️ Select a node to remove.", render_draft_nodes(draft_state), get_draft_node_choices(draft_state)

    new_draft = [n for n in draft_state if n["node_id"] != node_id_to_remove]
    _, err = api("DELETE", f"/api/nodes/{node_id_to_remove}")
    msg = f"🗑️ Removed node `{node_id_to_remove}`." if not err else f"Removed from draft (API error: {err})"
    return new_draft, msg, render_draft_nodes(new_draft), get_draft_node_choices(new_draft)


def set_start_node_in_draft(start_node_id, draft_state):
    """Mark the selected node as the start node in draft metadata."""
    if not start_node_id:
        return draft_state, render_draft_nodes(draft_state)
    new_draft = []
    for n in draft_state:
        n = dict(n)
        n["_is_start"] = (n["node_id"] == start_node_id)
        new_draft.append(n)
    return new_draft, render_draft_nodes(new_draft)


def create_agent_with_nodes(a_name, a_mode, a_system_prompt, a_lang, a_personality,
                             a_tools_str, draft_state, start_node_dd):
    """Create the agent using the draft node list."""
    if not a_name or not a_system_prompt:
        return "⚠️ Agent Name and System Prompt are required."

    if a_mode == "flow":
        if not draft_state:
            return "⚠️ Flow mode requires at least one node. Add nodes below."

        # Determine start node
        start_node = start_node_dd
        if not start_node:
            # Default: first node in draft
            start_node = draft_state[0]["node_id"]

        all_node_ids = [n["node_id"] for n in draft_state]
        global_node_ids = [n["node_id"] for n in draft_state if n.get("is_global")]
        regular_node_ids = [n["node_id"] for n in draft_state if not n.get("is_global")]

        payload = {
            "name": a_name.strip(),
            "mode": "flow",
            "system_prompt": a_system_prompt.strip(),
            "start_node": start_node,
            "nodes": all_node_ids,
            "global_nodes": global_node_ids,
            "tools": [],
            "language": a_lang or "english",
            "personality": a_personality.strip() if a_personality else None,
            "swap_rules": {},
        }
    else:
        # single_prompt mode
        payload = {
            "name": a_name.strip(),
            "mode": "single_prompt",
            "system_prompt": a_system_prompt.strip(),
            "start_node": None,
            "nodes": [],
            "global_nodes": [],
            "tools": [t.strip() for t in a_tools_str.split(",") if t.strip()] if a_tools_str else [],
            "language": a_lang or "english",
            "personality": a_personality.strip() if a_personality else None,
            "swap_rules": {},
        }

    data, err = api("POST", "/api/agents", payload)
    if err:
        return f"❌ Error: {err}", ""

    config_json = data.get("agent_config_json", {})
    import json as _json
    config_str = _json.dumps(config_json, indent=2, ensure_ascii=False) if config_json else "(not available)"

    status_msg = (
        f"✅ Agent **{data['name']}** (`{data['agent_id']}`) created! "
        f"Mode: {data.get('mode', 'flow')} | "
        f"Nodes: {len(data.get('nodes', []))} | "
        f"Start: {data.get('start_node', 'N/A')}"
    )
    return status_msg, config_str


def refresh_agents():
    data, err = api("GET", "/api/agents")
    if err:
        return f"Error: {err}", gr.update(choices=[])
    agents = data.get("agents", [])
    if not agents:
        return "No agents yet.", gr.update(choices=[])
    lines = []
    for a in agents:
        mode = a.get("mode", "flow").upper()
        lines.append(f"**{a['name']}** (`{a['agent_id']}`) — {mode}")
        if mode == "FLOW":
            lines.append(f"  Start: {a.get('start_node', 'N/A')} | Nodes: {len(a.get('nodes', []))} | Global: {len(a.get('global_nodes', []))}")
        else:
            tools = a.get("tools", [])
            lines.append(f"  Tools: {tools if tools else 'None'}")
        lines.append(f"  Lang: {a.get('language', 'en')} | Personality: {a.get('personality', 'N/A')}")
        lines.append("")
    choices = [(f"{a['name']} ({a['agent_id']})", a["agent_id"]) for a in agents]
    return "\n".join(lines), gr.update(choices=choices)


def delete_agent(agent_id):
    if not agent_id:
        return "Select an agent."
    _, err = api("DELETE", f"/api/agents/{agent_id}")
    if err:
        return f"Error: {err}"
    return f"Deleted: {agent_id}"


# ══════════════════════════════════════════════════════════════════════════════
# AGENT UPDATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_update_agent_choices():
    """Get agent list for the update dropdown."""
    data, err = api("GET", "/api/agents")
    if data:
        choices = [(f"{a['name']} ({a['agent_id']})", a["agent_id"]) for a in data.get("agents", [])]
        return gr.update(choices=choices)
    return gr.update(choices=[])


def load_agent_for_update(agent_id):
    """Load agent data into form fields for editing."""
    if not agent_id:
        return ("", "flow", "", "english", "", "", "", "_(Select an agent to load its details)_")

    data, err = api("GET", f"/api/agents/{agent_id}")
    if err:
        return ("", "flow", "", "english", "", "", "", f"❌ Error loading agent: {err}")

    # Format nodes list for display
    nodes_display = ", ".join(data.get("nodes", []))
    global_nodes_display = ", ".join(data.get("global_nodes", []))
    tools_display = ", ".join(data.get("tools", []))

    status = f"✅ Loaded: **{data['name']}** (`{agent_id}`) — ready to edit"

    return (
        data.get("name", ""),
        data.get("mode", "flow"),
        data.get("system_prompt", ""),
        data.get("language", "english"),
        data.get("personality", "") or "",
        tools_display,
        data.get("start_node", "") or "",
        status,
    )


def save_agent_update(agent_id, u_name, u_mode, u_system_prompt, u_lang,
                      u_personality, u_tools_str, u_start_node):
    """
    Save partial update — only send fields that the user actually changed.
    Compares current form values with what's stored in the DB and sends only diffs.
    """
    if not agent_id:
        return "⚠️ Select an agent first."

    # Fetch current agent data from DB for comparison
    current, err = api("GET", f"/api/agents/{agent_id}")
    if err:
        return f"❌ Error fetching agent: {err}"

    # Build payload with ONLY changed fields
    payload = {}

    if u_name.strip() and u_name.strip() != current.get("name", ""):
        payload["name"] = u_name.strip()

    if u_mode and u_mode != current.get("mode", "flow"):
        payload["mode"] = u_mode

    if u_system_prompt.strip() and u_system_prompt.strip() != current.get("system_prompt", ""):
        payload["system_prompt"] = u_system_prompt.strip()

    if u_lang and u_lang != current.get("language", "english"):
        payload["language"] = u_lang

    current_personality = current.get("personality") or ""
    if u_personality.strip() != current_personality:
        payload["personality"] = u_personality.strip() if u_personality.strip() else None

    # Tools comparison
    new_tools = [t.strip() for t in u_tools_str.split(",") if t.strip()] if u_tools_str else []
    current_tools = current.get("tools", [])
    if new_tools != current_tools:
        payload["tools"] = new_tools

    # Start node comparison
    current_start = current.get("start_node") or ""
    if u_start_node.strip() != current_start:
        payload["start_node"] = u_start_node.strip() if u_start_node.strip() else None

    if not payload:
        return "ℹ️ No changes detected — nothing to update."

    # Send only the changed fields
    data, err = api("PUT", f"/api/agents/{agent_id}", payload)
    if err:
        return f"❌ Error updating: {err}"

    changed_fields = ", ".join(payload.keys())
    return f"✅ Updated **{data['name']}** — changed: {changed_fields}"


def load_agent_nodes_for_update(agent_id):
    """Load the nodes belonging to this agent for individual node editing."""
    if not agent_id:
        return gr.update(choices=[]), "_(Select an agent first)_"

    data, err = api("GET", f"/api/agents/{agent_id}")
    if err:
        return gr.update(choices=[]), f"❌ Error: {err}"

    all_node_ids = list(set(data.get("nodes", []) + data.get("global_nodes", [])))
    if not all_node_ids:
        return gr.update(choices=[]), "_(This agent has no nodes)_"

    choices = []
    lines = []
    for nid in all_node_ids:
        node_data, nerr = api("GET", f"/api/nodes/{nid}")
        if node_data:
            label = f"{node_data['name']} ({nid})"
            choices.append((label, nid))
            globe = " 🌐" if node_data.get("is_global") else ""
            trans_count = len(node_data.get("transitions", []))
            lines.append(
                f"• **{node_data['name']}** (`{nid}`){globe} — "
                f"{node_data['type']} | Turns: {node_data.get('max_turns', 3)} | "
                f"Transitions: {trans_count}"
            )

    summary = "\n".join(lines) if lines else "_(No nodes found)_"
    return gr.update(choices=choices), summary


def load_node_for_update(node_id):
    """Load a single node's data into the node edit form."""
    if not node_id:
        return ("", "conversation", "", "", "", "", False, 3, "", "_(Select a node)_")

    data, err = api("GET", f"/api/nodes/{node_id}")
    if err:
        return ("", "conversation", "", "", "", "", False, 3, "", f"❌ Error: {err}")

    variables_str = ", ".join(data.get("variables_needed", []))
    tools_str = ", ".join(data.get("tools", []))

    # Format transitions back to editable text
    trans_lines = []
    for t in data.get("transitions", []):
        kw_str = ", ".join(t.get("keywords", []))
        line = f"{t.get('to_node', '')} | {t.get('condition', '')}"
        if kw_str:
            line += f" | {kw_str}"
        trans_lines.append(line)
    transitions_str = "\n".join(trans_lines)

    status = f"✅ Loaded node: **{data['name']}** (`{node_id}`)"

    return (
        data.get("name", ""),
        data.get("type", "conversation"),
        data.get("prompt", ""),
        data.get("static_message", "") or "",
        variables_str,
        tools_str,
        data.get("is_global", False),
        data.get("max_turns", 3),
        transitions_str,
        status,
    )


def save_node_update(node_id, un_name, un_type, un_prompt, un_static,
                     un_variables, un_tools, un_global, un_max_turns, un_transitions):
    """Save partial node update — only send fields that changed."""
    if not node_id:
        return "⚠️ Select a node first."

    # Fetch current node from DB
    current, err = api("GET", f"/api/nodes/{node_id}")
    if err:
        return f"❌ Error fetching node: {err}"

    payload = {}

    if un_name.strip() and un_name.strip() != current.get("name", ""):
        payload["name"] = un_name.strip()

    if un_type and un_type != current.get("type", "conversation"):
        payload["type"] = un_type

    if un_prompt.strip() and un_prompt.strip() != current.get("prompt", ""):
        payload["prompt"] = un_prompt.strip()

    current_static = current.get("static_message") or ""
    if un_static.strip() != current_static:
        payload["static_message"] = un_static.strip() if un_static.strip() else None

    new_vars = [v.strip() for v in un_variables.split(",") if v.strip()] if un_variables else []
    if new_vars != current.get("variables_needed", []):
        payload["variables_needed"] = new_vars

    new_tools = [t.strip() for t in un_tools.split(",") if t.strip()] if un_tools else []
    if new_tools != current.get("tools", []):
        payload["tools"] = new_tools

    if un_global != current.get("is_global", False):
        payload["is_global"] = un_global

    if int(un_max_turns) != current.get("max_turns", 3):
        payload["max_turns"] = int(un_max_turns)

    new_transitions = _parse_transitions(un_transitions)
    if new_transitions != current.get("transitions", []):
        payload["transitions"] = new_transitions

    if not payload:
        return "ℹ️ No changes detected — nothing to update."

    data, err = api("PUT", f"/api/nodes/{node_id}", payload)
    if err:
        return f"❌ Error updating node: {err}"

    changed_fields = ", ".join(payload.keys())
    return f"✅ Updated node **{data['name']}** — changed: {changed_fields}"


# ══════════════════════════════════════════════════════════════════════════════
# ADD NODE TO EXISTING AGENT — two options
# ══════════════════════════════════════════════════════════════════════════════

def _attach_node_to_agent(agent_id, node_id, is_global):
    """
    Internal helper: append node_id to the agent's nodes / global_nodes list
    and PUT the updated lists back to the agent.
    Returns (success_msg, err_msg).
    """
    current, err = api("GET", f"/api/agents/{agent_id}")
    if err:
        return None, f"❌ Error fetching agent: {err}"

    nodes = list(current.get("nodes", []))
    global_nodes = list(current.get("global_nodes", []))

    if is_global:
        if node_id not in global_nodes:
            global_nodes.append(node_id)
        if node_id not in nodes:
            nodes.append(node_id)
    else:
        if node_id not in nodes:
            nodes.append(node_id)

    payload = {"nodes": nodes, "global_nodes": global_nodes}
    data, err = api("PUT", f"/api/agents/{agent_id}", payload)
    if err:
        return None, f"❌ Error updating agent node list: {err}"
    return data, None


def add_new_node_to_agent(
    agent_id,
    an_name, an_type, an_prompt, an_static, an_variables,
    an_tools, an_global, an_max_turns, an_transitions,
):
    """
    Option 1 — Create a brand-new node via POST /api/nodes,
    then attach it to the selected agent.
    """
    if not agent_id:
        return "⚠️ Select an agent first.", gr.update()
    if not an_name or not an_prompt:
        return "⚠️ Node Name and Prompt are required.", gr.update()

    transitions = _parse_transitions(an_transitions)

    payload = {
        "name": an_name.strip(),
        "type": an_type,
        "prompt": an_prompt.strip(),
        "static_message": an_static.strip() if an_static else None,
        "variables_needed": [v.strip() for v in an_variables.split(",") if v.strip()] if an_variables else [],
        "tools": [t.strip() for t in an_tools.split(",") if t.strip()] if an_tools else [],
        "is_global": an_global,
        "max_turns": int(an_max_turns) if an_max_turns else 3,
        "transitions": transitions,
    }

    node_data, err = api("POST", "/api/nodes", payload)
    if err:
        return f"❌ Error creating node: {err}", gr.update()

    new_node_id = node_data["node_id"]
    _, attach_err = _attach_node_to_agent(agent_id, new_node_id, an_global)
    if attach_err:
        return attach_err, gr.update()

    # Reload node dropdown for the agent
    node_dd_update, summary = load_agent_nodes_for_update(agent_id)
    return (
        f"✅ Created & attached node **{node_data['name']}** (`{new_node_id}`) to agent.",
        node_dd_update,
    )


def add_existing_node_to_agent(agent_id, existing_node_id, as_global):
    """
    Option 2 — Attach an already-existing node (by node_id) to the selected agent.
    Validates that the node exists before attaching.
    """
    if not agent_id:
        return "⚠️ Select an agent first.", gr.update()
    if not existing_node_id or not existing_node_id.strip():
        return "⚠️ Provide an existing node_id.", gr.update()

    nid = existing_node_id.strip()

    # Validate node exists
    node_data, err = api("GET", f"/api/nodes/{nid}")
    if err:
        return f"❌ Node `{nid}` not found: {err}", gr.update()

    _, attach_err = _attach_node_to_agent(agent_id, nid, as_global)
    if attach_err:
        return attach_err, gr.update()

    # Reload node dropdown for the agent
    node_dd_update, _ = load_agent_nodes_for_update(agent_id)
    return (
        f"✅ Existing node **{node_data['name']}** (`{nid}`) attached to agent.",
        node_dd_update,
    )


def toggle_add_node_option(choice):
    """Show the correct sub-panel based on the user's addition method choice."""
    show_new = (choice == "option1")
    return gr.update(visible=show_new), gr.update(visible=not show_new)


def toggle_mode_panels(mode):
    """Show/hide node workflow panel vs single-prompt tools panel."""
    show_nodes = (mode == "flow")
    return gr.update(visible=show_nodes), gr.update(visible=not show_nodes)


def clear_draft():
    """Reset the draft state for a new agent."""
    return [], "_(No nodes added yet.)_", gr.update(choices=[], value=None), ""


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: CHAT
# ══════════════════════════════════════════════════════════════════════════════

def get_agent_choices():
    data, err = api("GET", "/api/agents")
    if data:
        return [(f"{a['name']} ({a['agent_id']})", a["agent_id"]) for a in data.get("agents", [])]
    return []


def get_customer_choices():
    choices = [("None — no customer record (ask for phone number)", None)]
    data, err = api("GET", "/api/cbs/customers")
    if data:
        choices += [(f"{c['name']} — {c['mobile']}", c["mobile"]) for c in data.get("customers", [])]
    return choices


def refresh_chat_dropdowns():
    agents = get_agent_choices()
    customers = get_customer_choices()
    return gr.update(choices=agents), gr.update(choices=customers, value=None)


def start_chat(agent_id, mobile):
    if not agent_id:
        return [{"role": "assistant", "content": "Select an agent first."}], "", None
    payload = {"agent_id": agent_id}
    if mobile:
        payload["mobile"] = mobile
    data, err = api("POST", "/api/chat/start", payload)
    if err:
        return [{"role": "assistant", "content": f"Error: {err}"}], "", None
    
    session_id = data["session_id"]
    greeting = data["greeting"]
    status = f"Session: {session_id} | Node: {data['current_node_name']}"
    return [{"role": "assistant", "content": greeting}], status, session_id


def send_message(message, history, session_state, agent_id, mobile, image_obj):
    """
    Send message and stream the response token-by-token from /api/chat/stream SSE endpoint.
    Gradio chatbot supports streaming via generator functions that yield partial updates.
    """
    # Determine if we have a message or an image (or both)
    has_text = bool(message and message.strip())
    has_image = bool(image_obj is not None)

    if not has_text and not has_image:
        yield history or [], "", "", gr.update()
        return
    
    if not session_state:
        history = history or []
        user_display = message if has_text else "[Image attached]"
        history.append({"role": "user", "content": user_display})
        history.append({"role": "assistant", "content": "Start a chat first (click Start Call)."})
        yield history, "", "", gr.update(value=None)
        return

    history = history or []

    payload = {
        "session_id": session_state,
        "message": message.strip() if has_text else "Please process the attached document.",
    }

    user_display = payload["message"]

    if has_image:
        try:
            import base64
            import os
            with open(image_obj.name, "rb") as f:
                file_bytes = f.read()
            b64_str = base64.b64encode(file_bytes).decode("utf-8")
            payload["image_base64"] = b64_str
            filename = os.path.basename(image_obj.name)
            user_display += f"\n\n[Attached: {filename}]"
        except Exception as e:
            history.append({"role": "user", "content": user_display})
            history.append({"role": "assistant", "content": f"Failed to read image: {e}"})
            yield history, "", "", gr.update(value=None)
            return

    # Add user message to history
    history.append({"role": "user", "content": user_display})
    # Add empty assistant message (will be filled by streaming)
    history.append({"role": "assistant", "content": ""})

    # Yield immediately to show user message
    yield history, "", "Generating...", gr.update(value=None)

    # Stream from SSE endpoint
    try:
        url = f"{API_BASE}/api/chat/stream"
        with httpx.stream("POST", url, json=payload, timeout=httpx.Timeout(5.0, read=120.0)) as response:
            if response.status_code != 200:
                history[-1]["content"] = f"Error: {response.status_code}"
                yield history, "", "Error", gr.update(value=None)
                return

            accumulated = ""
            status = ""

            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]  # strip "data: " prefix
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "token":
                    accumulated += event.get("content", "")
                    history[-1]["content"] = accumulated
                    yield history, "", "Streaming...", gr.update(value=None)

                elif event.get("type") == "done":
                    # Final event — update with complete response + metadata
                    accumulated = event.get("response", accumulated)
                    history[-1]["content"] = accumulated
                    trans = "YES" if event.get("transitioned") else "NO"
                    status = f"Node: {event.get('current_node_name', '?')} | Transitioned: {trans} | {event.get('duration_ms', 0):.0f}ms"
                    yield history, "", status, gr.update(value=None)

                elif event.get("type") == "error":
                    history[-1]["content"] = f"Error: {event.get('content', 'Unknown')}"
                    yield history, "", "Error", gr.update(value=None)
                    return

    except httpx.ConnectError:
        history[-1]["content"] = "Error: Server not running"
        yield history, "", "Connection failed", gr.update(value=None)
    except httpx.TimeoutException:
        history[-1]["content"] = "Error: Timeout"
        yield history, "", "Timeout", gr.update(value=None)
    except Exception as e:
        history[-1]["content"] = f"Error: {e}"
        yield history, "", "Error", gr.update(value=None)





# ══════════════════════════════════════════════════════════════════════════════
# BUILD UI
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="Knight Flow Generic") as app:

    gr.Markdown("# 🤖 Knight Flow Generic — Dynamic Multi-Agent Platform")
    gr.Markdown("Build conversational agents by configuring node workflows directly inside the **Agent Builder** tab.")

    with gr.Tabs():

        # ── TAB 1: Tool Registry ─────────────────────────────────────────────
        with gr.Tab("🔧 Tool Registry"):
            gr.Markdown("### Register APIs / Tools")
            tool_status = gr.Textbox(label="Status", interactive=False)
            tool_list = gr.Markdown("Click Refresh...")

            tool_refresh_btn = gr.Button("🔄 Refresh Tools", variant="secondary")

            gr.Markdown("---")
            gr.Markdown("### New Tool")
            with gr.Row():
                with gr.Column():
                    t_name = gr.Textbox(label="Name", placeholder="CBS Lookup")
                    t_desc = gr.Textbox(label="Description", placeholder="Fetch customer data")
                    t_method = gr.Dropdown(label="Method", choices=["GET", "POST", "PUT", "DELETE"], value="GET")
                    t_endpoint = gr.Textbox(label="Endpoint", placeholder="/api/cbs/summary/{mobile}")
                with gr.Column():
                    t_content_type = gr.Radio(
                        label="Content Type (quick-select)",
                        choices=[
                            "(none / keep as-is)",
                            "application/json",
                            "multipart/form-data",
                            "application/x-www-form-urlencoded",
                        ],
                        value="(none / keep as-is)",
                        info="For file-upload APIs choose multipart/form-data — the engine decodes {InputImage} to bytes automatically.",
                    )
                    t_params = gr.Textbox(
                        label="Request Body (JSON)",
                        lines=4,
                        placeholder=(
                            'JSON body — use {Variable} placeholders.\n'
                            'For multipart file upload:\n'
                            '{"file": "{InputImage}", "doc_type": "aadhaar", "customer_id": "{CustomerId}"}'
                        ),
                    )
                    t_headers = gr.Textbox(
                        label="Headers (key=value per line)",
                        lines=3,
                        placeholder=(
                            "Authorization=Bearer {AuthToken}\n"
                            "Content-Type=application/json\n"
                            "# For file upload: Content-Type=multipart/form-data"
                        ),
                    )
                    t_mapping = gr.Textbox(label="Response Body Template (JSON)", lines=4, placeholder='{"customer": {"name": "{CustomerName}"}, "loans": [{"emi": "{EMIAmount}"}]}')

            gr.Markdown("#### ⚡ Execution")
            with gr.Row():
                t_exec_mode = gr.Dropdown(
                    label="Execution Mode",
                    choices=["sync", "async"],
                    value="sync",
                    info="sync = agent waits for the real result. async = fires in the background; agent keeps talking and gets the result on a later turn."
                )
                with gr.Column(visible=False) as async_fields_panel:
                    t_timeout_ms = gr.Number(label="Timeout (ms)", value=30000, precision=0, minimum=1000)
                    t_on_complete = gr.Dropdown(
                        label="On Complete",
                        choices=["conversation", "next_tool_call"],
                        value="conversation",
                        info="What the agent should do once the real result arrives"
                    )
                    t_meanwhile = gr.Dropdown(
                        label="Meanwhile",
                        choices=["progress_the_journey", "small_talk"],
                        value="progress_the_journey",
                        info="What the agent does while the job runs — prefer progress_the_journey when there's a genuinely useful next step"
                    )

            tool_create_btn = gr.Button("➕ Register Tool", variant="primary")

        # ── TAB 3: AGENT BUILDER (with inline Node Workflow) ─────────────────
        with gr.Tab("🤖 Agent Builder"):
            gr.Markdown("### Configure Agent + Build Node Workflow")

            # ── Existing Agents ───────────────────────────────────────────────
            with gr.Accordion("📋 Existing Agents", open=False):
                agent_list = gr.Markdown("Click Refresh...")
                agent_select = gr.Dropdown(label="Select Agent (to delete)", choices=[])
                with gr.Row():
                    agent_refresh_btn = gr.Button("🔄 Refresh Agents", variant="secondary")
                    agent_delete_btn = gr.Button("🗑️ Delete Selected", variant="stop")
                agent_delete_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### ✨ Create New Agent")

            # ── Agent Meta Fields ─────────────────────────────────────────────
            with gr.Row():
                with gr.Column(scale=1):
                    a_name = gr.Textbox(label="Agent Name *", placeholder="e.g. Collection Agent")
                    a_mode = gr.Dropdown(
                        label="Mode *",
                        choices=["flow", "single_prompt"],
                        value="single_prompt",
                        info="'flow' = node-based workflow | 'single_prompt' = single LLM call with tools"
                    )
                    a_lang = gr.Dropdown(
                        label="Language",
                        choices=["english", "hindi", "hinglish"],
                        value="english"
                    )
                    a_personality = gr.Textbox(label="Personality", placeholder="professional, empathetic")

                with gr.Column(scale=2):
                    a_prompt = gr.Textbox(
                        label="System Prompt *",
                        lines=6,
                        placeholder="For flow mode: brief base prompt (e.g. 'You are a helpful collection agent...')\nFor single_prompt mode: full agent behavior here."
                    )

            # ── single_prompt tools (only visible in single_prompt mode) ──────
            with gr.Column(visible=False) as single_prompt_panel:
                gr.Markdown("#### 🔧 Agent-Level Tools (single_prompt mode)")
                a_tools_str = gr.Textbox(
                    label="Tool IDs — comma-separated",
                    placeholder="cbs_lookup, create_ptp, send_sms"
                )

            # ── Node Workflow Builder (only visible in flow mode) ─────────────
            with gr.Column(visible=False) as flow_panel:
                gr.Markdown("---")
                gr.Markdown(
                    "### 🔗 Node Workflow Builder\n"
                    "Add nodes one by one to build this agent's conversation flow. "
                    "The first node added becomes the default start node unless you change it."
                )

                # Draft state: list of node dicts
                draft_state = gr.State([])

                # Draft summary display
                draft_display = gr.Markdown("_(No nodes added yet. Use the form below to add nodes to this agent.)_")

                # Start node selector (populated dynamically from draft)
                with gr.Row():
                    start_node_dd = gr.Dropdown(
                        label="⭐ Set Start Node",
                        choices=[],
                        value=None,
                        info="Select which node the conversation begins at"
                    )
                    draft_clear_btn = gr.Button("🗑️ Clear All Draft Nodes", variant="stop", scale=0)

                node_draft_status = gr.Textbox(label="Node Status", interactive=False)

                gr.Markdown("---")

                # ── Inline Node Remove ────────────────────────────────────────
                with gr.Accordion("🗑️ Remove a Node from Draft", open=False):
                    remove_node_dd = gr.Dropdown(
                        label="Select Node to Remove",
                        choices=[],
                        info="This will also delete the node from the backend"
                    )
                    remove_node_btn = gr.Button("Remove Node", variant="stop")

                gr.Markdown("---")

                # ── Inline Node Form ──────────────────────────────────────────
                with gr.Accordion("➕ Add New Node to This Agent", open=True):
                    gr.Markdown("Fill in the details and click **Add Node** to create and attach it to this agent.")
                    with gr.Row():
                        with gr.Column():
                            n_name = gr.Textbox(label="Node Name *", placeholder="e.g. Greeting")
                            n_type = gr.Dropdown(
                                label="Node Type",
                                choices=["conversation", "function", "subagent"],
                                value="conversation"
                            )
                            n_global = gr.Checkbox(label="🌐 Global Node (reachable from any point)", value=False)
                            n_max_turns = gr.Number(label="Max Turns", value=3)
                            n_variables = gr.Textbox(
                                label="Variables Needed (comma-separated)",
                                placeholder="CustomerName, EMIAmount"
                            )
                            n_tools = gr.Textbox(
                                label="Tool IDs (comma-separated)",
                                placeholder="cbs_lookup, create_ptp"
                            )
                        with gr.Column():
                            n_prompt = gr.Textbox(
                                label="Node Prompt (LLM instruction) *",
                                lines=5,
                                placeholder="What should the agent do in this node?\ne.g. Greet the customer warmly, confirm their name."
                            )
                            n_static = gr.Textbox(
                                label="Static Message (optional — sent verbatim before LLM)",
                                lines=3,
                                placeholder="Hello {CustomerName}, I'm calling about your loan..."
                            )
                            n_transitions = gr.Textbox(
                                label="Transitions (one per line: to_node_id | condition | kw1, kw2)",
                                lines=4,
                                placeholder="node_abc123 | Customer confirms identity | yes, speaking, haan\nnode_def456 | Wrong number | wrong number, galat",
                                info="Use the node_id shown in draft above after adding the target node first."
                            )

                    add_node_btn = gr.Button("➕ Add Node to Agent", variant="primary")

            gr.Markdown("---")
            agent_status = gr.Textbox(label="Agent Creation Status", interactive=False)
            agent_create_btn = gr.Button("🚀 Create Agent", variant="primary", size="lg")

            # Consolidated JSON viewer — shows what was stored in DB after agent creation
            with gr.Accordion("📄 View Consolidated Agent Config JSON (stored in DB)", open=False):
                gr.Markdown(
                    "This is the **consolidated agent config JSON** built and stored in the database "
                    "when the agent was created. It follows Retell-style schema extended with internal fields "
                    "(tools, variables_needed, keywords, etc.). The orchestrator loads this at session-start "
                    "to avoid per-turn DB lookups."
                )
                agent_config_json_display = gr.Code(
                    label="agent_config_json",
                    language="json",
                    interactive=False,
                    lines=30,
                )

            # ── UPDATE EXISTING AGENT ─────────────────────────────────────────
            gr.Markdown("---")
            gr.Markdown("### ✏️ Update Existing Agent")
            gr.Markdown(
                "Select an agent to edit. Only the fields you change will be updated — "
                "everything else stays as-is."
            )

            with gr.Row():
                update_agent_dd = gr.Dropdown(label="Select Agent to Edit", choices=[], scale=3)
                update_agent_refresh_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)
                update_agent_load_btn = gr.Button("📥 Load Agent", variant="primary", scale=1)

            update_agent_status = gr.Markdown("_(Select an agent and click Load)_")

            with gr.Accordion("📝 Agent Fields", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        u_name = gr.Textbox(label="Agent Name", placeholder="Leave blank to keep current")
                        u_mode = gr.Dropdown(
                            label="Mode",
                            choices=["flow", "single_prompt"],
                            value="flow",
                        )
                        u_lang = gr.Dropdown(
                            label="Language",
                            choices=["english", "hindi", "hinglish"],
                            value="english",
                        )
                        u_personality = gr.Textbox(label="Personality", placeholder="professional, empathetic")
                        u_tools_str = gr.Textbox(
                            label="Tool IDs (comma-separated)",
                            placeholder="cbs_lookup, create_ptp"
                        )
                        u_start_node = gr.Textbox(
                            label="Start Node ID",
                            placeholder="node_xxxxx"
                        )
                    with gr.Column(scale=2):
                        u_system_prompt = gr.Textbox(
                            label="System Prompt",
                            lines=8,
                            placeholder="Edit system prompt here..."
                        )

                update_agent_save_btn = gr.Button("💾 Save Changes (Agent)", variant="primary", size="lg")
                update_agent_save_status = gr.Textbox(label="Update Status", interactive=False)

            # ── ADD NODE TO AGENT ──────────────────────────────────────────────
            with gr.Accordion("➕ Add Node to This Agent", open=False):
                gr.Markdown(
                    "Add a node to the selected agent — either **define a new node** from scratch, "
                    "or **attach an existing node** by its `node_id`."
                )

                add_node_method = gr.Radio(
                    label="How do you want to add the node?",
                    choices=[
                        ("Option 1 — Define a brand-new node", "option1"),
                        ("Option 2 — Attach an existing node by node_id", "option2"),
                    ],
                    value="option1",
                )

                # ── Option 1: Create a new node ───────────────────────────────
                with gr.Column(visible=True) as add_node_new_panel:
                    gr.Markdown("#### ✨ New Node Details")
                    with gr.Row():
                        with gr.Column():
                            an_name = gr.Textbox(label="Node Name *", placeholder="e.g. Verification")
                            an_type = gr.Dropdown(
                                label="Node Type",
                                choices=["conversation", "function", "subagent"],
                                value="conversation"
                            )
                            an_global = gr.Checkbox(label="🌐 Global Node", value=False)
                            an_max_turns = gr.Number(label="Max Turns", value=3)
                            an_variables = gr.Textbox(
                                label="Variables Needed (comma-separated)",
                                placeholder="CustomerName, EMIAmount"
                            )
                            an_tools = gr.Textbox(
                                label="Tool IDs (comma-separated)",
                                placeholder="cbs_lookup, create_ptp"
                            )
                        with gr.Column():
                            an_prompt = gr.Textbox(
                                label="Node Prompt (LLM instruction) *",
                                lines=5,
                                placeholder="What should the agent do in this node?"
                            )
                            an_static = gr.Textbox(
                                label="Static Message (optional)",
                                lines=3,
                                placeholder="Hello {CustomerName}, ..."
                            )
                            an_transitions = gr.Textbox(
                                label="Transitions (one per line: to_node_id | condition | kw1, kw2)",
                                lines=4,
                                placeholder="node_abc123 | Customer confirms identity | yes, speaking"
                            )

                    add_new_node_btn = gr.Button("➕ Create & Attach Node", variant="primary")

                # ── Option 2: Attach existing node ────────────────────────────
                with gr.Column(visible=False) as add_node_existing_panel:
                    gr.Markdown("#### 🔗 Attach Existing Node")
                    gr.Markdown(
                        "Enter the `node_id` of a node that already exists in the system. "
                        "It will be added to this agent's node list."
                    )
                    with gr.Row():
                        existing_node_id_input = gr.Textbox(
                            label="Existing Node ID *",
                            placeholder="node_xxxxxxxx",
                            scale=3,
                            info="The node_id of the existing node to attach"
                        )
                        existing_node_as_global = gr.Checkbox(
                            label="🌐 Attach as Global Node",
                            value=False,
                            scale=1
                        )

                    attach_existing_btn = gr.Button("🔗 Attach Existing Node", variant="primary")

                add_node_to_agent_status = gr.Textbox(label="Status", interactive=False)

            # ── UPDATE NODE (within selected agent) ───────────────────────────
            with gr.Accordion("🔗 Update Node (within selected agent)", open=False):
                gr.Markdown(
                    "Edit individual nodes of the selected agent. "
                    "Only changed fields will be updated."
                )
                update_nodes_summary = gr.Markdown("_(Load an agent first)_")

                with gr.Row():
                    update_node_dd = gr.Dropdown(label="Select Node to Edit", choices=[], scale=3)
                    update_node_load_btn = gr.Button("📥 Load Node", variant="primary", scale=1)

                update_node_status = gr.Markdown("_(Select a node)_")

                with gr.Row():
                    with gr.Column():
                        un_name = gr.Textbox(label="Node Name")
                        un_type = gr.Dropdown(
                            label="Node Type",
                            choices=["conversation", "function", "subagent"],
                            value="conversation"
                        )
                        un_global = gr.Checkbox(label="🌐 Global Node", value=False)
                        un_max_turns = gr.Number(label="Max Turns", value=3)
                        un_variables = gr.Textbox(
                            label="Variables Needed (comma-separated)",
                            placeholder="CustomerName, EMIAmount"
                        )
                        un_tools = gr.Textbox(
                            label="Tool IDs (comma-separated)",
                            placeholder="cbs_lookup, create_ptp"
                        )
                    with gr.Column():
                        un_prompt = gr.Textbox(
                            label="Node Prompt",
                            lines=5,
                            placeholder="LLM instruction for this node..."
                        )
                        un_static = gr.Textbox(
                            label="Static Message",
                            lines=3,
                            placeholder="Verbatim message before LLM..."
                        )
                        un_transitions = gr.Textbox(
                            label="Transitions (one per line: to_node_id | condition | kw1, kw2)",
                            lines=4,
                            placeholder="node_abc123 | Customer confirms | yes, haan"
                        )

                update_node_save_btn = gr.Button("💾 Save Changes (Node)", variant="primary")
                update_node_save_status = gr.Textbox(label="Node Update Status", interactive=False)

        # ── TAB 4: Chat ───────────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            gr.Markdown("### Chat with Dynamic Flow Agent")
            
            # This is the crucial fix: a per-browser-tab state variable
            session_state = gr.State(value=None)

            with gr.Row():
                chat_agent = gr.Dropdown(label="Agent", choices=[], scale=2)
                chat_customer = gr.Dropdown(label="Customer", choices=[], scale=2)
                chat_refresh_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)

            with gr.Row():
                chat_start_btn = gr.Button("📞 Start Call", variant="primary", scale=2)

            chat_status = gr.Textbox(label="Status", interactive=False)

            chatbot = gr.Chatbot(label="Conversation", height=450)

            with gr.Row():
                image_upload = gr.File(
                    label="Attach Document (Optional)",
                    file_types=[".jpg", ".jpeg", ".png", ".webp", ".pdf"],
                    scale=1,
                )
                chat_input = gr.Textbox(placeholder="Type response... (or leave blank if just uploading)", scale=4, show_label=False)
                chat_send_btn = gr.Button("Send ➤", variant="primary", scale=1)

            chat_clear_btn = gr.Button("🧹 Clear Chat", variant="stop")

    # ══════════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    # Tab 1 — Tool Registry
    tool_refresh_btn.click(fn=refresh_tools, outputs=[tool_list])
    t_exec_mode.change(fn=toggle_async_fields, inputs=[t_exec_mode], outputs=[async_fields_panel])
    t_content_type.change(
        fn=apply_content_type,
        inputs=[t_content_type, t_headers],
        outputs=[t_headers],
    )
    tool_create_btn.click(
        fn=create_tool,
        inputs=[
            t_name, t_desc, t_method, t_endpoint, t_params, t_headers, t_mapping,
            t_exec_mode, t_timeout_ms, t_on_complete, t_meanwhile,
        ],
        outputs=[tool_status],
    )

    # Tab 3 — Agent Builder: mode toggle
    a_mode.change(
        fn=toggle_mode_panels,
        inputs=[a_mode],
        outputs=[flow_panel, single_prompt_panel],
    )

    # Tab 3 — Agent Builder: Add node to draft
    add_node_btn.click(
        fn=add_node_to_draft,
        inputs=[
            n_name, n_type, n_prompt, n_static, n_variables, n_tools,
            n_global, n_max_turns, n_transitions, draft_state
        ],
        outputs=[draft_state, node_draft_status, draft_display, start_node_dd],
    ).then(
        # Also update the remove-node dropdown
        fn=get_draft_node_choices,
        inputs=[draft_state],
        outputs=[remove_node_dd],
    )

    # Tab 3 — Agent Builder: Remove node from draft
    remove_node_btn.click(
        fn=remove_node_from_draft,
        inputs=[remove_node_dd, draft_state],
        outputs=[draft_state, node_draft_status, draft_display, start_node_dd],
    ).then(
        fn=get_draft_node_choices,
        inputs=[draft_state],
        outputs=[remove_node_dd],
    )

    # Tab 3 — Agent Builder: Set start node (update display only)
    start_node_dd.change(
        fn=set_start_node_in_draft,
        inputs=[start_node_dd, draft_state],
        outputs=[draft_state, draft_display],
    )

    # Tab 3 — Agent Builder: Clear draft
    draft_clear_btn.click(
        fn=clear_draft,
        outputs=[draft_state, draft_display, start_node_dd, node_draft_status],
    ).then(
        fn=lambda: gr.update(choices=[]),
        outputs=[remove_node_dd],
    )

    # Tab 3 — Agent Builder: Create agent
    agent_refresh_btn.click(fn=refresh_agents, outputs=[agent_list, agent_select])
    agent_delete_btn.click(fn=delete_agent, inputs=[agent_select], outputs=[agent_delete_status])
    agent_create_btn.click(
        fn=create_agent_with_nodes,
        inputs=[a_name, a_mode, a_prompt, a_lang, a_personality, a_tools_str, draft_state, start_node_dd],
        outputs=[agent_status, agent_config_json_display],
    )

    # Tab 3 — Agent Update: Refresh dropdown
    update_agent_refresh_btn.click(
        fn=get_update_agent_choices,
        outputs=[update_agent_dd],
    )

    # Tab 3 — Agent Update: Load agent into form
    update_agent_load_btn.click(
        fn=load_agent_for_update,
        inputs=[update_agent_dd],
        outputs=[u_name, u_mode, u_system_prompt, u_lang, u_personality, u_tools_str, u_start_node, update_agent_status],
    ).then(
        fn=load_agent_nodes_for_update,
        inputs=[update_agent_dd],
        outputs=[update_node_dd, update_nodes_summary],
    )

    # Tab 3 — Agent Update: Save agent changes
    update_agent_save_btn.click(
        fn=save_agent_update,
        inputs=[update_agent_dd, u_name, u_mode, u_system_prompt, u_lang, u_personality, u_tools_str, u_start_node],
        outputs=[update_agent_save_status],
    )

    # Tab 3 — Add Node to Agent: toggle method panel
    add_node_method.change(
        fn=toggle_add_node_option,
        inputs=[add_node_method],
        outputs=[add_node_new_panel, add_node_existing_panel],
    )

    # Tab 3 — Add Node to Agent: Option 1 — create & attach new node
    add_new_node_btn.click(
        fn=add_new_node_to_agent,
        inputs=[
            update_agent_dd,
            an_name, an_type, an_prompt, an_static, an_variables,
            an_tools, an_global, an_max_turns, an_transitions,
        ],
        outputs=[add_node_to_agent_status, update_node_dd],
    )

    # Tab 3 — Add Node to Agent: Option 2 — attach existing node
    attach_existing_btn.click(
        fn=add_existing_node_to_agent,
        inputs=[update_agent_dd, existing_node_id_input, existing_node_as_global],
        outputs=[add_node_to_agent_status, update_node_dd],
    )

    # Tab 3 — Node Update: Load node into form
    update_node_load_btn.click(
        fn=load_node_for_update,
        inputs=[update_node_dd],
        outputs=[un_name, un_type, un_prompt, un_static, un_variables, un_tools, un_global, un_max_turns, un_transitions, update_node_status],
    )

    # Tab 3 — Node Update: Save node changes
    update_node_save_btn.click(
        fn=save_node_update,
        inputs=[update_node_dd, un_name, un_type, un_prompt, un_static, un_variables, un_tools, un_global, un_max_turns, un_transitions],
        outputs=[update_node_save_status],
    )

    # Tab 4 — Chat
    chat_refresh_btn.click(fn=refresh_chat_dropdowns, outputs=[chat_agent, chat_customer])
    
    # start_chat returns: chatbot, chat_status, session_state
    chat_start_btn.click(
        fn=start_chat, 
        inputs=[chat_agent, chat_customer], 
        outputs=[chatbot, chat_status, session_state]
    )
    
    # send_message takes session_state, returns: chatbot, chat_input, chat_status, image_upload
    chat_send_btn.click(
        fn=send_message,
        inputs=[chat_input, chatbot, session_state, chat_agent, chat_customer, image_upload],
        outputs=[chatbot, chat_input, chat_status, image_upload]
    )
    chat_input.submit(
        fn=send_message,
        inputs=[chat_input, chatbot, session_state, chat_agent, chat_customer, image_upload],
        outputs=[chatbot, chat_input, chat_status, image_upload]
    )
    # Clear chat also clears the session state
    chat_clear_btn.click(
        fn=lambda: ([], "", "", None, None), 
        outputs=[chatbot, chat_input, chat_status, image_upload, session_state]
    )


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Knight Flow Generic — Dynamic Multi-Agent Platform")
    print("  Backend:  python main.py  (port 8101)")
    print("  Frontend: python gradio_app.py")
    print("=" * 55 + "\n")
    app.launch(server_port=1338, share=True)
