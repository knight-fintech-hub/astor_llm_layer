"""
orchestrator.py — Runtime Orchestration Engine
================================================
The brain of the system. Handles:
  1. Starting conversations (create session, move to start node)
  2. Processing messages (evaluate transitions, execute tools, generate responses)
  3. Node transitions (keyword match → LLM evaluation → move)
  4. Global node handling (out-of-scope, escalation)
  5. Agent swapping (transfer memory between agents)

This replaces the hardcoded workflow_engine.py with a fully dynamic system
that reads nodes/edges/tools from JSON (configured via UI).
"""

from __future__ import annotations

import os
import re
import time
from datetime import date
from typing import Dict, List, Optional
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import cfg
from logger import get_logger
from node_manager import node_store
from tool_registry import (
    tool_store,
    dispatch_tool,
    is_async_stub,
    format_tool_result,
    format_background_result,
    poll_completed_jobs,
)
# ── Agent store: DB or local JSON based on USE_DB env flag ───────────────────
if cfg.USE_DB:
    from agent_manager import agent_store
else:
    from local_agent_loader import local_agent_store as agent_store  # type: ignore[assignment]
from memory_store import memory_store, Session
from llm_engine import llm_engine, llm_queue



logger = get_logger(__name__)

orchestrator_router = APIRouter(prefix="/api/chat", tags=["Orchestrator"])

_STORE_TAG_PATTERN = re.compile(r'\[STORE:\s*(.+?)\]', re.DOTALL)


def extract_and_store_variables(response_text: str, session: "Session") -> str:
    """Extract [STORE: key=val, key=val, ...] from LLM response, save to session, remove tag."""
    match = _STORE_TAG_PATTERN.search(response_text)
    if match:
        raw_pairs = match.group(1).split(',')
        stored = {}
        for pair in raw_pairs:
            if '=' in pair:
                key, value = pair.strip().split('=', 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    session.set_variable(key, value)
                    stored[key] = value
        if stored:
            logger.info(f"[STORE] Extracted {len(stored)} variables from LLM response: {list(stored.keys())}")
        # Remove all [STORE: ...] tags from response
        response_text = _STORE_TAG_PATTERN.sub('', response_text).strip()
    return response_text
# ── Async Tool Instructions (generic, one-time — see plan.md §6) ────────────
# Static text, safe to KV-cache. Applies to every agent so a per-tool note is
# only needed when a tool's *stub behavior* isn't obvious from its name.
ASYNC_TOOL_INSTRUCTIONS = (
    "Some tool results say \"Tool Dispatched\" instead of giving a real answer — "
    "that tool is running in the background. When you see this: do NOT say you "
    "are \"processing\" more than once, do NOT call that tool again to check on "
    "it, and do NOT invent a question just to stall. Move to the next useful "
    "step, or wait briefly and naturally if there is none. The real result is "
    "delivered to you automatically on a later turn — weave it in naturally then."
)


def _record_tool_result(session: "Session", tool_id: str, new_vars: Dict) -> None:
    """Generic per-tool retry counter.

    Increments TOOL_RETRY_COUNTS[tool_id] when a tool returns an empty /
    None result (failed call). Resets that tool's counter to 0 on a
    successful (non-empty) result. Stored without a leading underscore so
    build_session_context() exposes it to the LLM — the prompt handles
    deciding what to do when a count is high (e.g. call human-in-loop).
    """
    counts: Dict[str, int] = dict(session.variables.get("TOOL_RETRY_COUNTS", {}))
    if not new_vars:
        counts[tool_id] = counts.get(tool_id, 0) + 1
        logger.info(
            f"[Orchestrator] session={session.session_id} | "
            f"TOOL_RETRY_COUNT[{tool_id}] → {counts[tool_id]}"
        )
    elif counts.get(tool_id):
        counts[tool_id] = 0
        logger.info(
            f"[Orchestrator] session={session.session_id} | "
            f"TOOL_RETRY_COUNT[{tool_id}] reset (valid response)"
        )
    session.set_variable("TOOL_RETRY_COUNTS", counts)



# ── Cache-First Node Lookup ────────────────────────────────────────────────

def get_node(session: "Session", node_id: str) -> Optional[Dict]:
    """
    Cache-first node lookup. Returns the node dict from `session.node_cache`
    if populated (fast, no DB), otherwise falls back to `node_store.get()`.
    The cache is built once from `agent_config_json` when a session starts.
    """
    if session.node_cache:
        return session.node_cache.get(node_id)
    # Fallback: direct DB lookup (for sessions started before cache was introduced)
    return node_store.get(node_id)


def _build_node_cache_from_config(config_json: Dict) -> Dict[str, Dict]:
    """
    Build the session node_cache from an agent's consolidated config JSON.
    Converts Retell-style node/edge format back to the internal format used
    by the orchestrator (transitions[], prompt from instruction.text, etc.).
    Returns a dict keyed by node_id.
    """
    cache: Dict[str, Dict] = {}
    flow = config_json.get("conversationFlow", {})
    for node in flow.get("nodes", []):
        # Convert Retell-style `edges` → internal `transitions`
        transitions = []
        for edge in node.get("edges", []):
            transitions.append({
                "to_node": edge.get("destination_node_id"),
                "condition": edge.get("condition", edge.get("transition_condition", {}).get("prompt", "")),
                "keywords": edge.get("keywords", []),
                "on_transition_tool": edge.get("on_transition_tool"),
            })

        internal_node = {
            "node_id": node["id"],
            "name": node["name"],
            "type": node.get("type", "conversation"),
            "prompt": node.get("instruction", {}).get("text", ""),
            "static_message": node.get("static_message"),
            "variables_needed": node.get("variables_needed", []),
            "tools": [
                # node-level tool_ids: derive from tool names if embedded, else empty
                # (node-level tool IDs are stored in consolidated JSON as tool_id strings)
                t if isinstance(t, str) else t.get("tool_id", "")
                for t in node.get("node_tools", [])
            ],
            "is_global": node.get("is_global", False),
            "max_turns": node.get("max_turns", 3),
            "transitions": transitions,
            "created_at": "",
        }
        cache[node["id"]] = internal_node

    return cache


# ── Request/Response Models ──────────────────────────────────────────────────

class StartChatRequest(BaseModel):
    agent_id: str = Field(..., description="Agent ID")
    mobile: Optional[str] = Field(None, description="Customer mobile for data loading")


class StartChatResponse(BaseModel):
    session_id: str
    agent_id: str
    agent_name: str
    greeting: str
    current_node: str
    current_node_name: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")
    message: str = Field(..., min_length=1, description="User message")
    image_base64: Optional[str] = Field(None, description="Optional base64 image payload")


class ChatResponse(BaseModel):
    response: str
    session_id: str
    agent_name: str
    current_node: str
    current_node_name: str
    transitioned: bool
    duration_ms: float


# ── Out-of-Scope Detection ──────────────────────────────────────────────────

OUT_OF_SCOPE_INDICATORS = [
    "weather", "cricket", "movie", "song", "joke", "news", "politics",
    "recipe", "game", "stock market", "share price", "bitcoin",
    "write a poem", "tell me a story", "who is the president",
    "what is ai", "general knowledge", "history", "geography",
    "how are you", "what is your name", "your opinion",
    # Hindi/Hinglish (Roman script)
    "mausam", "gaana", "chutkula", "khabar", "rajneeti",
    "khana kaise banta", "recipe kaise", "kahani sunao",
    "kavita likho", "cricket match", "film", "picture kaun si",
    "tum kaise ho", "aap kaise ho", "tumhara naam kya hai",
    "tumhari raay", "president kaun hai",
]

LOAN_RELATED_WORDS = [
    "loan", "emi", "payment", "pay", "account", "balance",
    "transaction", "charge", "due", "overdue", "outstanding",
    "installment", "principal", "interest", "ptp", "promise",
    # Hindi/Hinglish (Roman script)
    "paisa", "paise", "rupaye", "rupya", "kist", "kisht",
    "bakaya", "udhaar", "karza", "karz", "khata", "jama",
    "vaada", "wapas", "chukana", "chukauta",
]


def is_out_of_scope(message: str) -> bool:
    """Check if message is off-topic."""
    msg_lower = message.lower()

    # If it contains loan-related words, it's in-scope
    if any(word in msg_lower for word in LOAN_RELATED_WORDS):
        return False

    # Check explicit out-of-scope indicators
    if any(indicator in msg_lower for indicator in OUT_OF_SCOPE_INDICATORS):
        return True

    # Check if it's a generic question not related to financial topics
    generic_patterns = [
        "who is", "what is", "tell me about", "explain",
        "how to", "why is", "when was", "where is",
        "write", "create", "generate", "make me",
        # Hindi/Hinglish (Roman script)
        "kaun hai", "kya hai", "bataiye", "samjhao", "samjhaiye",
        "kaise kare", "kyun hai", "kab tha", "kahan hai", "likh do",
    ]
    if any(p in msg_lower for p in generic_patterns):
        if not any(word in msg_lower for word in LOAN_RELATED_WORDS):
            return True

    return False


ESCALATION_INDICATORS = [
    "talk to a human", "talk to a person", "real agent", "human agent",
    "speak to your manager", "speak to a manager", "manager", "supervisor",
    "senior agent", "connect me to someone", "customer care", "complaint",
    # Hindi/Hinglish
    "insaan se baat", "manager se baat", "supervisor se baat",
    "senior se baat", "koi aur agent", "complaint karni hai",
]


def is_escalation_request(message: str) -> bool:
    """Check if customer wants to be transferred to a human/senior agent."""
    msg_lower = message.lower()
    return any(indicator in msg_lower for indicator in ESCALATION_INDICATORS)


def check_global_intercept(agent: Dict, user_message: str, session: Session) -> Optional[str]:
    """
    Check out-of-scope / escalation triggers that should interrupt the current node
    regardless of which node the conversation is in. Returns a response string if
    intercepted, or None to continue normal node processing.
    """
    checks = [
        ("scope", is_out_of_scope, "I am not an expert in this. Let me forward your query to another agent who can help you better."),
        ("escalat", is_escalation_request, "I understand. Let me connect you with a senior agent who can assist you further."),
    ]

    global_nodes = agent.get("global_nodes", [])
    # When node_cache is active, resolve global nodes from the cache
    if session.node_cache:
        global_node_ids = agent.get("agent_config_json", {}).get("conversationFlow", {}).get("global_node_ids", [])
        global_nodes = global_node_ids if global_node_ids else global_nodes
    for name_key, detector, fallback_msg in checks:
        if not detector(user_message):
            continue

        response = fallback_msg
        for gn_id in global_nodes:
            gn = get_node(session, gn_id)
            if gn and name_key in gn.get("name", "").lower():
                response = gn.get("static_message") or fallback_msg
                break

        return resolve_template(response, session.variables)

    return None


# ── Variable Resolution ──────────────────────────────────────────────────────

def _compute_dynamic_variable(name: str, variables: Dict) -> Optional[str]:
    """
    Live/derived variables that aren't stored directly in session state
    (Retell-style "dynamic variables") — computed fresh every time they're
    resolved, instead of relying on a pre-stored value that can go stale.
    """
    if name in ("DaysPastDue", "DPDLive"):
        due_date = variables.get("DueDate")
        if due_date:
            try:
                due = date.fromisoformat(str(due_date)[:10])
                return str(max((date.today() - due).days, 0))
            except ValueError:
                return None
    return None


def resolve_template(template: str, variables: Dict) -> str:
    """
    Replace {Variable} and {{Variable}} placeholders in a template with
    session data, or a live-computed value for dynamic variables (e.g.
    {{DaysPastDue}}). Unresolved placeholders are left as-is, not dropped.

    Lookup is case-insensitive: {{lender_name}}, {LenderName}, {LENDERNAME}}
    all resolve to the same session variable.
    """
    # Build a lowercase → value index for case-insensitive fallback
    lower_index = {k.lower(): v for k, v in variables.items()}

    def replacer(match):
        var_name = match.group(1)
        # 1. Exact match (fast path)
        if var_name in variables:
            return str(variables[var_name])
        # 2. Case-insensitive match (handles {{lender_name}} vs LenderName)
        lower_key = var_name.lower().replace("_", "")  # strip underscores too
        for k, v in lower_index.items():
            if k.replace("_", "") == lower_key:
                return str(v)
        # 3. Dynamic computed variable
        computed = _compute_dynamic_variable(var_name, variables)
        if computed is not None:
            return computed
        return match.group(0)

    # Double-brace (Retell-style {{Var}}) first, then single-brace {Var}
    template = re.sub(r"\{\{(\w+)\}\}", replacer, template)
    template = re.sub(r"\{(\w+)\}", replacer, template)
    return template


# ── Pre-Execution Variable Extraction ────────────────────────────────────────

def _get_tool_required_variables(tool_id: str) -> List[str]:
    """Extract all {Variable} placeholder names from a tool's endpoint and request_body."""
    tool = tool_store.get(tool_id)
    if not tool:
        return []
    all_vars = set()
    all_vars.update(re.findall(r"\{(\w+)\}", tool.get("endpoint", "")))
    
    def extract_vars_from_body(body: Any) -> set:
        vars_set = set()
        if isinstance(body, dict):
            for v in body.values():
                vars_set.update(extract_vars_from_body(v))
        elif isinstance(body, list):
            for v in body:
                vars_set.update(extract_vars_from_body(v))
        elif isinstance(body, str):
            vars_set.update(re.findall(r"\{(\w+)\}", body))
        return vars_set
        
    all_vars.update(extract_vars_from_body(tool.get("request_body", {})))
    return list(all_vars)


async def extract_and_execute_tool(
    tool_id: str,
    session: "Session",
    user_message: str,
    context: str = "",
) -> Dict[str, str]:
    """
    Smart tool execution: before calling the external API, find which variables
    the tool needs (from its params/endpoint), check which are missing from the
    session, and use the LLM to extract them from the user's latest message.
    Then execute the tool with the now-populated session variables.
    """
    required_vars = _get_tool_required_variables(tool_id)
    # missing_vars = [v for v in required_vars if not session.variables.get(v)]

    # Fallback for common patterns to avoid reliance on LLM extraction for simple IDs
    if "MobileNumber" in required_vars:
        import re
        match = re.search(r'\b[6-9]\d{9}\b|\b\d{10}\b', user_message)
        if match:
            mobile = match.group(0)
            session.set_variable("MobileNumber", mobile)
            session.set_variable("mobile_number", mobile)
            session.set_variable("phoneNumber", mobile)
            session.set_variable("mobile", mobile)
            # missing_vars.remove("MobileNumber")

    if required_vars and llm_engine.is_loaded:
        # if not context:
        #     recent_msgs = session.history[-4:] if session.history else []
        #     context = " | ".join(f"{m['role']}: {m['content']}" for m in recent_msgs)

        extracted = await llm_queue.run(llm_engine.extract_variables, user_message, required_vars)
        if extracted:
            session.set_variables(extracted)
            logger.info(
                f"[Orchestrator] Pre-tool extraction for {tool_id}: "
                f"needed={required_vars} → extracted={list(extracted.keys())}"
            )

    still_missing = [v for v in required_vars if not session.variables.get(v)]
    if still_missing:
        logger.info(f"[Orcestrator] Tool {tool_id} still missing after extraction: {still_missing}"
                    f"(will ateempt tool call anyway  - API may handle defaults)")
        
    new_vars = await dispatch_tool(tool_id, session.variables, session.session_id)
    return new_vars


# ── Transition Evaluation ────────────────────────────────────────────────────

async def evaluate_transitions(
    session: Session,
    user_message: str,
    use_llm: bool = True,
) -> Optional[str]:
    """
    Pure LLM transition evaluation.
    Single LLM call evaluates ALL transitions at once (multi-choice).
    No keywords — fully LLM-driven like Retell AI.
    """
    current_node_data = get_node(session, session.current_node)
    if not current_node_data:
        return None

    transitions = current_node_data.get("transitions", [])
    if not transitions:
        return None

    # If LLM not loaded, can't evaluate
    if not llm_engine.is_loaded:
        logger.warning("[Orchestrator] LLM not loaded, cannot evaluate transitions")
        return None

    # Special case: single transition with catch-all condition → always transition
    # (e.g., verification node: any response = move forward)
    if len(transitions) == 1:
        condition = transitions[0].get("condition", "").lower()
        # If condition is broadly accepting (any input), just transition
        if any(word in condition for word in ["any", "provides", "acknowledges", "after closing"]):
            logger.info(f"[Orchestrator] Single catch-all transition → {transitions[0]['to_node']}")
            return transitions[0]["to_node"]

    # Build conversation context from recent history
    recent_msgs = session.history[-6:] if session.history else []
    context = " | ".join(f"{m['role']}: {m['content']}" for m in recent_msgs)

    # Single LLM call — evaluate all transitions at once. Offloaded to a worker
    # thread so this blocking model.generate() call doesn't freeze the event
    # loop out from under any dispatched async tool tasks.
    target = await llm_queue.run(
        llm_engine.evaluate_all_transitions,
        user_message=user_message,
        transitions=transitions,
        current_node_name=current_node_data["name"],
        conversation_context=context,
    )

    return target


async def decide_transition_and_tool(
    session: Session,
    agent: Dict,
    node: Optional[Dict],
    user_message: str,
    skip_transition: bool = False,
) -> "tuple[Optional[str], Optional[str]]":
    """
    Flow-mode combined decision: evaluate transitions AND agent-level tool
    intent using as few LLM calls as possible (ideally one instead of two).
    Preserves the existing no-LLM fast paths (catch-all transition, no
    transitions, no tools) so calls are only spent when actually needed.

    Returns (target_node_id_or_None, tool_result_summary_or_None).
    """
    transitions = [] if skip_transition else (node.get("transitions", []) if node else [])

    tool_ids = set(agent.get("tools", []))
    available_tools = []
    for tid in tool_ids:
        tool = tool_store.get(tid)
        # Exclude image/file-upload tools from the LLM's candidate list when no
        # image is actually pending — the tool has no real file data to send,
        # so letting the LLM pick it anyway dispatches the call with an
        # unresolved {InputImage}/{InputFile} placeholder instead of real bytes.
        if tool and _has_input_image(tool.get("request_body", {})) and not (session.variables.get("InputImage") or session.variables.get("InputFile")):
            continue
        if tool:
            available_tools.append({
                "tool_id": tid,
                "name": tool["name"],
                "description": tool.get("description", ""),
            })

    # Fast path: single catch-all transition → always transition, no LLM needed for it
    catch_all_target = None
    if len(transitions) == 1:
        condition = transitions[0].get("condition", "").lower()
        if any(word in condition for word in ["any", "provides", "acknowledges", "after closing"]):
            catch_all_target = transitions[0]["to_node"]
            logger.info(f"[Orchestrator] Single catch-all transition → {catch_all_target}")

    needs_transition_llm = bool(transitions) and catch_all_target is None and llm_engine.is_loaded
    needs_tool_llm = bool(available_tools) and llm_engine.is_loaded

    target_node_id = catch_all_target
    tool_id = None

    if needs_transition_llm and needs_tool_llm:
        recent_msgs = session.history[-6:] if session.history else []
        context = " | ".join(f"{m['role']}: {m['content']}" for m in recent_msgs)
        decision = await llm_queue.run(
            llm_engine.decide_transition_and_tool,
            user_message=user_message,
            transitions=transitions,
            current_node_name=node["name"],
            available_tools=available_tools,
            conversation_context=context,
        )
        target_node_id = decision.get("transition")
        tool_id = decision.get("tool_id")
    elif needs_transition_llm:
        target_node_id = await evaluate_transitions(session, user_message, use_llm=True)
    elif needs_tool_llm:
        recent_msgs = session.history[-4:] if session.history else []
        context = " | ".join(f"{m['role']}: {m['content']}" for m in recent_msgs)
        tool_id = await llm_queue.run(llm_engine.detect_tool_intent, user_message, available_tools, context)

    tool_result = None
    if tool_id:
        new_vars = await extract_and_execute_tool(tool_id, session, user_message)
        if new_vars:
            if is_async_stub(new_vars):
                logger.info(f"[Orchestrator] Combined-decision tool dispatched async: {tool_id} | job={new_vars.get('_async_job_id')}")
            else:
                _clear_image_from_session(session, new_vars)  # no-op if not an image tool
                session.set_variables(new_vars)
                logger.info(f"[Orchestrator] Combined-decision tool executed: {tool_id} | new vars: {len(new_vars)}")
            tool_result = format_tool_result(tool_id, new_vars)

    return target_node_id, tool_result


# ── Flex Mode: LLM jumps to best node when no edge matches ───────────────────

async def flex_mode_jump(
    session: Session,
    agent: Dict,
    user_message: str,
) -> Optional[str]:
    """
    Edge-constrained fallback (Retell-style): when the normal transition
    decision returns no target, only reconsider the CURRENT node's own
    declared edge targets — never jump to an arbitrary node elsewhere in the
    graph. This prevents the conversation from teleporting to an unrelated
    node when a new/under-specified flow has vague conditions.

    Behavior is gated by cfg.FLEX_MODE_ENABLED:
      - False  → flex mode is fully off; the conversation simply stays put
                 (max_turns forcing still applies downstream).
      - True   → the LLM may only pick among the current node's edge targets
                 (i.e. legal transitions), not the whole node list.
    """
    if not llm_engine.is_loaded:
        return None

    # Global kill-switch — default OFF for edge-strict (Retell-like) behavior.
    if not getattr(cfg, "FLEX_MODE_ENABLED", False):
        return None

    current = session.current_node
    current_node = get_node(session, current)
    if not current_node:
        return None

    # Only the current node's legal edge destinations are candidates.
    legal_targets = [
        t.get("to_node")
        for t in current_node.get("transitions", [])
        if t.get("to_node") and t.get("to_node") != "__END__"
    ]
    if not legal_targets:
        return None

    available = []
    for nid in legal_targets:
        if nid == current:
            continue
        node = get_node(session, nid)
        if node and not node.get("is_global"):
            available.append({
                "node_id": nid,
                "name": node["name"],
                "prompt": node.get("prompt", ""),
            })

    if not available:
        return None

    # Build context
    recent_msgs = session.history[-4:] if session.history else []
    context = " | ".join(f"{m['role']}: {m['content'][:50]}" for m in recent_msgs)

    target = await llm_queue.run(llm_engine.flex_mode_evaluate, user_message, available, context)

    # Safety: never trust a target outside the legal edge set.
    if target and target not in legal_targets:
        logger.warning(f"[Orchestrator] Flex Mode returned illegal target '{target}' — ignoring")
        return None

    if target:
        logger.info(f"[Orchestrator] Flex Mode JUMP (edge-constrained): {current} → {target}")
    return target


# ── Subagent: LLM decides which tool to call ─────────────────────────────────

async def subagent_tool_execution(
    session: Session,
    node: Dict,
    user_message: str,
) -> Dict[str, str]:
    """
    Subagent Node: LLM autonomously decides if/which tool to call.
    Returns new variables from tool execution (empty dict if no tool called).

    If `InputImage` is present in session variables AND `_image_tool_fired` is
    NOT set, whichever of the node's tools consumes {InputImage} is force-selected
    (OCR path) instead of letting the LLM pick from the full tool list.
    After the tool runs, `_clear_image_from_session()` pops `InputImage` and
    sets `_image_tool_fired` so subsequent turns don't re-trigger OCR.
    """
    if not llm_engine.is_loaded:
        return {}

    tool_ids = node.get("tools", [])
    if not tool_ids:
        return {}

    # ── Image OCR fast-path: image already uploaded ──────────────────────────
    # Only fire when there is a pending image AND the image tool hasn't already
    # run for this upload (_image_tool_fired sentinel). This prevents re-triggering
    # on the user's next acknowledgement turn (e.g. "Yes, that's correct").
    if (session.variables.get("InputImage") or session.variables.get("InputFile")) and not session.variables.get("_image_tool_fired"):
        image_tool_id = await _find_image_tool(tool_ids, user_message, session, node=node)
        if image_tool_id:
            logger.info(f"[Orchestrator] InputImage/InputFile detected → forcing image tool: {image_tool_id} (subagent)")
            new_vars = await extract_and_execute_tool(image_tool_id, session, user_message)
            if new_vars:
                _clear_image_from_session(session, new_vars)  # pop InputImage/InputFile + set sentinel
                if not is_async_stub(new_vars):
                    session.set_variables(new_vars)
                    logger.info(f"[Orchestrator] Image OCR ({image_tool_id}) subagent → new vars: {list(new_vars.keys())}")
                else:
                    logger.info(f"[Orchestrator] Image OCR ({image_tool_id}) subagent dispatched async | job={new_vars.get('_async_job_id')}")
            return new_vars

    # Build tool descriptions for LLM
    available_tools = []
    for tid in tool_ids:
        tool = tool_store.get(tid)
        # unresolved {InputImage}/{InputFile} placeholder instead of real bytes.
        if tool and _has_input_image(tool.get("request_body", {})) and not (session.variables.get("InputImage") or session.variables.get("InputFile")):
            continue
        if tool:
            available_tools.append({
                "tool_id": tid,
                "name": tool["name"],
                "description": tool.get("description", ""),
            })

    if not available_tools:
        return {}

    # LLM decides which tool (if any) to call
    recent_msgs = session.history[-4:] if session.history else []
    context = " | ".join(f"{m['role']}: {m['content'][:50]}" for m in recent_msgs)

    tool_id = await llm_queue.run(llm_engine.detect_tool_intent, user_message, available_tools, context)

    if tool_id:
        new_vars = await extract_and_execute_tool(tool_id, session, user_message, context)
        if is_async_stub(new_vars):
            logger.info(f"[Orchestrator] Subagent dispatched tool async: {tool_id} | job={new_vars.get('_async_job_id')}")
        else:
            session.set_variables(new_vars)
            logger.info(f"[Orchestrator] Subagent executed tool: {tool_id} | new vars: {len(new_vars)}")
        return new_vars

    return {}


# ── Subflow Management ───────────────────────────────────────────────────────

import json as _json

SUBFLOWS_FILE = os.path.join(cfg.DATA_DIR, "subflows.json")


def load_subflows() -> List[Dict]:
    """Load subflows from JSON."""
    if os.path.exists(SUBFLOWS_FILE):
        try:
            with open(SUBFLOWS_FILE, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            pass
    return []


def save_subflows(subflows: List[Dict]):
    """Save subflows to JSON."""
    with open(SUBFLOWS_FILE, "w", encoding="utf-8") as f:
        _json.dump(subflows, f, indent=2, ensure_ascii=False)


def get_subflow(subflow_id: str) -> Optional[Dict]:
    """Get a subflow by ID."""
    subflows = load_subflows()
    for sf in subflows:
        if sf["subflow_id"] == subflow_id:
            return sf
    return None


# ── Declarative Trigger Engine (Retell-style "→ trigger X → Node Y") ─────────

async def fire_transition_tool(node: Optional[Dict], to_node: str, session: Session) -> Optional[str]:
    """
    If the transition that leads to `to_node` declares an `on_transition_tool`,
    execute it now — this is the real, code-executed version of prose like
    "→ trigger X → Node Y". Config-driven, not LLM-guessed, so it's reliable
    regardless of model size.

    Returns a tool-result summary string to inject into the next response,
    or None if no trigger tool fired.
    """
    if not node:
        return None

    for t in node.get("transitions", []):
        if t.get("to_node") != to_node:
            continue
        tool_id = t.get("on_transition_tool")
        if not tool_id:
            return None

        new_vars = await dispatch_tool(tool_id, session.variables, session.session_id)
        if new_vars and not is_async_stub(new_vars):
            session.set_variables(new_vars)
            logger.info(f"[Orchestrator] Transition-trigger executed: {tool_id} → {to_node}")
        elif new_vars:
            logger.info(f"[Orchestrator] Transition-trigger dispatched async: {tool_id} → {to_node} | job={new_vars.get('_async_job_id')}")

        return format_tool_result(tool_id, new_vars)

    return None


# ── Node Execution ───────────────────────────────────────────────────────────

def build_node_system_prompt(agent: Dict, node: Dict, session: Session, note: Optional[str] = None) -> str:
    """Build the STATIC system prompt for the current node.

    Variable-dependent content (session.variables dump, tool results) is
    intentionally excluded here so the returned string stays stable across
    turns and can be used as a KV-cache key.  Pass that content separately
    via build_session_context() + generate_response(dynamic_context=...).
    """
    parts = []

    # Response constraint
    parts.append(
        "CRITICAL: You are on a phone call. Keep responses to 1-2 sentences MAX. "
        "Be direct, natural, and to the point. No bullet points or explanations."
    )

    parts.append(ASYNC_TOOL_INSTRUCTIONS)

    # Out-of-scope rule
    parts.append(
        "If the customer asks anything outside loan collection, EMI, payments, "
        "or account queries, reply EXACTLY: "
        "\"I am not an expert in this. Let me forward your query to another agent who can help you better.\""
    )

    # Agent base prompt
    parts.append(f"\n{resolve_template(agent['system_prompt'], session.variables)}")

    # Language
    language = agent.get("language", "english")
    if language == "hinglish":
        parts.append("\nRespond in Hinglish (Hindi-English mix, Roman script).")
    elif language == "hindi":
        parts.append("\nRespond in Hindi.")

    # Node-specific prompt
    parts.append(f"\n\nCURRENT NODE: {node['name']}")
    parts.append(f"INSTRUCTION: {resolve_template(node['prompt'], session.variables)}")

    # ── Document OCR image context ────────────────────────────────────────────
    if session.variables.get("InputImage") or session.variables.get("InputFile"):
        parts.append(
            "\nNOTE: The customer has uploaded a document image. "
            "OCR will be triggered automatically on the correct tool — do NOT ask for the document number manually."
        )
    elif session.variables.get("Status") and session.variables.get("CustomerName"):
        # OCR already ran — confirm to LLM that we have the extracted data
        parts.append(
            "\nNOTE: Document OCR has already been completed. "
            "Use the extracted customer data (CustomerName, DOB, Gender) from the session variables below."
        )

    if note:
        parts.append(f"IMPORTANT: {note}")

    # Suggested response (if static_message exists)
    static_msg = node.get("static_message")
    if static_msg:
        resolved = resolve_template(static_msg, session.variables)
        parts.append(f"SUGGESTED RESPONSE: {resolved}")
        parts.append("Use the suggested response as your reply, adapted naturally.")

    # NOTE: session.variables are NOT included here — they go into dynamic_context
    # via build_session_context() so the cache key stays stable.

    return "\n".join(parts)


async def execute_node_tools(node: Dict, session: Session):
    """Execute tools assigned to a node (function nodes run tools on entry)."""
    tool_ids = node.get("tools", [])
    for tool_id in tool_ids:
        if node.get("type") == "function":
            # Function nodes always execute tools
            new_vars = await dispatch_tool(tool_id, session.variables, session.session_id)
            if is_async_stub(new_vars):
                logger.info(f"[Orchestrator] Function node tool dispatched async: {tool_id} | job={new_vars.get('_async_job_id')}")
            else:
                session.set_variables(new_vars)
        # Conversation/subagent nodes — tools executed by orchestrator when needed


async def extract_node_variables(session: Session, node: Optional[Dict], user_message: str) -> Dict[str, str]:
    """
    Extract structured fields (e.g. DOB, PTPDate, PTPAmount) that this node needs
    from the customer's free-text reply, and store them as session variables.
    Only asks the LLM for fields that are declared in variables_needed and not
    already known — existing values (e.g. from CBS lookup) are left untouched.
    """
    if not node or not llm_engine.is_loaded:
        return {}

    needed = node.get("variables_needed", [])
    missing = [v for v in needed if not session.variables.get(v)]
    if not missing:
        return {}

    recent_msgs = session.history[-4:] if session.history else []
    context = " | ".join(f"{m['role']}: {m['content'][:50]}" for m in recent_msgs)

    extracted = await llm_queue.run(llm_engine.extract_variables, user_message, missing, context)
    if extracted:
        session.set_variables(extracted)
        logger.info(f"[Orchestrator] Extracted variables: {list(extracted.keys())} for node={node.get('node_id')}")
    return extracted


async def maybe_run_node_tools(session: Session, node: Optional[Dict]):
    """
    Deterministically run a node's tools once all of its variables_needed are
    satisfied (e.g. verify identity once DOB is known, create PTP once date+amount
    are known). Runs at most once per node visit — reset when the session moves
    to a new node (see Session.move_to_node).
    """
    if not node or session.node_tools_executed:
        return

    tool_ids = node.get("tools", [])
    if not tool_ids:
        return

    needed = node.get("variables_needed", [])
    if needed and not all(session.variables.get(v) for v in needed):
        return

    for tool_id in tool_ids:
        new_vars = await dispatch_tool(tool_id, session.variables, session.session_id)
        if is_async_stub(new_vars):
            logger.info(f"[Orchestrator] Node tool dispatched async: {tool_id} | job={new_vars.get('_async_job_id')}")
        elif new_vars:
            session.set_variables(new_vars)
            logger.info(f"[Orchestrator] Node tool auto-executed: {tool_id} | new vars: {list(new_vars.keys())}")

    session.node_tools_executed = True


# ── Image-Tool Detector ──────────────────────────────────────────────────────

def _has_input_image(body) -> bool:
    """Recursively scan a request-body dict/list for {InputImage}."""
    if isinstance(body, dict):
        return any(_has_input_image(v) for v in body.values())
    if isinstance(body, list):
        return any(_has_input_image(v) for v in body)
    if isinstance(body, str):
        return bool(re.search(r"\{\{?Input(?:Image|File)\}?\}", body))
    return False


async def _find_image_tool(tool_ids, user_message: str, session: Session, node: Optional[Dict] = None) -> Optional[str]:
    """
    Return the tool_id to use for image OCR.

    Discovers every tool in `tool_ids` whose request_body references
    {InputImage} — no hardcoded tool IDs. If exactly one such tool exists,
    it's used directly. If several exist (e.g. separate PAN vs Aadhaar OCR
    tools registered by the user), the LLM picks the right one using each
    tool's configured name/description and the recent conversation context —
    the same dynamic tool-selection mechanism used everywhere else in the
    orchestrator, so tool behavior is driven entirely by what's registered
    """
    import json
    candidates = []
    for tid in tool_ids:
        tool = tool_store.get(tid)
        if tool and _has_input_image(tool.get("request_body", {})):
            body_str = json.dumps(tool.get("response_body", {}))
            provided_vars = set(re.findall(r"\{(\w+)\}", body_str))
            desc = tool.get("description", "")
            if provided_vars:
                desc += f" (Provides variables: {', '.join(provided_vars)})"
                
            candidates.append({
                "tool_id": tid,
                "name": tool["name"],
                "description": desc,
                "provided_vars": provided_vars,
            })

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]["tool_id"]

    # --- DETERMINISTIC MATCHING ---
    if node:
        needed_vars = set(node.get("variables_needed", []))
        if needed_vars:
            matching_tools = [cand for cand in candidates if cand["provided_vars"] & needed_vars]
            if len(matching_tools) == 1:
                logger.info(f"[ImageTool] Deterministic match! Tool {matching_tools[0]['tool_id']} provides needed vars: {matching_tools[0]['provided_vars'] & needed_vars}")
                return matching_tools[0]["tool_id"]
            
            if matching_tools:
                logger.info(f"[ImageTool] Filtered candidates to {len(matching_tools)} based on needed vars")
                candidates = matching_tools

    if not llm_engine.is_loaded:
        logger.warning("[ImageTool] Multiple image tools available but LLM not loaded — using first candidate")
        return candidates[0]["tool_id"]

    recent_msgs = session.history[-4:] if session.history else []
    context = " | ".join(f"{m['role']}: {m['content']}" for m in recent_msgs)
    
    msg_for_llm = user_message
    file_name = session.variables.get("InputImageFileName")
    if file_name:
        msg_for_llm = f"{user_message} [Uploaded File: {file_name}]".strip()

    picked = await llm_queue.run(
        llm_engine.select_image_tool, msg_for_llm, candidates, context, node.get("name") if node else None
    )
    if picked:
        logger.info(f"[ImageTool] LLM selected image tool: {picked}")
        return picked

    logger.info(f"[ImageTool] LLM made no selection among {len(candidates)} image tools — using first candidate")
    return candidates[0]["tool_id"]


def _clear_image_from_session(session: "Session", new_vars: Dict) -> None:
    """
    Post-tool cleanup for image OCR runs.

    Called by any code path that just executed an image tool.  Inspects the
    `_clear_input_image` sentinel that `execute_tool()` sets when it consumed
    {InputImage}, then:
      - Pops `InputImage` and `_image_pending` from the session
      - Sets `_image_tool_fired = "true"` so the OCR fast-path won't re-trigger
        on subsequent turns (until a new upload resets the sentinel)
      - Removes `_clear_input_image` from `new_vars` so private keys don't leak
        into the LLM prompt or the caller's variable store
    """
    if new_vars.pop("_clear_input_image", None):
        session.variables.pop("InputImage", None)
        session.variables.pop("InputFile", None)
        session.variables.pop("_image_pending", None)
        session.set_variable("_image_tool_fired", "true")
        logger.info("[ImageTool] InputImage cleared from session | _image_tool_fired=true")


# ── LLM-Driven Auto Tool Calling (works in both modes) ────────────────────────

async def auto_tool_call(
    session: Session,
    agent: Dict,
    user_message: str,
    node: Optional[Dict] = None,
) -> Optional[str]:
    """
    LLM decides if any tool should be called based on user's message.
    Works in both single_prompt and flow mode.

    If `InputImage` is present in session variables AND `_image_tool_fired` is
    NOT set, whichever agent-level tool consumes {InputImage} is force-selected
    (OCR path) instead of letting the LLM pick from the full tool list.
    After the tool runs, `_clear_image_from_session()` pops `InputImage` and
    sets `_image_tool_fired` so subsequent turns don't re-trigger OCR.

    Returns: tool result summary string (to inject into LLM context), or None.
    """
    if not llm_engine.is_loaded:
        return None

    # Agent-level tools only — node-level tools are handled deterministically by
    # maybe_run_node_tools() once their variables_needed are satisfied, so they're
    # excluded here to avoid double-execution (e.g. sending SMS twice).
    tool_ids = set(agent.get("tools", []))

    if not tool_ids:
        return None

    # ── Image OCR fast-path: image already uploaded ──────────────────────────
    # Only fire when there is a pending image AND the image tool hasn't already
    # run for this upload (_image_tool_fired sentinel). This prevents re-triggering
    # on the user's next acknowledgement turn (e.g. "Yes, that's correct").
    if (session.variables.get("InputImage") or session.variables.get("InputFile")) and not session.variables.get("_image_tool_fired"):
        image_tool_id = await _find_image_tool(tool_ids, user_message, session, node=node)
        if image_tool_id:
            logger.info(f"[AutoTool] InputImage/InputFile detected → forcing image tool: {image_tool_id}")
            new_vars = await extract_and_execute_tool(image_tool_id, session, user_message)
            if new_vars:
                _clear_image_from_session(session, new_vars)  # pop InputImage/InputFile + set sentinel
                if is_async_stub(new_vars):
                    logger.info(f"[AutoTool] Image OCR tool dispatched async ({image_tool_id}) | job={new_vars.get('_async_job_id')}")
                else:
                    session.set_variables(new_vars)
                    logger.info(f"[AutoTool] Image OCR tool executed ({image_tool_id}) | new vars: {len(new_vars)}")
                    _record_tool_result(session, image_tool_id, new_vars)
                return format_tool_result(image_tool_id, new_vars)
            return None

    # Build tool descriptions for LLM
    available_tools = []
    for tid in tool_ids:
        tool = tool_store.get(tid)
        # unresolved {InputImage}/{InputFile} placeholder instead of real bytes.
        if tool and _has_input_image(tool.get("request_body", {})) and not (session.variables.get("InputImage") or session.variables.get("InputFile")):
            continue
        if tool:
            available_tools.append({
                "tool_id": tid,
                "name": tool["name"],
                "description": tool.get("description", ""),
            })

    if not available_tools:
        return None

    # LLM decides which tool to call
    recent_msgs = session.history[-4:] if session.history else []
    context = " | ".join(f"{m['role']}: {m['content']}" for m in recent_msgs)

    tool_id = await llm_queue.run(llm_engine.detect_tool_intent, user_message, available_tools, context)

    if not tool_id:
        return None

    # Extract missing variables from user message, then execute the tool
    new_vars = await extract_and_execute_tool(tool_id, session, user_message, context)
    if new_vars:
        if is_async_stub(new_vars):
            logger.info(f"[AutoTool] Dispatched async: {tool_id} | job={new_vars.get('_async_job_id')}")
        else:
            session.set_variables(new_vars)
            logger.info(f"[AutoTool] Executed: {tool_id} | New vars: {len(new_vars)}")
            _record_tool_result(session, tool_id, new_vars)
        return format_tool_result(tool_id, new_vars)

    _record_tool_result(session, tool_id, new_vars)  # new_vars is None/empty — count the failure
    return None


# ── Single Prompt Mode System Builder ─────────────────────────────────────────

def build_single_prompt_system(agent: Dict, session: Session) -> str:
    """Build the STATIC system prompt for single_prompt mode.

    Variable-dependent content is excluded for the same KV-cache reason as
    build_node_system_prompt().  Use build_session_context() for the variable
    block and pass it via generate_response(dynamic_context=...).
    """
    parts = []

    # Response constraint
    parts.append(
        "You are on a phone call. Keep responses to 1-3 sentences. "
        "Be direct, natural, and conversational."
    )

    parts.append(ASYNC_TOOL_INSTRUCTIONS)

    # Agent's full prompt (contains all instructions) — resolve {Variable} placeholders
    parts.append(f"\n{resolve_template(agent['system_prompt'], session.variables)}")

    # Language
    language = agent.get("language", "english")
    if language == "hinglish":
        parts.append("\nRespond in Hinglish (Hindi-English mix, Roman script).")
    elif language == "hindi":
        parts.append("\nRespond in Hindi.")

    # Personality
    personality = agent.get("personality")
    if personality:
        parts.append(f"\nTone: Be {personality}.")

    # ── Document OCR image context ────────────────────────────────────────────
    if session.variables.get("InputImage") or session.variables.get("InputFile"):
        parts.append(
            "\nNOTE: The customer has uploaded a document image. "
            "OCR will be triggered automatically on the correct tool — do NOT ask for the document number manually."
        )
    elif session.variables.get("Status") and session.variables.get("CustomerName"):
        parts.append(
            "\nNOTE: Document OCR has already been completed. "
            "Use the extracted customer data (CustomerName, DOB, Gender) from the session variables below."
        )

    # NOTE: session.variables are NOT included here — they go into dynamic_context.

    return "\n".join(parts)


def build_session_context(session: Session, tool_result: str = "") -> str:
    """Build the dynamic per-turn context block (session variables + tool result).

    This is passed to generate_response(dynamic_context=...) so it rides in the
    user-message suffix rather than the system prompt, keeping the cache key stable.
    """
    lines = []

    if session.variables:
        lines.append("--- Available Data ---")
        for k, v in session.variables.items():
            if not k.startswith("_") and k not in ("InputImage", "InputFile"):
                lines.append(f"  {k}: {v}")

    if tool_result:
        lines.append("")
        lines.append(tool_result)

    return "\n".join(lines)


def combine_tool_results(*blocks: Optional[str]) -> Optional[str]:
    """Join tool-result text blocks (background results + this turn's own tool
    call), dropping any that are empty. Order matters: background results are
    passed first so the model sees them before this turn's own tool activity."""
    parts = [b for b in blocks if b]
    return "\n\n".join(parts) if parts else None


async def collect_background_results(session: Session) -> Optional[str]:
    """
    Pattern B (poll-and-inject): before generating this turn's response, pull
    any async tool jobs that finished since the last turn and haven't been
    shown to the conversation yet. Real result variables are merged into
    session state immediately; a combined text block (batching every job
    completed since last turn) is returned for injection into dynamic_context
    so the model naturally weaves it into its reply.
    """
    jobs = poll_completed_jobs(session.session_id)
    if not jobs:
        return None

    blocks = []
    for job in jobs:
        if job.get("status") == "done":
            result = job.get("result") or {}
            if isinstance(result, str):
                try:
                    result = _json.loads(result)
                except (ValueError, TypeError):
                    result = {}
            clean_vars = {k: v for k, v in result.items() if not str(k).startswith("_")}
            if clean_vars:
                session.set_variables(clean_vars)
        blocks.append(format_background_result(job))

    logger.info(f"[Orchestrator] Delivered {len(jobs)} background job result(s) to session={session.session_id}")
    return "\n\n".join(blocks)


# ── Main Chat Processing ─────────────────────────────────────────────────────

async def process_message(session_id: str, user_message: str) -> Dict:
    """
    Process a user message through the orchestrator.

    Flow:
      1. Get session & agent
      2. Check global nodes (out-of-scope, escalation)
      3. Evaluate transitions from current node
      4. If transition → move to new node, execute tools if function node
      5. Build prompt → Generate LLM response
      6. Return response
    """
    start_time = time.perf_counter()

    # 1. Get session
    session = memory_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    # Block messages after session ended
    if session.status == "completed" or session.current_node == "__END__":
        return {
            "response": "This call has ended. Thank you.",
            "session_id": session_id,
            "agent_name": "",
            "current_node": "__END__",
            "current_node_name": "End",
            "transitioned": False,
            "duration_ms": 0,
        }

    agent = agent_store.get(session.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {session.agent_id}")

    # ── SINGLE PROMPT MODE — skip all node/transition logic ──────────────
    if agent.get("mode") == "single_prompt":
        session.add_message("user", user_message)

        # End detection: if conversation has 30+ messages, auto-end
        if len(session.history) >= 30:
            session.end()
            return {
                "response": "Thank you for your time. This call is now complete. Goodbye.",
                "session_id": session_id,
                "agent_name": agent["name"],
                "current_node": "__single_prompt__",
                "current_node_name": "Single Prompt",
                "transitioned": False,
                "duration_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }

        # End detection: if user says bye/goodbye explicitly
        bye_words = ["bye", "goodbye", "bye bye", "good bye", "tata", "alvida"]
        if user_message.lower().strip() in bye_words:
            session.end()
            return {
                "response": "Thank you for your time. Have a wonderful day. Goodbye.",
                "session_id": session_id,
                "agent_name": agent["name"],
                "current_node": "__single_prompt__",
                "current_node_name": "Single Prompt",
                "transitioned": False,
                "duration_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }

        # Pattern B: deliver any background async-tool results finished since last turn
        background_result = await collect_background_results(session)

        # Auto tool call — LLM decides if API needed
        t_tool_start = time.perf_counter()
        tool_result = await auto_tool_call(session, agent, user_message)
        t_tool_ms = (time.perf_counter() - t_tool_start) * 1000

        system_prompt = build_single_prompt_system(agent, session)

        history = session.get_clean_history()

        # Cap history to prevent token overflow (max ~14 messages)
        history = history[-14:]

        t_llm_start = time.perf_counter()
        try:
            # Run the blocking model.generate() call on a worker thread instead of
            # the event loop — this is what actually lets a dispatched async tool's
            # HTTP call make progress concurrently with generation (a bare sync
            # call here freezes the whole event loop until generation finishes).
            response = await llm_queue.run(
                llm_engine.generate_response,
                system_prompt, history, user_message,
                dynamic_context=build_session_context(session, combine_tool_results(background_result, tool_result) or ""),
            )
        except Exception as e:
            logger.error(f"[Orchestrator] Single prompt LLM failed: {e}")
            response = "I apologize, I'm having trouble processing that. Could you repeat?"
        t_llm_ms = (time.perf_counter() - t_llm_start) * 1000
        
        # Extract [STORE: ...] tags from LLM response → save to session, strip from output
        response = extract_and_store_variables(response, session)

        session.add_message("assistant", response)
        duration_ms = (time.perf_counter() - start_time) * 1000

        cache_stats = llm_engine.cache_stats
        hit_rate = f"{cache_stats['hit_rate']:.1%}" if cache_stats else "n/a"
        logger.info(
            f"[Orchestrator] Done | mode=single_prompt | session={session_id} | "
            f"tool={t_tool_ms:.0f}ms | llm={t_llm_ms:.0f}ms | "
            f"total={duration_ms:.0f}ms | kv_hit_rate={hit_rate}"
        )

        return {
            "response": response,
            "session_id": session_id,
            "agent_name": agent["name"],
            "current_node": "__single_prompt__",
            "current_node_name": "Single Prompt",
            "transitioned": False,
            "duration_ms": round(duration_ms, 2),
        }

    # ── FLOW MODE — full node/transition logic below ─────────────────────

    # Record user message
    session.add_message("user", user_message)
    session.increment_turns()

    # 2. Check global intercepts (out-of-scope, escalation) — uses get_node() cache internally
    intercept_response = check_global_intercept(agent, user_message, session)
    if intercept_response:
        session.add_message("assistant", intercept_response)
        duration_ms = (time.perf_counter() - start_time) * 1000
        current_node = get_node(session, session.current_node)
        return {
            "response": intercept_response,
            "session_id": session_id,
            "agent_name": agent["name"],
            "current_node": session.current_node,
            "current_node_name": current_node["name"] if current_node else "Unknown",
            "transitioned": False,
            "duration_ms": round(duration_ms, 2),
        }

    # 3. Extract structured variables this node needs (e.g. DOB, PTPDate) from the
    #    reply, then deterministically run the node's tools once they're all present.
    pre_transition_node = get_node(session, session.current_node)
    await extract_node_variables(session, pre_transition_node, user_message)
    await maybe_run_node_tools(session, pre_transition_node)

    # If this node's tool set an IsVerified flag and it came back False, don't let
    # the LLM transition forward on a guess — keep the customer in this node.
    verification_failed = session.variables.get("IsVerified") == "False"

    # Pattern B: deliver any background async-tool results finished since last turn
    background_result = await collect_background_results(session)

    # 4. Combined transition + agent-tool decision — usually 1 LLM call instead
    #    of 2 (evaluate_transitions + auto_tool_call used to run separately).
    transitioned = False
    target_node_id, tool_result = await decide_transition_and_tool(
        session, agent, pre_transition_node, user_message, skip_transition=verification_failed
    )

    if target_node_id:
        if target_node_id == "__END__":
            session.end()
            response = "Thank you for your time. Have a wonderful day."
            session.add_message("assistant", response)
            duration_ms = (time.perf_counter() - start_time) * 1000
            return {
                "response": response,
                "session_id": session_id,
                "agent_name": agent["name"],
                "current_node": "__END__",
                "current_node_name": "End",
                "transitioned": True,
                "duration_ms": round(duration_ms, 2),
            }

        # Declarative trigger: if this specific transition declares an
        # on_transition_tool, fire it now (Retell-style "→ trigger X → Node Y")
        trigger_result = await fire_transition_tool(pre_transition_node, target_node_id, session)
        if trigger_result:
            tool_result = f"{tool_result}\n\n{trigger_result}" if tool_result else trigger_result

        session.move_to_node(target_node_id)
        transitioned = True

        # Execute tools if new node is a function node
        new_node = get_node(session, target_node_id)
        if new_node and new_node.get("type") == "function":
            await execute_node_tools(new_node, session)
    else:
        # No edge matched — try Flex Mode: LLM jumps to best-fit node
        # (skipped while verification is blocking — don't let flex mode route
        # the customer away before identity is confirmed)
        flex_target = None if verification_failed else await flex_mode_jump(session, agent, user_message)
        if flex_target:
            session.move_to_node(flex_target)
            transitioned = True

            new_node = get_node(session, flex_target)
            if new_node and new_node.get("type") == "function":
                await execute_node_tools(new_node, session)

        # Check max_turns — force transition if exceeded (only if flex mode didn't already move)
        if not transitioned:
            current_node_data = get_node(session, session.current_node)
            if current_node_data:
                max_turns = current_node_data.get("max_turns", 3)
                if session.node_turns >= max_turns:
                    # Force transition to first available transition in node
                    transitions = current_node_data.get("transitions", [])
                    if transitions:
                        force_target = transitions[0]["to_node"]
                        trigger_result = await fire_transition_tool(current_node_data, force_target, session)
                        if trigger_result:
                            tool_result = f"{tool_result}\n\n{trigger_result}" if tool_result else trigger_result
                        session.move_to_node(force_target)
                        transitioned = True
                        logger.info(f"[Orchestrator] Max turns exceeded → force transition to {force_target}")

    # 4. Get current node
    current_node = get_node(session, session.current_node)
    if not current_node:
        raise HTTPException(status_code=500, detail=f"Node not found: {session.current_node}")

    # 4.1. If current node has max_turns=0 and we're here, force END immediately
    if current_node.get("max_turns", 3) == 0 and not transitioned:
        transitions = current_node.get("transitions", [])
        if transitions:
            force_target = transitions[0]["to_node"]
            if force_target == "__END__":
                session.end()
                return {
                    "response": "This call has ended. Thank you.",
                    "session_id": session_id,
                    "agent_name": agent["name"],
                    "current_node": "__END__",
                    "current_node_name": "End",
                    "transitioned": True,
                    "duration_ms": round((time.perf_counter() - start_time) * 1000, 2),
                }
            else:
                session.move_to_node(force_target)
                transitioned = True
                current_node = get_node(session, session.current_node)

    # 4.5. tool_result was already decided in step 4 (combined call above) —
    #      no separate auto_tool_call here anymore.

    # 5. Generate response
    # If node has static_message and this is the first turn after transition, use it.
    # Skip this shortcut when a background result just arrived — it needs to be
    # woven into an LLM-generated reply rather than dropped in favor of the canned line.
    if transitioned and current_node.get("static_message") and not background_result:
        response = resolve_template(current_node["static_message"], session.variables)
    else:
        # Use LLM to generate
        verification_note = (
            "Identity verification failed. Politely let the customer know and ask them "
            "to re-confirm their date of birth before we continue."
            if verification_failed else None
        )
        system_prompt = build_node_system_prompt(agent, current_node, session, note=verification_note)

        history = session.get_clean_history()

        try:
            # Run on a worker thread so the event loop stays free for dispatched
            # async tools' HTTP calls to actually progress during generation.
            response = await llm_queue.run(
                llm_engine.generate_response,
                system_prompt, history[-10:], user_message,
                dynamic_context=build_session_context(session, combine_tool_results(background_result, tool_result) or ""),
            )
        except Exception as e:
            logger.error(f"[Orchestrator] LLM failed: {e}")
            # Fallback to static message
            if current_node.get("static_message"):
                response = resolve_template(current_node["static_message"], session.variables)
            else:
                response = "I understand. Could you please provide more details?"
                
    # Extract [STORE: ...] tags from LLM response → save to session, strip from output
    response = extract_and_store_variables(response, session)
    session.add_message("assistant", response)

    duration_ms = (time.perf_counter() - start_time) * 1000
    cache_stats = llm_engine.cache_stats
    hit_rate = f"{cache_stats['hit_rate']:.1%}" if cache_stats else "n/a"
    logger.info(
        f"[Orchestrator] Done | mode=flow | session={session_id} | "
        f"node={current_node['name']} | transitioned={transitioned} | "
        f"total={duration_ms:.0f}ms | kv_hit_rate={hit_rate}"
    )

    return {
        "response": response,
        "session_id": session_id,
        "agent_name": agent["name"],
        "current_node": session.current_node,
        "current_node_name": current_node["name"],
        "transitioned": transitioned,
        "duration_ms": round(duration_ms, 2),
    }


# ── API Endpoints ────────────────────────────────────────────────────────────

@orchestrator_router.post("/start", response_model=StartChatResponse)
async def start_chat(req: StartChatRequest):
    """
    Start a new conversation. Creates session, loads data, returns greeting.
    Supports both 'flow' mode (multi-node) and 'single_prompt' mode.
    """
    agent = agent_store.get(req.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {req.agent_id}")

    agent_mode = agent.get("mode", "flow")

    # Determine start node and populate node_cache from consolidated JSON
    if agent_mode == "single_prompt":
        start_node_id = "__single_prompt__"
    else:
        start_node_id = agent.get("start_node", "")
        # Try cache from consolidated JSON first
        config_json = agent.get("agent_config_json")
        if config_json:
            # Build session node cache from consolidated JSON (no DB needed during conversation)
            node_cache = _build_node_cache_from_config(config_json)
            start_node = node_cache.get(start_node_id)
            if not start_node:
                # Fallback to DB if node not in cache (shouldn't happen in normal operation)
                start_node = node_store.get(start_node_id)
            logger.info(
                f"[Orchestrator] Node cache built from agent_config_json: "
                f"{len(node_cache)} nodes for agent={req.agent_id}"
            )
        else:
            # No consolidated JSON yet — fall back to direct DB lookup (legacy path)
            node_cache = {}
            start_node = node_store.get(start_node_id)
            logger.info(
                f"[Orchestrator] No agent_config_json found for {req.agent_id}, "
                "using direct DB lookups (run add_config_json_column.py to migrate)"
            )
        if not start_node:
            raise HTTPException(status_code=500, detail=f"Start node not found: {start_node_id}")

    # Create session
    session = memory_store.create_session(agent_id=req.agent_id, start_node=start_node_id)
    # Attach node_cache to session (empty for single_prompt; populated for flow mode)
    if agent_mode != "single_prompt":
        session.node_cache = node_cache

    # Set default variables
    session.set_variable("LenderName", cfg.LENDER_NAME)

    # Load customer data if mobile provided
    if req.mobile:
        session.set_variable("mobile_number", req.mobile)
        session.set_variable("MobileNumber", req.mobile)

        from cbs_api import cbs_data
        summary = cbs_data.get_summary(req.mobile)
        if summary:
            customer = summary.get("customer", {})
            session.set_variable("CustomerName", customer.get("name", "Customer"))
            session.set_variable("CustomerId", customer.get("customer_id", ""))
            loans = summary.get("loans", [])
            if loans:
                loan = loans[0]
                session.set_variable("EMIAmount", loan.get("emi", "0"))
                session.set_variable("DueDate", loan.get("due_date", "N/A"))
                session.set_variable("DPD", loan.get("dpd", "0"))
                session.set_variable("OutstandingAmount", loan.get("outstanding", "0"))
                session.set_variable("OutstandingPrincipal", loan.get("principal", "0"))
                session.set_variable("RemainingEMIs", loan.get("remaining_emis", "0"))
            session.set_variable("TotalOverdue", str(summary.get("total_overdue", 0)))

    # Generate greeting based on mode
    if agent_mode == "single_prompt":
        # Single prompt mode — LLM generates first message
        system_prompt = build_single_prompt_system(agent, session)
        try:
            greeting = await llm_queue.run(
                llm_engine.generate_response,
                system_prompt,
                [],
                "Start the conversation. Greet the customer.",
            )
        except Exception:
            customer_name = session.get_variable("CustomerName", "Customer")
            greeting = f"Hello {customer_name}, how can I help you today?"
        node_name = "Single Prompt"
    else:
        # Flow mode — use start node's static message
        start_node = get_node(session, start_node_id)
        if start_node.get("static_message"):
            greeting = resolve_template(start_node["static_message"], session.variables)
        else:
            greeting = f"Hello {session.get_variable('CustomerName', 'Customer')}, I am calling from {cfg.LENDER_NAME}. Am I speaking with the right person?"
        node_name = start_node["name"]

    session.add_message("assistant", greeting)

    return StartChatResponse(
        session_id=session.session_id,
        agent_id=req.agent_id,
        agent_name=agent["name"],
        greeting=greeting,
        current_node=start_node_id,
        current_node_name=node_name,
    )


@orchestrator_router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Process a user message and return agent response."""
    if req.image_base64:
        session = memory_store.get_session(req.session_id)
        if session:
            session.set_variable("InputImage", req.image_base64)
            session.set_variable("InputFile", req.image_base64)
            session.set_variable("_image_pending", "true")
            session.variables.pop("_image_tool_fired", None)
            logger.info(f"[Chat] Stored InputImage from payload for session={req.session_id} | pending=true, fired sentinel cleared")

    result = await process_message(req.session_id, req.message)
    return ChatResponse(**result)


# ── SSE Streaming Chat Endpoint ──────────────────────────────────────────────

from fastapi.responses import StreamingResponse
import json as _json_mod
import asyncio


@orchestrator_router.post("/stream")
async def chat_stream(req: ChatRequest):
    """
    Process a user message and stream the LLM response token-by-token via SSE.
    
    Pre-processing (transitions, tool calls, etc.) runs first. Then the final
    LLM response generation is streamed as Server-Sent Events.
    
    SSE format:
      data: {"type": "token", "content": "Hello"}
      data: {"type": "token", "content": " how"}
      data: {"type": "done", "response": "full text", "current_node_name": "...", "transitioned": false, "duration_ms": 1234}
    """
    if req.image_base64:
        session = memory_store.get_session(req.session_id)
        if session:
            session.set_variable("InputImage", req.image_base64)
            session.set_variable("InputFile", req.image_base64)
            session.set_variable("_image_pending", "true")
            session.variables.pop("_image_tool_fired", None)

    async def event_generator():
        try:
            async for event in _process_message_stream(req.session_id, req.message):
                yield f"data: {_json_mod.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"[Stream] Error: {e}")
            error_event = {"type": "error", "content": str(e)}
            yield f"data: {_json_mod.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _process_message_stream(session_id: str, user_message: str):
    """
    Streaming variant of process_message(). Yields SSE event dicts.
    All pre-processing (transitions, tools) runs synchronously first,
    then the LLM generation phase streams token-by-token.
    """
    start_time = time.perf_counter()

    # 1. Get session
    session = memory_store.get_session(session_id)
    if not session:
        yield {"type": "error", "content": f"Session not found: {session_id}"}
        return

    # Block messages after session ended
    if session.status == "completed" or session.current_node == "__END__":
        yield {"type": "done", "response": "This call has ended. Thank you.",
               "current_node_name": "End", "transitioned": False, "duration_ms": 0}
        return

    agent = agent_store.get(session.agent_id)
    if not agent:
        yield {"type": "error", "content": f"Agent not found: {session.agent_id}"}
        return

    # ── SINGLE PROMPT MODE ───────────────────────────────────────────────
    if agent.get("mode") == "single_prompt":
        session.add_message("user", user_message)

        # End detection
        if len(session.history) >= 30:
            session.end()
            yield {"type": "done", "response": "Thank you for your time. This call is now complete. Goodbye.",
                   "current_node_name": "Single Prompt", "transitioned": False,
                   "duration_ms": round((time.perf_counter() - start_time) * 1000, 2)}
            return

        bye_words = ["bye", "goodbye", "bye bye", "good bye", "tata", "alvida"]
        if user_message.lower().strip() in bye_words:
            session.end()
            yield {"type": "done", "response": "Thank you for your time. Have a wonderful day. Goodbye.",
                   "current_node_name": "Single Prompt", "transitioned": False,
                   "duration_ms": round((time.perf_counter() - start_time) * 1000, 2)}
            return

        # Pattern B: deliver any background async-tool results finished since last turn
        background_result = await collect_background_results(session)

        # Tool call (non-streaming)
        tool_result = await auto_tool_call(session, agent, user_message)

        system_prompt = build_single_prompt_system(agent, session)
        history = session.get_clean_history()[-14:]
        dynamic_ctx = build_session_context(session, combine_tool_results(background_result, tool_result) or "")

        # Stream LLM response
        full_response = []
        try:
            async with llm_queue.acquire():
                async for token in llm_engine.agenerate_response_stream(
                    system_prompt, history, user_message, dynamic_context=dynamic_ctx
                ):
                    full_response.append(token)
                    yield {"type": "token", "content": token}
        except Exception as e:
            logger.error(f"[Stream] LLM failed: {e}")
            fallback = "I apologize, I'm having trouble processing that. Could you repeat?"
            full_response = [fallback]
            yield {"type": "token", "content": fallback}

        response_text = "".join(full_response)
        session.add_message("assistant", response_text)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        yield {"type": "done", "response": response_text,
               "current_node_name": "Single Prompt", "transitioned": False,
               "duration_ms": duration_ms}
        return

    # ── FLOW MODE ────────────────────────────────────────────────────────
    session.add_message("user", user_message)
    session.increment_turns()

    # Global intercept check
    intercept_response = check_global_intercept(agent, user_message, session)
    if intercept_response:
        session.add_message("assistant", intercept_response)
        current_node = get_node(session, session.current_node)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        yield {"type": "token", "content": intercept_response}
        yield {"type": "done", "response": intercept_response,
               "current_node_name": current_node["name"] if current_node else "Unknown",
               "transitioned": False, "duration_ms": duration_ms}
        return

    # Extract variables + run node tools
    pre_transition_node = get_node(session, session.current_node)
    await extract_node_variables(session, pre_transition_node, user_message)
    await maybe_run_node_tools(session, pre_transition_node)

    verification_failed = session.variables.get("IsVerified") == "False"

    # Pattern B: deliver any background async-tool results finished since last turn
    background_result = await collect_background_results(session)

    # Combined transition + tool decision
    transitioned = False
    target_node_id, tool_result = await decide_transition_and_tool(
        session, agent, pre_transition_node, user_message, skip_transition=verification_failed
    )

    if target_node_id:
        if target_node_id == "__END__":
            session.end()
            end_msg = "Thank you for your time. Have a wonderful day."
            session.add_message("assistant", end_msg)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            yield {"type": "token", "content": end_msg}
            yield {"type": "done", "response": end_msg,
                   "current_node_name": "End", "transitioned": True, "duration_ms": duration_ms}
            return

        # Fire transition tool if declared
        trigger_result = await fire_transition_tool(pre_transition_node, target_node_id, session)
        if trigger_result:
            tool_result = f"{tool_result}\n\n{trigger_result}" if tool_result else trigger_result

        session.move_to_node(target_node_id)
        transitioned = True

        new_node = get_node(session, target_node_id)
        if new_node and new_node.get("type") == "function":
            await execute_node_tools(new_node, session)
    else:
        # Flex mode
        flex_target = None if verification_failed else await flex_mode_jump(session, agent, user_message)
        if flex_target:
            session.move_to_node(flex_target)
            transitioned = True
            new_node = get_node(session, flex_target)
            if new_node and new_node.get("type") == "function":
                await execute_node_tools(new_node, session)

        # Max turns check
        if not transitioned:
            current_node_data = get_node(session, session.current_node)
            if current_node_data:
                max_turns = current_node_data.get("max_turns", 3)
                if session.node_turns >= max_turns:
                    transitions = current_node_data.get("transitions", [])
                    if transitions:
                        force_target = transitions[0]["to_node"]
                        trigger_result = await fire_transition_tool(current_node_data, force_target, session)
                        if trigger_result:
                            tool_result = f"{tool_result}\n\n{trigger_result}" if tool_result else trigger_result
                        session.move_to_node(force_target)
                        transitioned = True

    # Get current node
    current_node = get_node(session, session.current_node)
    if not current_node:
        yield {"type": "error", "content": f"Node not found: {session.current_node}"}
        return

    # max_turns=0 force END
    if current_node.get("max_turns", 3) == 0 and not transitioned:
        transitions = current_node.get("transitions", [])
        if transitions:
            force_target = transitions[0]["to_node"]
            if force_target == "__END__":
                session.end()
                end_msg = "This call has ended. Thank you."
                yield {"type": "token", "content": end_msg}
                yield {"type": "done", "response": end_msg,
                       "current_node_name": "End", "transitioned": True,
                       "duration_ms": round((time.perf_counter() - start_time) * 1000, 2)}
                return
            else:
                session.move_to_node(force_target)
                transitioned = True
                current_node = get_node(session, session.current_node)

    # 5. Generate response — STREAMING
    if transitioned and current_node.get("static_message"):
        # Static message — no streaming needed
        response_text = resolve_template(current_node["static_message"], session.variables)
        session.add_message("assistant", response_text)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        yield {"type": "token", "content": response_text}
        yield {"type": "done", "response": response_text,
               "current_node_name": current_node["name"], "transitioned": transitioned,
               "duration_ms": duration_ms}
    else:
        # LLM streaming
        verification_note = (
            "Identity verification failed. Politely let the customer know and ask them "
            "to re-confirm their date of birth before we continue."
            if verification_failed else None
        )
        system_prompt = build_node_system_prompt(agent, current_node, session, note=verification_note)
        history = session.get_clean_history()
        dynamic_ctx = build_session_context(session, combine_tool_results(background_result, tool_result) or "")

        full_response = []
        try:
            async with llm_queue.acquire():
                async for token in llm_engine.agenerate_response_stream(
                    system_prompt, history[-10:], user_message, dynamic_context=dynamic_ctx
                ):
                    full_response.append(token)
                    yield {"type": "token", "content": token}
        except Exception as e:
            logger.error(f"[Stream] LLM failed: {e}")
            if current_node.get("static_message"):
                fallback = resolve_template(current_node["static_message"], session.variables)
            else:
                fallback = "I understand. Could you please provide more details?"
            full_response = [fallback]
            yield {"type": "token", "content": fallback}

        response_text = "".join(full_response)
        session.add_message("assistant", response_text)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        yield {"type": "done", "response": response_text,
               "current_node_name": current_node["name"], "transitioned": transitioned,
               "duration_ms": duration_ms}


@orchestrator_router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session state."""
    session = memory_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return session.to_dict()


@orchestrator_router.get("/model/status")
def model_status():
    """Check LLM model status."""
    return {
        "loaded": llm_engine.is_loaded,
        "device": llm_engine.device,
        "model_path": cfg.MODEL_PATH,
    }


@orchestrator_router.post("/model/load")
def load_model():
    """Manually load the LLM model."""
    if llm_engine.is_loaded:
        return {"detail": "Model already loaded", "device": llm_engine.device}
    try:
        llm_engine.load()
        return {"detail": "Model loaded", "device": llm_engine.device}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


# ── Subflow API Endpoints ────────────────────────────────────────────────────

subflow_router = APIRouter(prefix="/api/subflows", tags=["Subflows"])


class SubflowCreate(BaseModel):
    name: str = Field(..., description="Subflow name")
    description: Optional[str] = None
    nodes: List[str] = Field(..., description="Node IDs in this subflow")
    edges: List[str] = Field(default_factory=list, description="Edge IDs in this subflow")
    entry_node: str = Field(..., description="Entry node of the subflow")
    exit_node: str = Field(..., description="Exit node of the subflow")


@subflow_router.post("")
def create_subflow(req: SubflowCreate):
    """Create a reusable subflow (package of nodes + edges)."""
    import uuid
    subflows = load_subflows()
    subflow = {
        "subflow_id": f"subflow_{uuid.uuid4().hex[:6]}",
        "name": req.name,
        "description": req.description,
        "nodes": req.nodes,
        "edges": req.edges,
        "entry_node": req.entry_node,
        "exit_node": req.exit_node,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    subflows.append(subflow)
    save_subflows(subflows)
    return subflow


@subflow_router.get("")
def list_subflows():
    """List all subflows."""
    subflows = load_subflows()
    return {"total": len(subflows), "subflows": subflows}


@subflow_router.get("/{subflow_id}")
def get_subflow_endpoint(subflow_id: str):
    """Get a specific subflow."""
    sf = get_subflow(subflow_id)
    if not sf:
        raise HTTPException(status_code=404, detail=f"Subflow not found: {subflow_id}")
    return sf


@subflow_router.delete("/{subflow_id}")
def delete_subflow(subflow_id: str):
    """Delete a subflow."""
    subflows = load_subflows()
    new_list = [sf for sf in subflows if sf["subflow_id"] != subflow_id]
    if len(new_list) == len(subflows):
        raise HTTPException(status_code=404, detail=f"Subflow not found: {subflow_id}")
    save_subflows(new_list)
    return {"detail": "Subflow deleted", "subflow_id": subflow_id}
