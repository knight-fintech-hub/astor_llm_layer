# Knight Flow Generic — Dynamic Multi-Agent Platform

The conversational agent platform where everything is configurable from UI — nodes, edges, tools, and agents. No code changes needed to build new conversation flows.

---

## How to Run

```bash
# Terminal 1 — Backend API (port 8100)
cd knight-flow-generic
python main.py

# Terminal 2 — Gradio UI (port 7870)
cd knight-flow-generic
python gradio_app.py
```

- API Docs: http://localhost:8100/docs
- Gradio UI: http://localhost:7870

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GRADIO UI (7870)                       │
├────────────┬────────────┬──────────────┬────────────────┤
│  Node      │  Tool      │   Agent      │   Chat         │
│  Builder   │  Registry  │   Builder    │   Interface    │
└────────────┴────────────┴──────────────┴────────────────┘
                         │
                   ┌─────▼─────┐
                   │ FastAPI    │ (8100)
                   │ main.py    │
                   └─────┬─────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
   ┌──────▼───┐  ┌──────▼───┐  ┌──────▼───┐
   │Orchestrator│  │CBS Data  │  │  LLM     │
   │(brain)    │  │API       │  │  Engine  │
   └───────────┘  └──────────┘  └──────────┘
```

---

## Folder Structure

```
knight-flow-generic/
├── .env                    — Environment configuration
├── config.py               — Config loader (reads .env)
├── logger.py               — Loguru logging setup
├── requirements.txt        — Python dependencies
├── main.py                 — FastAPI entry point, registers all routers
├── orchestrator.py         — Runtime brain (transitions, tools, responses)
├── llm_engine.py           — Local Qwen LLM interface
├── memory_store.py         — Session memory management
├── node_manager.py         — Node CRUD API
├── edge_manager.py         — Edge CRUD API
├── agent_manager.py        — Agent CRUD API
├── tool_registry.py        — Tool registration & execution
├── cbs_api.py              — CBS data API + backend services
├── gradio_app.py           — Gradio UI (5 tabs)
├── customer_master_data/   — Mock banking data (CSVs)
│   ├── customer_master_data.csv
│   ├── account_data.csv
│   ├── loan_data (1).csv
│   └── transaction_history.csv
└── data/                   — Runtime JSON data (user-configurable)
    ├── nodes.json
    ├── edges.json
    ├── tools.json
    ├── agents.json
    ├── subflows.json
    └── sessions.json
```

---

## File-by-File Explanation

### Core Files

| File | Purpose | What Happens Inside |
|------|---------|---------------------|
| **main.py** | FastAPI entry point | Creates app, adds CORS, registers all routers (nodes, edges, tools, agents, chat, cbs, services, subflows). Health check at `/health`. |
| **config.py** | Configuration | Loads `.env` file. Defines HOST, PORT, MODEL_PATH, MAX_NEW_TOKENS, TEMPERATURE, LENDER_NAME, GRADIO_PORT, DATA_DIR. |
| **logger.py** | Logging | Loguru setup with colored console output. All files import `get_logger()` from here. |
| **.env** | Environment variables | Server port (8100), model path, LLM params, lender name, Gradio port (7870). |

### Orchestrator (Brain)

| File | Purpose | What Happens Inside |
|------|---------|---------------------|
| **orchestrator.py** | Runtime engine | The main brain. Handles: (1) Start chat — create session, load CBS data, send greeting. (2) Process message — evaluate edge keywords, transition nodes, generate LLM response. (3) Out-of-scope detection — blocks off-topic questions. (4) Max turns enforcement — force-transitions when node exceeds turn limit. (5) __END__ blocking — stops accepting messages after call ends. (6) Subflow CRUD endpoints. |
| **llm_engine.py** | LLM interface | Loads Qwen model with 4-bit quantization on GPU. Three functions: `generate_response()` — produces agent replies. `flex_mode_evaluate()` — LLM picks best node (disabled currently). `detect_tool_intent()` — LLM decides which tool to call (for subagent nodes). |
| **memory_store.py** | Session state | Session class tracks: current_node, previous_node, variables dict, conversation history, node_turns counter, swap_history, status. MemoryStore manages all active sessions. Supports agent swapping with memory transfer. |

### CRUD Managers

| File | Purpose | What Happens Inside |
|------|---------|---------------------|
| **node_manager.py** | Node CRUD | Manages conversation nodes. Each node has: name, type (conversation/function/subagent), prompt (LLM instruction), static_message (fixed response), tools (assigned APIs), is_global flag, max_turns. Reads/writes `data/nodes.json`. API: POST/GET/PUT/DELETE `/api/nodes`. |
| **edge_manager.py** | Edge CRUD | Manages transitions between nodes. Each edge has: from_node, to_node, condition (natural language), keywords (fast-match list), priority. Reads/writes `data/edges.json`. API: POST/GET/PUT/DELETE `/api/edges`. |
| **tool_registry.py** | Tool CRUD + Execution | Registers external APIs. Each tool has: name, method, endpoint (with {Variable} placeholders), params, response_mapping (maps API response fields to session variables). `execute_tool()` resolves variables, makes HTTP call, maps response back. API: POST/GET/PUT/DELETE `/api/tools`. |
| **agent_manager.py** | Agent CRUD | Manages agents. Each agent has: name, system_prompt, start_node, assigned nodes list, global_nodes list, language,    personality, swap_rules. API: POST/GET/PUT/DELETE `/api/agents`. |

### Data & Services

| File | Purpose | What Happens Inside |
|------|---------|---------------------|
| **cbs_api.py** | CBS Data + Backend Services | Loads CSVs (customers, accounts, loans, transactions) into memory. Endpoints: `/api/cbs/customers`, `/api/cbs/summary/{mobile}`, `/api/cbs/transactions/{account_no}`. Also has simulated services: POST `/api/services/verify` (DOB check), POST `/api/services/ptp/create` (Promise to Pay), POST `/api/services/sms/send` (SMS notification). |
| **gradio_app.py** | UI | 5-tab Gradio interface. Tab 1: Node Builder (create/delete nodes). Tab 2: Edge Builder (hidden). Tab 3: Tool Registry (register APIs). Tab 4: Agent Builder (build agents from nodes). Tab 5: Chat (select agent + customer, start call, send messages). |

---

## Data Files (data/)

| File | What It Stores | User Editable From |
|------|----------------|-------------------|
| **nodes.json** | All conversation nodes (9 default: greeting, verification, loan_reminder, payment_discussion, payment_commitment, confirmation, closing, out_of_scope, escalation) | Node Builder tab |
| **edges.json** | All transitions (9 default: greeting→verification→loan_reminder→payment_discussion→payment_commitment→confirmation→closing→END) | Edge Builder tab |
| **tools.json** | Registered APIs (5 default: cbs_lookup, verify_customer, create_ptp, send_sms, get_transactions) | Tool Registry tab |
| **agents.json** | Agent configurations (1 default: Collection Agent) | Agent Builder tab |
| **subflows.json** | Reusable node groups (1 default: Identity Verification) | API only |
| **sessions.json** | Placeholder (sessions are in-memory) | — |

---

## How a Message Flows (Runtime)

```
1. User clicks "Start Call" with agent + customer selected
   → POST /api1623/start
   → Create session, load CBS data (CustomerName, EMI, etc.)
   → Move to start_node (greeting_01)
   → Return greeting: "Hello {CustomerName}, I am calling from Knight Gini..."

2. User types "yes"
   → POST /api/chat {session_id, message}
   → orchestrator.process_message():
     a. Check if session ended → No, continue
     b. Check out-of-scope → No
     c. Evaluate edges from greeting_01:
        - edge_001 keywords: ["yes", "speaking"...] → "yes" MATCHES!
        - Transition to verification_01
     d. verification_01 has static_message → return it
   → Response: "Please confirm your Date of Birth?"

3. User types "07071996"
   → Evaluate edges from verification_01:
     - edge_002: no keywords, no match
     - max_turns check: turns(1) < max(3) → stay
   → LLM generates response using verification node prompt
   → Response: "Thank you, verified. Your EMI of Rs.30843..."
   (Note: may transition via max_turns if no edge matches after 3 attempts)

4. User types "i dont have money" (in payment_discussion)
   → Evaluate edges:
     - edge_005 keywords: "next month", "will pay"... → NO match
     - edge_006 keywords: "stop calling", "never paying"... → NO match
   → No transition, stays in payment_discussion
   → LLM generates: "I understand. When do you expect to make the payment?"

5. User types "next month"
   → edge_005 keyword "next month" → MATCH
   → Transition to payment_commitment_01
   → Static message: "Would you be able to make payment by that date?"
```

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| **Keyword-only transitions (LLM evaluation disabled)** | Local Qwen 7B model gives unreliable yes/no answers for condition matching. Keywords are fast and deterministic. |
| **Flex Mode disabled** | Same reason — LLM picks wrong nodes. Can re-enable with better model (GPT-4o). |
| **Static message on transition** | Instant response when entering new node (no LLM call = zero latency). LLM used only when staying in same node. |
| **Max turns enforcement** | Prevents infinite loops. After N turns without edge match, force-transitions to first available edge. |
| **Session end blocking** | Once __END__ reached, all further messages return fixed response. |
| **max_new_tokens=150** | Keeps responses short (phone call style). |

---

## Node Types

| Type | Behavior |
|------|----------|
| **conversation** | Multi-turn dialogue. LLM generates responses. Most common. |
| **function** | Executes tools ON ENTRY. No dialogue. Auto-transitions after. |
| **subagent** | Dialogue + LLM-driven tool calling. LLM decides which tool to use based on conversation. |

---

## Features Implemented

- [x] Dynamic node creation (UI)
- [x] Dynamic edge creation with keywords + conditions
- [x] Tool/API registration and execution
- [x] Agent builder (assign nodes, global nodes)
- [x] Keyword-based transition matching
- [x] LLM transition evaluation (disabled, ready for cloud API)
- [x] Flex Mode — LLM jumps nodes (disabled, ready)
- [x] Subagent Node — LLM decides tools
- [x] Subflows — reusable node groups
- [x] Global nodes (out-of-scope, escalation)
- [x] Session memory with variable tracking
- [x] Agent swap rules (memory transfer ready)
- [x] Out-of-scope detection
- [x] Max turns enforcement
- [x] __END__ state enforcement
- [x] Word boundary matching for short keywords
- [x] CBS mock data integration
- [x] Simulated backend services (PTP, Verify, SMS)

---

## API Endpoints Summary

| Group | Endpoints |
|-------|-----------|
| **Nodes** | POST/GET/PUT/DELETE `/api/nodes`, GET `/api/nodes/{id}` |
| **Edges** | POST/GET/PUT/DELETE `/api/edges`, GET `/api/edges/from/{node_id}` |
| **Tools** | POST/GET/PUT/DELETE `/api/tools`, POST `/api/tools/{id}/execute` |
| **Agents** | POST/GET/PUT/DELETE `/api/agents` |
| **Chat** | POST `/api/chat/start`, POST `/api/chat`, GET `/api/chat/session/{id}` |
| **Model** | GET `/api/chat/model/status`, POST `/api/chat/model/load` |
| **Subflows** | POST/GET/DELETE `/api/subflows` |
| **CBS** | GET `/api/cbs/customers`, GET `/api/cbs/summary/{mobile}` |
| **Services** | POST `/api/services/verify`, POST `/api/services/ptp/create`, POST `/api/services/sms/send` |

---

## Dependencies

```
fastapi, uvicorn, pydantic, loguru, python-dotenv, httpx, gradio
torch, transformers, bitsandbytes, accelerate
```

---

## Configuration (.env)

```env
HOST=0.0.0.0
PORT=8100
MODEL_PATH=<path to Qwen model>
MAX_NEW_TOKENS=150
TEMPERATURE=0.7
LENDER_NAME=Knight Gini
GRADIO_PORT=7870
LOG_LEVEL=INFO
```
