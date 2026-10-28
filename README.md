# Knight Flow — Conversational Agent Platform

Knight Flow is an AI platform that lets you build, customize, and run conversational agents without writing code. You can design conversation flows, set transition rules, connect external APIs, and test your agents directly from a clean web interface.

---

## Key Features

- **Visual Flow Builder**: Create conversation steps (**nodes**) and movement rules (**edges**) straight from the UI.
- **Tool Integration**: Connect external APIs (banking services, SMS, databases) that agents can call automatically during conversation.
- **Smart Conversation Control**: Keeps dialogues natural and on-track using keyword matching, maximum turn limits, and out-of-scope protection.
- **Knowledge Base (RAG)**: Upload documents (PDFs, text files) so agents can answer questions from your reference materials.
- **Session Memory**: Tracks customer details, variables, and conversation history across the entire session.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy or update `.env` with your settings (database, model path, ports):
```env
HOST=192.168.0.155      # or localhost
PORT=1225               # Backend API port
GRADIO_PORT=7870        # Web UI port
DATABASE_URL=postgresql://<user>:<password>@localhost:5432/<dbname>
MODEL_PATH=<path_to_local_llm>
```

### 3. Run the Platform

**Terminal 1 — Backend API**
```bash
python main.py
```
- API Docs (Swagger): `http://localhost:1225/docs` (or your configured `PORT`)

**Terminal 2 — Gradio Web UI**
```bash
python gradio_app.py
```
- Web UI: `http://localhost:7870`

*(Optional)* **Terminal 3 — Knowledge Base (RAG Service)**
```bash
python rag_app.py
```

---

## How It Works

A conversation in Knight Flow runs through a straightforward lifecycle:

1. **Start Call**: A user selects an agent and customer. The engine loads user data and sends the initial greeting.
2. **Listen & Transition**: When the user replies, the engine checks the transition rules (keywords/intents) to decide the next step.
3. **Generate Response**:
   - If the new step has a fixed message, it responds immediately without delay.
   - Otherwise, the LLM generates a contextual response based on the step instructions.
4. **Run Tools**: If a step requires external data (e.g., checking an account balance or saving a promise-to-pay), the agent runs the API and saves the result to session memory.
5. **End Call**: Once the goal is met or the closing step triggers, the conversation safely concludes.

---

## Core Concepts

| Concept | What It Is |
|---|---|
| **Agent** | The AI assistant (e.g. Loan Officer, Support Bot) with a specific persona and assigned workflow. |
| **Node** | A single step or phase in the conversation (e.g. Greeting, Identity Check, Payment Discussion, Closing). |
| **Edge** | The rule connecting one node to another based on user responses. |
| **Tool** | An external API that agents can trigger to fetch or submit data. |
| **RAG** | Knowledge search that lets agents look up answers in uploaded reference documents. |

---

## Web UI Tabs

- **Tool Registry**: Register and test external APIs with custom parameters and responses.
- **Agent Builder**: Create agents, define conversation steps (nodes), and configure transition rules.
- **Chat**: Test agents in real time, view customer profile data, and observe live step transitions.

---

## Project Structure

```
knight/
├── main.py              # FastAPI server entry point & route definitions
├── orchestrator.py      # Runtime brain (conversation flow, transitions, session logic)
├── gradio_app.py        # Web interface (Agent Builder, Tool Registry, Chat)
├── llm_engine.py        # Local LLM loader and prompt generation
├── memory_store.py      # Session memory and state management
├── tool_registry.py     # API tool registration and execution
├── cbs_api.py           # Core banking mock data and service endpoints
├── rag_app.py           # Knowledge base document upload and retrieval service
├── database.py          # PostgreSQL database connection and setup
├── config.py            # Central configuration loader (.env)
└── requirements.txt     # Python package requirements
```

---

## Main API Endpoints

| Category | Method & Endpoint | Description |
|---|---|---|
| **Chat** | `POST /api/chat/start` | Start a new session with an agent |
| | `POST /api/chat` | Send a user message and receive the agent's response |
| | `GET /api/chat/session/{id}` | Retrieve session state and chat history |
| **Agents** | `GET /api/agents` | List all available agents |
| | `POST /api/agents` | Create or update an agent configuration |
| **Nodes** | `GET /api/nodes` | List all conversation nodes |
| **Tools** | `GET /api/tools` | List registered API tools |
| | `POST /api/tools/{id}/execute` | Test or run a specific API tool |
| **Knowledge Base** | `POST /api/knowledge-base/upload` | Upload documents to the knowledge base |
| **System** | `GET /health` | Service health and memory status |
