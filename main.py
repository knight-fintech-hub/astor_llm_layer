"""
main.py — Knight Flow Generic: Dynamic Multi-Agent Platform
==============================================================
  - Register tools (APIs) from UI
  - Build agents from UI — configure nodes (conversation phases) and their
    transitions inline as part of the agent's workflow
  - Chat with dynamic workflow execution

Run:
  python main.py         → FastAPI on port 8100
  python gradio_app.py   → Gradio UI on port 7870
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import cfg
from logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Knight Flow Generic — Dynamic Multi-Agent Platform",
    version="2.0.0",
    docs_url="/",
    redoc_url=None
    ,
    description="Build conversational agents with dynamic nodes, edges, and tools — no code changes needed.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routers ─────────────────────────────────────────────────────────
from tool_registry import tool_router, image_router
from orchestrator import orchestrator_router, subflow_router
from cbs_api import cbs_router, services_router

if cfg.USE_DB:
    from database import init_db
    from agent_manager import agent_router, node_router
    # Verify DB connection
    init_db()
    app.include_router(agent_router)
    app.include_router(node_router)
else:
    logger.info("[main] USE_DB=false — local JSON mode. Skipping PostgreSQL init and write routes.")
    # ── Read-only agent endpoints for local mode ──────────────────────────────
    from fastapi import APIRouter
    from local_agent_loader import local_agent_store
    _local_agent_router = APIRouter(prefix="/api/agents", tags=["Agent (local)"])

    @_local_agent_router.get("")
    def _list_local_agents():
        agents = local_agent_store.list_all()
        return {"total": len(agents), "agents": agents}

    @_local_agent_router.get("/{agent_id}")
    def _get_local_agent(agent_id: str):
        from fastapi import HTTPException
        agent = local_agent_store.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        return agent

    app.include_router(_local_agent_router)

app.include_router(tool_router)
app.include_router(image_router)
app.include_router(orchestrator_router)
app.include_router(subflow_router)
app.include_router(cbs_router)
app.include_router(services_router)



@app.get("/health")
def health():
    from memory_store import memory_store
    from llm_engine import llm_queue
    return {
        "status": "ok", 
        "service": "knight-flow-generic", 
        "version": "2.0.0",
        "cache": memory_store.cache_summary,
        "llm_queue": llm_queue.stats
    }


@app.get("/")
def root():
    return {
        "service": "Knight Flow Generic — Dynamic Multi-Agent Platform",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "nodes": "/api/nodes",
            "edges": "/api/edges",
            "tools": "/api/tools",
            "agents": "/api/agents",
            "chat": "/api/chat",
            "cbs": "/api/cbs/customers",
        },
    }


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Knight Flow Generic — Dynamic Multi-Agent Platform")
    logger.info(f"  Server: http://{cfg.HOST}:{cfg.PORT}")
    logger.info(f"  Docs:   http://localhost:{cfg.PORT}/docs")
    logger.info("=" * 60)
    uvicorn.run("main:app", host= "192.168.0.155", port=1230, reload=False)
