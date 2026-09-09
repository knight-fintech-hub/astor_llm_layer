"""
config.py — Central configuration loaded from .env
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Server
    HOST: str = os.getenv("HOST", "192.168.0.155")
    PORT: int = int(os.getenv("PORT", "1230"))

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    # If False, agent config is loaded from local agent_config.json (testing mode)
    USE_DB: bool = os.getenv("USE_DB", "true").lower() == "true"
    # Path to local agent config JSON (used when USE_DB=false)
    LOCAL_AGENT_CONFIG_PATH: str = os.getenv(
        "LOCAL_AGENT_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "agent_config.json"),
    )

    # LLM Model
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        r"E:\DS_Team_Share\hugging-face-model\models--Qwen--Qwen2.5-7B-Instruct-1M\snapshots\e28526f7bb80e2a9c8af03b831a9af3812f18fba",
    )
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "50"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.9"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))
    TOP_K: int = int(os.getenv("TOP_K", "40"))

    # Flow routing
    # Flex Mode = LLM fallback that reconsiders where to route when no edge
    # matched. When enabled it is now EDGE-CONSTRAINED (only the current node's
    # legal transition targets are candidates — Retell-style). Default OFF for
    # strict, predictable flow behavior; set FLEX_MODE_ENABLED=true to allow the
    # constrained fallback.
    FLEX_MODE_ENABLED: bool = os.getenv("FLEX_MODE_ENABLED", "false").lower() == "true"

    # Lender
    LENDER_NAME: str = os.getenv("LENDER_NAME", "Knight Gini")

    # Gradio
    GRADIO_PORT: int = int(os.getenv("GRADIO_PORT", "7870"))

    # KV Cache — prefill-phase system prompt caching
    # Caches past_key_values for the static system prompt prefix so subsequent
    # generate() calls skip re-processing the same tokens through the network.
    # Only active on CUDA; automatically disabled on CPU.
    KV_CACHE_ENABLED: bool = os.getenv("KV_CACHE_ENABLED", "true").lower() == "true"
    # Maximum number of distinct system prompts to keep cached simultaneously.
    # Each entry ≈ 73 MB on a 7B model with a 512-token prompt; 4 entries ≈ 292 MB.
    KV_CACHE_MAX_ENTRIES: int = int(os.getenv("KV_CACHE_MAX_ENTRIES", "4"))

    # Multithreading / LLM concurrency
    # MAX_CONCURRENT_LLM: how many model.generate() calls may run simultaneously.
    # Keep at 1 for a single GPU — increasing it only helps with multiple GPUs.
    MAX_CONCURRENT_LLM: int = int(os.getenv("MAX_CONCURRENT_LLM", "1"))
    # Maximum number of requests allowed to wait in the LLM queue before rejecting.
    LLM_QUEUE_SIZE: int = int(os.getenv("LLM_QUEUE_SIZE", "50"))
    # Seconds a request may wait in the queue before receiving a 503 "busy" error.
    LLM_REQUEST_TIMEOUT_SEC: int = int(os.getenv("LLM_REQUEST_TIMEOUT_SEC", "60"))

    # Session cache TTL — how long (seconds) an idle session stays in the
    # in-memory cache before being evicted (it remains safe in PostgreSQL).
    SESSION_CACHE_TTL_SEC: int = int(os.getenv("SESSION_CACHE_TTL_SEC", "3600"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Data directory (fallback for CBS CSV data)
    DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")


cfg = Config()
