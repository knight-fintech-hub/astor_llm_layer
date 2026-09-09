# from __future__ import annotations

# import json
# from typing import Any, Dict, List, Optional

# import psycopg2
# import psycopg2.extras
# from contextlib import contextmanager

# from config import cfg
# from logger import get_logger

# logger = get_logger(__name__)


# # ── Connection Pool ──────────────────────────────────────────────────────────

# _pool = None


# def get_connection():
#     """Get a database connection."""
#     global _pool
#     if _pool is None or _pool.closed:
#         try:
#             _pool = psycopg2.connect(cfg.DATABASE_URL)
#             _pool.autocommit = True
#             logger.info("[DB] Connected to PostgreSQL")
#         except Exception as e:
#             logger.error(f"[DB] Connection failed: {e}")
#             raise
#     return _pool


# @contextmanager
# def get_cursor():
#     """Context manager for database cursor."""
#     conn = get_connection()
#     cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#     try:
#         yield cursor
#         conn.commit()
#     except Exception as e:
#         conn.rollback()
#         logger.error(f"[DB] Query failed: {e}")
#         raise
#     finally:
#         cursor.close()


# # ── Helper Functions ─────────────────────────────────────────────────────────

# def execute_query(query: str, params: tuple = None) -> List[Dict]:
#     """Execute a SELECT query and return list of dicts."""
#     with get_cursor() as cur:
#         cur.execute(query, params)
#         if cur.description:
#             rows = cur.fetchall()
#             return [dict(row) for row in rows]
#         return []


# def execute_one(query: str, params: tuple = None) -> Optional[Dict]:
#     """Execute a query and return single row."""
#     with get_cursor() as cur:
#         cur.execute(query, params)
#         if cur.description:
#             row = cur.fetchone()
#             return dict(row) if row else None
#         return None


# def execute_write(query: str, params: tuple = None):
#     """Execute INSERT/UPDATE/DELETE."""
#     with get_cursor() as cur:
#         cur.execute(query, params)


# def init_db():
#     """Test database connection on startup."""
#     try:
#         with get_cursor() as cur:
#             cur.execute("SELECT 1")
#         logger.info("[DB] PostgreSQL connection verified")
#         return True
#     except Exception as e:
#         logger.error(f"[DB] Init failed: {e}")
#         return False



from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool
from contextlib import contextmanager

from config import cfg
from logger import get_logger

logger = get_logger(__name__)


# ── Connection Pool ──────────────────────────────────────────────────────────
# ThreadedConnectionPool maintains a set of reusable connections.
# Each concurrent request borrows one connection, uses it, then returns it —
# so 20 simultaneous users each get their own connection with no conflicts.
# Previously this was a single psycopg2.connect() which caused
# "connection already in use" crashes under any concurrent load.

_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None

# Min connections kept open at all times.
# Max connections allowed simultaneously — increase if you see
# "connection pool exhausted" under high load.
_POOL_MIN = 2
_POOL_MAX = 20


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Return the shared connection pool, creating it on first call."""
    global _pool
    if _pool is None or _pool.closed:
        try:
            _pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=_POOL_MIN,
                maxconn=_POOL_MAX,
                dsn=cfg.DATABASE_URL,
            )
            logger.info(
                f"[DB] ThreadedConnectionPool created "
                f"(min={_POOL_MIN}, max={_POOL_MAX})"
            )
        except Exception as e:
            logger.error(f"[DB] Pool creation failed: {e}")
            raise
    return _pool


@contextmanager
def get_cursor():
    """Context manager that borrows a connection from the pool, yields a
    RealDictCursor, commits on success, rolls back on error, and always
    returns the connection to the pool — even if an exception is raised."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"[DB] Query failed: {e}")
            raise
        finally:
            cursor.close()
    finally:
        # Always return the connection — whether success or exception.
        pool.putconn(conn)


# ── Helper Functions ─────────────────────────────────────────────────────────

def execute_query(query: str, params: tuple = None) -> List[Dict]:
    """Execute a SELECT query and return list of dicts."""
    with get_cursor() as cur:
        cur.execute(query, params)
        if cur.description:
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        return []


def execute_one(query: str, params: tuple = None) -> Optional[Dict]:
    """Execute a query and return single row."""
    with get_cursor() as cur:
        cur.execute(query, params)
        if cur.description:
            row = cur.fetchone()
            return dict(row) if row else None
        return None


def execute_write(query: str, params: tuple = None):
    """Execute INSERT/UPDATE/DELETE."""
    with get_cursor() as cur:
        cur.execute(query, params)


def init_db() -> bool:
    """Initialise the pool and verify connectivity on startup."""
    try:
        # Force pool creation now so any misconfiguration surfaces at startup.
        _get_pool()
        with get_cursor() as cur:
            cur.execute("SELECT 1")
        logger.info("[DB] PostgreSQL connection pool verified")
        return True
    except Exception as e:
        logger.error(f"[DB] Init failed: {e}")
        return False


def close_pool():
    """Gracefully close all pool connections (call on server shutdown)."""
    global _pool
    if _pool and not _pool.closed:
        _pool.closeall()
        logger.info("[DB] Connection pool closed")