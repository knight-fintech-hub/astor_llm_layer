"""
logger.py — Loguru-based logger setup
"""

import sys
from loguru import logger as _loguru_logger

from config import cfg

_loguru_logger.remove()
# Console Sink
_loguru_logger.add(
    sys.stdout,
    level=cfg.LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
)
# File Sink (Logs to logs/app.log, rotates at 10MB, keeps last 30 days)
_loguru_logger.add(
    "logs/app.log",
    level=cfg.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    rotation="10 MB",
    retention="30 days",
    enqueue=True, # Thread-safe async logging
)

def get_logger(name: str = __name__):
    return _loguru_logger.bind(name=name)
