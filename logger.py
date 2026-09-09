"""
logger.py — Loguru-based logger setup
"""

import sys
from loguru import logger as _loguru_logger

from config import cfg

_loguru_logger.remove()
_loguru_logger.add(
    sys.stdout,
    level=cfg.LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
)


def get_logger(name: str = __name__):
    return _loguru_logger.bind(name=name)
