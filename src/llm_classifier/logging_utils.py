"""日志横切关注点 — 共享 logger 和文件 handler 工具函数。"""

import logging
from pathlib import Path

log = logging.getLogger("classify")
log.setLevel(logging.INFO)
if not any(getattr(h, "_classify_console", False) for h in log.handlers):
    _ch = logging.StreamHandler()
    _ch._classify_console = True
    _ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_ch)


def add_file_log(log_path: Path) -> logging.FileHandler:
    """给 log 增加一个文件 handler，返回 handler 以便后续移除。"""
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(handler)
    return handler
