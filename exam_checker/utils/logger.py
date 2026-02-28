"""
Centralized logging setup with rotating file and console handlers.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


_loggers = {}


def get_logger(name: str, log_dir: str = None) -> logging.Logger:
    """Get or create a named logger with file + console handlers."""
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        _loggers[name] = logger
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO+)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (DEBUG+)
    if log_dir is None:
        log_dir = str(Path(__file__).parent.parent / "logs")
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_path / "exam_checker.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger
