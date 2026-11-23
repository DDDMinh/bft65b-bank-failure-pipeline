"""Deterministic console logging configuration."""

from __future__ import annotations

import logging
from logging.config import dictConfig

DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO", *, force: bool = False) -> None:
    """Configure package logging without writing log files by default.

    Logs are intentionally console-only because ``*.log`` files are ignored by
    Git; report artifacts must be generated explicitly by the reporting layer.
    """
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    normalized = str(level).upper()
    if normalized not in valid_levels:
        raise ValueError(f"Unsupported log level: {level!r}")

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": DEFAULT_FORMAT,
                    "datefmt": "%Y-%m-%dT%H:%M:%S%z",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": normalized,
                }
            },
            "root": {
                "handlers": ["console"],
                "level": normalized,
            },
        }
    )

    if force:
        for handler in logging.getLogger().handlers:
            handler.setLevel(normalized)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
