"""Shared logging setup.

Replaces the ad-hoc ``print()`` calls that used to be scattered through
``webui.py``, so that failures inside Gradio event handlers leave a real
traceback behind instead of only a one-line ``gr.Warning``.
"""

import logging
import os
import sys

logger = logging.getLogger("chatbot")

_LEVEL_ENV_VAR = "CHATBOT_LOG_LEVEL"


def setup_logging():
    """Configure the ``chatbot`` logger once, at process start."""
    if logger.handlers:
        return logger

    level_name = os.getenv(_LEVEL_ENV_VAR, "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))

    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def truncate(text, limit=2000):
    """Shorten text for logging so a base64 image or a whole PDF cannot flood the console."""
    text = "" if text is None else str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text) - limit} more chars]"
