"""
instrumentation.py

Shared latency-logging decorator.

Design principle
-----------------
Scott's working convention: "Instrument from day one, not retrofitted
after prototypes. Every module logs its inputs, outputs, and latency."
Timing every agent and ingestion stage the same way, with one line at each
call site, is what makes that convention actually followed consistently
rather than reimplemented (and drifting) six different times. What is
semantically interesting to log as "inputs, outputs" differs per stage
(a question and a question_type; a plan and a fact count), so this
decorator handles latency only; each call site adds its own one-line
summary log alongside it.

Author: Scott Josephson  |  Deloitte SEC Filing Intelligence take-home
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


def log_latency(logger: logging.Logger) -> Callable[[F], F]:
    """Log how long a function call took, at INFO level.

    Args:
        logger: The calling module's logger (`logging.getLogger(__name__)`),
            so latency logs are attributed to the stage that produced them.

    Returns:
        A decorator that wraps a function with latency logging.
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info("%s completed in %.1fms", fn.__qualname__, elapsed_ms)
        return wrapper
    return decorator
