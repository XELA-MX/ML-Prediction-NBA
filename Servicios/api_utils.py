"""
NBA Predictor — Utilidades compartidas para llamadas a API
Retry con backoff exponencial y logging para robustez en producción.
"""

import time
import logging
import functools

log = logging.getLogger(__name__)

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 2.0  # seconds


def retry_api(max_retries: int = DEFAULT_RETRIES,
              backoff: float = DEFAULT_BACKOFF,
              exceptions: tuple = (Exception,)):
    """
    Decorator that retries a function on failure with exponential backoff.

    Usage:
        @retry_api(max_retries=3, backoff=2.0)
        def my_api_call(...):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_retries:
                        wait = backoff * (2 ** (attempt - 1))
                        log.warning(
                            "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__, attempt, max_retries, e, wait
                        )
                        time.sleep(wait)
                    else:
                        log.error(
                            "%s failed after %d attempts: %s",
                            func.__name__, max_retries, e
                        )
            raise last_exc
        return wrapper
    return decorator
