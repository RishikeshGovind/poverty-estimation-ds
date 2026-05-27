"""Simple in-memory TTL cache so repeated API calls don't re-fetch."""
import time
from typing import Any

_cache: dict[str, tuple[Any, float]] = {}
TTL = 3600  # seconds


def get(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry and time.time() - entry[1] < TTL:
        return entry[0]
    return None


def set(key: str, value: Any) -> None:
    _cache[key] = (value, time.time())
