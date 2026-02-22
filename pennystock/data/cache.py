"""Simple file-based cache with TTL to avoid hammering APIs."""

import hashlib
import json
import os
import time

from loguru import logger

from pennystock.config import CACHE_DIR, CACHE_TTL_HOURS


def _cache_path(key: str) -> str:
    """Get filesystem path for a cache key."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    hashed = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hashed}.json")


def cache_get(key: str):
    """Retrieve a cached value if it exists and hasn't expired."""
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        age_hours = (time.time() - data["timestamp"]) / 3600
        if age_hours > CACHE_TTL_HOURS:
            os.remove(path)
            return None
        return data["value"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def cache_set(key: str, value):
    """Store a value in the cache."""
    path = _cache_path(key)
    try:
        with open(path, "w") as f:
            json.dump({"timestamp": time.time(), "value": value}, f)
    except (OSError, TypeError) as e:
        logger.warning(f"Cache write failed for {key}: {e}")


def cache_clear():
    """Remove all cached files."""
    if os.path.exists(CACHE_DIR):
        for fname in os.listdir(CACHE_DIR):
            fpath = os.path.join(CACHE_DIR, fname)
            if fname.endswith(".json"):
                os.remove(fpath)
        logger.info("Cache cleared")
