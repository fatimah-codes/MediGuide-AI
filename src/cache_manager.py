"""LLM caching: in-memory (RAM) vs SQLite (file on disk).

LangChain checks the registered cache automatically before every model call,
so we only have to register it once with set_llm_cache(...).
"""
from pathlib import Path

from langchain_community.cache import SQLiteCache
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
SQLITE_PATH = CACHE_DIR / "langchain_cache.db"

CACHE_OFF, CACHE_MEMORY, CACHE_SQLITE = "Off", "In-memory", "SQLite"
CACHE_MODES = [CACHE_OFF, CACHE_MEMORY, CACHE_SQLITE]

# A single in-memory cache object for the whole Streamlit process, so it
# survives script reruns (but NOT a server restart).
_MEMORY_CACHE = InMemoryCache()

# Streaming does not always go through the LLM cache, so streamed narratives are
# remembered in this small dictionary and replayed instantly on repeat requests.
_NARRATIVE_MEMO: dict[str, str] = {}


def setup_cache(mode: str) -> None:
    """Register the chosen cache globally (or switch caching off)."""
    if mode == CACHE_MEMORY:
        set_llm_cache(_MEMORY_CACHE)
    elif mode == CACHE_SQLITE:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=str(SQLITE_PATH)))
    else:
        set_llm_cache(None)


def clear_cache(mode: str | None = None) -> None:
    """Empty the chosen cache (or both when mode is None)."""
    if mode in (None, CACHE_MEMORY):
        _MEMORY_CACHE.clear()
    if mode in (None, CACHE_SQLITE) and SQLITE_PATH.exists():
        SQLiteCache(database_path=str(SQLITE_PATH)).clear()
    _NARRATIVE_MEMO.clear()


def sqlite_size_kb() -> float:
    """Size of the SQLite cache file in kilobytes (0 if it does not exist)."""
    return round(SQLITE_PATH.stat().st_size / 1024, 1) if SQLITE_PATH.exists() else 0.0


def get_narrative(key: str) -> str | None:
    return _NARRATIVE_MEMO.get(key)


def put_narrative(key: str, text: str) -> None:
    _NARRATIVE_MEMO[key] = text
