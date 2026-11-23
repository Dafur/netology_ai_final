"""Тесты кэша: ключ, TTL, LRU-вытеснение и выбор бэкенда."""

from __future__ import annotations

from typing import Any

from cache.base import NullCache, build_cache_key
from cache.factory import create_cache
from cache.memory import InMemoryLLMCache
from config.settings import Settings


def _run(coro: Any) -> Any:
    """Запускает корутину в синхронном тесте без pytest-asyncio."""
    import asyncio

    return asyncio.run(coro)


def test_cache_key_is_deterministic_and_order_independent() -> None:
    first = build_cache_key(
        "system", "user", {"temperature": 0.3, "max_tokens": 1000}
    )
    second = build_cache_key(
        "system", "user", {"max_tokens": 1000, "temperature": 0.3}
    )

    assert first == second
    assert len(first) == 64


def test_cache_key_depends_on_prompts_and_params() -> None:
    base = build_cache_key("system", "user", {"temperature": 0.3})
    other_prompt = build_cache_key("system", "other user", {"temperature": 0.3})
    other_params = build_cache_key("system", "user", {"temperature": 0.7})

    assert base != other_prompt
    assert base != other_params


def test_memory_cache_set_and_get() -> None:
    cache = InMemoryLLMCache(ttl=600)

    _run(cache.set("key", "value"))

    assert _run(cache.get("key")) == "value"
    assert _run(cache.get("missing")) is None


def test_memory_cache_expires_by_ttl() -> None:
    now = [0.0]
    cache = InMemoryLLMCache(ttl=600, clock=lambda: now[0])
    _run(cache.set("key", "value"))

    now[0] = 599.0
    assert _run(cache.get("key")) == "value"

    now[0] = 601.0
    assert _run(cache.get("key")) is None


def test_memory_cache_purges_expired_entries() -> None:
    now = [0.0]
    cache = InMemoryLLMCache(ttl=600, clock=lambda: now[0])
    _run(cache.set("first", "1"))
    _run(cache.set("second", "2"))

    now[0] = 601.0

    assert _run(cache.purge_expired()) == 2
    assert _run(cache.get("first")) is None


def test_memory_cache_delete() -> None:
    cache = InMemoryLLMCache(ttl=600)
    _run(cache.set("key", "value"))

    _run(cache.delete("key"))

    assert _run(cache.get("key")) is None


def test_memory_cache_evicts_least_recently_used() -> None:
    cache = InMemoryLLMCache(ttl=600, max_entries=2)
    _run(cache.set("first", "1"))
    _run(cache.set("second", "2"))

    # Обращение к first делает second самым давно использованным.
    assert _run(cache.get("first")) == "1"
    _run(cache.set("third", "3"))

    assert _run(cache.get("second")) is None
    assert _run(cache.get("first")) == "1"
    assert _run(cache.get("third")) == "3"
    assert len(cache) == 2


def test_memory_cache_evicts_expired_before_lru() -> None:
    now = [0.0]
    cache = InMemoryLLMCache(ttl=600, max_entries=2, clock=lambda: now[0])
    _run(cache.set("expired", "old"))
    # У «свежей» записи больший TTL, поэтому она не устаревает.
    _run(cache.set("fresh", "new", ttl=10000))

    now[0] = 601.0
    _run(cache.set("newest", "value"))

    assert _run(cache.get("expired")) is None
    assert _run(cache.get("fresh")) == "new"
    assert _run(cache.get("newest")) == "value"
    assert len(cache) == 2


def test_memory_cache_close_clears_entries() -> None:
    cache = InMemoryLLMCache(ttl=600)
    _run(cache.set("key", "value"))

    _run(cache.close())

    assert _run(cache.get("key")) is None
    assert len(cache) == 0


def test_factory_selects_memory_cache_when_enabled() -> None:
    settings = Settings(cache_enabled=True, cache_ttl=600, cache_max_entries=50)

    cache = create_cache(settings)

    assert isinstance(cache, InMemoryLLMCache)


def test_factory_selects_null_cache_when_disabled() -> None:
    settings = Settings(cache_enabled=False)

    cache = create_cache(settings)

    assert isinstance(cache, NullCache)
