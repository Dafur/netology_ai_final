"""Тесты кэша: ключ, TTL, in-memory и Redis-бэкенды."""

from __future__ import annotations

import fnmatch
from typing import Any

from cache.base import NullCache, build_cache_key
from cache.factory import create_cache
from cache.memory import InMemoryLLMCache
from cache.redis_cache import RedisLLMCache
from config.settings import Settings


def _run(coro: Any) -> Any:
    """Запускает корутину в синхронном тесте без pytest-asyncio."""
    import asyncio

    return asyncio.run(coro)


class FakeRedis:
    """Минимальная заглушка асинхронного Redis-клиента."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.closed = False

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.values[key] = value
        if ex is not None:
            self.ttls[key] = int(ex)

    async def delete(self, key: str) -> None:
        self.values.pop(key, None)
        self.ttls.pop(key, None)

    async def scan_iter(self, match: str | None = None):
        for key in list(self.values):
            if match is None or fnmatch.fnmatch(key, match):
                yield key

    async def ttl(self, key: str) -> int:
        return self.ttls.get(key, -1)

    async def aclose(self) -> None:
        self.closed = True


class BrokenRedis:
    """Redis, всегда выбрасывающий сетевую ошибку."""

    async def get(self, key: str) -> str | None:
        raise ConnectionError("redis is down")

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        raise ConnectionError("redis is down")


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


def test_redis_cache_uses_prefix_and_ttl() -> None:
    fake = FakeRedis()
    cache = RedisLLMCache(
        url="redis://test",
        ttl=600,
        key_prefix="llm:cache:",
        client=fake,
    )

    _run(cache.set("key", "value"))

    assert fake.values["llm:cache:key"] == "value"
    assert fake.ttls["llm:cache:key"] == 600
    assert _run(cache.get("key")) == "value"


def test_redis_cache_purges_keys_without_ttl() -> None:
    fake = FakeRedis()
    fake.values["llm:cache:stale"] = "value"
    cache = RedisLLMCache(
        url="redis://test",
        key_prefix="llm:cache:",
        client=fake,
    )

    removed = _run(cache.purge_expired())

    assert removed == 1
    assert "llm:cache:stale" not in fake.values


def test_redis_cache_is_fail_open() -> None:
    cache = RedisLLMCache(url="redis://test", client=BrokenRedis())

    assert _run(cache.get("key")) is None
    _run(cache.set("key", "value"))


def test_redis_cache_close() -> None:
    fake = FakeRedis()
    cache = RedisLLMCache(url="redis://test", client=fake)

    _run(cache.close())

    assert fake.closed is True


def test_factory_selects_memory_cache_without_redis() -> None:
    settings = Settings(cache_enabled=True, redis_url="", redis_host="")

    cache = create_cache(settings)

    assert isinstance(cache, InMemoryLLMCache)


def test_factory_selects_null_cache_when_disabled() -> None:
    settings = Settings(cache_enabled=False)

    cache = create_cache(settings)

    assert isinstance(cache, NullCache)


def test_factory_selects_redis_cache_when_configured() -> None:
    settings = Settings(
        cache_enabled=True,
        redis_url="redis://localhost:6379/0",
    )

    cache = create_cache(settings)

    assert isinstance(cache, RedisLLMCache)
