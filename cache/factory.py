"""Фабрика кэша и FastAPI-зависимость."""

from __future__ import annotations

import logging
from functools import lru_cache

from cache.base import NullCache, SupportsCache
from cache.memory import InMemoryLLMCache
from cache.redis_cache import RedisLLMCache
from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def create_cache(settings: Settings | None = None) -> SupportsCache:
    """Создаёт кэш по настройкам приложения.

    Приоритет:
    1. кэш отключён -> :class:`NullCache`;
    2. задан Redis DSN -> :class:`RedisLLMCache`;
    3. Redis не настроен -> :class:`InMemoryLLMCache` (dev/test fallback).
    """
    settings = settings or get_settings()

    if not settings.cache_enabled:
        logger.info(
            "cache_backend_selected",
            extra={"event": "cache_backend_selected", "backend": "null"},
        )
        return NullCache()

    dsn = settings.redis_dsn
    if dsn:
        logger.info(
            "cache_backend_selected",
            extra={
                "event": "cache_backend_selected",
                "backend": "redis",
                "ttl": settings.cache_ttl,
            },
        )
        return RedisLLMCache(
            url=dsn,
            ttl=settings.cache_ttl,
            key_prefix=settings.cache_key_prefix,
        )

    logger.warning(
        "cache_backend_selected",
        extra={
            "event": "cache_backend_selected",
            "backend": "memory",
            "reason": "REDIS_URL/REDIS_HOST не заданы",
        },
    )
    return InMemoryLLMCache(ttl=settings.cache_ttl)


@lru_cache(maxsize=1)
def get_cache() -> SupportsCache:
    """Возвращает singleton кэша (FastAPI-зависимость)."""
    return create_cache()


__all__ = ["create_cache", "get_cache"]
