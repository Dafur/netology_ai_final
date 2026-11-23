"""Фабрика кэша и FastAPI-зависимость."""

from __future__ import annotations

import logging
from functools import lru_cache

from cache.base import NullCache, SupportsCache
from cache.memory import InMemoryLLMCache
from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def create_cache(settings: Settings | None = None) -> SupportsCache:
    """Создаёт кэш по настройкам приложения.

    Кэш работает в памяти процесса:

    - при ``CACHE_ENABLED=false`` возвращается :class:`NullCache`;
    - иначе — :class:`InMemoryLLMCache` с TTL и ограничением числа записей.
    """
    settings = settings or get_settings()

    if not settings.cache_enabled:
        logger.info(
            "cache_backend_selected",
            extra={"event": "cache_backend_selected", "backend": "null"},
        )
        return NullCache()

    logger.info(
        "cache_backend_selected",
        extra={
            "event": "cache_backend_selected",
            "backend": "memory",
            "ttl": settings.cache_ttl,
            "max_entries": settings.cache_max_entries,
        },
    )
    return InMemoryLLMCache(
        ttl=settings.cache_ttl,
        max_entries=settings.cache_max_entries,
    )


@lru_cache(maxsize=1)
def get_cache() -> SupportsCache:
    """Возвращает singleton кэша (FastAPI-зависимость)."""
    return create_cache()


__all__ = ["create_cache", "get_cache"]
