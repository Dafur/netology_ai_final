"""In-memory реализация кэша (для разработки и тестов)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Entry:
    """Запись кэша с абсолютным временем истечения."""

    value: str
    expires_at: float | None


class InMemoryLLMCache:
    """Простой кэш в памяти процесса с поддержкой TTL.

    Используется, когда Redis не настроен, а также в тестах. Для
    детерминированной проверки истечения можно передать собственные часы.
    """

    def __init__(
        self,
        ttl: int | float = 600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl
        self._clock = clock
        self._data: dict[str, _Entry] = {}

    async def get(self, key: str) -> str | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if entry.expires_at is not None and entry.expires_at <= self._clock():
            self._data.pop(key, None)
            return None
        return entry.value

    async def set(
        self,
        key: str,
        value: str,
        ttl: int | float | None = None,
    ) -> None:
        effective_ttl = self._ttl if ttl is None else ttl
        expires_at = (
            self._clock() + effective_ttl
            if effective_ttl is not None and effective_ttl > 0
            else None
        )
        self._data[key] = _Entry(value=value, expires_at=expires_at)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def purge_expired(self) -> int:
        now = self._clock()
        expired = [
            key
            for key, entry in self._data.items()
            if entry.expires_at is not None and entry.expires_at <= now
        ]
        for key in expired:
            self._data.pop(key, None)
        return len(expired)

    async def close(self) -> None:
        self._data.clear()


__all__ = ["InMemoryLLMCache"]
