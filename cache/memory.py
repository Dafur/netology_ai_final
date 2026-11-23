"""In-memory реализация кэша результатов LLM."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Entry:
    """Запись кэша с абсолютным временем истечения."""

    value: str
    expires_at: float | None


class InMemoryLLMCache:
    """Кэш в памяти процесса с поддержкой TTL и ограничением размера.

    Записи хранятся в ``OrderedDict`` и упорядочены по времени последнего
    использования: при обращении запись перемещается в конец. Когда число
    записей превышает ``max_entries``, сначала удаляются устаревшие записи,
    а затем — самые давно использованные (LRU), чтобы кэш не рос бесконечно.

    Args:
        ttl: время жизни записи по умолчанию в секундах.
        max_entries: максимальное число записей; ``None`` — без ограничения.
        clock: источник времени (в тестах подменяется для детерминизма).
    """

    def __init__(
        self,
        ttl: int | float = 600,
        max_entries: int | None = 1000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl
        self._max_entries = max_entries
        self._clock = clock
        self._data: OrderedDict[str, _Entry] = OrderedDict()

    def __len__(self) -> int:
        """Текущее число записей в кэше (включая ещё не удалённые устаревшие)."""
        return len(self._data)

    def _is_expired(self, entry: _Entry, now: float) -> bool:
        return entry.expires_at is not None and entry.expires_at <= now

    async def get(self, key: str) -> str | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if self._is_expired(entry, self._clock()):
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
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
        self._data.move_to_end(key)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Удаляет устаревшие записи, затем самые давно использованные."""
        if self._max_entries is None:
            return
        if len(self._data) <= self._max_entries:
            return

        now = self._clock()
        expired = [
            key
            for key, entry in self._data.items()
            if self._is_expired(entry, now)
        ]
        for key in expired:
            self._data.pop(key, None)

        while len(self._data) > self._max_entries:
            self._data.popitem(last=False)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def purge_expired(self) -> int:
        now = self._clock()
        expired = [
            key
            for key, entry in self._data.items()
            if self._is_expired(entry, now)
        ]
        for key in expired:
            self._data.pop(key, None)
        return len(expired)

    async def close(self) -> None:
        self._data.clear()


__all__ = ["InMemoryLLMCache"]
