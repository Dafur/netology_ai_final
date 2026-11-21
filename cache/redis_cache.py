"""Redis-реализация кэша результатов LLM."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RedisLLMCache:
    """Кэш на базе Redis с TTL и устойчивостью к сбоям.

    Ошибки Redis не пробрасываются наружу: кэш работает по принципу
    fail-open — при недоступности хранилища приложение продолжает
    обрабатывать запросы без кэша.

    Args:
        url: DSN подключения к Redis.
        ttl: время жизни записи в секундах (по умолчанию 600).
        key_prefix: префикс ключей в Redis.
        client: готовый асинхронный Redis-клиент (используется в тестах).
    """

    def __init__(
        self,
        url: str,
        ttl: int | float = 600,
        key_prefix: str = "llm:cache:",
        client: Any | None = None,
    ) -> None:
        self._url = url
        self._ttl = ttl
        self._prefix = key_prefix
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            # Ленивый импорт: приложение может работать без установленного
            # redis, если Redis не настроен.
            import redis.asyncio as redis

            self._client = redis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> str | None:
        try:
            value = await self._get_client().get(self._full_key(key))
        except Exception as exc:  # noqa: BLE001 - кэш не должен ломать запрос
            logger.warning(
                "cache_get_failed",
                extra={"event": "cache_get_failed", "error": str(exc)},
            )
            return None
        return value if isinstance(value, str) else None

    async def set(
        self,
        key: str,
        value: str,
        ttl: int | float | None = None,
    ) -> None:
        effective_ttl = self._ttl if ttl is None else ttl
        try:
            await self._get_client().set(
                self._full_key(key),
                value,
                ex=max(1, int(effective_ttl)),
            )
        except Exception as exc:  # noqa: BLE001 - кэш не должен ломать запрос
            logger.warning(
                "cache_set_failed",
                extra={"event": "cache_set_failed", "error": str(exc)},
            )

    async def delete(self, key: str) -> None:
        try:
            await self._get_client().delete(self._full_key(key))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cache_delete_failed",
                extra={"event": "cache_delete_failed", "error": str(exc)},
            )

    async def purge_expired(self) -> int:
        """Удаляет ключи без TTL.

        Записи с истёкшим TTL Redis удаляет автоматически; этот метод
        дополнительно убирает «зависшие» ключи без срока жизни.
        """
        removed = 0
        try:
            client = self._get_client()
            async for raw_key in client.scan_iter(match=f"{self._prefix}*"):
                ttl = await client.ttl(raw_key)
                if ttl == -1:
                    await client.delete(raw_key)
                    removed += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cache_purge_failed",
                extra={"event": "cache_purge_failed", "error": str(exc)},
            )
            return removed
        return removed

    async def close(self) -> None:
        if self._client is None:
            return
        close = getattr(self._client, "aclose", None) or getattr(
            self._client, "close", None
        )
        if close is None:
            return
        try:
            await close()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cache_close_failed",
                extra={"event": "cache_close_failed", "error": str(exc)},
            )


__all__ = ["RedisLLMCache"]
