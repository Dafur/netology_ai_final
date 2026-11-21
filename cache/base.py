"""Базовые абстракции кэша результатов LLM."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

# Версия схемы ключа. Меняется при изменении состава ключа или формата
# сохраняемых данных, чтобы старые записи не смешивались с новыми.
CACHE_KEY_VERSION = "v1"


@runtime_checkable
class SupportsCache(Protocol):
    """Интерфейс кэша, используемого в пайплайне обработки запроса."""

    async def get(self, key: str) -> str | None:
        """Возвращает значение по ключу или ``None``, если его нет."""
        ...

    async def set(
        self,
        key: str,
        value: str,
        ttl: int | float | None = None,
    ) -> None:
        """Сохраняет значение с временем жизни ``ttl`` секунд."""
        ...

    async def delete(self, key: str) -> None:
        """Удаляет значение по ключу."""
        ...

    async def purge_expired(self) -> int:
        """Удаляет устаревшие записи и возвращает их количество."""
        ...

    async def close(self) -> None:
        """Освобождает ресурсы кэша."""
        ...


class NullCache:
    """Кэш-заглушка: всегда промах, ничего не сохраняет."""

    async def get(self, key: str) -> str | None:
        return None

    async def set(
        self,
        key: str,
        value: str,
        ttl: int | float | None = None,
    ) -> None:
        return None

    async def delete(self, key: str) -> None:
        return None

    async def purge_expired(self) -> int:
        return 0

    async def close(self) -> None:
        return None


def build_cache_key(
    system_prompt: str,
    user_prompt: str,
    params: Mapping[str, Any],
) -> str:
    """Формирует детерминированный ключ кэша.

    Ключ учитывает входные данные (system- и user-промпт) и параметры
    генерации (модель, температура, лимит токенов и т.п.), поэтому ответы
    для разных параметров не смешиваются.

    Args:
        system_prompt: системная инструкция.
        user_prompt: пользовательский запрос.
        params: параметры модели и версия схемы ключа.

    Returns:
        SHA-256 хеш канонического JSON-представления в hex-виде.
    """
    payload = {
        "version": CACHE_KEY_VERSION,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "params": dict(params),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "CACHE_KEY_VERSION",
    "NullCache",
    "SupportsCache",
    "build_cache_key",
]
