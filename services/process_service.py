"""Сервис получения описания лекарства через языковую модель.

Пайплайн:

1. валидация и нормализация названия препарата;
2. построение промптов и ключа кэша;
3. проверка кэша (cache hit/miss);
4. вызов LLM с обработкой ошибок и fallback-ответом;
5. пост-обработка ответа и сохранение результата в кэш.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from typing import Any, Protocol

from cache.base import SupportsCache, build_cache_key
from config.settings import Settings, get_settings
from errors import InternalAppError
from llm.drug_description import (
    DRUG_DESCRIPTION_SYSTEM_PROMPT,
    build_user_prompt,
    validate_drug_name,
)
from llm.yandex_gpt import LLMError, LLMNetworkError, LLMTimeoutError
from services.drug_response import (
    DrugDescription,
    InvalidResponseError,
    build_drug_description,
    build_fallback_description,
)

logger = logging.getLogger(__name__)


class SupportsGenerateWithPrompts(Protocol):
    """Интерфейс клиента языковой модели с system- и user-промптом."""

    async def generate_with_prompts(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Возвращает ответ модели на пару промптов."""
        ...


def _model_params(settings: Settings) -> dict[str, Any]:
    """Параметры генерации, влияющие на результат и ключ кэша."""
    return {
        "model": settings.yandex_gpt_model,
        "model_version": settings.yandex_gpt_model_version,
        "temperature": settings.yandex_gpt_temperature,
        "max_tokens": settings.yandex_gpt_max_tokens,
    }


def _short_key(cache_key: str) -> str:
    """Короткое представление ключа кэша для логов."""
    return cache_key[:12]


async def _load_from_cache(
    cache: SupportsCache,
    cache_key: str,
    drug_name: str,
) -> DrugDescription | None:
    """Пытается получить готовое описание из кэша."""
    try:
        payload = await cache.get(cache_key)
    except Exception as exc:  # noqa: BLE001 - кэш не должен ломать запрос
        logger.warning(
            "cache_read_failed",
            extra={
                "event": "cache_read_failed",
                "cache_key": _short_key(cache_key),
                "drug_name": drug_name,
                "error": str(exc),
            },
        )
        return None

    if payload is None:
        logger.info(
            "cache_miss",
            extra={
                "event": "cache_miss",
                "cache_key": _short_key(cache_key),
                "drug_name": drug_name,
            },
        )
        return None

    try:
        description = DrugDescription.from_dict(json.loads(payload))
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning(
            "cache_corrupted",
            extra={
                "event": "cache_corrupted",
                "cache_key": _short_key(cache_key),
                "drug_name": drug_name,
                "error": str(exc),
            },
        )
        return None

    logger.info(
        "cache_hit",
        extra={
            "event": "cache_hit",
            "cache_key": _short_key(cache_key),
            "drug_name": drug_name,
        },
    )
    return replace(description, cached=True)


async def _store_in_cache(
    cache: SupportsCache,
    cache_key: str,
    description: DrugDescription,
    ttl: int,
) -> None:
    """Сохраняет успешный результат в кэш с заданным TTL."""
    try:
        payload = json.dumps(description.to_dict(), ensure_ascii=False)
        await cache.set(cache_key, payload, ttl=ttl)
    except Exception as exc:  # noqa: BLE001 - кэш не должен ломать запрос
        logger.warning(
            "cache_store_failed",
            extra={
                "event": "cache_store_failed",
                "cache_key": _short_key(cache_key),
                "drug_name": description.drug_name,
                "error": str(exc),
            },
        )
        return

    logger.info(
        "cache_store",
        extra={
            "event": "cache_store",
            "cache_key": _short_key(cache_key),
            "drug_name": description.drug_name,
            "ttl": ttl,
        },
    )


async def describe_drug(
    drug_name: str,
    client: SupportsGenerateWithPrompts,
    cache: SupportsCache | None = None,
    settings: Settings | None = None,
) -> DrugDescription:
    """Запрашивает у модели описание лекарства и обрабатывает ответ.

    Args:
        drug_name: название препарата, введённое пользователем.
        client: клиент языковой модели.
        cache: необязательный кэш результатов.
        settings: настройки приложения; по умолчанию берутся из окружения.

    Returns:
        Проверенное, очищенное и отформатированное описание лекарства.

    Raises:
        llm.drug_description.InvalidDrugNameError: если название некорректно.
        llm.yandex_gpt.LLMError: при ошибке обращения к модели
            (если fallback отключён).
        services.drug_response.InvalidResponseError: если ответ модели
            не соответствует ожидаемому формату.
        errors.InternalAppError: при непредвиденной внутренней ошибке.
    """
    settings = settings or get_settings()
    cleaned = validate_drug_name(drug_name)

    system_prompt = DRUG_DESCRIPTION_SYSTEM_PROMPT
    user_prompt = build_user_prompt(cleaned)

    cache_key: str | None = None
    if cache is not None and settings.cache_enabled:
        cache_key = build_cache_key(
            system_prompt,
            user_prompt,
            _model_params(settings),
        )
        cached_description = await _load_from_cache(
            cache, cache_key, cleaned
        )
        if cached_description is not None:
            return cached_description
    else:
        logger.debug(
            "cache_disabled",
            extra={"event": "cache_disabled", "drug_name": cleaned},
        )

    started = time.perf_counter()
    logger.info(
        "llm_call_started",
        extra={
            "event": "llm_call_started",
            "drug_name": cleaned,
            "model": settings.yandex_gpt_model,
            "temperature": settings.yandex_gpt_temperature,
            "max_tokens": settings.yandex_gpt_max_tokens,
        },
    )

    try:
        raw_answer = await client.generate_with_prompts(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except (LLMTimeoutError, LLMNetworkError) as exc:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.error(
            "llm_unavailable",
            extra={
                "event": "llm_unavailable",
                "drug_name": cleaned,
                "error_type": exc.error_type,
                "duration_ms": duration_ms,
                "fallback_enabled": settings.llm_fallback_enabled,
            },
            exc_info=exc,
        )
        if settings.llm_fallback_enabled:
            logger.warning(
                "llm_fallback_used",
                extra={
                    "event": "llm_fallback_used",
                    "drug_name": cleaned,
                    "reason": exc.error_type,
                },
            )
            return build_fallback_description(cleaned)
        raise
    except LLMError as exc:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.error(
            "llm_call_failed",
            extra={
                "event": "llm_call_failed",
                "drug_name": cleaned,
                "error_type": exc.error_type,
                "duration_ms": duration_ms,
            },
            exc_info=exc,
        )
        raise
    except Exception as exc:  # noqa: BLE001 - классифицируем как внутреннюю
        logger.exception(
            "llm_unexpected_error",
            extra={"event": "llm_unexpected_error", "drug_name": cleaned},
        )
        raise InternalAppError(
            "Внутренняя ошибка при обращении к языковой модели"
        ) from exc

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "llm_call_finished",
        extra={
            "event": "llm_call_finished",
            "drug_name": cleaned,
            "duration_ms": duration_ms,
            "answer_chars": len(raw_answer),
        },
    )

    try:
        description = build_drug_description(cleaned, raw_answer)
    except InvalidResponseError as exc:
        logger.error(
            "llm_invalid_response",
            extra={
                "event": "llm_invalid_response",
                "drug_name": cleaned,
                "duration_ms": duration_ms,
                "error": str(exc),
            },
            exc_info=exc,
        )
        raise

    if cache is not None and settings.cache_enabled and cache_key is not None:
        await _store_in_cache(
            cache,
            cache_key,
            description,
            settings.cache_ttl,
        )

    return description
