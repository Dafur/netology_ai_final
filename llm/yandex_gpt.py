"""Асинхронный клиент YandexGPT."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from config.settings import Settings, get_settings
from errors import AppError

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "Ты — полезный ассистент. Отвечай кратко и по делу."
# Обратная совместимость с прежним именем константы.
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

MAX_ERROR_DETAIL_LENGTH = 500
MAX_PROMPT_PREVIEW_LENGTH = 200
MAX_RESPONSE_PREVIEW_LENGTH = 1000


class LLMError(AppError):
    """Базовая ошибка при обращении к языковой модели."""

    status_code = 503
    error_type = "external"
    default_message = "Сервис языковой модели недоступен"


class LLMTimeoutError(LLMError):
    """Истёк таймаут ожидания ответа от языковой модели."""

    status_code = 504
    error_type = "timeout"
    default_message = "Превышено время ожидания ответа языковой модели"


class LLMNetworkError(LLMError):
    """Сетевая ошибка при обращении к языковой модели."""

    status_code = 503
    error_type = "network"
    default_message = "Ошибка сети при обращении к языковой модели"


class LLMResponseError(LLMError):
    """Модель вернула некорректный или пустой ответ."""

    status_code = 502
    error_type = "model"
    default_message = "Модель вернула некорректный ответ"


class YandexGPTClient:
    """Клиент для YandexGPT Completion API.

    Args:
        settings: настройки приложения. Если не переданы, используются
            настройки из переменных окружения.
        transport: необязательный httpx-транспорт (используется в тестах).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._transport = transport

    def _auth_headers(self) -> dict[str, str]:
        """Формирует заголовки авторизации (API-ключ или IAM-токен)."""
        settings = self._settings
        if settings.yandex_gpt_api_key:
            authorization = f"Api-Key {settings.yandex_gpt_api_key}"
        elif settings.yandex_gpt_iam_token:
            authorization = f"Bearer {settings.yandex_gpt_iam_token}"
        else:
            raise LLMError(
                "Не задан YANDEX_GPT_API_KEY или YANDEX_GPT_IAM_TOKEN"
            )
        return {
            "Authorization": authorization,
            "Content-Type": "application/json",
        }

    def _build_timeout(self) -> httpx.Timeout:
        """Собирает таймауты подключения, чтения, записи и пула соединений."""
        settings = self._settings
        return httpx.Timeout(
            timeout=settings.request_timeout,
            connect=settings.llm_connect_timeout,
            read=settings.llm_read_timeout,
            write=settings.llm_write_timeout,
            pool=settings.llm_pool_timeout,
        )

    def _build_payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Формирует тело запроса к Completion API.

        Args:
            system_prompt: системная инструкция, задающая роль и формат ответа.
            user_prompt: пользовательский запрос.
        """
        settings = self._settings
        return {
            "modelUri": settings.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": settings.yandex_gpt_temperature,
                "maxTokens": str(settings.yandex_gpt_max_tokens),
            },
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_prompt},
            ],
        }

    async def generate(self, message: str) -> str:
        """Отправляет сообщение в YandexGPT и возвращает текст ответа."""
        return await self.generate_with_prompts(DEFAULT_SYSTEM_PROMPT, message)

    async def generate_with_prompts(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Отправляет system- и user-промпт в YandexGPT и возвращает ответ.

        Args:
            system_prompt: системная инструкция для модели.
            user_prompt: пользовательский запрос.

        Returns:
            Текст ответа модели без обрамляющих пробелов.

        Raises:
            LLMError: если не заданы обязательные настройки или промпты пусты.
            LLMTimeoutError: если истёк таймаут запроса.
            LLMNetworkError: при сетевой ошибке или недоступности сервиса.
            LLMResponseError: если ответ модели некорректен или пуст.
        """
        settings = self._settings
        if not settings.yandex_gpt_folder_id:
            raise LLMError("Не задан YANDEX_GPT_FOLDER_ID")

        system_prompt = system_prompt.strip()
        user_prompt = user_prompt.strip()
        if not system_prompt:
            raise LLMError("system_prompt не должен быть пустым")
        if not user_prompt:
            raise LLMError("user_prompt не должен быть пустым")

        headers = self._auth_headers()
        payload = self._build_payload(system_prompt, user_prompt)

        started = time.perf_counter()
        logger.info(
            "llm_request_started",
            extra={
                "event": "llm_request_started",
                "model": settings.yandex_gpt_model,
                "model_version": settings.yandex_gpt_model_version,
                "temperature": settings.yandex_gpt_temperature,
                "max_tokens": settings.yandex_gpt_max_tokens,
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
                "system_prompt_preview": system_prompt[:MAX_PROMPT_PREVIEW_LENGTH],
                "user_prompt_preview": user_prompt[:MAX_PROMPT_PREVIEW_LENGTH],
            },
        )
        logger.debug(
            "llm_prompt",
            extra={
                "event": "llm_prompt",
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
        )

        try:
            async with httpx.AsyncClient(
                timeout=self._build_timeout(),
                transport=self._transport,
            ) as client:
                response = await client.post(
                    settings.yandex_gpt_api_url,
                    headers=headers,
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            logger.error(
                "llm_timeout",
                extra={
                    "event": "llm_timeout",
                    "duration_ms": self._elapsed_ms(started),
                    "error": str(exc),
                },
                exc_info=exc,
            )
            raise LLMTimeoutError(
                "Превышено время ожидания ответа YandexGPT"
            ) from exc
        except httpx.HTTPError as exc:
            logger.error(
                "llm_network_error",
                extra={
                    "event": "llm_network_error",
                    "duration_ms": self._elapsed_ms(started),
                    "error": str(exc),
                },
                exc_info=exc,
            )
            raise LLMNetworkError(
                f"Ошибка сети при обращении к YandexGPT: {exc}"
            ) from exc

        if response.status_code != httpx.codes.OK:
            detail = response.text[:MAX_ERROR_DETAIL_LENGTH]
            logger.error(
                "llm_http_error",
                extra={
                    "event": "llm_http_error",
                    "status_code": response.status_code,
                    "duration_ms": self._elapsed_ms(started),
                    "error_detail": detail,
                },
            )
            raise LLMResponseError(
                self._status_error_message(response.status_code, detail)
            )

        answer = self._extract_answer(response)
        logger.info(
            "llm_response_received",
            extra={
                "event": "llm_response_received",
                "status_code": response.status_code,
                "duration_ms": self._elapsed_ms(started),
                "answer_chars": len(answer),
            },
        )
        logger.debug(
            "llm_response_body",
            extra={
                "event": "llm_response_body",
                "answer_preview": answer[:MAX_RESPONSE_PREVIEW_LENGTH],
            },
        )
        return answer

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        """Возвращает прошедшее время в миллисекундах."""
        return round((time.perf_counter() - started) * 1000, 2)

    @staticmethod
    def _status_error_message(status_code: int, detail: str) -> str:
        """Формирует понятное сообщение об ошибке по HTTP-статусу."""
        if status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            reason = "ошибка авторизации в YandexGPT"
        elif status_code == httpx.codes.TOO_MANY_REQUESTS:
            reason = "превышен лимит запросов к YandexGPT"
        elif status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
            reason = "YandexGPT временно недоступен"
        else:
            reason = "YandexGPT вернул ошибку"
        return f"{reason} ({status_code}): {detail}"

    @staticmethod
    def _extract_answer(response: httpx.Response) -> str:
        """Извлекает текст ответа из JSON-структуры Completion API."""
        try:
            data = response.json()
            answer = data["result"]["alternatives"][0]["message"]["text"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning(
                "Некорректный ответ YandexGPT: %s",
                response.text[:MAX_ERROR_DETAIL_LENGTH],
            )
            raise LLMResponseError("Некорректный ответ YandexGPT") from exc

        if not isinstance(answer, str) or not answer.strip():
            raise LLMResponseError("YandexGPT вернул пустой ответ")

        return answer.strip()
