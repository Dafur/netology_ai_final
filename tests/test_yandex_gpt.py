"""Тесты клиента YandexGPT."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from config.settings import Settings
from llm.yandex_gpt import (
    LLMError,
    LLMNetworkError,
    LLMResponseError,
    LLMTimeoutError,
    YandexGPTClient,
)


def _settings(**overrides: Any) -> Settings:
    """Настройки с тестовыми значениями, не зависящими от окружения."""
    values: dict[str, Any] = {
        "yandex_gpt_api_key": "test-api-key",
        "yandex_gpt_folder_id": "test-folder",
        "yandex_gpt_api_url": "https://llm.test/v1/completion",
        "request_timeout": 5.0,
        "llm_connect_timeout": 1.0,
        "llm_read_timeout": 2.0,
    }
    values.update(overrides)
    return Settings(**values)


def _run(coro: Any) -> Any:
    """Запускает корутину в синхронном тесте без pytest-asyncio."""
    return asyncio.run(coro)


def _ok_response(text: str = "  Ответ модели  ") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "result": {
                "alternatives": [{"message": {"text": text}}],
            }
        },
    )


def test_client_sends_system_and_user_prompts() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        captured["authorization"] = request.headers["authorization"]
        return _ok_response()

    client = YandexGPTClient(
        settings=_settings(),
        transport=httpx.MockTransport(handler),
    )

    answer = _run(client.generate_with_prompts("Системная роль", "Запрос"))

    assert answer == "Ответ модели"
    assert captured["authorization"] == "Api-Key test-api-key"
    messages = captured["payload"]["messages"]
    assert messages == [
        {"role": "system", "text": "Системная роль"},
        {"role": "user", "text": "Запрос"},
    ]
    assert captured["payload"]["completionOptions"]["stream"] is False


def test_client_uses_iam_token_when_api_key_missing() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["authorization"]
        return _ok_response()

    client = YandexGPTClient(
        settings=_settings(
            yandex_gpt_api_key="",
            yandex_gpt_iam_token="test-iam-token",
        ),
        transport=httpx.MockTransport(handler),
    )

    _run(client.generate("Привет"))

    assert captured["authorization"] == "Bearer test-iam-token"


def test_client_raises_timeout_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = YandexGPTClient(
        settings=_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMTimeoutError, match="время ожидания"):
        _run(client.generate("Привет"))


def test_client_raises_network_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = YandexGPTClient(
        settings=_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMNetworkError, match="Ошибка сети"):
        _run(client.generate("Привет"))


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, "авторизации"),
        (403, "авторизации"),
        (429, "лимит запросов"),
        (500, "временно недоступен"),
        (503, "временно недоступен"),
        (400, "вернул ошибку"),
    ],
)
def test_client_maps_http_status_to_llm_error(
    status_code: int,
    expected: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text="error details")

    client = YandexGPTClient(
        settings=_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMResponseError, match=expected):
        _run(client.generate("Привет"))


def test_client_rejects_invalid_json() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not a json")

    client = YandexGPTClient(
        settings=_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMResponseError, match="Некорректный ответ"):
        _run(client.generate("Привет"))


def test_client_rejects_empty_answer() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response("   ")

    client = YandexGPTClient(
        settings=_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMResponseError, match="пустой ответ"):
        _run(client.generate("Привет"))


def test_client_requires_folder_id() -> None:
    client = YandexGPTClient(settings=_settings(yandex_gpt_folder_id=""))

    with pytest.raises(LLMError, match="YANDEX_GPT_FOLDER_ID"):
        _run(client.generate("Привет"))


def test_client_requires_auth_credentials() -> None:
    client = YandexGPTClient(
        settings=_settings(yandex_gpt_api_key="", yandex_gpt_iam_token="")
    )

    with pytest.raises(LLMError, match="YANDEX_GPT_API_KEY"):
        _run(client.generate("Привет"))
