"""Эндпоинт POST /process: описание лекарства через LLM."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi import status as http_status

from api.schemas import ProcessRequest, ProcessResponse
from cache.base import SupportsCache
from cache.factory import get_cache
from llm.yandex_gpt import YandexGPTClient
from services.process_service import describe_drug

router = APIRouter(tags=["process"])


def get_llm_client() -> YandexGPTClient:
    """Фабрика клиента YandexGPT (переопределяется в тестах)."""
    return YandexGPTClient()


@router.post(
    "/process",
    response_model=ProcessResponse,
    response_model_exclude_none=True,
    summary="Получить описание лекарства",
    description=(
        "Принимает название препарата, проверяет кэш, при промахе "
        "запрашивает у YandexGPT структурированную справку, проверяет "
        "и очищает ответ модели, а затем возвращает его в ожидаемом "
        "формате. При недоступности модели возвращается безопасный "
        "fallback-ответ."
    ),
    responses={
        422: {
            "model": ProcessResponse,
            "description": "Ошибка валидации названия препарата",
        },
        http_status.HTTP_502_BAD_GATEWAY: {
            "model": ProcessResponse,
            "description": "Некорректный ответ модели",
        },
        http_status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ProcessResponse,
            "description": "Модель недоступна (fallback отключён)",
        },
        http_status.HTTP_504_GATEWAY_TIMEOUT: {
            "model": ProcessResponse,
            "description": "Таймаут обращения к модели (fallback отключён)",
        },
        http_status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ProcessResponse,
            "description": "Внутренняя ошибка сервера",
        },
    },
)
async def process(
    payload: ProcessRequest,
    client: YandexGPTClient = Depends(get_llm_client),
    cache: SupportsCache = Depends(get_cache),
) -> ProcessResponse:
    """Возвращает проверенное описание лекарства."""
    description = await describe_drug(
        payload.drug_name,
        client,
        cache=cache,
    )
    return ProcessResponse.success(description)
