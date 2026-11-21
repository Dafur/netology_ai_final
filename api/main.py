"""Точка входа FastAPI-приложения."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.process import router as process_router
from api.schemas import ProcessResponse
from cache.factory import get_cache
from config.logging import (
    configure_logging,
    reset_request_id,
    set_request_id,
)
from config.settings import get_settings
from errors import AppError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Настраивает логирование и освобождает ресурсы при остановке."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    logger.info(
        "app_started",
        extra={
            "event": "app_started",
            "version": app.version,
            "cache_ttl": settings.cache_ttl,
            "fallback_enabled": settings.llm_fallback_enabled,
        },
    )
    try:
        yield
    finally:
        try:
            cache = get_cache()
            await cache.close()
            get_cache.cache_clear()
        except Exception:  # noqa: BLE001 - не мешаем остановке приложения
            logger.warning(
                "cache_close_failed",
                extra={"event": "cache_close_failed"},
                exc_info=True,
            )
        logger.info("app_stopped", extra={"event": "app_stopped"})


app = FastAPI(
    title="Netology AI Final API",
    version="0.4.0",
    description=(
        "API для получения структурированного описания лекарства "
        "через YandexGPT с кэшированием, fallback-ответом и "
        "структурированным логированием."
    ),
    lifespan=lifespan,
)

app.include_router(process_router)


@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next,
):
    """Логирует ключевые этапы HTTP-запроса и прокидывает request_id."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = set_request_id(request_id)
    started = time.perf_counter()
    status_code = 500

    logger.info(
        "request_started",
        extra={
            "event": "request_started",
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None,
        },
    )

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:  # noqa: BLE001 - фиксируем и пробрасываем дальше
        logger.exception(
            "request_failed",
            extra={
                "event": "request_failed",
                "method": request.method,
                "path": request.url.path,
            },
        )
        raise
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "request_finished",
            extra={
                "event": "request_finished",
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )
        reset_request_id(token)


def _error_response(status_code: int, error: str) -> JSONResponse:
    """Формирует JSON-ответ об ошибке в едином формате."""
    content = ProcessResponse.failure(error).model_dump(exclude_none=True)
    return JSONResponse(status_code=status_code, content=content)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Приводит ошибки валидации Pydantic к единому формату ответа."""
    messages: list[str] = []
    for error in exc.errors():
        location = ".".join(
            str(part) for part in error.get("loc", ()) if part != "body"
        )
        message = str(error.get("msg", "Некорректное значение"))
        messages.append(f"{location}: {message}" if location else message)

    detail = "; ".join(messages) or "Некорректные входные данные"
    logger.warning(
        "validation_error",
        extra={
            "event": "validation_error",
            "path": request.url.path,
            "error_type": "validation",
            "errors": detail,
        },
    )
    return _error_response(422, detail)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Возвращает корректный код для типизированных ошибок приложения."""
    logger.error(
        "application_error",
        extra={
            "event": "application_error",
            "path": request.url.path,
            "error_type": exc.error_type,
            "status_code": exc.status_code,
        },
        exc_info=exc,
    )
    return _error_response(exc.status_code, exc.public_message)


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Приводит HTTPException к единому формату ответа."""
    detail = exc.detail if isinstance(exc.detail, str) else "Ошибка обработки запроса"
    logger.warning(
        "http_error",
        extra={
            "event": "http_error",
            "path": request.url.path,
            "status_code": exc.status_code,
            "error_type": "http",
        },
    )
    response = _error_response(exc.status_code, detail)
    if exc.headers:
        response.headers.update(exc.headers)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Скрывает детали внутренних ошибок от клиента."""
    logger.exception(
        "unhandled_exception",
        extra={
            "event": "unhandled_exception",
            "path": request.url.path,
            "error_type": "internal",
        },
        exc_info=exc,
    )
    return _error_response(500, "Внутренняя ошибка сервера")
