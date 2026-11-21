"""Настройка структурированного логирования.

Модуль предоставляет:

* JSON-форматтер для машиночитаемых логов (структурированный формат);
* человекочитаемый форматтер для локальной разработки;
* контекст ``request_id``, автоматически добавляемый во все записи;
* функцию :func:`configure_logging` для инициализации корневого логгера.

Использование::

    from config.logging import configure_logging, set_request_id

    configure_logging(level="INFO", json_logs=True)
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# Контекст текущего HTTP-запроса. Значение живёт в пределах одной задачи
# asyncio, поэтому безопасно при конкурентной обработке запросов.
_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

# Атрибуты стандартного LogRecord, которые не нужно дублировать в extra.
_RESERVED_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime"}

_HANDLER_MARKER = "_structured_logging_handler"


class RequestIdLogRecord(logging.LogRecord):
    """``LogRecord`` с гарантированно объявленным полем ``request_id``.

    В рантайме записи создаёт стандартный :class:`logging.LogRecord`, а
    :class:`RequestIdFilter` добавляет в них атрибут ``request_id``. Этот
    подкласс нужен только для статической типизации: он позволяет обращаться
    к ``record.request_id`` напрямую, без ``getattr`` и без предупреждений
    type checker'а.
    """

    request_id: str | None = None


def set_request_id(request_id: str) -> contextvars.Token[str | None]:
    """Устанавливает ``request_id`` в контексте текущей задачи."""
    return _request_id_var.set(request_id)


def reset_request_id(token: contextvars.Token[str | None]) -> None:
    """Сбрасывает ``request_id`` к предыдущему значению."""
    _request_id_var.reset(token)


def get_request_id() -> str | None:
    """Возвращает ``request_id`` текущего контекста, если он задан."""
    return _request_id_var.get()


class RequestIdFilter(logging.Filter):
    """Добавляет ``request_id`` из контекста в каждую запись лога."""

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = getattr(record, "request_id", None) or get_request_id()
        # Атрибут выставляется всегда (возможно, в ``None``), чтобы записи,
        # прошедшие фильтр, имели предсказуемый набор полей.
        record.request_id = request_id
        return True


def _extract_extra(record: logging.LogRecord) -> dict[str, Any]:
    """Возвращает пользовательские поля записи (переданные через ``extra``)."""
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _RESERVED_ATTRS
        and key != "request_id"
        and not key.startswith("_")
    }


def _timestamp(record: logging.LogRecord) -> str:
    """Форматирует время записи в ISO 8601 UTC."""
    created = datetime.fromtimestamp(record.created, tz=UTC)
    return created.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class JsonFormatter(logging.Formatter):
    """Форматтер, сериализующий запись лога в JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = getattr(record, "request_id", None) or get_request_id()
        if request_id:
            payload["request_id"] = request_id

        payload.update(_extract_extra(record))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


class HumanFormatter(logging.Formatter):
    """Человекочитаемый форматтер с дополнительными полями."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        parts = [base]

        request_id = getattr(record, "request_id", None) or get_request_id()
        if request_id:
            parts.append(f"request_id={request_id}")

        for key, value in _extract_extra(record).items():
            parts.append(f"{key}={value}")

        if record.exc_info:
            parts.append(self.formatException(record.exc_info))

        return " ".join(parts)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Настраивает корневой логгер приложения.

    Функция идемпотентна: повторный вызов обновляет уровень и формат
    единственного собственного обработчика, не удаляя сторонние
    (например, обработчик ``pytest``/``caplog``).

    Args:
        level: уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_logs: использовать структурированный JSON-формат.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()

    handler: logging.Handler | None = None
    for existing in root.handlers:
        if getattr(existing, _HANDLER_MARKER, False):
            handler = existing
            break

    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        setattr(handler, _HANDLER_MARKER, True)
        root.addHandler(handler)

    handler.setFormatter(JsonFormatter() if json_logs else HumanFormatter())
    handler.setLevel(numeric_level)
    if not any(isinstance(item, RequestIdFilter) for item in handler.filters):
        handler.addFilter(RequestIdFilter())

    root.setLevel(numeric_level)

    # Сторонние библиотеки логируют слишком подробно на уровне INFO.
    for noisy in ("httpx", "httpcore", "asyncio", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


__all__ = [
    "HumanFormatter",
    "JsonFormatter",
    "RequestIdFilter",
    "RequestIdLogRecord",
    "configure_logging",
    "get_request_id",
    "reset_request_id",
    "set_request_id",
]
