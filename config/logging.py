"""Настройка структурированного логирования.

Модуль предоставляет:

* JSON-форматтер для машиночитаемых логов (структурированный формат);
* человекочитаемый форматтер для локальной разработки;
* контекст ``request_id``, автоматически добавляемый во все записи;
* запись логов в stdout и в файл с ротацией;
* функцию :func:`configure_logging` для инициализации корневого логгера.

Использование::

    from config.logging import configure_logging, set_request_id

    configure_logging(level="INFO", json_logs=True, log_file="logs/app.log")
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
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

# Маркер общий для всех обработчиков, добавленных этим модулем, и отдельные
# маркеры для консольного и файлового — чтобы обновлять их независимо.
_HANDLER_MARKER = "_structured_logging_handler"
_STREAM_HANDLER_MARKER = "_structured_logging_stream_handler"
_FILE_HANDLER_MARKER = "_structured_logging_file_handler"
_FILE_PATH_ATTR = "_structured_logging_file_path"

# Значения по умолчанию для файлового логирования.
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5


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


def _find_handler(
    root: logging.Logger, marker: str
) -> logging.Handler | None:
    """Возвращает обработчик с заданным маркером, если он уже добавлен."""
    for existing in root.handlers:
        if getattr(existing, marker, False):
            return existing
    return None


def _add_handler(
    root: logging.Logger, handler: logging.Handler, marker: str
) -> None:
    """Помечает обработчик как собственный и добавляет его в логгер."""
    setattr(handler, marker, True)
    setattr(handler, _HANDLER_MARKER, True)
    root.addHandler(handler)


def _remove_handler(root: logging.Logger, handler: logging.Handler) -> None:
    """Удаляет обработчик из логгера и закрывает его."""
    root.removeHandler(handler)
    # Закрытие не должно ломать настройку логирования.
    with contextlib.suppress(Exception):
        handler.close()


def _configure_handler(
    handler: logging.Handler,
    formatter: logging.Formatter,
    level: int,
    request_id_filter: RequestIdFilter,
) -> None:
    """Применяет к обработчику форматтер, уровень и фильтр ``request_id``."""
    handler.setFormatter(formatter)
    handler.setLevel(level)
    if not any(
        isinstance(item, RequestIdFilter) for item in handler.filters
    ):
        handler.addFilter(request_id_filter)


def _configure_file_handler(
    root: logging.Logger,
    log_file: str | os.PathLike[str] | None,
    formatter: logging.Formatter,
    level: int,
    request_id_filter: RequestIdFilter,
    max_bytes: int,
    backup_count: int,
) -> None:
    """Создаёт/обновляет файловый обработчик с ротацией.

    ``log_file=None`` или пустая строка отключают запись в файл.
    """
    existing = _find_handler(root, _FILE_HANDLER_MARKER)

    if not log_file:
        if existing is not None:
            _remove_handler(root, existing)
        return

    target = Path(os.fspath(log_file)).expanduser().absolute()
    target_key = str(target)

    if existing is not None:
        if getattr(existing, _FILE_PATH_ATTR, None) == target_key:
            _configure_handler(existing, formatter, level, request_id_filter)
            return
        # Путь изменился — пересоздаём обработчик.
        _remove_handler(root, existing)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = RotatingFileHandler(
            target,
            maxBytes=max(0, max_bytes),
            backupCount=max(0, backup_count),
            encoding="utf-8",
        )
    except OSError as exc:
        # Не роняем приложение из-за недоступного пути к лог-файлу.
        logging.getLogger(__name__).warning(
            "log_file_unavailable",
            extra={
                "event": "log_file_unavailable",
                "log_file": target_key,
                "error": str(exc),
            },
        )
        return

    setattr(handler, _FILE_PATH_ATTR, target_key)
    _add_handler(root, handler, _FILE_HANDLER_MARKER)
    _configure_handler(handler, formatter, level, request_id_filter)


def configure_logging(
    level: str = "INFO",
    json_logs: bool = True,
    log_file: str | os.PathLike[str] | None = None,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> None:
    """Настраивает корневой логгер приложения.

    Логи всегда пишутся в stdout, а при заданном ``log_file`` — ещё и в файл
    с ротацией. Функция идемпотентна: повторный вызов обновляет уровень и
    формат собственных обработчиков, не удаляя сторонние (например,
    обработчик ``pytest``/``caplog``).

    Args:
        level: уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_logs: использовать структурированный JSON-формат.
        log_file: путь к файлу логов; ``None``/пустая строка отключают файл.
        max_bytes: максимальный размер файла до ротации в байтах
            (``0`` — без ротации по размеру).
        backup_count: количество хранимых резервных копий файла.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    formatter: logging.Formatter = (
        JsonFormatter() if json_logs else HumanFormatter()
    )
    request_id_filter = RequestIdFilter()

    stream_handler = _find_handler(root, _STREAM_HANDLER_MARKER)
    if stream_handler is None:
        stream_handler = logging.StreamHandler(sys.stdout)
        _add_handler(root, stream_handler, _STREAM_HANDLER_MARKER)
    _configure_handler(
        stream_handler, formatter, numeric_level, request_id_filter
    )

    _configure_file_handler(
        root,
        log_file,
        formatter,
        numeric_level,
        request_id_filter,
        max_bytes,
        backup_count,
    )

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
