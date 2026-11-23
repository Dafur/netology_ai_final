"""Тесты структурированного логирования."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from config.logging import (
    HumanFormatter,
    JsonFormatter,
    RequestIdFilter,
    RequestIdLogRecord,
    configure_logging,
    reset_request_id,
    set_request_id,
)

_NOISY_LOGGERS = ("httpx", "httpcore", "asyncio", "uvicorn.access")


def _make_record(
    message: str = "cache_hit",
    level: int = logging.INFO,
) -> RequestIdLogRecord:
    return RequestIdLogRecord(
        name="test.logger",
        level=level,
        pathname=__file__,
        lineno=10,
        msg=message,
        args=(),
        exc_info=None,
    )


@contextmanager
def _restore_root_logging() -> Iterator[logging.Logger]:
    """Сохраняет и восстанавливает состояние корневого логгера после теста."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    noisy_levels = {
        name: logging.getLogger(name).level for name in _NOISY_LOGGERS
    }
    try:
        yield root
    finally:
        for handler in root.handlers[:]:
            if getattr(handler, "_structured_logging_handler", False):
                root.removeHandler(handler)
                handler.close()
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
        for name, level in noisy_levels.items():
            logging.getLogger(name).setLevel(level)


def _file_handlers(root: logging.Logger) -> list[logging.Handler]:
    return [
        handler
        for handler in root.handlers
        if getattr(handler, "_structured_logging_file_handler", False)
    ]


def test_json_formatter_produces_structured_record() -> None:
    record = _make_record()
    record.event = "cache_hit"
    record.cache_key = "abc123"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["message"] == "cache_hit"
    assert payload["event"] == "cache_hit"
    assert payload["cache_key"] == "abc123"
    assert payload["timestamp"].endswith("Z")


def test_json_formatter_adds_request_id_from_context() -> None:
    record = _make_record()
    token = set_request_id("req-42")
    try:
        payload = json.loads(JsonFormatter().format(record))
    finally:
        reset_request_id(token)

    assert payload["request_id"] == "req-42"


def test_request_id_filter_injects_context_value() -> None:
    record = _make_record()
    token = set_request_id("req-7")
    try:
        RequestIdFilter().filter(record)
    finally:
        reset_request_id(token)

    assert record.request_id == "req-7"


def test_json_formatter_includes_exception() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _make_record(level=logging.ERROR)
        record.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: boom" in payload["exception"]


def test_human_formatter_contains_extra_fields() -> None:
    record = _make_record()
    record.event = "cache_miss"

    output = HumanFormatter().format(record)

    assert "cache_miss" in output
    assert "event=cache_miss" in output


def test_configure_logging_is_idempotent() -> None:
    with _restore_root_logging() as root:
        configure_logging("DEBUG", json_logs=True)
        configure_logging("INFO", json_logs=False)

        own_handlers = [
            handler
            for handler in root.handlers
            if getattr(handler, "_structured_logging_handler", False)
        ]

        assert len(own_handlers) == 1
        assert root.level == logging.INFO


def test_configure_logging_writes_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "app.log"

    with _restore_root_logging() as root:
        configure_logging("INFO", json_logs=True, log_file=log_file)

        logging.getLogger("test.file.logger").info(
            "file_event", extra={"event": "file_event"}
        )
        for handler in root.handlers:
            handler.flush()

        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "file_event" in content

        payload = json.loads(content.strip().splitlines()[-1])
        assert payload["event"] == "file_event"
        assert payload["message"] == "file_event"


def test_configure_logging_human_format_in_file(tmp_path: Path) -> None:
    log_file = tmp_path / "app.log"

    with _restore_root_logging():
        configure_logging("INFO", json_logs=False, log_file=log_file)

        logging.getLogger("test.file.logger").info(
            "human_file_event", extra={"event": "human_file_event"}
        )

        content = log_file.read_text(encoding="utf-8")
        assert "human_file_event" in content
        assert "event=human_file_event" in content


def test_file_handler_is_idempotent(tmp_path: Path) -> None:
    log_file = tmp_path / "app.log"

    with _restore_root_logging() as root:
        configure_logging("INFO", json_logs=True, log_file=log_file)
        configure_logging("DEBUG", json_logs=True, log_file=log_file)

        assert len(_file_handlers(root)) == 1


def test_switching_log_file_replaces_handler(tmp_path: Path) -> None:
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"

    with _restore_root_logging() as root:
        configure_logging("INFO", json_logs=True, log_file=first)
        configure_logging("INFO", json_logs=True, log_file=second)

        handlers = _file_handlers(root)
        assert len(handlers) == 1
        assert getattr(handlers[0], "baseFilename", "") == str(second)

        logging.getLogger("test.file.logger").info(
            "second_file_event", extra={"event": "second_file_event"}
        )
        for handler in root.handlers:
            handler.flush()

        assert "second_file_event" in second.read_text(encoding="utf-8")


def test_empty_log_file_disables_file_logging(tmp_path: Path) -> None:
    log_file = tmp_path / "app.log"

    with _restore_root_logging() as root:
        configure_logging("INFO", json_logs=True, log_file=log_file)
        assert len(_file_handlers(root)) == 1

        configure_logging("INFO", json_logs=True, log_file="")
        assert _file_handlers(root) == []
