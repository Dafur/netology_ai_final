"""Тесты структурированного логирования."""

from __future__ import annotations

import json
import logging

from config.logging import (
    HumanFormatter,
    JsonFormatter,
    RequestIdFilter,
    RequestIdLogRecord,
    configure_logging,
    reset_request_id,
    set_request_id,
)


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
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    noisy_levels = {
        name: logging.getLogger(name).level
        for name in ("httpx", "httpcore", "asyncio", "uvicorn.access")
    }
    try:
        configure_logging("DEBUG", json_logs=True)
        configure_logging("INFO", json_logs=False)

        own_handlers = [
            handler
            for handler in root.handlers
            if getattr(handler, "_structured_logging_handler", False)
        ]

        assert len(own_handlers) == 1
        assert root.level == logging.INFO
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
        for name, level in noisy_levels.items():
            logging.getLogger(name).setLevel(level)
