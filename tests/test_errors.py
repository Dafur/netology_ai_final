"""Тесты иерархии ошибок и корректных HTTP-кодов."""

from __future__ import annotations

from errors import (
    AppError,
    InternalAppError,
    ModelAppError,
    NetworkAppError,
    TimeoutAppError,
    ValidationAppError,
)
from llm.drug_description import InvalidDrugNameError
from llm.yandex_gpt import (
    LLMError,
    LLMNetworkError,
    LLMResponseError,
    LLMTimeoutError,
)
from services.drug_response import InvalidResponseError


def test_validation_error_status_code() -> None:
    assert InvalidDrugNameError("bad").status_code == 422
    assert InvalidDrugNameError("bad").error_type == "validation"


def test_network_error_status_codes() -> None:
    assert LLMNetworkError("down").status_code == 503
    assert LLMNetworkError("down").error_type == "network"
    assert LLMTimeoutError("slow").status_code == 504
    assert LLMTimeoutError("slow").error_type == "timeout"


def test_model_error_status_code() -> None:
    assert LLMResponseError("bad response").status_code == 502
    assert LLMResponseError("bad response").error_type == "model"
    assert InvalidResponseError("bad format").status_code == 502


def test_internal_error_status_code() -> None:
    assert InternalAppError("oops").status_code == 500
    assert InternalAppError("oops").error_type == "internal"


def test_error_hierarchy() -> None:
    assert issubclass(ValidationAppError, AppError)
    assert issubclass(NetworkAppError, AppError)
    assert issubclass(TimeoutAppError, NetworkAppError)
    assert issubclass(ModelAppError, AppError)
    assert issubclass(InternalAppError, AppError)
    assert issubclass(LLMError, AppError)
    assert issubclass(InvalidResponseError, LLMResponseError)
    assert issubclass(InvalidDrugNameError, ValueError)


def test_public_message_falls_back_to_default() -> None:
    error = InternalAppError()

    assert error.public_message == "Внутренняя ошибка сервера"
