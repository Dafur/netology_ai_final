"""Иерархия ошибок приложения.

Модуль разделяет ошибки по типам, чтобы HTTP-слой мог выбрать корректный
код ответа, а логирование — единообразно классифицировать инциденты:

* :class:`ValidationAppError` — некорректные входные данные (HTTP 422);
* :class:`NetworkAppError` — сетевая ошибка или недоступность внешнего
  сервиса (HTTP 503);
* :class:`TimeoutAppError` — истёк таймаут внешнего вызова (HTTP 504);
* :class:`ModelAppError` — внешняя модель вернула некорректный ответ
  (HTTP 502);
* :class:`InternalAppError` — непредвиденная внутренняя ошибка (HTTP 500).
"""

from __future__ import annotations


class AppError(Exception):
    """Базовая ошибка приложения с HTTP-статусом и типом для логов."""

    status_code: int = 500
    error_type: str = "internal"
    default_message: str = "Внутренняя ошибка сервера"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)

    @property
    def public_message(self) -> str:
        """Сообщение, безопасное для возврата клиенту."""
        return self.message


class ValidationAppError(AppError):
    """Некорректные входные данные (HTTP 422)."""

    status_code = 422
    error_type = "validation"
    default_message = "Некорректные входные данные"


class NetworkAppError(AppError):
    """Сетевая ошибка или недоступность внешнего сервиса (HTTP 503)."""

    status_code = 503
    error_type = "network"
    default_message = "Внешний сервис временно недоступен"


class TimeoutAppError(NetworkAppError):
    """Истёк таймаут внешнего вызова (HTTP 504)."""

    status_code = 504
    error_type = "timeout"
    default_message = "Превышено время ожидания внешнего сервиса"


class ModelAppError(AppError):
    """Внешняя модель вернула некорректный ответ (HTTP 502)."""

    status_code = 502
    error_type = "model"
    default_message = "Модель вернула некорректный ответ"


class InternalAppError(AppError):
    """Непредвиденная внутренняя ошибка (HTTP 500)."""

    status_code = 500
    error_type = "internal"
    default_message = "Внутренняя ошибка сервера"


__all__ = [
    "AppError",
    "InternalAppError",
    "ModelAppError",
    "NetworkAppError",
    "TimeoutAppError",
    "ValidationAppError",
]
