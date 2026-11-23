"""Настройки приложения, читаемые из переменных окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _env_float(name: str, default: float) -> float:
    """Читает float из переменной окружения, при ошибке возвращает default."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Читает int из переменной окружения, при ошибке возвращает default."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Читает bool из переменной окружения (true/1/yes/on)."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


@dataclass(frozen=True)
class Settings:
    """Настройки интеграции с YandexGPT."""

    yandex_gpt_api_key: str = field(
        default_factory=lambda: os.getenv("YANDEX_GPT_API_KEY", "")
    )
    yandex_gpt_iam_token: str = field(
        default_factory=lambda: os.getenv("YANDEX_GPT_IAM_TOKEN", "")
    )
    yandex_gpt_folder_id: str = field(
        default_factory=lambda: os.getenv("YANDEX_GPT_FOLDER_ID", "")
    )
    yandex_gpt_model: str = field(
        default_factory=lambda: os.getenv("YANDEX_GPT_MODEL", "yandexgpt-lite")
    )
    yandex_gpt_model_version: str = field(
        default_factory=lambda: os.getenv("YANDEX_GPT_MODEL_VERSION", "latest")
    )
    yandex_gpt_api_url: str = field(
        default_factory=lambda: os.getenv(
            "YANDEX_GPT_API_URL",
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        )
    )
    yandex_gpt_temperature: float = field(
        default_factory=lambda: _env_float("YANDEX_GPT_TEMPERATURE", 0.3)
    )
    yandex_gpt_max_tokens: int = field(
        default_factory=lambda: _env_int("YANDEX_GPT_MAX_TOKENS", 1000)
    )
    request_timeout: float = field(
        default_factory=lambda: _env_float("REQUEST_TIMEOUT", 30.0)
    )
    llm_connect_timeout: float = field(
        default_factory=lambda: _env_float("LLM_CONNECT_TIMEOUT", 5.0)
    )
    llm_read_timeout: float = field(
        default_factory=lambda: _env_float("LLM_READ_TIMEOUT", 30.0)
    )
    llm_write_timeout: float = field(
        default_factory=lambda: _env_float("LLM_WRITE_TIMEOUT", 10.0)
    )
    llm_pool_timeout: float = field(
        default_factory=lambda: _env_float("LLM_POOL_TIMEOUT", 5.0)
    )
    # Fallback-ответ при недоступности модели.
    llm_fallback_enabled: bool = field(
        default_factory=lambda: _env_bool("LLM_FALLBACK_ENABLED", True)
    )
    # Кэширование результатов LLM.
    cache_enabled: bool = field(
        default_factory=lambda: _env_bool("CACHE_ENABLED", True)
    )
    cache_ttl: int = field(
        default_factory=lambda: _env_int("CACHE_TTL", 600)
    )
    # Максимальное число записей в in-memory кэше (LRU-вытеснение).
    cache_max_entries: int = field(
        default_factory=lambda: _env_int("CACHE_MAX_ENTRIES", 1000)
    )
    # Логирование.
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    log_json: bool = field(
        default_factory=lambda: _env_bool("LOG_JSON", True)
    )
    # Путь к файлу логов; пустая строка отключает запись в файл.
    log_file: str = field(
        default_factory=lambda: os.getenv("LOG_FILE", "logs/app.log")
    )
    log_file_max_bytes: int = field(
        default_factory=lambda: _env_int(
            "LOG_FILE_MAX_BYTES", 10 * 1024 * 1024
        )
    )
    log_file_backup_count: int = field(
        default_factory=lambda: _env_int("LOG_FILE_BACKUP_COUNT", 5)
    )

    @property
    def model_uri(self) -> str:
        """URI модели в формате YandexGPT API."""
        return (
            f"gpt://{self.yandex_gpt_folder_id}/"
            f"{self.yandex_gpt_model}/{self.yandex_gpt_model_version}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает кэшированный экземпляр настроек."""
    return Settings()
