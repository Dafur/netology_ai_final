# Netology AI Final — FastAPI + YandexGPT

Приложение выполняет одну функцию: по названию лекарства запрашивает у LLM
структурированное описание и возвращает его через `POST /process`.
Дополнительно реализованы обработка ошибок, fallback-ответ, структурированное
логирование и кэширование результатов в Redis.

## Структура

```
api/            HTTP-слой: приложение, роутер /process, Pydantic-схемы
cache/          кэш результатов LLM: ключ, in-memory и Redis-бэкенды
config/         настройки из окружения и настройка логирования
llm/            клиент YandexGPT, промпты и валидация названия препарата
services/       вызов LLM, пост-обработка и fallback
errors.py       иерархия ошибок приложения (валидация/сеть/модель/внутренние)
tests/          тесты эндпоинта, пост-обработки, кэша, логов и клиента
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Настройка

Скопируйте `.env.example` в `.env` (или задайте переменные окружения).

### YandexGPT

- `YANDEX_GPT_API_KEY` **или** `YANDEX_GPT_IAM_TOKEN`;
- `YANDEX_GPT_FOLDER_ID` — идентификатор каталога Yandex Cloud;
- `YANDEX_GPT_MODEL`, `YANDEX_GPT_MODEL_VERSION`, `YANDEX_GPT_API_URL`;
- `YANDEX_GPT_TEMPERATURE`, `YANDEX_GPT_MAX_TOKENS`.

### Таймауты (секунды)

`REQUEST_TIMEOUT`, `LLM_CONNECT_TIMEOUT`, `LLM_READ_TIMEOUT`,
`LLM_WRITE_TIMEOUT`, `LLM_POOL_TIMEOUT`.

### Fallback

`LLM_FALLBACK_ENABLED=true` — при сетевой ошибке или таймауте вернуть
безопасный ответ с HTTP 200. При `false` вернётся 503/504.

### Кэш и Redis

- `CACHE_ENABLED` — включить кэширование;
- `CACHE_TTL` — время жизни записи в секундах (по умолчанию `600`, 10 минут);
- `CACHE_KEY_PREFIX` — префикс ключей в Redis;
- `REDIS_URL` **или** `REDIS_HOST`/`REDIS_PORT`/`REDIS_DB`/`REDIS_PASSWORD`.

Если Redis не настроен, используется in-memory кэш (удобно для разработки и
тестов). При недоступности Redis кэш работает fail-open: запросы продолжают
обрабатываться без кэша, в логах появляется `cache_get_failed` /
`cache_set_failed`.

### Логирование

- `LOG_LEVEL` — `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`;
- `LOG_JSON=true` — структурированный JSON, `false` — человекочитаемый формат.

## Запуск

```bash
uvicorn api.main:app --reload --env-file .env
```

Swagger: http://127.0.0.1:8000/docs

## POST /process

Запрос:

```json
{ "drug_name": "Парацетамол" }
```

Правила валидации: `drug_name` — обязательная строка, не пустая (в том числе не
из одних пробелов), не длиннее 100 символов. Лишние пробелы нормализуются.

Успешный ответ (`200`):

```json
{
  "status": "success",
  "drug_name": "Парацетамол",
  "sections": {
    "indications": "Для чего применяется...",
    "release_forms": "Формы выпуска...",
    "dosages": "Дозировки...",
    "administration": "Как принимать...",
    "side_effects": "Побочные эффекты...",
    "contraindications": "Противопоказания...",
    "price": "Средняя цена в Москве..."
  },
  "formatted": "## Для чего применяется\n...\n\n## Формы выпуска\n...",
  "disclaimer": "Информация носит справочный характер и не заменяет консультацию врача.",
  "cached": false,
  "fallback": false
}
```

- `cached: true` — ответ получен из кэша;
- `fallback: true` — модель была недоступна, возвращён безопасный ответ.

Ошибка:

```json
{ "status": "error", "error": "Описание ошибки" }
```

### Коды ответов

| Код | Тип ошибки | Когда возвращается |
|-----|------------|--------------------|
| `200` | — | Успех или fallback-ответ |
| `422` | `validation` | Некорректное название препарата / тело запроса |
| `502` | `model` | Модель вернула некорректный или неполный ответ |
| `503` | `network` | Модель недоступна по сети (fallback отключён) |
| `504` | `timeout` | Таймаут обращения к модели (fallback отключён) |
| `500` | `internal` | Непредвиденная внутренняя ошибка |

Типы ошибок описаны в [`errors.py`](errors.py). Клиент LLM преобразует
сетевые сбои в `LLMNetworkError`, таймауты — в `LLMTimeoutError`, некорректный
ответ — в `LLMResponseError` ([`llm/yandex_gpt.py`](llm/yandex_gpt.py)).

## Кэширование

Пайплайн [`services/process_service.py`](services/process_service.py):

1. валидирует название и строит system/user-промпты;
2. формирует ключ кэша `SHA-256(system_prompt + user_prompt + параметры модели)`;
3. проверяет кэш и логирует `cache_hit` / `cache_miss`;
4. при промахе вызывает LLM;
5. сохраняет успешный результат в кэш с TTL и логирует `cache_store`.

Ключ учитывает модель, версию, температуру и `max_tokens`, поэтому ответы для
разных параметров не смешиваются. Fallback-ответы в кэш не сохраняются.

Реализации:

- [`cache/redis_cache.py`](cache/redis_cache.py) — Redis с TTL и методом
  `purge_expired()` для удаления «зависших» ключей без срока жизни;
- [`cache/memory.py`](cache/memory.py) — in-memory кэш с TTL и удалением
  устаревших записей;
- [`cache/factory.py`](cache/factory.py) — выбор бэкенда по настройкам.

## Логирование

[`config/logging.py`](config/logging.py) настраивает JSON-формат и контекст
`request_id`. Ключевые события:

- `request_started` / `request_finished` — HTTP-запрос, статус, длительность;
- `llm_request_started`, `llm_prompt`, `llm_response_received` — обращение к LLM;
- `llm_timeout`, `llm_network_error`, `llm_http_error` — ошибки LLM;
- `cache_hit`, `cache_miss`, `cache_store`, `cache_get_failed` — кэш;
- `application_error`, `unhandled_exception`, `validation_error` — ошибки API.

Пример JSON-записи:

```json
{
  "timestamp": "2026-11-21T13:20:49.879Z",
  "level": "INFO",
  "logger": "services.process_service",
  "message": "cache_hit",
  "request_id": "a1b2c3d4",
  "event": "cache_hit",
  "drug_name": "Парацетамол",
  "cache_key": "9f8e7d6c5b4a"
}
```

## Пост-обработка ответа LLM

Модуль [`services/drug_response.py`](services/drug_response.py):

1. **Проверяет корректность ответа** — находит все обязательные разделы,
   проверяет их наличие и непустоту. При нарушении формата выбрасывается
   `InvalidResponseError`, и API возвращает `502`.
2. **Очищает результат** — убирает Markdown-разметку (`**`, `*`, `` ` ``,
   заголовки), HTML-теги, ссылки, zero-width-символы, code fences, лишние
   пробелы и пустые строки, нормализует маркеры списков, удаляет дубли
   дисклеймера и ограничивает чрезмерно длинные разделы.
3. **Приводит к ожидаемому формату** — возвращает словарь разделов, единый
   Markdown-текст `formatted` и дисклеймер.
4. **Строит fallback-описание** при недоступности модели.

Промпты и список разделов находятся в
[`llm/drug_description.py`](llm/drug_description.py), а оркестрация вызова,
кэша и пост-обработки — в [`services/process_service.py`](services/process_service.py).

## Тесты

```bash
.venv\Scripts\python.exe -m pytest
```

## Линтинг

Проект использует [Ruff](https://docs.astral.sh/ruff/) — быстрый линтер и
форматтер для Python. Настройки находятся в [`pyproject.toml`](pyproject.toml):
включены правила `pycodestyle`, `pyflakes`, `isort`, `flake8-bugbear`,
`pyupgrade`, `flake8-blind-except`, `flake8-comprehensions` и `flake8-simplify`.

```bash
# проверка
.venv\Scripts\python.exe -m ruff check .

# автоисправление
.venv\Scripts\python.exe -m ruff check . --fix

# форматирование
.venv\Scripts\python.exe -m ruff format .
```

Ruff устанавливается вместе с dev-зависимостями (`requirements-dev.txt`).
