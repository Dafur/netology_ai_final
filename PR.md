# PR: FastAPI-сервис структурированной справки о лекарствах на YandexGPT

## Цель изменений

Реализовать сервис, который по названию лекарственного препарата получает у
YandexGPT структурированное описание и возвращает его через единый эндпоинт
`POST /process`.

Задачи, которые закрывает PR:

- дать клиенту стабильный контракт ответа (разделы справки, Markdown-текст,
  дисклеймер) независимо от «сырого» формата ответа модели;
- защитить сервис от типичных сбоев LLM: таймаутов, сетевых ошибок,
  некорректного или неполного ответа;
- снизить число обращений к платной модели за счёт кэширования;
- обеспечить наблюдаемость: структурированные логи, `request_id`,
  классификация ошибок, ротация лог-файла;
- покрыть ключевые сценарии тестами и автоматической проверкой в CI.

## Архитектура решения

Приложение построено по слоям с явным разделением ответственности: HTTP-слой
не знает деталей работы с моделью, а клиент LLM не знает про HTTP-контракт API.

```
Клиент
  │  POST /process { "drug_name": "..." }
  ▼
api/process.py ──────────────► services/process_service.py
  (валидация схемы,               (оркестрация пайплайна)
   DI клиента и кэша)                    │
                                         ├─► llm/drug_description.py  (промпты, валидация названия)
                                         ├─► cache/                   (ключ, TTL, LRU)
                                         ├─► llm/yandex_gpt.py        (HTTP-клиент YandexGPT)
                                         └─► services/drug_response.py (проверка, очистка, формат)
  ◄──────────────────────────────────────┘
  ProcessResponse (success | error)
```

Пайплайн обработки запроса:

1. **Валидация и нормализация** названия препарата (Pydantic-схема и
   доменная проверка: не пусто, не из пробелов, не длиннее 100 символов).
2. **Построение промптов** (system + user) и **ключа кэша**
   `SHA-256(system_prompt + user_prompt + параметры модели)`.
3. **Проверка кэша**: при попадании возвращается готовое описание с
   `cached: true`, при промахе — вызов модели.
4. **Вызов LLM** с раздельными таймаутами; сетевые ошибки и таймауты
   преобразуются в типизированные исключения.
5. **Fallback**: при недоступности модели (если включён) возвращается
   безопасный ответ с HTTP 200 и `fallback: true`.
6. **Пост-обработка**: разбор ответа по обязательным разделам, проверка
   полноты, очистка от Markdown/HTML/служебных символов, сборка единого
   Markdown-текста и дисклеймера.
7. **Сохранение успешного результата в кэш** с TTL. Fallback-ответы не
   кэшируются.

Сквозные механизмы:

- **Единая модель ошибок** — иерархия `AppError` с HTTP-статусом и типом,
  обработчики FastAPI приводят любые исключения к формату
  `{ "status": "error", "error": "..." }`.
- **Наблюдаемость** — middleware присваивает `request_id` (из `X-Request-ID`
  или генерирует), прокидывает его через `contextvars` во все логи и
  возвращает в заголовке ответа.
- **Конфигурация из окружения** — `Settings` читает переменные с
  безопасными значениями по умолчанию и кэшируется.

## Основные реализованные компоненты

### HTTP-слой

- [`api/main.py`](api/main.py) — создание FastAPI-приложения, `lifespan`
  (настройка логирования, освобождение кэша), middleware логирования
  запросов и `request_id`, обработчики `RequestValidationError`,
  `AppError`, `HTTPException` и непредвиденных исключений.
- [`api/process.py`](api/process.py) — роутер `POST /process`, фабрики
  зависимостей клиента LLM и кэша (переопределяются в тестах), описание
  кодов ответов в OpenAPI.
- [`api/schemas.py`](api/schemas.py) — `ProcessRequest` (валидация и
  нормализация `drug_name`), `DrugSections`, `ProcessResponse` с фабриками
  `success()` / `failure()`.

### Сервисный слой

- [`services/process_service.py`](services/process_service.py) — оркестрация
  пайплайна: кэш → LLM → пост-обработка → кэш; логирование этапов,
  обработка таймаутов/сетевых ошибок и fallback.
- [`services/drug_response.py`](services/drug_response.py) — доменная модель
  `DrugDescription`, разбор ответа на разделы, валидация полноты, очистка
  текста, сборка Markdown и fallback-описания.

### Работа с LLM

- [`llm/yandex_gpt.py`](llm/yandex_gpt.py) — асинхронный клиент YandexGPT
  Completion API: авторизация по API-ключу или IAM-токену, раздельные
  таймауты (connect/read/write/pool), классификация ошибок
  (`LLMTimeoutError`, `LLMNetworkError`, `LLMResponseError`), извлечение
  текста ответа.
- [`llm/drug_description.py`](llm/drug_description.py) — system- и
  user-промпты, перечень и порядок обязательных разделов, валидация
  названия препарата.

### Кэширование

- [`cache/base.py`](cache/base.py) — протокол `SupportsCache`, `NullCache`,
  детерминированный `build_cache_key` с версией схемы.
- [`cache/memory.py`](cache/memory.py) — `InMemoryLLMCache`: TTL, удаление
  устаревших записей и LRU-вытеснение при превышении `CACHE_MAX_ENTRIES`.
- [`cache/factory.py`](cache/factory.py) — выбор бэкенда по настройкам
  (`InMemoryLLMCache` или `NullCache`) и singleton-зависимость для FastAPI.

### Конфигурация, ошибки и логирование

- [`config/settings.py`](config/settings.py) — настройки YandexGPT,
  таймаутов, fallback, кэша и логирования из переменных окружения.
- [`errors.py`](errors.py) — иерархия ошибок приложения
  (validation/network/timeout/model/internal) с HTTP-статусами.
- [`config/logging.py`](config/logging.py) — JSON- и человекочитаемый
  форматтеры, контекст `request_id`, вывод в stdout и файл с ротацией.

### Тесты и инфраструктура

- `tests/` — тесты эндпоинта, пост-обработки, кэша (TTL, LRU, выбор
  бэкенда), клиента YandexGPT, ошибок и логирования.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — CI на Python 3.11:
  установка зависимостей, проверка импортов, `ruff check` и `pytest`.
- [`pyproject.toml`](pyproject.toml) — конфигурация Ruff (pycodestyle,
  pyflakes, isort, bugbear, pyupgrade, blind-except, comprehensions,
  simplify).
- [`README.md`](README.md) и [`.env.example`](.env.example) — описание
  запуска, настроек, контракта API и кодов ответов.

## Контракт API

`POST /process` → `200` при успехе или fallback; `422` — ошибка валидации;
`502` — некорректный ответ модели; `503` — модель недоступна (fallback
отключён); `504` — таймаут (fallback отключён); `500` — внутренняя ошибка.

Успешный ответ содержит `status`, `drug_name`, `sections` (7 разделов),
`formatted` (очищенный Markdown), `disclaimer`, `cached` и `fallback`.

## Как проверить

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
uvicorn api.main:app --reload --env-file .env
```

Swagger доступен по адресу `http://127.0.0.1:8000/docs`.

## Ограничения и заметки

- Кэш хранится в памяти процесса: данные не разделяются между воркерами и
  теряются при перезапуске. Для горизонтального масштабирования потребуется
  внешний бэкенд, реализующий `SupportsCache`.
- Fallback-ответы намеренно не кэшируются, чтобы после восстановления
  модели клиент получил актуальные данные.
- Для работы с реальным YandexGPT нужны `YANDEX_GPT_FOLDER_ID` и
  `YANDEX_GPT_API_KEY` либо `YANDEX_GPT_IAM_TOKEN`.
