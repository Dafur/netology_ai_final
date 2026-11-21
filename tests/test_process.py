"""Тесты эндпоинта POST /process."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.process import get_cache, get_llm_client
from cache.memory import InMemoryLLMCache
from llm.yandex_gpt import LLMError, LLMNetworkError, LLMTimeoutError
from services.drug_response import DISCLAIMER, FALLBACK_SECTION_TEXT

VALID_ANSWER = """\
## Для чего применяется
Жаропонижающее и обезболивающее средство.

## Формы выпуска
Таблетки, сироп, ректальные свечи.

## Дозировки
Взрослым — 500 мг до 4 раз в сутки.

## Как принимать
После еды, запивая водой.

## Побочные эффекты
Редко — тошнота, аллергические реакции.

## Противопоказания
Повышенная чувствительность к компонентам.

## Средняя цена в Москве
100–200 ₽ за упаковку; цена зависит от аптеки и дозировки.

Информация носит справочный характер и не заменяет консультацию врача.
"""


class FakeLLMClient:
    """Подменный клиент LLM для тестов."""

    def __init__(
        self,
        answer: str = VALID_ANSWER,
        error: Exception | None = None,
    ) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def generate_with_prompts(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.error is not None:
            raise self.error
        return self.answer


@pytest.fixture()
def cache() -> InMemoryLLMCache:
    """Свежий in-memory кэш для каждого теста."""
    return InMemoryLLMCache(ttl=600)


@pytest.fixture()
def client(cache: InMemoryLLMCache):
    """TestClient без обращения к реальному YandexGPT и Redis."""
    fake = FakeLLMClient()
    app.dependency_overrides[get_llm_client] = lambda: fake
    app.dependency_overrides[get_cache] = lambda: cache
    with TestClient(app) as test_client:
        test_client.fake_llm = fake  # type: ignore[attr-defined]
        test_client.cache = cache  # type: ignore[attr-defined]
        yield test_client
    app.dependency_overrides.clear()


def test_process_success(client: TestClient) -> None:
    response = client.post("/process", json={"drug_name": "Парацетамол"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["drug_name"] == "Парацетамол"
    assert body["sections"] == {
        "indications": "Жаропонижающее и обезболивающее средство.",
        "release_forms": "Таблетки, сироп, ректальные свечи.",
        "dosages": "Взрослым — 500 мг до 4 раз в сутки.",
        "administration": "После еды, запивая водой.",
        "side_effects": "Редко — тошнота, аллергические реакции.",
        "contraindications": "Повышенная чувствительность к компонентам.",
        "price": (
            "100–200 ₽ за упаковку; цена зависит от аптеки и дозировки."
        ),
    }
    assert body["formatted"].startswith("## Для чего применяется")
    assert body["formatted"].endswith(DISCLAIMER)
    assert body["disclaimer"] == DISCLAIMER
    assert body["cached"] is False
    assert body["fallback"] is False
    assert "error" not in body
    assert response.headers["X-Request-ID"]


def test_process_passes_prompts_and_normalizes_name(client: TestClient) -> None:
    response = client.post("/process", json={"drug_name": "  Пара\nцетамол  "})

    assert response.status_code == 200
    system_prompt, user_prompt = client.fake_llm.calls[0]  # type: ignore[attr-defined]
    assert "фармацевтический" in system_prompt
    assert "«Пара цетамол»" in user_prompt
    assert response.json()["drug_name"] == "Пара цетамол"


def test_process_cleans_model_answer(client: TestClient) -> None:
    client.fake_llm.answer = (  # type: ignore[attr-defined]
        "```markdown\n"
        "## **Для чего применяется**\n"
        "  * **Жаропонижающее**   средство  \n"
        "\n"
        "## Формы выпуска\n"
        "- Таблетки\n"
        "\n"
        "## Дозировки\n"
        "500 мг\n"
        "\n"
        "## Как принимать\n"
        "После еды\n"
        "\n"
        "## Побочные эффекты\n"
        "Тошнота\n"
        "\n"
        "## Противопоказания\n"
        "Аллергия\n"
        "\n"
        "## Средняя цена в Москве\n"
        "100 ₽\n"
        "```\n"
    )

    response = client.post("/process", json={"drug_name": "Парацетамол"})

    assert response.status_code == 200
    sections = response.json()["sections"]
    assert sections["indications"] == "- Жаропонижающее средство"
    assert sections["release_forms"] == "- Таблетки"
    assert "**" not in response.json()["formatted"]
    assert "```" not in response.json()["formatted"]


@pytest.mark.parametrize(
    "payload",
    [
        {"drug_name": ""},
        {"drug_name": "     "},
        {"drug_name": "а" * 101},
        {},
    ],
)
def test_process_validation_error(
    client: TestClient,
    payload: dict[str, str],
) -> None:
    response = client.post("/process", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "error"
    assert body["error"]


def test_process_max_name_length(client: TestClient) -> None:
    response = client.post("/process", json={"drug_name": "а" * 100})

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_process_llm_error(client: TestClient) -> None:
    client.fake_llm.error = LLMError("YandexGPT недоступен")  # type: ignore[attr-defined]

    response = client.post("/process", json={"drug_name": "Парацетамол"})

    assert response.status_code == 503
    assert response.json() == {
        "status": "error",
        "error": "YandexGPT недоступен",
    }


def test_process_llm_timeout_returns_fallback(client: TestClient) -> None:
    client.fake_llm.error = LLMTimeoutError(  # type: ignore[attr-defined]
        "Превышено время ожидания ответа YandexGPT"
    )

    response = client.post("/process", json={"drug_name": "Парацетамол"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["fallback"] is True
    assert body["cached"] is False
    assert body["sections"]["indications"] == FALLBACK_SECTION_TEXT
    assert body["formatted"].endswith(DISCLAIMER)


def test_process_network_error_returns_fallback(client: TestClient) -> None:
    client.fake_llm.error = LLMNetworkError("нет соединения")  # type: ignore[attr-defined]

    response = client.post("/process", json={"drug_name": "Парацетамол"})

    assert response.status_code == 200
    assert response.json()["fallback"] is True


def test_process_internal_error_returns_500(client: TestClient) -> None:
    client.fake_llm.error = RuntimeError("unexpected")  # type: ignore[attr-defined]

    response = client.post("/process", json={"drug_name": "Парацетамол"})

    assert response.status_code == 500
    body = response.json()
    assert body["status"] == "error"
    assert body["error"] == "Внутренняя ошибка при обращении к языковой модели"


def test_process_invalid_model_answer(client: TestClient) -> None:
    client.fake_llm.answer = "Просто текст без разделов"  # type: ignore[attr-defined]

    response = client.post("/process", json={"drug_name": "Парацетамол"})

    assert response.status_code == 502
    body = response.json()
    assert body["status"] == "error"
    assert "разделов" in body["error"]


def test_process_incomplete_model_answer(client: TestClient) -> None:
    client.fake_llm.answer = (  # type: ignore[attr-defined]
        "## Для чего применяется\nЖаропонижающее.\n"
        "## Формы выпуска\nТаблетки.\n"
    )

    response = client.post("/process", json={"drug_name": "Парацетамол"})

    assert response.status_code == 502
    body = response.json()
    assert body["status"] == "error"
    assert "отсутствуют разделы" in body["error"]


def test_process_uses_cache_on_repeated_request(client: TestClient) -> None:
    first = client.post("/process", json={"drug_name": "Парацетамол"})
    second = client.post("/process", json={"drug_name": "Парацетамол"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert len(client.fake_llm.calls) == 1  # type: ignore[attr-defined]


def test_process_cache_is_per_drug(client: TestClient) -> None:
    client.post("/process", json={"drug_name": "Парацетамол"})
    client.post("/process", json={"drug_name": "Ибупрофен"})

    assert len(client.fake_llm.calls) == 2  # type: ignore[attr-defined]


def test_process_fallback_is_not_cached(client: TestClient) -> None:
    client.fake_llm.error = LLMNetworkError("нет соединения")  # type: ignore[attr-defined]
    first = client.post("/process", json={"drug_name": "Парацетамол"})

    client.fake_llm.error = None  # type: ignore[attr-defined]
    second = client.post("/process", json={"drug_name": "Парацетамол"})

    assert first.json()["fallback"] is True
    assert second.json()["fallback"] is False
    assert second.json()["cached"] is False
    assert len(client.fake_llm.calls) == 2  # type: ignore[attr-defined]
