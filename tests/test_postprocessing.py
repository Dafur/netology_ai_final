"""Тесты пост-обработки ответа LLM."""

from __future__ import annotations

import pytest

from llm.drug_description import SECTION_TITLES
from services.drug_response import (
    DISCLAIMER,
    MAX_SECTION_LENGTH,
    InvalidResponseError,
    build_drug_description,
    clean_section,
    format_description,
    parse_sections,
    validate_sections,
)

VALID_ANSWER = """\
## Для чего применяется
Жаропонижающее и обезболивающее средство.

## Формы выпуска
Таблетки и сироп.

## Дозировки
500 мг до 4 раз в сутки.

## Как принимать
После еды.

## Побочные эффекты
Тошнота.

## Противопоказания
Аллергия.

## Средняя цена в Москве
100–200 ₽.

Информация носит справочный характер и не заменяет консультацию врача.
"""


def test_parse_sections_returns_all_sections_in_order() -> None:
    sections = parse_sections(VALID_ANSWER)

    assert list(sections) == list(SECTION_TITLES)
    assert sections["indications"] == "Жаропонижающее и обезболивающее средство."
    assert sections["price"] == "100–200 ₽."


def test_parse_sections_supports_bold_and_colon_headings() -> None:
    raw = """\
**Для чего применяется**
Жаропонижающее.

Формы выпуска:
Таблетки.

Дозировки:
500 мг.

Как принимать:
После еды.

Побочные эффекты:
Тошнота.

Противопоказания:
Аллергия.

Средняя цена в Москве:
100 ₽.
"""

    sections = parse_sections(raw)

    assert sections["indications"] == "Жаропонижающее."
    assert sections["release_forms"] == "Таблетки."
    assert sections["price"] == "100 ₽."


def test_parse_sections_removes_disclaimer_from_sections() -> None:
    sections = parse_sections(VALID_ANSWER)

    assert DISCLAIMER not in sections["price"]


def test_parse_sections_rejects_answer_without_headings() -> None:
    with pytest.raises(InvalidResponseError, match="без обязательных разделов"):
        parse_sections("Просто текст без структуры")


def test_parse_sections_rejects_non_string() -> None:
    with pytest.raises(InvalidResponseError, match="не в виде строки"):
        parse_sections(None)  # type: ignore[arg-type]


def test_clean_section_removes_extra_symbols() -> None:
    raw = (
        "  * **Жаропонижающее**   средство  \n"
        "<b>Текст</b> [ссылка](https://example.com)\u200b\n"
        "```\n"
    )

    cleaned = clean_section(raw)

    assert cleaned == "- Жаропонижающее средство\nТекст ссылка"


def test_clean_section_limits_too_long_text() -> None:
    cleaned = clean_section("слово " * 1000)

    assert len(cleaned) <= MAX_SECTION_LENGTH + 1
    assert cleaned.endswith("…")


def test_validate_sections_rejects_missing_sections() -> None:
    with pytest.raises(InvalidResponseError, match="отсутствуют разделы"):
        validate_sections({"indications": "Жаропонижающее."})


def test_validate_sections_rejects_too_short_sections() -> None:
    sections = dict.fromkeys(SECTION_TITLES, "Текст")
    sections["dosages"] = "1"

    with pytest.raises(InvalidResponseError, match="слишком короткие разделы"):
        validate_sections(sections)


def test_build_drug_description_returns_expected_format() -> None:
    description = build_drug_description("Парацетамол", VALID_ANSWER)

    assert description.drug_name == "Парацетамол"
    assert list(description.sections) == list(SECTION_TITLES)
    assert description.disclaimer == DISCLAIMER
    assert description.formatted.startswith("## Для чего применяется")
    assert description.formatted.endswith(DISCLAIMER)
    assert "## Формы выпуска" in description.formatted


def test_format_description_contains_all_headings() -> None:
    sections = {key: f"Текст {key}" for key in SECTION_TITLES}

    formatted = format_description(sections)

    for title in SECTION_TITLES.values():
        assert f"## {title}" in formatted


def test_build_drug_description_rejects_incomplete_answer() -> None:
    raw = "## Для чего применяется\nЖаропонижающее.\n"

    with pytest.raises(InvalidResponseError, match="отсутствуют разделы"):
        build_drug_description("Парацетамол", raw)
