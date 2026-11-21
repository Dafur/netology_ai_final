"""Тесты промптов и валидации названия лекарства."""

from __future__ import annotations

import pytest

from llm.drug_description import (
    DRUG_DESCRIPTION_SYSTEM_PROMPT,
    MAX_DRUG_NAME_LENGTH,
    SECTION_TITLES,
    InvalidDrugNameError,
    build_user_prompt,
    validate_drug_name,
)


def test_system_prompt_contains_all_sections() -> None:
    for title in SECTION_TITLES.values():
        assert title in DRUG_DESCRIPTION_SYSTEM_PROMPT


def test_system_prompt_contains_safety_instructions() -> None:
    lowered = DRUG_DESCRIPTION_SYSTEM_PROMPT.lower()
    assert "не назначай лечение" in lowered
    assert "не выдумывай" in lowered
    assert "справочный характер" in lowered


def test_build_user_prompt_contains_drug_name() -> None:
    prompt = build_user_prompt("  Парацетамол  ")

    assert "Парацетамол" in prompt
    assert prompt == build_user_prompt("Парацетамол")


def test_validate_drug_name_normalizes_whitespace() -> None:
    assert validate_drug_name("  Пара\n\tцетамол  ") == "Пара цетамол"


def test_validate_drug_name_rejects_blank() -> None:
    with pytest.raises(InvalidDrugNameError):
        validate_drug_name("   \n\t ")


def test_validate_drug_name_rejects_too_long() -> None:
    with pytest.raises(InvalidDrugNameError):
        validate_drug_name("а" * (MAX_DRUG_NAME_LENGTH + 1))


def test_validate_drug_name_accepts_max_length() -> None:
    value = "а" * MAX_DRUG_NAME_LENGTH

    assert validate_drug_name(value) == value


def test_validate_drug_name_rejects_non_string() -> None:
    with pytest.raises(InvalidDrugNameError):
        validate_drug_name(None)  # type: ignore[arg-type]
