"""Pydantic-схемы запросов и ответов API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm.drug_description import MAX_DRUG_NAME_LENGTH
from services.drug_response import DrugDescription


class ProcessRequest(BaseModel):
    """Запрос на получение описания лекарства."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"drug_name": "Парацетамол"}}
    )

    drug_name: str = Field(
        ...,
        min_length=1,
        max_length=MAX_DRUG_NAME_LENGTH,
        description=(
            "Название лекарственного препарата. Не должно быть пустым, "
            f"максимум {MAX_DRUG_NAME_LENGTH} символов."
        ),
        examples=["Парацетамол"],
    )

    @field_validator("drug_name")
    @classmethod
    def drug_name_must_not_be_blank(cls, value: str) -> str:
        """Запрещает строки, состоящие только из пробельных символов."""
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError(
                "drug_name не должен быть пустым или состоять из пробелов"
            )
        return cleaned


class DrugSections(BaseModel):
    """Разделы описания лекарства в ожидаемом формате."""

    indications: str = Field(..., description="Для чего применяется")
    release_forms: str = Field(..., description="Формы выпуска")
    dosages: str = Field(..., description="Дозировки")
    administration: str = Field(..., description="Как принимать")
    side_effects: str = Field(..., description="Побочные эффекты")
    contraindications: str = Field(..., description="Противопоказания")
    price: str = Field(..., description="Средняя цена в Москве")


class ProcessResponse(BaseModel):
    """Унифицированный ответ API: статус и либо описание, либо ошибка."""

    status: Literal["success", "error"] = Field(
        ..., description="Статус обработки запроса"
    )
    drug_name: str | None = Field(
        default=None, description="Название препарата при status='success'"
    )
    sections: DrugSections | None = Field(
        default=None, description="Разделы описания при status='success'"
    )
    formatted: str | None = Field(
        default=None,
        description="Очищенное Markdown-описание при status='success'",
    )
    disclaimer: str | None = Field(
        default=None,
        description="Предупреждение о справочном характере информации",
    )
    cached: bool | None = Field(
        default=None,
        description="True, если ответ получен из кэша",
    )
    fallback: bool | None = Field(
        default=None,
        description=(
            "True, если модель была недоступна и возвращён "
            "безопасный fallback-ответ"
        ),
    )
    error: str | None = Field(
        default=None, description="Описание ошибки при status='error'"
    )

    @classmethod
    def success(cls, description: DrugDescription) -> ProcessResponse:
        """Создаёт успешный ответ из обработанного описания лекарства."""
        return cls(
            status="success",
            drug_name=description.drug_name,
            sections=DrugSections(**description.sections),
            formatted=description.formatted,
            disclaimer=description.disclaimer,
            cached=description.cached,
            fallback=description.fallback,
        )

    @classmethod
    def failure(cls, error: str) -> ProcessResponse:
        """Создаёт ответ с ошибкой."""
        return cls(status="error", error=error)
