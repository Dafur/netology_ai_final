"""Пост-обработка ответа LLM: проверка, очистка и приведение к формату API."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from llm.drug_description import SECTION_TITLES
from llm.yandex_gpt import LLMResponseError

DISCLAIMER = (
    "Информация носит справочный характер и не заменяет консультацию врача."
)

FALLBACK_SECTION_TEXT = (
    "Информация временно недоступна: сервис языковой модели не отвечает. "
    "Повторите запрос позже."
)

MIN_SECTION_LENGTH = 3
MAX_SECTION_LENGTH = 4000


class InvalidResponseError(LLMResponseError):
    """Модель вернула ответ, не соответствующий ожидаемому формату."""


@dataclass(frozen=True)
class DrugDescription:
    """Проверенное и нормализованное описание лекарства."""

    drug_name: str
    sections: dict[str, str]
    disclaimer: str
    formatted: str
    cached: bool = False
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Сериализует описание для сохранения в кэше."""
        return {
            "drug_name": self.drug_name,
            "sections": dict(self.sections),
            "disclaimer": self.disclaimer,
            "formatted": self.formatted,
            "fallback": self.fallback,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DrugDescription:
        """Восстанавливает описание из данных кэша.

        Raises:
            ValueError: если структура данных некорректна или неполна.
        """
        if not isinstance(data, Mapping):
            raise ValueError("Данные кэша должны быть объектом")

        sections_raw = data.get("sections")
        if not isinstance(sections_raw, Mapping):
            raise ValueError("В данных кэша отсутствует словарь sections")

        sections = {str(key): str(value) for key, value in sections_raw.items()}
        missing = [key for key in SECTION_TITLES if key not in sections]
        if missing:
            raise ValueError(
                "В данных кэша отсутствуют разделы: " + ", ".join(missing)
            )

        return cls(
            drug_name=str(data["drug_name"]),
            sections={key: sections[key] for key in SECTION_TITLES},
            disclaimer=str(data.get("disclaimer") or DISCLAIMER),
            formatted=str(data["formatted"]),
            fallback=bool(data.get("fallback", False)),
        )


_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
_CODE_FENCE_LINE_RE = re.compile(
    r"^[ \t]*```[a-zA-Z0-9_-]*[ \t]*$", re.MULTILINE
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BOLD_RE = re.compile(r"(\*\*|__)(.*?)\1", re.DOTALL)
_ITALIC_RE = re.compile(r"(?<!\w)([*_])(?!\s)(.+?)(?<!\s)\1(?!\w)", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_HEADING_MARK_RE = re.compile(r"^[ \t]*#{1,6}[ \t]*", re.MULTILINE)
_BULLET_RE = re.compile(r"^[ \t]*[*•+][ \t]+", re.MULTILINE)
_HORIZONTAL_RULE_RE = re.compile(r"^[ \t]*[-=_]{3,}[ \t]*$", re.MULTILINE)
_DISCLAIMER_RE = re.compile(
    r"Информация\s+носит\s+справочный\s+характер.*?врача\.?",
    re.IGNORECASE | re.DOTALL,
)
_MULTIPLE_SPACES_RE = re.compile(r"[ \t]{2,}")
_MULTIPLE_BLANK_LINES_RE = re.compile(r"\n{3,}")
_LEADING_LABEL_RE = re.compile(
    r"^\s*(?:ответ|answer|справка)\s*:\s*", re.IGNORECASE
)


def _heading_pattern(title: str) -> re.Pattern[str]:
    """Собирает регулярное выражение для заголовка раздела.

    Поддерживаются варианты ``## Заголовок``, ``**Заголовок**``,
    ``Заголовок:`` и их комбинации.
    """
    escaped = re.escape(title)
    return re.compile(
        rf"^[ \t]*(?:#{{1,6}}[ \t]*)?(?:\d+[.)][ \t]*)?"
        rf"(?:\*\*|__)?[ \t]*{escaped}[ \t]*(?:\*\*|__)?"
        rf"[ \t]*[.:]?[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )


def _prepare_raw(raw: str) -> str:
    """Убирает служебные символы, не затрагивая структуру разделов."""
    if not isinstance(raw, str):
        raise InvalidResponseError("Модель вернула ответ не в виде строки")

    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CODE_FENCE_LINE_RE.sub("", text)
    text = text.replace("```", "")
    text = _LEADING_LABEL_RE.sub("", text)
    return text.strip()


def _limit_section(text: str) -> str:
    """Ограничивает длину раздела, не разрывая слово."""
    if len(text) <= MAX_SECTION_LENGTH:
        return text
    truncated = text[:MAX_SECTION_LENGTH]
    cut = truncated.rsplit(" ", 1)[0].rstrip(" ,;:—-")
    return f"{cut}…"


def clean_section(text: str) -> str:
    """Очищает текст раздела от Markdown-разметки и лишних символов."""
    cleaned = _ZERO_WIDTH_RE.sub("", text)
    cleaned = _CODE_FENCE_LINE_RE.sub("", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    cleaned = _LINK_RE.sub(r"\1", cleaned)
    cleaned = _BOLD_RE.sub(r"\2", cleaned)
    cleaned = _ITALIC_RE.sub(r"\2", cleaned)
    cleaned = _INLINE_CODE_RE.sub(r"\1", cleaned)
    cleaned = _HEADING_MARK_RE.sub("", cleaned)
    cleaned = _BULLET_RE.sub("- ", cleaned)
    cleaned = _HORIZONTAL_RULE_RE.sub("", cleaned)
    cleaned = _DISCLAIMER_RE.sub("", cleaned)
    cleaned = _MULTIPLE_SPACES_RE.sub(" ", cleaned)

    lines = [line.strip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = _MULTIPLE_BLANK_LINES_RE.sub("\n\n", cleaned)
    cleaned = cleaned.strip()
    return _limit_section(cleaned)


def parse_sections(raw: str) -> dict[str, str]:
    """Разбирает ответ модели на разделы по их заголовкам.

    Args:
        raw: сырой текст ответа модели.

    Returns:
        Словарь ``ключ раздела -> очищенный текст``.

    Raises:
        InvalidResponseError: если в ответе не найдено ни одного заголовка.
    """
    prepared = _prepare_raw(raw)

    matches: list[tuple[int, int, str]] = []
    for key, title in SECTION_TITLES.items():
        match = _heading_pattern(title).search(prepared)
        if match is not None:
            matches.append((match.start(), match.end(), key))

    if not matches:
        raise InvalidResponseError(
            "Модель вернула ответ без обязательных разделов"
        )

    matches.sort(key=lambda item: item[0])

    sections: dict[str, str] = {}
    for index, (_, heading_end, key) in enumerate(matches):
        next_start = (
            matches[index + 1][0]
            if index + 1 < len(matches)
            else len(prepared)
        )
        if key in sections:
            continue
        sections[key] = clean_section(prepared[heading_end:next_start])

    return sections


def validate_sections(sections: dict[str, str]) -> None:
    """Проверяет, что присутствуют все обязательные непустые разделы.

    Raises:
        InvalidResponseError: если разделы отсутствуют или слишком короткие.
    """
    missing = [
        title
        for key, title in SECTION_TITLES.items()
        if not sections.get(key)
    ]
    if missing:
        raise InvalidResponseError(
            "Модель вернула неполный ответ: отсутствуют разделы: "
            + ", ".join(missing)
        )

    too_short = [
        title
        for key, title in SECTION_TITLES.items()
        if len(sections[key]) < MIN_SECTION_LENGTH
    ]
    if too_short:
        raise InvalidResponseError(
            "Модель вернула слишком короткие разделы: "
            + ", ".join(too_short)
        )


def format_description(sections: dict[str, str]) -> str:
    """Формирует единый Markdown-текст из проверенных разделов."""
    blocks = [
        f"## {title}\n{sections[key]}"
        for key, title in SECTION_TITLES.items()
    ]
    blocks.append(DISCLAIMER)
    return "\n\n".join(blocks)


def build_fallback_description(drug_name: str) -> DrugDescription:
    """Формирует безопасный ответ при недоступности модели.

    Используется, когда LLM не ответила по сети или таймауту, чтобы клиент
    получил валидную структуру вместо ошибки 5xx.
    """
    sections = dict.fromkeys(SECTION_TITLES, FALLBACK_SECTION_TEXT)
    return DrugDescription(
        drug_name=drug_name,
        sections=sections,
        disclaimer=DISCLAIMER,
        formatted=format_description(sections),
        fallback=True,
    )


def build_drug_description(
    drug_name: str,
    raw_answer: str,
) -> DrugDescription:
    """Проверяет, очищает и форматирует ответ модели.

    Args:
        drug_name: нормализованное название препарата.
        raw_answer: сырой ответ модели.

    Returns:
        Описание лекарства в ожидаемом формате.

    Raises:
        InvalidResponseError: если ответ модели некорректен или неполон.
    """
    parsed = parse_sections(raw_answer)
    validate_sections(parsed)

    ordered_sections = {
        key: parsed[key] for key in SECTION_TITLES
    }
    return DrugDescription(
        drug_name=drug_name,
        sections=ordered_sections,
        disclaimer=DISCLAIMER,
        formatted=format_description(ordered_sections),
    )
