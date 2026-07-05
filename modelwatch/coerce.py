from __future__ import annotations

from typing import Any


def null_if_string_null(value: Any) -> Any:
    return None if isinstance(value, str) and value.strip().lower() in {"null", "none"} else value


def as_list(value: Any, *, scalar: bool = True) -> list:
    value = null_if_string_null(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value] if scalar else []
