from __future__ import annotations

import re
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


def as_float(value: Any, default: float = 0) -> float:
    value = null_if_string_null(value)
    if value is None:
        return default
    if isinstance(value, str):
        label = value.strip().lower()
        if label in {"low", "weak"}:
            return 0.25
        if label in {"medium", "moderate"}:
            return 0.5
        if label in {"high", "strong"}:
            return 0.85
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any) -> int | None:
    value = null_if_string_null(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    match = re.search(r"\d[\d, _]*", str(value))
    return int(re.sub(r"\D", "", match.group(0))) if match else None
