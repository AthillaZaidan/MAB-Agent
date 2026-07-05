from __future__ import annotations

import re


def model_key(provider: str | None, name: str, parameter_size: str | None = None) -> str:
    cleaned = str(name).lower()
    cleaned = cleaned.split("/")[-1]
    cleaned = re.sub(r"\binstruct\b", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", "", cleaned)
    size = re.sub(r"[^a-z0-9]+", "", str(parameter_size or "").lower())
    prefix = re.sub(r"[^a-z0-9]+", "", str(provider or "").lower())
    return f"{prefix}:{cleaned}:{size}"
