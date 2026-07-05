from __future__ import annotations

import json
import re
from typing import Any

from modelwatch.http import post_json
from modelwatch.models import Candidate, RawItem


SCHEMA_DEFAULTS = {
    "model_name": None,
    "provider": None,
    "release_type": "irrelevant",
    "release_date": None,
    "modality": ["unknown"],
    "access_type": "unknown",
    "license": None,
    "parameter_size": None,
    "context_length": None,
    "claimed_strengths": [],
    "benchmark_claims": [],
    "availability": [],
    "confidence": 0,
    "evidence_urls": [],
}


class OllamaExtractor:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def __call__(self, item: RawItem) -> Candidate | None:
        prompt = build_prompt(item, strict=False)
        try:
            payload = self._extract(prompt)
        except (ValueError, KeyError, json.JSONDecodeError):
            payload = self._extract(build_prompt(item, strict=True))
        if payload.get("release_type") in {None, "irrelevant"} or not payload.get("model_name"):
            return None
        merged = SCHEMA_DEFAULTS | payload
        urls = list(dict.fromkeys([item.source_url, *merged["evidence_urls"]]))
        return Candidate(
            canonical_model_name=merged["model_name"],
            provider=merged["provider"] or item.author_or_provider,
            release_type=merged["release_type"],
            release_date=merged["release_date"],
            modality=merged["modality"] or ["unknown"],
            access_type=merged["access_type"] or "unknown",
            license=merged["license"],
            parameter_size=merged["parameter_size"],
            context_length=merged["context_length"],
            claimed_strengths=merged["claimed_strengths"] or [],
            benchmark_claims=merged["benchmark_claims"] or [],
            availability=merged["availability"] or [{"platform": item.source_name, "url": item.source_url}],
            confidence=float(merged["confidence"] or 0),
            evidence_urls=urls,
        )

    def _extract(self, prompt: str) -> dict[str, Any]:
        response = post_json(
            f"{self.base_url}/api/generate",
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
        )
        return parse_json_object(response["response"])


def build_prompt(item: RawItem, *, strict: bool) -> str:
    retry = "Return only one JSON object. No markdown. No prose." if strict else "Return valid JSON only."
    return f"""
Extract model release metadata from the source text.
Do not infer unsupported facts. Use null for unknown fields.
{retry}

Required keys:
model_name, provider, release_type, release_date, modality, access_type,
license, parameter_size, context_length, claimed_strengths, benchmark_claims,
availability, confidence, evidence_urls.

Source URL: {item.source_url}
Title: {item.title}
Text:
{item.raw_text[:6000]}
""".strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])
