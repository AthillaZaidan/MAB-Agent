from __future__ import annotations

import json
from typing import Any

from modelwatch.coerce import as_float
from modelwatch.extractor import parse_json_object
from modelwatch.http import post_json
from modelwatch.models import JudgeDecision, RawItem


class OllamaJudge:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def __call__(self, item: RawItem) -> JudgeDecision:
        response = post_json(
            f"{self.base_url}/api/generate",
            {
                "model": self.model,
                "prompt": build_judge_prompt(item),
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
        )
        return decision_from_payload(parse_json_object(response["response"]))


def decision_from_payload(payload: dict[str, Any]) -> JudgeDecision:
    keep = payload.get("keep", False)
    if isinstance(keep, str):
        keep = keep.strip().lower() in {"true", "yes", "keep", "benchmark-worthy"}
    reason = str(payload.get("reason") or payload.get("reject_reason") or "No benchmark-worthy release signal")
    return JudgeDecision(keep=bool(keep), reason=reason, confidence=as_float(payload.get("confidence")))


def build_judge_prompt(item: RawItem) -> str:
    return f"""
You are filtering model release candidates for a benchmark team.
Return JSON only with keys: keep, reason, confidence.

Keep benchmark-worthy model releases or availability updates:
- foundation/base/instruct/reasoning/VL/omni models
- official provider catalog models
- notable open-weight models from labs or organizations
- models likely comparable to GPT, Claude, Gemini, Grok, Qwen, DeepSeek, Llama, Gemma, Mistral, Kimi, GLM, Nemotron, SEA-LION, Falcon, Phi, MiniMax

Reject:
- personal LoRA/adapters
- random fine-tunes with no adoption signal
- test uploads
- quant-only repacks unless they are official/provider releases
- datasets, demos, or unrelated repos

Source URL: {item.source_url}
Title: {item.title}
Provider/author: {item.author_or_provider}
Text:
{item.raw_text[:4000]}
""".strip()
