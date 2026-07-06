from __future__ import annotations

import re
from dataclasses import replace

from modelwatch.models import Candidate


IMPORTANT_PROVIDERS = {
    "ai21",
    "aisingapore",
    "allenai",
    "anthropic",
    "baidu",
    "bytedance",
    "deepseek",
    "deepseek-ai",
    "google",
    "gotocompany",
    "huggingfacetb",
    "liquidai",
    "meta",
    "meta-llama",
    "microsoft",
    "minimax",
    "minimaxai",
    "mistral",
    "mistralai",
    "moonshotai",
    "nvidia",
    "opengvlab",
    "openai",
    "qwen",
    "tiiuae",
    "x-ai",
    "xiaomi",
    "z-ai",
    "zai-org",
}
MODEL_FAMILY_RE = re.compile(
    r"gpt|claude|gemini|grok|deepseek|qwen|glm|gemma|llama|mistral|kimi|minimax|"
    r"nemotron|nova|doubao|seed|falcon|phi|ernie|mimo|lfm|internvl|olmo|sea-lion|sahabat",
    re.IGNORECASE,
)
DERIVATIVE_RE = re.compile(
    r"lora|adapter|gguf|awq|gptq|quant|quantized|mlx|qat|sft|dpo|rlhf|finetune|fine-tune|"
    r"uncensored|abliterated|merge|merged|4bit|3bit|2bit|8bit|w\d+a\d+|debugger|custom",
    re.IGNORECASE,
)


def action_for_score(score: float) -> str:
    if score >= 80:
        return "Benchmark now"
    if score >= 60:
        return "Watchlist"
    if score >= 40:
        return "Store only"
    return "Ignore unless manually reviewed"


def score_candidate(candidate: Candidate) -> Candidate:
    novelty = 100 if candidate.release_type == "new_model" else 70 if candidate.release_type != "irrelevant" else 0
    provider = 100 if is_important_provider(candidate.provider) else 55
    availability = 100 if candidate.access_type in {"open_weight", "api_only"} else 35
    benchmark = 100 if candidate.benchmark_claims or candidate.claimed_strengths else 40
    modality = 100 if {"text", "vision"} & set(candidate.modality) else 45
    adoption = min(100, 40 + 20 * len(candidate.evidence_urls))
    score = (
        0.25 * novelty
        + 0.20 * provider
        + 0.20 * availability
        + 0.15 * benchmark
        + 0.10 * modality
        + 0.10 * adoption
    )
    score = round(score, 1)
    return replace(candidate, benchmark_relevance_score=score, recommended_action=action_for_score(score))


def rejection_reason(candidate: Candidate) -> str | None:
    if candidate.release_type in {None, "irrelevant"}:
        return "irrelevant release type"
    text = " ".join(
        [
            candidate.canonical_model_name,
            candidate.provider or "",
            " ".join(candidate.claimed_strengths),
            " ".join(candidate.evidence_urls),
        ]
    )
    hf_owner = huggingface_owner(candidate)
    if hf_owner and not is_important_provider(hf_owner) and DERIVATIVE_RE.search(text):
        return f"derivative or repack from untrusted Hugging Face owner: {hf_owner}"
    if not MODEL_FAMILY_RE.search(text):
        return "no frontier model family signal"
    return None


def is_important_provider(provider: str | None) -> bool:
    if not provider:
        return False
    return provider.lower().replace("_", "-") in IMPORTANT_PROVIDERS


def huggingface_owner(candidate: Candidate) -> str | None:
    for url in candidate.evidence_urls:
        match = re.search(r"https://huggingface\.co/([^/\s]+)", url)
        if match:
            return match.group(1)
    return None
