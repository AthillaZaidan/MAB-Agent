from __future__ import annotations

from dataclasses import replace

from modelwatch.models import Candidate


IMPORTANT_PROVIDERS = {"openai", "anthropic", "google", "meta", "qwen", "deepseek", "mistral"}


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
    provider = 100 if (candidate.provider or "").lower() in IMPORTANT_PROVIDERS else 55
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
