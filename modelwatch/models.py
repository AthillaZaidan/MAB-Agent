from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class RawItem:
    source_name: str
    source_type: str
    source_url: str
    title: str
    author_or_provider: str | None
    published_at: datetime | None
    updated_at: datetime | None
    raw_text: str
    raw_metadata: dict[str, Any]
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Candidate:
    canonical_model_name: str
    provider: str | None
    release_type: str
    release_date: str | None
    modality: list[str]
    access_type: str
    license: str | None
    parameter_size: str | None
    context_length: int | None
    claimed_strengths: list[str]
    benchmark_claims: list[dict[str, Any]]
    availability: list[dict[str, str]]
    confidence: float
    evidence_urls: list[str]
    evidence_chunks: list[dict[str, str]] = field(default_factory=list)
    benchmark_relevance_score: float = 0
    recommended_action: str = "Ignore unless manually reviewed"
    aliases: list[str] = field(default_factory=list)


@dataclass
class JudgeDecision:
    keep: bool
    reason: str
    confidence: float = 0


@dataclass
class PipelineResult:
    status: str
    source_count: int
    candidate_count: int
    digest_path: Any
    failures: dict[str, str]
