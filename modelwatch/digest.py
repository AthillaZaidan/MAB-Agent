from __future__ import annotations

from collections import defaultdict

from modelwatch.models import Candidate


GROUPS = {
    "Benchmark now": "Benchmark Now",
    "Watchlist": "Watchlist",
    "Store only": "Store Only",
    "Ignore unless manually reviewed": "Low Confidence",
}


def render_digest(
    *,
    date: str,
    window_hours: int,
    candidates: list[Candidate],
    failures: dict[str, str],
) -> str:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda c: c.benchmark_relevance_score, reverse=True):
        grouped[candidate.recommended_action].append(candidate)

    lines = [
        "# Daily Model Release Digest",
        f"Date: {date}",
        f"Window: last {window_hours} hours",
        "",
        "## Summary",
        f"- New model candidates: {len(candidates)}",
        f"- High-priority benchmark candidates: {len(grouped['Benchmark now'])}",
        f"- Watchlist candidates: {len(grouped['Watchlist'])}",
        f"- Ignored or low-confidence items: {len(grouped['Ignore unless manually reviewed'])}",
        f"- Source connector failures: {len(failures)}",
        "",
    ]

    for action, heading in GROUPS.items():
        lines.extend([f"## {heading}", ""])
        entries = grouped[action]
        if not entries:
            lines.extend(["None.", ""])
            continue
        for idx, candidate in enumerate(entries, 1):
            evidence = ", ".join(candidate.evidence_urls) or "No evidence URL"
            strengths = "; ".join(candidate.claimed_strengths) or "No specific claim extracted"
            snippets = " | ".join(chunk.get("text", "") for chunk in candidate.evidence_chunks[:2])
            lines.extend(
                [
                    f"{idx}. {candidate.canonical_model_name}",
                    f"   Provider: {candidate.provider or 'unknown'}",
                    f"   Release type: {candidate.release_type}",
                    f"   Access: {candidate.access_type}",
                    f"   Modality: {', '.join(candidate.modality) or 'unknown'}",
                    f"   Score: {candidate.benchmark_relevance_score}",
                    f"   Why it matters: {strengths}",
                    f"   Suggested benchmark suite: {suggest_suite(candidate)}",
                    f"   Evidence: {evidence}",
                    f"   Evidence snippets: {snippets or 'None'}",
                    f"   Recommended action: {candidate.recommended_action}",
                    "",
                ]
            )

    lines.extend(["## Source Connector Failures", ""])
    if failures:
        lines.extend(f"- {name}: {error}" for name, error in failures.items())
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def suggest_suite(candidate: Candidate) -> str:
    modalities = set(candidate.modality)
    if "vision" in modalities:
        return "vision-language benchmark suite"
    if "embedding" in modalities:
        return "embedding retrieval benchmark suite"
    if "reranker" in modalities:
        return "reranking benchmark suite"
    return "general language and reasoning benchmark suite"
