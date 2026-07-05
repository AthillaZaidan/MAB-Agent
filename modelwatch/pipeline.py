from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from modelwatch.digest import render_digest
from modelwatch.models import Candidate, PipelineResult, RawItem
from modelwatch.scoring import score_candidate
from modelwatch.store import Store

Connector = Callable[[int], list[RawItem]]
Extractor = Callable[[RawItem], Candidate | None]


def run_pipeline(
    *,
    connectors: list[Connector],
    extractor: Extractor,
    store: Store,
    output_dir: str | Path,
    window_hours: int = 48,
) -> PipelineResult:
    started = datetime.now(UTC)
    failures: dict[str, str] = {}
    source_count = 0

    for connector in connectors:
        name = getattr(connector, "__name__", connector.__class__.__name__)
        try:
            items = connector(window_hours)
        except Exception as exc:  # ponytail: keep connector isolation broad; add typed errors if retries/backoff matter.
            failures[name] = str(exc)
            continue
        for item in items:
            if store.save_source_item(item):
                source_count += 1
            candidate = extractor(item)
            if candidate is not None:
                store.upsert_candidate(score_candidate(candidate))

    candidates = store.list_candidates()
    status = "failed" if source_count == 0 and failures else "partial_success" if failures else "success"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    digest_path = output_path / f"digest-{datetime.now(UTC).date().isoformat()}.md"
    digest_path.write_text(
        render_digest(
            date=datetime.now(UTC).date().isoformat(),
            window_hours=window_hours,
            candidates=candidates,
            failures=failures,
        ),
        encoding="utf-8",
    )
    finished = datetime.now(UTC)
    store.save_run(started.isoformat(), finished.isoformat(), status, source_count, len(candidates), failures)
    return PipelineResult(status, source_count, len(candidates), digest_path, failures)
