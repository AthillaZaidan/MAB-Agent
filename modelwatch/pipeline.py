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
Logger = Callable[[str], None]


def run_pipeline(
    *,
    connectors: list[Connector],
    extractor: Extractor,
    store: Store,
    output_dir: str | Path,
    window_hours: int = 48,
    log: Logger | None = None,
) -> PipelineResult:
    started = datetime.now(UTC)
    failures: dict[str, str] = {}
    source_count = 0

    for connector in connectors:
        name = getattr(connector, "__name__", connector.__class__.__name__)
        emit(log, f"[source] {name} start")
        try:
            items = connector(window_hours)
        except Exception as exc:  # ponytail: keep connector isolation broad; add typed errors if retries/backoff matter.
            failures[name] = str(exc)
            emit(log, f"[source] {name} failed: {exc.__class__.__name__}: {exc}")
            continue
        for failed_name, reason in getattr(connector, "failures", {}).items():
            key = f"{name}:{failed_name}"
            failures[key] = reason
            emit(log, f"[source] {name} warning: {failed_name}: {reason}")
        emit(log, f"[source] {name} fetched {len(items)} items")
        for item in items:
            if store.save_source_item(item):
                source_count += 1
        for index, item in enumerate(items, 1):
            emit(log, f"[extract] {name} {index}/{len(items)} {item.title}")
            candidate = extractor(item)
            if candidate is not None:
                store.upsert_candidate(score_candidate(candidate))
                emit(log, f"[extract] {name} candidate: {candidate.canonical_model_name}")
            else:
                emit(log, f"[extract] {name} skipped")

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
    emit(log, f"[done] {status}: {source_count} source items, {len(candidates)} candidates, digest={digest_path}")
    return PipelineResult(status, source_count, len(candidates), digest_path, failures)


def emit(log: Logger | None, message: str) -> None:
    if log:
        log(message)
