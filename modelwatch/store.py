from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from modelwatch.coerce import as_list
from modelwatch.models import Candidate, RawItem
from modelwatch.normalize import model_key


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        self.db.executescript(
            """
            create table if not exists source_items (
                id integer primary key,
                source_name text not null,
                source_type text not null,
                source_url text not null,
                title text not null,
                author_or_provider text,
                published_at text,
                updated_at text,
                fetched_at text not null,
                raw_text text not null,
                raw_metadata text not null,
                content_hash text not null unique
            );
            create table if not exists candidates (
                model_key text primary key,
                payload text not null
            );
            create table if not exists pipeline_runs (
                id integer primary key,
                started_at text not null,
                finished_at text,
                status text not null,
                source_count integer not null,
                candidate_count integer not null,
                error_summary text not null
            );
            """
        )
        self.db.commit()

    def save_source_item(self, item: RawItem) -> bool:
        payload = f"{item.source_url}\n{item.title}\n{item.raw_text}".encode()
        content_hash = hashlib.sha256(payload).hexdigest()
        try:
            self.db.execute(
                """
                insert into source_items (
                    source_name, source_type, source_url, title, author_or_provider,
                    published_at, updated_at, fetched_at, raw_text, raw_metadata, content_hash
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.source_name,
                    item.source_type,
                    item.source_url,
                    item.title,
                    item.author_or_provider,
                    item.published_at.isoformat() if item.published_at else None,
                    item.updated_at.isoformat() if item.updated_at else None,
                    item.fetched_at.isoformat(),
                    item.raw_text,
                    json.dumps(item.raw_metadata, sort_keys=True),
                    content_hash,
                ),
            )
            self.db.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def upsert_candidate(self, candidate: Candidate) -> None:
        key = model_key(candidate.provider, candidate.canonical_model_name, candidate.parameter_size)
        row = self.db.execute("select payload from candidates where model_key = ?", (key,)).fetchone()
        if row:
            existing = self._candidate_from_payload(json.loads(row["payload"]))
            candidate = self._merge(existing, candidate)
        self.db.execute(
            "insert or replace into candidates (model_key, payload) values (?, ?)",
            (key, json.dumps(asdict(candidate), sort_keys=True)),
        )
        self.db.commit()

    def list_candidates(self) -> list[Candidate]:
        rows = self.db.execute("select payload from candidates order by model_key").fetchall()
        return [self._candidate_from_payload(json.loads(row["payload"])) for row in rows]

    def save_run(self, started_at: str, finished_at: str, status: str, source_count: int, candidate_count: int, failures: dict[str, str]) -> None:
        self.db.execute(
            """
            insert into pipeline_runs (started_at, finished_at, status, source_count, candidate_count, error_summary)
            values (?, ?, ?, ?, ?, ?)
            """,
            (started_at, finished_at, status, source_count, candidate_count, json.dumps(failures, sort_keys=True)),
        )
        self.db.commit()

    @staticmethod
    def _candidate_from_payload(payload: dict[str, Any]) -> Candidate:
        payload["modality"] = as_list(payload.get("modality")) or ["unknown"]
        payload["claimed_strengths"] = as_list(payload.get("claimed_strengths"))
        payload["benchmark_claims"] = [claim for claim in as_list(payload.get("benchmark_claims")) if isinstance(claim, dict)]
        payload["availability"] = [entry for entry in as_list(payload.get("availability")) if isinstance(entry, dict)]
        payload["evidence_urls"] = as_list(payload.get("evidence_urls"))
        payload["aliases"] = as_list(payload.get("aliases"))
        return Candidate(**payload)

    @staticmethod
    def _merge(existing: Candidate, new: Candidate) -> Candidate:
        aliases = list(dict.fromkeys(existing.aliases + ([] if existing.canonical_model_name == new.canonical_model_name else [new.canonical_model_name]) + new.aliases))
        evidence = list(dict.fromkeys(existing.evidence_urls + new.evidence_urls))
        strengths = list(dict.fromkeys(existing.claimed_strengths + new.claimed_strengths))
        availability = existing.availability + [entry for entry in new.availability if entry not in existing.availability]
        existing.aliases = aliases
        existing.evidence_urls = evidence
        existing.claimed_strengths = strengths
        existing.availability = availability
        existing.confidence = max(existing.confidence, new.confidence)
        existing.benchmark_relevance_score = max(existing.benchmark_relevance_score, new.benchmark_relevance_score)
        existing.recommended_action = new.recommended_action if new.benchmark_relevance_score > existing.benchmark_relevance_score else existing.recommended_action
        return existing
