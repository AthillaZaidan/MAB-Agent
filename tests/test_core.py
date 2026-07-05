import json
from dataclasses import asdict
from datetime import UTC, datetime

from modelwatch.digest import render_digest
from modelwatch.models import Candidate, RawItem
from modelwatch.normalize import model_key
from modelwatch.pipeline import run_pipeline
from modelwatch.scoring import action_for_score, score_candidate
from modelwatch.store import Store


def item(title: str, url: str = "https://example.test/model") -> RawItem:
    return RawItem(
        source_name="fixture",
        source_type="blog",
        source_url=url,
        title=title,
        author_or_provider="Qwen",
        published_at=datetime(2026, 7, 5, tzinfo=UTC),
        updated_at=None,
        raw_text=f"{title} is now available with benchmark claims.",
        raw_metadata={},
    )


def candidate(name: str, score: float = 0) -> Candidate:
    return Candidate(
        canonical_model_name=name,
        provider="Qwen",
        release_type="new_model",
        release_date="2026-07-05",
        modality=["text"],
        access_type="open_weight",
        license="apache-2.0",
        parameter_size="4B",
        context_length=32768,
        claimed_strengths=["coding"],
        benchmark_claims=[],
        availability=[{"platform": "Hugging Face", "url": "https://hf.test/qwen"}],
        confidence=0.92,
        evidence_urls=["https://example.test/model"],
        benchmark_relevance_score=score,
        recommended_action=action_for_score(score),
    )


def test_aliases_dedupe_to_one_candidate(tmp_path):
    store = Store(tmp_path / "modelwatch.sqlite")
    source_items = [item("Qwen3 4B Instruct"), item("Qwen/Qwen3-4B-Instruct")]

    def extractor(raw_item):
        return candidate(raw_item.title)

    result = run_pipeline(
        connectors=[lambda _window: source_items],
        extractor=extractor,
        store=store,
        output_dir=tmp_path,
    )

    assert result.status == "success"
    assert result.candidate_count == 1
    [merged] = store.list_candidates()
    assert merged.canonical_model_name == "Qwen3 4B Instruct"
    assert sorted(merged.aliases) == ["Qwen/Qwen3-4B-Instruct"]
    assert len(merged.evidence_urls) == 1


def test_scoring_maps_candidate_to_benchmark_now():
    scored = score_candidate(candidate("Qwen3 4B Instruct"))

    assert scored.benchmark_relevance_score >= 80
    assert scored.recommended_action == "Benchmark now"


def test_digest_groups_actions_and_includes_evidence():
    md = render_digest(
        date="2026-07-06",
        window_hours=48,
        candidates=[
            candidate("High", 86),
            candidate("Watch", 66),
            candidate("Store", 45),
            candidate("Low", 20),
        ],
        failures={"rss": "timeout"},
    )

    assert "Daily Model Release Digest" in md
    assert "Benchmark Now" in md
    assert "Watchlist" in md
    assert "Store Only" in md
    assert "Low Confidence" in md
    assert "https://example.test/model" in md
    assert "rss: timeout" in md


def test_pipeline_survives_failed_connector(tmp_path):
    store = Store(tmp_path / "modelwatch.sqlite")

    def bad_connector(_window):
        raise RuntimeError("boom")

    def good_connector(_window):
        return [item("Qwen3 4B Instruct")]

    result = run_pipeline(
        connectors=[bad_connector, good_connector],
        extractor=lambda raw_item: candidate(raw_item.title),
        store=store,
        output_dir=tmp_path,
    )

    assert result.status == "partial_success"
    assert result.source_count == 1
    assert result.candidate_count == 1
    assert result.digest_path.exists()


def test_store_normalizes_older_bad_candidate_payload(tmp_path):
    store = Store(tmp_path / "modelwatch.sqlite")
    bad = asdict(candidate("Qwen3 4B Instruct", 86))
    bad["availability"] = "Hugging Face"
    bad["evidence_urls"] = "https://example.test/model"
    store.db.execute(
        "insert into candidates (model_key, payload) values (?, ?)",
        ("qwen:qwen34b:4b", json.dumps(bad)),
    )
    store.db.commit()

    store.upsert_candidate(candidate("Qwen/Qwen3-4B-Instruct", 80))

    [merged] = store.list_candidates()
    assert merged.availability == [{"platform": "Hugging Face", "url": "https://hf.test/qwen"}]
    assert merged.evidence_urls == ["https://example.test/model"]


def test_model_key_accepts_numeric_parameter_size():
    assert model_key("Qwen", "Qwen3", 4) == "qwen:qwen3:4"
