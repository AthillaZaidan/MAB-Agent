import json
from dataclasses import asdict
from datetime import UTC, datetime

from modelwatch.digest import render_digest
from modelwatch.models import Candidate, JudgeDecision, RawItem
from modelwatch.normalize import model_key
from modelwatch.pipeline import run_pipeline
from modelwatch.rag import VectorStore
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


def test_pipeline_digest_filename_includes_timestamp(tmp_path):
    result = run_pipeline(
        connectors=[lambda _window: [item("Qwen3 4B Instruct")]],
        extractor=lambda raw_item: candidate(raw_item.title),
        store=Store(tmp_path / "modelwatch.sqlite"),
        output_dir=tmp_path,
    )

    assert result.digest_path.name.startswith("digest-")
    assert result.digest_path.name.endswith(".md")
    assert len(result.digest_path.stem) == len("digest-2026-07-06-013045")


def test_pipeline_digest_only_uses_candidates_from_current_run(tmp_path):
    store = Store(tmp_path / "modelwatch.sqlite")
    store.upsert_candidate(score_candidate(candidate("old-random-model")))

    result = run_pipeline(
        connectors=[lambda _window: [item("Qwen/Qwen3-30B-A3B")]],
        extractor=lambda raw_item: candidate(raw_item.title),
        store=store,
        output_dir=tmp_path,
    )

    digest = result.digest_path.read_text(encoding="utf-8")
    assert "Qwen/Qwen3-30B-A3B" in digest
    assert "old-random-model" not in digest


def test_pipeline_judges_items_before_extraction(tmp_path):
    store = Store(tmp_path / "modelwatch.sqlite")
    logs = []
    extracted = []
    source_items = [item("random-user/cool-lora"), item("Qwen/Qwen3-30B-A3B")]

    def judge(raw_item):
        return JudgeDecision(
            keep=raw_item.title.startswith("Qwen/"),
            reason="benchmark-worthy foundation model" if raw_item.title.startswith("Qwen/") else "personal LoRA",
            confidence=0.9,
        )

    def extractor(raw_item):
        extracted.append(raw_item.title)
        return candidate(raw_item.title)

    result = run_pipeline(
        connectors=[lambda _window: source_items],
        extractor=extractor,
        judge=judge,
        store=store,
        output_dir=tmp_path,
        log=logs.append,
    )

    assert extracted == ["Qwen/Qwen3-30B-A3B"]
    assert result.candidate_count == 1
    assert "[judge] <lambda> random-user/cool-lora rejected: personal LoRA" in logs
    assert "[judge] <lambda> Qwen/Qwen3-30B-A3B kept: benchmark-worthy foundation model" in logs


def test_pipeline_stores_and_retrieves_rag_evidence(tmp_path):
    store = Store(tmp_path / "modelwatch.sqlite")
    vectors = VectorStore(tmp_path / "vectors.sqlite")

    result = run_pipeline(
        connectors=[lambda _window: [item("Qwen/Qwen3-30B-A3B")]],
        extractor=lambda raw_item: candidate(raw_item.title),
        store=store,
        output_dir=tmp_path,
        vector_store=vectors,
        embed=lambda text: [1.0, 0.0] if "Qwen" in text else [0.0, 1.0],
    )

    digest = result.digest_path.read_text(encoding="utf-8")
    assert "Evidence snippets:" in digest
    assert "Qwen/Qwen3-30B-A3B is now available" in digest


def test_pipeline_only_indexes_rag_for_judge_kept_items(tmp_path):
    source_items = [
        item("random-user/cool-lora", "https://example.test/rejected"),
        item("Qwen/Qwen3-30B-A3B", "https://example.test/kept"),
    ]
    embedded_texts = []

    def judge(raw_item):
        return JudgeDecision(keep="Qwen/" in raw_item.title, reason="ok")

    def embed(text):
        embedded_texts.append(text)
        return [1.0, 0.0]

    run_pipeline(
        connectors=[lambda _window: source_items],
        extractor=lambda raw_item: candidate(raw_item.title),
        judge=judge,
        store=Store(tmp_path / "modelwatch.sqlite"),
        output_dir=tmp_path,
        vector_store=VectorStore(tmp_path / "vectors.sqlite"),
        embed=embed,
    )

    assert not any("random-user/cool-lora" in text for text in embedded_texts)
    assert any("Qwen/Qwen3-30B-A3B" in text for text in embedded_texts)


def test_pipeline_scopes_rag_evidence_to_candidate_source_url(tmp_path):
    store = Store(tmp_path / "modelwatch.sqlite")
    vectors = VectorStore(tmp_path / "vectors.sqlite")
    vectors.add("wrong but very similar Qwen evidence", "https://example.test/wrong", [1.0, 0.0])

    run_pipeline(
        connectors=[lambda _window: [item("Qwen/Qwen3-30B-A3B", "https://example.test/right")]],
        extractor=lambda raw_item: candidate(raw_item.title),
        store=store,
        output_dir=tmp_path,
        vector_store=vectors,
        embed=lambda text: [0.9, 0.1] if "available" in text else [1.0, 0.0],
    )

    [stored] = store.list_candidates()
    assert stored.evidence_chunks
    assert {chunk["source_url"] for chunk in stored.evidence_chunks} == {"https://example.test/right"}


def test_pipeline_continues_when_rag_embedding_fails(tmp_path):
    logs = []
    calls = 0

    def broken_embed(_text):
        nonlocal calls
        calls += 1
        raise RuntimeError("embedding model missing")

    result = run_pipeline(
        connectors=[lambda _window: [item("Qwen/Qwen3-30B-A3B")]],
        extractor=lambda raw_item: candidate(raw_item.title),
        store=Store(tmp_path / "modelwatch.sqlite"),
        output_dir=tmp_path,
        vector_store=VectorStore(tmp_path / "vectors.sqlite"),
        embed=broken_embed,
        log=logs.append,
    )

    assert result.status == "success"
    assert result.candidate_count == 1
    assert any(line.startswith("[rag]") and "embedding model missing" in line for line in logs)
    assert calls == 1


def test_pipeline_disables_rag_after_first_embedding_failure(tmp_path):
    calls = 0
    logs = []

    def broken_embed(_text):
        nonlocal calls
        calls += 1
        raise RuntimeError("embedding model missing")

    run_pipeline(
        connectors=[lambda _window: [item("Qwen/Qwen3-30B-A3B"), item("Qwen/Qwen3-14B")]],
        extractor=lambda raw_item: candidate(raw_item.title),
        store=Store(tmp_path / "modelwatch.sqlite"),
        output_dir=tmp_path,
        vector_store=VectorStore(tmp_path / "vectors.sqlite"),
        embed=broken_embed,
        log=logs.append,
    )

    rag_logs = [line for line in logs if line.startswith("[rag]")]
    assert calls == 1
    assert len(rag_logs) == 1


def test_pipeline_logs_connector_progress_and_failures(tmp_path):
    store = Store(tmp_path / "modelwatch.sqlite")
    logs = []

    def bad_connector(_window):
        raise RuntimeError("HTTP Error 429: Too Many Requests")

    def good_connector(_window):
        return [item("Qwen3 4B Instruct")]
    good_connector.failures = {"https://bad.test/rss.xml": "HTTPError: HTTP Error 404: Not Found"}

    run_pipeline(
        connectors=[bad_connector, good_connector],
        extractor=lambda raw_item: candidate(raw_item.title),
        store=store,
        output_dir=tmp_path,
        log=logs.append,
    )

    assert "[source] bad_connector start" in logs
    assert "[source] bad_connector failed: RuntimeError: HTTP Error 429: Too Many Requests" in logs
    assert "[source] good_connector warning: https://bad.test/rss.xml: HTTPError: HTTP Error 404: Not Found" in logs
    assert "[source] good_connector fetched 1 items" in logs
    assert "[extract] good_connector 1/1 Qwen3 4B Instruct" in logs
    assert any(line.startswith("[done] partial_success:") for line in logs)


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
