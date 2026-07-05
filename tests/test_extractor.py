from datetime import UTC, datetime

from modelwatch.extractor import OllamaExtractor, parse_json_object
from modelwatch.models import RawItem


def test_parse_json_object_accepts_fenced_ollama_text():
    payload = parse_json_object(
        """
        ```json
        {"model_name": "Qwen3 4B Instruct", "confidence": 0.9}
        ```
        """
    )

    assert payload["model_name"] == "Qwen3 4B Instruct"
    assert payload["confidence"] == 0.9


def test_extractor_accepts_null_list_fields_from_ollama():
    class StubExtractor(OllamaExtractor):
        def _extract(self, prompt):
            return {
                "model_name": "Qwen3 4B Instruct",
                "provider": "Qwen",
                "release_type": "new_model",
                "release_date": "2026-07-05",
                "modality": None,
                "access_type": "open_weight",
                "license": None,
                "parameter_size": "4B",
                "context_length": None,
                "claimed_strengths": None,
                "benchmark_claims": None,
                "availability": None,
                "confidence": 0.8,
                "evidence_urls": None,
            }

    candidate = StubExtractor("http://ollama.test", "qwen3:4b-instruct")(
        RawItem(
            source_name="fixture",
            source_type="blog",
            source_url="https://example.test/qwen",
            title="Qwen3 4B Instruct",
            author_or_provider="Qwen",
            published_at=datetime(2026, 7, 5, tzinfo=UTC),
            updated_at=None,
            raw_text="Qwen3 4B Instruct is released.",
            raw_metadata={},
        )
    )

    assert candidate is not None
    assert candidate.modality == ["unknown"]
    assert candidate.claimed_strengths == []
    assert candidate.benchmark_claims == []
    assert candidate.availability == [{"platform": "fixture", "url": "https://example.test/qwen"}]
    assert candidate.evidence_urls == ["https://example.test/qwen"]


def test_extractor_accepts_string_null_confidence_from_ollama():
    class StubExtractor(OllamaExtractor):
        def _extract(self, prompt):
            return {
                "model_name": "Qwen3 4B Instruct",
                "provider": "Qwen",
                "release_type": "new_model",
                "modality": ["text"],
                "access_type": "open_weight",
                "confidence": "null",
                "evidence_urls": [],
            }

    candidate = StubExtractor("http://ollama.test", "qwen3:4b-instruct")(
        RawItem(
            source_name="fixture",
            source_type="blog",
            source_url="https://example.test/qwen",
            title="Qwen3 4B Instruct",
            author_or_provider="Qwen",
            published_at=datetime(2026, 7, 5, tzinfo=UTC),
            updated_at=None,
            raw_text="Qwen3 4B Instruct is released.",
            raw_metadata={},
        )
    )

    assert candidate is not None
    assert candidate.confidence == 0


def test_extractor_accepts_confidence_labels_from_ollama():
    class StubExtractor(OllamaExtractor):
        def _extract(self, prompt):
            return {
                "model_name": "Qwen3 4B Instruct",
                "provider": "Qwen",
                "release_type": "new_model",
                "modality": ["text"],
                "access_type": "open_weight",
                "confidence": "medium",
                "evidence_urls": [],
            }

    candidate = StubExtractor("http://ollama.test", "qwen3:4b-instruct")(
        RawItem(
            source_name="fixture",
            source_type="blog",
            source_url="https://example.test/qwen",
            title="Qwen3 4B Instruct",
            author_or_provider="Qwen",
            published_at=datetime(2026, 7, 5, tzinfo=UTC),
            updated_at=None,
            raw_text="Qwen3 4B Instruct is released.",
            raw_metadata={},
        )
    )

    assert candidate is not None
    assert candidate.confidence == 0.5


def test_extractor_coerces_context_length_from_ollama():
    class StubExtractor(OllamaExtractor):
        def _extract(self, prompt):
            return {
                "model_name": "Qwen3 4B Instruct",
                "provider": "Qwen",
                "release_type": "new_model",
                "modality": ["text"],
                "access_type": "open_weight",
                "context_length": "32,768 tokens",
                "confidence": "unknown",
                "evidence_urls": [],
            }

    candidate = StubExtractor("http://ollama.test", "qwen3:4b-instruct")(
        RawItem(
            source_name="fixture",
            source_type="blog",
            source_url="https://example.test/qwen",
            title="Qwen3 4B Instruct",
            author_or_provider="Qwen",
            published_at=datetime(2026, 7, 5, tzinfo=UTC),
            updated_at=None,
            raw_text="Qwen3 4B Instruct is released.",
            raw_metadata={},
        )
    )

    assert candidate is not None
    assert candidate.context_length == 32768
    assert candidate.confidence == 0


def test_extractor_normalizes_scalar_list_fields_from_ollama():
    class StubExtractor(OllamaExtractor):
        def _extract(self, prompt):
            return {
                "model_name": "Qwen3 4B Instruct",
                "provider": "Qwen",
                "release_type": "new_model",
                "modality": "text",
                "access_type": "open_weight",
                "claimed_strengths": "coding",
                "benchmark_claims": "none",
                "availability": "Hugging Face",
                "confidence": 0.8,
                "evidence_urls": "https://example.test/qwen",
            }

    candidate = StubExtractor("http://ollama.test", "qwen3:4b-instruct")(
        RawItem(
            source_name="fixture",
            source_type="blog",
            source_url="https://example.test/qwen",
            title="Qwen3 4B Instruct",
            author_or_provider="Qwen",
            published_at=datetime(2026, 7, 5, tzinfo=UTC),
            updated_at=None,
            raw_text="Qwen3 4B Instruct is released.",
            raw_metadata={},
        )
    )

    assert candidate is not None
    assert candidate.modality == ["text"]
    assert candidate.claimed_strengths == ["coding"]
    assert candidate.benchmark_claims == []
    assert candidate.availability == [{"platform": "fixture", "url": "https://example.test/qwen"}]
    assert candidate.evidence_urls == ["https://example.test/qwen"]
