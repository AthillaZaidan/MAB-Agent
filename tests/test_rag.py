from urllib.error import HTTPError

import pytest

import modelwatch.rag as rag
from modelwatch.rag import EvidenceChunk, OllamaEmbedder, VectorStore, chunk_text


def test_chunk_text_splits_overlap_and_keeps_url():
    chunks = chunk_text(
        "alpha beta gamma delta epsilon zeta",
        source_url="https://example.test/model",
        max_words=3,
        overlap=1,
    )

    assert chunks == [
        EvidenceChunk("alpha beta gamma", "https://example.test/model"),
        EvidenceChunk("gamma delta epsilon", "https://example.test/model"),
        EvidenceChunk("epsilon zeta", "https://example.test/model"),
    ]


def test_vector_store_retrieves_similar_chunks(tmp_path):
    store = VectorStore(tmp_path / "vectors.sqlite")
    store.add("Qwen reasoning model", "https://example.test/qwen", [1.0, 0.0])
    store.add("random cooking recipe", "https://example.test/cook", [0.0, 1.0])

    results = store.search([0.9, 0.1], limit=1)

    assert len(results) == 1
    assert results[0].text == "Qwen reasoning model"
    assert results[0].source_url == "https://example.test/qwen"


def test_ollama_embedder_explains_missing_model(monkeypatch):
    def missing_model(*_args, **_kwargs):
        raise HTTPError("http://ollama.test/api/embeddings", 404, "Not Found", {}, None)

    monkeypatch.setattr(rag, "post_json", missing_model)

    with pytest.raises(RuntimeError, match="ollama pull nomic-embed-text"):
        OllamaEmbedder("http://ollama.test", "nomic-embed-text")("hello")
