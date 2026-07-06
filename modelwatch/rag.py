from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from modelwatch.http import post_json


@dataclass(frozen=True)
class EvidenceChunk:
    text: str
    source_url: str


@dataclass(frozen=True)
class SearchResult:
    text: str
    source_url: str
    score: float


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def __call__(self, text: str) -> list[float]:
        response = post_json(f"{self.base_url}/api/embeddings", {"model": self.model, "prompt": text})
        return [float(value) for value in response["embedding"]]


class VectorStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            """
            create table if not exists evidence_vectors (
                id integer primary key,
                text text not null,
                source_url text not null,
                embedding text not null,
                unique(text, source_url)
            )
            """
        )
        self.db.commit()

    def add(self, text: str, source_url: str, embedding: list[float]) -> None:
        self.db.execute(
            "insert or ignore into evidence_vectors (text, source_url, embedding) values (?, ?, ?)",
            (text, source_url, json.dumps(embedding)),
        )
        self.db.commit()

    def search(self, embedding: list[float], limit: int = 3) -> list[SearchResult]:
        rows = self.db.execute("select text, source_url, embedding from evidence_vectors").fetchall()
        results = [
            SearchResult(row["text"], row["source_url"], cosine(embedding, json.loads(row["embedding"])))
            for row in rows
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]


def chunk_text(text: str, *, source_url: str, max_words: int = 120, overlap: int = 20) -> list[EvidenceChunk]:
    words = text.split()
    if not words:
        return []
    step = max(1, max_words - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_words])
        if chunk:
            chunks.append(EvidenceChunk(chunk, source_url))
        if start + max_words >= len(words):
            break
    return chunks


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0
    return dot / (left_norm * right_norm)
