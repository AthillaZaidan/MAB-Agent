from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    output_dir: Path = Path("data/digests")
    database_path: Path = Path("data/modelwatch.sqlite")
    vector_database_path: Path = Path("data/vectors.sqlite")
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b-instruct"
    ollama_embedding_model: str = "nomic-embed-text"
    rss_urls: list[str] = field(default_factory=list)
    github_repos: list[str] = field(default_factory=list)
    arxiv_query: str = 'cat:cs.CL AND (LLM OR "large language model" OR multimodal)'
    max_items_per_source: int = 20
    huggingface_fetch_limit: int = 1000
    huggingface_top_items: int = 150
    email_to: str | None = None


def load_config(path: str | Path | None) -> Config:
    if path is None:
        return Config()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = Config()
    for key, value in payload.items():
        if key in {"output_dir", "database_path", "vector_database_path"}:
            value = Path(value)
        setattr(config, key, value)
    return config
