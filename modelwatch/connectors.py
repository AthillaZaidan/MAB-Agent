from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

from modelwatch.http import get_json, get_text
from modelwatch.models import RawItem
from modelwatch.scoring import DERIVATIVE_RE, MODEL_FAMILY_RE, is_important_provider


class HuggingFaceConnector:
    __name__ = "huggingface"

    def __init__(self, max_items: int = 150, model_prefixes: list[str] | None = None, fetch_limit: int = 1000):
        self.max_items = max_items
        self.fetch_limit = fetch_limit
        self.model_prefixes = model_prefixes or []

    def __call__(self, window_hours: int) -> list[RawItem]:
        payload = get_json(
            "https://huggingface.co/api/models",
            params={"sort": "lastModified", "direction": -1, "limit": self.fetch_limit, "filter": "text-generation"},
        )
        cutoff = datetime.now(UTC) - timedelta(hours=window_hours)
        models = []
        for model in payload:
            updated = parse_dt(model.get("lastModified"))
            if updated and updated < cutoff:
                continue
            model_id = model.get("modelId") or model.get("id") or "unknown"
            if not matches_prefix(model_id, self.model_prefixes):
                continue
            models.append(model)
        items = []
        for model in sorted(models, key=huggingface_model_score, reverse=True)[: self.max_items]:
            model_id = model.get("modelId") or model.get("id") or "unknown"
            updated = parse_dt(model.get("lastModified"))
            text = " ".join(str(part) for part in [model_id, model.get("pipeline_tag"), ",".join(model.get("tags") or [])] if part)
            items.append(
                RawItem(
                    source_name="Hugging Face Hub",
                    source_type="huggingface",
                    source_url=f"https://huggingface.co/{model_id}",
                    title=model_id,
                    author_or_provider=model_id.split("/")[0] if "/" in model_id else None,
                    published_at=None,
                    updated_at=updated,
                    raw_text=text,
                    raw_metadata=model,
                )
            )
        return items


class OpenRouterConnector:
    __name__ = "openrouter"

    def __init__(self, max_items: int = 20, model_prefixes: list[str] | None = None):
        self.max_items = max_items
        self.model_prefixes = model_prefixes or []

    def __call__(self, window_hours: int) -> list[RawItem]:
        payload = get_json("https://openrouter.ai/api/v1/models")
        cutoff = datetime.now(UTC) - timedelta(hours=window_hours)
        models = []
        for model in payload.get("data", []):
            created = datetime.fromtimestamp(model["created"], UTC) if model.get("created") else None
            if created and created < cutoff:
                continue
            model_id = model.get("id") or model.get("name") or "unknown"
            if not matches_prefix(model_id, self.model_prefixes):
                continue
            models.append(model)
        items = []
        for model in sorted(models, key=openrouter_model_score, reverse=True)[: self.max_items]:
            created = datetime.fromtimestamp(model["created"], UTC) if model.get("created") else None
            model_id = model.get("id") or model.get("name") or "unknown"
            items.append(
                RawItem(
                    source_name="OpenRouter Models API",
                    source_type="openrouter",
                    source_url=f"https://openrouter.ai/{model_id}",
                    title=model.get("name") or model_id,
                    author_or_provider=model_id.split("/")[0] if "/" in model_id else None,
                    published_at=created,
                    updated_at=None,
                    raw_text=" ".join(str(model.get(key) or "") for key in ["id", "name", "description"]),
                    raw_metadata=model,
                )
            )
        return items


class ArxivConnector:
    __name__ = "arxiv"

    def __init__(self, query: str, max_items: int = 20):
        self.query = query
        self.max_items = max_items

    def __call__(self, window_hours: int) -> list[RawItem]:
        xml = get_text(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": self.query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": self.max_items,
            },
        )
        root = ET.fromstring(xml)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        cutoff = datetime.now(UTC) - timedelta(hours=window_hours)
        items = []
        for entry in root.findall("atom:entry", ns):
            title = text(entry, "atom:title", ns)
            published = parse_dt(text(entry, "atom:published", ns))
            updated = parse_dt(text(entry, "atom:updated", ns))
            if published and published < cutoff and (not updated or updated < cutoff):
                continue
            url = text(entry, "atom:id", ns)
            summary = text(entry, "atom:summary", ns)
            items.append(
                RawItem(
                    source_name="arXiv",
                    source_type="arxiv",
                    source_url=url,
                    title=title,
                    author_or_provider=None,
                    published_at=published,
                    updated_at=updated,
                    raw_text=f"{title}\n{summary}",
                    raw_metadata={},
                )
            )
        return items


class GitHubReleasesConnector:
    __name__ = "github"

    def __init__(self, repos: list[str], max_items: int = 20):
        self.repos = repos
        self.max_items = max_items

    def __call__(self, window_hours: int) -> list[RawItem]:
        cutoff = datetime.now(UTC) - timedelta(hours=window_hours)
        items = []
        for repo in self.repos:
            releases = get_json(f"https://api.github.com/repos/{repo}/releases", params={"per_page": self.max_items})
            for release in releases:
                published = parse_dt(release.get("published_at"))
                if published and published < cutoff:
                    continue
                items.append(
                    RawItem(
                        source_name="GitHub Releases",
                        source_type="github",
                        source_url=release.get("html_url") or f"https://github.com/{repo}/releases",
                        title=f"{repo} {release.get('name') or release.get('tag_name')}",
                        author_or_provider=repo.split("/")[0],
                        published_at=published,
                        updated_at=None,
                        raw_text=f"{release.get('name') or ''}\n{release.get('body') or ''}",
                        raw_metadata=release,
                    )
                )
        return items


class RssConnector:
    __name__ = "rss"

    def __init__(self, urls: list[str], max_items: int = 20):
        self.urls = urls
        self.max_items = max_items
        self.failures: dict[str, str] = {}

    def __call__(self, window_hours: int) -> list[RawItem]:
        cutoff = datetime.now(UTC) - timedelta(hours=window_hours)
        items = []
        self.failures = {}
        for feed_url in self.urls:
            try:
                root = ET.fromstring(get_text(feed_url))
            except Exception as exc:  # ponytail: per-feed isolation, not a full RSS connector failure.
                self.failures[feed_url] = f"{exc.__class__.__name__}: {exc}"
                continue
            entries = root.findall(".//item") or root.findall("{http://www.w3.org/2005/Atom}entry")
            for entry in entries[: self.max_items]:
                title = child_text(entry, "title")
                url = child_text(entry, "link") or feed_url
                raw_date = child_text(entry, "pubDate") or child_text(entry, "published") or child_text(entry, "updated")
                published = parse_dt(raw_date)
                if published and published < cutoff:
                    continue
                summary = child_text(entry, "description") or child_text(entry, "summary")
                items.append(
                    RawItem(
                        source_name=feed_url,
                        source_type="rss",
                        source_url=url,
                        title=title,
                        author_or_provider=None,
                        published_at=published,
                        updated_at=None,
                        raw_text=strip_html(f"{title}\n{summary}"),
                        raw_metadata={"feed_url": feed_url},
                    )
                )
        return items


def default_connectors(config) -> list[Any]:
    return [
        HuggingFaceConnector(config.huggingface_top_items, fetch_limit=config.huggingface_fetch_limit),
        OpenRouterConnector(config.max_items_per_source),
        ArxivConnector(config.arxiv_query, config.max_items_per_source),
        GitHubReleasesConnector(config.github_repos, config.max_items_per_source),
        RssConnector(config.rss_urls, config.max_items_per_source),
    ]


def matches_prefix(model_id: str, prefixes: list[str]) -> bool:
    if not prefixes:
        return True
    lowered = model_id.lower()
    return any(lowered.startswith(prefix.lower()) for prefix in prefixes)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        try:
            return parsedate_to_datetime(value).astimezone(UTC)
        except (TypeError, ValueError):
            return None


def text(node, path: str, ns: dict[str, str]) -> str:
    found = node.find(path, ns)
    return (found.text or "").strip() if found is not None else ""


def child_text(node, name: str) -> str:
    found = node.find(name)
    if found is None:
        found = node.find(f"{{http://www.w3.org/2005/Atom}}{name}")
    if found is None:
        return ""
    if name == "link" and found.attrib.get("href"):
        return found.attrib["href"]
    return (found.text or "").strip()


def strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value)


def huggingface_model_score(model: dict[str, Any]) -> float:
    model_id = model.get("modelId") or model.get("id") or ""
    owner = model_id.split("/")[0] if "/" in model_id else ""
    tags = " ".join(model.get("tags") or [])
    text = f"{model_id} {model.get('pipeline_tag') or ''} {tags}"
    trusted = is_important_provider(owner)
    score = 0.0
    if trusted:
        score += 100
    if MODEL_FAMILY_RE.search(text):
        score += 60
    score += min(50, math.log10(float(model.get("downloads") or 0) + 1) * 10)
    score += min(40, math.log10(float(model.get("likes") or 0) + 1) * 15)
    if "text-generation" in text:
        score += 10
    if DERIVATIVE_RE.search(text) and not trusted:
        score -= 120
    return score


def openrouter_model_score(model: dict[str, Any]) -> float:
    model_id = model.get("id") or model.get("name") or ""
    provider = model_id.split("/")[0] if "/" in model_id else ""
    text = " ".join(str(model.get(key) or "") for key in ["id", "name", "description"])
    score = float(model.get("created") or 0) / 10_000_000
    if is_important_provider(provider):
        score += 100
    if MODEL_FAMILY_RE.search(text):
        score += 60
    return score
