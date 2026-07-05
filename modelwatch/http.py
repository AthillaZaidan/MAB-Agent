from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def get_json(url: str, *, params: dict[str, str | int] | None = None, timeout: int = 30):
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "ModelWatch/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_text(url: str, *, params: dict[str, str | int] | None = None, timeout: int = 30) -> str:
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "ModelWatch/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def post_json(url: str, payload: dict, *, timeout: int = 120):
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))
