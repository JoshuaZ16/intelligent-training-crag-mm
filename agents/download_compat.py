"""Small retry primitive for external CRAG-MM image downloads."""

from __future__ import annotations

import time
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit


DEFAULT_IMAGE_USER_AGENT = (
    "CRAGMMCourseResearch/1.0 "
    "(https://github.com/JoshuaZ16/intelligent-training-crag-mm)"
)


def wikimedia_thumbnail_url(url: str, *, width: int = 1280) -> str | None:
    """Return Wikimedia's official thumbnail URL for a Commons original."""
    parsed = urlsplit(url)
    prefix = "/wikipedia/commons/"
    if parsed.hostname != "upload.wikimedia.org" or not parsed.path.startswith(prefix):
        return None
    if width <= 0:
        raise ValueError("width must be positive")
    relative = parsed.path[len(prefix) :]
    filename = relative.rsplit("/", 1)[-1]
    thumb_path = f"{prefix}thumb/{relative}/{width}px-{filename}"
    return urlunsplit((parsed.scheme, parsed.netloc, thumb_path, parsed.query, ""))


def request_with_retries(
    request_get: Callable[..., Any],
    url: str,
    *,
    attempts: int,
    timeout: int,
    headers: dict | None = None,
    sleep: Callable[[float], None] = time.sleep,
):
    """Run a GET request with bounded exponential backoff."""
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    for attempt in range(attempts):
        try:
            return request_get(
                url,
                stream=True,
                timeout=timeout,
                headers=headers or {},
            )
        except Exception:
            if attempt + 1 == attempts:
                raise
            sleep(2**attempt)
