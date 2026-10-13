"""Small in-memory sliding-window rate limiter.

Single-process only, which matches the one-container demo deployment. It exists
to stop an open sign-up form from burning the Groq quota.
"""

import os
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request, status

from auth import get_current_user

_hits: dict = defaultdict(deque)


def _check(key: str, limit: int, window_s: int) -> None:
    now = time.monotonic()
    q = _hits[key]
    while q and now - q[0] > window_s:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit reached ({limit} requests per {window_s // 60} min). Try again later.",
        )
    q.append(now)


def limit_per_user(scope: str, limit: int, window_s: int = 3600):
    """Dependency that limits an authenticated user."""

    async def dep(user: dict = Depends(get_current_user)) -> None:
        _check(f"{scope}:user:{user['id']}", limit, window_s)

    return dep


def limit_per_ip(scope: str, limit: int, window_s: int = 3600):
    """Dependency that limits an unauthenticated client by IP (honours the proxy header)."""

    async def dep(request: Request) -> None:
        forwarded = request.headers.get("x-forwarded-for", "")
        ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
        _check(f"{scope}:ip:{ip}", limit, window_s)

    return dep


CHAT_LIMIT_PER_HOUR = int(os.getenv("CHAT_LIMIT_PER_HOUR", "30"))
UPLOAD_LIMIT_PER_HOUR = int(os.getenv("UPLOAD_LIMIT_PER_HOUR", "20"))
REGISTER_LIMIT_PER_HOUR = int(os.getenv("REGISTER_LIMIT_PER_HOUR", "10"))
