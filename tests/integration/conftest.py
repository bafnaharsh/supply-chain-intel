"""Integration tests need a running stack. Locally, without one, they are skipped (not failed)
so a bare `pytest` on a fresh clone stays green and says why. In CI a missing stack is an
error: a green run must mean the four required scenarios actually ran."""

import os

import httpx
import pytest

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


def pytest_collection_modifyitems(config, items):
    try:
        httpx.get(f"{BASE_URL}/health", timeout=3)
        return
    except httpx.HTTPError:
        if os.getenv("CI"):  # GitHub Actions sets CI=true
            raise pytest.UsageError(f"no API at {BASE_URL} in CI; the stack must be up") from None
        reason = f"no API at {BASE_URL}: start it with `docker compose up --wait` (or set BASE_URL)"
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip)
