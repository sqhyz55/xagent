"""Tests for KB web ingestion input validation."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.web.api.kb import kb_router


@pytest.fixture
def mock_user():
    """Minimal user-like object for ingest dependency."""

    return type("User", (), {"id": 1, "is_admin": False})()


@pytest.fixture
def app_with_kb(mock_user):
    """FastAPI app with kb_router and mocked auth."""

    from unittest.mock import MagicMock

    from xagent.web.api.kb import get_current_user
    from xagent.web.models.database import get_db

    def override_get_current_user():
        return mock_user

    def override_get_db():
        yield MagicMock()

    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db
    return app


@pytest.mark.parametrize(
    "start_url",
    [
        "xinference.cn",
        " www.xinference.cn ",
        "ftp://xinference.cn",
        "https://",
        "http://",
        "://xinference.cn",
        "",
        "   ",
    ],
)
def test_ingest_web_rejects_invalid_start_url(app_with_kb, start_url: str) -> None:
    client = TestClient(app_with_kb)

    resp = client.post(
        "/api/kb/ingest-web",
        data={
            "collection": "test_coll",
            "start_url": start_url,
        },
    )

    assert resp.status_code == 422
    body = resp.json()
    detail = body.get("detail", "")

    # Different FastAPI/Starlette versions may treat empty form fields as missing
    # ("Field required") or pass through as empty string (our custom validator).
    # Accept either behavior for blank inputs.
    if not start_url.strip():
        if isinstance(detail, list):
            assert any(
                (isinstance(item, dict) and item.get("msg") == "Field required")
                for item in detail
            )
        else:
            assert "Invalid start_url" in str(detail)
    else:
        assert "Invalid start_url" in str(detail)


def test_ingest_web_accepts_stripped_url(app_with_kb) -> None:
    """Whitespace/newline around a valid URL should be accepted after strip()."""

    client = TestClient(app_with_kb)
    resp = client.post(
        "/api/kb/ingest-web",
        data={
            "collection": "test_coll",
            "start_url": "  https://xinference.cn\n",
            "max_pages": "1",
            "max_depth": "1",
            "respect_robots_txt": "false",
        },
    )

    # We don't assert success here (depends on network), only that validation passed.
    assert resp.status_code != 422
