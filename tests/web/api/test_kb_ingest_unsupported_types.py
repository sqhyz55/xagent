"""Tests for KB ingest friendly errors on unsupported formats.

These tests define the expected fail-fast behavior for the /ingest endpoint:
- Files whose extension is allowed to upload but has no available parser → 422
- Explicit parser that is incompatible with the file extension → 422
- HTML / HTM files with 'unstructured' parser available → accepted (not 422)
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.core.tools.core.RAG_tools.core.parser_registry import get_supported_parsers
from xagent.web.api.kb import kb_router
from xagent.web.models.database import get_db


@pytest.fixture
def mock_user():
    """Minimal user-like object for ingest dependency."""
    return type("User", (), {"id": 1, "is_admin": False})()


@pytest.fixture
def mock_db():
    """Minimal mock DB session so FastAPI dependency resolution succeeds."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    return db


@pytest.fixture
def app_with_kb(mock_user, mock_db):
    """FastAPI app with kb_router and mocked auth + db."""
    from xagent.web.api.kb import get_current_user

    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_db] = lambda: mock_db
    return app


def _post_ingest(client: TestClient, filename: str, content: bytes = b"x", **form_data):
    """Helper to POST a file to /api/kb/ingest."""
    return client.post(
        "/api/kb/ingest",
        data={"collection": "test_coll", **form_data},
        files={"file": (filename, io.BytesIO(content), "application/octet-stream")},
    )


# ---------------------------------------------------------------------------
# Registry-level tests
# ---------------------------------------------------------------------------


class TestParserRegistrySupport:
    """Verify parser registry declares correct support for file types."""

    def test_html_has_unstructured(self):
        supported = get_supported_parsers(".html")
        assert "unstructured" in supported

    def test_htm_has_unstructured(self):
        supported = get_supported_parsers(".htm")
        assert "unstructured" in supported

    def test_code_files_have_no_parsers(self):
        for ext in (".py", ".go", ".rs", ".sh"):
            supported = get_supported_parsers(ext)
            assert supported == [], f"Expected no parsers for {ext}, got {supported}"

    def test_pdf_has_multiple_parsers(self):
        supported = get_supported_parsers(".pdf")
        assert len(supported) >= 2
        assert "deepdoc" in supported


# ---------------------------------------------------------------------------
# Ingest endpoint: reject unsupported / incompatible files (fail-fast)
# ---------------------------------------------------------------------------


class TestIngestFailFast:
    """Fail-fast validation at API boundary before file is written to disk."""

    def test_rejects_allowed_but_unparsable_extension(self, app_with_kb):
        """Uploading an allowed-but-unparsable file (.py) returns a friendly 422."""
        client = TestClient(app_with_kb)

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("xagent.web.api.kb.get_upload_path") as mock_path,
                patch("xagent.web.api.kb.run_document_ingestion") as mock_ingest,
            ):
                mock_path.return_value = str(Path(tmpdir) / "test.py")
                resp = _post_ingest(client, "test.py", b"print('hi')\n")

        assert resp.status_code == 422
        body = resp.json()
        assert "Unsupported file type" in body.get("detail", "")
        mock_ingest.assert_not_called()

    def test_rejects_incompatible_explicit_parser(self, app_with_kb):
        """Specifying deepdoc for a .pptx file should be rejected as incompatible.

        .pptx only supports 'unstructured', and 'deepdoc' is not a PDF-only
        parser (so it won't be auto-normalized to default).
        """
        client = TestClient(app_with_kb)

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("xagent.web.api.kb.get_upload_path") as mock_path,
                patch("xagent.web.api.kb.run_document_ingestion") as mock_ingest,
            ):
                mock_path.return_value = str(Path(tmpdir) / "slides.pptx")
                resp = _post_ingest(
                    client,
                    "slides.pptx",
                    b"PK\x03\x04",
                    parse_method="deepdoc",
                )

        assert resp.status_code == 422
        detail = resp.json().get("detail", "")
        assert "not compatible" in detail or "incompatible" in detail.lower()
        mock_ingest.assert_not_called()

    def test_html_not_rejected_by_fail_fast(self, app_with_kb):
        """HTML files must pass the parser validation check.

        The request may fail later (deeper mocking gaps) but must NOT be rejected
        with an 'Unsupported file type' or 'not compatible' 422 error.
        """
        client = TestClient(app_with_kb, raise_server_exceptions=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "xagent.web.api.kb.get_upload_path",
                return_value=str(Path(tmpdir) / "page.html"),
            ):
                resp = _post_ingest(
                    client, "page.html", b"<html><body>hi</body></html>"
                )

        if resp.status_code == 422:
            detail = resp.json().get("detail", "")
            assert "Unsupported file type" not in detail, (
                f"HTML should not be rejected as unsupported: {detail}"
            )
            assert "not compatible" not in detail.lower(), (
                f"HTML should not be rejected as incompatible: {detail}"
            )

    def test_htm_not_rejected_by_fail_fast(self, app_with_kb):
        """HTM files must also pass the parser validation check."""
        client = TestClient(app_with_kb, raise_server_exceptions=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "xagent.web.api.kb.get_upload_path",
                return_value=str(Path(tmpdir) / "page.htm"),
            ):
                resp = _post_ingest(client, "page.htm", b"<html><body>hi</body></html>")

        if resp.status_code == 422:
            detail = resp.json().get("detail", "")
            assert "Unsupported file type" not in detail, (
                f"HTM should not be rejected as unsupported: {detail}"
            )
